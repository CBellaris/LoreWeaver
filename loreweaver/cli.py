"""LoreWeaver command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loreweaver import __version__
from loreweaver.config import load_config
from loreweaver.evidence.assembler import assemble_evidence_pack_from_retrieval_report
from loreweaver.eval.corpus import build_chapter_corpus
from loreweaver.eval.generator import generate_question_set
from loreweaver.eval.runner import run_eval, summarize_eval_run
from loreweaver.extraction.extractor import extract_document_windows, list_extraction_windows
from loreweaver.graph.center_span import build_m15_graph, list_graph_clusters
from loreweaver.ingest.pipeline import ingest_text
from loreweaver.ingest.window_splitter import build_candidate_windows
from loreweaver.indexing.pipeline import (
    build_m14_indexes,
    search_bm25_index,
    search_vector_index,
)
from loreweaver.logging import configure_logging, new_run_id
from loreweaver.progress import build_cli_progress_reporter
from loreweaver.qa.answerer import ask_m18
from loreweaver.retrieval.pipeline import retrieve_m16
from loreweaver.storage.sqlite_store import SQLiteStore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loreweaver",
        description="LoreWeaver M1 command line interface.",
    )
    parser.add_argument("--version", action="version", version=f"LoreWeaver {__version__}")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the LoreWeaver config file.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--progress",
        choices=("auto", "rich", "text", "jsonl", "none"),
        default="auto",
        help="Progress renderer for long-running commands.",
    )

    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser("status", help="Show project bootstrap status.")
    status_parser.set_defaults(func=_status)

    web_parser = subparsers.add_parser(
        "web",
        help="Run the local debugging Web UI.",
    )
    web_parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    web_parser.add_argument("--port", type=int, default=7860, help="Port to bind.")
    web_parser.add_argument(
        "--storage-config",
        default="configs/storage.yaml",
        help="Path to storage config containing the SQLite path.",
    )
    web_parser.add_argument(
        "--models-config",
        default="configs/models.yaml",
        help="Path to provider/model config.",
    )
    web_parser.set_defaults(func=_web)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="M1.1 ingest: normalize raw text, split chapters, and write SQLite metadata.",
    )
    ingest_parser.add_argument(
        "--source",
        help="Raw .txt source path. Defaults to sample.source_path in config.",
    )
    ingest_parser.add_argument("--title", help="Document title override.")
    ingest_parser.add_argument("--author", help="Document author override.")
    ingest_parser.add_argument(
        "--max-chapters",
        type=int,
        help="Only persist the first N detected chapters for early M1 runs.",
    )
    ingest_parser.add_argument(
        "--storage-config",
        default="configs/storage.yaml",
        help="Path to storage config containing the SQLite path.",
    )
    ingest_parser.set_defaults(func=_ingest)

    windows_parser = subparsers.add_parser(
        "windows",
        help="M1.2 windows: build chapter windows or fallback sliding windows.",
    )
    windows_parser.add_argument(
        "--document-id",
        help="Document id to split. Defaults to the latest SQLite document.",
    )
    windows_parser.add_argument(
        "--storage-config",
        default="configs/storage.yaml",
        help="Path to storage config containing the SQLite path.",
    )
    windows_parser.add_argument(
        "--window-size",
        type=int,
        help="Window size in normalized-text characters.",
    )
    windows_parser.add_argument(
        "--overlap-ratio",
        type=float,
        help="Window overlap ratio in the range [0, 1).",
    )
    windows_parser.add_argument(
        "--min-chars",
        type=int,
        help="Minimum standalone tail-window length before merging.",
    )
    windows_parser.add_argument(
        "--max-chars",
        type=int,
        help="Maximum expected window length for validation warnings.",
    )
    windows_parser.add_argument(
        "--window-mode",
        choices=("auto", "by_chapter", "sliding"),
        help=(
            "Windowing strategy. auto uses one window per detected chapter, "
            "or sliding windows over the whole-document fallback chapter."
        ),
    )
    windows_parser.set_defaults(func=_windows)

    extract_parser = subparsers.add_parser(
        "extract",
        help="M1.3 extract: call an LLM for multi-Span metadata and locate anchors.",
    )
    extract_parser.add_argument(
        "--document-id",
        help="Document id to extract. Defaults to the latest SQLite document.",
    )
    extract_parser.add_argument(
        "--storage-config",
        default="configs/storage.yaml",
        help="Path to storage config containing the SQLite path.",
    )
    extract_parser.add_argument(
        "--models-config",
        default="configs/models.yaml",
        help="Path to provider/model config containing API env var and model price.",
    )
    extract_parser.add_argument(
        "--limit",
        type=int,
        help="Only extract N windows for small paid-API test runs.",
    )
    extract_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N windows before extraction.",
    )
    extract_parser.add_argument(
        "--window-id",
        action="append",
        help="Extract a candidate window id. Can be repeated or comma-separated.",
    )
    extract_parser.add_argument(
        "--window-range",
        action="append",
        help="Extract 1-based global window range, for example 21-40 or 21:40.",
    )
    extract_parser.add_argument(
        "--list-windows",
        action="store_true",
        help="List candidate windows and extraction status instead of extracting.",
    )
    extract_parser.add_argument(
        "--only",
        choices=("all", "extracted", "pending"),
        default="all",
        help="When used with --list-windows, filter by extraction status.",
    )
    extract_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic local mock extraction without calling an LLM API.",
    )
    extract_parser.add_argument(
        "--batch",
        action="store_true",
        help="Submit selected windows as a SiliconFlow/OpenAI-compatible batch job.",
    )
    extract_parser.add_argument(
        "--batch-id",
        help="Retrieve and apply an existing batch job. The batch must contain window_id custom_ids.",
    )
    extract_parser.add_argument(
        "--batch-model",
        default=None,
        help=(
            "Model used for batch extraction. Defaults to services.extraction.batch_model "
            "from the models config."
        ),
    )
    extract_parser.add_argument(
        "--batch-wait",
        action="store_true",
        help="Poll after batch submission/retrieval until the batch reaches a terminal status.",
    )
    extract_parser.add_argument(
        "--batch-poll-interval",
        type=float,
        default=30.0,
        help="Seconds between batch status polls when --batch-wait is set.",
    )
    extract_parser.add_argument(
        "--batch-timeout",
        type=float,
        default=None,
        help="Maximum seconds to wait for a batch before returning the current status.",
    )
    extract_parser.add_argument(
        "--batch-completion-window",
        default="24h",
        help="Batch completion window passed to the provider.",
    )
    extract_parser.set_defaults(func=_extract)

    index_parser = subparsers.add_parser(
        "index",
        help="M1.4 index: build local Qdrant vector index and BM25 index from located spans.",
    )
    index_parser.add_argument(
        "--document-id",
        help="Document id to index. Defaults to the latest SQLite document.",
    )
    index_parser.add_argument(
        "--storage-config",
        default="configs/storage.yaml",
        help="Path to storage config containing SQLite, Qdrant, and BM25 settings.",
    )
    index_parser.add_argument(
        "--models-config",
        default="configs/models.yaml",
        help="Path to provider/model config containing embedding settings.",
    )
    index_parser.add_argument(
        "--limit",
        type=int,
        help="Only index N located spans for small plumbing checks.",
    )
    index_parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use deterministic local mock embeddings without calling an API.",
    )
    index_parser.set_defaults(func=_index)

    search_vector_parser = subparsers.add_parser(
        "search-vector",
        help="M1.4 debug: search the vector index.",
    )
    search_vector_parser.add_argument("query", help="Query text.")
    search_vector_parser.add_argument(
        "--document-id",
        help="Document id to search. Defaults to the latest SQLite document.",
    )
    search_vector_parser.add_argument(
        "--storage-config",
        default="configs/storage.yaml",
        help="Path to storage config containing SQLite and Qdrant settings.",
    )
    search_vector_parser.add_argument(
        "--models-config",
        default="configs/models.yaml",
        help="Path to provider/model config containing embedding settings.",
    )
    search_vector_parser.add_argument("--top-k", type=int, default=5, help="Number of hits.")
    search_vector_parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use deterministic local mock embeddings for query embedding.",
    )
    search_vector_parser.set_defaults(func=_search_vector)

    search_bm25_parser = subparsers.add_parser(
        "search-bm25",
        help="M1.4 debug: search the BM25 index.",
    )
    search_bm25_parser.add_argument("query", help="Query text.")
    search_bm25_parser.add_argument(
        "--document-id",
        help="Document id to search. Defaults to the latest SQLite document.",
    )
    search_bm25_parser.add_argument(
        "--storage-config",
        default="configs/storage.yaml",
        help="Path to storage config containing SQLite and BM25 settings.",
    )
    search_bm25_parser.add_argument("--top-k", type=int, default=5, help="Number of hits.")
    search_bm25_parser.set_defaults(func=_search_bm25)

    spans_parser = subparsers.add_parser(
        "spans",
        help="M1.5 debug: list high-salience located spans for center-span selection.",
    )
    spans_parser.add_argument(
        "--document-id",
        help="Document id to inspect. Defaults to the latest SQLite document.",
    )
    spans_parser.add_argument(
        "--storage-config",
        default="configs/storage.yaml",
        help="Path to storage config containing the SQLite path.",
    )
    spans_parser.add_argument(
        "--top-salience",
        type=int,
        default=30,
        help="Number of located spans to show by salience.",
    )
    spans_parser.set_defaults(func=_spans)

    graph_parser = subparsers.add_parser(
        "graph",
        help="M1.5 graph: build or inspect the Center Span Cluster skeleton.",
    )
    graph_parser.add_argument(
        "--document-id",
        help="Document id to process. Defaults to the latest SQLite document.",
    )
    graph_parser.add_argument(
        "--storage-config",
        default="configs/storage.yaml",
        help="Path to storage config containing SQLite and optional Neo4j settings.",
    )
    graph_parser.add_argument(
        "--cluster-count",
        type=int,
        help="Maximum number of CenterSpanClusters to build.",
    )
    graph_parser.add_argument(
        "--members-per-cluster",
        type=int,
        help="Member Span count per cluster.",
    )
    graph_parser.add_argument(
        "--min-members",
        type=int,
        help="Minimum member Span count required for a cluster.",
    )
    graph_parser.add_argument(
        "--sync-neo4j",
        action="store_true",
        help="Sync the graph to Neo4j even if storage.neo4j.enabled is false.",
    )
    graph_parser.add_argument(
        "--no-neo4j",
        action="store_true",
        help="Skip Neo4j sync even if storage.neo4j.enabled is true.",
    )
    graph_parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Use the deterministic rule-only graph scoring fallback.",
    )
    graph_parser.add_argument(
        "--list",
        action="store_true",
        help="List existing clusters instead of rebuilding them.",
    )
    graph_parser.add_argument(
        "--cluster-id",
        help="When used with --list, show one cluster and its member spans.",
    )
    graph_parser.set_defaults(func=_graph)

    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="M1.6 retrieve: run graph + vector + BM25 hybrid retrieval and rerank candidates.",
    )
    retrieve_parser.add_argument("question", help="User question to retrieve evidence for.")
    retrieve_parser.add_argument(
        "--document-id",
        help="Document id to retrieve from. Defaults to the latest SQLite document.",
    )
    retrieve_parser.add_argument(
        "--storage-config",
        default="configs/storage.yaml",
        help="Path to storage config containing SQLite, Qdrant, and BM25 settings.",
    )
    retrieve_parser.add_argument(
        "--models-config",
        default="configs/models.yaml",
        help="Path to provider/model config containing embedding and reranker settings.",
    )
    retrieve_parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use deterministic local mock embeddings for vector query embedding.",
    )
    retrieve_parser.add_argument(
        "--mock-reranker",
        action="store_true",
        help="Use deterministic local mock reranking instead of a live reranker API.",
    )
    retrieve_parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip reranking and keep Union fused-score ordering.",
    )
    retrieve_parser.set_defaults(func=_retrieve)

    evidence_parser = subparsers.add_parser(
        "evidence",
        help="M1.7 evidence: assemble an Evidence Pack from hybrid retrieval Top-K spans.",
    )
    evidence_parser.add_argument("question", help="User question to assemble evidence for.")
    evidence_parser.add_argument(
        "--document-id",
        help="Document id to retrieve from. Defaults to the latest SQLite document.",
    )
    evidence_parser.add_argument(
        "--storage-config",
        default="configs/storage.yaml",
        help="Path to storage config containing SQLite, Qdrant, and BM25 settings.",
    )
    evidence_parser.add_argument(
        "--models-config",
        default="configs/models.yaml",
        help="Path to provider/model config containing embedding and reranker settings.",
    )
    evidence_parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use deterministic local mock embeddings for vector query embedding.",
    )
    evidence_parser.add_argument(
        "--mock-reranker",
        action="store_true",
        help="Use deterministic local mock reranking instead of a live reranker API.",
    )
    evidence_parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip reranking and keep Union fused-score ordering.",
    )
    evidence_parser.set_defaults(func=_evidence)

    ask_parser = subparsers.add_parser(
        "ask",
        help="M1.8 ask: run hybrid retrieval, assemble evidence, and answer with citations.",
    )
    ask_parser.add_argument("question", help="User question to answer.")
    ask_parser.add_argument(
        "--document-id",
        help="Document id to retrieve from. Defaults to the latest SQLite document.",
    )
    ask_parser.add_argument(
        "--storage-config",
        default="configs/storage.yaml",
        help="Path to storage config containing SQLite, Qdrant, and BM25 settings.",
    )
    ask_parser.add_argument(
        "--models-config",
        default="configs/models.yaml",
        help="Path to provider/model config containing embedding, reranker, and QA settings.",
    )
    ask_parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use deterministic local mock embeddings for vector query embedding.",
    )
    ask_parser.add_argument(
        "--mock-reranker",
        action="store_true",
        help="Use deterministic local mock reranking instead of a live reranker API.",
    )
    ask_parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip reranking and keep Union fused-score ordering.",
    )
    ask_parser.add_argument(
        "--mock-answer",
        action="store_true",
        help="Use deterministic local mock answer generation without calling a QA model.",
    )
    ask_parser.set_defaults(func=_ask)

    eval_parser = subparsers.add_parser(
        "eval",
        help="M1.9 eval: build long-context question sets and score chapter recall.",
    )
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command")
    eval_parser.set_defaults(func=_eval_help)

    eval_corpus_parser = eval_subparsers.add_parser(
        "build-corpus",
        help="Export persisted normalized chapters for long-context question generation.",
    )
    eval_corpus_parser.add_argument(
        "--document-id",
        help="Document id to export. Defaults to the latest SQLite document.",
    )
    eval_corpus_parser.add_argument(
        "--storage-config",
        default="configs/storage.yaml",
        help="Path to storage config containing the SQLite path.",
    )
    eval_corpus_parser.add_argument(
        "--chapter-start",
        type=int,
        default=1,
        help="First 1-based chapter index to include.",
    )
    eval_corpus_parser.add_argument(
        "--chapter-end",
        type=int,
        default=100,
        help="Last 1-based chapter index to include.",
    )
    eval_corpus_parser.add_argument("--output", help="Output corpus JSON path.")
    eval_corpus_parser.set_defaults(func=_eval_build_corpus)

    eval_generate_parser = eval_subparsers.add_parser(
        "generate",
        help="Call the configured long-context LLM to create a JSONL question set.",
    )
    eval_generate_parser.add_argument("corpus", help="Corpus JSON from eval build-corpus.")
    eval_generate_parser.add_argument(
        "--models-config",
        default="configs/models.yaml",
        help="Path to provider/model config containing eval_question_generator.",
    )
    eval_generate_parser.add_argument(
        "--question-count",
        type=int,
        default=200,
        help="Requested number of generated questions.",
    )
    eval_generate_parser.add_argument(
        "--profile",
        choices=("broad", "pinpoint", "mixed"),
        default="broad",
        help="Question generation profile. broad is the default recall stress test.",
    )
    eval_generate_parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Override eval_question_generator.max_output_tokens for this generation call.",
    )
    eval_generate_parser.add_argument("--output", help="Output question set JSONL path.")
    eval_generate_parser.set_defaults(func=_eval_generate)

    eval_run_parser = eval_subparsers.add_parser(
        "run",
        help="Run LoreWeaver retrieval for each question and score chapter-level recall.",
    )
    eval_run_parser.add_argument("questions", help="Question set JSONL path.")
    eval_run_parser.add_argument(
        "--document-id",
        help="Document id to retrieve from. Defaults to the latest SQLite document.",
    )
    eval_run_parser.add_argument(
        "--storage-config",
        default="configs/storage.yaml",
        help="Path to storage config containing SQLite, Qdrant, and BM25 settings.",
    )
    eval_run_parser.add_argument(
        "--models-config",
        default="configs/models.yaml",
        help="Path to provider/model config containing embedding and reranker settings.",
    )
    eval_run_parser.add_argument("--output", help="Output predictions JSONL path.")
    eval_run_parser.add_argument("--limit", type=int, help="Only run the first N questions.")
    eval_run_parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use deterministic local mock embeddings for vector query embedding.",
    )
    eval_run_parser.add_argument(
        "--mock-reranker",
        action="store_true",
        help="Use deterministic local mock reranking instead of a live reranker API.",
    )
    eval_run_parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip reranking and keep Union fused-score ordering.",
    )
    eval_run_parser.set_defaults(func=_eval_run)

    eval_report_parser = eval_subparsers.add_parser(
        "report",
        help="Recompute summary and failure markdown from an eval predictions JSONL.",
    )
    eval_report_parser.add_argument("predictions", help="Predictions JSONL from eval run.")
    eval_report_parser.set_defaults(func=_eval_report)

    return parser


def _status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run_id = new_run_id("status")
    sample_path = config.sample_source_path

    print(f"run_id: {run_id}")
    print(f"version: {__version__}")
    print(f"config: {config.path}")
    print(f"stage: {config.values.get('project', {}).get('stage', 'unknown')}")
    print(f"data_dir: {config.data_dir}")

    if sample_path is None:
        print("sample: not configured")
    else:
        status = "found" if sample_path.exists() else "missing"
        size = sample_path.stat().st_size if sample_path.exists() else 0
        print(f"sample: {sample_path} ({status}, {size} bytes)")

    required_dirs = [
        config.data_dir / "raw",
        config.data_dir / "normalized",
        config.data_dir / "runs",
        config.data_dir / "indexes",
        config.data_dir / "eval",
    ]
    missing_dirs = [str(path) for path in required_dirs if not Path(path).exists()]
    if missing_dirs:
        print("missing_dirs:")
        for path in missing_dirs:
            print(f"  - {path}")
        return 1

    print("bootstrap: ok")
    return 0


def _web(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError(
            "The Web UI requires optional web dependencies. "
            "Install them with: python -m pip install -e '.[web]'"
        ) from error

    from loreweaver.web.app import create_app

    app = create_app(
        config_path=args.config,
        storage_config_path=args.storage_config,
        models_config_path=args.models_config,
    )
    print(f"web_url: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _ingest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage_config = _load_storage_config(args.storage_config)
    run_id = new_run_id("ingest")
    progress, progress_sink = build_cli_progress_reporter(
        command="ingest",
        run_id=run_id,
        mode=args.progress,
    )

    source_path = Path(args.source) if args.source else config.sample_source_path
    if source_path is None:
        raise ValueError("No source path provided and sample.source_path is not configured.")

    try:
        report = ingest_text(
            config=config,
            storage_config=storage_config,
            run_id=run_id,
            source_path=source_path,
            title=args.title,
            author=args.author,
            max_chapters=args.max_chapters,
            progress=progress,
        )
    finally:
        _close_progress_sink(progress_sink)

    document = report["document"]
    split = report["chapter_split"]
    normalization = report["normalization"]
    warnings = split["boundary_warnings"]

    print(f"run_id: {run_id}")
    print("command: ingest")
    print(f"document_id: {document['document_id']}")
    print(f"content_hash: {document['content_hash']}")
    print(f"normalized_path: {document['normalized_path']}")
    print(f"sqlite_path: {report['sqlite_path']}")
    print(f"report_path: {report['report_path']}")
    print(f"total_chars: {document['total_chars']}")
    print(f"total_chapters: {document['total_chapters']}")
    print(f"chapter_strategy: {split['strategy']}")
    print(f"shortest_chapter_chars: {split['shortest_chapter_chars']}")
    print(f"longest_chapter_chars: {split['longest_chapter_chars']}")
    print(f"chars_removed_by_normalization: {normalization['chars_removed']}")
    print(f"boundary_warnings: {len(warnings)}")
    for warning in warnings[:10]:
        print(f"  - {warning}")
    if len(warnings) > 10:
        print(f"  - ... {len(warnings) - 10} more")
    print("status: ok")
    return 0


def _windows(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage_config = _load_storage_config(args.storage_config)
    run_id = new_run_id("windows")
    progress, progress_sink = build_cli_progress_reporter(
        command="windows",
        run_id=run_id,
        mode=args.progress,
    )

    try:
        report = build_candidate_windows(
            config=config,
            storage_config=storage_config,
            run_id=run_id,
            document_id=args.document_id,
            window_size_chars=args.window_size,
            overlap_ratio=args.overlap_ratio,
            min_window_chars=args.min_chars,
            max_window_chars=args.max_chars,
            window_mode=args.window_mode,
            progress=progress,
        )
    finally:
        _close_progress_sink(progress_sink)

    document = report["document"]
    split = report["window_split"]
    warnings = split["boundary_warnings"]

    print(f"run_id: {run_id}")
    print("command: windows")
    print(f"document_id: {document['document_id']}")
    print(f"normalized_path: {document['normalized_path']}")
    print(f"sqlite_path: {report['sqlite_path']}")
    print(f"report_path: {report['report_path']}")
    print(f"total_chapters: {split['total_chapters']}")
    print(f"requested_mode: {split['requested_mode']}")
    print(f"split_mode: {split['split_mode']}")
    print(f"total_windows: {split['total_windows']}")
    print(f"average_window_chars: {split['average_window_chars']}")
    print(f"shortest_window_chars: {split['shortest_window_chars']}")
    print(f"longest_window_chars: {split['longest_window_chars']}")
    print(f"short_window_count: {split['short_window_count']}")
    print(f"window_size_chars: {split['configured_window_size_chars']}")
    print(f"overlap_ratio: {split['configured_overlap_ratio']}")
    print(f"effective_stride_chars: {split['effective_stride_chars']}")
    print(f"boundary_warnings: {len(warnings)}")
    for warning in warnings[:10]:
        print(f"  - {warning}")
    if len(warnings) > 10:
        print(f"  - ... {len(warnings) - 10} more")
    print("status: ok")
    return 0


def _extract(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage_config = _load_storage_config(args.storage_config)
    models_config = load_config(args.models_config)
    if args.list_windows:
        report = list_extraction_windows(
            storage_config=storage_config,
            document_id=args.document_id,
            only=args.only,
            limit=args.limit,
        )
        print("command: extract")
        print("mode: list-windows")
        print(f"document_id: {report['document_id']}")
        print(f"sqlite_path: {report['sqlite_path']}")
        print(f"only: {report['only']}")
        print(f"window_count: {report['window_count']}")
        _print_extraction_window_status(report["windows"])
        print("status: ok")
        return 0

    run_id = new_run_id("extract")
    progress, progress_sink = build_cli_progress_reporter(
        command="extract",
        run_id=run_id,
        mode=args.progress,
    )
    try:
        report = extract_document_windows(
            config=config,
            storage_config=storage_config,
            models_config=models_config,
            run_id=run_id,
            document_id=args.document_id,
            limit=args.limit,
            offset=args.offset,
            window_ids=args.window_id,
            window_ranges=args.window_range,
            mock=args.mock,
            batch=args.batch,
            batch_id=args.batch_id,
            batch_model=args.batch_model,
            batch_wait=args.batch_wait,
            batch_poll_interval_seconds=args.batch_poll_interval,
            batch_timeout_seconds=args.batch_timeout,
            batch_completion_window=args.batch_completion_window,
            progress=progress,
        )
    finally:
        _close_progress_sink(progress_sink)

    print(f"run_id: {run_id}")
    print("command: extract")
    print(f"document_id: {report['document']['document_id']}")
    print(f"model: {report['model']}")
    print(f"mock: {report['mock']}")
    if report.get("mode") == "batch":
        print("mode: batch")
        print(f"batch_id: {report['batch_id']}")
        print(f"batch_status: {report['batch_status']}")
        if report.get("input_file_id"):
            print(f"input_file_id: {report['input_file_id']}")
        if report.get("output_file_id"):
            print(f"output_file_id: {report['output_file_id']}")
        if report.get("error_file_id"):
            print(f"error_file_id: {report['error_file_id']}")
    print(f"sqlite_path: {report['sqlite_path']}")
    print(f"report_path: {report['report_path']}")
    print(f"window_count: {report['window_count']}")
    print(f"span_count: {report['span_count']}")
    print(f"extraction_success_count: {report['extraction_success_count']}")
    print(f"extraction_failed_count: {report['extraction_failed_count']}")
    print(f"locator_success_rate: {report['locator_success_rate']}")
    print(f"estimated_input_tokens: {report['usage']['input_tokens']}")
    print(f"estimated_output_tokens: {report['usage']['output_tokens']}")
    print(f"estimated_cost_yuan: {report['estimated_cost_yuan']}")
    for failure in report["failed_windows"][:10]:
        print(f"failed: {failure['window_id']} - {failure['reason']}")
    print("status: ok")
    return 0


def _index(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage_config = _load_storage_config(args.storage_config)
    models_config = load_config(args.models_config)
    run_id = new_run_id("index")
    progress, progress_sink = build_cli_progress_reporter(
        command="index",
        run_id=run_id,
        mode=args.progress,
    )
    try:
        report = build_m14_indexes(
            config=config,
            storage_config=storage_config,
            models_config=models_config,
            run_id=run_id,
            document_id=args.document_id,
            limit=args.limit,
            mock_embeddings=args.mock_embeddings,
            progress=progress,
        )
    finally:
        _close_progress_sink(progress_sink)

    print(f"run_id: {run_id}")
    print("command: index")
    print(f"document_id: {report['document']['document_id']}")
    print(f"sqlite_path: {report['sqlite_path']}")
    print(f"report_path: {report['report_path']}")
    print(f"located_span_count: {report['located_span_count']}")
    print(f"embedding_model: {report['embedding']['model']}")
    print(f"embedding_dimensions: {report['embedding']['dimensions']}")
    print(f"embedding_cache_hits: {report['embedding']['cache_hits']}")
    print(f"embedding_cache_misses: {report['embedding']['cache_misses']}")
    print(f"embedding_cost_yuan: {report['embedding']['estimated_cost_yuan']}")
    print(f"qdrant_collection: {report['qdrant']['collection_name']}")
    print(f"qdrant_count: {report['qdrant']['collection_count']}")
    if report["qdrant"]["local_path"]:
        print(f"qdrant_local_path: {report['qdrant']['local_path']}")
    print(f"bm25_index_path: {report['bm25']['index_path']}")
    print(f"bm25_document_count: {report['bm25']['document_count']}")
    print("status: ok")
    return 0


def _search_vector(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage_config = _load_storage_config(args.storage_config)
    models_config = load_config(args.models_config)
    report = search_vector_index(
        config=config,
        storage_config=storage_config,
        models_config=models_config,
        query=args.query,
        document_id=args.document_id,
        top_k=args.top_k,
        mock_embeddings=args.mock_embeddings,
    )

    print("command: search-vector")
    print(f"document_id: {report['document_id']}")
    print(f"query: {report['query']}")
    print(f"result_count: {len(report['results'])}")
    _print_search_results(report["results"])
    print("status: ok")
    return 0


def _search_bm25(args: argparse.Namespace) -> int:
    storage_config = _load_storage_config(args.storage_config)
    report = search_bm25_index(
        storage_config=storage_config,
        query=args.query,
        document_id=args.document_id,
        top_k=args.top_k,
    )

    print("command: search-bm25")
    print(f"document_id: {report['document_id']}")
    print(f"query: {report['query']}")
    print(f"index_path: {report['index_path']}")
    print(f"result_count: {len(report['results'])}")
    _print_search_results(report["results"])
    print("status: ok")
    return 0


def _spans(args: argparse.Namespace) -> int:
    storage_config = _load_storage_config(args.storage_config)
    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    document = store.get_document(args.document_id)
    spans = store.list_top_salience_spans(document.document_id, limit=args.top_salience)

    print("command: spans")
    print(f"document_id: {document.document_id}")
    print(f"result_count: {len(spans)}")
    for index, span in enumerate(spans, start=1):
        print(
            f"{index}. span_id={span.span_id} "
            f"salience={span.salience_score:.3f} "
            f"type={span.span_type} "
            f"chapter_id={span.chapter_id} "
            f"range={span.span_start_idx}-{span.span_end_idx}"
        )
        print(f"   summary: {_truncate(span.summary, 120)}")
        if span.entities:
            print(f"   entities: {', '.join(span.entities[:8])}")
    print("status: ok")
    return 0


def _graph(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage_config = _load_storage_config(args.storage_config)
    if args.list:
        report = list_graph_clusters(
            storage_config=storage_config,
            document_id=args.document_id,
            cluster_id=args.cluster_id,
        )
        print("command: graph")
        print("mode: list")
        print(f"document_id: {report['document_id']}")
        print(f"cluster_count: {report['cluster_count']}")
        print(f"edge_count: {report['edge_count']}")
        _print_graph_clusters(report["clusters"])
        print("status: ok")
        return 0

    run_id = new_run_id("graph")
    sync_neo4j = None
    if args.sync_neo4j:
        sync_neo4j = True
    if args.no_neo4j:
        sync_neo4j = False
    progress, progress_sink = build_cli_progress_reporter(
        command="graph",
        run_id=run_id,
        mode=args.progress,
    )
    try:
        report = build_m15_graph(
            config=config,
            storage_config=storage_config,
            run_id=run_id,
            document_id=args.document_id,
            cluster_count=args.cluster_count,
            members_per_cluster=args.members_per_cluster,
            min_members=args.min_members,
            use_embeddings=False if args.no_embeddings else None,
            sync_neo4j=sync_neo4j,
            progress=progress,
        )
    finally:
        _close_progress_sink(progress_sink)

    print(f"run_id: {run_id}")
    print("command: graph")
    print("mode: build")
    print(f"document_id: {report['document_id']}")
    print(f"sqlite_path: {report['sqlite_path']}")
    print(f"report_path: {report['report_path']}")
    print(f"cluster_count: {report['cluster_count']}")
    print(f"edge_count: {report['edge_count']}")
    scoring = report.get("scoring", {})
    vector_load = scoring.get("vector_load", {})
    if vector_load:
        print(
            "vector_load: "
            f"{vector_load.get('loaded_count')}/{vector_load.get('requested_count')} "
            f"coverage={vector_load.get('coverage')} "
            f"source={vector_load.get('source')}"
        )
    print(f"edge_counts: {report['edge_counts']}")
    print(f"neo4j: {report['neo4j']}")
    _print_graph_clusters(report["clusters"])
    print("status: ok")
    return 0


def _retrieve(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage_config = _load_storage_config(args.storage_config)
    models_config = load_config(args.models_config)
    progress, progress_sink = build_cli_progress_reporter(
        command="retrieve",
        run_id=None,
        mode=args.progress,
    )
    try:
        report = retrieve_m16(
            config=config,
            storage_config=storage_config,
            models_config=models_config,
            question=args.question,
            document_id=args.document_id,
            mock_embeddings=args.mock_embeddings,
            mock_reranker=args.mock_reranker,
            no_reranker=args.no_reranker,
            progress=progress,
        )
    finally:
        _close_progress_sink(progress_sink)

    print(f"run_id: {report['run_id']}")
    print("command: retrieve")
    print(f"document_id: {report['document_id']}")
    print(f"query_type: {report['query_type']}")
    print(f"question: {report['question']}")
    print(f"report_path: {report['report_path']}")
    retrieval = report["retrieval"]
    print(
        "source_counts: "
        f"graph={retrieval['graph'].get('count', 0)} "
        f"vector={retrieval['vector'].get('count', 0)} "
        f"bm25={retrieval['bm25'].get('count', 0)} "
        f"union={retrieval['union'].get('candidate_count', 0)}"
    )
    print(
        "reranker: "
        f"{report['reranker']['provider']} "
        f"{report['reranker']['model']} "
        f"inputs={report['reranker']['input_count']}"
    )
    _print_retrieve_results(report["top_results"])
    print("status: ok")
    return 0


def _evidence(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage_config = _load_storage_config(args.storage_config)
    models_config = load_config(args.models_config)
    progress, progress_sink = build_cli_progress_reporter(
        command="evidence",
        run_id=None,
        mode=args.progress,
    )
    try:
        retrieval_report = retrieve_m16(
            config=config,
            storage_config=storage_config,
            models_config=models_config,
            question=args.question,
            document_id=args.document_id,
            mock_embeddings=args.mock_embeddings,
            mock_reranker=args.mock_reranker,
            no_reranker=args.no_reranker,
            progress=progress.child(command="retrieve") if progress is not None else None,
        )
        report = assemble_evidence_pack_from_retrieval_report(
            config=config,
            storage_config=storage_config,
            retrieval_report=retrieval_report,
            progress=progress,
        )
    finally:
        _close_progress_sink(progress_sink)
    pack = report["evidence_pack"]
    assembly = report["assembly"]

    print(f"run_id: {report['run_id']}")
    print("command: evidence")
    print(f"query_id: {report['query_id']}")
    print(f"document_id: {report['document_id']}")
    print(f"query_type: {report['query_type']}")
    print(f"question: {report['question']}")
    print(f"retrieval_report_path: {retrieval_report['report_path']}")
    print(f"evidence_report_path: {report['report_path']}")
    print(f"evidence_block_count: {assembly['evidence_block_count']}")
    print(f"evidence_chars: {assembly['evidence_chars']}")
    print(f"token_estimate: {pack['token_estimate']}")
    print(f"warnings: {len(assembly['warnings'])}")
    _print_evidence_blocks(pack["evidence_blocks"])
    print("status: ok")
    return 0


def _ask(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage_config = _load_storage_config(args.storage_config)
    models_config = load_config(args.models_config)
    progress, progress_sink = build_cli_progress_reporter(
        command="ask",
        run_id=None,
        mode=args.progress,
    )
    try:
        report = ask_m18(
            config=config,
            storage_config=storage_config,
            models_config=models_config,
            question=args.question,
            document_id=args.document_id,
            mock_embeddings=args.mock_embeddings,
            mock_reranker=args.mock_reranker,
            no_reranker=args.no_reranker,
            mock_answer=args.mock_answer,
            progress=progress,
        )
    finally:
        _close_progress_sink(progress_sink)

    validation = report["answer_validation"]
    qa = report["qa"]
    pack = report["evidence_pack"]
    print(f"run_id: {report['run_id']}")
    print("command: ask")
    print(f"query_id: {report['query_id']}")
    print(f"document_id: {report['document_id']}")
    print(f"query_type: {report['query_type']}")
    print(f"question: {report['question']}")
    print(f"retrieval_report_path: {report['source_retrieval_report_path']}")
    print(f"evidence_report_path: {report['source_evidence_report_path']}")
    print(f"answer_report_path: {report['report_path']}")
    print(
        "answer_model: "
        f"{qa['provider']} {qa['model']} mock={qa['mock']} "
        f"citations_ok={validation['ok']}"
    )
    if validation["errors"]:
        print(f"citation_errors: {'; '.join(validation['errors'])}")
    print(f"evidence_block_count: {len(pack['evidence_blocks'])}")
    _print_evidence_blocks(pack["evidence_blocks"])
    print("answer:")
    print(report["answer"])
    print("status: ok" if validation["ok"] else "status: citation_validation_failed")
    return 0 if validation["ok"] else 1


def _eval_help(args: argparse.Namespace) -> int:
    del args
    print("command: eval")
    print("subcommands: build-corpus, generate, run, report")
    print("status: missing_subcommand")
    return 1


def _eval_build_corpus(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage_config = _load_storage_config(args.storage_config)
    report = build_chapter_corpus(
        config=config,
        storage_config=storage_config,
        document_id=args.document_id,
        chapter_start=args.chapter_start,
        chapter_end=args.chapter_end,
        output_path=args.output,
    )
    print(f"run_id: {report['run_id']}")
    print("command: eval build-corpus")
    print(f"document_id: {report['document']['document_id']}")
    print(f"chapter_range: {report['chapter_start']}-{report['chapter_end']}")
    print(f"chapter_count: {report['chapter_count']}")
    print(f"char_count: {report['char_count']}")
    print(f"corpus_path: {report['corpus_path']}")
    print("status: ok")
    return 0


def _eval_generate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    models_config = load_config(args.models_config)
    report = generate_question_set(
        config=config,
        models_config=models_config,
        corpus_path=args.corpus,
        output_path=args.output,
        question_count=args.question_count,
        profile=args.profile,
        max_output_tokens=args.max_output_tokens,
    )
    print(f"run_id: {report['run_id']}")
    print("command: eval generate")
    print(f"document_id: {report['document_id']}")
    print(f"corpus_path: {report['corpus_path']}")
    print(f"question_set_path: {report['question_set_path']}")
    print(f"profile: {report['profile']}")
    print(f"generated_question_count: {report['generated_question_count']}")
    generator = report["generator"]
    print(f"generator: {generator['provider']} {generator['model']}")
    print(f"report_path: {report['report_path']}")
    print("status: ok")
    return 0


def _eval_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage_config = _load_storage_config(args.storage_config)
    models_config = load_config(args.models_config)
    progress, progress_sink = build_cli_progress_reporter(
        command="eval",
        run_id=None,
        mode=args.progress,
    )
    try:
        report = run_eval(
            config=config,
            storage_config=storage_config,
            models_config=models_config,
            question_set_path=args.questions,
            document_id=args.document_id,
            output_path=args.output,
            limit=args.limit,
            mock_embeddings=args.mock_embeddings,
            mock_reranker=args.mock_reranker,
            no_reranker=args.no_reranker,
            progress=progress,
        )
    finally:
        _close_progress_sink(progress_sink)
    _print_eval_summary(report, command="eval run")
    return 0


def _eval_report(args: argparse.Namespace) -> int:
    report = summarize_eval_run(args.predictions)
    _print_eval_summary(report, command="eval report")
    return 0


def _close_progress_sink(progress_sink: object | None) -> None:
    close = getattr(progress_sink, "close", None)
    if callable(close):
        close()


def _print_eval_summary(report: dict, *, command: str) -> None:
    metrics = report["metrics"]
    overall = metrics["overall"]
    print(f"run_id: {report.get('run_id', '<recomputed>')}")
    print(f"command: {command}")
    if report.get("document_id"):
        print(f"document_id: {report['document_id']}")
    print(f"question_count: {report['question_count']}")
    if report.get("predictions_path"):
        print(f"predictions_path: {report['predictions_path']}")
    print(f"summary_path: {report['report_path']}")
    print(f"failures_path: {report['failures_path']}")
    print(
        "overall: "
        f"recall@3={overall['weighted_recall_at_3']:.4f} "
        f"recall@20={overall['weighted_recall_at_20']:.4f} "
        f"ndcg@20={overall['ndcg_at_20']:.4f} "
        f"facet@20={overall['facet_coverage_at_20']:.4f} "
        f"noise@20={overall['noise_at_20']:.4f} "
        f"mrr={overall['mrr']:.4f}"
    )
    print("status: ok")


def _print_search_results(results: list[dict]) -> None:
    for result in results:
        summary = result.get("summary") or ""
        if len(summary) > 120:
            summary = summary[:117] + "..."
        print(
            f"{result['rank']}. span_id={result['span_id']} "
            f"score={result['score']:.6f} "
            f"chapter_id={result.get('chapter_id')} "
            f"range={result.get('span_start_idx')}-{result.get('span_end_idx')}"
        )
        print(f"   summary: {summary}")
        entities = result.get("entities") or []
        if entities:
            print(f"   entities: {', '.join(str(entity) for entity in entities[:8])}")


def _print_extraction_window_status(windows: list[dict]) -> None:
    for window in windows:
        print(
            f"{window['global_index']}. window_id={window['window_id']} "
            f"status={window['status']} "
            f"chapter={window['chapter_index']} "
            f"window_index={window['window_index']} "
            f"range={window['window_start']}-{window['window_end']} "
            f"chars={window['char_count']} "
            f"spans={window['span_count']} "
            f"located={window['located_count']} "
            f"failed={window['failed_count']}"
        )


def _print_graph_clusters(clusters: list[dict]) -> None:
    for index, cluster in enumerate(clusters, start=1):
        print(
            f"{index}. cluster_id={cluster['cluster_id']} "
            f"type={cluster['cluster_type']} "
            f"members={cluster['member_count']} "
            f"confidence={cluster['confidence']:.4f}"
        )
        print(f"   name: {cluster['cluster_name']}")
        print(f"   center_span_id: {cluster['center_span_id']}")
        print(f"   summary: {_truncate(cluster['summary'], 140)}")
        for member in cluster.get("members", [])[:8]:
            reasons = member.get("reasons")
            reason_text = f" reasons={','.join(reasons[:3])}" if reasons else ""
            print(
                f"   - span_id={member['span_id']} "
                f"score={member.get('score', 0):.4f} "
                f"type={member.get('span_type')} "
                f"chapter_id={member.get('chapter_id')} "
                f"range={member.get('range')}"
                f"{reason_text}"
            )
            components = member.get("component_scores") or {}
            if components:
                print(
                    "     scores: "
                    f"vector={components.get('vector', 0):.3f} "
                    f"entity={components.get('entity', 0):.3f} "
                    f"bm25={components.get('bm25', 0):.3f} "
                    f"chapter={components.get('chapter', 0):.3f} "
                    f"salience={components.get('salience', 0):.3f}"
                )
            summary = member.get("summary")
            if summary:
                print(f"     summary: {_truncate(summary, 100)}")


def _print_retrieve_results(results: list[dict]) -> None:
    for result in results:
        print(
            f"{result['rank']}. span_id={result['span_id']} "
            f"rerank={result['rerank_score']:.6f} "
            f"fused={result['fused_score']:.6f} "
            f"sources={','.join(result['sources'])} "
            f"chapter_id={result['chapter_id']} "
            f"range={result['span_start_idx']}-{result['span_end_idx']}"
        )
        print(f"   summary: {_truncate(result['summary'], 140)}")
        entities = result.get("entities") or []
        if entities:
            print(f"   entities: {', '.join(str(entity) for entity in entities[:8])}")


def _print_evidence_blocks(blocks: list[dict]) -> None:
    for block in blocks:
        preview = _truncate(block.get("text", "").replace("\n", " "), 140)
        print(
            f"{block['citation_id']} "
            f"chapter={block['chapter_title']} "
            f"range={block['start_idx']}-{block['end_idx']} "
            f"spans={','.join(block['source_span_ids'])} "
            f"sources={','.join(block['retrieval_sources'])}"
        )
        print(f"   text: {preview}")


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _placeholder(args: argparse.Namespace) -> int:
    run_id = new_run_id(args.command)
    print(f"run_id: {run_id}")
    print(f"command: {args.command}")
    print("status: placeholder")
    print("message: This command surface is ready; implementation begins in later M1 substages.")
    if args.args:
        print("received_args:")
        for item in args.args:
            print(f"  - {item}")
    return 0


def _load_storage_config(path: str) -> object:
    storage_path = Path(path)
    if storage_path.exists():
        return load_config(storage_path)
    return load_config("configs/default.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = sys.argv[1:] if argv is None else argv
    args, unknown_args = parser.parse_known_args(raw_argv)
    configure_logging(args.verbose)

    if not hasattr(args, "func"):
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        parser.print_help()
        return 0

    if unknown_args and hasattr(args, "args"):
        command_index = raw_argv.index(args.command)
        args.args = raw_argv[command_index + 1 :]
    elif unknown_args:
        parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
