"""LoreWeaver command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loreweaver import __version__
from loreweaver.config import load_config
from loreweaver.extraction.extractor import extract_document_windows
from loreweaver.ingest.pipeline import ingest_text
from loreweaver.ingest.window_splitter import build_candidate_windows
from loreweaver.logging import configure_logging, new_run_id


PIPELINE_COMMANDS = (
    "index",
    "graph",
    "retrieve",
    "ask",
    "eval",
)


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

    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser("status", help="Show project bootstrap status.")
    status_parser.set_defaults(func=_status)

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
        help="M1.2 windows: split persisted chapters into overlapping candidate windows.",
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
        "--by-chapter",
        action="store_true",
        help="Use each natural chapter as one candidate window instead of sliding windows.",
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
        help="Extract exactly one candidate window.",
    )
    extract_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic local mock extraction without calling an LLM API.",
    )
    extract_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-window extraction progress output.",
    )
    extract_parser.set_defaults(func=_extract)

    for command in PIPELINE_COMMANDS:
        command_parser = subparsers.add_parser(
            command,
            help=f"M1 command placeholder: {command}.",
        )
        command_parser.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="Arguments reserved for later M1 stages.",
        )
        command_parser.set_defaults(func=_placeholder)

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


def _ingest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    storage_config = _load_storage_config(args.storage_config)
    run_id = new_run_id("ingest")

    source_path = Path(args.source) if args.source else config.sample_source_path
    if source_path is None:
        raise ValueError("No source path provided and sample.source_path is not configured.")

    report = ingest_text(
        config=config,
        storage_config=storage_config,
        run_id=run_id,
        source_path=source_path,
        title=args.title,
        author=args.author,
        max_chapters=args.max_chapters,
    )

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

    report = build_candidate_windows(
        config=config,
        storage_config=storage_config,
        run_id=run_id,
        document_id=args.document_id,
        window_size_chars=args.window_size,
        overlap_ratio=args.overlap_ratio,
        min_window_chars=args.min_chars,
        max_window_chars=args.max_chars,
        split_by_chapter=args.by_chapter,
    )

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
    run_id = new_run_id("extract")
    progress = _build_extract_progress_printer() if not args.no_progress else None

    report = extract_document_windows(
        config=config,
        storage_config=storage_config,
        models_config=models_config,
        run_id=run_id,
        document_id=args.document_id,
        limit=args.limit,
        offset=args.offset,
        window_id=args.window_id,
        mock=args.mock,
        progress_callback=progress,
    )

    print(f"run_id: {run_id}")
    print("command: extract")
    print(f"document_id: {report['document']['document_id']}")
    print(f"model: {report['model']}")
    print(f"mock: {report['mock']}")
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


def _build_extract_progress_printer() -> object:
    totals = {
        "spans": 0,
        "located": 0,
        "failed": 0,
        "cost": 0.0,
    }

    def progress(event: str, payload: dict) -> None:
        if event == "planned":
            print(
                "[extract] "
                f"document={payload['document_id']} "
                f"model={payload['model']} "
                f"mock={payload['mock']} "
                f"windows={payload['total_windows']}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "window_start":
            print(
                "[extract] "
                f"window {payload['window_index']}/{payload['total_windows']} "
                f"{payload['window_id']} "
                f"chars={payload['char_count']} ...",
                file=sys.stderr,
                flush=True,
            )
        elif event == "api_start":
            print(
                "[extract] "
                f"api start {payload['window_index']}/{payload['total_windows']} "
                f"{payload['window_id']} "
                f"attempt={payload['attempt']}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "api_done":
            usage = ""
            if payload["input_tokens"] or payload["output_tokens"]:
                usage = (
                    f" input_tokens={payload['input_tokens']}"
                    f" output_tokens={payload['output_tokens']}"
                )
            print(
                "[extract] "
                f"api done {payload['window_index']}/{payload['total_windows']} "
                f"{payload['window_id']} "
                f"attempt={payload['attempt']} "
                f"elapsed={payload['elapsed_seconds']}s"
                f"{usage}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "parse_locate_done":
            print(
                "[extract] "
                f"parse+locate {payload['window_index']}/{payload['total_windows']} "
                f"{payload['window_id']} "
                f"spans={payload['span_count']} "
                f"located={payload['located_count']} "
                f"failed={payload['failed_count']} "
                f"elapsed={payload['elapsed_seconds']}s",
                file=sys.stderr,
                flush=True,
            )
        elif event == "db_write_done":
            print(
                "[extract] "
                f"db write {payload['window_index']}/{payload['total_windows']} "
                f"{payload['window_id']} "
                f"elapsed={payload['elapsed_seconds']}s",
                file=sys.stderr,
                flush=True,
            )
        elif event == "window_done":
            totals["spans"] += payload["span_count"]
            totals["located"] += payload["located_count"]
            totals["failed"] += payload["failed_count"]
            totals["cost"] += payload["estimated_cost_yuan"]
            print(
                "[extract] "
                f"done {payload['window_index']}/{payload['total_windows']} "
                f"{payload['window_id']} "
                f"spans={payload['span_count']} "
                f"located={payload['located_count']} "
                f"failed={payload['failed_count']} "
                f"cost=CNY {payload['estimated_cost_yuan']:.6f} "
                f"uncovered_chars={payload['uncovered_chars']} "
                f"elapsed={payload['elapsed_seconds']}s "
                f"total_spans={totals['spans']} "
                f"total_failed={totals['failed']} "
                f"total_cost=CNY {totals['cost']:.6f}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "completed":
            print(
                "[extract] "
                f"completed spans={payload['span_count']} "
                f"located={payload['located_count']} "
                f"failed={payload['failed_count']} "
                f"cost=CNY {payload['estimated_cost_yuan']:.6f} "
                f"report={payload['report_path']}",
                file=sys.stderr,
                flush=True,
            )

    return progress


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
