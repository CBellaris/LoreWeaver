# LoreWeaver

LoreWeaver is an LLM-driven analysis engine for long-form fictional worlds. The current implementation is starting with Milestone 1: single-book ingestion and evidence-grounded question answering.

## Current Stage

M1.4 is the metadata, vector index, and BM25 index stage. The current implementation provides:

- Python package skeleton
- CLI entry point
- centralized configuration files
- data directory layout
- first raw sample registration
- run id and logging helpers
- M1.1 raw text ingestion, normalization, chapter splitting, and SQLite metadata
- M1.2 chapter-bounded overlapping candidate windows with SQLite records and JSON reports
- M1.3 OpenAI-compatible LLM extraction, multi-Span discovery per window, start/end anchor location with overlong-anchor trimming, Span persistence with optional `located_text`, window-level `uncovered_text` debug output, failure queues, progress timing, and API cost estimates
- M1.4 embedding cache in SQLite, local-or-remote Qdrant vector indexing, local BM25 indexing with Chinese-friendly tokenization, and standalone `search-vector` / `search-bm25` debug commands

The first test sample is:

```text
data/raw/DawnSword_Chapter_1_260.txt
```

## Quick Start

Run the CLI directly from the workspace:

```bash
python3 -m loreweaver.cli --help
python3 -m loreweaver.cli status
```

After installing the package, the same CLI will be available as:

```bash
loreweaver --help
loreweaver status
```

## M1 Command Surface

The command surface is intentionally created before the full pipeline is implemented:

```bash
loreweaver ingest
loreweaver windows
loreweaver extract
loreweaver index
loreweaver search-vector "问题"
loreweaver search-bm25 "问题"
loreweaver graph
loreweaver retrieve
loreweaver ask
loreweaver eval
```

`ingest`, `windows`, `extract`, `index`, `search-vector`, and `search-bm25` are implemented; later M1 commands still report their run id and placeholder status.

For a paid API smoke test, set `SILICONFLOW_API_KEY` in your environment and start with a small limit:

```bash
python3 -m pip install -e ".[m1]"
python3 -m loreweaver.cli extract --limit 10
```

For local plumbing checks without API calls:

```bash
python3 -m loreweaver.cli extract --limit 10 --mock
python3 -m loreweaver.cli index --mock-embeddings
```

## Data Directories

```text
data/raw/          source text files
data/normalized/   normalized canonical text files
data/runs/         command run outputs and diagnostics
data/indexes/      generated vector/BM25/local indexes
data/eval/         evaluation questions and reports
```

Generated artifacts should include a `document_id` or `run_id` so they can be inspected and reproduced.

## Configuration

Main configuration lives in:

```text
configs/default.yaml
configs/models.yaml
configs/storage.yaml
```

Copy `.env.example` to `.env` when model or database credentials are needed.
