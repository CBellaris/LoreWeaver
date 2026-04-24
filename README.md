# LoreWeaver

LoreWeaver is an LLM-driven analysis engine for long-form fictional worlds. The current implementation is starting with Milestone 1: single-book ingestion and evidence-grounded question answering.

## Current Stage

M1.5 is the lightweight Center Span graph skeleton stage. The current implementation provides:

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
- M1.5 high-salience Span inspection, embedding-aware CenterSpanCluster construction with deterministic rule fallback, SQLite graph mirror, optional Neo4j sync, graph reports, and cluster member inspection

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
loreweaver spans --top-salience 30
loreweaver graph
loreweaver retrieve
loreweaver ask
loreweaver eval
```

`ingest`, `windows`, `extract`, `index`, `search-vector`, `search-bm25`, `spans`, and `graph` are implemented; later M1 commands still report their run id and placeholder status.

For a paid API smoke test, set `SILICONFLOW_API_KEY` in your environment and start with a small limit:

```bash
python3 -m pip install -e ".[m1]"
python3 -m loreweaver.cli extract --limit 10
```

For local plumbing checks without API calls:

```bash
python3 -m loreweaver.cli extract --limit 10 --mock
python3 -m loreweaver.cli extract --list-windows --only pending --limit 20
python3 -m loreweaver.cli extract --window-range 21-40
python3 -m loreweaver.cli extract --window-id doc_59331b17113e_ch0021_win0001
python3 -m loreweaver.cli index --mock-embeddings
python3 -m loreweaver.cli spans --top-salience 30
python3 -m loreweaver.cli graph --no-neo4j
python3 -m loreweaver.cli graph --no-neo4j --no-embeddings
python3 -m loreweaver.cli graph --list
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
The CLI loads local `.env` values automatically without overriding existing shell variables.

## Local Neo4j Visualization

For local graph inspection, run a test Neo4j container and sync the current graph:

```bash
docker run -d --name loreweaver-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/loreweaver-test \
  neo4j:5

python3 -m loreweaver.cli graph --sync-neo4j
```

Open `http://localhost:7474` and log in with:

```text
username: neo4j
password: loreweaver-test
```

Useful starter query:

```cypher
MATCH p=(:LoreWeaverCluster)-[:SUPPORTS]->(:LoreWeaverSpan)
RETURN p
LIMIT 80;
```
