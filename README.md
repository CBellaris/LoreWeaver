# LoreWeaver

LoreWeaver is an LLM-driven analysis engine for long-form fictional worlds. The current implementation is starting with Milestone 1: single-book ingestion and evidence-grounded question answering.

## Current Stage

M1.9 is the chapter-level evaluation stage. The current implementation provides:

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
- M1.6 graph + vector + BM25 hybrid retrieval, Union candidate fusion, pluggable reranker interface, SiliconFlow reranker provider, mock/noop reranker fallbacks, retrieval reports, and `query_runs` persistence
- M1.7 Evidence Pack assembly from retrieval Top-K spans, interval expansion/merge, citation ids, and SQLite persistence
- M1.8 evidence-grounded QA with citation validation and repair
- M1.9 long-context LLM question-set generation, chapter-level gold labels, span-to-chapter prediction aggregation, Recall/NDCG/MRR scoring, and failure reports

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

Start the local debugging Web UI with live server logs:

```bash
conda run --no-capture-output -n loreweaver python -m loreweaver.cli web
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
loreweaver eval build-corpus
loreweaver eval generate
loreweaver eval run
loreweaver eval report
```

`ingest`, `windows`, `extract`, `index`, `search-vector`, `search-bm25`, `spans`, `graph`, `retrieve`, `evidence`, `ask`, and `eval` are implemented.

For a paid API smoke test, set `SILICONFLOW_API_KEY` in your environment and start with a small limit:

```bash
python3 -m pip install -e ".[m1]"
python3 -m loreweaver.cli extract --limit 10
```

For larger extraction runs through SiliconFlow batch mode:

```bash
python3 -m loreweaver.cli extract --batch --limit 1000
python3 -m loreweaver.cli extract --batch-id <batch_id>
python3 -m loreweaver.cli extract --batch --batch-wait --batch-poll-interval 60
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
python3 -m loreweaver.cli retrieve "塞西尔家族为什么衰落？" --mock-embeddings --mock-reranker
```

Progress output is controlled by the global `--progress` option:

```bash
python3 -m loreweaver.cli --progress rich extract --limit 10 --mock
python3 -m loreweaver.cli --progress jsonl ask "塞西尔家族为什么衰落？" --mock-embeddings --mock-reranker --mock-answer
python3 -m loreweaver.cli --progress none index --mock-embeddings
```

CLI rendering and Web UI task updates are both fed by the same structured progress
event stream.

For M1.9 chapter-level recall evaluation, build a long-context corpus, generate a JSONL question set with the configured OpenAI-compatible eval model, then run LoreWeaver retrieval against that set:

```bash
python3 -m loreweaver.cli eval build-corpus --chapter-start 1 --chapter-end 100
python3 -m loreweaver.cli eval generate data/eval/corpora/doc_59331b17113e_ch001_100.json --profile broad --question-count 50 --max-output-tokens 384000
python3 -m loreweaver.cli eval run data/eval/question_sets/doc_59331b17113e_ch001_100_broad_v001.jsonl --no-reranker
python3 -m loreweaver.cli eval report data/eval/runs/<run_id>_predictions.jsonl
```

Set `DEEPSEEK_API_KEY` to use the default `deepseek-v4-pro` eval question generator.

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
