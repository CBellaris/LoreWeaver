# LoreWeaver M1 当前交付文档

更新时间：2026-05-14

本文档以当前源码、配置和 git 工作树为准，用于后续开发人员快速接管 LoreWeaver 第一阶段。旧进度文档仅作为背景参考；若旧文档与本文档冲突，以本文档和源码为准。

## 1. 交付结论

LoreWeaver M1 已实现单本文本的证据化知识库闭环：原始 txt 小说文本可以被规范化、切章、生成候选窗口，通过 OpenAI-compatible LLM 抽取多 Span 并定位到原文坐标，再构建 SQLite 元数据、Qdrant 向量索引、BM25 索引、CenterSpanCluster 图结构，最终支持混合检索、Evidence Pack 组装、带引用校验的问答，以及章节级召回评估。

当前产品形态是研发/调试阶段，不是生产服务。CLI 是主入口；Web UI 是本地调试面板；评估框架已具备可运行骨架，但还需要完整评测集和端到端回归来稳定质量。

## 2. 项目背景与愿景

LoreWeaver 的长期目标是做一个面向超长篇连续文本的世界观分析引擎，服务网文作者、IP 架构师、设定考据者和剧本/游戏内容团队。它要解决的不是“问一句、搜几段、拼一个答案”的普通 RAG 问题，而是让系统能够围绕一整本作品进行全局感知、时序追踪和隐性设定整理，回答角色关系演变、势力兴衰、地理变迁、力量体系、历史伏笔等需要跨章节综合的问题。

这个项目的核心判断是：长篇文学文本不能被压缩成若干静态三元组，也不能只靠固定切块向量检索。传统 RAG 容易丢失上下文拓扑，知识图谱容易过度压缩文学语境，直接把整本书塞进长上下文模型又昂贵且不稳定。LoreWeaver 因此采用“原文档案馆 + 多层导航索引”的思路：原文始终是事实地基，Span、向量、BM25、图、聚类和报告都只是帮助模型快速找到证据的导航层。

第一阶段的工程目标已经收缩为“单书建库与证据问答”。它不试图一次完成完整的世界观分析平台，而是先验证最关键的可信闭环：离线建库可以慢，但每个抽取结果必须能回到 normalized 原文坐标；在线回答可以借助 LLM 综合，但关键判断必须能引用 Evidence Pack 中的原文证据。这个原则贯穿当前代码：模型负责发现语义 Span 和输出 anchor，程序负责反查坐标、合并区间、校验引用和落盘复盘。

从产品演进看，M1 是底座：证明单书可入库、可检索、可问答、可评估。后续阶段可以在这个底座上发展设定报告模板、中心 Span 聚合优化、半自动主题聚类、冲突/不确定性标注，以及更接近“世界观整理工具”的稳定输出层。

## 3. 当前仓库基线

- 当前分支：`main`
- 当前 HEAD：`379edda split extractor.py & refine`
- git 工作树：干净，无 staged/unstaged/merge-conflict 改动
- 本地分支状态：`main` 相对 `origin/main` ahead 5
- 包版本：`loreweaver 0.1.0`
- Python 要求：`>=3.11`

## 4. 当前功能边界

已实现范围：

- 单书 txt 输入，当前样本为《黎明之剑》前 260 章文件。
- UTF-8/UTF-8-SIG 读取、文本规范化、章节标题识别、整书 fallback 章节。
- 章节窗口与滑动窗口两种候选窗口策略，默认 `auto`。
- LLM 多 Span 结构化抽取，支持 live、mock、SiliconFlow/OpenAI-compatible batch。
- start/end anchor 定位、模糊匹配、过长 anchor 裁剪、定位候选记录、失败队列。
- Span 持久化、`located_text` 可选保存、窗口 `uncovered_text` 调试保存。
- Embedding 缓存、Qdrant 本地或远程向量索引、中文友好的 BM25 索引。
- CenterSpanCluster 图构建、Span 边构建、SQLite 图镜像、可选 Neo4j 同步。
- graph/vector/BM25 三路召回、Union 融合、可插拔 reranker、noop/mock fallback。
- Evidence Pack：基于 Top-K span 坐标扩展区间、合并、裁剪、生成 citation id。
- QA：基于 Evidence Pack 生成答案，校验引用，必要时触发一次引用修复。
- M1.9 eval：长上下文语料导出、问题集生成、章节级预测聚合、Recall/NDCG/MRR 等指标。
- 本地 Debug Web UI：查看文档、窗口、Span、报告、图摘要，触发后台任务，查看进度事件。

非目标或尚未完成范围：

- 多书、多租户、多用户、权限、认证、生产部署。
- 稳定的数据迁移和 schema version 管理。
- 真实端到端集成测试、性能压测、成本上限控制。
- 完整参数实验平台和检索质量 A/B 对比。
- 生产级 reranker 验证、Neo4j 大规模同步验证。
- 完整依赖锁文件。

## 5. 快速上手

安装基础包：

```bash
python3 -m pip install -e .
```

安装 M1 完整依赖：

```bash
python3 -m pip install -e ".[m1,web,dev]"
```

查看状态：

```bash
python3 -m loreweaver.cli status
python3 -m loreweaver.cli --help
```

无 API 成本的局部烟测：

```bash
python3 -m loreweaver.cli ingest
python3 -m loreweaver.cli windows
python3 -m loreweaver.cli extract --limit 2 --mock
python3 -m loreweaver.cli index --mock-embeddings
python3 -m loreweaver.cli graph --no-embeddings --no-neo4j
python3 -m loreweaver.cli retrieve "塞西尔家族为什么衰落？" --mock-embeddings --mock-reranker
python3 -m loreweaver.cli ask "塞西尔家族为什么衰落？" --mock-embeddings --mock-reranker --mock-answer
```

启动本地调试 UI：

```bash
python3 -m loreweaver.cli web
```

默认地址为 `http://127.0.0.1:7860`。

## 6. 配置与环境变量

主配置文件：

- `configs/default.yaml`：项目阶段、样本、ingest、window、extraction、locator、retrieval、eval、graph、indexing、evidence、qa 参数。
- `configs/models.yaml`：provider、模型 profile、服务绑定、定价。
- `configs/storage.yaml`：SQLite、Qdrant、Neo4j、BM25 路径和环境变量名。

本地环境模板：`.env.example`。`loreweaver.config.load_config()` 会自动加载 `.env`，且不会覆盖已有 shell 环境变量。

常用环境变量：

- `SILICONFLOW_API_KEY`：抽取、QA、Embedding、Reranker 的默认付费 provider。
- `DEEPSEEK_API_KEY`：M1.9 eval 问题生成默认 provider。
- `OPENAI_API_KEY`：备用 OpenAI-compatible provider。
- `QDRANT_URL` / `QDRANT_API_KEY`：远程 Qdrant。为空时使用本地 `data/indexes/qdrant`。
- `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD`：可选 Neo4j 同步。

当前模型绑定：

- extraction / qa：`Pro/deepseek-ai/DeepSeek-V3.2` via SiliconFlow。
- batch extraction：`deepseek-ai/DeepSeek-V3.1-Terminus`。
- embedding：`Qwen/Qwen3-Embedding-0.6B`，期望 1024 维。
- reranker：`Qwen/Qwen3-Reranker-0.6B`，配置中 `enabled: false`，默认可走 noop 或 mock。
- eval question generator：`deepseek-v4-pro` via DeepSeek。

## 7. CLI 命令面

全局参数：

- `--config`：默认 `configs/default.yaml`。
- `--verbose`：开启 debug 日志。
- `--progress`：`auto` / `rich` / `text` / `jsonl` / `none`。

主要命令：

| 命令 | 阶段 | 作用 | 常用参数 |
| --- | --- | --- | --- |
| `status` | 管理 | 查看配置、样本和 bootstrap 状态 | 无 |
| `web` | 管理 | 启动本地 FastAPI Debug UI | `--host`, `--port` |
| `ingest` | M1.1 | 读取、规范化、切章、写 SQLite | `--source`, `--title`, `--author`, `--max-chapters` |
| `windows` | M1.2 | 生成候选窗口 | `--document-id`, `--window-mode`, `--window-size`, `--overlap-ratio` |
| `extract` | M1.3 | LLM 抽取 Span 并定位 | `--limit`, `--offset`, `--window-id`, `--window-range`, `--mock`, `--batch`, `--batch-id`, `--batch-wait` |
| `extract --list-windows` | M1.3 debug | 查看窗口抽取状态 | `--only all|extracted|pending`, `--limit` |
| `index` | M1.4 | 构建 Qdrant 与 BM25 索引 | `--limit`, `--mock-embeddings` |
| `search-vector` | M1.4 debug | 向量检索调试 | `query`, `--top-k`, `--mock-embeddings` |
| `search-bm25` | M1.4 debug | BM25 检索调试 | `query`, `--top-k` |
| `spans` | M1.5 debug | 查看高 salience Span | `--top-salience` |
| `graph` | M1.5 | 构建或查看 CenterSpanCluster 图 | `--cluster-count`, `--members-per-cluster`, `--no-embeddings`, `--sync-neo4j`, `--list` |
| `retrieve` | M1.6 | 三路混合召回与 rerank | `question`, `--mock-embeddings`, `--mock-reranker`, `--no-reranker` |
| `evidence` | M1.7 | 召回并组装 Evidence Pack | 同 `retrieve` |
| `ask` | M1.8 | 召回、证据组装、生成带引用答案 | `--mock-answer` 以及 retrieve 参数 |
| `eval build-corpus` | M1.9 | 导出章节长上下文语料 | `--chapter-start`, `--chapter-end`, `--output` |
| `eval generate` | M1.9 | 生成 JSONL 评测问题集 | `corpus`, `--question-count`, `--profile`, `--max-output-tokens` |
| `eval run` | M1.9 | 跑章节级召回评测 | `questions`, `--limit`, `--no-reranker` |
| `eval report` | M1.9 | 汇总已有 predictions | `predictions` |

## 8. 模块职责

| 模块 | 关键文件 | 职责 |
| --- | --- | --- |
| `loreweaver/cli.py` | `cli.py` | CLI 参数定义、配置加载、命令调度、控制台输出。 |
| `loreweaver/config.py` | `config.py` | YAML/.env 加载，提供 `AppConfig`。 |
| `loreweaver/progress.py` | `progress.py` | 结构化进度事件，CLI 与 Web 共用。 |
| `loreweaver/ingest/` | `pipeline.py`, `normalizer.py`, `chapter_splitter.py`, `window_splitter.py` | 文本读取、规范化、章节切分、窗口生成。 |
| `loreweaver/extraction/` | `extractor.py`, `window_processing.py`, `locator.py`, `schemas.py`, `batch.py`, `persistence.py`, `reports.py` | LLM 抽取、JSON schema 解析、anchor 定位、batch 提交/回收/重试、Span 持久化。 |
| `loreweaver/indexing/` | `pipeline.py`, `embeddings.py` | Embedding 输入构造、缓存、Qdrant/BM25 索引构建与搜索。 |
| `loreweaver/graph/` | `center_span.py`, `edge_builder.py` | CenterSpanCluster 构建、成员评分、Span/Cluster 边构建、Neo4j 同步入口。 |
| `loreweaver/retrieval/` | `pipeline.py`, `graph_retriever.py`, `vector_retriever.py`, `bm25_retriever.py`, `union.py`, `reranker.py` | query routing、三路召回、候选融合、rerank。 |
| `loreweaver/evidence/` | `assembler.py`, `interval.py`, `citation.py` | 从检索结果生成可引用证据包。 |
| `loreweaver/qa/` | `answerer.py`, `prompts.py` | 基于 Evidence Pack 的答案生成、引用校验和修复。 |
| `loreweaver/eval/` | `corpus.py`, `generator.py`, `question_set.py`, `runner.py`, `metrics.py` | 章节级 eval 语料、问题、运行、指标。 |
| `loreweaver/model_services/` | `factory.py`, `config.py`, `clients/` | Provider/model profile 解析、OpenAI-compatible chat/embedding/batch、rerank HTTP、mock。 |
| `loreweaver/storage/` | `sqlite_store.py`, `qdrant_store.py`, `bm25_store.py`, `neo4j_store.py` | SQLite、Qdrant、BM25、Neo4j 存储适配。 |
| `loreweaver/models/` | `document.py`, `chapter.py`, `window.py`, `span.py`, `cluster.py`, `evidence.py` | dataclass 数据模型。 |
| `loreweaver/web/` | `app.py`, `api.py`, `jobs.py`, `inspectors.py`, `templates/`, `static/` | 本地 Debug UI、REST API、后台任务、SSE 进度。 |

## 9. 数据与持久化

目录约定：

```text
data/raw/          原始输入文本
data/normalized/   规范化后的坐标基准文本
data/runs/         SQLite、运行报告、batch 输入/输出、检索/问答报告
data/indexes/      Qdrant 本地索引、BM25 JSON 索引
data/eval/         eval corpus、question set、predictions、summary、failures
```

SQLite 默认路径：`data/runs/loreweaver_m1.sqlite3`。

主要 SQLite 表：

- `documents`：文档元数据、content hash、规范化文本路径。
- `chapters`：章节边界和标题。
- `candidate_windows`：候选窗口文本、原文坐标、`uncovered_text`。
- `ingest_reports` / `window_reports` / `extraction_reports` / `index_reports` / `graph_reports`：各阶段运行摘要，便于调试和复盘。
- `spans`：抽取出的 Span、实体、主题、salience、anchor、坐标、located text、locator 状态。
- `locator_candidates`：anchor 定位候选。
- `extraction_failures`：抽取、解析、定位失败队列。
- `embedding_cache`：embedding 输入和向量缓存。
- `center_span_clusters`：中心 Span 聚类。
- `span_edges`：Span/Cluster 图边。
- `query_runs`：检索或问答运行记录。
- `evidence_packs`：证据包和答案。

## 10. 完整运行建库流程

下面流程从空数据目录开始，描述如何完整重建一个可检索、可问答的单书知识库。当前仓库中的既有 SQLite、索引、报告、batch 输入输出都应视为测试产物；它们可能不完整、不同步或只覆盖部分章节，不应作为接管判断依据。

### 10.1 准备环境

推荐使用项目约定的 conda 环境；也可以使用任意 Python 3.11+ 环境。

```bash
conda run -n loreweaver python -m pip install -e ".[m1,web,dev]"
conda run -n loreweaver python -m loreweaver.cli status
```

如果不使用 conda，则保持同样的模块入口：

```bash
python3 -m pip install -e ".[m1,web,dev]"
python3 -m loreweaver.cli status
```

复制 `.env.example` 为 `.env`，至少按要运行的链路配置：

```text
SILICONFLOW_API_KEY=       # live extraction / embedding / qa / reranker
DEEPSEEK_API_KEY=          # eval question generation
QDRANT_URL=                # 可空，空值使用本地 data/indexes/qdrant
QDRANT_API_KEY=
NEO4J_URI=                 # 可选
NEO4J_USERNAME=
NEO4J_PASSWORD=
```

### 10.2 从空目录重建

如需从干净状态开始，先备份或删除生成目录中的旧测试产物。`data/raw/` 中的源文本按需保留。

```bash
rm -f data/runs/loreweaver_m1.sqlite3
rm -f data/runs/*_report.json data/runs/*_batch_*.jsonl data/runs/*_evidence_pack.json
rm -f data/indexes/*_bm25.json
rm -rf data/indexes/qdrant
```

这些命令会删除本地构建结果，请只在确认不需要旧测试产物时执行。

### 10.3 Ingest：原文规范化与章节入库

```bash
conda run -n loreweaver python -m loreweaver.cli ingest \
  --source data/raw/DawnSword_Chapter_1_260.txt
```

输出结果会建立 normalized 文本、`documents`、`chapters` 和 ingest 报告。`document_id` 由 normalized 文本 hash 决定；后续命令可以省略 `--document-id`，默认使用最新文档。

### 10.4 Windows：生成候选窗口

```bash
conda run -n loreweaver python -m loreweaver.cli windows
```

默认 `window.mode=auto`。检测到真实章节时，当前实现倾向每章一个窗口；如果只有整书 fallback 章节，则使用滑动窗口。窗口是 LLM 抽取输入，不是最终知识单元。

### 10.5 Extraction：抽取 Span 并定位坐标

先做无成本 smoke test：

```bash
conda run -n loreweaver python -m loreweaver.cli extract --limit 3 --mock
conda run -n loreweaver python -m loreweaver.cli extract --list-windows --only all --limit 10
```

确认链路正常后，可以选择 live 逐窗口抽取：

```bash
conda run -n loreweaver python -m loreweaver.cli extract --limit 10
conda run -n loreweaver python -m loreweaver.cli extract --window-range 11-30
conda run -n loreweaver python -m loreweaver.cli extract --window-id <window_id>
```

大批量抽取建议使用 batch：

```bash
conda run -n loreweaver python -m loreweaver.cli extract --batch --limit 1000
conda run -n loreweaver python -m loreweaver.cli extract --batch-id <batch_id>
conda run -n loreweaver python -m loreweaver.cli extract --batch --batch-wait --batch-poll-interval 60
```

批处理完成后，用窗口状态命令确认是否还有待处理窗口：

```bash
conda run -n loreweaver python -m loreweaver.cli extract --list-windows --only pending --limit 50
```

### 10.6 Index：构建 Embedding、Qdrant 与 BM25

无 API smoke test：

```bash
conda run -n loreweaver python -m loreweaver.cli index --mock-embeddings
```

真实索引：

```bash
conda run -n loreweaver python -m loreweaver.cli index
```

调试搜索：

```bash
conda run -n loreweaver python -m loreweaver.cli search-bm25 "高文" --top-k 5
conda run -n loreweaver python -m loreweaver.cli search-vector "塞西尔家族衰落" --top-k 5
```

### 10.7 Graph：构建中心 Span 图

```bash
conda run -n loreweaver python -m loreweaver.cli spans --top-salience 30
conda run -n loreweaver python -m loreweaver.cli graph --no-neo4j
conda run -n loreweaver python -m loreweaver.cli graph --list
```

如果要同步到本地 Neo4j，可先启动容器，再运行：

```bash
docker run -d --name loreweaver-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/loreweaver-test \
  neo4j:5

conda run -n loreweaver python -m loreweaver.cli graph --sync-neo4j
```

Neo4j 是可视化和调试路径，不是 M1 本地主链路的硬依赖。

### 10.8 Retrieve / Evidence / Ask：验证在线链路

```bash
conda run -n loreweaver python -m loreweaver.cli retrieve \
  "塞西尔家族为什么会衰落？" --no-reranker

conda run -n loreweaver python -m loreweaver.cli evidence \
  "塞西尔家族为什么会衰落，高文和这个家族的关系如何变化？" --no-reranker

conda run -n loreweaver python -m loreweaver.cli ask \
  "塞西尔家族和高文有什么关系？"
```

无 API 端到端 smoke test 可使用 mock：

```bash
conda run -n loreweaver python -m loreweaver.cli ask \
  "塞西尔家族和高文有什么关系？" \
  --mock-embeddings --mock-reranker --mock-answer
```

### 10.9 Eval：章节级召回评估

```bash
conda run -n loreweaver python -m loreweaver.cli eval build-corpus \
  --chapter-start 1 --chapter-end 100

conda run -n loreweaver python -m loreweaver.cli eval generate \
  data/eval/corpora/<corpus>.json \
  --profile broad \
  --question-count 50 \
  --max-output-tokens 384000

conda run -n loreweaver python -m loreweaver.cli eval run \
  data/eval/question_sets/<question_set>.jsonl \
  --no-reranker

conda run -n loreweaver python -m loreweaver.cli eval report \
  data/eval/runs/<run_id>_predictions.jsonl
```

M1.9 的 gold label 是章节级，适合做自动回归、坏例筛选和参数对比。正式质量判断仍应抽查低分样本和低置信样本。

## 11. 阶段实现说明

### M1.1 Ingest

入口：`loreweaver.ingest.pipeline.ingest_text()`。

流程：读取 raw txt，规范化换行和空行，计算 normalized text SHA-256，生成 `document_id=doc_<hash12>`，写入 `data/normalized/<document_id>.txt`，按配置的章节正则切章，持久化 `documents` / `chapters` / `ingest_reports`。

特性：若没有章节标题，会 fallback 为整书一个章节。重新 ingest 同一 document 会清理该文档下游 windows、extraction、graph、evidence 输出。

### M1.2 Windows

入口：`loreweaver.ingest.window_splitter.build_candidate_windows()`。

当前默认 `window.mode=auto`。检测到真实章节时，使用每章一个窗口；fallback 整书章节时，使用滑动窗口。窗口写入 `candidate_windows`，报告写入 `window_reports` 和 `data/runs/*_windows_report.json`。

重新生成窗口会清理该文档下游 extraction、graph、evidence 输出。

### M1.3 Extraction

入口：`loreweaver.extraction.extractor.extract_document_windows()`。

支持三种执行方式：

- `--mock`：本地 deterministic mock，用于单元测试和无 API 烟测。
- live：逐窗口调用配置的 chat provider。
- `--batch` / `--batch-id`：提交或回收 OpenAI-compatible batch，支持 batch 输出解析、失败窗口重试、batch 报告。

LLM 输出 schema 位于 `loreweaver/extraction/schemas.py`。抽取后会做 anchor 定位，定位结果写入 `spans`、`locator_candidates`；失败写入 `extraction_failures`。配置项 `store_located_text` 与 `store_uncovered_text` 控制调试文本持久化。

注意：当前源码中非 batch extraction 在启用 progress 且走到完成事件时，`extractor.py` 使用了未定义局部变量 `report_path`。CLI 默认 `--progress auto` 可能触发该路径。修复方式应改为读取 `_persist_extraction_report()` 已写入的 `report["report_path"]`，或让该函数返回 `report_path`。

### M1.4 Indexing

入口：`loreweaver.indexing.pipeline.build_m14_indexes()`。

只索引 `locator_status='located'` 的 Span。Embedding 输入由 summary、entities 等字段组合，是否包含 `key_quote` 和 `located_text` 由 `indexing.embedding_input` 控制。Embedding 缓存在 SQLite 的 `embedding_cache`。向量进入 Qdrant collection：`loreweaver_<document_id>_spans`；BM25 保存到 `data/indexes/<document_id>_bm25.json`。

如果 Span 表变化，需要重新运行 `index`，否则检索会基于旧索引。

### M1.5 Graph

入口：`loreweaver.graph.center_span.build_m15_graph()`。

算法从 located spans 中按类型、实体、主题、salience 等信号选中心 Span，构建 `CenterSpanCluster`，再由 `edge_builder` 生成图边。可加载 Qdrant 向量增强成员评分；也可 `--no-embeddings` 走纯规则。结果写入 SQLite，也可 `--sync-neo4j` 同步到 Neo4j。

当前配置默认：`cluster_count=4`，`members_per_cluster=8`，`min_members_per_cluster=5`。

### M1.6 Retrieval

入口：`loreweaver.retrieval.pipeline.retrieve_m16()`。

步骤：query routing、graph retrieval、vector retrieval、BM25 retrieval、Union merge、rerank、报告持久化。Union 会保留各来源分数和归一化分数。Reranker 当前可用 noop、mock 或配置的 HTTP rerank provider；默认配置里 Qwen reranker disabled。

检索结果写入 `data/runs/*_retrieval_report.json` 和 `query_runs`。

### M1.7 Evidence Pack

入口：`loreweaver.evidence.assembler.assemble_evidence_pack_from_retrieval_report()`。

从 retrieval top results 中读取 span 坐标，扩展前后文，按章节边界裁剪，合并近邻区间，按 `max_evidence_chars` 和 `max_blocks` 做预算选择，生成 `E001` 形式 citation id。结果写入 `data/runs/*_evidence_pack.json` 和 `evidence_packs`。

### M1.8 QA

入口：`loreweaver.qa.answerer.ask_m18()`。

先调用 retrieval，再组装 Evidence Pack，最后用 QA 模型生成答案。答案必须引用 Evidence Pack 内存在的 citation id；若校验失败且有证据块，会触发一次 repair prompt。结果写回 `evidence_packs.answer`，并写入 answer report。

### M1.9 Eval

入口：`loreweaver.eval.*`。

`build-corpus` 从 SQLite 章节坐标导出长上下文 JSON；`generate` 调用长上下文模型生成 JSONL question set；`run` 对每个问题调用 retrieval，并把 span-level top results 聚合为 chapter-level ranking；`report` 汇总 predictions。指标包括 weighted recall、hit、NDCG、core recall、facet coverage、noise、MRR。

M1.9 更像评估基础设施交付：问题集、predictions、summary 和 failures 应按目标作品与评估范围重新生成。章节级 gold label 适合做自动回归和坏例筛选，正式质量判断仍需要人工抽查。

### Web Debug UI

入口：`loreweaver.web.app.create_app()`，CLI 命令为 `web`。

后端为 FastAPI，前端为 Jinja2 + 静态 JS/CSS。主要 API：overview、documents、windows、spans、span-review、graph、reports、commands、jobs、Neo4j status/start。任务通过 `JobManager` 后台线程执行，使用 SSE 推送 progress event。Web UI 支持从 payload 临时注入环境变量，但不提供生产认证或访问控制。

## 12. 测试状态

单元测试目录：`tests/unit/`。

当前测试文件覆盖：

- `test_m1_1_ingest.py`
- `test_m1_2_windows.py`
- `test_m1_3_extraction.py`
- `test_m1_4_indexing.py`
- `test_m1_5_graph.py`
- `test_m1_6_retrieval.py`
- `test_m1_7_evidence.py`
- `test_m1_8_qa.py`
- `test_m1_9_eval.py`
- `test_model_services.py`
- `test_web_inspectors.py`

源码中未发现显式 `TODO`、`FIXME` 或 `NotImplementedError`。`tests/integration/` 目前只有 `.gitkeep`，没有集成测试。后续接管时应至少补一条 mock 端到端 smoke test，覆盖 ingest -> windows -> extract mock -> index mock -> graph no-embedding -> retrieve mock -> evidence -> ask mock。

推荐测试命令：

```bash
python3 -m pytest tests/unit/
python3 -m ruff check loreweaver tests
```

## 13. 已知风险和接管注意事项

1. 配置阶段标识落后：`configs/default.yaml` 仍写 `M1.8`，README 和源码已扩展到 M1.9。
2. 非 batch extraction 的 progress 完成事件存在 `report_path` 未定义风险。
3. Reranker provider 配置存在但 disabled，生产质量需要验证真实 rerank。
4. Neo4j 默认为 disabled，本地容器脚本可用，但未证明大规模同步性能。
5. 当前默认窗口策略在真实章节存在时倾向一章一个窗口，可能显著大于早期计划中的小窗口；后续评估需确认对抽取成本和质量的影响。
6. 原文坐标依赖 normalized text。任何规范化规则变更都会使已有 span 坐标、索引和证据包失效。
7. 依赖没有 lock file，不同机器可能解析出不同版本。交付生产或长期实验前建议补 `uv.lock`、`requirements.lock` 或等价方案。
8. `data/*` 生成产物被 `.gitignore` 忽略；需要迁移环境时，应选择重新建库，或显式打包同一批 SQLite、normalized 文本、Qdrant/BM25 索引与报告。
9. 工作区存在 `__pycache__` 等本地缓存文件，受 `.gitignore` 保护，不应作为交付内容。
10. 样本文本为完整小说片段，后续分发或公开演示前需要确认文本来源与授权边界。

## 14. 建议的下一步接管路线

短期修复：

- 修复 `extractor.py` 中 progress 完成事件的 `report_path` 引用。
- 将 `configs/default.yaml` 的 `project.stage` 与 README/源码统一到 M1.9。
- 按第 10 节从空目录完整重建一次单书库，确认 ingest、windows、extract、index、graph、retrieve、evidence、ask 都能连续运行。
- 为重建流程沉淀一条无 API mock smoke test 和一条 live 小样本 smoke test。

质量稳定：

- 增加 mock 端到端集成测试。
- 为 eval 生成一份小规模可复跑 question set，并把 `eval run --limit` 纳入回归。
- 建立索引与 Span 表一致性检查，例如记录 indexed span count、latest extraction run id、document content hash。
- 为 reranker 做真实 provider smoke test，并记录 latency、成本、质量增益。

工程化：

- 补依赖锁文件和最小 CI。
- 明确数据产物迁移/备份策略。
- 给 SQLite schema 增加版本号或迁移机制。
- 将 Web UI 标记为本地调试用途，避免误作生产服务部署。
