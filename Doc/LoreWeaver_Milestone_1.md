# LoreWeaver Milestone 1 详细落地计划

版本：v0.1 计划稿  
阶段代号：M1 - 单书建库与证据问答  
目标周期：建议 3-5 周，按个人开发节奏可拉长  
阶段定位：把 LoreWeaver 从概念计划推进为一条可运行、可复盘、可验收的最小闭环

项目文档链接：[LoreWeaver.md](/LoreWeaver.md)

conda环境：loreweaver
请在环境内运行测试，例如：`conda run -n loreweaver python -m loreweaver.cli`

---

## 1. Milestone 1 总目标

Milestone 1 的目标不是一次性完成完整的“世界观分析引擎”，而是先跑通首个可信闭环：

> 对一份 3-5 万字的连续文本样本，完成原文入库、章节切分、Span 抽取、引用定位、向量/BM25/图索引写入、混合召回、证据区间合并，并最终支持带原文引用的宏观问答。

M1 完成后，系统应该能回答类似问题：

- “这个势力与主角的关系如何变化？”
- “故事中力量体系目前有哪些明确规则？”
- “某个地点在剧情中承担了什么作用？”
- “哪些事件暗示了某个隐藏设定？”

回答不要求像最终产品一样优雅，但必须做到：

- 能说明证据来自哪些章节或原文区间；
- 关键判断能附带原文引用；
- 证据不足时能明确说“不足以确认”；
- 建库、召回、回答链路可以重复运行和调试。

---

## 2. M1 产品边界

### 2.1 必做范围

M1 只服务于“单书、单样本、离线建库、在线证据问答”。

必须完成：

- 支持导入一个 `.txt` 文本样本；
- 识别章节边界并建立全局字符坐标；
- 将章节文本切成带重叠的候选窗口；
- 调用 LLM 在窗口内发现多个微观 Span；
- 用程序反查 Span 的 start/end anchor，生成可信坐标；
- 将 Span 元数据写入工程数据库；
- 建立摘要向量索引；
- 建立 BM25 关键词索引；
- 建立最小中心 Span 图骨架；
- 实现图召回、向量召回、BM25 召回的 Union 合并；
- 接入 Reranker 或预留可替换精排接口；
- 合并证据区间，切出原文证据包；
- 生成带引用的回答；
- 保存每次查询的证据包与调试记录；
- 建立最小评估集并跑通人工验收。

### 2.2 明确不做

M1 不做以下内容：

- 不做完整 Web UI；
- 不做多用户系统；
- 不做通用知识图谱平台；
- 不做全自动“概念锻造炉”；
- 不做全书百万字规模优化；
- 不做复杂 Agent 调度；
- 不追求 Neo4j 可视化炫技；
- 不承诺任意问题都有答案；
- 不做报告生成主链路，报告放到 M2。

### 2.3 M1 的成功定义

M1 成功的标志是：即使界面只有 CLI，系统也能稳定完成一次完整链路：

```text
txt 样本
  -> 章节切分
  -> 候选窗口
  -> LLM 抽取
  -> 引用定位
  -> Span 入库
  -> 向量/BM25/图索引
  -> 用户问题
  -> 混合召回
  -> Rerank
  -> 原文证据包
  -> 带引用回答
  -> 评估记录
```

---

## 3. M1 核心设计原则

### 3.1 原文坐标是系统地基

任何摘要、聚合、回答都不作为最终事实来源。最终可信依据只能来自：

- `document_id`
- `chapter_id`
- `start_idx`
- `end_idx`
- 原文切片内容

模型不得直接写入最终可信坐标。模型只输出 `start_anchor_quote` / `end_anchor_quote`，程序负责反查并组合定位。

### 3.2 先做可复盘，再做智能化

M1 每一步都必须能落盘：

- 每个窗口为什么生成哪些微观 Span；
- 每个 Span 的 anchor 是否定位成功；
- 每条召回结果来自图、向量还是 BM25；
- Reranker 为什么保留某些候选；
- 最终证据包包含哪些原文区间；
- 回答引用对应哪些证据块。

只要一次回答质量不好，必须能追到是抽取、定位、召回、精排、合并还是生成环节出了问题。

### 3.3 图结构先轻后重

M1 必须有图，但只做最小骨架：

- `Span`
- `CenterSpanCluster`
- `Entity`
- `SUPPORTS`
- `RELATED_TO`
- `MENTIONS_ENTITY`
- `ADJACENT_CHAPTER`

不急着做复杂实体关系、不急着追求全自动聚类。中心 Span 可以人工指定，重点是验证“图召回能否帮助宏观问题组织证据”。

### 3.4 多路召回是主链路，不是锦上添花

M1 的在线查询必须默认执行：

- 图召回；
- 向量召回；
- BM25 召回；
- Union 去重；
- Reranker 精排；
- 坐标扩展与合并。

任何单路召回都只能作为降级模式，不能成为默认模式。

### 3.5 每个子阶段都要能独立验收

M1 拆成 10 个子阶段。每个子阶段都必须定义：

- 输入；
- 输出；
- 开发任务；
- 验收方法；
- 常见失败；
- 进入下一阶段的门槛。

从 M1.0 开始，每完成一个子阶段，都必须在对应阶段定义内追加 `### 验收记录` 小节。验收记录至少包含：

- 完成日期；
- 实际完成内容；
- 关键产物路径；
- 执行过的验证命令；
- 验收结论；
- 遗留问题或进入下一阶段的注意事项。

后续阶段不得只在对话中口头说明完成情况，必须把验收结果沉淀回本文档。

这样可以避免“所有模块都写了一点，但没有任何一段能确认可用”。

---

## 4. 推荐工程结构

M1 阶段建议先采用 Python CLI 项目结构：

```text
LoreWeaver/
  LoreWeaver.md
  LoreWeaver_Milestone_1.md
  pyproject.toml
  README.md
  .env.example
  configs/
    default.yaml
    models.yaml
    storage.yaml
  data/
    raw/
    normalized/
    runs/
    indexes/
    eval/
  loreweaver/
    __init__.py
    cli.py
    config.py
    logging.py
    models/
      document.py
      chapter.py
      span.py
      cluster.py
      evidence.py
    ingest/
      reader.py
      normalizer.py
      chapter_splitter.py
      window_splitter.py
    extraction/
      schemas.py
      prompts.py
      extractor.py
      locator.py
      retry.py
    storage/
      sqlite_store.py
      qdrant_store.py
      bm25_store.py
      neo4j_store.py
    graph/
      center_span.py
      edge_builder.py
    retrieval/
      query_router.py
      graph_retriever.py
      vector_retriever.py
      bm25_retriever.py
      union.py
      reranker.py
    evidence/
      interval.py
      assembler.py
      citation.py
    qa/
      answerer.py
      prompts.py
    eval/
      question_set.py
      runner.py
      metrics.py
  tests/
    fixtures/
    unit/
    integration/
```

说明：

- `data/raw/` 存原始文本；
- `data/normalized/` 存规范化后的文本；
- `data/runs/` 存每次建库与查询的运行记录；
- `data/indexes/` 存 BM25、本地缓存、临时索引；
- `data/eval/` 存问题集、人工评分、评测输出；
- `loreweaver/models/` 只放 Pydantic 数据模型；
- `loreweaver/storage/` 封装存储后端，避免业务逻辑直接依赖数据库 SDK。

---

## 5. 数据对象冻结版

M1 可以允许字段扩展，但核心对象和主键应尽早稳定。

### 5.1 Document

```text
document_id: str
title: str
author: str | None
source_path: str
normalized_path: str
total_chars: int
total_chapters: int
content_hash: str
created_at: datetime
```

验收要求：

- 同一份文本重复导入时，`content_hash` 应一致；
- 原文坐标必须基于 `normalized_path` 对应内容；
- 不允许后续流程混用 raw 文本坐标和 normalized 文本坐标。

### 5.2 Chapter

```text
chapter_id: str
document_id: str
chapter_index: int
chapter_title: str
start_idx: int
end_idx: int
char_count: int
```

验收要求：

- 所有章节区间不重叠；
- 章节按 `chapter_index` 连续；
- `text[start_idx:end_idx]` 必须能切出章节原文；
- 章节总覆盖率应接近 100%，允许少量前言、目录、尾声作为特殊章节。

### 5.3 CandidateWindow

```text
window_id: str
document_id: str
chapter_id: str
window_index: int
window_start: int
window_end: int
text: str
```

验收要求：

- 每个窗口必须落在单一章节内；
- 相邻窗口有配置化重叠；
- 窗口坐标必须可直接回切原文；
- 窗口不是最终知识单元，只是抽取输入。

### 5.4 Span

```text
span_id: str
document_id: str
chapter_id: str
window_id: str
span_index_in_window: int
window_start: int
window_end: int
span_type: str
micro_summary: str
entities: list[str]
topics: list[str]
salience_score: float
start_anchor_quote: str
end_anchor_quote: str
key_quote: str
overlap_reason: str
span_start_idx: int | None
span_end_idx: int | None
located_text: str
locator_confidence: float
locator_status: str
created_at: datetime
```

`locator_status` 建议枚举：

- `exact_match`
- `fuzzy_match`
- `multi_match_resolved`
- `failed`
- `manual_required`

验收要求：

- `locator_status=failed` 的 Span 不进入主检索索引；
- `span_start_idx` 和 `span_end_idx` 必须基于 normalized 全文；
- `start_anchor_quote` 和 `end_anchor_quote` 必须逐字来自窗口原文；
- `key_quote` 只作为核心证据短引，不承担完整定位；
- `located_text` 默认保存定位后的原文切片，便于 DBeaver 快速抽查；大规模建库时可通过配置关闭；
- 同一 `window_id` 可以生成多个 `span_index_in_window`，允许坐标重叠；
- `salience_score` 保留模型评分，但不能单独决定事实可信度。

### 5.5 CenterSpanCluster

```text
cluster_id: str
document_id: str
center_span_id: str
cluster_name: str
cluster_type: str
micro_summary: str
member_span_ids: list[str]
confidence: float
status: str
created_at: datetime
```

`cluster_type` 首版限定为：

- `character`
- `faction`
- `location`
- `power_system`
- `history`
- `mystery`

验收要求：

- M1 至少完成 2 个、建议完成 4 个中心 Span Cluster；
- 每个 Cluster 至少包含 5 个成员 Span；
- 每个成员 Span 必须能解释为什么归属该 Cluster；
- Cluster 进入问答主链路，而不是只存在于数据库里。

### 5.6 SpanEdge

```text
edge_id: str
document_id: str
from_id: str
to_id: str
from_type: str
to_type: str
edge_type: str
weight: float
source: str
created_at: datetime
```

`edge_type` 首版限定为：

- `SUPPORTS`
- `RELATED_TO`
- `MENTIONS_ENTITY`
- `ADJACENT_CHAPTER`

`source` 建议枚举：

- `rule`
- `manual`
- `llm`
- `imported`

验收要求：

- M1 中 `manual` 和 `rule` 边优先；
- LLM 生成边必须经过可解释理由或人工确认；
- 每类边至少有查询或调试方法能查看。

### 5.7 QueryEvidencePack

```text
query_id: str
document_id: str
user_question: str
query_type: str
retrieved_span_ids: list[str]
cluster_ids: list[str]
merged_intervals: list[dict]
evidence_blocks: list[dict]
retrieval_sources: dict
rerank_scores: dict
token_estimate: int
answer: str | None
created_at: datetime
```

验收要求：

- 每次问答都必须落盘 Evidence Pack；
- Evidence Pack 必须能复现最终送入模型的证据内容；
- 回答中的引用编号必须能映射回 `evidence_blocks`。

---

## 6. M1 子阶段拆分

## M1.0 项目骨架与样本锁定

### 目标

建立最小工程骨架，选定第一份测试文本，并冻结 M1 的配置入口、目录约定和开发命令。

### 输入

- `LoreWeaver.md`
- 一份 3-5 万字连续文本样本；
- OpenAI 或其他 LLM API 凭据；
- 本地 Python 环境；
- 可选：Qdrant、Neo4j、本地 Reranker 环境。

### 输出

- Python 项目骨架；
- 配置文件；
- `.env.example`；
- 第一份样本文本；
- 最小 README；
- M1 开发命令清单。

### 任务拆分

1. 初始化 Python 项目。
2. 确定依赖管理方式，建议使用 `uv` 或 `poetry`。
3. 建立 `configs/default.yaml`。
4. 建立 `data/` 目录结构。
5. 放入第一份样本文本到 `data/raw/`。
6. 创建最小 CLI：

```text
loreweaver ingest
loreweaver extract
loreweaver index
loreweaver graph
loreweaver ask
loreweaver eval
```

7. 建立基础日志格式，所有命令输出 `run_id`。

### 设计思路

M1 不应先写复杂服务端。CLI 更适合前期调试离线流水线。等 M1 闭环跑通后，M2/M3 再决定是否加 FastAPI 或 UI。

配置必须集中化，尤其是：

- 窗口大小；
- 窗口重叠；
- 抽取模型；
- embedding 模型；
- top-k 参数；
- token 预算；
- 数据库连接；
- 是否启用 Neo4j；
- 是否启用真实 Reranker。

### 独立验收

- 运行 `loreweaver --help` 能看到所有命令；
- 运行任一命令都会生成 `run_id`；
- 样本文本路径能被配置读取；
- 项目能通过基础 import 测试；
- README 说明如何准备 `.env` 和运行第一条命令。

### 常见失败与处理

- 依赖还未确定：先冻结最小依赖，不要提前引入重型框架；
- 数据目录混乱：所有生成物必须带 `document_id` 或 `run_id`；
- 配置分散：任何超参数不得硬编码在业务逻辑深处。

### 进入下一阶段门槛

项目骨架可运行，样本文本已锁定，配置系统可读取。

### 验收记录

完成日期：2026-04-21

验收结论：M1.0 已完成，可以进入 M1.1。

实际完成内容：

- 建立 Python 项目骨架；
- 建立 `loreweaver/` 包结构；
- 建立 M1 各阶段模块占位；
- 建立 CLI 入口；
- 建立集中配置文件；
- 建立数据目录结构；
- 建立环境变量模板；
- 建立最小 README；
- 锁定第一份测试文本；
- 确认基础命令可运行并输出 `run_id`。

关键产物：

- `pyproject.toml`
- `README.md`
- `.env.example`
- `.gitignore`
- `configs/default.yaml`
- `configs/models.yaml`
- `configs/storage.yaml`
- `loreweaver/cli.py`
- `loreweaver/config.py`
- `loreweaver/logging.py`
- `loreweaver/models/`
- `loreweaver/ingest/`
- `loreweaver/extraction/`
- `loreweaver/storage/`
- `loreweaver/graph/`
- `loreweaver/retrieval/`
- `loreweaver/evidence/`
- `loreweaver/qa/`
- `loreweaver/eval/`
- `tests/`
- `data/raw/DawnSword_Chapter_1_260.txt`

样本文本记录：

- 原始测试文件：`DawnSword_Chapter_1_260.txt`
- 已复制到：`data/raw/DawnSword_Chapter_1_260.txt`
- 文件大小：`2625682 bytes`
- 说明：该样本覆盖《黎明之剑》第 1 章到第 260 章，体量大于 M1 首轮建议的 3-5 万字。M1.1 实现真实入库时，可以先支持完整文件读取，同时允许通过配置限制首轮建库章节范围，避免早期抽取成本失控。

已执行验证命令：

```bash
python3 -m loreweaver.cli --help
python3 -m loreweaver.cli status
python3 -m loreweaver.cli ingest --source data/raw/DawnSword_Chapter_1_260.txt
python3 -m compileall loreweaver
```

验证结果：

- `--help` 能看到 `status / ingest / windows / extract / index / graph / retrieve / ask / eval` 命令；
- `status` 能读取 `configs/default.yaml`；
- `status` 能确认 `data/raw/DawnSword_Chapter_1_260.txt` 存在；
- `status` 输出 `bootstrap: ok`；
- 占位 `ingest` 命令能输出 `run_id`，并保留后续阶段将要使用的参数；
- `python3 -m compileall loreweaver` 通过，当前 Python 模块无语法错误；
- 当前环境未安装 `PyYAML`，因此已在 `loreweaver/config.py` 中提供 M1.0 轻量 YAML fallback，保证未安装依赖时基础命令仍可运行。

遗留问题与 M1.1 注意事项：

- 当前 `ingest / windows / extract / index / graph / retrieve / ask / eval` 仍是占位命令，M1.1 开始需要逐步替换为真实实现；
- 当前尚未建立 SQLite 表，M1.1 需要实现 `Document / Chapter` 落盘；
- 当前样本文本较大，M1.1 需要支持首轮限制章节范围；
- 当前仓库没有 git 初始化记录，本次验收基于工作区文件状态。

---

## M1.1 文本入库、规范化与章节切分

### 目标

把原始 `.txt` 转换成唯一可信的 normalized 文本，并建立章节坐标表。

### 输入

- `data/raw/<sample>.txt`
- 章节识别规则配置。

### 输出

- `data/normalized/<document_id>.txt`
- `Document` 记录；
- `Chapter` 记录；
- 入库报告。

### 任务拆分

1. 实现文本读取，统一编码为 UTF-8。
2. 规范化换行符。
3. 清理明显噪声：
   - 多余空白；
   - 连续空行；
   - 页面广告；
   - 重复标题；
   - 非正文尾巴。
4. 实现章节识别：
   - 支持 `第x章`；
   - 支持 `Chapter x`；
   - 支持手工 fallback：按固定长度伪章节切分。
5. 为每章记录全局 `[start_idx, end_idx]`。
6. 生成 `content_hash`。
7. 将 `Document`、`Chapter` 写入 SQLite。
8. 输出入库报告：
   - 总字符数；
   - 章节数；
   - 最短章节；
   - 最长章节；
   - 章节边界异常；
   - 文本清理前后字符差异。

### 设计思路

normalized 文本是 M1 的 Layer 0。之后所有坐标都必须基于 normalized 文本，不能再引用 raw 文本坐标。

章节切分不要追求一次支持所有小说格式。M1 的优先级是样本稳定，规则可配置，异常可发现。

### 独立验收

- 能导入一份 3-5 万字样本；
- `Document.total_chars` 与 normalized 文本长度一致；
- 每个 `Chapter` 区间可准确切回原文；
- 章节区间按顺序排列且不重叠；
- 入库报告能指出异常章节；
- 重复导入同一文本不会生成冲突记录。

### 建议测试

- 单章文本；
- 多章文本；
- 没有标准章节标题的文本；
- 含连续空行的文本；
- 含中英文章节标题的文本。

### 常见失败与处理

- 章节标题误识别正文：加入标题行长度、位置、前后换行约束；
- 清洗导致坐标不可追踪：M1 只保留 normalized 坐标，不做 raw-to-normalized 映射；
- 样本章节太短：允许合并过短章节或 fallback 为伪章节。

### 进入下一阶段门槛

样本文本能稳定生成 `Document` 与 `Chapter`，且章节坐标人工抽查无误。

### 验收记录

完成日期：2026-04-22

验收结论：M1.1 已完成，可以进入 M1.2。

实际完成内容：

- 实现真实 `ingest` CLI；
- 实现 raw `.txt` 读取与 UTF-8/GB18030 编码 fallback；
- 实现 normalized 文本生成，包括换行统一、行尾空白清理、连续空行压缩、分隔线与常见广告行清理；
- 实现章节识别，优先使用真实章标题，必要时才使用配置中的包装标题或固定长度伪章节 fallback；
- 支持中文数字章节、阿拉伯数字章节和 `Chapter x` 章节；
- 生成稳定 `content_hash` 与 `document_id`；
- 将 `Document`、`Chapter`、入库报告写入 SQLite；
- 输出 JSON 入库报告；
- 支持 `--max-chapters` 限制首轮落库章节数；
- 增加 M1.1 单元测试。

关键产物路径：

- `loreweaver/cli.py`
- `loreweaver/config.py`
- `loreweaver/ingest/reader.py`
- `loreweaver/ingest/normalizer.py`
- `loreweaver/ingest/chapter_splitter.py`
- `loreweaver/ingest/pipeline.py`
- `loreweaver/storage/sqlite_store.py`
- `tests/unit/test_m1_1_ingest.py`
- `data/normalized/doc_59331b17113e.txt`
- `data/runs/loreweaver_m1.sqlite3`
- `data/runs/ingest_20260422T014532Z_61052e59_ingest_report.json`

样本文本入库结果：

- 原始文本：`data/raw/DawnSword_Chapter_1_260.txt`
- normalized 文本：`data/normalized/doc_59331b17113e.txt`
- `document_id`：`doc_59331b17113e`
- `content_hash`：`59331b17113ef174b9f334c427a811b6957701de649ecc091f99d61c5e3c3f36`
- normalized 字符数：`882478`
- 章节数：`258`
- 章节识别策略：`real_chapter_patterns`
- 最短章节：`3169` 字符
- 最长章节：`4348` 字符
- 清理前后字符差异：移除 `10961` 字符
- 章节边界异常：`0`

已执行验证命令：

```bash
python3 -m unittest tests.unit.test_m1_1_ingest
python3 -m compileall loreweaver tests
python3 -m loreweaver.cli ingest --source data/raw/DawnSword_Chapter_1_260.txt --max-chapters 20
python3 -m loreweaver.cli ingest --source data/raw/DawnSword_Chapter_1_260.txt
python3 -m loreweaver.cli status
python3 -m loreweaver.cli ingest --source data/raw/DawnSword_Chapter_1_260.txt --max-chapters 258
sqlite3 data/runs/loreweaver_m1.sqlite3 "select count(*) from documents; select count(*) from chapters where document_id='doc_59331b17113e'; select count(*) from ingest_reports;"
```

验证结果：

- 单元测试通过；
- `compileall` 通过；
- `status` 显示阶段为 `M1.1`，样本文本存在，bootstrap 正常；
- 小范围 `--max-chapters 20` 入库成功，无章节边界告警；
- 完整样本入库成功，生成 258 个真实章节；
- 重复导入同一 normalized 文本未生成重复 `Document`，SQLite 中 `documents=1`；
- SQLite 中 `doc_59331b17113e` 对应 `chapters=258`；
- 人工抽查首尾章节坐标可回切，例如第 1 章为 `[68, 3366]`，第 258 章为 `[878974, 882478]`。

遗留问题与 M1.2 注意事项：

- 当前 normalized 坐标从真实章标题开始，样本开头的书名、作者、提取范围等元信息未作为章节覆盖，这是有意保留的前置信息；
- M1.2 切窗口时应只遍历 `Chapter` 表，不要把 raw 文本或前置信息重新纳入坐标体系；
- `data/normalized/` 与 `data/runs/` 仍按 `.gitignore` 作为本地生成物管理。

---

## M1.2 候选窗口切分

### 目标

将章节切成可供 LLM 抽取的候选窗口，同时保留全局坐标。

### 输入

- normalized 文本；
- Chapter 表；
- 窗口配置。

### 输出

- CandidateWindow 列表；
- 窗口切分报告。

### 推荐参数

中文样本初始参数：

```text
window_size_chars: 1200
overlap_ratio: 0.2
min_window_chars: 300
max_window_chars: 1600
```

### 任务拆分

1. 按章节遍历文本。
2. 对每章做滑动窗口切分。
3. 窗口不得跨章节。
4. 对最后一个过短窗口做合并或保留策略。
5. 每个窗口保存：
   - `window_id`
   - `chapter_id`
   - `window_start`
   - `window_end`
   - `window_index`
6. 输出统计：
   - 总窗口数；
   - 每章窗口数；
   - 平均窗口长度；
   - 重叠比例；
   - 过短窗口数量。

### 设计思路

候选窗口只是抽取输入。它的任务是保证 LLM 看到足够上下文，而不是成为最终证据粒度。

重叠窗口会造成重复 Span，这是可接受的。后续通过 anchor 定位、区间合并和去重处理。

### 独立验收

- 每个窗口都能从 normalized 文本中准确切出；
- 所有窗口都落在对应章节边界内；
- 重叠比例符合配置；
- 窗口数量在合理范围内；
- 对 3-5 万字样本，窗口数大约在 30-60 个左右，具体取决于章节长度。

### 建议测试

- 极短章节；
- 超长章节；
- 章节长度刚好等于窗口大小；
- 最后一段不足 `min_window_chars`。

### 常见失败与处理

- 窗口跨章：强制按章节独立切分；
- 末尾窗口过短：并入前一个窗口；
- 重叠导致抽取成本过高：优先调小 overlap，而不是调大窗口到失控。

### 进入下一阶段门槛

窗口坐标全部可回切，窗口报告无明显异常。

### 验收记录

完成日期：2026-04-22

验收结论：M1.2 已完成，可以进入 M1.3。

实际完成内容：

- 实现 `CandidateWindow` 数据模型；
- 实现按章节边界独立切分的滑动窗口算法；
- 支持配置化 `window_size_chars / overlap_ratio / min_window_chars / max_window_chars`；
- 对末尾过短窗口执行并入前一窗口策略；
- 实现窗口坐标校验，确认窗口不跨章、坐标可回切 normalized 文本；
- 将 `candidate_windows` 与 `window_reports` 写入 SQLite；
- 实现真实 `windows` CLI；
- 输出 JSON 窗口切分报告；
- 增加 M1.2 单元测试；
- 修复 M1.1 normalized 文本写入在当前 Python 版本下的 `newline` 兼容性问题。

关键产物路径：

- `loreweaver/models/window.py`
- `loreweaver/ingest/window_splitter.py`
- `loreweaver/storage/sqlite_store.py`
- `loreweaver/cli.py`
- `loreweaver/ingest/pipeline.py`
- `tests/unit/test_m1_2_windows.py`
- `data/runs/loreweaver_m1.sqlite3`
- `data/runs/windows_20260422T035226Z_0f1461e1_windows_report.json`

样本文本窗口切分结果：

- `document_id`：`doc_59331b17113e`
- normalized 文本：`data/normalized/doc_59331b17113e.txt`
- 章节数：`258`
- 总窗口数：`1030`
- 平均窗口长度：`1036.59`
- 最短窗口长度：`303`
- 最长窗口长度：`1255`
- 过短窗口数：`0`
- 窗口大小：`1200`
- 重叠比例：`0.2`
- 有效步长：`960`
- 窗口跨章异常：`0`
- 边界告警：`0`

已执行验证命令：

```bash
conda run -n loreweaver python -m unittest tests.unit.test_m1_1_ingest tests.unit.test_m1_2_windows
PYTHONPYCACHEPREFIX=/tmp/loreweaver_pycache conda run -n loreweaver python -m compileall loreweaver tests
conda run -n loreweaver python -m loreweaver.cli windows --document-id doc_59331b17113e
conda run -n loreweaver python -m loreweaver.cli status
sqlite3 data/runs/loreweaver_m1.sqlite3 "select count(*) from candidate_windows where document_id='doc_59331b17113e'; select count(*) from window_reports; select min(window_end-window_start), max(window_end-window_start), round(avg(window_end-window_start), 2) from candidate_windows where document_id='doc_59331b17113e'; select count(*) from candidate_windows w join chapters c on w.chapter_id=c.chapter_id where w.window_start < c.start_idx or w.window_end > c.end_idx;"
conda run -n loreweaver python -c "import sqlite3; text=open('data/normalized/doc_59331b17113e.txt', encoding='utf-8').read(); con=sqlite3.connect('data/runs/loreweaver_m1.sqlite3'); rows=con.execute(\"select window_id, window_start, window_end, text from candidate_windows where document_id='doc_59331b17113e' order by window_id limit 20\").fetchall(); print(len(rows)); print(sum(1 for _, s, e, t in rows if text[s:e] != t))"
```

验证结果：

- M1.1 与 M1.2 单元测试通过；
- `compileall` 通过；
- `windows` CLI 成功生成窗口报告并写入 SQLite；
- SQLite 中 `candidate_windows=1030`；
- SQLite 中 `window_reports=1`；
- SQLite 抽查窗口长度统计为 `303 / 1255 / 1036.59`；
- SQLite 查询跨章节窗口数量为 `0`；
- 抽查前 20 个窗口的存储文本与 normalized 文本坐标回切结果完全一致；
- `status` 显示阶段为 `M1.2`，bootstrap 正常。

遗留问题与 M1.3 注意事项：

- M1.3 抽取应优先从 `candidate_windows` 表读取窗口，必要时用 `window_start/window_end` 从 normalized 文本重新回切；
- `CandidateWindow.text` 是抽取输入缓存，不是最终可信证据源，最终可信坐标仍必须以 normalized 文本和 anchor 定位结果为准；
- 当前窗口数量基于完整 258 章样本为 `1030`，如果首轮 LLM 抽取成本过高，可先重新 ingest 较小 `--max-chapters` 样本或在 M1.3 增加窗口范围过滤。

---

## M1.3 结构化抽取与引用定位

### 目标

对每个候选窗口抽取多个微观 Span 元数据，并通过程序定位 `start_anchor_quote` / `end_anchor_quote` 的全局坐标区间。

### 输入

- CandidateWindow；
- 抽取 Prompt；
- Pydantic Schema；
- LLM API。

### 输出

- Span 记录，允许一个 CandidateWindow 对应多个 Span；
- 定位结果；
- 抽取失败队列；
- 定位失败队列；
- 抽取质量报告。

### 抽取 Schema

模型输出必须符合结构化格式。窗口只是 LLM 的阅读上下文，不是 Span 粒度本身：

```text
spans: list[
  span_type: str
  micro_summary: str
  entities: list[str]
  topics: list[str]
  salience_score: float
  start_anchor_quote: str
  end_anchor_quote: str
  key_quote: str
  overlap_reason: str
]
```

字段约束：

- `spans`：一个窗口输出多个微观 Span，建议 2-12 个；窗口很短时至少 1 个；
- `span_type`：首版限定为 `dialogue_exchange / relationship_signal / location_lore / faction_lore / power_rule / event / mystery_clue / object_lore / scene_action / other`；
- `micro_summary`：1-2 句，只概括当前小 Span，不写窗口或整章总结；需自包含主体、动作或设定点；
- `entities`：人物、地点、势力、物品、特殊术语；
- `topics`：更抽象的主题，如权力斗争、魔法规则、地理线索；
- `salience_score`：0-1；
- `start_anchor_quote`：Span 起点附近的短原文锚点，必须逐字来自窗口，建议 4-80 字符；若模型偶尔输出过长锚点，程序会优先保留靠近起点的一侧用于定位；
- `end_anchor_quote`：Span 终点附近的短原文锚点，必须逐字来自窗口，建议 4-80 字符；若模型偶尔输出过长锚点，程序会优先保留靠近终点的一侧用于定位；
- `key_quote`：可选，代表 Span 核心证据的一小段原文，不承担完整定位；
- `overlap_reason`：可选，说明该 Span 与其他 Span 重叠的必要性。

示例：

```text
原文：
主角B一进入房间，主角A便开口说道：“闲聊内容。XX森林你知道吗？那里......”

应拆为：
1. A 与 B 的对话互动 Span，覆盖 A 开口到该轮对话结束；
2. XX森林设定 Span，只覆盖关于 XX森林的描述；
3. 若闲聊体现 A/B 相处方式，可另拆人物关系 Span。
```

### 任务拆分

1. 编写抽取 Prompt。
2. 定义 Pydantic Schema。
3. 实现 LLM 调用封装。
4. 支持 temperature=0。
5. 对窗口逐个抽取，每个窗口产出多个 Span candidate。
6. 实现失败重试：
   - JSON 格式错误；
   - 字段缺失；
   - anchor 为空；
   - anchor 明显不在窗口中；
   - anchor 顺序错误或区间过大。
7. 实现 start/end anchor 精确匹配。
8. 实现 start/end anchor 局部模糊匹配。
9. 实现多命中消歧：
   - start anchor 必须早于 end anchor；
   - 优先跨度长度落在配置范围内；
   - 优先 anchor 匹配分最高；
   - 分数接近时优先更短、更聚焦的 Span；
   - 保留所有候选到调试日志。
10. 写入 Span 表。
11. 定位失败的 Span 标记为 `failed`，不进入主索引。

### 设计思路

这一步是 M1 的第一道高危点。绝不能相信模型坐标，也不能把“模型说这段在第几章”当作事实。

可靠链路应该是：

```text
窗口原文
  -> LLM 发现多个微观 Span
  -> LLM 输出每个 Span 的 start/end anchor
  -> 程序在窗口内查 anchor 并组合区间
  -> 程序映射到 normalized 全文坐标区间
  -> 落盘 locator_status 与 locator_confidence
```

### 定位策略

第一层：双锚点精确匹配。

```text
window_text.find(start_anchor_quote)
window_text.find(end_anchor_quote)
```

第二层：轻清洗后匹配。

- 去掉多余空白；
- 统一中文/英文引号；
- 统一省略号；
- 统一全角/半角空格。

第三层：双锚点模糊匹配。

- 使用相似度算法；
- 仅在当前窗口附近搜索；
- 分数低于阈值则失败。

第四层：区间组合与消歧。

- `start_anchor_quote` 命中必须早于 `end_anchor_quote` 命中；
- 区间长度必须落在 `target_span_chars_min` 和 `target_span_chars_max` 附近；
- 多组候选同时存在时，优先匹配分高、跨度短、靠近窗口中心的组合；
- 若 anchor 可定位但区间过大或过小，保留候选并标记 Span 定位失败。

### 独立验收

- 对样本窗口完成抽取；
- 单个窗口可产出多个 Span；
- 支持重叠或嵌套 Span；
- 结构化输出成功率 >= 95%；
- anchor 定位成功率 >= 90%；
- `failed` Span 不进入后续主索引；
- 随机抽查 20 条 Span，坐标切片与微话题对应；
- 每条失败记录能看到失败原因。

### 建议测试

- 单窗口多 Span；
- 重叠 Span；
- anchor 精确存在；
- anchor 中含换行；
- anchor 中含引号；
- anchor 在窗口中出现多次；
- 模型改写 anchor；
- anchor 可定位但 start/end 组合区间过大。

### 常见失败与处理

- 模型喜欢整章总结：Prompt 强调微观 Span 发现，不允许把窗口压缩成一个总结；
- 模型喜欢转述 anchor：Prompt 强调 anchor 必须逐字引用，并在失败时自动重试；
- anchor 太短导致多命中：设置最小长度；
- anchor 太长导致微小差异无法匹配：Prompt 设置最大长度，定位前对过长 start/end anchor 做边界侧裁剪；
- Span 区间过大：调小窗口或要求模型选择更近的 start/end anchor；
- LLM 成本过高：先用 10 个窗口做小样本调试，再全量跑。

### 进入下一阶段门槛

Span 抽取和定位稳定，定位成功的 Span 能够作为后续索引输入。

### 验收记录

完成日期：2026-04-22

实际完成内容：

- 定义 M1.3 多 Span 结构化抽取 Schema，优先使用 Pydantic，并提供最小依赖环境下的校验 fallback；
- 新增 OpenAI-compatible LLM 调用封装，支持 SiliconFlow `base_url` 与 `SILICONFLOW_API_KEY` 环境变量；
- 新增抽取 Prompt，强约束窗口内发现多个微观 Span，禁止默认整章总结；
- 新增双锚点定位器，覆盖 start/end anchor 的精确匹配、轻清洗匹配、局部模糊匹配、多候选消歧与过长锚点边界侧裁剪；
- 新增 Span、定位候选、抽取失败队列、抽取报告 SQLite 表，并支持一个窗口落多个 Span；
- 在 `candidate_windows` 增加 `uncovered_text` 调试字段，用 located Span 区间反算并合并窗口内未覆盖原文，便于 DBeaver 人工抽查是否确实不需要索引；
- 新增 `loreweaver extract` CLI，可用 `--limit` 做小样本 API 调试，也可用 `--mock` 做无 API 管线验收；流程反馈由统一 ProgressEvent 事件流驱动，可用全局 `--progress auto|rich|text|jsonl|none` 选择 CLI 渲染方式；
- 新增输入/输出 token 与人民币成本预估，当前测试模型价格配置为输入 `¥0.002 / K Tokens`、输出 `¥0.003 / K Tokens`；
- 更新 `.env.example`、`configs/models.yaml`、`configs/default.yaml` 与 README，密钥只通过环境变量读取，不写入业务代码。

关键产物路径：

- `loreweaver/extraction/extractor.py`
- `loreweaver/extraction/locator.py`
- `loreweaver/extraction/prompts.py`
- `loreweaver/extraction/schemas.py`
- `loreweaver/storage/sqlite_store.py`
- `loreweaver/cli.py`
- `tests/unit/test_m1_3_extraction.py`
- `data/runs/extract_20260422T064033Z_a332b5ab_extraction_report.json`

执行过的验证命令：

```bash
conda run -n loreweaver python -m pip install -e ".[m1,dev]"
conda run -n loreweaver python -m unittest discover -s tests/unit
conda run -n loreweaver python -m pytest tests/unit
conda run -n loreweaver python -m loreweaver.cli status
conda run -n loreweaver python -m loreweaver.cli extract --limit 2 --mock
conda run -n loreweaver python -m loreweaver.cli extract --limit 1 --mock
conda run -n loreweaver python -m loreweaver.cli ingest
conda run -n loreweaver python -m loreweaver.cli windows
conda run -n loreweaver env SILICONFLOW_API_KEY=<local-secret> python -m loreweaver.cli extract --limit 1
conda run -n loreweaver python -m compileall loreweaver tests/unit
conda run -n loreweaver python -m ruff check .
```

验收结论：M1.3 的确定性链路已完成并通过本地验收；mock 抽取在现有样本库上完成窗口内多 Span 抽取，Span 落库成功，anchor 定位成功率为 100%，失败队列与报告表可用。安装完整依赖后，真实 API 小样本流程 `ingest -> windows -> extract --limit 1` 曾跑通：258 章入库、1030 个候选窗口生成、1 个窗口真实 LLM 抽取成功、定位成功率 100%、预估成本 `¥0.002495`。双锚点多 Span 版本已通过 mock CLI 和真实 API 验证：mock 模式 1 个窗口生成 2 个 located Span；真实 API 1 个窗口生成 12 个 located Span，定位成功率 100%，预估成本 `¥0.01039`。后续真实 API `--limit 3` 质量检查得到 29 个 Span、25 个 located Span，暴露的主要问题是个别锚点过长、个别 Span 区间偏长；已补充锚点裁剪、Prompt 粒度约束与耗时进度。可以进入 M1.4。

遗留问题或进入下一阶段的注意事项：

- 当前 conda 环境已安装完整 M1/dev 依赖，`pytest`、`pydantic`、`openai` 可用；schema fallback 仅保留给最小 bootstrap 环境；
- 当前未把用户提供的 API Key 写入仓库，live LLM 调用需要先在本地环境设置 `SILICONFLOW_API_KEY`；
- 早期测试阶段不维护抽取表迁移；`extract` 会按当前 schema 重建抽取相关表，后续稳定后再引入正式迁移策略；
- `store_located_text` 默认开启，方便在 DBeaver 里抽查 Span 语义切分；大规模建库时可关闭以减少 SQLite 体积；
- `store_uncovered_text` 默认开启，方便在 DBeaver 里抽查窗口内未进入任何 located Span 的原文片段；大规模建库时可关闭；
- M1.4 只应读取 `locator_status = 'located'` 的 Span 进入向量/BM25/图索引；
- 全量抽取成本与耗时可能较高，建议先用 `loreweaver extract --limit 3` 做真实 API 小样本质量检查；若要测试整章窗口，先运行 `loreweaver windows --by-chapter`，再小范围抽取。

---

## M1.4 元数据存储、向量索引与 BM25 索引

### 目标

将定位成功的 Span 写入结构化元数据库、向量库和关键词索引，形成最小可检索底座。

### 输入

- 定位成功的 Span；
- embedding 模型；
- Qdrant 配置；
- BM25 存储配置。

### 输出

- SQLite 元数据库；
- Qdrant collection；
- BM25 索引文件；
- 索引构建报告。

### 任务拆分

1. 建立 SQLite 表：
   - documents；
   - chapters；
   - windows；
   - spans；
   - extraction_runs；
   - locator_failures；
   - query_runs。
2. 实现 Span 写入与查询。
3. 选择 embedding 输入：
   - M1 默认使用 `micro_summary + entities + topics`；
   - 可选加入 `key_quote`。
4. 批量生成 embedding。
5. 写入 Qdrant：
   - vector；
   - `span_id`；
   - `document_id`；
   - `chapter_id`；
   - `salience_score`；
   - `entities`；
   - `topics`；
   - `span_start_idx`；
   - `span_end_idx`。
6. 建立 BM25 文档：
   - `micro_summary`；
   - `entities`；
   - `topics`；
   - `key_quote`。
7. 保存 BM25 索引到 `data/indexes/`。
8. 实现最小检索命令：

```text
loreweaver search-vector "问题"
loreweaver search-bm25 "问题"
```

### 设计思路

M1 的索引层要避免“存了但不可查”。每种索引都必须有单独调试入口。

向量检索擅长语义相似，BM25 擅长专有名词和术语召回。二者都不能替代原文证据，只用于找到 Span 坐标。

### 独立验收

- SQLite 中 Span 数量与定位成功数量一致；
- Qdrant collection 中点数量与定位成功 Span 数量一致；
- BM25 索引能持久化并重新加载；
- 给定实体名，BM25 能召回含该实体的 Span；
- 给定抽象主题，向量检索能召回语义相关 Span；
- 检索结果能回查到原文 Span 区间。

### 建议测试

- 空实体；
- 重复 Span；
- embedding API 失败；
- Qdrant 连接失败；
- BM25 重建；
- 索引与 SQLite 数量不一致。

### 常见失败与处理

- embedding 输入太短：拼接 micro_summary、entities、topics，必要时加入 key_quote 或 located_text；
- BM25 中文分词效果差：先用简单分词跑通，再替换更好的中文分词；
- Qdrant 与 SQLite 不一致：每次索引构建记录 `run_id`，并做数量校验。

### 进入下一阶段门槛

向量与 BM25 都能独立检索，检索结果可以回查原文坐标。

### 验收记录

完成日期：2026-04-23

实际完成内容：

- 新增 SQLite 索引层元数据：`embedding_cache`、`index_reports`、`query_runs`，支持 embedding 缓存与索引报告落盘；
- 新增 OpenAI-compatible embedding 调用封装，当前配置默认使用 SiliconFlow `Qwen/Qwen3-Embedding-0.6B`，并保留 `--mock-embeddings` 便于本地无 API 验收；
- 实现 embedding 输入构建与缓存键逻辑，M1 默认输入为 `micro_summary + entities + topics`，可通过配置控制是否加入 `key_quote` 或 `located_text`；
- 实现 Qdrant 向量存储适配层：优先读取 `QDRANT_URL` 远程配置，未配置时默认落到本地 `data/indexes/qdrant`，并按文档维度重建 collection；
- 实现 BM25 本地索引适配层：基于 `rank-bm25` 落盘 `data/indexes/<document_id>_bm25.json`，包含中文友好的字/二元/三元 token 化，支持持久化与重新加载；
- 新增 `loreweaver index` CLI，输出 embedding 缓存命中、Qdrant collection 计数、BM25 文档数与索引报告路径；
- 新增 `loreweaver search-vector` 与 `loreweaver search-bm25` CLI，作为 M1.4 的独立调试入口，可直接回查 `span_id`、章节与原文坐标；
- 新增 M1.4 单元测试，覆盖本地 Qdrant + BM25 + SQLite 缓存链路，以及中文 BM25 tokenizer 的实体命中能力；
- 更新 README、配置文件与 `.env.example`，明确默认 embedding 模型、Qdrant 本地路径和 M1.4 命令面。

关键产物路径：

- `loreweaver/indexing/embeddings.py`
- `loreweaver/indexing/pipeline.py`
- `loreweaver/storage/qdrant_store.py`
- `loreweaver/storage/bm25_store.py`
- `loreweaver/storage/sqlite_store.py`
- `loreweaver/cli.py`
- `tests/unit/test_m1_4_indexing.py`
- `data/runs/index_20260423T080908Z_07dbcd10_index_report.json`

执行过的验证命令：

```bash
conda run -n loreweaver python -m pytest tests/unit/test_m1_4_indexing.py
conda run -n loreweaver python -m pytest tests/unit
conda run -n loreweaver python -m ruff check .
conda run -n loreweaver python -m compileall loreweaver tests/unit
conda run -n loreweaver python -m loreweaver.cli index --limit 10 --mock-embeddings
conda run -n loreweaver python -m loreweaver.cli search-bm25 高文 --top-k 3
conda run -n loreweaver python -m loreweaver.cli search-vector 旧时代秘密 --top-k 3 --mock-embeddings
```

验收结论：M1.4 的本地可运行闭环已完成并通过验收。当前仓库上的 mock embedding 验证链路已确认：定位成功的 Span 可以完成 embedding 缓存、Qdrant 本地 collection 写入、BM25 持久化落盘，并能通过 `search-vector` / `search-bm25` 两个独立入口回查到 `span_id`、章节与原文区间。全量 unit 当前为 `16/16` 通过；在现有样本库上，CLI 烟测已成功写入本地 Qdrant collection `loreweaver_doc_59331b17113e_spans` 10 个点，并生成 `data/indexes/doc_59331b17113e_bm25.json`，`search-bm25 高文` 可稳定命中相关 Span。

遗留问题或进入下一阶段的注意事项：

- 当前 `search-vector --mock-embeddings` 只验证向量链路连通性，不代表真实语义召回质量；真实 embedding 质量应在后续使用 SiliconFlow API 做小样本人工抽查；
- 目前 Qdrant collection 采用按文档重建策略，适合单书 M1；后续若要支持增量建库，需要补 collection 级别的 run/version 管理；
- BM25 目前使用轻量中文 token 化方案，足够支撑 M1 实体验证；如后续术语召回仍弱，再评估更强中文分词器；
- `query_runs` 表已建，但 M1.4 尚未写入在线查询记录，留待 M1.6 的混合召回链路接入；
- M1.5 若开始写图骨架，应继续坚持只消费 `locator_status = 'located'` 的 Span，避免图索引引入不可信区间。

---

## M1.5 最小中心 Span 图骨架

### 目标

建立 M1 的轻量图结构，让系统具备“从宏观主题下钻到底层 Span”的能力。

### 输入

- 定位成功的 Span；
- 高 salience Span；
- 人工指定的聚类方向；
- Neo4j 配置。

### 输出

- 2-4 个 CenterSpanCluster；
- 对应成员 Span；
- 显式边；
- 图构建报告。

### 推荐 M1 聚类方向

从样本文本中选择最明显的 2-4 类：

- 主角相关关系；
- 核心势力；
- 关键地点；
- 力量体系；
- 历史事件；
- 悬疑/异常现象。

不要为了凑类型硬建 Cluster。宁愿少而准。

### 任务拆分

1. 实现高 salience Span 列表查看命令：

```text
loreweaver spans --top-salience 30
```

2. 人工挑选中心 Span。
3. 为每个中心 Span 填写：
   - `cluster_name`
   - `cluster_type`
   - `micro_summary`
4. 召回候选成员 Span：
   - 向量相似；
   - 共享实体；
   - BM25 命中；
   - 同章或邻近章节。
5. 生成候选成员列表。
6. 人工确认或规则过滤成员。
7. 写入 CenterSpanCluster。
8. 写入边：
   - Center -> Span: `SUPPORTS`
   - Span -> Entity: `MENTIONS_ENTITY`
   - Span -> Span: `RELATED_TO`
   - Chapter -> Chapter: `ADJACENT_CHAPTER`
9. 同步写入 Neo4j。
10. 保留 SQLite mirror 或导出文件用于调试。

### 设计思路

M1 的图不是“知识图谱完成品”，而是宏观检索的导航骨架。

中心 Span 不是真理，只是索引锚点。它的价值在于：

- 让宏观问题先命中主题框架；
- 顺着显式边找到一批底层证据；
- 与向量/BM25 形成互补；
- 为 M2 的报告结构提供雏形。

### 独立验收

- 至少有 2 个 CenterSpanCluster；
- 每个 Cluster 至少有 5 个成员 Span；
- 每个 Cluster 能解释成员归属；
- Neo4j 中可以查到 Cluster、Span 和边；
- CLI 能按 cluster 查看成员和引用；
- 后续问答能实际使用 Cluster。

### 建议测试

- 中心 Span 被删除或定位失败；
- 成员 Span 重复归属多个 Cluster；
- Cluster 无成员；
- Neo4j 不可用；
- 人工配置格式错误。

### 常见失败与处理

- 中心 Span 代表性不足：允许人工替换，并记录替换历史；
- 成员过杂：先用共享实体和类型过滤，再看向量相似；
- Neo4j 增加复杂度：业务逻辑通过 `GraphStore` 接口访问，方便临时降级。

### 进入下一阶段门槛

中心 Span 图可以被查询，并且能返回可追溯的底层 Span。

### 验收记录

完成日期：2026-04-24

实际完成内容：

- 新增 `CenterSpanCluster` 与 `SpanEdge` 数据模型；
- 新增 SQLite 图骨架 mirror：`center_span_clusters`、`span_edges`、`graph_reports`；
- 实现高 salience Span 查看入口 `loreweaver spans --top-salience 30`，用于人工挑选中心 Span；
- 实现 M1.5 中心 Span 图构建管线：按 Span 类型、实体、主题、章节邻近与词面重合挑选中心 Span 和成员 Span；
- 实现 `SUPPORTS / RELATED_TO / MENTIONS_ENTITY / ADJACENT_CHAPTER` 四类显式边；
- 实现 `loreweaver graph` 构建命令与 `loreweaver graph --list` 调试查看命令；
- 新增 Neo4j 可选同步适配层，默认不启用，避免 M1 本地验收被外部服务阻塞；
- 新增 M1.5 单元测试，覆盖 Cluster 构建、边构建、SQLite 落库、图报告与查询读取；
- 更新配置与 README，将当前阶段推进到 M1.5。

关键产物路径：

- `loreweaver/models/cluster.py`
- `loreweaver/graph/center_span.py`
- `loreweaver/graph/edge_builder.py`
- `loreweaver/storage/sqlite_store.py`
- `loreweaver/storage/neo4j_store.py`
- `loreweaver/cli.py`
- `tests/unit/test_m1_5_graph.py`
- `configs/default.yaml`
- `README.md`
- `data/runs/graph_20260424T014038Z_afd76ef5_graph_report.json`

执行过的验证命令：

```bash
conda run -n loreweaver python -m pytest tests/unit/test_m1_5_graph.py
conda run -n loreweaver python -m ruff check loreweaver tests/unit/test_m1_5_graph.py
conda run -n loreweaver python -m compileall loreweaver
conda run -n loreweaver python -m loreweaver.cli spans --top-salience 8
conda run -n loreweaver python -m loreweaver.cli graph --no-neo4j
conda run -n loreweaver python -m loreweaver.cli graph --list
```

验收结论：

M1.5 本地轻量图骨架已完成并通过验收。当前样本库中已有 `35` 个 located Span；`graph --no-neo4j` 在其上生成 `4` 个 CenterSpanCluster、`351` 条边，其中 `SUPPORTS=32`、`RELATED_TO=28`、`MENTIONS_ENTITY=34`、`ADJACENT_CHAPTER=257`。每个 Cluster 当前包含 `8` 个成员 Span，并可通过 `graph --list` 查看成员、章节、坐标与主题。

遗留问题或进入下一阶段的注意事项：

- 当前 Cluster 成员确认采用规则启发式，已经满足 M1.5 可调试骨架要求，但不是最终自动聚类质量；
- 当前 Neo4j 同步为可选路径，未在本机凭据下做 live 验证；M1.6 默认应先消费 SQLite mirror，待 Neo4j 凭据稳定后再接入图召回；
- 现有样本抽取只覆盖早期少量窗口，Cluster 主题偏向高文早期穿越、异常、身份与亡灵复生规则；后续若要更丰富的势力/地点/力量体系 Cluster，需要继续抽取更多窗口；
- M1.6 应优先实现图召回读取 `center_span_clusters` 与 `span_edges`，再与向量、BM25 召回做 Union。

#### M1.5 embedding-aware 改进记录

完成日期：2026-04-24

改进内容：

- M1.5 图构建接入 M1.4 Qdrant 向量索引，优先读取已建好的 Span embedding；
- 成员评分从规则加分改为分项加权：`vector / entity / topic / bm25 / chapter / salience`；
- 每个成员在 graph report 中保存 `component_scores` 与 reasons，便于复盘向量和规则各自贡献；
- 中心 Span 选择改为“类型代表性优先 + embedding medoid + salience”；
- 实体命名增加类型偏好，降低高频主角实体对 faction/location 命名的干扰；
- 新增 `loreweaver graph --no-embeddings`，可做规则 fallback 与 embedding-aware 的 A/B 对照。

真实样本验证：

- 当前 SQLite 中 located Span：`102`；
- 当前 Qdrant 向量覆盖：`102/102`；
- 规则 fallback 报告：`data/runs/graph_20260424T031633Z_ca30a4df_graph_report.json`；
- embedding-aware 报告：`data/runs/graph_20260424T031807Z_7448c7ff_graph_report.json`；
- embedding-aware 当前生成：`4` 个 Cluster、`384` 条边，其中 `SUPPORTS=32`、`RELATED_TO=28`、`MENTIONS_ENTITY=67`、`ADJACENT_CHAPTER=257`。

A/B 初步结论：

- embedding-aware 版本能利用跨章节语义相似度，例如将“塞西尔家族衰落”“法师道路旧规”“传承知识缺失”等 Span 拉入同一 faction 骨架；
- 规则 fallback 在强实体/强地点线索上仍然可靠，例如 `塞西尔领` 与 `刚铎帝国` 相关簇较清晰；
- embedding-aware 初版曾把 location 中心拉向“逃生方案”，说明纯 medoid 不足以代表类型；已加入类型代表性优先策略；
- 当前改进版的四个簇为：`历史事件：高文`、`关键地点：先祖陵寝`、`角色关系：瑞贝卡·塞西尔`、`核心势力：塞西尔家族`；
- 仍需在 M1.6 通过真实查询评估图召回贡献，单看簇标题和成员还不能证明最终问答质量。

#### M1.3 增量抽取命令改进记录

完成日期：2026-04-24

改进内容：

- `extract` 不再在每次运行时重建并清空所有抽取表；
- 新增 `extract --list-windows`，可查看候选窗口是否已经抽取、Span 数、located 数和失败数；
- `extract --list-windows --only extracted|pending` 支持只看已抽取或未抽取窗口；
- `--window-id` 改为可重复、可逗号分隔；
- 新增 `--window-range`，按 1-based 全局窗口序号指定范围，例如 `--window-range 21-40`；
- 指定窗口重跑时会覆盖这些窗口的旧 Span、locator candidates、failures 和 uncovered_text，但不会影响未指定窗口；
- 重建 windows 或重新 ingest 文档时仍会清理对应旧抽取结果，避免坐标和窗口不一致。

示例命令：

```bash
conda run -n loreweaver python -m loreweaver.cli extract --list-windows --only pending --limit 20
conda run -n loreweaver python -m loreweaver.cli extract --window-range 21-40
conda run -n loreweaver python -m loreweaver.cli extract --window-id doc_59331b17113e_ch0021_win0001
```

---

## M1.6 混合召回、Union 与 Reranker

### 目标

实现在线查询的核心证据召回链路：图召回、向量召回、BM25 召回、Union 合并、Reranker 精排。

### 输入

- 用户问题；
- CenterSpanCluster；
- Qdrant 索引；
- BM25 索引；
- Reranker 配置。

### 输出

- 候选 Span 池；
- 去重后的候选列表；
- 精排后的 Top-K Span；
- 召回调试报告。

### 推荐参数

M1 初始参数：

```text
graph_cluster_top_k: 4
graph_span_per_cluster: 12
vector_top_k: 30
bm25_top_k: 30
union_max_candidates: 80
rerank_top_k: 15
min_rerank_score: optional
```

### 任务拆分

1. 实现 Query Router：
   - `character_relation`
   - `faction_history`
   - `location`
   - `power_system`
   - `timeline`
   - `unknown`
2. 实现图召回：
   - 问题匹配 Cluster；
   - 沿 `SUPPORTS` 下钻成员 Span；
   - 补充中心 Span 本身。
3. 实现向量召回：
   - 问题 embedding；
   - Qdrant Top-K；
   - payload 返回。
4. 实现 BM25 召回：
   - 问题分词；
   - Top-K；
   - 返回 span_id 和 score。
5. 实现 Union 合并：
   - 按 `span_id` 去重；
   - 保留来源列表；
   - 保留各路分数；
   - 计算初始融合分。
6. 实现 Reranker 接口：
   - 输入 `question + span_text_for_rerank`；
   - 输出统一 score；
   - 支持真实模型；
   - 支持 mock reranker 便于开发。
7. 输出调试报告：
   - 每路召回数量；
   - Union 后数量；
   - Rerank Top-K；
   - 每个 Top-K 的来源；
   - 是否命中 Cluster。

### 设计思路

混合召回必须是主路径。

图召回负责宏观结构，向量召回负责语义近邻，BM25 负责专有名词和原文术语。Reranker 用来统一判断“这个 Span 是否真的回答这个问题”。

M1 需要特别关注“语义相似但对象错误”的假阳性。例如问题问 A 国，向量召回可能返回 B 国的相似段落。BM25 与实体覆盖规则可以提前压制这类错误。

### 可选增强：实体覆盖门控

参考 `VAC Mem.md` 中 MCA 思路，M1 可实现一个轻量实体覆盖分：

```text
coverage = 问题关键词与候选 Span 实体/关键词的交集比例
```

用法：

- 不作为硬过滤的唯一依据；
- 作为 Union 初始融合分的一部分；
- 用于压低对象明显不匹配的候选。

### 独立验收

- 任意问题都能返回召回报告；
- 图、向量、BM25 三路至少两路能正常参与；
- Union 后能保留每个 Span 的来源；
- Reranker 能输出稳定 Top-K；
- 对 10 个手工问题，Top-K 中至少有部分人工认为相关的证据；
- 图召回不足时系统能降级到向量 + BM25。

### 建议测试

- 问题包含明确实体；
- 问题是抽象主题；
- 问题命中未建图主题；
- BM25 无结果；
- Qdrant 无结果；
- Neo4j 无结果；
- Reranker 不可用。

### 常见失败与处理

- 候选池太小：增大各路 Top-K；
- 候选池太大导致精排慢：设置 `union_max_candidates`；
- BM25 对中文效果差：优化分词；
- 图召回过窄：Cluster 成员补充向量相似 Span；
- Reranker 吃不下长文本：使用 micro_summary + key_quote + entities 作为精排文本。

### 进入下一阶段门槛

混合召回链路可运行，Top-K 结果能为证据包组装提供稳定输入。

### 技术选型记录

M1.6 Reranker 采用“可插拔接口 + 远程优先 + 本地/禁用降级”的策略：

- 主选型：`Qwen/Qwen3-Reranker-0.6B` via SiliconFlow `/v1/rerank`；
- fallback 模型：`BAAI/bge-reranker-v2-m3`；
- 工程接口：`MockReranker`、`ServiceReranker`、`NoopReranker` 共用 `question + span_text_for_rerank` 输入；
- M1.6 初期不启用硬 `min_rerank_score`，先保存原始分数、模型、provider、rerank 文本 hash 与 Top-K 排名，等人工评估后再定阈值；
- 默认开发验收优先使用 `--mock-reranker` 或 `--no-reranker`，避免本地环境和远程模型可用性阻塞主链路。

### 验收记录

完成日期：2026-04-24

实际完成内容：

- 新增 `loreweaver retrieve` CLI，跑通图召回、向量召回、BM25 召回、Union 融合与 Reranker 精排；
- 实现 `Query Router`，支持 `character_relation / faction_history / location / power_system / timeline / unknown`；
- 实现 SQLite graph mirror 召回，读取 `center_span_clusters` 与 `SUPPORTS` 边，并补充中心 Span；
- 实现 BM25 与 Qdrant 向量召回适配，向量召回失败时会记录错误并允许其余召回路径继续；
- 实现 Union 去重与融合分，保留 `sources`、各路原始分、归一化分、Cluster 命中信息；
- 实现 Reranker 接口与 `MockReranker`、`ServiceReranker`、`NoopReranker`；
- 每次查询输出 retrieval report，并写入 SQLite `query_runs`。

关键产物路径：

- `loreweaver/retrieval/pipeline.py`
- `loreweaver/retrieval/graph_retriever.py`
- `loreweaver/retrieval/vector_retriever.py`
- `loreweaver/retrieval/bm25_retriever.py`
- `loreweaver/retrieval/union.py`
- `loreweaver/retrieval/reranker.py`
- `tests/unit/test_m1_6_retrieval.py`
- `data/runs/retrieve_20260424T084827Z_73ec4d0a_retrieval_report.json`

执行过的验证命令：

```bash
conda run -n loreweaver python -m compileall loreweaver
conda run -n loreweaver python -m pytest tests/unit/test_m1_6_retrieval.py
conda run -n loreweaver python -m pytest
conda run -n loreweaver python -m ruff check loreweaver tests
conda run -n loreweaver python -m loreweaver.cli retrieve 塞西尔家族为什么会衰落 --mock-embeddings --mock-reranker
```

验收结论：M1.6 的本地可运行闭环已完成。当前样本文档上，CLI 烟测返回 `graph=36`、`vector=30`、`bm25=30`、`union=70`，并生成 mock rerank Top-15；全量 unit 为 `18/18` 通过。

遗留问题或进入下一阶段的注意事项：

- 真实 SiliconFlow Reranker 已完成 6 题小样本 live API 烟测，代表性报告包括 `data/runs/retrieve_20260424T085431Z_5064fbc4_retrieval_report.json`、`data/runs/retrieve_20260424T085503Z_c024fdf6_retrieval_report.json`、`data/runs/retrieve_20260424T085539Z_eb78d7d5_retrieval_report.json`；
- live 烟测显示明确实体/地点/事实类问题 Top-K 质量较好，抽象力量体系与伏笔异常类问题仍需强化 query router、候选池构造与 rerank 输入；
- 当前人工验收尚未覆盖 10 个手工问题，M1.7 前后应补齐 Top-K 证据相关性人工评分；
- mock embedding 只验证链路连通性，不代表真实语义召回质量；
- `min_rerank_score` 暂不启用，待人工评估后再按模型分布设置。

---

## M1.7 证据区间合并与 Evidence Pack 组装

### 目标

把精排后的 Span 转换为可送入长上下文模型的原文证据包。

### 输入

- Rerank Top-K Span；
- Chapter 表；
- normalized 文本；
- token 预算配置。

### 输出

- 合并后的原文区间；
- evidence_blocks；
- QueryEvidencePack；
- 引用编号。

### 推荐参数

M1 初始参数：

```text
pre_context_chars: 300
post_context_chars: 500
max_evidence_chars: 40000
merge_gap_chars: 500
max_blocks: 12
```

### 任务拆分

1. 读取每个 Span 的 `span_start_idx`、`span_end_idx`。
2. 根据章节边界做上下文扩展：
   - 向前扩展；
   - 向后扩展；
   - 不跨章节，除非配置允许。
3. 区间裁剪到章节边界内。
4. 区间排序。
5. 合并重叠区间。
6. 合并间距小于 `merge_gap_chars` 的近邻区间。
7. 控制总字符数：
   - 优先保留 Rerank 分高的区间；
   - 优先保留图 + 向量 + BM25 多源命中的区间；
   - 保留章节多样性。
8. 生成 evidence_blocks：
   - `citation_id`
   - `chapter_title`
   - `start_idx`
   - `end_idx`
   - `text`
   - `source_span_ids`
9. 生成 QueryEvidencePack。
10. 写入 SQLite。

### 设计思路

Span 是导航点，不一定包含足够上下文。因此要适度扩展。但扩展必须克制，否则宏观问题会被大量噪声吞没。

证据区间合并是确定性程序逻辑，不交给模型处理。

### 独立验收

- evidence_blocks 能准确从 normalized 文本切出；
- 合并后区间不重叠；
- 引用编号唯一；
- 每个 evidence_block 能追溯到 source_span_ids；
- 总证据长度不超过配置预算；
- QueryEvidencePack 能完整复现输入给回答模型的内容。

### 建议测试

- 多个 Span 区间重叠；
- 多个 Span 相距很近；
- Span 在章节开头；
- Span 在章节末尾；
- 证据总长度超预算；
- Top-K 中有无效坐标。

### 常见失败与处理

- 合并后证据过长：降低 Top-K 或上下文扩展；
- 扩展跨章导致噪声大：默认不跨章；
- 引用编号混乱：由 Evidence Pack 统一生成，不让模型自行编号；
- 证据块丢失来源：每个 block 保存 source_span_ids。

### 进入下一阶段门槛

Evidence Pack 可稳定生成，并且证据文本、引用编号、原文坐标全部可追溯。

### 验收记录

- 完成日期：2026-04-27
- 实际完成内容：
  - 实现 M1.7 Evidence Pack 组装链路：接收 M1.6 Rerank Top-K，按章节边界扩展 Span 坐标，合并重叠/近邻区间，按证据预算和块数上限裁剪。
  - 生成稳定引用编号 `[E001]` 形式的 `evidence_blocks`，每个 block 保留 `chapter_title`、`start_idx`、`end_idx`、`text`、`source_span_ids`、`retrieval_sources`、`rerank_score`。
  - 新增 `QueryEvidencePack` 落盘表 `evidence_packs`，并提供 `loreweaver evidence` CLI，从混合召回直接产出 Evidence Pack 报告。
  - 将默认阶段号推进为 `M1.7`。
- 关键产物路径：
  - `loreweaver/evidence/interval.py`
  - `loreweaver/evidence/assembler.py`
  - `loreweaver/evidence/citation.py`
  - `loreweaver/models/evidence.py`
  - `loreweaver/storage/sqlite_store.py`
  - `loreweaver/cli.py`
  - `tests/unit/test_m1_7_evidence.py`
  - `configs/default.yaml`
- 执行过的验证命令：
  - `conda run -n loreweaver python -m pytest tests/unit/test_m1_7_evidence.py`
  - `python -m compileall loreweaver`
  - `conda run -n loreweaver python -m pytest tests/unit`
  - `env SILICONFLOW_API_KEY=<redacted> conda run -n loreweaver python -m loreweaver.cli evidence "塞西尔家族为什么会衰落，高文和这个家族的关系如何变化？" --no-reranker`
- 验收结论：
  - M1.7 独立单测通过，完整单元测试通过：`20 passed`。
  - 真实 SiliconFlow embedding API 测试通过：Graph 36、Vector 30、BM25 30，Union 59 个候选，生成 12 个 evidence_blocks，0 个 warning，并写入 SQLite `evidence_packs`。
  - Evidence Pack 可复现送入后续回答模型的证据块，引用编号、原文坐标和来源 Span 可追溯。
- 遗留问题或进入下一阶段的注意事项：
  - 当前预算裁剪为确定性启发式：优先高 Rerank 分、多源命中和章节多样性；M1.8 人工问答验收时需要观察是否过度裁剪关键上下文。
  - `token_estimate` 仍为字符级保守估算，后续接入真实回答模型时可替换为模型 tokenizer。
  - M1.8 需要在回答生成阶段强制只使用 `evidence_blocks`，并验证引用编号能回映射到本阶段生成的 Evidence Pack。

---

## M1.8 在线证据问答与引用输出

### 目标

基于 Evidence Pack 生成谨慎、可引用、可复盘的回答。

### 输入

- 用户问题；
- QueryEvidencePack；
- 回答 Prompt；
- 长上下文模型。

### 输出

- 带引用的回答；
- 不确定性标记；
- 查询运行记录。

### 回答格式建议

M1 的回答不用追求最终产品文风，建议固定结构：

```text
结论：
...

证据：
[E1] ...
[E2] ...

分析：
...

不确定性：
...
```

也可以用更自然的 Markdown，但必须包含引用编号。

### 任务拆分

1. 编写回答 Prompt。
2. Prompt 中明确：
   - 只能基于 evidence_blocks 回答；
   - 关键结论必须引用 `[E#]`；
   - 证据不足必须说明；
   - 推测必须标记为“推测”；
   - 不得编造章节或引用。
3. 组装模型输入：
   - 用户问题；
   - Cluster 摘要；
   - evidence_blocks；
   - 输出约束。
4. 调用回答模型。
5. 校验回答引用：
   - 引用编号是否存在；
   - 是否至少包含一个引用；
   - 是否出现不存在的 `[E99]`。
6. 对引用失败进行一次修复或重试。
7. 保存最终答案到 QueryEvidencePack。
8. CLI 输出回答与证据摘要。

### 设计思路

M1 的回答模型不是自由创作模型，而是“证据解释器”。它的任务是阅读已经组装好的证据，然后给出有边界的分析。

引用校验必须由程序完成。模型输出后如果引用编号不存在，不能直接放行。

### 独立验收

- `loreweaver ask "问题"` 能输出回答；
- 回答至少包含一个有效引用；
- 引用编号能映射回 evidence_blocks；
- 证据不足的问题不会被强行编答案；
- 回答记录能在 SQLite 中回查；
- 同一个问题重复运行，召回与回答基本稳定。

### 建议测试

- 有明确证据的问题；
- 证据分散的问题；
- 证据不足的问题；
- 问题包含错误前提；
- 问题要求宏观总结；
- 问题要求时间演变。

### 常见失败与处理

- 回答没有引用：强制引用校验，不合格重试；
- 模型过度推断：Prompt 中加入结论等级；
- evidence_blocks 太多：控制证据长度；
- 回答引用无关证据：人工评估时标记，回看 Reranker 与证据包。

### 进入下一阶段门槛

在线问答主链路可用，并能输出可追溯引用。

### 验收记录

- 完成日期：2026-04-27
- 实际完成内容：
  - 实现 M1.8 在线证据问答主链路：`ask` 命令串联 M1.6 混合召回、M1.7 Evidence Pack 组装、M1.8 回答生成与引用校验。
  - 编写回答 Prompt 与引用修复 Prompt，要求回答只能基于 `evidence_blocks`，关键结论必须使用 `[E###]` 引用，证据不足时明确标记。
  - 新增 OpenAI-compatible 回答客户端与 deterministic mock answerer，支持无 API 的端到端烟测。
  - 实现程序化引用校验：检查回答是否引用 Evidence Pack 中存在的编号，发现缺失或不存在引用时执行一次修复。
  - 将最终回答、引用校验结果、模型信息、来源报告路径写入 answer report，并回写 SQLite `evidence_packs.answer` 与 `query_runs`。
  - 将默认阶段号推进为 `M1.8`。
- 关键产物路径：
  - `loreweaver/qa/answerer.py`
  - `loreweaver/qa/prompts.py`
  - `loreweaver/cli.py`
  - `loreweaver/storage/sqlite_store.py`
  - `tests/unit/test_m1_8_qa.py`
  - `data/runs/retrieve_20260427T031746Z_382f200c_answer_report.json`
- 执行过的验证命令：
  - `conda run -n loreweaver python -m pytest tests/unit/test_m1_8_qa.py tests/unit/test_m1_7_evidence.py`
  - `conda run -n loreweaver python -m pytest tests/unit`
  - `conda run -n loreweaver python -m loreweaver.cli ask "塞西尔家族和高文有什么关系？" --mock-embeddings --no-reranker --mock-answer`
- 验收结论：
  - M1.8 独立单测通过，M1.7 回归通过，完整单元测试通过：`23 passed`。
  - `ask` CLI 可输出带有效 `[E001]` 引用的回答，引用编号可回映射到本次 Evidence Pack。
  - 回答记录已可在 SQLite 与 answer report 中回查。
- 遗留问题或进入下一阶段的注意事项：
  - 当前烟测使用 `--mock-answer`，尚未做真实长上下文回答模型的人工质量验收。
  - Mock 回答只验证链路与引用约束，不代表最终回答质量；M1.9 需要用人工评估集检查证据相关性、引用贴合度与证据不足问题的保守性。
  - 若后续切换真实 QA 模型，需要观察一次引用修复是否足够，必要时增加更细粒度的引用-句子一致性检查。

---

## M1.9 评估集、验收与稳定性打磨

更完整的可调决策与实验清单见 `Doc/LoreWeaver_M1_Adjustable_Decisions.md`

## 7. 端到端开发顺序

推荐实际开发顺序如下：

1. M1.0 项目骨架与样本锁定；
2. M1.1 文本入库、规范化与章节切分；
3. M1.2 候选窗口切分；
4. M1.3 结构化抽取与引用定位；
5. M1.4 SQLite + BM25，先不急着接 Qdrant；
6. M1.4 Qdrant 向量索引；
7. M1.6 先实现向量 + BM25 双路召回；
8. M1.7 证据包组装；
9. M1.8 最小问答；
10. M1.5 中心 Span 图骨架；
11. M1.6 补齐图召回与 Reranker；
12. M1.9 评估与打磨。

说明：

- 文档编号按架构层次排列；
- 实际开发可先绕过 Neo4j，先让 SQLite + Qdrant + BM25 的证据问答跑通；
- 图骨架必须在 M1 完成前进入主链路，但不必是最早实现的模块；
- Reranker 可先用 mock 或简单打分接口，后续替换真实 Cross-Encoder。

---

## 8. M1 关键命令设计

### 8.1 入库

```bash
loreweaver ingest --source data/raw/sample.txt --title "Sample"
```

输出：

- `document_id`
- normalized 文本路径；
- 章节数；
- 入库报告路径。

### 8.2 切窗口

```bash
loreweaver windows build --document-id <document_id>
```

输出：

- 窗口数量；
- 平均窗口长度；
- 异常窗口。

### 8.3 抽取

```bash
loreweaver extract run --document-id <document_id> --limit 10
loreweaver extract run --document-id <document_id> --all
```

输出：

- 抽取成功数；
- 定位成功数；
- 失败队列。

### 8.4 建索引

```bash
loreweaver index build --document-id <document_id>
```

输出：

- SQLite 校验；
- Qdrant 写入数量；
- BM25 索引路径。

### 8.5 建图

```bash
loreweaver graph candidates --document-id <document_id>
loreweaver graph build --document-id <document_id> --clusters configs/clusters.yaml
```

输出：

- Cluster 数；
- 成员 Span 数；
- 边数量。

### 8.6 搜索调试

```bash
loreweaver search-vector --document-id <document_id> "问题"
loreweaver search-bm25 --document-id <document_id> "问题"
loreweaver search-graph --document-id <document_id> "问题"
loreweaver retrieve --document-id <document_id> "问题"
```

输出：

- 各路 Top-K；
- Union 后候选；
- Rerank 结果。

### 8.7 问答

```bash
loreweaver ask --document-id <document_id> "问题"
```

输出：

- 回答；
- 引用；
- Evidence Pack 路径；
- 耗时与 token 估算。

### 8.8 评估

```bash
loreweaver eval run --document-id <document_id> --questions data/eval/m1_questions.yaml
loreweaver eval report --run-id <run_id>
```

输出：

- 每题结果；
- 汇总指标；
- 坏例列表。

---

## 9. 配置文件建议

### 9.1 `configs/default.yaml`

```yaml
ingest:
  normalize_newlines: true
  remove_extra_blank_lines: true
  chapter_patterns:
    - "^第[一二三四五六七八九十百千万0-9]+章"
    - "^Chapter\\s+[0-9]+"

window:
  size_chars: 1200
  overlap_ratio: 0.2
  min_chars: 300
  max_chars: 1600

extraction:
  model: "Pro/deepseek-ai/DeepSeek-V3.2"
  temperature: 0
  max_retries: 2
  min_spans_per_window: 2
  max_spans_per_window: 12
  target_span_chars_min: 30
  target_span_chars_max: 800
  anchor_min_chars: 4
  anchor_max_chars: 80
  allow_overlap: true
  store_located_text: true
  store_uncovered_text: true

locator:
  fuzzy_threshold: 0.86
  search_scope: "window_first"

retrieval:
  graph_cluster_top_k: 4
  graph_span_per_cluster: 12
  vector_top_k: 30
  bm25_top_k: 30
  union_max_candidates: 80
  rerank_top_k: 15

evidence:
  pre_context_chars: 300
  post_context_chars: 500
  merge_gap_chars: 500
  max_evidence_chars: 40000
  max_blocks: 12

qa:
  model: "gpt-4o"
  temperature: 0
  require_citations: true
```

### 9.2 `configs/clusters.yaml`

```yaml
clusters:
  - cluster_name: "主角与核心势力关系"
    cluster_type: "character"
    center_span_id: "span_xxx"
    seed_entities: ["主角名", "势力名"]
    include_span_ids:
      - "span_aaa"
      - "span_bbb"

  - cluster_name: "力量体系线索"
    cluster_type: "power_system"
    center_span_id: "span_yyy"
    seed_keywords: ["灵力", "禁术", "仪式"]
```

说明：

- M1 允许人工配置 Cluster；
- 后续 M2 再把这部分做成半自动流程。

---

## 10. Prompt 设计要点

### 10.1 抽取 Prompt 要点

必须强调：

- 只基于给定窗口；
- 输出多个微观 Span，不要默认做整章总结；
- 允许 Span 重叠和嵌套；
- `start_anchor_quote` / `end_anchor_quote` 必须逐字来自原文；
- 不要改写 anchor；
- `key_quote` 只表达核心证据，不承担完整定位；
- 如果窗口信息贫乏，也要给低 salience，而不是硬编。

抽取关注点：

- 人物关系变化；
- 势力冲突；
- 地点与空间线索；
- 力量体系规则；
- 历史事件；
- 异常现象；
- 伏笔与隐喻。

### 10.2 回答 Prompt 要点

必须强调：

- 只使用证据块；
- 每个关键结论附 `[E#]`；
- 不存在的引用编号不得使用；
- 证据不足时明确说明；
- 推测和事实分开；
- 优先回答用户问题，不写泛泛总结。

### 10.3 Reranker 输入文本

每个候选 Span 的精排文本建议格式：

```text
章节：<chapter_title>
摘要：<micro_summary>
实体：<entities>
主题：<topics>
原文短引：<key_quote>
```

不要把超长原文直接送进 Reranker。

---

## 11. 测试策略

### 11.1 单元测试

必须覆盖：

- 章节切分；
- 窗口切分；
- anchor 精确定位；
- anchor 模糊定位；
- start/end anchor 组合消歧；
- 区间合并；
- 引用编号校验；
- Union 去重；
- Evidence Pack 生成。

### 11.2 集成测试

必须覆盖：

- 小文本端到端建库；
- 抽取 mock 模式；
- BM25 检索；
- 向量检索 mock 或真实模式；
- 问答 mock 模式；
- Evidence Pack 落盘。

### 11.3 人工验收测试

必须覆盖：

- 20-30 个问题；
- 每题 Evidence Pack；
- 每题人工评分；
- 至少 3 个坏例复盘。

---

## 12. 风险清单与应对

### 12.1 LLM 抽取 anchor 不可定位

风险：

- 模型改写原文；
- anchor 过短；
- anchor 过长；
- start/end anchor 顺序错误；
- start/end anchor 组合区间过大；
- 原文中重复出现。

应对：

- Prompt 约束逐字引用；
- 设置 anchor 长度范围；
- 精确匹配失败后局部模糊匹配；
- 多命中时结合窗口位置、跨度长度和匹配分消歧；
- 定位失败不进入主索引。

### 12.2 中文 BM25 效果不稳定

风险：

- 分词差导致召回弱；
- 专有名词被切碎。

应对：

- M1 先用简单方案跑通；
- 对实体词加入自定义词典；
- BM25 文档拼接 entities/topics；
- 后续替换更好的中文分词器。

### 12.3 图结构做重

风险：

- 过早陷入复杂知识图谱建模；
- 大量时间花在关系类型设计。

应对：

- 只保留 4 类边；
- 中心 Span 人工指定；
- 图只服务于召回和下钻；
- 复杂概念锻造放到后续阶段。

### 12.4 Reranker 本地部署阻塞

风险：

- 模型下载慢；
- 环境复杂；
- 推理耗时高。

应对：

- 先定义 Reranker 接口；
- 提供 mock reranker；
- 支持关闭 reranker 的降级模式；
- 真实模型接入作为 M1 后半段任务。

### 12.5 回答幻觉

风险：

- 模型把推测写成事实；
- 模型编造引用；
- 模型忽略证据不足。

应对：

- 强制引用校验；
- Prompt 区分事实、推测、未知；
- 无有效证据时允许拒答；
- 保存 Evidence Pack 便于复查。

## 13. M1 结束时应该留下什么

M1 完成后，仓库中应至少留下：

- 一条可运行的单书建库流水线；
- 一个可检索的样本文档；
- 一批定位成功的 Span；
- 一个可查询的向量索引；
- 一个可查询的 BM25 索引；
- 一个最小中心 Span 图；
- 一个可复现的混合召回流程；
- 一个 Evidence Pack 生成器；
- 一个带引用的问答 CLI；
- 一份 20-30 题评估集；
- 一份 M1 验收报告；
- 一批坏例与改进建议。

这些产物将直接支撑 Milestone 2：

- 报告生成模板；
- 更稳定的中心 Span 聚合；
- 主题级设定整理；
- 半自动聚类探索；
- 更细的引用与冲突标记。

## 14. M1 最小演示脚本

M1 完成时，应该能用下面脚本完成演示：

```bash
# 1. 入库
loreweaver ingest --source data/raw/sample.txt --title "M1 Sample"

# 2. 切窗口
loreweaver windows build --document-id <document_id>

# 3. 抽取与定位
loreweaver extract run --document-id <document_id> --all

# 4. 建索引
loreweaver index build --document-id <document_id>

# 5. 建图
loreweaver graph build --document-id <document_id> --clusters configs/clusters.yaml

# 6. 提问
loreweaver ask --document-id <document_id> "这个故事里的力量体系目前有哪些明确规则？"

# 7. 评估
loreweaver eval run --document-id <document_id> --questions data/eval/m1_questions.yaml
loreweaver eval report --run-id <run_id>
```

演示时必须展示：

- 章节坐标；
- Span anchor 定位；
- 三路召回结果；
- Rerank Top-K；
- Evidence Pack；
- 带引用回答；
- 评估摘要。

---

## 15. M1 的核心判断

M1 不是为了证明 LoreWeaver 已经“理解整本书”，而是为了证明以下工程命题：

1. 原文坐标体系可行；
2. LLM 抽取可以通过 anchor 反查变得可控；
3. Span 能作为可追溯证据单元；
4. 向量 + BM25 + 图召回能互补；
5. Reranker 能帮助筛掉错误候选；
6. Evidence Pack 能把检索结果变成可阅读原文；
7. 回答可以基于证据而不是凭空生成；
8. 系统失败时可以定位原因。

只要这 8 件事成立，LoreWeaver 就有了继续走向 M2/M3 的真正地基。
