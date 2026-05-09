# LoreWeaver Milestone 1 交付文档

版本：v0.1  
更新日期：2026-05-08  
阶段代号：M1 - 单书建库与证据问答  
当前实现阶段：M1.9  
原始详细计划：[LoreWeaver_Milestone_1.md](LoreWeaver_Milestone_1.md)

---

## 1. 阶段目标与交付状态

Milestone 1 的目标是验证 LoreWeaver 的第一条可信闭环：

```text
txt 样本
  -> 章节切分
  -> 候选窗口
  -> LLM 多 Span 抽取
  -> anchor 反查定位
  -> SQLite / Qdrant / BM25 / 图索引
  -> 图 + 向量 + BM25 混合召回
  -> Rerank
  -> Evidence Pack
  -> 带原文引用的回答
```

当前仓库已实现 M1.0 - M1.9，能够通过 CLI 完成单书入库、窗口切分、结构化抽取、索引构建、中心 Span 图、混合召回、证据包组装、带引用问答，以及章节级召回评估。M1.9 采用超长上下文 LLM 生成问题集与章节级 gold labels，LoreWeaver 输出 span 级召回后聚合为章节排序，并计算 Recall / NDCG / MRR 与坏例报告。

M1 不追求完整世界观分析产品，也不做 Web UI、多用户系统、百万字规模优化或复杂 Agent 调度。它的核心价值是证明“原文坐标 + 可复盘抽取 + 多路召回 + 证据问答”这条工程路线可行。

---

## 2. 核心原则

1. **原文坐标是事实地基**  
   所有可信证据最终都必须落到 `document_id / chapter_id / start_idx / end_idx / normalized text slice`。模型不得直接写入最终可信坐标，只能输出 `start_anchor_quote` 和 `end_anchor_quote`，由程序反查定位。

2. **每一步都必须可复盘**  
   入库报告、窗口报告、抽取报告、索引报告、图报告、召回报告、Evidence Pack 和 answer report 都落盘到 `data/runs/`，并带 `run_id`。

3. **Span 是可追溯证据单元**  
   LLM 在窗口内发现多个微观 Span，Span 可以重叠或嵌套；只有 `locator_status = located` 的 Span 进入索引、图和问答主链路。

4. **混合召回是默认主链路**  
   在线查询默认走图召回、向量召回、BM25 召回、Union 去重、Rerank 精排和证据区间合并。单路召回只作为调试或降级路径。

5. **回答模型是证据解释器**  
   `ask` 只能基于 Evidence Pack 回答，关键结论必须带 `[E###]` 引用。程序会校验引用编号是否存在，并把回答回写到报告与 SQLite。

---

## 3. 当前样本与关键产物

样本文本：

```text
data/raw/DawnSword_Chapter_1_260.txt
```

已入库文档：

```text
document_id: doc_59331b17113e
title: 黎明之剑
author: 远瞳
normalized: data/normalized/doc_59331b17113e.txt
sqlite: data/runs/loreweaver_m1.sqlite3
```

入库统计：

```text
raw size: 2625682 bytes
normalized chars: 882478
chapters: 258
content_hash: 59331b17113ef174b9f334c427a811b6957701de649ecc091f99d61c5e3c3f36
chapter boundary warnings: 0
```

窗口统计：

```text
candidate_windows: 1030
window_size_chars: 1200
overlap_ratio: 0.2
avg_window_length: 1036.59
min / max_window_length: 303 / 1255
cross_chapter_windows: 0
```

索引与图：

```text
BM25 index: data/indexes/doc_59331b17113e_bm25.json
Qdrant local path: data/indexes/qdrant
Qdrant collection: loreweaver_doc_59331b17113e_spans
graph mirror tables: center_span_clusters, span_edges
Neo4j: optional, disabled by default
```

---

## 4. 工程结构

主要目录：

```text
configs/
  default.yaml       # 阶段、样本、窗口、抽取、召回、证据、QA 配置
  models.yaml        # SiliconFlow/OpenAI-compatible 模型与价格配置
  storage.yaml       # SQLite / Qdrant / Neo4j / BM25 存储配置

data/
  raw/               # 原始文本
  normalized/        # normalized 可信坐标文本
  runs/              # SQLite、运行报告、调试记录
  indexes/           # Qdrant 本地索引、BM25 索引
  eval/              # 评估语料、问题集、预测与报告

loreweaver/
  ingest/            # 读取、规范化、章节切分、窗口切分
  extraction/        # LLM 抽取、Prompt、Schema、anchor 定位
  indexing/          # embedding 与索引构建管线
  storage/           # SQLite / Qdrant / BM25 / Neo4j 适配
  graph/             # CenterSpanCluster 与边构建
  retrieval/         # 图、向量、BM25、Union、Reranker
  evidence/          # 区间扩展、合并、Evidence Pack
  qa/                # 回答生成、引用校验与修复
  eval/              # M1.9 章节级召回评估入口

tests/unit/
  test_m1_1_ingest.py
  test_m1_2_windows.py
  test_m1_3_extraction.py
  test_m1_4_indexing.py
  test_m1_5_graph.py
  test_m1_6_retrieval.py
  test_m1_7_evidence.py
  test_m1_8_qa.py
  test_m1_9_eval.py
```

---

## 5. 核心数据对象

### Document

```text
document_id, title, author, source_path, normalized_path,
total_chars, total_chapters, content_hash, created_at
```

同一份 normalized 文本重复导入时 `content_hash` 应一致，不允许后续流程混用 raw 坐标和 normalized 坐标。

### Chapter

```text
chapter_id, document_id, chapter_index, chapter_title,
start_idx, end_idx, char_count
```

章节区间必须有序、不重叠，`normalized_text[start_idx:end_idx]` 必须能切回章节原文。

### CandidateWindow

```text
window_id, document_id, chapter_id, window_index,
window_start, window_end, text, uncovered_text
```

窗口只能落在单一章节内，是抽取输入缓存，不是最终知识单元。

### Span

```text
span_id, document_id, chapter_id, window_id, span_index_in_window,
micro_topic, span_type, micro_summary, entities, topics,
salience_score, start_anchor_quote, end_anchor_quote, key_quote,
span_start_idx, span_end_idx, located_text,
locator_confidence, locator_status, created_at
```

`locator_status = failed` 的 Span 不进入主索引。`located_text` 便于 DBeaver 或 SQLite 人工抽查，大规模建库时可通过配置关闭。

### CenterSpanCluster / SpanEdge

```text
CenterSpanCluster:
cluster_id, document_id, center_span_id, cluster_name,
cluster_type, micro_summary, member_span_ids, confidence, status

SpanEdge:
edge_id, document_id, from_id, to_id, from_type, to_type,
edge_type, weight, source
```

M1 图只保留轻量骨架：`SUPPORTS / RELATED_TO / MENTIONS_ENTITY / ADJACENT_CHAPTER`。

### QueryEvidencePack

```text
query_id, document_id, user_question, query_type,
retrieved_span_ids, cluster_ids, merged_intervals,
evidence_blocks, retrieval_sources, rerank_scores,
token_estimate, answer, created_at
```

每次 `evidence` 或 `ask` 都应落盘 Evidence Pack，回答引用编号必须能映射回 `evidence_blocks`。

### EvalQuestion / EvalPrediction

```text
EvalQuestion:
question_id, question, answer, profile, query_type,
required_facets,
expected_chapters[{chapter_id, chapter_index, relevance, weight, facet, reason}],
negative_chapters[{chapter_id, chapter_index, reason}]

EvalPrediction:
question_id, expected_chapters, predicted_chapters,
score{weighted_recall_at_k, hit_at_k, ndcg_at_k, core_recall_at_k,
facet_coverage_at_k, noise_at_k, mrr},
retrieval_report_path
```

M1.9 的 gold label 只标到章节级。`eval run` 将 `retrieve` 产出的 span 级 Top-K 按 `chapter_id` 聚合为章节排名，默认使用同章最高 `rerank_score` 作为章节分数，再计算 `weighted_recall_at_1/3/5/10/20`、`hit_at_k`、`ndcg_at_k`、`core_recall_at_k`、`facet_coverage_at_k`、`noise_at_k` 和 `mrr`。Broad profile 会扩大检索候选池与 `rerank_top_k`，用于暴露广域问题的遗失章节和无效章节。

---

## 6. 配置与外部依赖

推荐在 conda 环境内运行：

```bash
conda run -n loreweaver python -m loreweaver.cli status
```

安装依赖：

```bash
conda run -n loreweaver python -m pip install -e ".[m1,dev]"
```

环境变量：

```text
SILICONFLOW_API_KEY   # live 抽取、embedding、reranker、QA
DEEPSEEK_API_KEY      # M1.9 long-context eval question generation
QDRANT_URL            # 可选；未设置时使用 data/indexes/qdrant
QDRANT_API_KEY        # 可选
NEO4J_URI             # 可选
NEO4J_USERNAME        # 可选
NEO4J_PASSWORD        # 可选
```

当前模型配置：

```text
extraction: Pro/deepseek-ai/DeepSeek-V3.2
batch extraction: deepseek-ai/DeepSeek-V3.1-Terminus
embedding: Qwen/Qwen3-Embedding-0.6B
reranker: Qwen/Qwen3-Reranker-0.6B
qa: Pro/deepseek-ai/DeepSeek-V3.2
eval question generator: deepseek-v4-pro
providers: SiliconFlow / DeepSeek OpenAI-compatible API
```

关键参数：

```text
window.size_chars: 1200
window.overlap_ratio: 0.2
extraction.min/max_spans_per_window: 2 / 12
locator.fuzzy_threshold: 0.86
retrieval.graph_cluster_top_k: 4
retrieval.vector_top_k: 30
retrieval.bm25_top_k: 30
retrieval.union_max_candidates: 80
retrieval.rerank_top_k: 15
evidence.max_blocks: 12
evidence.max_evidence_chars: 40000
```

---

## 7. 常用命令

查看状态：

```bash
conda run -n loreweaver python -m loreweaver.cli --help
conda run -n loreweaver python -m loreweaver.cli status
```

入库与窗口切分：

```bash
conda run -n loreweaver python -m loreweaver.cli ingest \
  --source data/raw/DawnSword_Chapter_1_260.txt

conda run -n loreweaver python -m loreweaver.cli windows \
  --document-id doc_59331b17113e
```

小样本抽取调试：

```bash
conda run -n loreweaver python -m loreweaver.cli extract --limit 3 --mock
conda run -n loreweaver python -m loreweaver.cli extract --limit 3
```

增量抽取与窗口查看：

```bash
conda run -n loreweaver python -m loreweaver.cli extract \
  --list-windows --only pending --limit 20

conda run -n loreweaver python -m loreweaver.cli extract \
  --window-range 21-40

conda run -n loreweaver python -m loreweaver.cli extract \
  --window-id doc_59331b17113e_ch0021_win0001
```

批量抽取：

```bash
conda run -n loreweaver python -m loreweaver.cli extract --batch --limit 1000
conda run -n loreweaver python -m loreweaver.cli extract --batch-id <batch_id>
conda run -n loreweaver python -m loreweaver.cli extract --batch --batch-wait --batch-poll-interval 60
```

索引与检索调试：

```bash
conda run -n loreweaver python -m loreweaver.cli index --mock-embeddings
conda run -n loreweaver python -m loreweaver.cli search-bm25 高文 --top-k 3
conda run -n loreweaver python -m loreweaver.cli search-vector 旧时代秘密 --top-k 3 --mock-embeddings
```

图骨架：

```bash
conda run -n loreweaver python -m loreweaver.cli spans --top-salience 30
conda run -n loreweaver python -m loreweaver.cli graph --no-neo4j
conda run -n loreweaver python -m loreweaver.cli graph --no-neo4j --no-embeddings
conda run -n loreweaver python -m loreweaver.cli graph --list
```

混合召回、证据包与问答：

```bash
conda run -n loreweaver python -m loreweaver.cli retrieve \
  "塞西尔家族为什么会衰落？" --mock-embeddings --mock-reranker

conda run -n loreweaver python -m loreweaver.cli evidence \
  "塞西尔家族为什么会衰落，高文和这个家族的关系如何变化？" --no-reranker

conda run -n loreweaver python -m loreweaver.cli ask \
  "塞西尔家族和高文有什么关系？" --mock-embeddings --no-reranker --mock-answer
```

章节级召回评估：

```bash
conda run -n loreweaver python -m loreweaver.cli eval build-corpus \
  --chapter-start 1 --chapter-end 100

conda run -n loreweaver python -m loreweaver.cli eval generate \
  data/eval/corpora/doc_59331b17113e_ch001_100.json \
  --profile broad \
  --question-count 50 \
  --max-output-tokens 384000

conda run -n loreweaver python -m loreweaver.cli eval run \
  data/eval/question_sets/doc_59331b17113e_ch001_100_broad_v001.jsonl \
  --no-reranker

conda run -n loreweaver python -m loreweaver.cli eval report \
  data/eval/runs/<run_id>_predictions.jsonl
```

---

## 8. 端到端演示脚本

无 API 管线烟测：

```bash
conda run -n loreweaver python -m loreweaver.cli status
conda run -n loreweaver python -m loreweaver.cli ingest --source data/raw/DawnSword_Chapter_1_260.txt
conda run -n loreweaver python -m loreweaver.cli windows
conda run -n loreweaver python -m loreweaver.cli extract --limit 10 --mock
conda run -n loreweaver python -m loreweaver.cli index --mock-embeddings
conda run -n loreweaver python -m loreweaver.cli graph --no-neo4j
conda run -n loreweaver python -m loreweaver.cli ask \
  "塞西尔家族和高文有什么关系？" \
  --mock-embeddings --mock-reranker --mock-answer
```

live API 小样本：

```bash
export SILICONFLOW_API_KEY=<local-secret>

conda run -n loreweaver python -m loreweaver.cli extract --limit 3
conda run -n loreweaver python -m loreweaver.cli index --limit 30
conda run -n loreweaver python -m loreweaver.cli graph --no-neo4j
conda run -n loreweaver python -m loreweaver.cli evidence \
  "塞西尔家族为什么会衰落，高文和这个家族的关系如何变化？" \
  --no-reranker
conda run -n loreweaver python -m loreweaver.cli ask \
  "塞西尔家族和高文有什么关系？"
```

演示时应展示：

- `run_id` 与 JSON 报告路径；
- normalized 文本和章节坐标；
- Span 的 start/end anchor 与定位后的 `located_text`；
- BM25 / Qdrant / 图召回数量；
- Union 与 Rerank Top-K；
- Evidence Pack 的 `[E001]` 证据块；
- 最终回答中的引用编号和回映射结果。

---

## 9. 验证命令

基础验证：

```bash
conda run -n loreweaver python -m compileall loreweaver tests/unit
conda run -n loreweaver python -m pytest tests/unit
conda run -n loreweaver python -m ruff check loreweaver tests
```

分阶段单测：

```bash
conda run -n loreweaver python -m pytest tests/unit/test_m1_1_ingest.py
conda run -n loreweaver python -m pytest tests/unit/test_m1_2_windows.py
conda run -n loreweaver python -m pytest tests/unit/test_m1_3_extraction.py
conda run -n loreweaver python -m pytest tests/unit/test_m1_4_indexing.py
conda run -n loreweaver python -m pytest tests/unit/test_m1_5_graph.py
conda run -n loreweaver python -m pytest tests/unit/test_m1_6_retrieval.py
conda run -n loreweaver python -m pytest tests/unit/test_m1_7_evidence.py
conda run -n loreweaver python -m pytest tests/unit/test_m1_8_qa.py
```

SQLite 抽查：

```bash
sqlite3 data/runs/loreweaver_m1.sqlite3 \
  "select count(*) from documents;
   select count(*) from chapters where document_id='doc_59331b17113e';
   select count(*) from candidate_windows where document_id='doc_59331b17113e';
   select count(*) from spans where document_id='doc_59331b17113e' and locator_status='located';
   select count(*) from center_span_clusters where document_id='doc_59331b17113e';
   select count(*) from evidence_packs where document_id='doc_59331b17113e';"
```


---

## 10. Prompt 与质量约束

抽取 Prompt 必须约束：

- 只基于当前窗口；
- 输出多个微观 Span，不做整章总结；
- 允许 Span 重叠和嵌套；
- `start_anchor_quote` / `end_anchor_quote` 必须逐字来自窗口原文；
- `key_quote` 只表达核心证据，不承担完整定位；
- 信息不足时给低 salience，不硬编。

回答 Prompt 必须约束：

- 只能使用 Evidence Pack；
- 每个关键结论必须附 `[E###]`；
- 不存在的引用编号不得使用；
- 证据不足必须明确说明；
- 事实、推测、未知要分开。

Reranker 输入建议：

```text
章节：<chapter_title>
小话题：<micro_topic>
摘要：<micro_summary>
实体：<entities>
主题：<topics>
原文短引：<key_quote>
```

不要把超长原文直接送入 Reranker。

---

## 11. 已知限制与后续补齐

1. **M1.9 gold labels 仍是 silver set**  
   章节级 gold 由超长上下文 LLM 生成，适合做自动回归与坏例筛选；正式验收仍需要对低分样本和低置信样本做人工抽查。

2. **真实回答质量仍需人工评估**  
   M1.8 的 mock answer 只验证链路和引用约束，不代表最终问答质量。切换 live QA 模型后，要检查引用贴合度和证据不足时的保守性。

3. **抽象问题仍是风险区**  
   live 召回烟测显示明确实体、地点、事实类问题较稳定；力量体系、伏笔异常、宏观主题类问题仍需强化 query router、候选池构造和 rerank 输入。

4. **图骨架是 M1 轻量版本**  
   当前 Cluster 构建为 embedding-aware + 规则启发式，足够调试主链路，但不是最终自动聚类质量。

5. **Qdrant collection 采用按文档重建策略**  
   适合单书 M1。后续支持增量建库时，需要补 collection 级别的 run/version 管理。

6. **BM25 中文分词仍是轻量方案**  
   当前使用字、二元、三元 token 化，足够支撑 M1 实体验证；若术语召回不足，再引入更强中文分词器或实体词典。

7. **Neo4j 是可选调试路径**  
   M1 主链路优先消费 SQLite graph mirror。Neo4j 可用于可视化，不应成为本地验收阻塞项。

---

## 12. M1 交付结论

M1 已经证明以下工程命题：

1. normalized 原文坐标体系可以作为可信地基；
2. LLM 抽取结果可以通过 anchor 反查变得可控；
3. Span 可以作为可追溯的微观证据单元；
4. 向量、BM25、图召回能形成互补；
5. Reranker 接口和降级策略已经具备；
6. Evidence Pack 能把检索结果转化为可阅读原文证据；
7. 问答可以被约束在证据块内，并用程序校验引用；
8. 系统失败时可以通过落盘报告定位到入库、抽取、索引、召回、证据组装或回答阶段。

M2 可以在此基础上推进报告生成模板、更稳定的中心 Span 聚合、主题级设定整理、半自动聚类探索，以及更细粒度的引用与冲突标记。
