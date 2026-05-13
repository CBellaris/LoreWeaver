# LoreWeaver M1 可调决策与实验清单

版本：v0.2
创建日期：2026-04-24
更新日期：2026-04-27
定位：记录 M1 开发过程中那些“不是简单参数，但也不应过早定死”的工程与架构决策，供 M1.9 评估、A/B、盲测和 M2 前稳定性打磨使用。

---

## 1. 文档目的

LoreWeaver M1 的核心链路已经基本成型，但很多模块都存在可调整空间：

- 有些是明确参数，例如 `top_k`、窗口大小、上下文扩展长度；
- 有些是策略选择，例如 Query Router 的 bucket 设计、Union 融合方式、Cluster 召回文本字段；
- 有些是阶段性妥协，例如当前 Cluster Summary 仍主要来自规则/Span 摘要拼接，后续可能替换为 LLM 聚合总结；
- 有些是可作为产品能力演进的架构接口，例如是否建立 Cluster 向量索引、是否支持多标签 query routing。

这份文档的目的不是马上修改主链路，而是把这些决策抓出来，避免后续只凭单题观感调参。M1.9 应将其中重要条目转化为可重复的 A/B 或盲测实验。

原则：

- 每个实验只改少数变量；
- 使用同一批评估问题对照 baseline；
- 保留 query run、retrieval report、Evidence Pack 与人工评分；
- 先评分再揭示实验配置；
- 若只在个别题目变好但整体下降，不进入默认主链路，只作为问题类型特化策略保留。

---

## 2. 当前已整理范围与 M1 Baseline

本版本已补齐 M1.1-M1.8 的主要可调决策。当前主链路为：

```text
raw txt
  -> 文本读取 / normalized 文本
  -> 章节切分
  -> CandidateWindow
  -> LLM 多 Span 抽取
  -> start/end anchor 定位
  -> Span 入 SQLite
  -> Embedding / Qdrant / BM25
  -> CenterSpanCluster / SpanEdge
  -> Query Router
  -> 图召回 / 向量召回 / BM25 召回
  -> Union 融合
  -> Reranker 或 Noop Reranker
  -> Top-K Span
  -> Evidence Pack 区间扩展 / 合并 / 预算裁剪
  -> QA Answerer
  -> 引用校验 / 一次修复
```

当前代码与运行产物显示，M1 已经具备从建库到带引用回答的最小闭环。M1.9 的重点应从“继续堆功能”切换为“把可调决策变成可重复实验”。

当前样本与运行基线：

- 样本：`data/raw/DawnSword_Chapter_1_260.txt`；
- `document_id`：`doc_59331b17113e`；
- normalized 字符数：`882478`；
- 章节数：`258`；
- 章节识别策略：`real_chapter_patterns`；
- 最近一次窗口产物：`258` 个 chapter 窗口，平均 `3420.19` 字符；
- 最近一次付费抽取小批：`10` 个窗口，`108` 个 Span，`107` 个 located，locator 成功率 `0.9907`；
- 当前索引产物：`209` 个 located Span，Embedding 模型 `Qwen/Qwen3-Embedding-0.6B`，维度 `1024`；
- 当前图产物：`4` 个 Cluster，`407` 条边，其中 `SUPPORTS=32`、`RELATED_TO=28`、`MENTIONS_ENTITY=90`、`ADJACENT_CHAPTER=257`；
- 最近一次问答样例：“塞西尔家族和高文有什么关系？”；
- 最近一次召回：graph `36`、vector `30`、BM25 `30`，Union `61` 个候选，multi-source `24`；
- 最近一次 Evidence Pack：Top-K `15` 个 Span 扩展为 `11` 个证据块，证据文本 `13300` 字符；
- 最近一次 QA：`Pro/deepseek-ai/DeepSeek-V3.2`，引用校验通过，未触发 repair。

仍未真正实现的 M1.9 评估模块：

- `loreweaver/eval/question_set.py`；
- `loreweaver/eval/runner.py`；
- `loreweaver/eval/metrics.py`；
- CLI 中 `eval` 仍是 placeholder。

## 3. 所有模块待测试/调整项

### 3.1 M1.1 文本规范化与章节切分

当前做法：

- 读取编码优先级为 `utf-8-sig / utf-8 / gb18030`；
- normalized 文本作为唯一可信坐标层；
- 清理 BOM、换行差异、行尾空白、分隔线、连续空行和常见广告行；
- 章节识别优先使用真实章标题，再回落到配置 pattern 或固定长度伪章节；
- 支持 `--max-chapters` 做小范围建库；
- 同一 normalized 文本通过 `content_hash` 与稳定 `document_id` 去重。

当前理解：

M1.1 的核心不是“清洗得最干净”，而是建立一个后续永远能回切的坐标基座。当前样本章节识别很稳，258 章无边界告警。但清洗规则一旦过强，会把某些“像广告/分隔线”的正文也移除；一旦过弱，又会让抽取模型读到噪声。

待测试项：

- `normalize_newlines` 与 `remove_extra_blank_lines` 的开关组合；
- 广告行正则的 strict / loose / off 三档；
- 是否把样本前置信息纳入特殊章节，而不是保留为 uncovered prefix；
- 章节标题 pattern 对番外、序章、卷标题、`【 第 N 部分 】` 的误伤率；
- `fallback_chapter_chars` 在无标准章节文本上的 8000 / 12000 / 16000 对照；
- 极短章节是否合并，还是保留为独立章节；
- 是否需要 raw-to-normalized 映射，支撑未来回到原始文件定位。

当前倾向：

继续把 normalized 文本作为唯一坐标真相。M2 前不要引入 raw 坐标双轨，避免调试复杂度上升。更值得补的是清洗报告，把每条被清掉的广告/分隔线样例落盘，便于人工抽查误删。

关联模块/文件：

- `loreweaver/ingest/reader.py`
- `loreweaver/ingest/normalizer.py`
- `loreweaver/ingest/chapter_splitter.py`
- `loreweaver/ingest/pipeline.py`
- `loreweaver/storage/sqlite_store.py`
- `configs/default.yaml`

### 3.2 M1.2 候选窗口切分

当前做法：

- 支持章节内滑窗，也支持 `--by-chapter` 把每章作为一个窗口；
- 配置默认值为 `size_chars=1200`、`overlap_ratio=0.2`、`min_chars=300`、`max_chars=1600`；
- 滑窗模式下末尾过短窗口并入前一窗口；
- 每个窗口不跨章节，窗口文本缓存到 SQLite；
- `candidate_windows.uncovered_text` 用于保存 located Span 未覆盖的窗口片段。

当前理解：

M1 曾跑过滑窗产物，也有最近一次 chapter-window 产物。二者取舍明显：

- 滑窗更接近低成本、低延迟抽取，但会产生更多窗口、更多重复 Span；
- 章节整窗保留完整局部叙事，适合少量章节精抽，但单窗 3000-4000 字会让模型倾向输出更多 Span，且定位和抽取失败的影响范围更大。

待测试项：

- `sliding_window` vs `chapter` 对抽取质量、成本、locator 成功率的影响；
- `window_size_chars`：800 / 1200 / 1600；
- `overlap_ratio`：0.1 / 0.2 / 0.3；
- 是否按自然段落边界吸附窗口起止点；
- 是否为对话密集章节使用更小窗口；
- `min_window_chars` 与末尾并入策略；
- `uncovered_text` 抽查比例：每批窗口抽查 5%、10% 还是只抽查高 salience 章节；
- 是否在同一章节内限制重复 Span，减少重叠窗口引发的重复抽取。

当前倾向：

保留两种模式。M1.9 应用同一批章节做 `chapter` vs `1200/0.2 sliding` 对照，不只看 locator 成功率，也看最终问答证据是否更完整。若只是建库成本可控，章节整窗可作为“高质量抽取档”；滑窗作为“常规成本档”。

关联模块/文件：

- `loreweaver/ingest/window_splitter.py`
- `loreweaver/models/window.py`
- `loreweaver/cli.py`
- `configs/default.yaml`

### 3.3 M1.3 LLM Span 抽取与 Anchor 定位

当前做法：

- 抽取模型默认 `Pro/deepseek-ai/DeepSeek-V3.2`，`temperature=0`；
- 每窗目标输出 2-12 个微观 Span；
- Span 类型限定为 `dialogue_exchange / relationship_signal / location_lore / faction_lore / power_rule / event / mystery_clue / object_lore / scene_action / other`；
- 模型只输出 `start_anchor_quote / end_anchor_quote`，程序负责定位全局坐标；
- anchor 配置为 4-80 字符，过长 anchor 按 start/end 边界方向裁剪；
- 定位顺序为 exact -> normalized -> fuzzy；
- fuzzy threshold 当前为 `0.86`；
- 目标 Span 长度为 30-800 字符；
- 支持 `store_located_text` 与 `store_uncovered_text`；
- JSON 解析支持从非严格输出中抽取 `{...}`，并有 best-effort payload 兜底。

当前理解：

M1.3 决定了后续所有索引和证据的原料质量。当前小批抽取的 locator 成功率很高，但这不等于 Span 粒度已经最佳。每窗输出数量、类型体系、summary 风格和 anchor 长度都会影响召回、图聚合和 Evidence Pack 的最终质量。

待测试项：

- 抽取模型对照：当前 DeepSeek vs 低成本模型 vs 更强模型；
- `min_spans_per_window / max_spans_per_window`：更保守少拆 vs 更激进多拆；
- `target_span_chars_min/max`：短 Span 是否导致证据碎片化，长 Span 是否导致引用过宽；
- `anchor_min_chars / anchor_max_chars`：短 anchor 的多命中风险与长 anchor 的失配风险；
- `fuzzy_threshold`：0.82 / 0.86 / 0.90；
- span_type 是否新增 `law_policy`、`economy_resource`、`social_order`；
- `micro_summary` 更偏事实复述 vs 因果解释；
- `salience_score` 是否应分成剧情重要性、世界观重要性、证据可靠性；
- 是否启用 provider 的 JSON response format；
- `located_text` 是否长期保存，还是只在抽查与 debug 运行中保存；
- 是否把失败 Span 的 raw output 和 locator candidates 做更完整的二次重跑队列。

当前倾向：

定位链路继续保持程序确定性，不让模型写坐标。M1.9 优先评估 Span 粒度，而不是急着换模型：同一批窗口分别用“保守拆分”和“激进拆分”，看后续召回 Top-K 与 Evidence Pack 是否更可读。

关联模块/文件：

- `loreweaver/extraction/prompts.py`
- `loreweaver/extraction/schemas.py`
- `loreweaver/extraction/extractor.py`
- `loreweaver/extraction/locator.py`
- `loreweaver/extraction/retry.py`
- `loreweaver/models/span.py`
- `configs/default.yaml`
- `configs/models.yaml`

### 3.4 M1.4 索引与 Embedding

当前做法：

- Embedding 模型为 `Qwen/Qwen3-Embedding-0.6B`，期望维度 `1024`；
- embedding input 当前由 `micro_summary + entities + topics` 组成；
- `include_key_quote=false`；
- `include_located_text=false`；
- SQLite 保存 embedding cache；
- Qdrant 支持远程 URL 或本地 path；
- Qdrant payload 保存 `span_type / salience_score / span_start_idx / span_end_idx / locator_confidence` 等字段；
- BM25 文档由 `micro_summary + entities + topics + key_quote` 拼接；
- BM25 tokenizer 对中文生成 char、bigram、trigram。

当前理解：

Embedding 当前更像“摘要语义索引”，BM25 当前更像“短文本精确保险”。这个分工合理，但还不够稳定：摘要语义可能漏掉原文关键措辞，BM25 又缺少字段权重和实体 boost。

待测试项：

- embedding input 加入 `key_quote`；
- embedding input 加入短截断 `located_text`；
- `key_quote` 与 `located_text` 分开建多向量，而不是拼进同一个向量；
- 是否给 `span_type` 进入 embedding input；
- 是否建立 Cluster 级向量索引；
- embedding cache key 是否需要包含输入字段配置版本；
- Qdrant payload 是否补充 `cluster_ids`，便于过滤或解释；
- BM25 字段权重：entities > topics > summary > key_quote；
- BM25 query 侧实体/别名扩展；
- 轻量 tokenizer vs 更强中文分词器；
- `vector_top_k / bm25_top_k` 30、50、80 的召回-噪声曲线。

当前倾向：

先测试 `include_key_quote=true`，再谨慎测试 `located_text`。`located_text` 可能提升细节召回，但也会把叙事噪声带进摘要向量。Cluster 向量索引值得作为 M1.9 的独立实验，不要和 Span embedding 输入改动混在一次实验里。

关联模块/文件：

- `loreweaver/indexing/embeddings.py`
- `loreweaver/indexing/pipeline.py`
- `loreweaver/storage/qdrant_store.py`
- `loreweaver/storage/bm25_store.py`
- `configs/default.yaml`
- `configs/models.yaml`
- `configs/storage.yaml`

### 3.5 M1.5 CenterSpanCluster 构建

当前做法：

- Cluster 类型限定为 `character / faction / location / power_system / history / mystery`；
- 先按规则把 Span 分类到 cluster_type；
- 中心 Span 优先使用向量 medoid + salience + 类型 bonus；
- 成员打分由 vector、entity、topic、BM25 lexical、chapter proximity、salience 加权；
- 默认 `cluster_count=4`、`members_per_cluster=8`、`min_members=5`；
- 当前权重为 vector 0.4、entity 0.2、topic 0.15、BM25 0.1、chapter 0.1、salience 0.05；
- Cluster name 与 micro_summary 由规则生成；
- 生成 `SUPPORTS / RELATED_TO / MENTIONS_ENTITY / ADJACENT_CHAPTER`；
- SQLite 是主落盘，Neo4j 是可选同步。

当前理解：

M1.5 已经证明“图骨架能进入问答主链路”，但它仍是很轻的图。当前 4 个 Cluster 对样例问题有效，但会漏掉 `power_system` 和 `mystery` 等小样本中数量较少、但对世界观分析价值很高的主题。

待测试项：

- `cluster_count`：4 / 6 / 8 / 10；
- `members_per_cluster`：8 / 12 / 20；
- `min_members_per_cluster` 是否压掉少量但高价值的 mystery/power cluster；
- 类型分类规则是否过度依赖关键词；
- 中心 Span 选择：自动 medoid vs salience top vs 人工 seed；
- 成员权重：vector 权重提高是否带来语义泛化，entity/topic 权重提高是否更稳；
- chapter proximity 是否让 Cluster 过度局部化；
- `cluster_name / micro_summary` 是否改为 LLM summary；
- `member_span_ids` 是否需要人工确认状态；
- `SUPPORTS` 边权是否应保留各 component score；
- `MENTIONS_ENTITY` 是否只对 cluster members 建，还是对全量 located Span 建。

当前倾向：

M1.9 优先做 `cluster_count=4` vs `8`，并测试 LLM 生成 `cluster_retrieval_summary`。图的目标是帮助宏观问题找到阅读框架，不要过早追求全自动实体关系图。

关联模块/文件：

- `loreweaver/graph/center_span.py`
- `loreweaver/graph/edge_builder.py`
- `loreweaver/models/cluster.py`
- `loreweaver/storage/sqlite_store.py`
- `loreweaver/storage/neo4j_store.py`
- `configs/default.yaml`
- `configs/storage.yaml`

### 3.6 M1.6 召回模块

#### Query Router Buckets

当前 buckets：

```text
character_relation
faction_history
location
power_system
timeline
unknown
```

当前理解：

这些 buckets 作为 M1 的最小路由是合理的，但更像“报告主题分类”，不完全等同于“检索意图分类”。真实问题常常跨类，例如：

- “塞西尔家族为什么衰落”既是 `faction_history`，也带 `timeline`；
- “塞西尔领遭遇了什么灾难”更像 `event_disaster`，当前容易落到 `unknown`；
- “高文复活有什么异常”应该进入 `mystery/anomaly`，当前也容易落到 `unknown`；
- “力量体系有哪些规则”里的“规则”可能误召回社会规则、贵族习俗、法律权利。

待测试项：

- 单标签路由 vs 多标签路由，例如输出 `{faction_history: 0.7, timeline: 0.4}`；
- 是否新增 `mystery/anomaly`、`event_disaster`、`social_order/law_policy`；
- `unknown` 是否只作为兜底，而不是实际检索策略；
- Router 使用关键词规则、LLM 分类、embedding 分类的 A/B；
- 不同问题类型是否应影响 `top_k`、graph cluster 偏置、reranker 输入、Evidence Pack 组装。

当前倾向：

M1.9 优先测试多标签轻量路由，并新增 `mystery/anomaly`。这对伏笔、异常、复活疑点类问题可能收益最大。

#### 图召回的 Cluster 元数据字段

当前图召回使用：

```text
cluster_name
cluster_type
micro_summary
```

然后与用户 query 做 BM25/token overlap。

当前理解：

这个选择稳、噪声少、容易解释。但 Cluster 元数据较短，一旦用户问题用词与 cluster 摘要不一致，就会漏召回。尤其是中文语义表达中，“复活异常 / 身份疑点 / 黑钢棺材 / 精神状态”可能属于同一 mystery 主题，但词面碰撞不一定强。

待测试项：

- 只用 `cluster_name + cluster_type + micro_summary`；
- 加入中心 Span 的 `micro_summary / key_quote`；
- 加入成员 Span 的 top entities/topics；
- 加入由 LLM 生成的 `cluster_retrieval_summary`；
- 对字段加权：`cluster_type`、`cluster_name`、`summary`、`member keywords` 权重不同；
- 对比“短而干净的 Cluster 文本”与“包含更多成员信息的 Cluster 文本”。

当前倾向：

不要把所有成员文本直接塞进 Cluster 检索文本。更稳的候选结构是：

```text
cluster_retrieval_text =
  cluster_name
  cluster_type_label
  llm_cluster_summary
  top_entities
  top_topics
  center_span_summary
```

#### 图召回打分方式

当前方式：

```text
BM25 / token overlap over cluster metadata
```

当前理解：

只用 BM25 不够。BM25 对实体、术语、专名很好，但对抽象问法弱。例如“这个家族为什么不行了”不一定命中“衰落”。图召回应支持 lexical + embedding 双分。

待测试项：

- BM25-only；
- embedding-only；
- BM25 + embedding 加权；
- query type 自适应权重：实体类偏 BM25，抽象类偏 embedding；
- Cluster embedding 使用 `cluster_retrieval_text`；
- 是否复用 M1.5 的 embedding-aware 成员向量信息；
- Cluster 向量索引是否单独建立，而不是只查 Span 向量。

当前倾向：

M1.9 做 `BM25-only` vs `BM25+embedding` 对照。不要直接替换，先看图召回命中率和最终 Top-K 质量。

#### 图召回下钻策略

当前方式：

- 选中 Cluster；
- 补充中心 Span；
- 沿 `SUPPORTS` 取成员 Span；
- 按边权排序，取 `graph_span_per_cluster`。

待测试项：

- 中心 Span 是否始终强制保留；
- `graph_span_per_cluster` 固定值 vs 按 cluster_score 动态分配；
- 是否补充 `RELATED_TO` 二跳 Span；
- 是否加入章节覆盖约束，避免候选集中在同一章节；
- 是否按 query type 做 span_type 过滤或加权；
- 图召回不足时，是否用 cluster name 做二次向量召回扩展。

当前倾向：

图召回的价值是提供宏观框架和高价值锚点，不应承担全量召回。它应给 Union 提供高置信候选，而不是包办证据池。

#### 向量召回策略

当前 Span embedding 输入：

```text
micro_summary
entities
topics
```

当前理解：

这是 M1 的合理选择，偏向“摘要语义检索”。但对细节证据和原文表达可能不足。

待测试项：

- embedding 输入是否加入 `key_quote`；
- 是否加入短截断 `located_text`；
- 是否按字段加权或多向量存储；
- query 是否做改写/扩展，例如把“为什么衰落”扩写成“原因、历史、领地、资源、贵族地位”；
- 不同 query type 是否使用不同 `vector_top_k`；
- 是否建立 Cluster 向量索引；
- 是否保留不同 embedding 模型的 A/B 结果。

当前倾向：

先测试加入 `key_quote`，谨慎测试 `located_text`。`located_text` 可能提升细节召回，也可能引入叙事噪声。

#### BM25 召回策略

当前 BM25：

- 中文字、bigram、trigram token；
- 文档文本由 `micro_summary + entities + topics + key_quote` 拼接。

当前理解：

足以支撑 M1 实体召回，但不是最终形态。BM25 应定位为专名、术语、别称、精确表述的保险丝。

待测试项：

- 引入实体词典，把抽取出的 entities/topics 作为强 token；
- 字段权重：entities > topics > summary > key_quote；
- query 侧同义词/别名扩展；
- 人名、地名、势力名精确命中 boost；
- BM25 top_k 增大后，Reranker 是否能压住噪声；
- 更强中文分词器与当前轻量 tokenizer 的对照。

当前倾向：

优先测试实体词典和字段权重，而不是立刻替换分词器。

#### Union 融合策略

当前 Union 保留：

- `sources`
- 各路原始分；
- 各路归一化分；
- `fused_score`
- 多源命中奖励；
- graph bonus；
- entity coverage。

当前理解：

Union 是可调最多的地方，但它不应追求最终排序。它的目标是形成“足够宽但不过脏”的候选池，最终质量交给 Reranker 和 Evidence Pack。

待测试项：

- 多源命中奖励是否过强；
- graph bonus 是否导致图里的弱相关候选挤掉向量强相关候选；
- entity coverage 是否按 query type 启用；
- BM25 精确实体命中是否应强 boost；
- 是否加入 diversity 约束，避免候选都来自同一章节或同一 cluster；
- `union_max_candidates` 对 Reranker 质量与耗时的影响；
- 是否按 query type 使用不同 Union 权重。

当前倾向：

Union 应更偏召回，不要过早剪掉候选；但需要观察候选池扩大后 Reranker 是否还能稳定压住噪声。

#### Reranker 策略

当前 Reranker：

- 默认配置主选型为 `Qwen/Qwen3-Reranker-0.6B` via SiliconFlow；
- fallback 记录为 `BAAI/bge-reranker-v2-m3`；
- 支持 mock/noop；
- 默认配置仍关闭 live reranker，避免本地误触付费 API。

当前输入格式：

```text
章节：<chapter_title>
摘要：<micro_summary>
实体：<entities>
主题：<topics>
原文短引：<key_quote>
```

当前理解：

真实 API 小样本显示，实体事实类问题表现很好；抽象设定类与伏笔异常类问题更依赖候选池和 reranker 输入结构。Reranker 分数存在饱和现象，不适合现在启用硬阈值。

待测试项：

- 输入是否加入 `span_type`；
- 输入是否加入 `cluster_name / cluster_type`；
- 输入是否加入 `located_text` 的 100-200 字短截断；
- 是否按 query type 使用不同输入模板；
- `Qwen/Qwen3-Reranker-0.6B` vs `BAAI/bge-reranker-v2-m3`；
- 是否补测更大 Qwen3 Reranker；
- 是否保留 rerank score 阈值，或只信排序；
- 是否把多源命中信息显式传入 reranker 输入。

当前倾向：

下一步最值得测试的是加入：

```text
类型：<span_type>
所属簇：<cluster_name>
```

这可能比直接加入长原文更稳。

#### 召回可观测性与评估

当前已经保存：

- retrieval report；
- 每路召回数量；
- Union 候选；
- Rerank Top-K；
- 每个 Top-K 的来源；
- query_runs。

待新增观测项：

- 每题三路召回各自贡献了哪些最终 Top-K；
- 人工标记 Top-K 为“核心证据 / 辅助证据 / 噪声”；
- 每个坏例归因：router、graph、vector、BM25、union、reranker、evidence merge、answerer；
- baseline/variant 盲测标签；
- 按 category 分开统计，不只看总分；
- 记录召回耗时、rerank 耗时、候选池大小与最终 evidence block 数量。

当前倾向：

M1.9 的核心不是“调到某个最好参数”，而是识别哪些旋钮值得保留。召回模块优先验证：

```text
Query Router
Cluster metadata
Graph BM25+embedding
Union weights
Reranker input
```
### 3.7 M1.7 Evidence Pack 组装

当前做法：

- 从 retrieval Top-K 中读取带坐标的 Span；
- 对每个 Span 做章节内上下文扩展；
- 默认 `pre_context_chars=300`、`post_context_chars=500`；
- 同章 interval 在 `merge_gap_chars=500` 内合并；
- 默认 `max_evidence_chars=40000`、`max_blocks=12`；
- priority score 为 rerank score 加多来源 bonus；
- 先保证章节覆盖，再按优先级补齐；
- EvidenceBlock 引用编号为 `[E001]` 递增；
- citation id 做格式与唯一性校验。

当前理解：

M1.7 的风险不在“能不能切出原文”，而在证据包是否刚好适合 QA 阅读。当前样例问题的 Evidence Pack 达到 11 个块、13300 字符，对单个关系问题已经偏宽。宽证据包会提高覆盖率，也可能让回答模型在引用时抓取不够精确的段落。

待测试项：

- `pre_context_chars / post_context_chars`：100/200、200/300、300/500；
- `merge_gap_chars`：0 / 200 / 500；
- `max_blocks`：6 / 8 / 12；
- `max_evidence_chars`：12000 / 20000 / 40000；
- citation 顺序：文本时间顺序 vs rerank 顺序；
- 是否把 Span metadata 附在 EvidenceBlock 前，帮助模型理解块为何入选；
- 是否保留 Top-K 中未入选的 Span 摘要作为“证据候补”；
- 是否按 query_type 选择扩展长度：timeline 更长，实体事实更短；
- 是否避免多个 evidence blocks 覆盖近似重复语义；
- 合并后 source_span_ids 多时，是否需要标明每个 Span 在块内的大概位置。

当前倾向：

先收紧 evidence 上下文：对实体/关系类问题测试 `pre=150/post=250/merge_gap=200/max_blocks=8`。M1.9 评分时要单独标注“证据完整性”和“证据噪声”，不要只看最终回答是否顺口。

关联模块/文件：

- `loreweaver/evidence/interval.py`
- `loreweaver/evidence/assembler.py`
- `loreweaver/evidence/citation.py`
- `loreweaver/models/evidence.py`
- `configs/default.yaml`

### 3.8 M1.8 QA Answerer

当前做法：

- `ask` 会串起 retrieve -> evidence -> answer；
- QA provider 使用 OpenAI-compatible chat client；
- 支持 `--mock-answer`；
- Prompt 明确要求只能基于 `evidence_blocks`，且输出包含“结论、证据、分析、不确定性”；
- Prompt 会附带相关 Cluster 摘要；
- `validate_answer_citations` 校验未知引用和缺失引用；
- 如果首次回答引用校验失败，会用同一 client 做一次 repair；
- 通过 `evidence_packs.answer` 与 answer report 落盘。

当前理解：

M1.8 已经从“能回答”走到“能校验引用编号”。但当前校验只验证引用编号存在，不验证每条结论是否真的被引用文本支持。QA 的下一步风险是“引用格式正确，但证据归因过宽或解释过度”。

待测试项：

- QA 模型对照：当前 DeepSeek vs 更强长上下文模型 vs 小模型；
- Prompt 结构：固定四段式 vs 按 query_type 输出；
- 是否要求每个证据块先提炼一句“证据事实”再综合；
- evidence blocks 是否按时间顺序、rerank 顺序或混合顺序输入；
- Cluster 摘要是否会引入未由 EvidenceBlock 直接支持的判断；
- 证据不足时是否允许无引用拒答，还是必须引用“证据不足的原因块”；
- repair 是否应换用更严格的小模型/规则，而不是同一模型；
- 是否引入 claim-level citation audit：抽取回答中的结论句，逐句验证引用；
- `configs/default.yaml` 中 `qa.model` 与 `configs/models.yaml` 中 `models.qa.name` 的优先级是否需要统一，避免实验记录混乱；
- 是否把最终 answer 输出为结构化 JSON，便于评估。

当前倾向：

保留当前引用编号校验作为硬门槛，再增加“人工 claim 支持度评分”。自动 claim audit 可以作为 M2 工具，不必阻塞 M1.9，但 M1.9 人工评分表里应把“引用是否真的支持结论”作为单独列。

关联模块/文件：

- `loreweaver/qa/prompts.py`
- `loreweaver/qa/answerer.py`
- `loreweaver/cli.py`
- `configs/default.yaml`
- `configs/models.yaml`

### 3.9 SQLite、运行报告与可观测性

当前做法：

- SQLite 保存 documents、chapters、candidate_windows、spans、locator_candidates、failures、embedding_cache、clusters、edges、query_runs、evidence_packs；
- 每个阶段会生成 `data/runs/<run_id>_*_report.json`；
- 重跑 ingest/windows/extract 会清理其下游产物，避免旧索引混入新坐标；
- query_runs 保存 retrieval 和 answer report；
- Evidence Pack 能复现送给回答模型的原文证据。

当前理解：

M1 的可复盘性已经打了很好的底。现在缺的是“实验级别”的可观测性：同一题不同配置之间如何对齐比较，如何记录 variant，如何让人工评分不被配置名称影响。

待测试项：

- 每个 report 增加 `experiment_id / variant_id / baseline_id`；
- 保存有效配置快照，而不是只保存默认路径；
- 对同一 query 的多个 run 做成对比较；
- 每题标注坏例归因：ingest/window/extract/index/graph/retrieval/evidence/qa；
- 记录耗时：embedding、vector search、BM25、graph、union、rerank、evidence、answer；
- 记录 token 与成本：extraction、embedding、qa、repair；
- query_runs 是否应按 `query_id + variant_id` 存多条，避免覆盖；
- 是否把人工评分表放入 `data/eval/` 并与 report path 互链。

当前倾向：

M1.9 先不做复杂 dashboard。增加一个简单 JSONL/CSV 评估记录即可：每行一题一配置，包含路径、评分、归因、人工备注。

关联模块/文件：

- `loreweaver/storage/sqlite_store.py`
- `loreweaver/logging.py`
- `data/runs/`
- `data/eval/`

### 3.10 M1.9 Eval 模块

当前做法：

- `loreweaver/eval/question_set.py`、`runner.py`、`metrics.py` 仍是占位；
- CLI 中 `eval` 仍是 placeholder；
- 当前评估主要依赖人工查看 retrieval/evidence/answer report。

当前理解：

没有 M1.9，前面的“可调决策”都会退化成凭单题观感调参。M1.9 不需要一开始就做复杂自动评分，但必须把问题集、运行配置、人工评分和产物路径绑定起来。

待测试项：

- 问题集最小规模：20 题、40 题还是 60 题；
- 问题类型配比：character、faction、location、power、timeline、mystery、unknown；
- 每题是否预先写 gold evidence span，还是只做人工主观评分；
- 评分维度：证据召回、证据噪声、回答正确性、引用支持度、不确定性表达；
- 盲测方式：隐藏 variant 名称，只展示 answer/evidence；
- 是否允许同题多答案并排比较；
- 是否自动汇总每个模块归因；
- 是否将 mock runs 和 live runs 分开统计。

当前倾向：

M1.9 第一版采用人工评分 + report 路径索引，不急着做 LLM-as-judge。最小问题集建议 30 题：每个主要 query type 至少 4 题，再补 5-6 个跨类型难题。

关联模块/文件：

- `loreweaver/eval/question_set.py`
- `loreweaver/eval/runner.py`
- `loreweaver/eval/metrics.py`
- `loreweaver/cli.py`
- `data/eval/`

---

## 4. M1.9 优先实验矩阵

M1.9 不应一次测试所有旋钮。建议先挑 6 组高收益实验，每组只改 1-2 个变量。

| 优先级 | 实验 | Baseline | Variant | 主要观察 |
| --- | --- | --- | --- | --- |
| P0 | 窗口模式 | chapter window | sliding `1200/0.2` | 抽取成本、Span 粒度、locator 成功率、最终证据完整性 |
| P0 | Embedding 输入 | summary fields only | + `key_quote` | vector 命中率、细节证据召回、噪声 |
| P0 | Evidence 上下文 | `300/500/500/12` | `150/250/200/8` | 证据块可读性、引用精度、回答遗漏 |
| P1 | Cluster 数量 | `cluster_count=4` | `cluster_count=8` | 图召回覆盖率、错误 cluster 下钻 |
| P1 | Cluster 检索文本 | 规则 summary | LLM `cluster_retrieval_summary` | 抽象问题召回、mystery/power 命中 |
| P1 | Reranker | noop fused score | live Qwen3 reranker | Top-K 相关性、候选池压噪能力、耗时 |
| P2 | Query Router | 单标签关键词 | 多标签轻量路由 | 跨类型问题覆盖、图偏置是否改善 |
| P2 | QA Prompt | 固定四段式 | query_type 模板 | 回答结构、证据不足表达、引用支持度 |

每个实验至少记录：

```text
experiment_id
variant_id
git_commit_or_worktree_note
config_snapshot
question_set_path
retrieval_report_path
evidence_report_path
answer_report_path
manual_scores
failure_attribution
decision
```

建议人工评分维度：

```text
retrieval_core_evidence: 0-3
retrieval_noise: 0-3
evidence_pack_readability: 0-3
answer_correctness: 0-3
citation_support: 0-3
uncertainty_calibration: 0-3
overall_preference: baseline / variant / tie
failure_owner: router / graph / vector / bm25 / union / reranker / evidence / qa / source
```

---

## 5. 使用方式

开发时如果发现某个策略“现在先这么做，但后面可能要测”，应追加到本文档，而不是只留在对话里。

推荐每条记录至少包含：

```text
当前做法
为什么这样做
可能的问题
待测试变体
当前倾向
关联模块/文件
```

M1.9 开始前，应从本文档中挑选优先级最高的实验项，形成正式的 `eval experiments` 清单。
