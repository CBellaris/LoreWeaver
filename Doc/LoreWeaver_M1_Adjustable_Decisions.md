# LoreWeaver M1 可调决策与实验清单

版本：v0.1  
创建日期：2026-04-24  
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

## 2. 当前已整理范围

本版本先整理 **M1.6 召回模块**：

```text
用户问题
  -> Query Router
  -> 图召回 / 向量召回 / BM25 召回
  -> Union 融合
  -> Reranker 精排
  -> Top-K Span
```

其他模块待补充，包括但不限于：

- M1.1 文本规范化与章节切分；
- M1.2 窗口切分；
- M1.3 LLM Span 抽取与 anchor 定位；
- M1.4 embedding 输入与索引策略；
- M1.5 CenterSpanCluster 构建；
- M1.7 Evidence Pack 组装；
- M1.8 QA 生成与引用约束；
- M1.9 评估集与人工评分体系。

---

## 3. M1.6 召回模块待测试/调整项

### 3.1 Query Router Buckets

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

### 3.2 图召回的 Cluster 元数据字段

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
- 加入中心 Span 的 `micro_topic / micro_summary / key_quote`；
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

### 3.3 图召回打分方式

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

### 3.4 图召回下钻策略

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

### 3.5 向量召回策略

当前 Span embedding 输入：

```text
micro_topic
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

### 3.6 BM25 召回策略

当前 BM25：

- 中文字、bigram、trigram token；
- 文档文本由 `micro_topic + micro_summary + entities + topics + key_quote` 拼接。

当前理解：

足以支撑 M1 实体召回，但不是最终形态。BM25 应定位为专名、术语、别称、精确表述的保险丝。

待测试项：

- 引入实体词典，把抽取出的 entities/topics 作为强 token；
- 字段权重：entities > micro_topic > topics > summary > key_quote；
- query 侧同义词/别名扩展；
- 人名、地名、势力名精确命中 boost；
- BM25 top_k 增大后，Reranker 是否能压住噪声；
- 更强中文分词器与当前轻量 tokenizer 的对照。

当前倾向：

优先测试实体词典和字段权重，而不是立刻替换分词器。

### 3.7 Union 融合策略

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

### 3.8 Reranker 策略

当前 Reranker：

- 默认配置主选型为 `Qwen/Qwen3-Reranker-0.6B` via SiliconFlow；
- fallback 记录为 `BAAI/bge-reranker-v2-m3`；
- 支持 mock/noop；
- 默认配置仍关闭 live reranker，避免本地误触付费 API。

当前输入格式：

```text
章节：<chapter_title>
小话题：<micro_topic>
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

### 3.9 召回可观测性与评估

当前已经保存：

- retrieval report；
- 每路召回数量；
- Union 候选；
- Rerank Top-K；
- 每个 Top-K 的来源；
- query_runs。

待补充观测项：

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

---

## 4. 其他模块待补充

以下模块也有大量类似“可调决策”，后续应逐步补齐：

### M1.1 文本规范化与章节切分

待补充。

可能方向：

- 广告/噪声清洗强度；
- 章节标题规则；
- fallback chapter 切分长度；
- 是否保留原始换行与排版符号。

### M1.2 候选窗口切分

待补充。

可能方向：

- window size；
- overlap ratio；
- 按章节整窗 vs 滑窗；
- 是否按自然段落边界调整窗口；
- 未覆盖文本的人工抽查策略。

### M1.3 Span 抽取与定位

待补充。

可能方向：

- Span 类型体系；
- 每窗最小/最大 Span 数；
- salience_score 解释与使用；
- anchor 长度；
- fuzzy threshold；
- 抽取 prompt 的保守/激进程度；
- `micro_summary` 是否应更偏“事实”还是“解释”。

### M1.4 索引与 Embedding

待补充。

可能方向：

- embedding 模型；
- embedding 输入字段；
- 是否多向量；
- Qdrant payload 字段；
- BM25 文档构造与字段权重。

### M1.5 CenterSpanCluster 构建

待补充。

可能方向：

- 人工中心 Span 选择；
- 自动 medoid 选择；
- LLM cluster summary；
- member_span_ids 的确认策略；
- `SUPPORTS / RELATED_TO` 边权；
- Cluster 类型体系。

### M1.7 Evidence Pack

待补充。

可能方向：

- 上下文扩展长度；
- merge gap；
- max blocks；
- 多来源证据优先级；
- 时间线排序 vs rerank 排序；
- 引用编号稳定性。

### M1.8 QA Answerer

待补充。

可能方向：

- 回答 prompt；
- 引用约束；
- 证据不足时的拒答策略；
- 是否要求逐条证据归纳；
- 是否按问题类型输出不同结构。

### M1.9 评估体系

待补充。

可能方向：

- 评估问题集结构；
- 人工评分表；
- 盲测流程；
- baseline/variant 对照；
- 失败归因标签；
- M1 完成阈值。

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
