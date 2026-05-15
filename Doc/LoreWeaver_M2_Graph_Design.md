# LoreWeaver M2 Graph 设计文档

状态：草案 v0.2
更新时间：2026-05-15
适用阶段：M2（图层重构）
依赖前置：M1 已交付（参见 [LoreWeaver_M1_Current_Delivery.md](LoreWeaver_M1_Current_Delivery.md)）

本文档定义 LoreWeaver M2 阶段图层的设计哲学、存储边界、节点与边谱系、应用层抽象、迁移机制和实施路线。M1 阶段的 `loreweaver/graph/center_span.py` 与 `loreweaver/graph/edge_builder.py` 在 M2.0 开始时直接删除，旧图数据可清空重建，不保留兼容路径。

---

## 1. 设计哲学

### 1.1 根原则

1. **原文是唯一事实地基**：所有抽象层（Span、Cluster、Edge、Hypothesis）必须能回溯到 normalized text 的字符坐标区间。
2. **Span 是事实层与图层的边界**：Span 的"产生与权威定义"在 SQLite；Span 的"导航与组合"在 Neo4j。Neo4j 不复制 Span 原文，只持有 span_id 指针和导航必需的元数据。
3. **Cluster 不是真相，是视图（view）**：同一个 Span 可以同时属于多个 cluster；不同 cluster 可以重叠、冲突、矛盾。Cluster 是导航工具，不是分类学。
4. **导航视角由结构表达**：早期不引入全局 `layer` / `view_family` / `source` 属性；用节点 label、边 type 和 GraphExplorer 的边组配置表达导航方式。
5. **中心节点优先于两两连边**：表达 N 元关系一律用 hub-and-spoke 模式（中心节点 + spoke 边），不用 $O(N^2)$ 直连。
6. **推理边必须可溯源**：非 Span-origin 的派生/推理边必须携带 `evidence_span_ids` 或等价 trace；Span-origin 的 spoke 边以起点 Span 本身作为证据。
7. **schema 在应用层治理**：数据库只提供必要唯一约束和索引；业务一致性靠 NodeSpec/EdgeSpec 抽象层和迁移脚本保证。

### 1.2 三类图内容

| 类别 | 名称 | 节点/边来源 | 物化策略 | 失效条件 |
|---|---|---|---|---|
| structure | naive 结构边 | 文本结构（章节、窗口、相邻） | 全量物化 | Span 表变更 |
| navigation | 程序化导航中心 | 实体、span_type、向量聚类 | 全量物化（中心节点 + spoke） | Span 表变更 或 聚类规则版本变更 |
| exploration | Agent 探索 | 问题、假设、答案、用户交互痕迹 | 全量物化 | 显式 supersede / archive，永不自动删除 |

### 1.3 Agent 走图的目标形态

Agent 不直接遍历 Span 列表。它的入口永远是某种"中心节点"——某个 Entity、某个 Chapter、某个已有的 Question 节点——然后通过 spoke 边走到 Span，再从 Span 走到其他中心节点，形成跨层级跳转。探索写回让图逐渐积累"可复用的思考痕迹"。

---

## 2. 存储边界

### 2.1 SQLite（事实层，权威）

保留：

- `documents`、`chapters`、`candidate_windows`
- `spans`、`locator_candidates`、`extraction_failures`
- `embedding_cache`
- `ingest_reports` / `window_reports` / `extraction_reports` / `index_reports`
- `query_runs` / `evidence_packs`
- 新增 `graph_reports`（M2 各阶段构建报告，与现存 `graph_reports` 表复用但 schema 调整）
- 新增 `graph_migrations`（迁移记录，见 §7）

清理（M2.0 必须删除）：

- `center_span_clusters` 表及其全部数据
- `span_edges` 表及其全部数据
- 旧的 `graph_reports` 历史记录（M2.0 迁移脚本清空）

### 2.2 Neo4j（图层，权威）

接管：

- 所有图节点：Span 镜像节点、Chapter/Window 镜像、Entity、EntityType、SpanType、MicroSpanType、VectorCluster、Question、Hypothesis、Answer
- 所有图边：structure 边、navigation spoke 边、向量聚类 spoke 边、agent 探索边
- 所有 cluster 元数据（cluster 本身就是中心节点）

不接管：

- Span 的原文（located_text、anchor 等），仍在 SQLite
- 任何 chat completion 原始返回、batch 输出
- 向量本体（仍在 Qdrant）

### 2.3 边界规则

- **Neo4j → SQLite 的查询只发生在一个接口**：`span_id → (chapter_id, char_start, char_end, located_text)`。这是图层与事实层之间唯一的同步点。
- **写入方向单向**：SQLite 是源，Neo4j 是从 SQLite 派生的。Agent 探索产物虽然只在 Neo4j 里，但它的"证据"指向的是 SQLite 中的 span，不反向写回 SQLite。
- **重建语义**：structure/navigation 内容在 Neo4j 中可全量重建。exploration 内容不能重建（是用户/agent 交互产物），M2 中通过 Neo4j 的 dump/restore 备份。

### 2.4 Neo4j 部署假设

- M2 阶段假设单 Neo4j 实例、单 database
- 当前期内不区分多文档：可只挂一本书；如需多本书共存，节点和边携带 `document_id` property，通过复合索引隔离
- 不引入 `user_id` / `session_id`，待 exploration 多用户场景出现时再加（LPG 加 property 是 O(1)）

---

## 3. 节点谱系

所有节点共有 property：

| property | 类型 | 说明 |
|---|---|---|
| `node_id` | string | 业务唯一键，UUID5 或确定性 hash |
| `document_id` | string | 所属文档；跨文档节点（如全局 EntityType）为 `"_global"` |
| `status` | string | `active` / `deprecated` / `archived` |

### 3.1 镜像类节点（指向 SQLite 实体）

#### `:Span`
导航投影，原文与 anchor 不在此存储。

| property | 必需 | 说明 |
|---|---|---|
| `span_id` | ✓ | SQLite `spans.span_id` |
| `chapter_id` | ✓ | 反范式，加速按章过滤 |
| `window_id` | ✓ | 反范式 |
| `span_type` | ✓ | progression / interaction / ... |
| `salience_score` | ✓ | float |
| `entities` | ✓ | list[string]，反范式用于 ad-hoc 过滤；权威列表通过 spoke 边 |
| `char_start` | ✓ | normalized text 起点（反范式） |
| `char_end` | ✓ | normalized text 终点（反范式） |
| `micro_span_type` | 延后 | 在后期增强阶段引入（见 §10 后期增强），M2 初期不在 Span 节点上 |

Span 节点是图层一切操作的基础节点。

#### `:Chapter`
镜像 SQLite `chapters` 表。

| property | 必需 | 说明 |
|---|---|---|
| `chapter_id` | ✓ | SQLite 主键 |
| `chapter_index` | ✓ | 章节序号 |
| `title` | ✓ | 章节标题 |
| `char_start` / `char_end` | ✓ | 章节在 normalized text 中的坐标 |

#### `:Window`
镜像 `candidate_windows`，主要用于"同窗口"导航。

### 3.2 导航中心节点

#### `:Entity`
| property | 必需 | 说明 |
|---|---|---|
| `entity_name` | ✓ | 规范化后的实体名（唯一键的一部分） |
| `mention_count` | ✓ | 反范式聚合，便于排序 |
| `entity_type` | 延后 | character / location / faction / item / power / event / unknown，后期增强引入 |
| `aliases` | 延后 | list[string]，共指消解结果，后期增强引入 |

`node_id` = `entity::{document_id}::{entity_name}`。

#### `:SpanType`
类型本身作为节点存在，便于挂属性（如 type 的描述、统计量），也便于将来引入"类型间关系"。

| property | 必需 | 说明 |
|---|---|---|
| `type_value` | ✓ | 枚举值字符串 |
| `mention_count` | ✓ | 反范式聚合 |
| `description` | optional | 人工或 LLM 生成的类型说明 |

类型节点的 `document_id` 默认 `"_global"`，不随文档变化。

#### `:EntityType` / `:MicroSpanType`（后期增强阶段引入）
等 extraction 扩展产出 `entity_type` 和 `micro_span_type` 后再引入对应的中心节点和 spoke 边。当前阶段不预创建空节点，避免半成品占位。

#### `:VectorCluster`
embedding 自动聚类的中心。

| property | 必需 | 说明 |
|---|---|---|
| `cluster_index` | ✓ | 聚类编号 |
| `algorithm` | ✓ | hdbscan / leiden / kmeans |
| `params_hash` | ✓ | 聚类参数 hash，参数变即新 cluster |
| `member_count` | ✓ | 成员 Span 数 |
| `centroid_summary` | optional | LLM 生成的 cluster 主题描述 |

### 3.3 Agent 探索节点

#### `:Question`
| property | 必需 | 说明 |
|---|---|---|
| `question_id` | ✓ | UUID4 |
| `question_text` | ✓ | 自然语言问题 |
| `origin` | ✓ | `user` / `agent_self_play` / `eval` |
| `status` | ✓ | `open` / `answered` / `archived` |

#### `:Hypothesis`
| property | 必需 | 说明 |
|---|---|---|
| `hypothesis_id` | ✓ | UUID4 |
| `statement` | ✓ | 命题陈述 |
| `confidence` | ✓ | 0~1 |
| `supersedes` | optional | 旧 hypothesis_id（链式演进） |

#### `:Answer`
| property | 必需 | 说明 |
|---|---|---|
| `answer_id` | ✓ | UUID4 |
| `answer_text` | ✓ | 最终答案文本 |
| `evidence_pack_id` | optional | SQLite `evidence_packs.evidence_pack_id` |

### 3.4 节点 ID 约定

所有 `node_id` 必须确定性生成（除 Agent 层 UUID4 外），格式：

```
{prefix}::{document_id}::{natural_key}
```

例如 `entity::doc_59331b17113e::高文`、`vector_cluster::doc_59331b17113e::hdbscan::abc123::0042`。

确定性 ID 保证 MERGE 幂等、迁移可重跑、跨环境可对比。

---

## 4. 边谱系

所有边共有 property：

| property | 类型 | 说明 |
|---|---|---|
| `edge_id` | string | 确定性 UUID5（from + to + type） |
| `document_id` | string | 边所属文档 |
| `weight` | float | 0~1，可选业务语义 |
| `confidence` | float | 0~1，Agent 推理边必需 |
| `evidence_span_ids` | list[string] | 非 Span-origin 的派生/推理边必需；Span-origin spoke 边可空 |
| `created_at` | datetime | UTC ISO8601 |

### 4.1 Structure 边

| 边 type | from → to | 语义 | 数量级 |
|---|---|---|---|
| `:IN_CHAPTER` | Span → Chapter | 所属章节 | $O(\text{spans})$ |
| `:IN_WINDOW` | Span → Window | 所属窗口 | $O(\text{spans})$ |
| `:CONTAINS_SPAN` | Chapter → Span（IN_CHAPTER 反向） | 仅在需要正向枚举时建；否则用反向 MATCH | $O(\text{spans})$ |
| `:NEXT_CHAPTER` | Chapter → Chapter | 章节顺序 | $O(\text{chapters})$ |
| `:NEXT_WINDOW` | Window → Window | 窗口顺序 | $O(\text{windows})$ |
| `:NEXT_SPAN_IN_TEXT` | Span → Span | 原文内相邻（按 char_start 排序） | $O(\text{spans})$ |

注：是否需要正反两向取决于 Cypher 查询模式。Neo4j 边天然带方向，`MATCH (c:Chapter)<-[:IN_CHAPTER]-(s:Span)` 已经能反向走，因此 `:CONTAINS_SPAN` 默认不建。

### 4.2 Navigation 边

| 边 type | from → to | 语义 | 数量级 |
|---|---|---|---|
| `:MENTIONS_ENTITY` | Span → Entity | 提及该实体 | $O(\text{spans} \cdot \overline{\text{entities/span}})$ |
| `:HAS_SPAN_TYPE` | Span → SpanType | span_type spoke | $O(\text{spans})$ |
| `:IN_VECTOR_CLUSTER` | Span → VectorCluster | 向量聚类成员 | $O(\text{spans})$ |

后期增强引入：

| 边 type | from → to | 引入阶段 |
|---|---|---|
| `:HAS_MICRO_TYPE` | Span → MicroSpanType | 后期增强 |
| `:OF_ENTITY_TYPE` | Entity → EntityType | 后期增强 |
| `:SUBTYPE_OF` | MicroSpanType → SpanType | 后期增强 |

所有 navigation 边的查询模式都是"Span → 中心节点 → Span"两跳，永远不在 Span 之间两两连。

### 4.3 Agent 边

| 边 type | from → to | 语义 |
|---|---|---|
| `:ASKS_ABOUT` | Question → Entity / Chapter / VectorCluster / Span | 问题指向 |
| `:DERIVED_FROM` | Hypothesis → Question | 假设产生自问题 |
| `:SUPPORTED_BY` | Hypothesis → Span | 假设的原文证据 |
| `:CONTRADICTS` | Hypothesis → Hypothesis | 冲突关系 |
| `:SUPERSEDES` | Hypothesis → Hypothesis | 演进关系 |
| `:ANSWERS` | Answer → Question | 回答指向 |
| `:CITES` | Answer → Span | 答案引用 |
| `:USES_HYPOTHESIS` | Answer → Hypothesis | 答案采纳的假设 |

### 4.4 边 ID 约定

```python
edge_id = "edge_" + uuid5(NAMESPACE_URL, f"{document_id}::{from_id}::{to_id}::{edge_type}")[:20]
```

若同一对节点需要表达不同语义，应使用不同 edge type；禁止用同 type 的重复边承载不同业务含义。

---

## 5. NodeSpec / EdgeSpec 抽象层

### 5.1 目标

把"节点/边长什么样"集中在一个文件，让数据库写入、Cypher 生成、校验、迁移、文档同步全部以 spec 为单一事实源。

### 5.2 接口骨架

```python
# loreweaver/graph/spec.py

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

@dataclass(frozen=True)
class PropertySpec:
    name: str
    py_type: type
    required: bool = True
    default: Any = None
    description: str = ""

@dataclass(frozen=True)
class NodeSpec:
    label: ClassVar[str]                  # Neo4j label
    version: ClassVar[int] = 1
    properties: ClassVar[tuple[PropertySpec, ...]] = ()
    id_field: ClassVar[str] = "node_id"   # property used as MERGE key

    @classmethod
    def to_merge_cypher(cls) -> str: ...
    @classmethod
    def validate(cls, payload: Mapping[str, Any]) -> None: ...

@dataclass(frozen=True)
class EdgeSpec:
    edge_type: ClassVar[str]
    from_label: ClassVar[str]
    to_label: ClassVar[str]
    version: ClassVar[int] = 1
    properties: ClassVar[tuple[PropertySpec, ...]] = ()
    requires_evidence: ClassVar[bool] = False

    @classmethod
    def to_merge_cypher(cls) -> str: ...
    @classmethod
    def validate(cls, payload: Mapping[str, Any]) -> None: ...

# 注册表
NODE_SPECS: dict[str, type[NodeSpec]] = {}
EDGE_SPECS: dict[str, type[EdgeSpec]] = {}

def register_node(cls: type[NodeSpec]) -> type[NodeSpec]: ...
def register_edge(cls: type[EdgeSpec]) -> type[EdgeSpec]: ...
```

### 5.3 使用示例

```python
@register_node
class SpanNode(NodeSpec):
    label = "Span"
    version = 1
    id_field = "span_id"
    properties = (
        PropertySpec("span_id", str, required=True),
        PropertySpec("document_id", str, required=True),
        PropertySpec("chapter_id", str, required=True),
        PropertySpec("window_id", str, required=True),
        PropertySpec("span_type", str, required=True),
        PropertySpec("salience_score", float, required=True),
        PropertySpec("entities", list, required=True),
        PropertySpec("char_start", int, required=True),
        PropertySpec("char_end", int, required=True),
    )

@register_edge
class MentionsEntityEdge(EdgeSpec):
    edge_type = "MENTIONS_ENTITY"
    from_label = "Span"
    to_label = "Entity"
    version = 1
    properties = (
        PropertySpec("weight", float, required=False, default=1.0),
    )
    requires_evidence = False  # entity 边的 evidence 就是 Span 本身
```

### 5.4 演进规则

- 加 property：直接在 spec 加一行，bump `version`，写一段迁移脚本回填默认值
- 删 property：先在 spec 标 `deprecated=True`（PropertySpec 加该字段），停止写入；下一个 minor 版本删除字段并跑迁移
- 加新节点/边类型：注册新 spec，无需迁移
- 改 ID 计算规则：禁止。要换语义就新增一个节点类型

---

## 6. GraphOps API（业务原语）

应用代码（retrieval、qa、web、cli）**永远不直接写 Cypher**，全部通过 GraphOps。

### 6.1 写入侧

```python
class GraphWriter:
    # Structure
    def upsert_span_node(self, span: Span) -> None
    def upsert_chapter_node(self, chapter: Chapter) -> None
    def upsert_window_node(self, window: Window) -> None
    def link_span_in_chapter(self, span_id: str, chapter_id: str) -> None
    def link_next_span(self, prev_span_id: str, next_span_id: str) -> None

    # Navigation
    def upsert_entity(self, name: str, *, entity_type: str | None = None) -> str
    def link_span_mentions_entity(self, span_id: str, entity_id: str) -> None
    def upsert_type_node(self, kind: str, value: str) -> str
    def link_span_has_type(self, span_id: str, type_node_id: str, kind: str) -> None
    def upsert_vector_cluster(self, cluster_index: int, params_hash: str, ...) -> str
    def link_span_in_vector_cluster(self, span_id: str, cluster_id: str) -> None

    # Exploration
    def record_question(self, text: str, origin: str) -> str
    def record_hypothesis(self, statement: str, confidence: float,
                          question_id: str, evidence_span_ids: list[str]) -> str
    def record_answer(self, question_id: str, answer_text: str,
                      cited_span_ids: list[str], used_hypotheses: list[str]) -> str
```

### 6.2 读取侧

```python
class GraphReader:
    def get_span(self, span_id: str) -> dict
    def neighbors(self, node_id: str, *, edge_types: list[str] | None = None,
                  limit: int = 50) -> list[dict]
    def spans_of_entity(self, entity_id: str, limit: int = 100) -> list[str]
    def spans_in_chapter(self, chapter_id: str) -> list[str]
    def spans_in_vector_cluster(self, cluster_id: str) -> list[str]
    def walk(self, start: str, pattern: list[str], max_hops: int = 3) -> list[list[dict]]
    # Exploration
    def find_related_questions(self, text: str, top_k: int = 5) -> list[dict]
    def hypotheses_for_question(self, question_id: str) -> list[dict]
```

### 6.3 设计约束

- 所有方法返回 `dict` 或 `list[dict]`，不暴露 Neo4j driver 对象。这样上层不绑定 Neo4j。
- 所有写入幂等（基于 MERGE）。重跑 graph seed 不产生重复节点。
- 大批量写入走 `UNWIND` 批处理，单次事务 500~1000 条。

---

## 7. 迁移机制

### 7.1 迁移记录

Neo4j 中维护单一节点：

```cypher
MERGE (m:SchemaMigration {migration_id: $id})
SET m.applied_at = $now, m.description = $desc
```

SQLite 中维护 `graph_migrations` 表镜像，便于离线查询：

```sql
CREATE TABLE graph_migrations (
    migration_id TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT NOT NULL,
    cypher_hash TEXT NOT NULL
);
```

### 7.2 迁移脚本目录

```
loreweaver/graph/migrations/
    0001_initial_schema.py
    0002_add_micro_span_type.py
    0003_add_vector_cluster.py
    ...
```

每个文件结构：

```python
MIGRATION_ID = "0002_add_micro_span_type"
DESCRIPTION = "Backfill MicroSpanType nodes and HAS_MICRO_TYPE edges"

def up(driver, sqlite_store): ...
def down(driver, sqlite_store): ...  # 可选，破坏性迁移可省略
```

CLI：

```bash
python -m loreweaver.cli graph migrate          # 跑所有未应用
python -m loreweaver.cli graph migrate --target 0003
python -m loreweaver.cli graph migrate --status # 列出已应用/未应用
```

### 7.3 destructive 迁移规则

涉及删 property、删节点类型、改 ID 算法的迁移：

1. 先写 `--dry-run` 模式，输出影响 N 个节点 / M 条边
2. 必须有 Neo4j 全量 dump 备份证据
3. 在 PR 描述中显式标记 `[BREAKING]`

---

## 8. 关键流程

### 8.1 `graph seed`（M2.1 新增 CLI 命令）

从 SQLite 全量构建 Neo4j structure + navigation 静态部分（不含向量聚类）：

```
1. 读取 documents / chapters / windows / spans
2. 写入 Span / Chapter / Window 镜像节点（UNWIND 批量 MERGE）
3. 写入 :IN_CHAPTER / :IN_WINDOW / :NEXT_CHAPTER / :NEXT_WINDOW / :NEXT_SPAN_IN_TEXT
4. 聚合 entities → 写入 :Entity 节点 + :MENTIONS_ENTITY 边
5. 聚合 span_type → 写入类型中心节点 + spoke 边（micro_span_type 留待 M2.5 引入）
6. 输出 seed 报告（节点数、边数、耗时、错误清单）
```

幂等：重跑只会 MERGE，不会重复创建。

### 8.2 `graph vector-cluster`（M2.2 新增）

```
1. 从 Qdrant 拉取该文档全部 span 向量
2. 运行聚类算法（默认 HDBSCAN）
3. 计算 params_hash → 决定是否新建 VectorCluster 节点
4. MERGE VectorCluster 节点 + :IN_VECTOR_CLUSTER 边，边上记录 `membership_score`
5. 可选：调用 LLM 生成 centroid_summary 写回节点
```

规则：

- 允许 Span 不属于任何 VectorCluster（例如 HDBSCAN noise）。
- 允许同一 Span 在多套 `params_hash` 下属于多个 VectorCluster。
- `params_hash` 不同即视为不同聚类视图，旧聚类不覆盖新聚类。
- 大簇必须记录代表性成员（如 `representative_span_ids` 或按 `membership_score` 查询 top members），GraphExplorer 不无界展开大簇。

### 8.3 GraphExplorer：图原生 QA 流程

一旦 structure + navigation 静态部分就绪（M2.1 退出标准达成），图层就具备独立承载召回与问答的条件。M1 的"retrieve → evidence → qa"三段式不再合适，因为：

- 入口不是一次性向量召回，而是图节点定位（entity / chapter / question / vector_cluster）
- 主循环是"走图 → 评估覆盖 → 决定下一跳"的迭代过程，可能多轮
- evidence 组装在每一跳之间动态发生，不是末端单次操作
- vector / BM25 不再是并行召回路径，而是图遍历过程中"按需调用的工具"

因此 M2 引入一个独立模块 **`loreweaver/explorer/`**（名字待定，候选还有 `graphqa` / `walker` / `inquirer`；本文档统一称为 GraphExplorer），承担从问题到答案的完整闭环。这是与 M1 `retrieval/` + `evidence/` + `qa/` **平行**存在的新流程，不通过 Union 与旧流程混合。

#### 8.3.1 流程骨架

```
ask(question) :
  ┌─────────────────────────────────────────────┐
  │ 1. 入口定位（Seed）                         │
  │    - 全文检索 Question 节点：复用历史回答?  │
  │    - 抽取 query 中的实体/章节/类型关键词    │
  │    - MERGE seeds: list[node_id]             │
  └─────────────────────────────────────────────┘
                       │
  ┌─────────────────────────────────────────────┐
  │ 2. 走图循环（最多 N 轮）                    │
  │   ├ Step A: 邻域扩展                        │
  │   │   GraphReader.walk(seeds, patterns)    │
  │   │   产出候选 Span 集合                    │
  │   ├ Step B: 工具调用（按需）                │
  │   │   - vector_search(query, scope=spans)   │
  │   │   - bm25_search(rare_token)             │
  │   │   两者结果作为新 seeds 注入             │
  │   ├ Step C: 证据片段挑选                    │
  │   │   选最近相关 span 加入 working_evidence │
  │   ├ Step D: 覆盖评估                        │
  │   │   足够 / 需要换方向 / 终止              │
  │   └ 继续 or 终止                            │
  └─────────────────────────────────────────────┘
                       │
  ┌─────────────────────────────────────────────┐
  │ 3. 答案合成                                 │
  │    用 working_evidence 生成带 citation 答案 │
  │    引用校验（按 M1.8 原则重写）             │
  └─────────────────────────────────────────────┘
                       │
  ┌─────────────────────────────────────────────┐
  │ 4. 痕迹写回 exploration                     │
  │    Question / Answer / (Hypothesis) 节点    │
  │    CITES / ANSWERS / SUPPORTED_BY 边        │
  └─────────────────────────────────────────────┘
```

#### 8.3.2 工具内化原则

vector / BM25 不再作为独立召回器与图召回并列，而是 GraphExplorer 在走图过程中调用的"扩展工具"：

| 工具 | 调用时机 | 作用域 |
|---|---|---|
| `graph.walk` | 主循环每一轮 | 从已有 seeds 沿 edge 扩散 |
| `vector_search` | 邻域扩展不足 / 入口定位失败 | 全文档 spans 或限定子集 |
| `bm25_search` | 出现疑似专有名词、自造词、生僻词，图中无对应 Entity 节点 | 全文档 spans |
| `fetch_span_text` | 选定证据后从 SQLite 取原文 | 单 span |

这样三种"召回方法"在 GraphExplorer 内部被组合调用，外层 API 只暴露 `ask(question)`。

#### 8.3.3 模块结构（候选）

```
loreweaver/explorer/
    __init__.py
    pipeline.py            # ask() 主入口
    seeding.py             # 入口节点定位
    walker.py              # 走图循环
    tools/
        vector.py          # 向量工具
        bm25.py            # BM25 工具
        graph.py           # GraphReader 包装
        span_text.py       # 原文取回
    evidence_buffer.py     # 动态 evidence 累积
    synthesizer.py         # 答案合成 + 引用校验
    trace.py               # exploration 写回
```

#### 8.3.4 与旧 retrieve/evidence/qa 的关系

- **并存阶段**：GraphExplorer 引入后，旧 `loreweaver/retrieval/`、`loreweaver/evidence/`、`loreweaver/qa/` 仅保留为对照基线，CLI 通过 `--explorer` / `--legacy` 切换
- **评估阶段**：用 M1.9 eval 框架对比两套流程在同一 question set 上的指标
- **替换阶段**：当 GraphExplorer 在 weighted recall / NDCG / 答案引用准确率上**至少持平**旧流程，且在至少一个 question profile 上明显胜出后，删除旧三组件
- **不做 Union 融合**：两套流程逻辑结构差异太大（一次性 vs 迭代式），强行合并会让评估失去对比意义
- **不交叉引用旧模块**：GraphExplorer 内部重写走图、证据缓冲、引用校验和答案合成能力，不从旧 retrieval/evidence/qa import 业务逻辑。

#### 8.3.5 Exploration 写回（GraphExplorer 内化）

GraphExplorer 流程末端自动写入 Question / Answer 节点：

```python
question_id = graph_writer.record_question(query, origin="user")
answer_id = graph_writer.record_answer(
    question_id=question_id,
    answer_text=answer.text,
    cited_span_ids=answer.cited_spans,
    used_hypotheses=[],
)
```

Hypothesis 节点的引入暂缓到 M2.4，初版 GraphExplorer 先把 Question / Answer 落进图，并以"历史 Question 节点全文检索"作为入口定位的第一步。

---

## 9. 现有代码影响清单

### 9.1 M2.0 立即删除

- `loreweaver/graph/center_span.py`：整个文件
- `loreweaver/graph/edge_builder.py`：整个文件
- `loreweaver/models/cluster.py`：`CenterSpanCluster` 与 `SpanEdge` 类
- `loreweaver/storage/sqlite_store.py`：`replace_graph`、`list_graph_*`、`insert_graph_report` 中与 cluster/edge 表相关的部分；`center_span_clusters` / `span_edges` 表 DDL

### 9.2 GraphExplorer 替换后删除（M2.3 之后，eval 通过为前提）

- `loreweaver/retrieval/`：整目录（包括 `pipeline.py`、`graph_retriever.py`、`vector_retriever.py`、`bm25_retriever.py`、`union.py`、`reranker.py`）
- `loreweaver/evidence/`：整目录
- `loreweaver/qa/`：整目录
- `loreweaver/cli.py` 中 `retrieve` / `evidence` / `ask` 旧命令实现
- `configs/default.yaml` 中 `retrieval` / `evidence` / `qa` 段

保留：`evidence_packs` SQLite 表与 `query_runs` 表，作为 GraphExplorer 的运行记录沉淀目标（schema 复用，字段可能调整）。

### 9.3 必须新增

- `loreweaver/graph/spec.py`：NodeSpec / EdgeSpec 框架
- `loreweaver/graph/specs/`：所有节点/边类型定义
- `loreweaver/graph/ops/writer.py`：GraphWriter
- `loreweaver/graph/ops/reader.py`：GraphReader
- `loreweaver/graph/seed.py`：seed pipeline
- `loreweaver/graph/vector_cluster.py`：M2.2 引入
- `loreweaver/graph/migrations/`：迁移目录
- `loreweaver/storage/neo4j_driver.py`：升级现有 `neo4j_store.py`，提供低层 driver 包装
- `loreweaver/explorer/`：GraphExplorer 模块（M2.3 引入，结构见 §8.3.3）

### 9.4 必须调整

- `loreweaver/cli.py`：
  - 移除 `graph --cluster-count` 等旧参数
  - 新增 `graph seed`、`graph vector-cluster`、`graph migrate`、`graph stats` 子命令
  - M2.3 起新增 `explore` 子命令（或为 `ask` 增加 `--explorer` 开关并存）
- `loreweaver/extraction/schemas.py`：M2.0~M2.4 不动；待 M2.5 后期增强阶段引入 `micro_span_type` 与 `entity_type` 字段
- `loreweaver/web/`：图查看页面切到 Neo4j 数据源；问答页面 M2.3 起切换到 GraphExplorer
- `configs/default.yaml`：移除 `graph.cluster_count` / `members_per_cluster`，加 `graph.vector_cluster.*` 与 `explorer.*` 参数

### 9.5 必须保留兼容

- M1.9 eval 框架：不动（GraphExplorer 复用同一 question set 与指标）
- ingest / windows / extract / index：不动
- 旧 retrieve / evidence / qa：M2.3 引入 GraphExplorer 后**并存**，作为 eval baseline；通过验证后再按 §9.2 删除

---

## 10. M2 路线图

### M2.0 — Schema 清理与基础设施（最小落地）

目标：图层基础设施 ready，旧 cluster 代码移除。**无任何新功能**，但端到端仍可跑通（图层暂时为空）。

- 删除 `center_span.py` / `edge_builder.py` / 相关 SQLite 表
- 实现 `spec.py`、`graph/specs/`、`GraphWriter`、`GraphReader`、`Neo4jDriver` 包装
- 实现 `0001_initial_schema` 迁移：建立 Neo4j 索引和约束（仅覆盖 M2.0~M2.3 涉及的节点/边类型）
- CLI 新增 `graph migrate` / `graph stats`
- **不动 extraction schema**（沿用 M1 现有字段）

退出标准：`graph migrate` 可跑通，`graph stats` 返回空图统计；旧测试通过（除被废弃的 graph 测试）。

### M2.1 — Structure + Navigation 静态部分

目标：图层有 Span/Chapter/Window 镜像、Entity/Type 中心节点和 spoke 边。

- 实现 `graph seed` 命令
- 实现 structure 边构造
- 实现 navigation spoke 边构造（entity / span_type）
- 实现新的 `GraphRetriever`，先支持 entity / chapter 入口
- 旧 retrieval Union 中替换原 `graph_retriever.py` 为基于 Neo4j 的新实现（保持 Union 接口不变，便于 eval 对照）
- 用 M1.9 eval 框架对比 v1 vs v2，要求至少持平

退出标准：eval 不退步；`graph seed` 幂等可重跑；web UI 能显示 Neo4j 中的实体邻域。

### M2.2 — Vector Cluster

目标：把向量聚类作为中心节点接入图。

- 引入 HDBSCAN 依赖
- 实现 `graph vector-cluster` 命令
- VectorCluster 节点带 LLM 生成的 centroid_summary
- 旧 `GraphRetriever` 接入 vector cluster 入口（仍走 Union）
- eval 对比 vector-only vs vector + graph cluster

退出标准：eval 在至少一个 question profile 上优于 M2.1，且图结构具备承载迭代走图的能力。

### M2.3 — GraphExplorer 初版（图原生 QA 流程）

目标：上线 §8.3 描述的 GraphExplorer 流程，与旧 retrieve/evidence/qa 并存。

- 新建 `loreweaver/explorer/` 模块（结构见 §8.3.3）
- 实现 seeding（基于全文索引 Question 节点 + 实体/章节关键词抽取）
- 实现 walker 主循环，初版用规则化策略，最多 N 轮
- 内化 vector / BM25 / fetch_span_text 三件工具
- 动态 evidence_buffer 与 synthesizer（按 M1.8 引用校验原则重写）
- exploration 写回：Question / Answer 节点 + CITES / ANSWERS 边
- CLI 新增 `explore` 命令（或 `ask --explorer`），并存旧 `ask`
- 用 M1.9 eval 框架做 GraphExplorer vs legacy 对比报告

退出标准：GraphExplorer 在 weighted recall / NDCG 上至少持平旧流程，引用准确率不低于旧流程。

### M2.4 — GraphExplorer 进阶 + Hypothesis

目标：让 GraphExplorer 具备假设级推理，并替换旧流程。

- 引入 Hypothesis 节点与 supersede/contradicts 边
- walker 中增加 LLM 自主决策步（决定下一跳方向、是否调用工具）
- 答案合成时把 Hypothesis 作为中间结构
- 冲突检测：同问题多次回答出现矛盾时标 `:CONTRADICTS`
- eval 通过后**删除旧 retrieve/evidence/qa**（按 §9.2 清单）

退出标准：GraphExplorer 在至少一个 question profile 上明显胜出旧流程；定义并通过一次"长时间跨章节推理"评估用例；旧三组件成功移除。

### M2.5（后期增强，时间点暂不固定）— 类型细化

目标：扩展 extraction 让 LLM 同步产出 `micro_span_type` 与 `entity_type`，并将对应的中心节点和 spoke 边纳入图。

- `loreweaver/extraction/schemas.py` 在 `SpanCandidatePayload` 增加 `micro_span_type` 字段，并对 `entities` 做 type 标注（结构待定，可能是并行字段 `entity_types: dict[str, str]`）
- 设计 micro_span_type 的枚举集与 prompt 约束
- 决定是否重跑历史 window 还是只对新 window 生效（兼容策略）
- 引入 `:MicroSpanType` / `:EntityType` 节点与 `:HAS_MICRO_TYPE` / `:OF_ENTITY_TYPE` / `:SUBTYPE_OF` 边
- GraphRetriever 增加 micro_type / entity_type 入口
- eval 验证类型细化对检索质量的增量

这一阶段需要重新审视抽取成本、prompt 长度、retry 成功率，因此从 M2.4 后开始独立评估，不绑死时间点。

---

## 11. 未决问题与延后决策

1. **共指消解**：Entity 名规范化仅做 trim/lowercase，是否引入 alias 表或 LLM 共指消解？延后到 M2.2 后评估。
2. **Hypothesis 的 confidence 校准**：M2.4 待定。
3. **多用户/多 session 字段**：M2 不引入。exploration 节点默认全局可见，未来按需加 `visibility` / `owner_id`。
4. **Neo4j 备份策略**：M2.3 前确定。倾向用 `neo4j-admin database dump` + 周期归档。
5. **Neo4j 版本与社区/企业**：M2 全程社区版即可（单 database 够用）。
6. **类型细化时机**：`micro_span_type` 与 `entity_type` 的引入归到 M2.5 后期增强，依赖一次 extraction prompt 改动 + 重跑评估，不在 M2.0~M2.4 主路径中。
7. **GraphExplorer 模块名**：`explorer` / `graphqa` / `walker` / `inquirer` 待定，M2.3 启动前敲定。
8. **走图主循环的决策机制**：M2.3 初版用规则（覆盖度阈值、最大轮数），M2.4 升级为 LLM 决策。决策日志（trace）是否独立持久化、是否入图，M2.3 设计时确定。
9. **工具调用预算控制**：vector_search / bm25_search / LLM 决策步在单次 ask 中的调用上限，M2.3 启动前在 `configs/default.yaml` `explorer.*` 段配置。
10. **GraphExplorer 与 eval 接口**：M1.9 eval 当前面向 retrieval（章节级 gold label），GraphExplorer 是端到端答案；是否在 eval 框架中新增"答案级"指标，M2.3 设计时确定。

---

## 12. 与 M1 文档的关系

本文档定义 M2 的方向。M1 实现细节仍以 [LoreWeaver_M1_Current_Delivery.md](LoreWeaver_M1_Current_Delivery.md) 为准。M2.0 完成后，M1 文档中第 11 节"M1.5 Graph"段落标记为"已废弃"，并指向本文档。

---

## 附录 A：Cypher 索引与约束（M2.0 迁移内容）

```cypher
// 唯一约束（同时充当索引）
CREATE CONSTRAINT span_id_unique IF NOT EXISTS
  FOR (n:Span) REQUIRE n.span_id IS UNIQUE;
CREATE CONSTRAINT chapter_id_unique IF NOT EXISTS
  FOR (n:Chapter) REQUIRE n.chapter_id IS UNIQUE;
CREATE CONSTRAINT window_id_unique IF NOT EXISTS
  FOR (n:Window) REQUIRE n.window_id IS UNIQUE;
CREATE CONSTRAINT entity_node_id_unique IF NOT EXISTS
  FOR (n:Entity) REQUIRE n.node_id IS UNIQUE;
CREATE CONSTRAINT span_type_node_id_unique IF NOT EXISTS
  FOR (n:SpanType) REQUIRE n.node_id IS UNIQUE;
// :MicroSpanType / :EntityType 的约束随 M2.5 引入
CREATE CONSTRAINT vector_cluster_id_unique IF NOT EXISTS
  FOR (n:VectorCluster) REQUIRE n.node_id IS UNIQUE;
CREATE CONSTRAINT question_id_unique IF NOT EXISTS
  FOR (n:Question) REQUIRE n.question_id IS UNIQUE;
CREATE CONSTRAINT hypothesis_id_unique IF NOT EXISTS
  FOR (n:Hypothesis) REQUIRE n.hypothesis_id IS UNIQUE;
CREATE CONSTRAINT answer_id_unique IF NOT EXISTS
  FOR (n:Answer) REQUIRE n.answer_id IS UNIQUE;

// 复合索引（用于按文档过滤）
CREATE INDEX span_doc_chapter IF NOT EXISTS
  FOR (n:Span) ON (n.document_id, n.chapter_id);
CREATE INDEX entity_doc_name IF NOT EXISTS
  FOR (n:Entity) ON (n.document_id, n.entity_name);
CREATE INDEX vector_cluster_doc IF NOT EXISTS
  FOR (n:VectorCluster) ON (n.document_id, n.params_hash);

// 全文索引（用于 question 匹配）
CREATE FULLTEXT INDEX question_text_fulltext IF NOT EXISTS
  FOR (n:Question) ON EACH [n.question_text];
```

---

## 附录 B：Span 节点写入示例

```python
# 通过 GraphWriter
writer.upsert_span_node(span_record)

# 底层等价 Cypher（由 NodeSpec 生成）
"""
MERGE (n:Span {span_id: $span_id})
SET n.document_id = $document_id,
    n.chapter_id = $chapter_id,
    n.window_id = $window_id,
    n.span_type = $span_type,
    n.salience_score = $salience_score,
    n.entities = $entities,
    n.char_start = $char_start,
    n.char_end = $char_end,
    n.status = 'active'
"""
```
