from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest

from loreweaver.config import AppConfig
from loreweaver.graph.center_span import build_m15_graph, list_graph_clusters
from loreweaver.models.chapter import Chapter
from loreweaver.models.document import Document
from loreweaver.models.span import Span
from loreweaver.storage.sqlite_store import SQLiteStore


class M15GraphTests(unittest.TestCase):
    def test_builds_center_span_clusters_and_edges(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = AppConfig(
                path=root / "default.yaml",
                values={
                    "project": {"data_dir": str(root / "data")},
                    "graph": {
                        "cluster_count": 2,
                        "members_per_cluster": 5,
                        "min_members_per_cluster": 5,
                    },
                },
            )
            storage_config = AppConfig(
                path=root / "storage.yaml",
                values={
                    "sqlite": {"path": str(root / "data" / "runs" / "test.sqlite3")},
                    "neo4j": {"enabled": False},
                },
            )
            store = SQLiteStore(storage_config.sqlite_path)
            store.initialize()
            store.initialize_extraction_tables()
            store.initialize_graph_tables()

            document = Document(
                document_id="doc_graph",
                title="Graph Sample",
                author=None,
                source_path=str(root / "raw.txt"),
                normalized_path=str(root / "normalized.txt"),
                total_chars=3000,
                total_chapters=3,
                content_hash="hash_graph",
                created_at=datetime.now(timezone.utc),
            )
            chapters = [
                Chapter(
                    chapter_id=f"doc_graph_ch{index:04d}",
                    document_id=document.document_id,
                    chapter_index=index,
                    chapter_title=f"第{index}章",
                    start_idx=(index - 1) * 1000,
                    end_idx=index * 1000,
                    char_count=1000,
                )
                for index in range(1, 4)
            ]
            store.upsert_document_with_chapters(document, chapters)
            store.initialize_extraction_tables()
            for span in _sample_spans(document.document_id):
                store.upsert_span(span)

            report = build_m15_graph(
                config=config,
                storage_config=storage_config,
                run_id="graph_test",
                document_id=document.document_id,
                sync_neo4j=False,
            )
            listed = list_graph_clusters(
                storage_config=storage_config,
                document_id=document.document_id,
            )

            self.assertTrue(Path(report["report_path"]).exists())
            self.assertEqual(report["cluster_count"], 2)
            self.assertEqual(listed["cluster_count"], 2)
            self.assertGreaterEqual(report["edge_counts"]["SUPPORTS"], 10)
            self.assertGreaterEqual(report["edge_counts"]["RELATED_TO"], 8)
            self.assertGreaterEqual(report["edge_counts"]["MENTIONS_ENTITY"], 10)
            self.assertEqual(report["edge_counts"]["ADJACENT_CHAPTER"], 2)
            self.assertTrue(all(cluster["member_count"] >= 5 for cluster in report["clusters"]))
            self.assertEqual({cluster["cluster_type"] for cluster in report["clusters"]}, {"mystery", "power_system"})

            with sqlite3.connect(storage_config.sqlite_path) as connection:
                cluster_count = connection.execute(
                    "SELECT COUNT(*) FROM center_span_clusters"
                ).fetchone()[0]
                edge_count = connection.execute("SELECT COUNT(*) FROM span_edges").fetchone()[0]
                report_count = connection.execute("SELECT COUNT(*) FROM graph_reports").fetchone()[0]

            self.assertEqual(cluster_count, 2)
            self.assertEqual(edge_count, report["edge_count"])
            self.assertEqual(report_count, 1)


def _sample_spans(document_id: str) -> list[Span]:
    now = datetime.now(timezone.utc)
    spans: list[Span] = []
    rows = [
        ("exposition", "精神视角规则", "高文以漂浮视角观察世界，精神状态遵循异常规则。", ["高文"], ["精神规则", "观察限制"]),
        ("reflection", "观察距离限制", "高文发现视角固定，无法随意移动。", ["高文"], ["精神规则", "观察限制"]),
        ("reflection", "意识维持方式", "高文依靠持续思考维持自我。", ["高文"], ["精神规则", "生存危机"]),
        ("exposition", "异常能力边界", "异常能力并非万能，只能记录和观察。", ["高文"], ["精神规则", "能力边界"]),
        ("progression", "精神规则转折", "精神状态濒临消散时出现新的转折。", ["高文"], ["精神规则", "转折点"]),
        ("reflection", "火种时间异常", "火种诞生后，高文的时间感知发生异常。", ["高文", "火种"], ["时间异常", "伏笔"]),
        ("progression", "逃逸程序提示", "神秘声音提示逃逸程序启动。", ["高文"], ["逃逸程序", "伏笔"]),
        ("progression", "黑钢棺材怪响", "黑钢棺材内传来怪响，众人感到不安。", ["黑钢棺材", "瑞贝卡"], ["异常现象", "伏笔"]),
        ("reflection", "身份复苏异常", "高文复苏后的身份与年代形成疑点。", ["高文", "瑞贝卡"], ["身份异常", "伏笔"]),
        ("exposition", "古老秘密线索", "旧时代秘密在对话中被反复暗示。", ["高文", "塞西尔家族"], ["秘密", "伏笔"]),
    ]
    for index, (span_type, _topic, summary, entities, topics) in enumerate(rows, start=1):
        chapter_index = 1 if index <= 4 else 2 if index <= 8 else 3
        spans.append(
            Span(
                span_id=f"span_{index:03d}",
                document_id=document_id,
                chapter_id=f"{document_id}_ch{chapter_index:04d}",
                window_id=f"window_{chapter_index:03d}",
                span_index_in_window=index,
                window_start=(chapter_index - 1) * 1000,
                window_end=chapter_index * 1000,
                span_type=span_type,
                summary=summary,
                entities=entities,
                topics=topics,
                salience_score=0.95 - index * 0.01,
                start_anchor_quote=summary[:8],
                end_anchor_quote=summary[-8:],
                key_quote=summary[:20],
                span_start_idx=(index - 1) * 100,
                span_end_idx=(index - 1) * 100 + 80,
                located_text=summary,
                locator_confidence=1.0,
                locator_status="located",
                created_at=now,
            )
        )
    return spans


if __name__ == "__main__":
    unittest.main()
