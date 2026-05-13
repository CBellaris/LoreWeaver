from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest

from loreweaver.config import AppConfig
from loreweaver.models.chapter import Chapter
from loreweaver.models.cluster import CenterSpanCluster, SpanEdge
from loreweaver.models.document import Document
from loreweaver.models.span import Span
from loreweaver.retrieval.pipeline import retrieve_m16
from loreweaver.storage.bm25_store import BM25Index, bm25_index_path
from loreweaver.storage.sqlite_store import SQLiteStore


class M16RetrievalTests(unittest.TestCase):
    def test_retrieves_unions_and_mock_reranks_graph_and_bm25_hits(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = AppConfig(
                path=root / "default.yaml",
                values={
                    "project": {"data_dir": str(root / "data")},
                    "retrieval": {
                        "graph_cluster_top_k": 2,
                        "graph_span_per_cluster": 3,
                        "vector_top_k": 3,
                        "bm25_top_k": 5,
                        "union_max_candidates": 10,
                        "rerank_top_k": 4,
                    },
                    "indexing": {"embedding_dimensions": 8},
                },
            )
            storage_config = AppConfig(
                path=root / "storage.yaml",
                values={
                    "sqlite": {"path": str(root / "data" / "runs" / "test.sqlite3")},
                    "bm25": {"index_dir": str(root / "data" / "indexes")},
                    "qdrant": {"local_path": str(root / "data" / "qdrant")},
                    "neo4j": {"enabled": False},
                },
            )
            models_config = AppConfig(
                path=root / "models.yaml",
                values={
                    "providers": {"siliconflow": {"api_key_env": "SILICONFLOW_API_KEY"}},
                    "models": {
                        "embedding": {
                            "provider": "siliconflow",
                            "name": "Qwen/Qwen3-Embedding-0.6B",
                            "expected_dimensions": 8,
                        },
                        "reranker": {
                            "provider": "siliconflow",
                            "name": "Qwen/Qwen3-Reranker-0.6B",
                            "enabled": False,
                        },
                    },
                },
            )
            store = SQLiteStore(storage_config.sqlite_path)
            store.initialize()
            store.initialize_extraction_tables()
            store.initialize_index_tables()
            store.initialize_graph_tables()
            document = _document(root)
            chapters = _chapters(document.document_id)
            store.upsert_document_with_chapters(document, chapters)
            spans = _spans(document.document_id)
            for span in spans:
                store.upsert_span(span)
            store.replace_graph(
                document_id=document.document_id,
                clusters=[_cluster(document.document_id)],
                edges=_edges(document.document_id),
            )
            BM25Index.from_spans(document_id=document.document_id, spans=spans).save(
                bm25_index_path(root / "data" / "indexes", document.document_id)
            )

            report = retrieve_m16(
                config=config,
                storage_config=storage_config,
                models_config=models_config,
                question="塞西尔家族为什么衰落，和高文有什么关系？",
                document_id=document.document_id,
                mock_embeddings=True,
                mock_reranker=True,
            )

            self.assertTrue(Path(report["report_path"]).exists())
            self.assertEqual(report["query_type"], "character_relation")
            self.assertGreaterEqual(report["retrieval"]["graph"]["count"], 3)
            self.assertGreaterEqual(report["retrieval"]["bm25"]["count"], 1)
            self.assertGreaterEqual(report["retrieval"]["union"]["candidate_count"], 3)
            self.assertEqual(report["reranker"]["provider"], "mock")
            self.assertTrue(report["top_results"])
            self.assertIn("span_cecil_1", {item["span_id"] for item in report["top_results"]})

            with sqlite3.connect(storage_config.sqlite_path) as connection:
                query_count = connection.execute("SELECT COUNT(*) FROM query_runs").fetchone()[0]
            self.assertEqual(query_count, 1)


def _document(root: Path) -> Document:
    return Document(
        document_id="doc_m16",
        title="Retrieval Sample",
        author=None,
        source_path=str(root / "raw.txt"),
        normalized_path=str(root / "normalized.txt"),
        total_chars=2000,
        total_chapters=2,
        content_hash="hash_m16",
        created_at=datetime.now(timezone.utc),
    )


def _chapters(document_id: str) -> list[Chapter]:
    return [
        Chapter(
            chapter_id=f"{document_id}_ch{index:04d}",
            document_id=document_id,
            chapter_index=index,
            chapter_title=f"第{index}章",
            start_idx=(index - 1) * 1000,
            end_idx=index * 1000,
            char_count=1000,
        )
        for index in range(1, 3)
    ]


def _spans(document_id: str) -> list[Span]:
    now = datetime.now(timezone.utc)
    rows = [
        (
            "span_cecil_1",
            "doc_m16_ch0001",
            "塞西尔家族衰落",
            "高文发现塞西尔家族因为旧贵族内耗和边境压力逐渐衰落。",
            ["高文", "塞西尔家族"],
            ["家族衰落", "边境压力"],
        ),
        (
            "span_cecil_2",
            "doc_m16_ch0001",
            "高文继承家族责任",
            "高文复苏后被视为塞西尔先祖，需要重新承担家族责任。",
            ["高文", "塞西尔家族"],
            ["责任", "复苏"],
        ),
        (
            "span_magic_1",
            "doc_m16_ch0002",
            "法师道路旧规",
            "法师道路依赖旧时代知识，传承缺失导致规则变得混乱。",
            ["法师"],
            ["力量体系", "旧时代知识"],
        ),
        (
            "span_location_1",
            "doc_m16_ch0002",
            "边境营地",
            "边境营地暴露了塞西尔领地资源不足的问题。",
            ["塞西尔领地"],
            ["地点", "资源"],
        ),
    ]
    spans: list[Span] = []
    for index, (span_id, chapter_id, _topic, summary, entities, topics) in enumerate(rows):
        spans.append(
            Span(
                span_id=span_id,
                document_id=document_id,
                chapter_id=chapter_id,
                window_id=f"window_{index}",
                span_index_in_window=index,
                window_start=index * 100,
                window_end=index * 100 + 100,
                span_type="faction" if "cecil" in span_id else "setting",
                micro_summary=summary,
                entities=entities,
                topics=topics,
                salience_score=0.9 - index * 0.05,
                start_anchor_quote=summary[:6],
                end_anchor_quote=summary[-6:],
                key_quote=summary,
                overlap_reason="",
                span_start_idx=index * 100,
                span_end_idx=index * 100 + 80,
                located_text=summary,
                locator_confidence=1.0,
                locator_status="located",
                created_at=now,
            )
        )
    return spans


def _cluster(document_id: str) -> CenterSpanCluster:
    return CenterSpanCluster(
        cluster_id="cluster_cecil",
        document_id=document_id,
        center_span_id="span_cecil_1",
        cluster_name="塞西尔家族衰落与高文复苏",
        cluster_type="faction",
        micro_summary="围绕塞西尔家族衰落、高文身份和家族责任的证据簇。",
        member_span_ids=["span_cecil_1", "span_cecil_2", "span_location_1"],
        confidence=0.9,
        status="active",
        created_at=datetime.now(timezone.utc),
    )


def _edges(document_id: str) -> list[SpanEdge]:
    now = datetime.now(timezone.utc)
    return [
        SpanEdge(
            edge_id=f"edge_{index}",
            document_id=document_id,
            from_id="cluster_cecil",
            to_id=span_id,
            from_type="CenterSpanCluster",
            to_type="Span",
            edge_type="SUPPORTS",
            weight=weight,
            source="test",
            created_at=now,
        )
        for index, (span_id, weight) in enumerate(
            [("span_cecil_1", 0.95), ("span_cecil_2", 0.9), ("span_location_1", 0.75)]
        )
    ]


if __name__ == "__main__":
    unittest.main()
