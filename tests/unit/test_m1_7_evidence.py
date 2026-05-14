from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest

from loreweaver.config import AppConfig
from loreweaver.evidence.assembler import assemble_evidence_pack_from_retrieval_report
from loreweaver.models.chapter import Chapter
from loreweaver.models.document import Document
from loreweaver.storage.sqlite_store import SQLiteStore


class M17EvidencePackTests(unittest.TestCase):
    def test_assembles_merged_traceable_evidence_pack(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config, storage_config, document = _setup_store(root)
            retrieval_report = {
                "query_id": "query_m17",
                "document_id": document.document_id,
                "question": "塞西尔家族衰落和高文有什么关系？",
                "query_type": "character_relation",
                "report_path": str(root / "retrieval.json"),
                "retrieval": {"graph": {"count": 2}, "vector": {"count": 1}, "bm25": {"count": 1}},
                "top_results": [
                    _top_result(
                        "span_a",
                        "doc_m17_ch0001",
                        100,
                        180,
                        ["graph", "vector"],
                        0.92,
                        ["cluster_cecil"],
                    ),
                    _top_result("span_b", "doc_m17_ch0001", 240, 300, ["bm25"], 0.84),
                    _top_result("span_c", "doc_m17_ch0002", 1120, 1180, ["graph"], 0.72),
                ],
            }

            report = assemble_evidence_pack_from_retrieval_report(
                config=config,
                storage_config=storage_config,
                retrieval_report=retrieval_report,
            )
            pack = report["evidence_pack"]
            blocks = pack["evidence_blocks"]

            self.assertTrue(Path(report["report_path"]).exists())
            self.assertEqual(pack["query_id"], "query_m17")
            self.assertEqual(pack["cluster_ids"], ["cluster_cecil"])
            self.assertEqual(len(blocks), 2)
            self.assertEqual([block["citation_id"] for block in blocks], ["[E001]", "[E002]"])
            self.assertEqual(blocks[0]["start_idx"], 80)
            self.assertEqual(blocks[0]["end_idx"], 330)
            self.assertEqual(blocks[0]["source_span_ids"], ["span_a", "span_b"])
            self.assertEqual(set(blocks[0]["retrieval_sources"]), {"graph", "vector", "bm25"})
            self.assertEqual(blocks[0]["text"], _normalized_text()[80:330])
            self.assertLessEqual(
                sum(len(block["text"]) for block in blocks),
                config.values["evidence"]["max_evidence_chars"],
            )

            with sqlite3.connect(storage_config.sqlite_path) as connection:
                pack_count = connection.execute(
                    "SELECT COUNT(*) FROM evidence_packs WHERE query_id = ?",
                    ("query_m17",),
                ).fetchone()[0]
            self.assertEqual(pack_count, 1)

    def test_skips_invalid_coordinates_and_enforces_budget(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config, storage_config, document = _setup_store(
                root,
                evidence_overrides={"max_evidence_chars": 100, "max_blocks": 1},
            )
            retrieval_report = {
                "query_id": "query_budget",
                "document_id": document.document_id,
                "question": "证据预算测试",
                "query_type": "setting",
                "top_results": [
                    _top_result("span_good", "doc_m17_ch0002", 1120, 1180, ["graph"], 0.9),
                    _top_result("span_bad", "doc_m17_ch0001", 990, 1020, ["bm25"], 0.95),
                ],
            }

            report = assemble_evidence_pack_from_retrieval_report(
                config=config,
                storage_config=storage_config,
                retrieval_report=retrieval_report,
            )
            blocks = report["evidence_pack"]["evidence_blocks"]

            self.assertEqual(len(blocks), 1)
            self.assertEqual(len(blocks[0]["text"]), 100)
            self.assertEqual(blocks[0]["source_span_ids"], ["span_good"])
            self.assertEqual(report["assembly"]["warnings"][0]["span_id"], "span_bad")
            self.assertEqual(report["assembly"]["warnings"][0]["reason"], "coordinates outside chapter")


def _setup_store(
    root: Path,
    *,
    evidence_overrides: dict | None = None,
) -> tuple[AppConfig, AppConfig, Document]:
    data_dir = root / "data"
    normalized_path = root / "normalized.txt"
    normalized_path.write_text(_normalized_text(), encoding="utf-8")
    evidence_config = {
        "pre_context_chars": 20,
        "post_context_chars": 30,
        "merge_gap_chars": 100,
        "max_evidence_chars": 1000,
        "max_blocks": 12,
    }
    if evidence_overrides:
        evidence_config.update(evidence_overrides)
    config = AppConfig(
        path=root / "default.yaml",
        values={"project": {"data_dir": str(data_dir)}, "evidence": evidence_config},
    )
    storage_config = AppConfig(
        path=root / "storage.yaml",
        values={"sqlite": {"path": str(data_dir / "runs" / "test.sqlite3")}},
    )
    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    store.initialize_evidence_tables()
    document = Document(
        document_id="doc_m17",
        title="Evidence Sample",
        author=None,
        source_path=str(root / "raw.txt"),
        normalized_path=str(normalized_path),
        total_chars=len(_normalized_text()),
        total_chapters=2,
        content_hash="hash_m17",
        created_at=datetime.now(timezone.utc),
    )
    store.upsert_document_with_chapters(
        document,
        [
            Chapter("doc_m17_ch0001", document.document_id, 1, "第一章", 0, 1000, 1000),
            Chapter("doc_m17_ch0002", document.document_id, 2, "第二章", 1000, 2000, 1000),
        ],
    )
    return config, storage_config, document


def _normalized_text() -> str:
    return "0123456789" * 200


def _top_result(
    span_id: str,
    chapter_id: str,
    start_idx: int,
    end_idx: int,
    sources: list[str],
    score: float,
    cluster_ids: list[str] | None = None,
) -> dict:
    return {
        "span_id": span_id,
        "chapter_id": chapter_id,
        "span_start_idx": start_idx,
        "span_end_idx": end_idx,
        "sources": sources,
        "rerank_score": score,
        "fused_score": score,
        "source_scores": {},
        "normalized_scores": {},
        "cluster_ids": cluster_ids or [],
        "summary": span_id,
    }


if __name__ == "__main__":
    unittest.main()
