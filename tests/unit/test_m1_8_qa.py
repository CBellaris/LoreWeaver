from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest

from loreweaver.config import AppConfig
from loreweaver.models.chapter import Chapter
from loreweaver.models.document import Document
from loreweaver.qa.answerer import (
    AnswerClient,
    _qa_settings,
    answer_evidence_pack,
    validate_answer_citations,
)
from loreweaver.storage.sqlite_store import SQLiteStore


class M18QATests(unittest.TestCase):
    def test_generates_persists_mock_answer_with_valid_citation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config, storage_config, models_config, document = _setup_store(root)
            evidence_report = _evidence_report(root, document.document_id)

            report = answer_evidence_pack(
                config=config,
                storage_config=storage_config,
                models_config=models_config,
                evidence_report=evidence_report,
                mock_answer=True,
            )

            self.assertTrue(report["answer_validation"]["ok"])
            self.assertEqual(report["answer_validation"]["citations"], ["[E001]"])
            self.assertIn("[E001]", report["answer"])
            self.assertTrue(Path(report["report_path"]).exists())

            with sqlite3.connect(storage_config.sqlite_path) as connection:
                answer = connection.execute(
                    "SELECT answer FROM evidence_packs WHERE query_id = ?",
                    ("query_m18",),
                ).fetchone()[0]
                query_count = connection.execute(
                    "SELECT COUNT(*) FROM query_runs WHERE query_id = ?",
                    ("query_m18",),
                ).fetchone()[0]
            self.assertEqual(answer, report["answer"])
            self.assertEqual(query_count, 1)

    def test_validates_unknown_and_missing_citations(self) -> None:
        unknown = validate_answer_citations(
            "结论：不支持 [E099]",
            valid_citation_ids=["[E001]"],
        )
        missing = validate_answer_citations(
            "结论：没有引用。",
            valid_citation_ids=["[E001]"],
        )

        self.assertFalse(unknown.ok)
        self.assertIn("unknown citations: [E099]", unknown.errors)
        self.assertFalse(missing.ok)
        self.assertIn("answer contains no citations", missing.errors)

    def test_repairs_invalid_citation_once(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config, storage_config, models_config, document = _setup_store(root)
            evidence_report = _evidence_report(root, document.document_id)

            report = answer_evidence_pack(
                config=config,
                storage_config=storage_config,
                models_config=models_config,
                evidence_report=evidence_report,
                mock_answer=False,
                answer_client=RepairingAnswerClient(),
            )

            self.assertTrue(report["answer_validation"]["ok"])
            self.assertTrue(report["answer_validation"]["repaired"])
            self.assertEqual(report["answer_validation"]["citations"], ["[E001]"])

    def test_qa_settings_use_model_config_when_default_has_no_model(self) -> None:
        config = AppConfig(
            path=Path("default.yaml"),
            values={"qa": {"temperature": 0, "require_citations": True}},
        )
        models_config = AppConfig(
            path=Path("models.yaml"),
            values={
                "providers": {
                    "siliconflow": {
                        "api_key_env": "SILICONFLOW_API_KEY",
                        "base_url": "https://api.siliconflow.cn/v1",
                    }
                },
                "models": {
                    "qa": {
                        "provider": "siliconflow",
                        "name": "deepseek-ai/DeepSeek-V3.2",
                    }
                },
            },
        )

        settings = _qa_settings(config=config, models_config=models_config)

        self.assertEqual(settings["provider"], "siliconflow")
        self.assertEqual(settings["model"], "deepseek-ai/DeepSeek-V3.2")
        self.assertEqual(settings["api_key_env"], "SILICONFLOW_API_KEY")


def _setup_store(
    root: Path,
) -> tuple[AppConfig, AppConfig, AppConfig, Document]:
    data_dir = root / "data"
    storage_config = AppConfig(
        path=root / "storage.yaml",
        values={"sqlite": {"path": str(data_dir / "runs" / "test.sqlite3")}},
    )
    config = AppConfig(
        path=root / "default.yaml",
        values={
            "project": {"data_dir": str(data_dir)},
            "qa": {"model": "mock-answerer", "temperature": 0, "require_citations": True},
        },
    )
    models_config = AppConfig(
        path=root / "models.yaml",
        values={
            "providers": {"mock": {"api_key_env": "MOCK_API_KEY"}},
            "models": {"qa": {"provider": "mock", "name": "mock-answerer", "temperature": 0}},
        },
    )
    normalized_path = root / "normalized.txt"
    normalized_path.write_text("塞西尔家族与高文的证据文本。" * 10, encoding="utf-8")
    document = Document(
        document_id="doc_m18",
        title="QA Sample",
        author=None,
        source_path=str(root / "raw.txt"),
        normalized_path=str(normalized_path),
        total_chars=len(normalized_path.read_text(encoding="utf-8")),
        total_chapters=1,
        content_hash="hash_m18",
        created_at=datetime.now(timezone.utc),
    )
    store = SQLiteStore(storage_config.sqlite_path)
    store.initialize()
    store.initialize_index_tables()
    store.initialize_graph_tables()
    store.initialize_evidence_tables()
    store.upsert_document_with_chapters(
        document,
        [Chapter("doc_m18_ch0001", document.document_id, 1, "第一章", 0, 200, 200)],
    )
    store.initialize_index_tables()
    store.initialize_graph_tables()
    store.initialize_evidence_tables()
    store.insert_evidence_pack(_pack(document.document_id), report={})
    return config, storage_config, models_config, document


def _evidence_report(root: Path, document_id: str) -> dict:
    pack = _pack_payload(document_id)
    return {
        "run_id": "evidence_run",
        "query_id": "query_m18",
        "document_id": document_id,
        "question": pack["user_question"],
        "query_type": pack["query_type"],
        "report_path": str(root / "query_m18_evidence_pack.json"),
        "evidence_pack": pack,
        "assembly": {"evidence_block_count": 1},
    }


def _pack(document_id: str):
    from loreweaver.models.evidence import QueryEvidencePack

    payload = _pack_payload(document_id)
    return QueryEvidencePack(
        query_id=payload["query_id"],
        document_id=payload["document_id"],
        user_question=payload["user_question"],
        query_type=payload["query_type"],
        retrieved_span_ids=payload["retrieved_span_ids"],
        cluster_ids=payload["cluster_ids"],
        merged_intervals=payload["merged_intervals"],
        evidence_blocks=payload["evidence_blocks"],
        retrieval_sources=payload["retrieval_sources"],
        rerank_scores=payload["rerank_scores"],
        token_estimate=payload["token_estimate"],
        answer=None,
        created_at=datetime.now(timezone.utc),
    )


def _pack_payload(document_id: str) -> dict:
    return {
        "query_id": "query_m18",
        "document_id": document_id,
        "user_question": "塞西尔家族和高文有什么关系？",
        "query_type": "character_relation",
        "retrieved_span_ids": ["span_a"],
        "cluster_ids": [],
        "merged_intervals": [],
        "evidence_blocks": [
            {
                "citation_id": "[E001]",
                "document_id": document_id,
                "chapter_id": "doc_m18_ch0001",
                "chapter_title": "第一章",
                "start_idx": 0,
                "end_idx": 20,
                "text": "高文醒来后面对塞西尔家族的困境。",
                "source_span_ids": ["span_a"],
                "retrieval_sources": ["graph", "bm25"],
                "rerank_score": 0.9,
            }
        ],
        "retrieval_sources": {"summary": {}, "by_span_id": {"span_a": ["graph"]}},
        "rerank_scores": {"span_a": 0.9},
        "token_estimate": 20,
        "answer": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


class RepairingAnswerClient(AnswerClient):
    provider = "mock"
    model = "repairing"

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> tuple[str, dict[str, int]]:
        del messages, temperature
        self.calls += 1
        if self.calls == 1:
            return "结论：错误引用 [E099]", {}
        return "结论：修复后引用 [E001]", {}


if __name__ == "__main__":
    unittest.main()
