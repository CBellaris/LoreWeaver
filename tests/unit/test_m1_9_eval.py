from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import json
import unittest

from loreweaver.config import AppConfig
from loreweaver.eval.corpus import build_chapter_corpus
from loreweaver.eval.metrics import chapter_ranking_from_retrieval_report, score_question
from loreweaver.eval.question_set import load_question_set
from loreweaver.model_services.errors import EmptyModelResponse
from loreweaver.model_services.json_utils import chat_content_from_response
from loreweaver.models.chapter import Chapter
from loreweaver.models.document import Document
from loreweaver.storage.sqlite_store import SQLiteStore


class M19EvalTests(unittest.TestCase):
    def test_question_set_normalizes_gold_weights(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "questions.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "question_id": "q_0001",
                        "question": "塞西尔家族为什么衰落？",
                        "answer": "因为旧贵族内耗和边境压力。",
                        "profile": "broad",
                        "query_type": "causality",
                        "required_facets": ["家族内耗", "边境压力"],
                        "expected_chapters": [
                            {
                                "chapter_id": "doc_ch0001",
                                "chapter_index": 1,
                                "relevance": 3,
                                "weight": 3,
                                "facet": "家族内耗",
                            },
                            {
                                "chapter_id": "doc_ch0002",
                                "chapter_index": 2,
                                "relevance": 1,
                                "weight": 1,
                                "facet": "边境压力",
                            },
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            questions = load_question_set(path)

            self.assertEqual(len(questions), 1)
        self.assertAlmostEqual(questions[0].expected_chapters[0].weight, 0.75)
        self.assertAlmostEqual(questions[0].expected_chapters[1].weight, 0.25)
        self.assertEqual(questions[0].required_facets, ["家族内耗", "边境压力"])

    def test_scores_span_results_as_chapter_recall(self) -> None:
        question = load_question_set(_write_question_set())[0]
        retrieval_report = {
            "top_results": [
                {
                    "rank": 1,
                    "span_id": "span_a",
                    "chapter_id": "doc_ch0003",
                    "rerank_score": 0.9,
                    "sources": ["bm25"],
                },
                {
                    "rank": 2,
                    "span_id": "span_b",
                    "chapter_id": "doc_ch0001",
                    "rerank_score": 0.8,
                    "sources": ["vector"],
                },
                {
                    "rank": 3,
                    "span_id": "span_c",
                    "chapter_id": "doc_ch0001",
                    "rerank_score": 0.7,
                    "sources": ["graph"],
                },
            ]
        }

        ranking = chapter_ranking_from_retrieval_report(retrieval_report)
        score = score_question(question=question, predicted_chapters=ranking)

        self.assertEqual([item["chapter_id"] for item in ranking], ["doc_ch0003", "doc_ch0001"])
        self.assertEqual(ranking[1]["hit_count"], 2)
        self.assertAlmostEqual(score["weighted_recall_at_1"], 0.0)
        self.assertAlmostEqual(score["weighted_recall_at_3"], 1.0)
        self.assertAlmostEqual(score["facet_coverage_at_3"], 1.0)
        self.assertAlmostEqual(score["mrr"], 0.5)

    def test_build_corpus_exports_chapter_text_from_sqlite_coordinates(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            normalized_path = root / "normalized.txt"
            normalized_path.write_text("第一章 A\n正文一\n第二章 B\n正文二\n", encoding="utf-8")
            storage_config = AppConfig(
                path=root / "storage.yaml",
                values={"sqlite": {"path": str(root / "data" / "runs" / "test.sqlite3")}},
            )
            config = AppConfig(
                path=root / "default.yaml",
                values={"project": {"data_dir": str(root / "data")}},
            )
            store = SQLiteStore(storage_config.sqlite_path)
            store.initialize()
            document = Document(
                document_id="doc_eval",
                title="Eval Sample",
                author=None,
                source_path=str(root / "raw.txt"),
                normalized_path=str(normalized_path),
                total_chars=20,
                total_chapters=2,
                content_hash="hash_eval",
                created_at=datetime.now(timezone.utc),
            )
            chapters = [
                Chapter(
                    chapter_id="doc_eval_ch0001",
                    document_id=document.document_id,
                    chapter_index=1,
                    chapter_title="第一章 A",
                    start_idx=0,
                    end_idx=10,
                    char_count=10,
                ),
                Chapter(
                    chapter_id="doc_eval_ch0002",
                    document_id=document.document_id,
                    chapter_index=2,
                    chapter_title="第二章 B",
                    start_idx=10,
                    end_idx=20,
                    char_count=10,
                ),
            ]
            store.upsert_document_with_chapters(document, chapters)

            report = build_chapter_corpus(
                config=config,
                storage_config=storage_config,
                document_id=document.document_id,
                chapter_start=1,
                chapter_end=1,
            )

            self.assertEqual(report["chapter_count"], 1)
            self.assertTrue(Path(report["corpus_path"]).exists())
            self.assertEqual(report["chapters"][0]["text"], "第一章 A\n正文一\n")

    def test_empty_generation_response_reports_provider_payload(self) -> None:
        response = SimpleNamespace(
            choices=None,
            model_dump=lambda: {
                "id": "chatcmpl_empty",
                "choices": None,
                "error": {"message": "context length exceeded"},
            },
        )

        with self.assertRaisesRegex(
            EmptyModelResponse,
            "context length exceeded",
        ):
            chat_content_from_response(response, context="Eval question generation")

    def test_null_generation_response_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(
            EmptyModelResponse,
            "null response",
        ):
            chat_content_from_response(None, context="Eval question generation")


def _write_question_set() -> Path:
    tmpdir = TemporaryDirectory()
    root = Path(tmpdir.name)
    path = root / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "q_0001",
                "question": "塞西尔家族为什么衰落？",
                "answer": "因为旧贵族内耗和边境压力。",
                "profile": "broad",
                "query_type": "causality",
                "required_facets": ["家族衰落"],
                "expected_chapters": [
                    {
                        "chapter_id": "doc_ch0001",
                        "chapter_index": 1,
                        "relevance": 3,
                        "weight": 1,
                        "facet": "家族衰落",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _TEMP_DIRS.append(tmpdir)
    return path


_TEMP_DIRS: list[TemporaryDirectory] = []


if __name__ == "__main__":
    unittest.main()
