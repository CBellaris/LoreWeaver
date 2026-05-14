from pathlib import Path
from tempfile import TemporaryDirectory
import os
import sqlite3
import unittest

from loreweaver.config import AppConfig
from loreweaver.extraction.extractor import extract_document_windows
from loreweaver.indexing.pipeline import build_m14_indexes, search_bm25_index, search_vector_index
from loreweaver.ingest.pipeline import ingest_text
from loreweaver.ingest.window_splitter import build_candidate_windows
from loreweaver.storage.bm25_store import BM25Index, tokenize_for_bm25


class M14IndexingTests(unittest.TestCase):
    def test_chinese_bm25_tokenizer_keeps_entity_ngrams(self) -> None:
        tokens = tokenize_for_bm25("塞西尔领的魔网")

        self.assertIn("塞西", tokens)
        self.assertIn("西尔", tokens)
        self.assertIn("魔网", tokens)

    def test_build_indexes_and_debug_searches_with_local_backends(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            os.environ.pop("LOREWEAVER_TEST_QDRANT_URL", None)
            source = root / "raw.txt"
            source.write_text(
                "第一章 A\n"
                + ("高文在陌生的大厅中醒来，并意识到这座城堡隐藏着旧时代留下的秘密。" * 18)
                + "\n第二章 B\n"
                + ("瑞贝卡提到塞西尔领、魔网和家族危机，这些信息共同构成早期世界观线索。" * 18),
                encoding="utf-8",
            )
            config = AppConfig(
                path=root / "default.yaml",
                values={
                    "project": {"data_dir": str(root / "data")},
                    "sample": {"title": "T", "author": "A"},
                    "ingest": {
                        "normalize_newlines": True,
                        "remove_extra_blank_lines": True,
                        "chapter_patterns": [
                            r"^第[一二三四五六七八九十百千万零〇两0-9]+章",
                        ],
                    },
                    "window": {
                        "mode": "auto",
                        "size_chars": 500,
                        "overlap_ratio": 0.2,
                        "min_chars": 120,
                        "max_chars": 800,
                    },
                    "extraction": {
                        "model": "mock",
                        "temperature": 0,
                        "max_retries": 0,
                        "anchor_min_chars": 8,
                        "anchor_max_chars": 80,
                    },
                    "locator": {"fuzzy_threshold": 0.86},
                    "indexing": {
                        "embedding_dimensions": 8,
                        "embedding_batch_size": 2,
                        "embedding_input": {
                            "include_key_quote": False,
                            "include_located_text": False,
                        },
                    },
                },
            )
            storage_config = AppConfig(
                path=root / "storage.yaml",
                values={
                    "sqlite": {"path": str(root / "data" / "runs" / "test.sqlite3")},
                    "qdrant": {
                        "url_env": "LOREWEAVER_TEST_QDRANT_URL",
                        "api_key_env": "LOREWEAVER_TEST_QDRANT_API_KEY",
                        "local_path": str(root / "data" / "indexes" / "qdrant"),
                        "collection_prefix": "test",
                        "distance": "cosine",
                    },
                    "bm25": {"index_dir": str(root / "data" / "indexes")},
                },
            )
            models_config = AppConfig(
                path=root / "models.yaml",
                values={
                    "providers": {"mock": {"adapter": "mock"}},
                    "services": {
                        "extraction": {
                            "capability": "chat",
                            "provider": "mock",
                            "model": "mock",
                            "temperature": 0,
                        },
                        "embedding": {
                            "capability": "embedding",
                            "provider": "mock",
                            "model": "mock-embedding",
                            "expected_dimensions": 8,
                            "batch_size": 2,
                        },
                    },
                },
            )

            ingest_report = ingest_text(
                config=config,
                storage_config=storage_config,
                run_id="ingest_test",
                source_path=source,
            )
            build_candidate_windows(
                config=config,
                storage_config=storage_config,
                run_id="windows_test",
                document_id=ingest_report["document"]["document_id"],
            )
            extract_document_windows(
                config=config,
                storage_config=storage_config,
                models_config=models_config,
                run_id="extract_test",
                document_id=ingest_report["document"]["document_id"],
                limit=2,
                mock=True,
            )

            index_report = build_m14_indexes(
                config=config,
                storage_config=storage_config,
                models_config=models_config,
                run_id="index_test",
                document_id=ingest_report["document"]["document_id"],
                mock_embeddings=True,
            )
            vector_report = search_vector_index(
                config=config,
                storage_config=storage_config,
                models_config=models_config,
                query="旧时代秘密",
                document_id=ingest_report["document"]["document_id"],
                top_k=2,
                mock_embeddings=True,
            )
            bm25_report = search_bm25_index(
                storage_config=storage_config,
                query="高文",
                document_id=ingest_report["document"]["document_id"],
                top_k=2,
            )

            self.assertTrue(Path(index_report["report_path"]).exists())
            self.assertTrue(Path(index_report["bm25"]["index_path"]).exists())
            self.assertEqual(index_report["located_span_count"], 4)
            self.assertEqual(index_report["qdrant"]["collection_count"], 4)
            self.assertEqual(index_report["bm25"]["document_count"], 4)
            self.assertEqual(len(vector_report["results"]), 2)
            self.assertGreaterEqual(len(bm25_report["results"]), 1)
            self.assertIn("高文", bm25_report["results"][0]["summary"])

            with sqlite3.connect(storage_config.sqlite_path) as connection:
                cache_count = connection.execute(
                    "SELECT COUNT(*) FROM embedding_cache"
                ).fetchone()[0]
                report_count = connection.execute(
                    "SELECT COUNT(*) FROM index_reports"
                ).fetchone()[0]

            self.assertEqual(cache_count, 4)
            self.assertEqual(report_count, 1)

            loaded_bm25 = BM25Index.load(index_report["bm25"]["index_path"])
            self.assertEqual(len(loaded_bm25.search("高文", top_k=1)), 1)


if __name__ == "__main__":
    unittest.main()
