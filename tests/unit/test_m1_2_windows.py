from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest

from loreweaver.config import AppConfig
from loreweaver.ingest.pipeline import ingest_text
from loreweaver.ingest.window_splitter import build_candidate_windows, split_candidate_windows
from loreweaver.models.chapter import Chapter
from loreweaver.storage.sqlite_store import SQLiteStore


class M12WindowTests(unittest.TestCase):
    def test_split_windows_stay_inside_chapters_and_match_text_slices(self) -> None:
        text = "第一章 A\n" + ("甲" * 2500) + "\n第二章 B\n" + ("乙" * 900)
        chapters = [
            Chapter("doc_test_ch0001", "doc_test", 1, "第一章 A", 0, 2507, 2507),
            Chapter("doc_test_ch0002", "doc_test", 2, "第二章 B", 2507, len(text), len(text) - 2507),
        ]

        windows, report = split_candidate_windows(
            text,
            document_id="doc_test",
            chapters=chapters,
            window_size_chars=1000,
            overlap_ratio=0.2,
            min_window_chars=300,
            max_window_chars=1400,
        )

        self.assertEqual(report.total_chapters, 2)
        self.assertEqual(report.boundary_warnings, [])
        self.assertGreater(report.total_windows, 2)

        chapters_by_id = {chapter.chapter_id: chapter for chapter in chapters}
        for window in windows:
            chapter = chapters_by_id[window.chapter_id]
            self.assertGreaterEqual(window.window_start, chapter.start_idx)
            self.assertLessEqual(window.window_end, chapter.end_idx)
            self.assertEqual(window.text, text[window.window_start : window.window_end])

    def test_short_tail_window_is_merged_into_previous_window(self) -> None:
        text = "第一章 A\n" + ("甲" * 2480)
        chapters = [Chapter("doc_test_ch0001", "doc_test", 1, "第一章 A", 0, len(text), len(text))]

        windows, report = split_candidate_windows(
            text,
            document_id="doc_test",
            chapters=chapters,
            window_size_chars=1000,
            overlap_ratio=0.2,
            min_window_chars=300,
            max_window_chars=1400,
        )

        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[-1].window_end, len(text))
        self.assertGreaterEqual(windows[-1].char_count, 300)
        self.assertEqual(report.short_window_count, 0)

    def test_split_by_chapter_uses_one_window_per_natural_chapter(self) -> None:
        text = "第一章 A\n" + ("甲" * 2500) + "\n第二章 B\n" + ("乙" * 900)
        chapters = [
            Chapter("doc_test_ch0001", "doc_test", 1, "第一章 A", 0, 2507, 2507),
            Chapter("doc_test_ch0002", "doc_test", 2, "第二章 B", 2507, len(text), len(text) - 2507),
        ]

        windows, report = split_candidate_windows(
            text,
            document_id="doc_test",
            chapters=chapters,
            window_size_chars=1000,
            overlap_ratio=0.2,
            min_window_chars=300,
            max_window_chars=1400,
            split_by_chapter=True,
        )

        self.assertEqual(report.split_mode, "chapter")
        self.assertEqual(report.total_windows, 2)
        self.assertEqual(report.boundary_warnings, [])
        self.assertEqual([window.window_index for window in windows], [1, 1])
        self.assertEqual(windows[0].window_start, chapters[0].start_idx)
        self.assertEqual(windows[0].window_end, chapters[0].end_idx)
        self.assertEqual(windows[1].window_start, chapters[1].start_idx)
        self.assertEqual(windows[1].window_end, chapters[1].end_idx)

    def test_windows_pipeline_persists_sqlite_records_and_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "raw.txt"
            source.write_text(
                "第一章 A\n" + ("甲" * 1800) + "\n第二章 B\n" + ("乙" * 1200),
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
                        "fallback_chapter_chars": 1000,
                        "chapter_patterns": [
                            r"^第[一二三四五六七八九十百千万零〇两0-9]+章",
                            r"^[一二三四五六七八九十百千万零〇两0-9]+章",
                        ],
                    },
                    "window": {
                        "size_chars": 1000,
                        "overlap_ratio": 0.2,
                        "min_chars": 300,
                        "max_chars": 1400,
                    },
                },
            )
            storage_config = AppConfig(
                path=root / "storage.yaml",
                values={"sqlite": {"path": str(root / "data" / "runs" / "test.sqlite3")}},
            )

            ingest_report = ingest_text(
                config=config,
                storage_config=storage_config,
                run_id="ingest_test",
                source_path=source,
            )
            windows_report = build_candidate_windows(
                config=config,
                storage_config=storage_config,
                run_id="windows_test",
                document_id=ingest_report["document"]["document_id"],
            )

            self.assertTrue(Path(windows_report["report_path"]).exists())
            store = SQLiteStore(windows_report["sqlite_path"])
            persisted_windows = store.list_candidate_windows(
                ingest_report["document"]["document_id"]
            )
            with sqlite3.connect(windows_report["sqlite_path"]) as connection:
                window_count = connection.execute(
                    "SELECT COUNT(*) FROM candidate_windows"
                ).fetchone()[0]
                report_count = connection.execute(
                    "SELECT COUNT(*) FROM window_reports"
                ).fetchone()[0]

            self.assertEqual(window_count, windows_report["window_split"]["total_windows"])
            self.assertEqual(len(persisted_windows), window_count)
            self.assertEqual(persisted_windows[0].window_index, 1)
            self.assertEqual(report_count, 1)

    def test_windows_pipeline_can_persist_chapter_windows(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "raw.txt"
            source.write_text(
                "第一章 A\n" + ("甲" * 1800) + "\n第二章 B\n" + ("乙" * 1200),
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
                        "fallback_chapter_chars": 1000,
                        "chapter_patterns": [
                            r"^第[一二三四五六七八九十百千万零〇两0-9]+章",
                            r"^[一二三四五六七八九十百千万零〇两0-9]+章",
                        ],
                    },
                    "window": {
                        "size_chars": 1000,
                        "overlap_ratio": 0.2,
                        "min_chars": 300,
                        "max_chars": 1400,
                    },
                },
            )
            storage_config = AppConfig(
                path=root / "storage.yaml",
                values={"sqlite": {"path": str(root / "data" / "runs" / "test.sqlite3")}},
            )

            ingest_report = ingest_text(
                config=config,
                storage_config=storage_config,
                run_id="ingest_test",
                source_path=source,
            )
            windows_report = build_candidate_windows(
                config=config,
                storage_config=storage_config,
                run_id="windows_test",
                document_id=ingest_report["document"]["document_id"],
                split_by_chapter=True,
            )

            self.assertEqual(windows_report["window_split"]["split_mode"], "chapter")
            self.assertEqual(windows_report["window_split"]["total_windows"], 2)


if __name__ == "__main__":
    unittest.main()
