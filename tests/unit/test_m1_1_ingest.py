from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest

from loreweaver.config import AppConfig
from loreweaver.ingest.pipeline import ingest_text
from loreweaver.ingest.chapter_splitter import split_chapters


class M11IngestTests(unittest.TestCase):
    def test_split_prefers_real_chapter_titles_over_part_wrappers(self) -> None:
        text = (
            "书名: 示例\n\n"
            "【 第 1 部分 】\n\n"
            "第一章 开始\n正文一\n\n"
            "【 第 2 部分 】\n\n"
            "第二章 继续\n正文二\n"
        )

        chapters, report = split_chapters(
            text,
            document_id="doc_test",
            chapter_patterns=[
                r"^第[一二三四五六七八九十百千万零〇两0-9]+章",
                r"^[一二三四五六七八九十百千万零〇两0-9]+章",
                r"^【 第 [0-9]+ 部分 】",
            ],
        )

        self.assertEqual(report.strategy, "real_chapter_patterns")
        self.assertEqual(
            [chapter.chapter_title for chapter in chapters],
            ["第一章 开始", "第二章 继续"],
        )
        self.assertEqual(
            text[chapters[0].start_idx : chapters[0].end_idx].splitlines()[0],
            "第一章 开始",
        )

    def test_ingest_writes_normalized_text_and_sqlite_records(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "raw.txt"
            source.write_text(
                "第一章 A\r\n内容\r\n\r\n\r\n第二章 B\r\n内容\n",
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
                            r"^[一二三四五六七八九十百千万零〇两0-9]+章",
                        ],
                    },
                },
            )
            storage_config = AppConfig(
                path=root / "storage.yaml",
                values={"sqlite": {"path": str(root / "data" / "runs" / "test.sqlite3")}},
            )

            report = ingest_text(
                config=config,
                storage_config=storage_config,
                run_id="ingest_test",
                source_path=source,
            )

            normalized_path = Path(report["document"]["normalized_path"])
            normalized_text = normalized_path.read_text(encoding="utf-8")
            self.assertNotIn("\r", normalized_text)
            self.assertEqual(report["document"]["total_chars"], len(normalized_text))
            self.assertEqual(report["document"]["total_chapters"], 2)

            with sqlite3.connect(report["sqlite_path"]) as connection:
                document_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                chapter_count = connection.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]

            self.assertEqual(document_count, 1)
            self.assertEqual(chapter_count, 2)

    def test_split_without_headings_uses_one_whole_document_fallback_chapter(self) -> None:
        text = "无标题正文\n" + ("内容" * 400)

        chapters, report = split_chapters(
            text,
            document_id="doc_fallback",
            chapter_patterns=[r"^第[一二三四五六七八九十百千万零〇两0-9]+章"],
        )

        self.assertEqual(report.strategy, "whole_document_fallback")
        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0].chapter_id, "doc_fallback_ch0000")
        self.assertEqual(chapters[0].chapter_index, 0)
        self.assertEqual(chapters[0].start_idx, 0)
        self.assertEqual(chapters[0].end_idx, len(text))


if __name__ == "__main__":
    unittest.main()
