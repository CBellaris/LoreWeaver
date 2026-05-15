from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from loreweaver.models.chapter import Chapter
from loreweaver.models.document import Document
from loreweaver.models.span import Span
from loreweaver.models.window import CandidateWindow
from loreweaver.storage.sqlite_store import SQLiteStore
from loreweaver.web.inspectors import DebugInspector


class WebInspectorTests(unittest.TestCase):
    def test_span_review_reports_window_gaps_and_segments(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sqlite_path = root / "loreweaver.sqlite3"
            config_path = root / "default.yaml"
            storage_path = root / "storage.yaml"
            models_path = root / "models.yaml"
            config_path.write_text(
                "project:\n  data_dir: data\n",
                encoding="utf-8",
            )
            storage_path.write_text(
                f"sqlite:\n  path: {sqlite_path}\n",
                encoding="utf-8",
            )
            models_path.write_text("providers: {}\n", encoding="utf-8")

            store = SQLiteStore(sqlite_path)
            store.initialize()
            store.initialize_extraction_tables()

            text = "开头闲笔。主角发现古门。中间过渡。古门发光并显出符文。结尾闲笔。"
            document = Document(
                document_id="doc",
                title="Debug Doc",
                author=None,
                source_path="source.txt",
                normalized_path="normalized.txt",
                total_chars=len(text),
                total_chapters=1,
                content_hash="hash",
                created_at=datetime.now(timezone.utc),
            )
            chapter = Chapter(
                chapter_id="doc_ch0001",
                document_id="doc",
                chapter_index=1,
                chapter_title="第一章",
                start_idx=0,
                end_idx=len(text),
                char_count=len(text),
            )
            window = CandidateWindow(
                window_id="doc_ch0001_win0001",
                document_id="doc",
                chapter_id="doc_ch0001",
                window_index=1,
                window_start=0,
                window_end=len(text),
                text=text,
            )
            second_window = CandidateWindow(
                window_id="doc_ch0001_win0002",
                document_id="doc",
                chapter_id="doc_ch0001",
                window_index=2,
                window_start=100,
                window_end=110,
                text="第二窗口无抽取。",
            )
            store.upsert_document_with_chapters(document, [chapter])
            store.upsert_candidate_windows(document.document_id, [window, second_window])

            first_start = text.index("主角发现古门")
            first_end = first_start + len("主角发现古门")
            second_start = text.index("古门发光")
            second_end = second_start + len("古门发光并显出符文")
            now = datetime.now(timezone.utc)
            store.upsert_span(
                Span(
                    span_id="span_1",
                    document_id="doc",
                    chapter_id="doc_ch0001",
                    window_id=window.window_id,
                    span_index_in_window=1,
                    window_start=0,
                    window_end=len(text),
                    span_type="progression",
                    summary="主角发现古门。",
                    entities=["主角", "古门"],
                    salience_score=0.6,
                    start_anchor_quote="主角发现古门",
                    end_anchor_quote="主角发现古门",
                    key_quote="主角发现古门",
                    span_start_idx=first_start,
                    span_end_idx=first_end,
                    located_text=text[first_start:first_end],
                    locator_confidence=1.0,
                    locator_status="located",
                    created_at=now,
                )
            )
            store.upsert_span(
                Span(
                    span_id="span_2",
                    document_id="doc",
                    chapter_id="doc_ch0001",
                    window_id=window.window_id,
                    span_index_in_window=2,
                    window_start=0,
                    window_end=len(text),
                    span_type="exposition",
                    summary="古门发光并显出符文。",
                    entities=["古门", "符文"],
                    salience_score=0.7,
                    start_anchor_quote="古门发光",
                    end_anchor_quote="显出符文",
                    key_quote="古门发光",
                    span_start_idx=second_start,
                    span_end_idx=second_end,
                    located_text=text[second_start:second_end],
                    locator_confidence=1.0,
                    locator_status="located",
                    created_at=now,
                )
            )
            store.upsert_span(
                Span(
                    span_id="span_failed",
                    document_id="doc",
                    chapter_id="doc_ch0001",
                    window_id=window.window_id,
                    span_index_in_window=3,
                    window_start=0,
                    window_end=len(text),
                    span_type="other",
                    summary="一个定位失败的候选。",
                    entities=[],
                    salience_score=0.3,
                    start_anchor_quote="不存在",
                    end_anchor_quote="不存在",
                    key_quote="",
                    span_start_idx=None,
                    span_end_idx=None,
                    located_text="",
                    locator_confidence=0.0,
                    locator_status="failed",
                    created_at=now,
                )
            )
            store.insert_extraction_failure(
                window_id=window.window_id,
                span_id="span_failed",
                stage="locator",
                reason="start anchor not found",
                attempts=1,
                raw_output="{}",
            )

            inspector = DebugInspector(
                config_path=str(config_path),
                storage_config_path=str(storage_path),
                models_config_path=str(models_path),
            )

            review = inspector.span_review(document_id="doc", window_range="0-0", limit=10)
            self.assertEqual(review["summary"]["window_count"], 1)
            self.assertEqual(review["window_range"]["start"], 0)
            self.assertEqual(review["window_range"]["end"], 0)
            self.assertEqual(review["windows"][0]["global_window_index"], 0)
            self.assertEqual(review["windows"][0]["failed_count"], 1)
            self.assertIn("locator_failed_likely", review["windows"][0]["hint_tags"])
            self.assertGreater(review["windows"][0]["max_gap_chars"], 0)

            second_review = inspector.span_review(document_id="doc", window_range="1-1", limit=10)
            self.assertEqual(second_review["summary"]["window_count"], 1)
            self.assertEqual(second_review["windows"][0]["global_window_index"], 1)
            self.assertEqual(second_review["windows"][0]["window_id"], second_window.window_id)

            span_only_review = inspector.span_review(
                document_id="doc",
                window_range="0-1",
                with_spans_only=True,
                limit=10,
            )
            self.assertEqual(span_only_review["summary"]["window_count"], 1)
            self.assertEqual(span_only_review["windows"][0]["window_id"], window.window_id)

            detail = inspector.span_review_window(window.window_id)
            self.assertEqual(detail["audit"]["located_count"], 2)
            self.assertTrue(any(not segment["span_ids"] for segment in detail["segments"]))
            self.assertTrue(any(segment["span_ids"] for segment in detail["segments"]))
            self.assertIn("开头闲笔", detail["gaps"][0]["text_preview"])


if __name__ == "__main__":
    unittest.main()
