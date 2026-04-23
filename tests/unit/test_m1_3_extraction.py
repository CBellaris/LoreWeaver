from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sqlite3
import unittest

from loreweaver.config import AppConfig
from loreweaver.extraction.extractor import (
    MockChatClient,
    TokenPrice,
    build_uncovered_text,
    estimate_cost,
    extract_document_windows,
    extract_window,
)
from loreweaver.extraction.locator import locate_quote, locate_span_anchors
from loreweaver.extraction.retry import RetryPolicy
from loreweaver.ingest.pipeline import ingest_text
from loreweaver.ingest.window_splitter import build_candidate_windows
from loreweaver.models.window import CandidateWindow
from loreweaver.storage.sqlite_store import SQLiteStore


class StaticJsonClient:
    def __init__(self, raw_output: str) -> None:
        self.raw_output = raw_output

    def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> tuple[str, dict[str, int]]:
        del messages, model, temperature
        return self.raw_output, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}


class M13ExtractionTests(unittest.TestCase):
    def test_locator_handles_exact_normalized_and_fuzzy_quotes(self) -> None:
        window = CandidateWindow(
            window_id="w1",
            document_id="doc",
            chapter_id="ch1",
            window_index=1,
            window_start=100,
            window_end=170,
            text="他说：“魔法来自星空。”\n这句话后来被反复引用。",
        )

        exact = locate_quote(window, "“魔法来自星空。”")
        normalized = locate_quote(window, '"魔法来自星空。" 这句话后来被反复引用')
        fuzzy = locate_quote(window, "魔法来自星空。这句话后来被反复引用")

        self.assertEqual(exact.status, "located")
        self.assertEqual(window.text[exact.start_idx - 100 : exact.end_idx - 100], "“魔法来自星空。”")
        self.assertEqual(normalized.status, "located")
        self.assertEqual(fuzzy.status, "located")
        self.assertGreaterEqual(fuzzy.confidence, 0.86)

    def test_anchor_locator_returns_micro_span_interval(self) -> None:
        text = (
            "主角B一进入房间，主角A便开口说道：“今天天气不错。"
            "XX森林你知道吗？那里有古老遗迹，也有危险的雾。”"
        )
        window = CandidateWindow(
            window_id="w1",
            document_id="doc",
            chapter_id="ch1",
            window_index=1,
            window_start=100,
            window_end=100 + len(text),
            text=text,
        )

        relationship = locate_span_anchors(
            window,
            start_anchor_quote="主角B一进入房间，主角A便开口说道",
            end_anchor_quote="也有危险的雾。”",
            min_span_chars=20,
            max_span_chars=120,
        )
        location = locate_span_anchors(
            window,
            start_anchor_quote="XX森林你知道吗？",
            end_anchor_quote="也有危险的雾",
            min_span_chars=10,
            max_span_chars=60,
        )

        self.assertEqual(relationship.status, "located")
        self.assertEqual(location.status, "located")
        self.assertLessEqual(relationship.start_idx, location.start_idx)
        self.assertGreaterEqual(relationship.end_idx, location.end_idx)

    def test_extract_window_trims_overlong_anchors_for_location(self) -> None:
        text = (
            "随后她回过头，打量着身边仅剩的几个人：三名士兵正在举着火把警戒四周，"
            "赫蒂姑妈则手托着一个燃烧的火球认真打量着石厅尽头的墙壁，"
            "算上她自己和拜伦骑士，眼下这七个人恐怕就是最后的幸存者了。"
        )
        window = CandidateWindow(
            window_id="doc_ch0002_win0001",
            document_id="doc",
            chapter_id="doc_ch0002",
            window_index=1,
            window_start=0,
            window_end=len(text),
            text=text,
        )
        raw_output = json.dumps(
            {
                "spans": [
                    {
                        "micro_topic": "幸存者队伍成员状态",
                        "span_type": "scene_action",
                        "micro_summary": "瑞贝卡清点身边幸存者，确认当前只剩七人。",
                        "entities": ["瑞贝卡", "拜伦骑士", "赫蒂姑妈", "士兵"],
                        "topics": ["队伍构成", "幸存者"],
                        "salience_score": 0.5,
                        "start_anchor_quote": (
                            "随后她回过头，打量着身边仅剩的几个人："
                            "三名士兵正在举着火把警戒四周，"
                            "赫蒂姑妈则手托着一个燃烧的火球认真打量着石厅尽头的墙壁"
                        ),
                        "end_anchor_quote": (
                            "算上她自己和拜伦骑士，"
                            "眼下这七个人恐怕就是最后的幸存者了。"
                        ),
                        "key_quote": "这七个人恐怕就是最后的幸存者了",
                        "overlap_reason": "",
                    }
                ]
            },
            ensure_ascii=False,
        )

        results = extract_window(
            window,
            client=StaticJsonClient(raw_output),
            model="mock",
            temperature=0,
            retry_policy=RetryPolicy(max_retries=0),
            min_spans_per_window=1,
            max_spans_per_window=12,
            anchor_min_chars=8,
            anchor_max_chars=80,
            target_span_chars_min=20,
            target_span_chars_max=220,
            store_located_text=True,
            fuzzy_threshold=0.86,
            token_price=TokenPrice(input_yuan_per_1k=0.002, output_yuan_per_1k=0.003),
        )

        self.assertEqual(results[0].status, "located")
        self.assertIn("这七个人恐怕就是最后的幸存者", results[0].span.located_text)

    def test_uncovered_text_merges_fragments_outside_located_spans(self) -> None:
        text = "开头闲笔。主角发现古门。中间过渡。古门发光并显出符文。结尾闲笔。"
        window = CandidateWindow(
            window_id="doc_ch0001_win0001",
            document_id="doc",
            chapter_id="doc_ch0001",
            window_index=1,
            window_start=100,
            window_end=100 + len(text),
            text=text,
        )
        raw_output = json.dumps(
            {
                "spans": [
                    {
                        "micro_topic": "主角发现古门",
                        "span_type": "event",
                        "micro_summary": "主角发现一扇古门。",
                        "entities": ["主角", "古门"],
                        "topics": ["发现"],
                        "salience_score": 0.6,
                        "start_anchor_quote": "主角发现古门",
                        "end_anchor_quote": "主角发现古门",
                    },
                    {
                        "micro_topic": "古门显出符文",
                        "span_type": "mystery_clue",
                        "micro_summary": "古门发光并显出符文。",
                        "entities": ["古门", "符文"],
                        "topics": ["伏笔"],
                        "salience_score": 0.7,
                        "start_anchor_quote": "古门发光",
                        "end_anchor_quote": "显出符文",
                    },
                ]
            },
            ensure_ascii=False,
        )
        results = extract_window(
            window,
            client=StaticJsonClient(raw_output),
            model="mock",
            temperature=0,
            retry_policy=RetryPolicy(max_retries=0),
            min_spans_per_window=1,
            max_spans_per_window=12,
            anchor_min_chars=4,
            anchor_max_chars=80,
            target_span_chars_min=4,
            target_span_chars_max=80,
            store_located_text=True,
            fuzzy_threshold=0.86,
            token_price=TokenPrice(input_yuan_per_1k=0.002, output_yuan_per_1k=0.003),
        )

        uncovered_text = build_uncovered_text(window, [result.span for result in results])

        self.assertIn("开头闲笔", uncovered_text)
        self.assertIn("中间过渡", uncovered_text)
        self.assertIn("结尾闲笔", uncovered_text)
        self.assertNotIn("主角发现古门", uncovered_text)
        self.assertNotIn("古门发光并显出符文", uncovered_text)

    def test_extract_window_with_mock_returns_multiple_located_spans_and_cost(self) -> None:
        text = "第一章 A\n" + "高文在陌生的大厅中醒来，并意识到这座城堡隐藏着旧时代留下的秘密。" * 3
        window = CandidateWindow(
            window_id="doc_ch0001_win0001",
            document_id="doc",
            chapter_id="doc_ch0001",
            window_index=1,
            window_start=0,
            window_end=len(text),
            text=text,
        )

        results = extract_window(
            window,
            client=MockChatClient(),
            model="mock",
            temperature=0,
            retry_policy=RetryPolicy(max_retries=0),
            min_spans_per_window=2,
            max_spans_per_window=12,
            anchor_min_chars=8,
            anchor_max_chars=80,
            target_span_chars_min=20,
            target_span_chars_max=180,
            store_located_text=True,
            fuzzy_threshold=0.86,
            token_price=TokenPrice(input_yuan_per_1k=0.002, output_yuan_per_1k=0.003),
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.status == "located" for result in results))
        self.assertEqual(results[0].span.span_index_in_window, 1)
        self.assertEqual(results[1].span.span_index_in_window, 2)
        self.assertIsNotNone(results[0].span.span_start_idx)
        self.assertTrue(results[0].span.located_text)
        self.assertGreater(results[0].cost.estimated_yuan, 0)
        self.assertEqual(results[1].cost.estimated_yuan, 0)

    def test_extraction_pipeline_persists_spans_failures_and_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "raw.txt"
            source.write_text(
                "第一章 A\n"
                + ("高文在陌生的大厅中醒来，并意识到这座城堡隐藏着旧时代留下的秘密。" * 35)
                + "\n第二章 B\n"
                + ("瑞贝卡提到领地、魔网和家族危机，这些信息共同构成早期世界观线索。" * 30),
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
                        ],
                    },
                    "window": {
                        "size_chars": 500,
                        "overlap_ratio": 0.2,
                        "min_chars": 120,
                        "max_chars": 800,
                    },
                    "extraction": {
                        "model": "mock",
                        "temperature": 0,
                        "max_retries": 0,
                        "target_span_chars_min": 20,
                        "target_span_chars_max": 700,
                        "anchor_min_chars": 8,
                        "anchor_max_chars": 80,
                        "input_yuan_per_1k": 0.002,
                        "output_yuan_per_1k": 0.003,
                    },
                    "locator": {"fuzzy_threshold": 0.86},
                },
            )
            storage_config = AppConfig(
                path=root / "storage.yaml",
                values={"sqlite": {"path": str(root / "data" / "runs" / "test.sqlite3")}},
            )
            models_config = AppConfig(
                path=root / "models.yaml",
                values={
                    "providers": {"mock": {"api_key_env": "MOCK_API_KEY"}},
                    "models": {
                        "extraction": {
                            "provider": "mock",
                            "name": "mock",
                            "temperature": 0,
                            "input_yuan_per_1k": 0.002,
                            "output_yuan_per_1k": 0.003,
                        }
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
            extraction_report = extract_document_windows(
                config=config,
                storage_config=storage_config,
                models_config=models_config,
                run_id="extract_test",
                document_id=ingest_report["document"]["document_id"],
                limit=2,
                mock=True,
            )

            self.assertTrue(Path(extraction_report["report_path"]).exists())
            self.assertEqual(extraction_report["window_count"], 2)
            self.assertEqual(extraction_report["span_count"], 4)
            self.assertEqual(extraction_report["locator_success_count"], 4)

            store = SQLiteStore(storage_config.sqlite_path)
            spans = store.list_spans(ingest_report["document"]["document_id"], located_only=True)
            with sqlite3.connect(storage_config.sqlite_path) as connection:
                report_count = connection.execute(
                    "SELECT COUNT(*) FROM extraction_reports"
                ).fetchone()[0]
                candidate_count = connection.execute(
                    "SELECT COUNT(*) FROM locator_candidates"
                ).fetchone()[0]
                uncovered_count = connection.execute(
                    """
                    SELECT COUNT(*) FROM candidate_windows
                    WHERE document_id = ? AND uncovered_text != ''
                    """,
                    (ingest_report["document"]["document_id"],),
                ).fetchone()[0]

            self.assertEqual(len(spans), 4)
            self.assertEqual(report_count, 1)
            self.assertGreaterEqual(candidate_count, 4)
            self.assertGreaterEqual(uncovered_count, 1)
            self.assertTrue(all(span.micro_topic for span in spans))
            self.assertTrue(all(span.micro_summary for span in spans))
            self.assertTrue(all(span.located_text for span in spans))

    def test_cost_estimate_uses_configured_prices(self) -> None:
        cost = estimate_cost(
            {"input_tokens": 1500, "output_tokens": 500},
            TokenPrice(input_yuan_per_1k=0.002, output_yuan_per_1k=0.003),
        )

        self.assertEqual(cost.estimated_yuan, 0.0045)


if __name__ == "__main__":
    unittest.main()
