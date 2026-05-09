from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sqlite3
import unittest

from loreweaver.config import AppConfig
from loreweaver.extraction.extractor import (
    MockChatClient,
    TokenPrice,
    _apply_batch_outputs,
    _batch_output_from_line,
    _build_batch_request_line,
    _results_from_raw_window_output,
    build_uncovered_text,
    estimate_cost,
    extract_document_windows,
    extract_window,
    list_extraction_windows,
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


class SequencedJsonClient:
    def __init__(self, raw_outputs: list[str]) -> None:
        self.raw_outputs = list(raw_outputs)
        self.calls = 0

    def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> tuple[str, dict[str, int]]:
        del messages, model, temperature
        self.calls += 1
        return self.raw_outputs.pop(0), {"input_tokens": 7, "output_tokens": 8, "total_tokens": 15}


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

    def test_batch_request_line_uses_window_id_and_batch_model(self) -> None:
        window = CandidateWindow(
            window_id="doc_ch0001_win0001",
            document_id="doc",
            chapter_id="doc_ch0001",
            window_index=1,
            window_start=0,
            window_end=20,
            text="高文发现旧城堡里有异常魔力。",
        )

        line = _build_batch_request_line(
            window=window,
            model="deepseek-ai/DeepSeek-V3.1-Terminus",
            temperature=0,
            min_spans_per_window=1,
            max_spans_per_window=3,
            anchor_min_chars=4,
            anchor_max_chars=80,
            target_span_chars_min=4,
            target_span_chars_max=80,
            json_response_format=True,
        )

        self.assertEqual(line["custom_id"], window.window_id)
        self.assertEqual(line["method"], "POST")
        self.assertEqual(line["url"], "/v1/chat/completions")
        self.assertEqual(line["body"]["model"], "deepseek-ai/DeepSeek-V3.1-Terminus")
        self.assertEqual(line["body"]["response_format"], {"type": "json_object"})
        self.assertTrue(line["body"]["messages"])

    def test_batch_output_can_be_parsed_and_located(self) -> None:
        text = "高文醒来后发现自己站在陌生大厅中央，墙上的魔法阵仍在微光闪烁。"
        window = CandidateWindow(
            window_id="doc_ch0001_win0001",
            document_id="doc",
            chapter_id="doc_ch0001",
            window_index=1,
            window_start=0,
            window_end=len(text),
            text=text,
        )
        raw_payload = {
            "spans": [
                {
                    "micro_topic": "陌生大厅苏醒",
                    "span_type": "event",
                    "micro_summary": "高文醒来并察觉大厅中的魔法阵。",
                    "entities": ["高文", "魔法阵"],
                    "topics": ["苏醒", "魔法"],
                    "salience_score": 0.6,
                    "start_anchor_quote": "高文醒来后发现自己站在陌生大厅中央",
                    "end_anchor_quote": "墙上的魔法阵仍在微光闪烁",
                }
            ]
        }
        line = {
            "custom_id": window.window_id,
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": json.dumps(raw_payload, ensure_ascii=False)}}],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 13,
                        "total_tokens": 24,
                    },
                },
            },
        }

        output = _batch_output_from_line(line)
        results = _results_from_raw_window_output(
            window,
            raw_output=output.raw_output or "",
            attempts=1,
            usage=output.usage,
            token_price=TokenPrice(input_yuan_per_1k=0.002, output_yuan_per_1k=0.003),
            max_spans_per_window=12,
            anchor_min_chars=4,
            anchor_max_chars=80,
            target_span_chars_min=4,
            target_span_chars_max=120,
            store_located_text=True,
            fuzzy_threshold=0.86,
        )

        self.assertIsNone(output.error)
        self.assertEqual(output.usage["total_tokens"], 24)
        self.assertEqual(results[0].status, "located")
        self.assertIn("魔法阵", results[0].span.located_text)

    def test_batch_parse_failure_retries_window_live(self) -> None:
        with TemporaryDirectory() as tmpdir:
            text = "高文醒来后发现自己站在陌生大厅中央，墙上的魔法阵仍在微光闪烁。"
            window = CandidateWindow(
                window_id="doc_ch0001_win0001",
                document_id="doc",
                chapter_id="doc_ch0001",
                window_index=1,
                window_start=0,
                window_end=len(text),
                text=text,
            )
            bad_payload = {
                "spans": [
                    {
                        "micro_topic": "陌生大厅苏醒",
                        "span_type": "event",
                        "micro_summary": "高文醒来并察觉大厅中的魔法阵。",
                        "entities": ["高文", "魔法阵"],
                        "topics": ["苏醒", "魔法"],
                        "salience_score": 0.6,
                        "start_anchor_quote": "高文醒来后发现自己站在陌生大厅中央",
                        "end_quote": "墙上的魔法阵仍在微光闪烁",
                    }
                ]
            }
            retry_payload = {
                "spans": [
                    {
                        "micro_topic": "陌生大厅苏醒",
                        "span_type": "event",
                        "micro_summary": "高文醒来并察觉大厅中的魔法阵。",
                        "entities": ["高文", "魔法阵"],
                        "topics": ["苏醒", "魔法"],
                        "salience_score": 0.6,
                        "start_anchor_quote": "高文醒来后发现自己站在陌生大厅中央",
                        "end_anchor_quote": "墙上的魔法阵仍在微光闪烁",
                    }
                ]
            }
            output = _batch_output_from_line(
                {
                    "custom_id": window.window_id,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {"message": {"content": json.dumps(bad_payload, ensure_ascii=False)}}
                            ],
                            "usage": {
                                "prompt_tokens": 11,
                                "completion_tokens": 13,
                                "total_tokens": 24,
                            },
                        },
                    },
                }
            )
            client = SequencedJsonClient([json.dumps(retry_payload, ensure_ascii=False)])
            store = SQLiteStore(Path(tmpdir) / "test.sqlite3")
            store.initialize()
            store.initialize_extraction_tables()

            results = _apply_batch_outputs(
                store=store,
                windows=[window],
                outputs=[output],
                client=client,
                model="mock",
                temperature=0,
                retry_policy=RetryPolicy(max_retries=0),
                min_spans_per_window=1,
                max_spans_per_window=12,
                anchor_min_chars=4,
                anchor_max_chars=80,
                target_span_chars_min=4,
                target_span_chars_max=120,
                store_located_text=True,
                store_uncovered_text=True,
                fuzzy_threshold=0.86,
                token_price=TokenPrice(input_yuan_per_1k=0.002, output_yuan_per_1k=0.003),
                progress=None,
            )

            self.assertEqual(client.calls, 1)
            self.assertEqual(results[0].status, "located")
            self.assertIn("魔法阵", results[0].span.located_text)
            self.assertEqual(results[0].usage["total_tokens"], 39)

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

            status_report = list_extraction_windows(
                storage_config=storage_config,
                document_id=ingest_report["document"]["document_id"],
                only="all",
            )
            self.assertEqual(status_report["windows"][0]["status"], "extracted")
            self.assertEqual(status_report["windows"][1]["status"], "extracted")
            self.assertEqual(status_report["windows"][2]["status"], "pending")

            range_report = extract_document_windows(
                config=config,
                storage_config=storage_config,
                models_config=models_config,
                run_id="extract_range_test",
                document_id=ingest_report["document"]["document_id"],
                window_ranges=["3-3"],
                mock=True,
            )
            self.assertEqual(range_report["window_count"], 1)
            self.assertEqual(range_report["span_count"], 2)

            window_id = status_report["windows"][0]["window_id"]
            rerun_report = extract_document_windows(
                config=config,
                storage_config=storage_config,
                models_config=models_config,
                run_id="extract_window_test",
                document_id=ingest_report["document"]["document_id"],
                window_ids=[window_id],
                mock=True,
            )
            self.assertEqual(rerun_report["window_count"], 1)
            self.assertEqual(rerun_report["span_count"], 2)

            final_spans = store.list_spans(
                ingest_report["document"]["document_id"],
                located_only=True,
            )
            self.assertEqual(len(final_spans), 6)

    def test_cost_estimate_uses_configured_prices(self) -> None:
        cost = estimate_cost(
            {"input_tokens": 1500, "output_tokens": 500},
            TokenPrice(input_yuan_per_1k=0.002, output_yuan_per_1k=0.003),
        )

        self.assertEqual(cost.estimated_yuan, 0.0045)


if __name__ == "__main__":
    unittest.main()
