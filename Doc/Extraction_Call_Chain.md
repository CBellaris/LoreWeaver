# Extraction 调用链

## 顶层入口

`extract_document_windows` 是对 CLI/Web 暴露的主入口，负责：

1. 打开 SQLite，并初始化 extraction tables。
2. 读取 document 和 candidate windows。
3. 根据 `window_id`、`window_ids`、`window_ranges`、`offset`、`limit` 选择窗口。
4. 解析 extraction / locator / model 配置。
5. 分流到 live 模式或 batch 模式。
6. 汇总 report，并写入 runs 目录和 SQLite。

## Live 模式

```text
extract_document_windows
  -> extract_window
     -> build_extraction_messages
     -> ChatClient.complete
     -> _parse_payload
     -> _results_from_window_payload
        -> anchor_constraints_ok
        -> locate_span_anchors
        -> _span_from_payload
  -> _persist_window_results
  -> _build_report
  -> _persist_extraction_report
```

## Batch 模式

```text
extract_document_windows
  -> _run_batch_extraction
     -> _write_batch_input_file
        -> _build_batch_request_line
     -> client.submit_chat_batch
     -> _wait_for_batch_status
     -> client.download_file_text
     -> _parse_batch_output_lines
        -> _batch_output_from_line
     -> _apply_batch_outputs
        -> _results_from_raw_window_output
        -> _persist_window_results
     -> _run_batch_retry_round
     -> _retry_windows_live
        -> extract_window
        -> _persist_window_results
     -> _record_deferred_batch_retry_windows
  -> _build_report
  -> _persist_extraction_report
```

## 模块分工

- `extractor.py`: 外部兼容入口和 live/batch 分流编排。
- `batch.py`: batch 提交、等待、下载、应用输出、重试编排。
- `window_processing.py`: 单窗口 LLM 输出解析、定位、Span 构建。
- `persistence.py`: live 和 batch 共用的窗口结果持久化。
- `reports.py`: extraction report 构建和写入。
- `selection.py`: window id/range 选择。
- `settings.py`: extraction model 配置解析。
- `usage.py`: token 估算和费用计算。
- `types.py`: extraction 运行期 dataclass 和 protocol。
- `mock_client.py`: dry-run/test 用 mock LLM client。

## 清理结果

- 删除了 extraction 内部未使用的 `_optional_str`。
- 收敛了三处重复的 span/failure/locator/uncovered-text 写入逻辑。
- `extractor.py` 从 2158 行瘦身到约 380 行，同时保留原 import 路径的兼容 re-export。
