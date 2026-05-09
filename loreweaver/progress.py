"""Structured progress events shared by CLI and web runners."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, TextIO


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ProgressEvent:
    name: str
    command: str
    stage: str
    label: str
    run_id: str | None = None
    current: float | int | None = None
    total: float | int | None = None
    unit: str | None = None
    status: str = "running"
    message: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now)

    @property
    def percent(self) -> float | None:
        if self.current is None or self.total in (None, 0):
            return None
        return round(max(0.0, min(100.0, (float(self.current) / float(self.total)) * 100.0)), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.name,
            "command": self.command,
            "run_id": self.run_id,
            "stage": self.stage,
            "label": self.label,
            "current": self.current,
            "total": self.total,
            "unit": self.unit,
            "percent": self.percent,
            "status": self.status,
            "message": self.message,
            "detail": self.detail,
            "time": self.timestamp,
        }


class ProgressSink(Protocol):
    def emit(self, event: ProgressEvent) -> None:
        """Consume one progress event."""


class ProgressReporter:
    def __init__(
        self,
        *,
        command: str,
        run_id: str | None = None,
        sinks: list[ProgressSink] | None = None,
    ) -> None:
        self.command = command
        self.run_id = run_id
        self._sinks = list(sinks or [])

    def emit(
        self,
        name: str,
        *,
        stage: str,
        label: str,
        current: float | int | None = None,
        total: float | int | None = None,
        unit: str | None = None,
        status: str = "running",
        message: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> ProgressEvent:
        event = ProgressEvent(
            name=name,
            command=self.command,
            run_id=self.run_id,
            stage=stage,
            label=label,
            current=current,
            total=total,
            unit=unit,
            status=status,
            message=message,
            detail=detail or {},
        )
        for sink in self._sinks:
            sink.emit(event)
        return event

    def child(self, *, command: str | None = None, run_id: str | None = None) -> "ProgressReporter":
        return ProgressReporter(
            command=command or self.command,
            run_id=run_id or self.run_id,
            sinks=self._sinks,
        )


class TextProgressSink:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stderr

    def emit(self, event: ProgressEvent) -> None:
        progress = ""
        if event.current is not None and event.total is not None:
            unit = f" {event.unit}" if event.unit else ""
            progress = f" {event.current:g}/{event.total:g}{unit}"
            if event.percent is not None:
                progress += f" {event.percent:.1f}%"
        detail = ""
        if event.message:
            detail = f" - {event.message}"
        print(
            f"[{event.command}] {event.stage}: {event.label}{progress}{detail}",
            file=self.stream,
            flush=True,
        )


class JsonlProgressSink:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stderr

    def emit(self, event: ProgressEvent) -> None:
        print(json.dumps(event.to_dict(), ensure_ascii=False), file=self.stream, flush=True)


class RichProgressSink:
    """Rich-backed CLI renderer with graceful text fallback when Rich is unavailable."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._fallback: TextProgressSink | None = None
        try:
            from rich.console import Console
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )
        except ImportError:
            self._fallback = TextProgressSink(stream=stream)
            self._progress = None
            self._task_ids: dict[str, int] = {}
            return

        self._console = Console(file=stream or sys.stderr, stderr=True)
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.fields[stage]}[/bold]"),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self._console,
            transient=False,
        )
        self._task_ids: dict[str, int] = {}
        self._progress.start()

    def emit(self, event: ProgressEvent) -> None:
        if self._fallback is not None or self._progress is None:
            assert self._fallback is not None
            self._fallback.emit(event)
            return

        key = event.stage
        total = float(event.total) if event.total is not None else None
        completed = float(event.current) if event.current is not None else 0.0
        if key not in self._task_ids:
            self._task_ids[key] = self._progress.add_task(
                event.label,
                total=total,
                completed=completed,
                stage=event.stage,
            )
        else:
            self._progress.update(
                self._task_ids[key],
                description=event.label,
                total=total,
                completed=completed,
                stage=event.stage,
            )
        if event.status in {"completed", "failed", "cancelled"}:
            self._progress.update(self._task_ids[key], completed=total or completed)

    def close(self) -> None:
        if self._progress is not None:
            self._progress.stop()


class NullProgressSink:
    def emit(self, event: ProgressEvent) -> None:
        del event


def build_cli_progress_reporter(
    *,
    command: str,
    run_id: str | None,
    mode: str,
) -> tuple[ProgressReporter | None, Any | None]:
    if mode == "none":
        return None, None
    if mode == "jsonl":
        sink: ProgressSink = JsonlProgressSink()
    elif mode == "rich":
        sink = RichProgressSink()
    elif mode == "text":
        sink = TextProgressSink()
    elif mode == "auto":
        sink = RichProgressSink() if sys.stderr.isatty() else TextProgressSink()
    else:
        raise ValueError(f"Unsupported progress mode: {mode}")
    return ProgressReporter(command=command, run_id=run_id, sinks=[sink]), sink
