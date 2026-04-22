"""Raw text reading utilities for M1.1."""

from __future__ import annotations

from pathlib import Path


SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


def read_text_file(path: str | Path) -> tuple[str, str]:
    """Read a raw text file with a small, deterministic encoding fallback chain."""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source text not found: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Source path is not a file: {source_path}")

    raw_bytes = source_path.read_bytes()
    last_error: UnicodeDecodeError | None = None
    for encoding in SUPPORTED_ENCODINGS:
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise UnicodeDecodeError(
            last_error.encoding,
            last_error.object,
            last_error.start,
            last_error.end,
            f"Unable to decode {source_path} with {SUPPORTED_ENCODINGS}",
        ) from last_error

    return "", "utf-8"
