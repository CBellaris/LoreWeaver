"""Configuration loading for LoreWeaver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from ast import literal_eval
from typing import Any


@dataclass(frozen=True)
class AppConfig:
    path: Path
    values: dict[str, Any]

    @property
    def data_dir(self) -> Path:
        data_dir = self.values.get("project", {}).get("data_dir", "data")
        return Path(data_dir)

    @property
    def sample_source_path(self) -> Path | None:
        source_path = self.values.get("sample", {}).get("source_path")
        return Path(source_path) if source_path else None


def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = handle.read()

    try:
        import yaml
    except ImportError:
        values = _load_simple_yaml(raw_config)
    else:
        values = yaml.safe_load(raw_config) or {}

    if not isinstance(values, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    return AppConfig(path=config_path, values=values)


def _load_simple_yaml(raw_config: str) -> dict[str, Any]:
    """Load the small YAML subset used by the bootstrap config.

    This fallback keeps M1.0 runnable before project dependencies are installed.
    It supports nested mappings, simple lists, quoted strings, numbers, booleans,
    and nulls. Full YAML support still comes from PyYAML once dependencies exist.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any, Any, str | None]] = [(-1, root, None, None)]

    for raw_line in raw_config.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        while indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]

        if stripped.startswith("- "):
            if not isinstance(parent, list):
                parent_ref = stack[-1][2]
                parent_key = stack[-1][3]
                if isinstance(parent, dict) and not parent and isinstance(parent_ref, dict) and parent_key:
                    replacement: list[Any] = []
                    parent_ref[parent_key] = replacement
                    stack[-1] = (stack[-1][0], replacement, parent_ref, parent_key)
                    parent = replacement
                else:
                    raise ValueError(f"Unsupported YAML list location: {raw_line}")

            parent.append(_parse_scalar(stripped[2:].strip()))
            continue

        key, separator, value = stripped.partition(":")
        if not separator:
            raise ValueError(f"Unsupported YAML line: {raw_line}")

        key = key.strip()
        value = value.strip()

        if value:
            if not isinstance(parent, dict):
                raise ValueError(f"Cannot assign mapping key below list item: {raw_line}")
            parent[key] = _parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            if not isinstance(parent, dict):
                raise ValueError(f"Cannot create mapping below list item: {raw_line}")
            parent[key] = child
            stack.append((indent, child, parent, key))

    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None

    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        try:
            return literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value
