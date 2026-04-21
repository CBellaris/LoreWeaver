"""Small logging helpers shared by CLI commands."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4


def new_run_id(prefix: str = "run") -> str:
    """Create a readable run id for reproducible command outputs."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}"


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

