"""Extraction retry helpers for M1.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2

    @property
    def max_attempts(self) -> int:
        return max(1, self.max_retries + 1)


def run_with_retries(operation: Callable[[int], T], policy: RetryPolicy) -> T:
    """Run an operation with simple numbered attempts.

    The operation receives the 1-based attempt number. Exceptions are retried
    until the policy is exhausted, then the last exception is raised.
    """
    last_error: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation(attempt)
        except Exception as error:  # noqa: BLE001 - callers need raw validation/API errors.
            last_error = error
    if last_error is None:
        raise RuntimeError("retry operation did not run")
    raise last_error
