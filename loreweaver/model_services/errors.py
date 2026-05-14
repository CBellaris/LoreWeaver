"""Model service error types."""

from __future__ import annotations


class ModelServiceError(RuntimeError):
    """Base class for model service failures."""


class MissingApiKeyError(ModelServiceError, ValueError):
    """Raised when a configured provider has no available API key."""


class MissingDependencyError(ModelServiceError):
    """Raised when an optional provider dependency is not installed."""


class EmptyModelResponse(ModelServiceError):
    """Raised when a provider returns no usable model output."""


class InvalidJsonResponse(ModelServiceError, ValueError):
    """Raised when a model response cannot be parsed as a JSON object."""
