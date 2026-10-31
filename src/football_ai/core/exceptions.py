"""Exception hierarchy shared by all modules."""

from __future__ import annotations

from typing import Any


class FootballAIError(Exception):
    """Base class of every application error."""


class ConfigurationError(FootballAIError):
    """Invalid or missing configuration."""


class DataProviderError(FootballAIError):
    """Base class for errors raised by external data providers."""


class MissingApiKeyError(DataProviderError, ConfigurationError):
    """The provider API key is not configured."""


class AuthenticationError(DataProviderError):
    """The provider rejected the credentials (invalid / suspended key)."""


class QuotaExceededError(DataProviderError):
    """The request would exceed (or has exceeded) the allowed quota."""


class RateLimitError(DataProviderError):
    """Per-minute rate limit hit and retries exhausted."""


class PlanRestrictionError(DataProviderError):
    """The subscription plan does not give access to the requested data."""


class ApiRequestError(DataProviderError):
    """The provider reported an error for the request (bad parameters...)."""

    def __init__(self, message: str, errors: Any = None, status_code: int | None = None) -> None:
        super().__init__(message)
        self.errors = errors
        self.status_code = status_code


class ApiServerError(DataProviderError):
    """Server-side (5xx) error after retries were exhausted."""


class ApiNetworkError(DataProviderError):
    """Network / timeout error after retries were exhausted."""


class ApiResponseFormatError(DataProviderError):
    """The response body is not in the expected format."""
