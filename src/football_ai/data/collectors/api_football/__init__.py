"""API-Football (api-sports.io v3) connector."""

from football_ai.data.collectors.api_football.client import ApiFootballClient, ApiResponse
from football_ai.data.collectors.api_football.provider import ApiFootballProvider
from football_ai.data.collectors.api_football.quota import QuotaManager, QuotaSnapshot

__all__ = [
    "ApiFootballClient",
    "ApiFootballProvider",
    "ApiResponse",
    "QuotaManager",
    "QuotaSnapshot",
]
