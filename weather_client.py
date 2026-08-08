"""
Client for the National Weather Service (NWS) API.

Fetches active alerts and narrative forecasts for a list of locations and
normalizes them into documents ready for Lakebase.
"""

import hashlib
import os
import re
from datetime import UTC, datetime
from typing import Any

import requests

_NWS_BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov")
_GEOCODER_BASE_URL = os.environ.get(
    "GEOCODER_BASE_URL",
    "https://nominatim.openstreetmap.org",
)
_DEFAULT_TIMEOUT = 30

_COORDINATES_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$"
)


class WeatherClient:
    """HTTP client for NWS alerts and detailed forecasts."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        self.base_url = (base_url or _NWS_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

        # NWS asks API clients to identify themselves.
        self._session.headers.update(
            {
                "User-Agent": os.environ.get(
                    "NWS_USER_AGENT",
                    "weather-rag-app/1.0 (student@example.com)",
                ),
                "Accept": "application/geo+json, application/json",
            }
        )

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """GET JSON from api.weather.gov."""
        response = self._session.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def resolve_location(self, location: str) -> dict[str, Any]:
        """
        Convert either 'Chicago, IL' or '41.8781,-87.6298' into coordinates.

        NWS needs latitude/longitude. City/state names are resolved through
        OpenStreetMap Nominatim before calling NWS.
        """
        location = location.strip()
        match = _COORDINATES_RE.match(location)

        if match:
            return {
                "location": location,
                "latitude": float(match.group(1)),
                "longitude": float(match.group(2)),
            }

        response = self._session.get(
            f"{_GEOCODER_BASE_URL.rstrip('/')}/search",
            params={
                "q": location,
                "format": "jsonv2",
                "limit": 1,
                "countrycodes": "us",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        results = response.json()

        if not results:
            raise ValueError(f"Location not found: {location}")

        result = results[0]
        return {
            "location": location,
            "latitude": float(result["lat"]),
            "longitude": float(result["lon"]),
        }

    def get_gridpoint(self, latitude: float, longitude: float) -> dict[str, Any]:
        """Resolve NWS grid point via GET /points/{lat},{lon}."""
        return self.get(f"/points/{latitude},{longitude}")

    def get_active_alerts(
        self,
        latitude: float,
        longitude: float,
    ) -> list[dict[str, Any]]:
        """Fetch active alerts affecting one coordinate pair."""
        data = self.get(
            "/alerts/active",
            params={"point": f"{latitude},{longitude}"},
        )
        return data.get("features", [])

    def get_forecast_periods(self, forecast_url: str) -> list[dict[str, Any]]:
        """Fetch detailed narrative forecast periods from NWS."""
        response = self._session.get(
            forecast_url,
            timeout=self.timeout,
        )
        response.raise_for_status()

        return response.json().get("properties", {}).get("periods", [])

    def fetch_documents_for_location(
        self,
        location: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch and normalize alert and forecast documents for one location."""
        place = self.resolve_location(location)
        latitude = place["latitude"]
        longitude = place["longitude"]
        synced_at = datetime.now(UTC).isoformat()

        documents: list[dict[str, Any]] = []

        # NWS grid metadata includes the forecast endpoint for this point.
        gridpoint = self.get_gridpoint(latitude, longitude)

        # Active alerts -> source_type = alert
        for feature in self.get_active_alerts(latitude, longitude):
            properties = feature.get("properties", {})

            narrative_text = "\n\n".join(
                value
                for value in (
                    properties.get("description"),
                    properties.get("instruction"),
                )
                if value
            )

            if not narrative_text:
                continue

            documents.append(
                {
                    "id": str(feature["id"]),  # stable NWS alert URL/id
                    "location": place["location"],
                    "source_type": "alert",
                    "headline": (
                        properties.get("headline")
                        or properties.get("event")
                        or "Weather alert"
                    ),
                    "narrative_text": narrative_text,
                    "issued_at": properties.get("sent"),
                    "effective_at": properties.get("effective"),
                    "payload": feature,
                    "synced_at": synced_at,
                }
            )

        # Forecast periods -> source_type = forecast
        forecast_url = gridpoint.get("properties", {}).get("forecast")
        if forecast_url:
            for period in self.get_forecast_periods(forecast_url):
                narrative_text = (
                    period.get("detailedForecast")
                    or period.get("shortForecast")
                )
                if not narrative_text:
                    continue

                # Hash is stable across repeat syncs of the same forecast period.
                dedup_input = "|".join(
                    (
                        place["location"],
                        str(period.get("number")),
                        str(period.get("startTime")),
                    )
                )
                document_id = hashlib.sha256(
                    dedup_input.encode("utf-8")
                ).hexdigest()

                documents.append(
                    {
                        "id": f"forecast:{document_id}",
                        "location": place["location"],
                        "source_type": "forecast",
                        "headline": period.get("name", "Weather forecast"),
                        "narrative_text": narrative_text,
                        "issued_at": period.get("startTime"),
                        "effective_at": period.get("startTime"),
                        "payload": period,
                        "synced_at": synced_at,
                    }
                )

        return documents[:limit]

    def fetch_documents(
        self,
        locations: list[str],
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Fetch normalized weather documents for every requested location.

        `limit` is the maximum number of documents per location.
        """
        documents: list[dict[str, Any]] = []

        for location in locations:
            documents.extend(
                self.fetch_documents_for_location(location, limit=limit)
            )

        return documents