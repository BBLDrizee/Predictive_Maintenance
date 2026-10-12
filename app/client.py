"""Data access for the dashboard: the REST API when TURBORUL_API_URL is set, otherwise in-process calls.

Both paths return the same JSON-shaped data, so the dashboard runs the same way
locally against the API and on a single-process free host.
"""

from __future__ import annotations

import os
from typing import Any

import requests


class LocalClient:
    mode = "in-process"

    def __getattr__(self, name: str):
        from turborul import service

        return {
            "subsets": service.subsets, "fleet": service.fleet, "importance": service.global_importance,
            "engine": service.engine, "results": service.model_comparison, "test_predictions": service.test_predictions,
        }[name]


class HttpClient:
    mode = "API"

    def __init__(self, base_url: str, timeout: float = 30) -> None:
        self.base_url, self.timeout = base_url.rstrip("/"), timeout

    def _get(self, path: str, **params: Any) -> Any:
        r = requests.get(f"{self.base_url}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def subsets(self) -> list[dict[str, Any]]:
        return self._get("/subsets")

    def fleet(self, subset: str, tier: str | None = None) -> list[dict[str, Any]]:
        return self._get(f"/fleet/{subset}", tier=tier)

    def importance(self, subset: str) -> list[dict[str, Any]]:
        return self._get(f"/fleet/{subset}/importance")

    def engine(self, subset: str, unit: int) -> dict[str, Any]:
        return self._get(f"/engines/{subset}/{unit}")

    def results(self) -> dict[str, Any]:
        return self._get("/results")

    def test_predictions(self, subset: str, model: str) -> list[dict[str, Any]]:
        return self._get(f"/results/{subset}/{model}/predictions")


def get_client() -> LocalClient | HttpClient:
    url = os.environ.get("TURBORUL_API_URL")
    return HttpClient(url) if url else LocalClient()
