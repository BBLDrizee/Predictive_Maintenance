"""Headless smoke test of every dashboard screen (FR-11), skipped until models are exported."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from turborul.config import REPO_ROOT, load_config

pytestmark = pytest.mark.skipif(not (load_config()["paths"]["models_dir"] / "FD001" / "meta.json").exists(),
                                reason="no exported models; run python -m turborul.export")
APP = str(REPO_ROOT / "app" / "dashboard.py")


@pytest.mark.parametrize("page", ["Fleet overview", "Engine detail", "Model comparison", "About"])
def test_page_renders_without_errors(page: str) -> None:
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["page"] = page
    at.run()
    assert not at.exception, at.exception
    assert at.title or at.markdown


def test_fleet_counts_match_api() -> None:
    from turborul import service

    at = AppTest.from_file(APP, default_timeout=120).run()
    rows = service.fleet("FD001")
    critical = sum(r["tier"] == "Critical" for r in rows)
    assert any(f">{critical}<" in m.value for m in at.markdown), "Critical count card missing"


def test_open_engine_button_switches_page() -> None:
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["page"] == "Engine detail"
