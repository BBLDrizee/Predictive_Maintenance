"""Capture dashboard screenshots for the progress report.

Starts the dashboard headless, opens each screen in the locally installed Edge or
Chrome (via Playwright, no browser download), and saves PNGs to reports/screenshots/.

    python reports/screenshots.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "screenshots"
PORT = 8599
URL = f"http://localhost:{PORT}"


def wait_for(url: str, timeout: float = 90) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if requests.get(url, timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(url)


def settle(page, text: str, seconds: float = 2.5) -> None:
    """Wait until the page shows ``text`` and Streamlit has stopped running, then let Plotly draw."""
    page.get_by_text(text).first.wait_for(timeout=180_000)
    page.wait_for_function("() => !document.body.innerText.includes('RUNNING...')", timeout=180_000)
    time.sleep(seconds)


def shoot(page, path: Path) -> None:
    """Resize the window to the page content (Streamlit scrolls inside its own container), then capture."""
    height = page.evaluate("() => { const m = document.querySelector('[data-testid=\"stMain\"]');"
                           " return m ? m.scrollHeight : document.body.scrollHeight; }")
    page.set_viewport_size({"width": 1440, "height": max(900, min(int(height) + 40, 4000))})
    time.sleep(1.5)
    page.screenshot(path=path)


def choose_page(page, name: str, text: str) -> None:
    page.get_by_text(name, exact=True).first.click()
    settle(page, text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "dashboard.py"),
                               "--server.headless", "true", "--server.port", str(PORT), "--browser.gatherUsageStats", "false"],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(f"{URL}/_stcore/health")
        with sync_playwright() as p:
            browser = None
            for channel in ("msedge", "chrome"):
                try:
                    browser = p.chromium.launch(channel=channel)
                    break
                except Exception:  # noqa: BLE001 - try the next installed browser
                    continue
            if browser is None:
                raise RuntimeError("no Edge or Chrome found")
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
            start = time.perf_counter()
            page.goto(URL)
            settle(page, "Which engines need attention?", 3)
            first_load = time.perf_counter() - start
            print(f"first load: {first_load:.1f}s")
            (OUT / "timing.json").write_text(json.dumps({"first_load_seconds": round(first_load, 2)}))
            shoot(page, OUT / "fleet.png")

            choose_page(page, "Engine detail", "Why this prediction")
            shoot(page, OUT / "engine.png")

            choose_page(page, "Model comparison", "Error by true-RUL bucket")
            shoot(page, OUT / "models.png")
            # A second visitor on the already-running server (imports and caches warm).
            fresh = browser.new_page(viewport={"width": 1440, "height": 1000})
            start = time.perf_counter()
            fresh.goto(URL)
            settle(fresh, "Which engines need attention?", 0)
            warm_load = time.perf_counter() - start
            print(f"warm load: {warm_load:.1f}s")
            (OUT / "timing.json").write_text(json.dumps({"first_load_seconds": round(first_load, 2),
                                                          "warm_load_seconds": round(warm_load, 2)}))
            browser.close()
    finally:
        server.terminate()
    print("saved", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
