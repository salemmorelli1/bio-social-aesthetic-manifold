"""Opt-in rendering and privacy check of the built AME static preview."""

import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.environ.get("MANIFOLD_BROWSER_TESTS") != "1", reason="Opt-in installed Edge verification")
def test_built_preview_renders_local_capture_without_starting_detection():
    from playwright.sync_api import sync_playwright

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    assert (ROOT / "out/index.html").is_file(), "Run npm run build first."
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(ROOT / "out")))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=os.environ.get("MANIFOLD_TEST_BROWSER", "msedge"))
            try:
                context = browser.new_context(viewport={"width": 1365, "height": 1000})
                external = []
                errors = []

                def guard(route):
                    if urlsplit(route.request.url).hostname == "127.0.0.1":
                        route.continue_()
                    else:
                        external.append(route.request.url)
                        route.abort()

                context.route("**/*", guard)
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/", wait_until="networkidle")
                assert page.get_by_role("button", name="Start camera").is_enabled()
                assert page.get_by_label("Upload a JPEG, PNG, or WebP facial photo").is_enabled()
                assert "Research preview" in page.locator(".brand").inner_text()
                assert "Awaiting input" in page.locator(".stage-caption").inner_text()
                geometry = page.locator(".sr-only").evaluate(
                    "node=>({width:node.getBoundingClientRect().width,position:getComputedStyle(node).position})"
                )
                assert geometry["width"] <= 1 and geometry["position"] == "absolute"
                columns = page.locator(".content-grid").evaluate(
                    "node=>getComputedStyle(node).gridTemplateColumns.split(' ').length"
                )
                assert columns == 2
                assert external == [], "Capture/model downloads must wait for an explicit user action."
                assert errors == []
                if os.environ.get("MANIFOLD_SCREENSHOT"):
                    destination = Path(os.environ["MANIFOLD_SCREENSHOT"])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(destination), full_page=True)
            finally:
                browser.close()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
