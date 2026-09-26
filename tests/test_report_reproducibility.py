"""Regression tests for the committed technical-report artifact."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_apa_report  # noqa: E402


def _report_narrative_text() -> str:
    return " ".join(
        str(block.get("text", ""))
        for page in build_apa_report.PAGES
        for block in page["blocks"]
    )


def test_report_narrative_matches_current_interface_and_test_surface():
    """Prevent assessment-era UI and validation prose from drifting backward."""

    narrative = _report_narrative_text()
    assert "accessible four-tab workspace" in narrative
    assert "The Python and JavaScript test suites cover" in narrative
    assert "three-tab workspace" not in narrative
    assert "Eighty-six tests" not in narrative


def test_report_builder_reproduces_committed_bytes(tmp_path, monkeypatch):
    """Two clean builds and the committed PDF must be byte-identical.

    This guards both sources of the audit failure: volatile PDF timestamps and
    host-dependent font selection.
    """

    committed_path = build_apa_report.OUTPUT
    committed_bytes = committed_path.read_bytes()

    first_path = tmp_path / "first.pdf"
    monkeypatch.setattr(build_apa_report, "OUTPUT", first_path)
    build_apa_report.build_report()
    first_bytes = first_path.read_bytes()

    second_path = tmp_path / "second.pdf"
    monkeypatch.setattr(build_apa_report, "OUTPUT", second_path)
    build_apa_report.build_report()
    second_bytes = second_path.read_bytes()

    assert first_bytes == second_bytes
    assert first_bytes == committed_bytes

    reader = PdfReader(first_path)
    assert len(reader.pages) == 27
    assert hashlib.sha256(first_bytes).hexdigest() == hashlib.sha256(
        committed_bytes
    ).hexdigest()


def test_failed_report_build_preserves_the_last_valid_artifact(tmp_path, monkeypatch):
    output = tmp_path / "report.pdf"
    previous_bytes = b"previous validated report"
    output.write_bytes(previous_bytes)
    monkeypatch.setattr(build_apa_report, "OUTPUT", output)

    def fail_during_drawing(_pdf):
        raise RuntimeError("forced drawing failure")

    monkeypatch.setattr(build_apa_report, "draw_title_page", fail_during_drawing)
    with pytest.raises(RuntimeError, match="forced drawing failure"):
        build_apa_report.build_report()

    assert output.read_bytes() == previous_bytes
    assert not output.with_suffix(".pdf.tmp").exists()
