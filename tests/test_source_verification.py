"""Verify source-byte corruption and unsafe manifest paths fail closed."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_source import verify  # noqa: E402


def test_source_tampering_is_rejected_without_overwriting_input(tmp_path):
    source = tmp_path / "engine.py"
    source.write_bytes(b"original\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (tmp_path / "SOURCE_SHA256SUMS.json").write_text(json.dumps({"engine.py": digest}))
    assert verify(tmp_path) == 1
    source.write_bytes(b"altered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(tmp_path)
    assert source.read_bytes() == b"altered\n"


def test_source_manifest_cannot_escape_the_checkout(tmp_path):
    (tmp_path / "SOURCE_SHA256SUMS.json").write_text(json.dumps({"../engine.py": "0" * 64}))
    with pytest.raises(ValueError, match="unsafe"):
        verify(tmp_path)


def test_binary_report_bytes_are_part_of_the_verified_delivery(tmp_path):
    report = tmp_path / "report.pdf"
    data = bytes(range(256))
    report.write_bytes(data)
    (tmp_path / "SOURCE_SHA256SUMS.json").write_text(
        json.dumps({"report.pdf": hashlib.sha256(data).hexdigest()})
    )
    assert verify(tmp_path) == 1
    report.write_bytes(data[:-1])
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(tmp_path)
