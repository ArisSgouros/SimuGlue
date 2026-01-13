from __future__ import annotations

from pathlib import Path

import pytest

import simuglue.workflow.cij.markers as m


def test_markers_finalize_success_and_failure(tmp_path: Path):
    d = tmp_path / "case"
    d.mkdir()

    # success
    m.finalize_success(d)
    assert m.is_done(d)
    assert not m.is_running(d)
    assert not m.is_failed(d)

    # failure should clear .running and write .failed
    (d / ".running").write_text("running\n", encoding="utf-8")
    m.finalize_failure(d, "boom")
    assert m.is_failed(d)
    assert not m.is_running(d)
    assert (d / ".failed").read_text(encoding="utf-8").strip() == "boom"


def test_prepare_run_creates_running_and_clears_failed(tmp_path: Path):
    d = tmp_path / "case"
    d.mkdir()

    (d / ".failed").write_text("old\n", encoding="utf-8")
    m.prepare_run(d, verbose=False)

    assert m.is_running(d)
    assert not m.is_failed(d)


def test_prepare_run_respects_done_or_running(tmp_path: Path):
    d = tmp_path / "case"
    d.mkdir()

    (d / ".done").write_text("done\n", encoding="utf-8")
    should = m.prepare_run(d, verbose=False)
    assert should is False

    (d / ".done").unlink()
    (d / ".running").write_text("running\n", encoding="utf-8")
    should = m.prepare_run(d, verbose=False)
    assert should is False
