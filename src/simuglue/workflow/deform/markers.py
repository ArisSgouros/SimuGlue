# simuglue/workflow/markers.py
from __future__ import annotations
from pathlib import Path

def m_running(case_dir: Path) -> Path: return case_dir / ".running"
def m_done(case_dir: Path)    -> Path: return case_dir / ".done"
def m_failed(case_dir: Path)  -> Path: return case_dir / ".failed"

def is_done(case_dir: Path) -> bool:
    return m_done(case_dir).exists()

def is_running(case_dir: Path) -> bool:
    return m_running(case_dir).exists()

def is_failed(case_dir: Path) -> bool:
    return m_failed(case_dir).exists()

def prepare_run(case_dir: Path, *, verbose: bool = True) -> bool:
    """
    Returns True if we should proceed with running.
    Sets .running and clears any stale .failed.
    Returns False (and prints a short message) if .done or .running exist.
    """
    # mark running and clear old failed
    m_running(case_dir).write_text("running\n", encoding="utf-8")
    m_failed(case_dir).unlink(missing_ok=True)

def finalize_success(case_dir: Path) -> None:
    m_done(case_dir).write_text("done\n", encoding="utf-8")
    m_running(case_dir).unlink(missing_ok=True)

def finalize_failure(case_dir: Path, msg: str) -> None:
    m_failed(case_dir).write_text(msg.rstrip() + "\n", encoding="utf-8")
    m_running(case_dir).unlink(missing_ok=True)

