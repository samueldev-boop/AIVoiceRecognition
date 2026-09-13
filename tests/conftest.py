"""Tests must never publish audit events into the developer's real spool."""

import pytest


@pytest.fixture(autouse=True)
def isolated_audit_spool(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_SPOOL_DIR", str(tmp_path / "audit-spool"))
