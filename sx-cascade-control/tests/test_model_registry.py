"""The model registry (src/model_registry.py): checks, review, activation, rollback, journal."""
import json

import pytest

from src.model_registry import ModelRegistry

BOUNDS = (-0.8, 0.5)


def registry(tmp_path=None):
    return ModelRegistry({"log_beta_pr_lace": 0.0, "log_beta_nd_pr": 0.0, "log_k": 0.0}, (2.0, 1.5), BOUNDS,
                         path=None if tmp_path is None else tmp_path / "models.json")


def test_a_small_clean_fit_passes_every_check_and_the_residual_does_not_decide():
    r = registry()
    v = r.propose(24.0, [-0.02, 0.01, 0.1], "daily fit", 1.1)
    assert all(c["passed"] for c in v.checks.values())
    v = r.propose(48.0, [-0.02, 0.01, 0.1], "daily fit", 50.0)
    assert not v.checks["residual"]["passed"] and not v.checks["residual"]["decides"]
    assert all(c["passed"] for k, c in v.checks.items() if c["decides"])


def test_a_fit_on_its_bound_or_jumping_fails_the_checks():
    r = registry()
    v = r.propose(24.0, [-0.02, 0.01, 0.5], "daily fit", 1.1)        # k on its upper bound, and a big step
    assert not v.checks["bounds"]["passed"] and not v.checks["step"]["passed"]
    assert "log_k" in v.checks["bounds"]["detail"]


def test_only_an_approved_version_goes_in_use_and_rollback_returns_the_last(tmp_path):
    r = registry(tmp_path)
    v1 = r.propose(24.0, [-0.02, 0.0, 0.0], "day 1", 1.0)
    with pytest.raises(ValueError):
        r.activate(v1.id)
    r.approve(v1.id, "engineer", "checks pass")
    r.activate(v1.id)
    v2 = r.propose(48.0, [-0.04, 0.0, 0.0], "day 2", 1.0)
    r.approve(v2.id, "engineer", "checks pass")
    r.activate(v2.id)
    assert r.active_id == v2.id
    r.rollback(50.0, "engineer", "product off after the update")
    assert r.active_id == v1.id and r.versions[v2.id].status == "rejected"
    saved = json.loads((tmp_path / "models.json").read_text())
    assert saved["active"] == v1.id and any("rolled back" in e["event"] for e in saved["journal"])
