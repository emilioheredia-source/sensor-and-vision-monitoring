"""The controller's model, versioned: each update proposed, checked, reviewed, then put in use.

A model fitted to the plant's readings is a proposal, not an update (README, "Learn the plant,
under supervision"). This file is the record that makes that concrete. Every fit becomes a version
with:
    - the fitted numbers (log factors on the Pr/(La+Ce) and Nd/Pr separation factors and on the
      stage transfer rate, as notebook 31 fits them) and where they came from;
    - automatic checks the reviewer sees first:
        step      the change from the model in use is no bigger than a slow drift gives in a day
        bounds    no fitted number sits on the limit the fit allows. One that does is the fit
                  bending a number to make up for something the model doesn't have: a fault,
                  a fouled instrument, a section running differently (notebook 33)
        range     the separation factors stay inside what the chemistry allows
        residual  how well the fitted model reproduces the readings. Shown to the reviewer, but it
                  doesn't decide: on notebook 37's plant it ran from 1 to 82 per reading on normal
                  days and 22 to 68 on fault days, so it separates nothing there. A first version
                  let it decide, with a limit chosen before looking (4 per reading); it rejected
                  every fit, and the controller ran a drifting plant on a stale model, worse than
                  with no review at all. Limits come from the plant's normal history.
    - a status: proposed, then approved or rejected by a named reviewer with a note, and an
      approved version made active. The version in use before it is kept: `rollback()` goes back.
Every event is journalled with its time.

The checks advise; the reviewer decides. In notebook 37 a scripted reviewer approves a version only
when every check passes, which stands in for an engineer reading the same checks.

Judgement calls: step limits 0.05 on a log separation factor (5 %) and 0.25 on the log rate factor;
"on its bound" within 0.01 of the fit's limit; the separation factors between 1.2 and 3.5.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

NAMES = ("log_beta_pr_lace", "log_beta_nd_pr", "log_k")
STEP_LIMIT = np.array([0.05, 0.05, 0.25])
BOUND_MARGIN = 0.01
RESIDUAL_LIMIT = 4.0                          # reported against, doesn't decide (see above)
DECIDING = ("step", "bounds", "range")
BETA_RANGE = (1.2, 3.5)


@dataclass
class Version:
    """One version of the controller's model: the fitted numbers, the checks and the review."""
    id: int
    t: float                                  # h, plant time when proposed
    params: dict                              # NAMES -> value
    source: str
    checks: dict = field(default_factory=dict)  # check -> {"passed": bool, "detail": str}
    status: str = "proposed"                  # proposed, approved, rejected, active, retired
    reviewer: str = ""
    note: str = ""


class ModelRegistry:
    """The versions of the controller's model, the one in use, and the journal of what happened."""

    def __init__(self, initial: dict, base_betas: tuple, bounds: tuple, path: Path | None = None):
        """initial: the model in use at the start (usually all zeros: the nominal model).
        base_betas: the nominal model's (Pr/(La+Ce), Nd/Pr), to check the range in plain units.
        bounds: (low, high), the limits the fit allows each log factor, for the bounds check.
        path: a JSON file to keep the registry in, rewritten at every event, or None."""
        self.base_betas, self.bounds, self.path = np.asarray(base_betas, float), bounds, path
        self.versions = [Version(0, 0.0, dict(zip(NAMES, map(float, [initial[n] for n in NAMES]))), "nominal model",
                                 status="active", reviewer="commissioning")]
        self.active_id = 0
        self.journal: list[dict] = [{"t": 0.0, "event": "version 0 active (nominal model)"}]

    @property
    def active(self) -> Version:
        """The version in use."""
        return self.versions[self.active_id]

    def propose(self, t: float, params, source: str, residual_per_reading: float) -> Version:
        """Add a fitted model as a proposal, with the checks the reviewer sees.

        Args:
            t: plant time, h. params: the fitted log factors, in NAMES order. source: where the fit
            came from. residual_per_reading: the fit's chi-squared per reading.
        Returns:
            the new Version, status "proposed".
        """
        v = Version(len(self.versions), t, dict(zip(NAMES, map(float, params))), source)
        v.checks = self._checks(np.array([v.params[n] for n in NAMES]), residual_per_reading)
        self.versions.append(v)
        failed = [k for k, c in v.checks.items() if not c["passed"] and c["decides"]]
        self._log(t, f"version {v.id} proposed ({source}); deciding checks failed: {', '.join(failed) or 'none'}")
        return v

    def _on_bound(self, x: np.ndarray) -> list[str]:
        """The fitted numbers sitting on (within BOUND_MARGIN of) the limits the fit allows."""
        lo, hi = self.bounds
        return [n for n, xi in zip(NAMES, x) if not lo + BOUND_MARGIN < xi < hi - BOUND_MARGIN]

    def _checks(self, x: np.ndarray, residual: float) -> dict:
        """The four checks of a fit x (log factors) against the version in use."""
        step = np.abs(x - np.array([self.active.params[n] for n in NAMES]))
        on_bound = self._on_bound(x)
        betas = self.base_betas * np.exp(x[:2])
        checks = {
            "step": (bool(np.all(step <= STEP_LIMIT)),
                     ", ".join(f"{n} moved {s:.3f} (limit {lim})" for n, s, lim in zip(NAMES, step, STEP_LIMIT))),
            "bounds": (not on_bound, "on its bound: " + ", ".join(on_bound) if on_bound else "none on a bound"),
            "residual": (bool(residual <= RESIDUAL_LIMIT), f"{residual:.2f} per reading (for reference: {RESIDUAL_LIMIT})"),
            "range": (bool(np.all((betas >= BETA_RANGE[0]) & (betas <= BETA_RANGE[1]))),
                      f"separation factors {betas[0]:.3f}, {betas[1]:.3f}"),
        }
        return {k: {"passed": ok, "detail": detail, "decides": k in DECIDING} for k, (ok, detail) in checks.items()}

    def approve(self, vid: int, reviewer: str, note: str = "", t: float | None = None):
        """Record a reviewer's approval of version vid. It isn't in use until activate().
        t: plant time, h, for the journal; None means the time the version was proposed."""
        self._review(vid, "approved", reviewer, note, t)

    def reject(self, vid: int, reviewer: str, note: str, t: float | None = None):
        """Record a reviewer's rejection of version vid, with the reason in the note (t as approve)."""
        self._review(vid, "rejected", reviewer, note, t)

    def _review(self, vid: int, status: str, reviewer: str, note: str, t: float | None):
        """Set a version's status, reviewer and note, and journal it.

        Why five parameters: they're the fields of one journal entry.
        """
        v = self.versions[vid]
        v.status, v.reviewer, v.note = status, reviewer, note
        self._log(t if t is not None else v.t, f"version {vid} {status} by {reviewer}: {note}")

    def activate(self, vid: int, t: float | None = None):
        """Put an approved version in use; the one in use before is retired (and kept)."""
        v = self.versions[vid]
        if v.status != "approved":
            raise ValueError(f"version {vid} is {v.status}; only an approved version can be put in use")
        self.active.status = "retired"
        v.status, self.active_id = "active", vid
        self._log(t if t is not None else v.t, f"version {vid} active")

    def rollback(self, t: float, reviewer: str, note: str):
        """Go back to the last version that was in use before this one."""
        retired = [v for v in self.versions[:self.active_id] if v.status == "retired"]
        if not retired:
            raise ValueError("nothing to roll back to")
        back = retired[-1]
        self.active.status = "rejected"
        back.status, self.active_id = "active", back.id
        self._log(t, f"rolled back to version {back.id} by {reviewer}: {note}")

    def _log(self, t, event):
        """Add an event to the journal, and rewrite the registry file if there is one."""
        self.journal.append({"t": float(t), "event": event})
        if self.path is not None:
            self.save(self.path)

    def save(self, path: Path):
        """Write the versions, the one in use and the journal to a JSON file."""
        Path(path).write_text(json.dumps({"active": self.active_id, "versions": [asdict(v) for v in self.versions],
                                          "journal": self.journal}, indent=1), encoding="utf-8")
