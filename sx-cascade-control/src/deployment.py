"""Putting a controller on a plant in steps, with guards, and a journal of everything it did.

A model-based controller doesn't go from a simulation to closing the loop on a plant in one move.
It goes in steps, each one earned by the last:

    shadow     the controller runs on the live readings and says what it would do; nothing it
               says reaches the plant. The plant runs as it did before (here: handles held).
               Its advice is compared with what the plant did, for days if need be.
    advisory   its advice goes to an operator, who applies it at each review (every
               `review_every_h` hours) or doesn't. The operator stays in the loop.
    closed     it moves the handles itself, inside limits.

In every mode the guard sits between the controller and the plant:
    - a handle moves at most `max_move` (fraction of its commissioned value) per interval, and stays
      within `max_offset` of its commissioned value; anything beyond is clipped and journalled;
    - a watchdog: if the controller raises, or fails `fail_limit` solves in a row, the guard holds
      the last applied handles, raises an alarm and stays out until someone re-arms it
      (`rearm()`); the level loops underneath keep running all along;
    - a change of mode is bumpless: the controller is re-synced to the handles actually applied, so
      it doesn't start from moves the plant never saw.
Every interval goes in `journal`: time, mode, what was proposed, what was applied, what was clipped
and why, alarms. That's the record an engineer reviews before moving to the next step, and the
record that explains, afterwards, what the controller did and why.

Judgement calls: max_move 2 % and max_offset 20 % of the commissioned value, fail_limit 3, a
review every 2 h in advisory mode. They're settings; a plant would set them in its management of
change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("shadow", "advisory", "closed")


@dataclass(frozen=True)
class Limits:
    """What the guard lets through. max_move and max_offset are fractions of the commissioned
    value (per interval, and in all); fail_limit is failed solves in a row before the watchdog
    trips; review_every_h is how often an operator applies advice in advisory mode."""
    max_move: float = 0.02
    max_offset: float = 0.20
    fail_limit: int = 3
    review_every_h: float = 2.0


class DeploymentGuard:
    """Wraps a controller with the loop.Controller interface (a CircuitMPC, possibly under level loops)."""

    def __init__(self, inner, mode: str = "shadow", limits: Limits | None = None):
        """Take over from the controller at the handles it holds now (the commissioned ones).

        Args:
            inner: the controller to guard; it must have handles(), step(), apply() and last.
            mode: "shadow", "advisory" or "closed" (see the module docstring).
            limits: what the guard lets through; None for the defaults in Limits.

        Raises:
            ValueError: for a mode not in MODES.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.inner, self.mode, self.limits = inner, mode, limits or Limits()
        self.level_loops = inner.level_loops             # the settler loops go on under the guard
        self.commissioned = _copy(inner.handles())             # where the plant was handed over
        self.applied = _copy(self.commissioned)                 # what the plant runs on now
        self.t = 0.0
        self._fails, self._since_review = 0, 0.0
        self.tripped = False
        self.journal: list[dict] = []
        self.last = {}

    # the loop.Controller interface, passed through to the wrapped controller
    @property
    def u_nominal(self):
        return self.inner.u_nominal

    @property
    def cut(self):
        return self.inner.cut

    @property
    def t_step(self) -> float:
        return self.inner.t_step

    def apply(self, handles: dict):
        return self.inner.apply(handles)

    def handles(self) -> dict:
        return _copy(self.applied)

    # ---- changing what the guard lets through ---------------------------------------------------
    def set_mode(self, mode: str):
        """Move to another mode, bumpless: the controller starts from the handles actually applied."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != self.mode:
            self._resync()
            self.journal.append({"t": self.t, "event": f"mode {self.mode} -> {mode}"})
            self.mode, self._since_review = mode, 0.0

    def rearm(self):
        """After a watchdog trip, once someone has looked: re-sync and let the controller back in."""
        self._resync()
        self.tripped, self._fails = False, 0
        self.journal.append({"t": self.t, "event": "re-armed"})

    # ---- one interval ---------------------------------------------------------------------------
    def step(self, obs) -> dict:
        """One interval: ask the controller (unless the watchdog holds it out), decide what reaches
        the plant in this mode, journal it. Returns the handles the plant gets."""
        self.t += self.t_step
        entry = {"t": self.t, "mode": self.mode, "alarm": "", "clipped": ""}
        proposed = None if self.tripped else self._ask(obs, entry)
        entry["proposed"] = proposed
        self._apply(proposed, entry)
        if self.mode != "closed" or self.tripped:
            self._tell(self.applied)                        # the controller knows what the plant really got
        entry["applied"] = _copy(self.applied)
        self.journal.append(entry)
        self.last = dict(getattr(self.inner, "last", {}), mode=self.mode, guard_alarm=entry["alarm"],
                         guard_clipped=entry["clipped"])
        return _copy(self.applied)

    def _ask(self, obs, entry: dict) -> dict | None:
        """The controller's proposal, or None if it raised. The watchdog counts failed solves and
        trips after fail_limit in a row, or at once on an exception."""
        proposed, ok = None, False
        try:
            proposed = _copy(self.inner.step(obs))
            ok = self.inner.last.get("qp_ok", True) is not False
        except Exception as exc:                            # a controller that raises is a failed controller
            entry["alarm"] = f"controller raised {type(exc).__name__}: {exc}"
        self._watch(ok, proposed is None, entry)
        return proposed

    def _watch(self, ok: bool, raised: bool, entry: dict):
        """The watchdog: count failed solves in a row; trip at fail_limit, or at once if it raised."""
        self._fails = 0 if ok else self._fails + 1
        if raised or self._fails >= self.limits.fail_limit:
            self._trip(entry)

    def _trip(self, entry: dict):
        """Hold the controller out until rearm(), and say why in the journal entry."""
        self.tripped = True
        entry["alarm"] = (entry["alarm"] or f"{self._fails} failed solves in a row") + \
            "; watchdog: holding the handles, controller out until re-armed"

    def _apply(self, proposed: dict | None, entry: dict):
        """What reaches the plant: nothing new in shadow mode or when the watchdog holds; the advice
        at each review in advisory mode; the proposal itself in closed loop. Always inside the limits."""
        if self._plant_keeps_its_handles(proposed):
            return
        if self.mode == "advisory":
            self._review(proposed, entry)
            return
        self.applied, entry["clipped"] = self._limited(proposed)

    def _plant_keeps_its_handles(self, proposed: dict | None) -> bool:
        """True when nothing new reaches the plant this interval: the watchdog holds, the controller
        raised (no proposal), or the guard is in shadow mode."""
        return self.tripped or proposed is None or self.mode == "shadow"

    def _review(self, proposed: dict, entry: dict):
        """Advisory mode: every review_every_h hours the operator applies the advice, which may move as
        far as the intervals since the last review allow."""
        self._since_review += self.t_step
        if self._since_review < self.limits.review_every_h - 1e-9:
            return
        intervals = round(self._since_review / self.t_step)
        self.applied, entry["clipped"] = self._limited(proposed, intervals)
        self._since_review, entry["review"] = 0.0, True

    def _limited(self, proposed: dict, intervals: int = 1) -> tuple[dict, str]:
        """The proposal after the move and offset limits, and what was clipped. A review in
        advisory mode stands for `intervals` intervals, so it may move that many times as far."""
        out, notes = {}, []
        for name, value in proposed.items():
            v = self._clip_handle(name, value, intervals)
            if not np.allclose(v, value):
                notes.append(name)
            out[name] = float(v) if np.ndim(v) == 0 else v
        if notes:
            self._tell(out)
        return out, ", ".join(notes)

    def _clip_handle(self, name: str, value, intervals: int) -> np.ndarray:
        """One handle's proposed value, clipped first to max_move per interval from what the plant
        has now, then to max_offset from its commissioned value."""
        base = np.asarray(self.commissioned[name], float)
        prev = np.asarray(self.applied[name], float)
        step = self.limits.max_move * np.abs(base) * intervals
        v = np.clip(np.asarray(value, float), prev - step, prev + step)
        return np.clip(v, base - self.limits.max_offset * np.abs(base), base + self.limits.max_offset * np.abs(base))

    # ---- keeping the controller in step with the plant ------------------------------------------
    def _tell(self, handles: dict):
        """Set the controller's held handles to what was applied (CircuitMPC's u_prev)."""
        mpc = _mpc(self.inner)
        if mpc is None:
            return
        mpc.u_prev = np.concatenate([np.atleast_1d(np.asarray(handles[n], float)) for n in mpc._u_names])

    def _resync(self):
        """Put the controller's model state at its rest point for the applied handles and the feed
        last measured, and clear its bias (CircuitMPC.resync; decision 21: the MPC works on its own
        model)."""
        mpc = _mpc(self.inner)
        if mpc is None:
            return
        self._tell(self.applied)
        mpc.resync()


def _mpc(controller):
    """The CircuitMPC inside a controller (the controller itself, or what it wraps), or None."""
    while controller is not None and not _is_mpc(controller):
        controller = getattr(controller, "inner", None)
    return controller


def _is_mpc(controller) -> bool:
    """True for a CircuitMPC, recognised by the held handles it carries."""
    return hasattr(controller, "u_prev") and hasattr(controller, "_u_names")


def _copy(handles: dict) -> dict:
    """A copy of a handles dict that shares no arrays: scalars as floats, arrays as new arrays."""
    return {k: (np.array(v, float) if np.ndim(v) else float(v)) for k, v in handles.items()}
