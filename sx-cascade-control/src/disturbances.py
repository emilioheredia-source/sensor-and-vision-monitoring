"""The upset types, as functions of time acting on the inputs.

What this file does
-------------------
An upset is something that happens to the plant that nobody asked for: the
feed gets stronger, a pump stops, a flowmeter reads wrong, a settler's
interface drifts. Here each kind of upset is a small object that takes the
inputs the plant would otherwise get, u, and returns the inputs it really
gets at time t:

    u_plant(t) = upset.apply(t, u)

Every upset starts at its time `t` (hours into the run). It either jumps
there (ramp_h = 0, a step) or moves in a straight line over `ramp_h` hours.
The helper `_ramp` gives that shape as a number a between 0 and 1:

    a = 0                    before t
    a = (time - t) / ramp_h  during the ramp
    a = 1                    after t + ramp_h

and each upset uses a to move part of the way from the normal input to the
upset one. For example, a feed acidity upset gives

    h_feed(t) = (1 - a) * h_feed_normal + a * value

Each upset also lists its `breakpoints`: the times where the input changes
abruptly (the start of a step, the end of a ramp, the end of an
interruption). The integrator restarts at each one so it doesn't step over
a corner.

A Scenario is a list of upsets, applied in order, one on top of the other.

Where each upset comes from:
  FeedSwitch, FlowmeterBias, InterfaceDrift                Carester
  FeedStep, FeedAcid, FlowStep, RefluxAcid, EfficiencyLoss  Idaho thesis

The named catalogue of upsets built from these is in scenarios.py. The
controller loop applies them after the controller's handles are written in,
so a flowmeter bias scales what was commanded and a drift adds to the
jackleg that was set (ARCHITECTURE.md, decision 16).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .circuit import Inputs
from .params import Params, feed_concentrations

# The four pumped streams an upset can act on, and the Inputs field (L/h) of each.
_FLOW_FIELD = {"organic": "Q_org", "feed": "Q_feed", "scrub": "Q_scrub", "strip": "Q_strip"}


# ---------------------------------------------------------------------------
# The shape in time, and the base class
# ---------------------------------------------------------------------------

# Ramps are applied in an open-loop simulation as a staircase of steps this long (h); see
# Disturbance.breakpoints.
RAMP_STEP_H = 0.25


def _ramp(t, t0, ramp_h):
    """How far the upset has gone at time t: 0 before t0, 1 after t0 + ramp_h, linear in between.

    t, t0 and ramp_h are in hours. ramp_h = 0 (or less) is a step at t0.
    """
    if ramp_h <= 0:
        return 1.0 if t >= t0 else 0.0
    return float(np.clip((t - t0) / ramp_h, 0.0, 1.0))


@dataclass
class Disturbance:
    """The base of every upset.

    t        when the upset starts, hours into the run
    ramp_h   how long it takes to reach its full size, hours (0 = a step)
    """
    t: float
    ramp_h: float = 0.0

    @property
    def breakpoints(self):
        """The times (h) where the input changes: the start, then every RAMP_STEP_H through the ramp,
        and the end of the ramp.

        simulate() holds the inputs constant between breakpoints, at their value just after each
        segment starts. With only the start and the end listed, an open-loop run held a ramp at
        zero for its whole length and then jumped to full size in one step. Listing points inside
        the ramp makes it a staircase of quarter-hour steps, finer than the controller's half-hour
        step, so open-loop and controlled runs see the same upset.
        """
        if self.ramp_h <= 0:
            return [self.t]
        n_steps = max(1, int(np.ceil(self.ramp_h / RAMP_STEP_H - 1e-9)))
        return [self.t + self.ramp_h * k / n_steps for k in range(n_steps + 1)]

    def apply(self, t: float, u: Inputs) -> Inputs:
        """The inputs the plant gets at time t (h), given the inputs u it would get without the upset."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Feed upsets
# ---------------------------------------------------------------------------

@dataclass
class FeedSwitch(Disturbance):
    """Change deposit: the feed composition moves from the nominal one to another (Carester).

        c_feed(t) = (1 - a) * c_feed_normal + a * c_feed_new          (mol/L, per component)

    to                 the deposit switched to, a deposit name in params/feeds.toml (e.g. "yangibana")
    total_RE_g_per_L   the new feed's total metal, g/L; None keeps the strength of the current feed
    removed_upstream   the components taken out before this circuit. It must match the run file's
                       feed, so there's deliberately no default.
    params             the parameter set, for the molar masses and the feed table

    Raises ValueError from apply() if removed_upstream wasn't given.
    """
    to: str = "yangibana"
    total_RE_g_per_L: float | None = None       # None: keep the nominal strength
    removed_upstream: tuple | None = None       # must match the run file's feed (no default on purpose)
    params: Params | None = None
    _c_to: np.ndarray = field(default=None, repr=False)   # the new feed, mol/L; worked out on first use

    def apply(self, t, u):
        """The inputs at time t with the feed composition moved part of the way to the new deposit's."""
        if self.removed_upstream is None:
            raise ValueError("FeedSwitch needs removed_upstream, the same list as the run file's feed")
        c_to = self._new_feed(u)
        a = _ramp(t, self.t, self.ramp_h)
        return u.replace(c_feed=(1 - a) * np.asarray(u.c_feed) + a * c_to)

    def _new_feed(self, u: Inputs) -> np.ndarray:
        """The new deposit's feed composition, mol/L.

        Worked out once, the first time it's needed, and kept. Its strength (g/L) is
        total_RE_g_per_L, or else the current feed's: sum of c_feed (mol/L) times molar mass M (g/mol).
        """
        if self._c_to is None:
            total = self.total_RE_g_per_L or float(np.asarray(u.c_feed) @ self.params.M)
            self._c_to = feed_concentrations(self.to, total, self.params, tuple(self.removed_upstream))
        return self._c_to


@dataclass
class FeedStep(Disturbance):
    """Feed strength (total metal) scaled by `factor`, composition unchanged (Idaho: day-to-day variation).

        c_feed(t) = c_feed_normal * (1 + a * (factor - 1))       factor 1.2 = 20 % stronger
    """
    factor: float = 1.2

    def apply(self, t, u):
        """The inputs at time t with the feed strength scaled (see the class docstring)."""
        a = _ramp(t, self.t, self.ramp_h)
        return u.replace(c_feed=np.asarray(u.c_feed) * (1 + a * (self.factor - 1)))


@dataclass
class FeedAcid(Disturbance):
    """Feed acidity set to `value` mol/L (Idaho: process sensitivity to pH).

        h_feed(t) = (1 - a) * h_feed_normal + a * value          (mol/L)
    """
    value: float = 0.3

    def apply(self, t, u):
        """The inputs at time t with the feed acidity moved toward `value`."""
        a = _ramp(t, self.t, self.ramp_h)
        return u.replace(h_feed=(1 - a) * u.h_feed + a * self.value)


# ---------------------------------------------------------------------------
# Flow upsets
# ---------------------------------------------------------------------------

@dataclass
class FlowStep(Disturbance):
    """A stream's flow scaled by `factor`; with `duration_h` it returns to normal.

    An interruption is factor 0.

        Q(t) = Q_normal * (1 + a * (factor - 1))       while the upset lasts
        Q(t) = Q_normal                                from t + duration_h on

    stream       "organic", "feed", "scrub" or "strip"
    factor       what the flow is multiplied by (0 = the pump stops)
    duration_h   how long it lasts, hours; None = for the rest of the run
    """
    stream: str = "feed"              # organic | feed | scrub | strip
    factor: float = 0.0
    duration_h: float | None = None

    @property
    def breakpoints(self):
        """The base breakpoints, plus the time the flow comes back if the upset has a duration."""
        b = super().breakpoints
        return b + ([self.t + self.duration_h] if self.duration_h else [])

    def apply(self, t, u):
        """The inputs at time t with the stream's flow scaled, or as normal once the duration is over."""
        a = 0.0 if self._over(t) else _ramp(t, self.t, self.ramp_h)
        key = _FLOW_FIELD[self.stream]
        return u.replace(**{key: getattr(u, key) * (1 + a * (self.factor - 1))})

    def _over(self, t: float) -> bool:
        """True from t + duration_h on, when the flow is back to normal; never without a duration."""
        return self.duration_h is not None and t >= self.t + self.duration_h


@dataclass
class FlowmeterBias(Disturbance):
    """The plant delivers `1 + bias` times what the meter says (Carester: 'precision of flowmeters').

    In open loop it's a hidden flow step. It matters once a controller trusts the meter.

        Q_delivered(t) = Q_commanded * (1 + a * bias)

    stream   "organic", "feed", "scrub" or "strip"
    bias     the fraction delivered beyond what's commanded (0.05 = 5 % more)
    """
    stream: str = "organic"
    bias: float = 0.05

    def apply(self, t, u):
        """The inputs at time t with the stream delivering more (or less) than commanded."""
        key = _FLOW_FIELD[self.stream]
        a = _ramp(t, self.t, self.ramp_h)
        return u.replace(**{key: getattr(u, key) * (1 + a * self.bias)})


# ---------------------------------------------------------------------------
# Settler and mixer upsets
# ---------------------------------------------------------------------------

@dataclass
class InterfaceDrift(Disturbance):
    """The hydraulic balance of some settlers drifts (crud, a partly blocked jackleg).

    The effective jackleg height of `stages` ramps up by `delta_m` (Carester: 'variation
    of the interface positions'). The form is chosen; the size is the knob.

        H_jackleg[j](t) = H_jackleg_set[j] + a * delta_m          (m), for j in stages

    stages    the stage indices affected (0-based)
    delta_m   how far the effective jackleg moves, m
    """
    stages: tuple = (5,)
    delta_m: float = 0.02

    def apply(self, t, u):
        """The inputs at time t with the affected stages' effective jacklegs raised."""
        a = _ramp(t, self.t, self.ramp_h)
        H = np.array(u.H_jackleg, float).copy()
        for j in self.stages:
            H[j] += a * self.delta_m
        return u.replace(H_jackleg=H)


@dataclass
class RefluxAcid(Disturbance):
    """Scrub liquor acidity set to `value` mol/L: too strong and the scrub section starts stripping
    (Idaho 4.1.3).

        h_scrub(t) = (1 - a) * h_scrub_normal + a * value          (mol/L)
    """
    value: float = 1.5

    def apply(self, t, u):
        """The inputs at time t with the scrub acidity moved toward `value`."""
        a = _ramp(t, self.t, self.ramp_h)
        return u.replace(h_scrub=(1 - a) * u.h_scrub + a * self.value)


@dataclass
class EfficiencyLoss(Disturbance):
    """Transfer rate of `stages` scaled by `factor` (a mixer running slow; Idaho: stage efficiency).

        k_factor[j](t) = k_factor_normal[j] * (1 + a * (factor - 1))       for j in stages

    stages   the stage indices affected (0-based)
    factor   what their transfer rate is multiplied by (0.3 = down to 30 %)
    """
    stages: tuple = (0, 1)
    factor: float = 0.3

    def apply(self, t, u):
        """The inputs at time t with the affected stages' transfer rates scaled."""
        a = _ramp(t, self.t, self.ramp_h)
        kf = self._current_k_factor(u)
        for j in self.stages:
            kf[j] *= 1 + a * (self.factor - 1)
        return u.replace(k_factor=kf)

    @staticmethod
    def _current_k_factor(u: Inputs) -> np.ndarray:
        """A writable copy of the per-stage transfer-rate multipliers. None in u means all ones."""
        if u.k_factor is None:
            return np.ones(len(np.asarray(u.H_jackleg)))
        return np.array(u.k_factor, float).copy()


# ---------------------------------------------------------------------------
# A scenario: several upsets together
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """A list of upsets, applied in order. An empty list is a run with no upset.

    items   the Disturbance objects
    """
    items: list

    @property
    def breakpoints(self) -> list[float]:
        """Every upset's breakpoints (h), merged and sorted, without repeats."""
        return sorted({b for d in self.items for b in d.breakpoints})

    def u_of_t(self, u_nominal: Inputs):
        """A function of time t (h) that gives the plant's inputs with every upset applied.

        Example
        -------
            u_of_t = Scenario([FeedStep(t=5.0, factor=1.2)]).u_of_t(u)
            u_of_t(4.0).c_feed      # the normal feed
            u_of_t(6.0).c_feed      # 20 % stronger
        """
        def f(t):
            """The inputs at time t (h), each upset applied on top of the one before."""
            u = u_nominal
            for d in self.items:
                u = d.apply(t, u)
            return u
        return f
