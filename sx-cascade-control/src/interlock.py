"""A feed-loss interlock around any controller: what a plant does when the feed pump stops.

The pump stop is the upset the MPC has handled worst all along (decision 26). Two things go wrong
while the feed is off. Recovery, product now over feed now, reads nonsense, and the controller acts
on it. And with no feed there's no acid coming in, so a controller that keeps saponifying the
organic drives the extraction section out of acid (decision 33). A plant doesn't ask its optimizer
to ride through that: an interlock takes over.

    feed flow below `trip_fraction` of nominal  ->  trip: freeze the handles where they were
                                                    (and, with stop_base, stop saponifying)
    feed back above it for `rearm_h` hours      ->  re-arm: put the controller's model state back on
                                                    the model's rest point for the flows now held,
                                                    clear its output bias, hand control back

Re-arming from the model's rest point, not from the plant's state, follows decision 21: the MPC
linearizes on its own model's trajectory, never at the plant's state.

Judgement calls: trip at 20 % of nominal feed flow; re-arm after 2 h of steady feed; with
stop_base the saponification goes to zero while tripped and back to its held value on re-arm.
"""
from __future__ import annotations


class FeedLossInterlock:
    """Wraps a controller (CircuitMPC or anything with the loop.Controller interface)."""

    def __init__(self, inner, trip_fraction: float = 0.2, rearm_h: float = 2.0, stop_base: bool = False):
        """Arm the interlock around a controller.

        Args:
            inner: the controller it wraps.
            trip_fraction: trip when the measured feed flow falls below this fraction of nominal.
            rearm_h: hours of steady feed before control goes back to the controller.
            stop_base: stop saponifying while tripped.
        """
        self.inner, self.trip_fraction, self.rearm_h, self.stop_base = inner, trip_fraction, rearm_h, stop_base
        self.level_loops = inner.level_loops             # the settler loops go on under the interlock
        self.q_nominal = float(inner.u_nominal.Q_feed)
        self.tripped, self._back_since, self._held = False, None, None
        self.trips = 0
        self.last = {}

    # the loop.Controller interface, passed through
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
        return self._held if self.tripped else self.inner.handles()

    def step(self, obs) -> dict:
        """One interval: trip when the feed drops, hold while tripped, hand back once the feed has
        been steady for rearm_h; otherwise the controller's own step."""
        feed_on = float(obs.Q_feed) >= self.trip_fraction * self.q_nominal
        if not self.tripped and not feed_on:
            self._trip()
        if self.tripped and not self._feed_back_long_enough(feed_on):
            return self._hold()
        if self.tripped:
            self._rearm()
            self.tripped = False
        out = self.inner.step(obs)
        self.last = dict(self.inner.last, interlock="armed")
        return out

    def _trip(self):
        """Freeze the handles where they are."""
        self.tripped, self._back_since, self.trips = True, None, self.trips + 1
        self._held = dict(self.inner.handles())

    def _feed_back_long_enough(self, feed_on: bool) -> bool:
        """Count the hours the feed has been back; a drop starts the count again."""
        if not feed_on:
            self._back_since = None
            return False
        self._back_since = 0.0 if self._back_since is None else self._back_since + self.t_step
        return self._back_since >= self.rearm_h

    def _hold(self) -> dict:
        """The frozen handles (with stop_base, no saponification while the feed is off)."""
        held = dict(self._held)
        if self.stop_base and "saponified_fraction" in held:
            held["saponified_fraction"] = 0.0
        self.last = {"interlock": "tripped", "qp_ok": True}
        return held

    def _rearm(self):
        """Hand control back: the controller holds the frozen handles, its model state sits at the
        model's rest point for them, and its output bias starts again from zero."""
        if hasattr(self.inner, "resync"):                         # not an MPC: nothing to re-sync
            self.inner.resync()
