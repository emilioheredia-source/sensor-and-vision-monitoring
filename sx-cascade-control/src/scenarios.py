"""The catalogue of upsets, defined once.

What this file does
-------------------
disturbances.py has the upset types (a feed switch, a flow step, a drift and
so on). This file picks one size and place for each, gives it a name, and
lists them all in one catalogue. The command line, the matrix and the
workbench (through runs.py) take their upsets from here. Notebooks 03, 05,
05b and 08 were written before this module and build the same upsets
themselves, with the same keys and sizes.

Each entry carries a key (what a run spec asks for), a one-line title, the
source it comes from, and a longer explanation for anyone choosing it from
a list.

Every upset starts at UPSET_TIME_H, 5 hours into the run, after a stretch
with nothing happening. Where an upset acts on a stage, the stage is worked
out from the run file's stage counts, so the same key means the same kind
of upset on a small circuit and a large one:

    feed stage          stage n_ext - 1 (0-based), the last extraction stage, where the feed enters
    middle stage        (n_ext + n_scrub + n_strip) // 2

The catalogue, in order:

    none             nothing happens
    feed_switch      Eneabba feed -> Yangibana feed over 1 h, same total metal (or back, on a Yangibana run file)
    feed_step        feed 20 % stronger
    feed_acid        feed acidity 0.05 -> 0.3 mol/L
    feed_interrupt   feed pump off for 2 h
    flowmeter_bias   5 % more organic than the meter says
    interface_drift  the middle settler's effective jackleg +10 cm over 5 h
    reflux_acid      scrub liquor acid x1.5 (at most 4 mol/L)
    efficiency_loss  the feed stage's mixer and the first scrub mixer at 30 % transfer rate
"""
from __future__ import annotations

from dataclasses import dataclass

from .circuit import Inputs
from .config import RunConfig
from .disturbances import (Disturbance, EfficiencyLoss, FeedAcid, FeedStep, FeedSwitch, FlowmeterBias,
                           FlowStep, InterfaceDrift, RefluxAcid, Scenario)

UPSET_TIME_H = 5.0          # chosen: every upset starts 5 h into the run, after a stretch at rest


# ---------------------------------------------------------------------------
# One entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UpsetEntry:
    """One named upset.

    key          the name a run spec uses, e.g. "feed_switch"
    title        a one-line title for menus and figures
    source       where the upset comes from (Carester, the Idaho thesis)
    explanation  a few sentences for someone choosing it from a list
    disturbance  the upset itself, or None for "no upset"
    """
    key: str
    title: str
    source: str
    explanation: str
    disturbance: Disturbance | None      # None: no upset, the circuit at rest

    def scenario(self) -> Scenario:
        """The upset as a Scenario (an empty one for "none"), ready for simulate() or the loop."""
        return Scenario([] if self.disturbance is None else [self.disturbance])


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

def catalogue(rc: RunConfig, u: Inputs, t0: float = UPSET_TIME_H) -> dict[str, UpsetEntry]:
    """The upsets for this run file, at time t0.

    Parameters
    ----------
    rc   the run file; its stage counts place the stage upsets and its feed's
         removed_upstream list goes to the feed switch
    u    the operating point. The scrub-acid upset is set relative to the scrub
         acid the plant runs at.
    t0   when every upset starts, hours into the run

    Returns
    -------
    {key: UpsetEntry}, in the order listed in the module docstring.

    Example
    -------
        entry = catalogue(rc, u_plant)["feed_switch"]
        u_of_t = entry.scenario().u_of_t(u_plant)
    """
    entries = ([_no_upset()]
               + _feed_upsets(rc, t0)
               + _flow_upsets(t0)
               + _stage_upsets(rc, u, t0))
    return {e.key: e for e in entries}


def _no_upset() -> UpsetEntry:
    """The reference: nothing happens."""
    return UpsetEntry("none", "No upset", "-",
                      "The circuit left at its operating point. Shows what the controller does when "
                      "nothing happens.",
                      None)


def _feed_upsets(rc: RunConfig, t0: float) -> list[UpsetEntry]:
    """What changes in the feed: its composition, its strength, its acidity."""
    # The feed switch must take out the same components upstream as the run file's feed does.
    removed = tuple(rc["feed"]["removed_upstream"])
    # The switch goes to the other published deposit: lean Eneabba to rich Yangibana, or back.
    rich_base = rc["feed"]["deposit"] == "yangibana"
    to, change = ("eneabba", "41 % to 21 %") if rich_base else ("yangibana", "21 % to 41 %")
    return [
        UpsetEntry("feed_switch", f"Feed switch {rc['feed']['deposit'].capitalize()} -> {to.capitalize()} over 1 h",
                   "Carester (Rollat et al. 2024)",
                   "The feed changes from one published deposit to another at the same total metal: NdPr goes "
                   f"from about {change} of the metal. The upset a plant fed from more than one source will see.",
                   FeedSwitch(t=t0, ramp_h=1.0, to=to, params=rc.params, removed_upstream=removed)),
        UpsetEntry("feed_step", "Feed strength +20 %", "Idaho thesis (day-to-day variation)",
                   "The feed gets 20 % stronger at the same composition.",
                   FeedStep(t=t0, factor=1.2)),
        UpsetEntry("feed_acid", "Feed acidity 0.05 -> 0.3 mol/L", "Idaho thesis (pH sensitivity)",
                   "The feed arrives six times more acidic. Acid stops extraction.",
                   FeedAcid(t=t0, value=0.3)),
    ]


def _flow_upsets(t0: float) -> list[UpsetEntry]:
    """What goes wrong with a pump or a meter."""
    return [
        UpsetEntry("feed_interrupt", "Feed pump stops for 2 h", "Idaho thesis (flow upsets)",
                   "No feed for two hours, then back. The one upset where this controller does worse than "
                   "doing nothing.",
                   FlowStep(t=t0, stream="feed", factor=0.0, duration_h=2.0)),
        UpsetEntry("flowmeter_bias", "Organic flowmeter reads 5 % low", "Carester",
                   "The plant gets 5 % more organic than commanded, and the flowmeter says all is well.",
                   FlowmeterBias(t=t0, stream="organic", bias=0.05)),
    ]


def _stage_upsets(rc: RunConfig, u: Inputs, t0: float) -> list[UpsetEntry]:
    """What goes wrong inside the circuit: a settler's level, the scrub liquor, a pair of mixers."""
    # Stages are counted from 0. The feed enters stage n_ext - 1, the last extraction stage;
    # stage n_ext is the first scrub stage. Those two are the mixers next to the feed point.
    n_ext = rc["circuit"]["n_ext"]
    mid = (n_ext + rc["circuit"]["n_scrub"] + rc["circuit"]["n_strip"]) // 2
    return [
        UpsetEntry("interface_drift", f"Hydraulic drift on settler {mid}: +10 cm over 5 h", "Carester",
                   "One settler's interface drifts up as if its jackleg were raised, with nobody touching it.",
                   InterfaceDrift(t=t0, ramp_h=5.0, stages=(mid,), delta_m=0.10)),
        # The new scrub acid is 1.5 times the plant's own, capped at 4 mol/L.
        UpsetEntry("reflux_acid", "Scrub liquor acid x1.5", "Idaho thesis 4.1.3",
                   "The scrub liquor arrives half again as acidic as set: the scrub starts stripping product.",
                   RefluxAcid(t=t0, value=min(1.5 * u.h_scrub, 4.0))),
        UpsetEntry("efficiency_loss", f"Mixers {n_ext - 1} and {n_ext} at 30 %", "Idaho thesis (stage efficiency)",
                   "Two mixers next to the feed stage slow to 30 % of their transfer rate.",
                   EfficiencyLoss(t=t0, stages=(n_ext - 1, n_ext), factor=0.3)),
    ]
