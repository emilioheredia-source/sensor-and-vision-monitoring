"""Purity, recovery, losses and transition cost. All defined from the cut.

What this file does
-------------------
Everything the project reports about how well the circuit runs is worked out
here: product purity, recovery, the metal lost to the raffinate, the reagent
spent, where the interfaces sit, and, over a whole run, what an upset cost.

The cut
-------
The cut names the last component meant to leave in the raffinate; everything
heavier goes to the product. The components are, in order, (La+Ce), Pr, Nd,
(Sm and heavier).

    cut = "LaCe"   the main case. La and Ce go to the raffinate; Pr and Nd go
                   together to the product, as NdPr.
    cut = "Pr"     the Pr | Nd split, used only to check the model.

Purity and recovery are on a mole-of-metal basis:

    purity   = product-side metal in the product / all metal in the product
    recovery = product-side metal leaving in the product / product-side metal fed

The transition cost (APPENDIX_B_control.md, B.6)
------------------------------------------------
Over a run, from time series of the point outputs:

    off-spec          integral of the product metal rate while purity < spec        mol
    lost              integral of the product-side metal rate to the raffinate      mol
    lost excess       the same, minus its value at the start of the run             mol
    interface         integral of how far the interfaces sit outside their window   m*h
    reagent           integral of base + acid fed                                   mol
    time to settle    the last time purity leaves a +-0.002 band around its final value   h

The spec is 0.99. Off-spec counts every mol of product made while purity is
below it, so two runs that both dip near the spec can differ a lot for a small
difference in purity.
"""
from __future__ import annotations

import numpy as np

from .circuit import Circuit, Inputs
from .params import Params
from .simulate import Trajectory


# ---------------------------------------------------------------------------
# Which components are product
# ---------------------------------------------------------------------------

def heavy_mask(p: Params, cut: str) -> np.ndarray:
    """True for components meant to go to the product (heavier than the cut).

    Example: with components (LaCe, Pr, Nd, SmPlus) and cut "LaCe" the mask is
    [False, True, True, True].
    """
    return np.arange(len(p.names)) > p.index(cut)


# ---------------------------------------------------------------------------
# Purity and recovery at one instant
# ---------------------------------------------------------------------------

def product_purity(c_prod: np.ndarray, p: Params, cut: str) -> float:
    """Product-side metal over all metal in the product stream, 0..1.

    c_prod is the product's concentration of each component, mol/L.
    Returns NaN when the product holds no metal at all.
    """
    total = float(np.sum(c_prod))
    return float(np.sum(c_prod[heavy_mask(p, cut)]) / total) if total > 0 else float("nan")


def raffinate_purity(c_raff: np.ndarray, p: Params, cut: str) -> float:
    """Raffinate-side metal over all metal in the raffinate stream, 0..1.

    c_raff is the raffinate's concentration of each component, mol/L.
    Returns NaN when the raffinate holds no metal at all.
    """
    total = float(np.sum(c_raff))
    return float(np.sum(c_raff[~heavy_mask(p, cut)]) / total) if total > 0 else float("nan")


def recovery(out: dict, u: Inputs, p: Params, cut: str) -> float:
    """Fraction of the product-side metal in the feed that leaves in the product, 0..1.

        recovery = Q_prod * sum(c_prod[heavy]) / (Q_feed * sum(c_feed[heavy]))

    out is circuit.outlets(...) (flows in L/h, concentrations in mol/L); u the inputs.
    Returns NaN when no product-side metal is fed.
    """
    m = heavy_mask(p, cut)
    fed = float(u.Q_feed * np.sum(np.asarray(u.c_feed)[m]))       # mol/h
    got = float(out["Q_prod"] * np.sum(out["c_prod"][m]))          # mol/h
    return got / fed if fed > 0 else float("nan")


def summarize_steady(circuit: Circuit, x: np.ndarray, u: Inputs, cut: str) -> dict:
    """Purities, recovery and outlet strengths at one state, as a dict for printing.

    Keys: product_purity, raffinate_purity, recovery (fractions), product_total_M and
    raffinate_total_M (total metal in each outlet, mol/L).
    """
    out = circuit.outlets(x, u.numpy())
    return {
        "product_purity": product_purity(out["c_prod"], circuit.p, cut),
        "raffinate_purity": raffinate_purity(out["c_raff"], circuit.p, cut),
        "recovery": recovery(out, u, circuit.p, cut),
        "product_total_M": float(np.sum(out["c_prod"])),
        "raffinate_total_M": float(np.sum(out["c_raff"])),
    }


# ---------------------------------------------------------------------------
# Interfaces and reagent
# ---------------------------------------------------------------------------

def interface_window(p: Params) -> tuple[float, float]:
    """(lowest, highest) interface height allowed, m.

    Each limit is a fraction of the settler height (parameters.toml [stage_geometry]
    interface_window_fraction times settler_height_m).
    """
    geo = p.raw["stage_geometry"]
    return tuple(f * geo["settler_height_m"] for f in geo["interface_window_fraction"])


def reagent_rate(u: Inputs, p: Params) -> float:
    """mol/h of base (saponification) and acid (strip, scrub) fed.

        reagent = s * L_mono * Q_org + Q_strip * h_strip + Q_scrub * h_scrub

    s is the saponified fraction, L_mono the extractant concentration as monomer (mol/L),
    flows in L/h and acidities in mol/L. design.py uses the same formula.
    """
    return (float(u.saponified_fraction) * p.raw["extractant"]["concentration_monomer"] * float(u.Q_org)
            + float(u.Q_strip) * float(u.h_strip) + float(u.Q_scrub) * float(u.h_scrub))


# ---------------------------------------------------------------------------
# What an operator would read
# ---------------------------------------------------------------------------

def point_outputs(circuit: Circuit, x: np.ndarray, uv: np.ndarray, cut: str) -> dict:
    """What a plant operator would read at one instant.

    Parameters
    ----------
    circuit   the circuit model
    x         the state at this instant
    uv        the inputs at this instant, as a vector (Inputs.numpy())
    cut       the last component meant for the raffinate

    Returns
    -------
    dict with
        purity               product purity, 0..1
        recovery             recovery, 0..1
        product_rate         all metal leaving in the product, mol/h
        lost_rate            product-side metal leaving in the raffinate, mol/h
        interfaces           every stage's interface height, m
        interface_violation  how far the interfaces sit outside their window, summed over stages, m
        reagent_rate         base + acid fed, mol/h
        lost_pct_of_feed     lost_rate as a % of the product-side metal being fed now
        reagent_per_product  reagent_rate per mol of product-side metal leaving in the product now

    Why the last two: a loss in mol/h means little without knowing how much is coming in. Against
    the feed of the moment, a switch to a richer or leaner feed changes the yardstick with it, so
    no baseline has to be chosen. Reagent per mol of product is the plant's specific consumption.
    """
    p = circuit.p
    out = circuit.outlets(x, uv)
    u = Inputs.from_vector(uv, circuit.n, circuit.N)
    z = np.asarray(circuit.unpack(x)[3], float)      # interface height per stage, m
    lo, hi = interface_window(p)
    heavy = heavy_mask(p, cut)
    lost_rate = out["Q_raff"] * float(np.sum(out["c_raff"][heavy]))
    fed_rate = float(u.Q_feed) * float(np.sum(np.asarray(u.c_feed, float)[heavy]))
    made_rate = out["Q_prod"] * float(np.sum(out["c_prod"][heavy]))
    return {
        "purity": product_purity(out["c_prod"], p, cut),
        "recovery": recovery(out, u, p, cut),
        "product_rate": out["Q_prod"] * float(np.sum(out["c_prod"])),
        "lost_rate": lost_rate,
        "lost_pct_of_feed": 100.0 * lost_rate / fed_rate if fed_rate > 0 else float("nan"),
        "reagent_per_product": reagent_rate(u, p) / made_rate if made_rate > 0 else float("nan"),
        "interfaces": z,
        # sum over stages of max(0, lo - z) + max(0, z - hi): zero when every interface is in its window
        "interface_violation": float(np.sum(np.maximum(0, lo - z) + np.maximum(0, z - hi))),
        "reagent_rate": reagent_rate(u, p),
    }


def stage_profile(circuit: Circuit, x: np.ndarray) -> dict:
    """Every stage at one instant.

    Returns a dict with
        aqueous     concentration of each component in the aqueous phase (stage x component, mol/L)
        organic     the same in the organic phase (stage x component, mol/L)
        acid        aqueous acid per stage, mol/L
        interfaces  interface height per stage, m
    """
    c_aq, c_org, h, z = circuit.unpack(x)
    return {"aqueous": _rows_as_array(c_aq), "organic": _rows_as_array(c_org),
            "acid": np.asarray(h, float).ravel(), "interfaces": np.asarray(z, float).ravel()}


def _rows_as_array(rows) -> np.ndarray:
    """A list of per-stage vectors as one 2-D float array, one row per stage."""
    return np.array([np.asarray(r, float).ravel() for r in rows])


# ---------------------------------------------------------------------------
# What an upset cost, over a whole run
# ---------------------------------------------------------------------------

def transition_cost(circuit: Circuit, traj: Trajectory, cut: str, spec: float,
                    settle_band: float = 0.002) -> dict:
    """What a feed switch (or any upset) costs, scored the way the plan says.

    Parameters
    ----------
    circuit       the circuit the trajectory ran on
    traj          the simulated run: times t (h), states x and inputs u
    cut           the last component meant for the raffinate
    spec          the product purity spec, 0..1 (0.99)
    settle_band   the band around the final purity used for the time to settle

    Returns
    -------
    The dict from score():
        off_spec_mol              product metal made while purity < spec
        lost_mol                  product-side metal (NdPr for cut "LaCe") that left in the raffinate
        lost_excess_mol           the same, beyond its value at the start of the run
        time_to_settle_h          when product purity last left a band around its final value
        interface_excursion_m_h   integral of how far any interface sits outside its window
        reagent_mol               base dose (saponification) + strip acid + scrub acid fed, over the run
        purity_series             product purity at every time point

    Why five parameters: what was run (circuit, trajectory) and how to score it (cut, spec, settling band) are separate facts.
    """
    # The point outputs at every saved time, then each one as a time series.
    # Judgement call: the saved times are the 0.5 h control grid, so a dip shorter than a step can
    # be missed by the trapezoid rule. Scoring from a finer plant output would check it.
    points = [point_outputs(circuit, xk, uk, cut) for xk, uk in zip(traj.x, traj.u)]
    series = {k: np.array([pt[k] for pt in points]) for k in points[0]}
    return score(traj.t, series, spec, settle_band)


def score(t: np.ndarray, series: dict, spec: float, settle_band: float = 0.002) -> dict:
    """The transition cost from series of point_outputs over time t (see transition_cost).

    t is in hours and the rates are in mol/h, so each integral is in mol. The
    integrals use the trapezoid rule over the saved time points.

    Parameters
    ----------
    t             times, h
    series        {name: array over t} for purity, product_rate, lost_rate,
                  interface_violation and reagent_rate (point_outputs' keys)
    spec          the product purity spec, 0..1
    settle_band   half-width of the band around the final purity, for the time to settle

    Returns
    -------
    The dict described in transition_cost.
    """
    purity, lost_rate = series["purity"], series["lost_rate"]
    # Why (one score for everything): the open loop, every sensor set and the RL policy are scored
    # by the same five numbers on the plant's true state, so the comparisons are like for like.

    # Off-spec rate: the product rate where purity < spec, zero elsewhere (mol/h).
    # Judgement call: all-or-nothing at the spec, which is how a batch is judged, but it jumps:
    # 98.99 % counts the whole product rate and 99.01 % counts nothing (215 against 0 mol between
    # two similar runs, REPORT.md 5.2). Reporting the purity shortfall integrated over time next to
    # it would show which differences are real.
    off = np.where(purity < spec, series["product_rate"], 0.0)

    # Time to settle: find the last sample outside the band around the final purity.
    # Purity has settled from the sample after it on. If no sample is ever outside, that's 0 h.
    # Judgement call: +-0.002 is a chosen band, and it is measured against the purity at the end of
    # the run, which may not have settled yet in 40 h.
    outside = np.where(np.abs(purity - purity[-1]) > settle_band)[0]
    t_settle = 0.0 if len(outside) == 0 else float(t[min(outside[-1] + 1, len(t) - 1)])

    return {
        "off_spec_mol": float(np.trapezoid(off, t)),
        "lost_mol": float(np.trapezoid(lost_rate, t)),
        # Lost excess: the loss rate minus what the circuit was already losing at t = 0.
        # Judgement call: counted over the run only (40 h), while the circuit's inventory settles
        # over ~400 h. A loss the run sets up but that only reaches the raffinate after 40 h isn't
        # counted, and neither is a change in the metal held up in the circuit. The steady-state
        # floors guard the controller against borrowing from it, but the score doesn't. Adding the
        # change in held-up metal over the run, or scoring at 200 h, would settle it.
        "lost_excess_mol": float(np.trapezoid(lost_rate - lost_rate[0], t)),
        "time_to_settle_h": t_settle,
        "interface_excursion_m_h": float(np.trapezoid(series["interface_violation"], t)),
        "reagent_mol": float(np.trapezoid(series["reagent_rate"], t)),
        "purity_series": purity,
    }
