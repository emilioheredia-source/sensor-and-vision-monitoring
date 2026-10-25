"""Checks on the operating-point optimizer (src/design.py).

The optimizer picks flows, scrub acid and saponification so the circuit's
steady state meets the purity spec with as much recovery as it can get. The
run files' design points come from it, so its answer has to be trustworthy:

- it has to be a real steady state of the model, not just a point the solver
  stopped at,
- the project's own metrics, computed separately, have to agree with the
  purity and recovery the optimizer reports,
- it has to meet spec, and do at least as well as a coarse grid search did.

This runs on the Pr|Nd validation split (cut = "Pr") with a didymium feed, on
the Idaho-size circuit. The plant's product is NdPr; Pr|Nd is a check that
the method works on a split with published data.
"""
import numpy as np
import pytest

from src.circuit import Circuit, Inputs, build_map
from src.design import optimize_operating_point, reflux_link
from src.params import load_params
from src.metrics import summarize_steady

p = load_params()
n = 4                                                  # components: LaCe, Pr, Nd, SmPlus
SPEC = p.raw["targets"]["product_purity_fraction"]     # product purity spec, fraction (0.99)
DIDYMIUM = np.array([0.0, 0.0885, 0.2602, 0.0])         # feed, mol/L: Pr and Nd only


def _u(cmap, Q_org=1200.0, Q_scrub=300.0, Q_strip=600.0, h_scrub=0.5, sapon=0.4):
    """Starting inputs for the optimizer on the given map.

    Flows L/h (feed fixed at 600), acids mol/L (feed 0.05, strip 2.0), the
    scrub carries no metal of its own, and every jackleg sits at 0.41 m.

    Why six parameters: the map and the five handles' starting values, each with a default.
    """
    return Inputs(Q_org, 600.0, Q_scrub, Q_strip, DIDYMIUM, 0.05, np.zeros(n), h_scrub, 2.0, sapon,
                  np.full(cmap.N, 0.41))


@pytest.mark.parametrize("reflux", [False, True])
def test_optimizer_meets_spec_on_idaho_circuit(reflux):
    """The optimizer finds a steady state on spec, its numbers check out, and it beats the coarse grid.

    Three things are checked. The model's right-hand side is zero at the
    answer (to 1e-6), so it's a genuine steady state. The purity and recovery
    that metrics.summarize_steady computes from that state match what the
    optimizer reports. And purity is at spec while recovery is above 5 %,
    better than the best a coarse grid found on this circuit (2 % with an
    acid-only scrub, 0.7 % with reflux). Both scrub modes are run.
    """
    # 2 extraction, 12 scrub, 4 strip stages: the Idaho pilot layout
    cmap = build_map(2, 12, 4)
    circ = Circuit(cmap, p, scrub_reflux=reflux)
    u0 = _u(cmap)
    # what the optimizer may move, with bounds: flows L/h, scrub acid mol/L, saponified fraction 0..1
    free = {"Q_org": (600.0, 3000.0), "Q_scrub": (50.0, 1500.0), "h_scrub": (0.05, 2.0),
            "saponified_fraction": (0.0, 0.6)}
    # with reflux, strip flow is tied to scrub flow: strip = 300 L/h net product + scrub
    res = optimize_operating_point(circ, u0, "Pr", SPEC, free, link=reflux_link(300.0) if reflux else None)
    assert res.success, res.status
    # the answer is a genuine steady state, and the summary agrees with the optimizer's own numbers
    assert np.abs(circ.rhs(res.x, res.u.numpy())).max() < 1e-6
    s = summarize_steady(circ, res.x, res.u, "Pr")
    assert s["product_purity"] == pytest.approx(res.purity, abs=1e-6)
    assert s["recovery"] == pytest.approx(res.recovery, abs=1e-6)
    assert res.purity >= SPEC - 1e-6
    # the coarse grid's best at spec was 0.02 (acid-only) and 0.007 (reflux) on this circuit
    assert res.recovery > 0.05
