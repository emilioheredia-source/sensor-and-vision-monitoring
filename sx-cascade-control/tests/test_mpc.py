"""Checks on the linear MPC (src/mpc.py) in closed loop with the circuit model.

The MPC works from a linearization of the circuit at its operating point. So
the first check is that the linear model is stable, the same as the circuit
it comes from. Then two closed-loop runs:

- nothing happens: the controller should hold the operating point, keep every
  QP solve feasible, and make no off-spec product,
- the feed gets 30 % stronger: the controller should act (ask for more
  organic) and do no worse on purity than leaving the handles alone.

These run on prnd_small, the Pr|Nd validation split (cut = "Pr") on the
Idaho-size circuit. It's small and fast, which is why the controller was
developed on it. The plant's product is NdPr; Pr|Nd is for checking.
"""
import numpy as np
import pytest

from src.commission import commission
from src.config import build_circuit, load_config, nominal_inputs
from src.disturbances import FeedStep, Scenario
from src.metrics import transition_cost
from src.mpc import CircuitMPC, MPCSettings, run_closed_loop
from src.simulate import simulate


@pytest.fixture(scope="module")
def small():
    """prnd_small commissioned at its design point, with an MPC built there.

    Returns (run config, circuit, steady state x0, inputs u with the jacklegs
    set, controller). The MPC holds purity at 0.995, above the 0.99 spec, and
    recovery at the run file's target.
    """
    rc = load_config("prnd_small")
    circ = build_circuit(rc)
    x0, u = commission(circ, nominal_inputs(rc, circ))
    spec, rec = rc.targets()                     # purity spec and recovery target, fractions
    ctl = CircuitMPC(circ, x0, u, rc.cut, spec, 0.995, rec, MPCSettings())
    return rc, circ, x0, u, ctl


def test_linearization_is_stable(small):
    """The circuit's Jacobian at the operating point has every eigenvalue in the left half plane.

    A solvent-extraction circuit left alone settles to a steady state, so its
    linearization has to be stable too. If it weren't, the MPC's prediction
    would run away even with no input moves, and every plan it made would be wrong.
    """
    _, circ, x0, u, _ = small
    A = circ.jac(x0, u.numpy())        # d(dx/dt)/dx at the operating point
    assert np.max(np.linalg.eigvals(A).real) < 0


def test_holds_the_operating_point(small):
    """With no upset, every QP solves, the handles barely move, and no product goes off spec.

    The plant here is the controller's own model, sitting at steady state.
    Over 6 h the organic flow and saponification must stay within 2 % of
    where they started. A controller that drifts with nothing to fix is
    fighting its own model or its own tuning.
    """
    rc, circ, x0, u, ctl = small
    # 6 h closed loop, no scenario (None): the inputs stay nominal except what the MPC moves
    traj, log = run_closed_loop(circ, x0, ctl, None, 6.0)
    assert all(h["qp_ok"] for h in log)
    # nothing happened, so the controller should barely move anything
    for h in log:
        assert abs(h["Q_org"] / u.Q_org - 1) < 0.02
        assert abs(h["saponified_fraction"] / u.saponified_fraction - 1) < 0.02
    spec, _ = rc.targets()
    cost = transition_cost(circ, traj, rc.cut, spec)
    assert cost["off_spec_mol"] == 0.0


def test_acts_on_a_feed_step(small):
    """After a 30 % stronger feed, the MPC raises the organic flow and keeps purity at least as well
    as doing nothing.

    More metal coming in needs more organic to carry it, so the organic flow
    has to move by more than 2 %. The closed-loop run is then compared with an
    open-loop run of the same upset: the lowest purity under control must not
    be worse than the lowest purity with the handles held (to 0.001).
    """
    rc, circ, x0, u, _ = small
    spec, rec = rc.targets()
    # the feed gets 30 % stronger at t = 1 h and stays that way
    sc = Scenario([FeedStep(t=1.0, factor=1.3)])
    ctl = CircuitMPC(circ, x0, u, rc.cut, spec, 0.995, rec, MPCSettings())
    traj, log = run_closed_loop(circ, x0, ctl, sc, 12.0)
    assert all(h["qp_ok"] for h in log)
    # largest relative change in organic flow over the run
    moved = max(abs(h["Q_org"] / u.Q_org - 1) for h in log)
    assert moved > 0.02                                 # more metal in: it should ask for more organic
    cl = transition_cost(circ, traj, rc.cut, spec)      # closed loop
    # open loop: the same upset over the same 12 h, handles held, purity sampled every 0.25 h
    ol = transition_cost(circ, simulate(circ, x0, sc.u_of_t(u), 12.0, dt_out=0.25, breakpoints=sc.breakpoints), rc.cut, spec)
    assert cl["purity_series"].min() >= ol["purity_series"].min() - 1e-3
