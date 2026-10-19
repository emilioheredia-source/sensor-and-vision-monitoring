"""Checks on the plant/model mismatch (src/mismatch.py) and the trim that copes with it.

The controller plans on the nominal parameters. The plant it drives is built
from a perturbed copy of them: different extraction constants, a different
acid exponent, uneven stage efficiencies. That gap is what makes the control
problem real, so these tests make sure it's what it says it is:

- strength 0 gives the nominal model back exactly,
- the same seed gives the same plant every time, and the spreads stay in the
  stated bounds,
- the run file's [plant] setting switches between the perturbed and the
  matched plant,
- the commissioning trim, with full information, puts the perturbed plant on
  spec, and leaves a matched plant's handles where they were.

The runs here use ndpr_small: the NdPr product, cut between La+Ce and Pr.
"""
import numpy as np

from src.config import build_circuit, build_plant, load_config
from src.mismatch import Mismatch, perturbed_params, stage_efficiency_factors
from src.params import load_params

p = load_params()


def test_strength_zero_is_nominal():
    """With strength 0 the perturbed plant is the model, number for number.

    Every spread in Mismatch is scaled by strength. At 0 the K values, the
    acid exponent, the organic density and all the stage efficiencies have to
    come back unchanged. That's what lets the matched-plant runs act as the
    reference for the perturbed ones.
    """
    q = perturbed_params(p, Mismatch(seed=3, strength=0.0))
    assert np.allclose(q.K, p.K) and q.n_acid == p.n_acid and q.rho_org0 == p.rho_org0
    assert np.all(stage_efficiency_factors(10, Mismatch(seed=3, strength=0.0)) == 1.0)


def test_perturbation_is_bounded_and_reproducible():
    """The same seed gives the same plant, and the differences stay in the stated sizes.

    Two draws with seed 1 must give identical K values, so a run can be
    repeated. Each K is off by a factor drawn with a 10-20 % spread, so a
    ratio outside 0.5 to 2 would mean something went wrong. The acid exponent
    goes from 3.0 to 3.3. Stage efficiencies sit between 85 % and 100 % of
    nominal and aren't all the same.
    """
    a = perturbed_params(p, Mismatch(seed=1))
    b = perturbed_params(p, Mismatch(seed=1))
    assert np.allclose(a.K, b.K)
    ratio = a.K / p.K                                         # plant K over model K, per component
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0)        # a few sigma of 10-20 %
    assert a.n_acid == p.n_acid + 0.3
    # per-stage multipliers on the transfer rate for a 32-stage circuit
    kf = stage_efficiency_factors(32, Mismatch(seed=1))
    assert kf.min() >= 0.85 and kf.max() <= 1.0 and kf.std() > 0


def test_run_file_switch():
    """The run file decides whether the plant differs from the model.

    ndpr_small has [plant] mismatch = "perturbed", so its plant's K values
    differ from the model's. ndpr_small_matched is the same file with
    mismatch = "none", so its plant has the model's K values and every stage
    at full efficiency.
    """
    rc = load_config("ndpr_small")
    plant, kf = build_plant(rc)                  # kf: the plant's per-stage efficiency multipliers
    model = build_circuit(rc)
    assert not np.allclose(plant.p.K, model.p.K)
    rc_m = load_config("ndpr_small_matched")
    plant_m, kf_m = build_plant(rc_m)
    assert np.allclose(plant_m.p.K, model.p.K) and np.all(kf_m == 1.0)


def _trim(name, t_settle):
    """Commission the plant on the model's design point, then trim it under the MPC.

    This is the same sequence the runs use to find the plant's own operating
    point. The plant is run up on the model's design inputs and its jacklegs
    are set. Then the MPC, seeing the full plant state, runs for t_settle hours
    and moves the handles until the product is on spec.

    Returns (product purity, recovery, the design inputs, the trimmed inputs,
    the number of failed controller solves).
    """
    from src.commission import commission, trim_under_control
    from src.config import nominal_inputs
    from src.metrics import product_purity, recovery
    from src.mpc import CircuitMPC, MPCSettings
    from src.simulate import steady_state
    rc = load_config(name)
    spec, rec = rc.targets()                     # purity spec and recovery target, fractions
    circ, (plant, kf) = build_circuit(rc), build_plant(rc)
    ones = np.ones(circ.N)                       # nominal stage efficiency: what the model assumes
    # the plant at rest on the model's design inputs, with its jacklegs set
    x_rest, u_design = commission(plant, nominal_inputs(rc, plant).replace(k_factor=kf))
    # the model's own steady state at those inputs: where the controller linearizes
    x_ss = steady_state(circ, u_design.replace(k_factor=ones), x0=x_rest)
    # MPC holding purity at 0.995 (above the spec) and recovery at the run file's target
    ctl = CircuitMPC(circ, x_ss, u_design.replace(k_factor=ones), rc.cut, spec, 0.995, rec, MPCSettings())
    x0, u = trim_under_control(plant, x_rest, kf, ctl, t_settle=t_settle)
    out = plant.outlets(x0, u.replace(k_factor=kf).numpy())
    return (product_purity(out["c_prod"], plant.p, rc.cut), recovery(out, u, plant.p, rc.cut),
            u_design, u, ctl.failures)


def test_trim_puts_the_perturbed_plant_on_spec():
    """The controller, with full information, brings the real plant from the model's design point
    to spec and holds recovery; on the matched plant it leaves the handles alone.

    On the perturbed plant the model's design point is off. After 20 h of trim
    the product has to be at 0.994 purity or better (the target is 0.995) and
    recovery at 0.515 or better (the target is 0.522), with no failed solves.
    On the matched plant the design point is already right, so after 5 h every
    handle has to be within 2 % of where it started. A controller that moved
    things on a plant with nothing wrong would be chasing its own errors.
    """
    pur, rec, u_design, u, failures = _trim("ndpr_small", 20.0)
    assert failures == 0
    assert pur >= 0.994 and rec >= 0.52 - 0.005
    pur_m, rec_m, u_design_m, u_m, failures_m = _trim("ndpr_small_matched", 5.0)
    assert failures_m == 0 and pur_m >= 0.994
    # the four handles the MPC moves: organic flow, scrub flow, scrub acid, saponified fraction
    for h in ("Q_org", "Q_scrub", "h_scrub", "saponified_fraction"):
        assert abs(getattr(u_m, h) - getattr(u_design_m, h)) <= 0.02 * abs(getattr(u_design_m, h)) + 1e-9


def test_ratio_generator_keeps_separation_factors_in_published_ranges():
    """The "ratios" generator draws the separation factors themselves, inside their published
    ranges widened 10 % on each side, with no tails (decision 31). Over 50 seeds every Pr/(La+Ce),
    Nd/Pr and Sm/Nd stays inside its range, and seed 1 of the independent generator (the report's
    plant) is unchanged by the option existing."""
    p = load_params()
    m0 = Mismatch(generator="ratios")
    names = list(p.names)
    i_la, i_pr, i_nd, i_sm = (names.index(c) for c in ("LaCe", "Pr", "Nd", "SmPlus"))
    for seed in range(1, 51):
        K = perturbed_params(p, Mismatch(seed=seed, generator="ratios")).K
        assert m0.beta_pr_lace[0] - 1e-9 <= K[i_pr] / K[i_la] <= m0.beta_pr_lace[1] + 1e-9
        assert m0.beta_nd_pr[0] - 1e-9 <= K[i_nd] / K[i_pr] <= m0.beta_nd_pr[1] + 1e-9
        assert m0.beta_sm_nd[0] - 1e-9 <= K[i_sm] / K[i_nd] <= m0.beta_sm_nd[1] + 1e-9
    a = perturbed_params(p, Mismatch(seed=1)).K
    b = perturbed_params(p, Mismatch(seed=1, generator="independent")).K
    assert np.array_equal(a, b)
