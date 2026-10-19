"""Reference runs for the refactor: the numbers the restructured code must reproduce.

What it's for
-------------
A refactor should change how the code is laid out, not what it computes. The
unit tests check behaviour against tolerances, so a small change in the
numbers could slip past them. This script runs a fixed set of real runs
(commissioning and trim, open loop, the MPC on three upsets, the MPC with two
sensor sets, the MPC with one handle, and the design optimizer) and saves
every resulting array to baseline.npz. After a refactor it runs them all
again and compares, array by array. The arrays were recorded on the code
before the refactor, so any difference means the refactor changed behaviour.

"The same" here means no element of any array differs from the recorded one
by more than 1e-7, relative. That's far tighter than any tolerance in the
unit tests: the refactored code should give the same numbers, not close ones.

How to use it
-------------
Recorded once on the code before the refactor (commit 65b094f), then checked
after each step:
    pixi run python tests/baseline/make_baseline.py write     # record (only on the old code)
    pixi run python tests/baseline/make_baseline.py check     # compare the current code

`write` overwrites baseline.npz. Only run it on code you trust; on refactored
code it would just record whatever that code does now. `check` is the default
when no mode is given. It prints the largest relative difference for each
array, then SAME, or DIFFERENT with the worst difference.

    pixi run python tests/baseline/make_baseline.py check --with-runs

adds one more comparison: the same MPC feed-switch run done through the new
front door, runs.execute (the path the command line and the workbench use).
Its state history has to match the recorded "full_feed_switch" array, which
was made by wiring the pieces together by hand. That shows the new run
machinery builds the same run the old code did.

The whole thing takes a while: every run is 40 h of plant time on ndpr_plant,
the full-size NdPr circuit (8 extraction, 20 scrub, 4 strip stages).

Uses only the API every version keeps: load_config, build_circuit/build_plant, commission,
trim_under_control, CircuitMPC, run_closed_loop, SensorSet, Estimator, optimize_operating_point.
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# make the project folder importable, so `src` is found when this is run as a script
sys.path.insert(0, str(HERE.parent.parent))
from src.commission import commission, trim_under_control                     # noqa: E402
from src.config import build_circuit, build_plant, load_config, nominal_inputs  # noqa: E402
from src.design import optimize_operating_point, reflux_link                  # noqa: E402
from src.disturbances import FeedSwitch, FlowmeterBias, FlowStep, Scenario    # noqa: E402
from src.estimator import Estimator                                           # noqa: E402
from src.mpc import CircuitMPC, MPCSettings, run_closed_loop                  # noqa: E402
from src.sensors import Sensor, SensorSet                                     # noqa: E402
from src.simulate import simulate, steady_state                               # noqa: E402

FILE = HERE / "baseline.npz"      # the recorded arrays, one entry per name in results()
T_END = 40.0                      # h, length of every closed- and open-loop run


def prepared(name):
    """Commission the plant for run file `name` and trim it under the MPC, as a run starts.

    The plant (perturbed from the model, per the run file) is run up on the
    model's design inputs and its jacklegs are set. The MPC, seeing the full
    state, then trims the handles until the product is on spec. Last, the
    model's own steady state at the trimmed inputs is found: that's where the
    controllers in results() linearize.

    Returns (run config, purity spec, recovery target, model circuit, plant,
    plant stage efficiencies kf, plant state after the trim x0, trimmed inputs
    u_ctl, model steady state x_ss).
    """
    rc = load_config(name)
    spec, rec = rc.targets()
    circ = build_circuit(rc)
    plant, kf = build_plant(rc)
    ones = np.ones(circ.N)                    # nominal stage efficiency: what the model assumes
    x_rest, u_design = commission(plant, nominal_inputs(rc, plant).replace(k_factor=kf))
    x_ss = steady_state(circ, u_design.replace(k_factor=ones), x0=x_rest)
    # the trim controller holds purity at 0.995 (above the spec) and recovery at the target
    trim = CircuitMPC(circ, x_ss, u_design.replace(k_factor=ones), rc.cut, spec, 0.995, rec, MPCSettings())
    x0, u_ctl = trim_under_control(plant, x_rest, kf, trim)
    x_ss = steady_state(circ, u_ctl, x0=x0)
    return rc, spec, rec, circ, plant, kf, x0, u_ctl, x_ss


def results():
    """Run every reference case and return {array name: numpy array}.

    The names are the keys in baseline.npz:
      trim_x0, trim_u                 plant state and inputs after commissioning and trim
      open_feed_switch                plant states, feed switch, nobody touching the handles
      full_<upset>, full_<upset>_u    states and inputs, MPC with full state, for the feed
                                      switch, a 2 h feed interruption and a 5 % organic
                                      flowmeter bias
      sensors_prodraff_feed           states, MPC + estimator, analysers on product, raffinate, feed
      sensors_ecjtu                   states, MPC + estimator, analysers mid-extraction and mid-scrub
      actuators_org_only              states, MPC moving the organic flow only, no jacklegs
      design_small_x, design_small_v  the optimizer's steady state on ndpr_small, and its
                                      Q_org, Q_scrub, h_scrub, saponified fraction and recovery
    Every upset starts at t = 5 h.
    """
    out = {}
    rc, spec, rec, circ, plant, kf, x0, u_ctl, x_ss = prepared("ndpr_plant")
    out["trim_x0"], out["trim_u"] = x0, u_ctl.numpy()
    p = rc.params
    # the three upsets: feed switch Eneabba -> Yangibana over 1 h, feed flow to zero for 2 h,
    # and the organic pump delivering 5 % more than its flowmeter says
    scen = {"feed_switch": Scenario([FeedSwitch(t=5.0, ramp_h=1.0, to="yangibana", params=p,
                                                removed_upstream=tuple(rc["feed"]["removed_upstream"]))]),
            "feed_interrupt": Scenario([FlowStep(t=5.0, stream="feed", factor=0.0, duration_h=2.0)]),
            "flowmeter_bias": Scenario([FlowmeterBias(t=5.0, stream="organic", bias=0.05)])}
    # the trimmed inputs with the plant's own stage efficiencies, for the plant simulations
    u = u_ctl.replace(k_factor=kf)
    # open loop: the feed switch with the handles held, state saved every 0.5 h
    ol = simulate(plant, x0, scen["feed_switch"].u_of_t(u), T_END, dt_out=0.5, breakpoints=scen["feed_switch"].breakpoints)
    out["open_feed_switch"] = ol.x
    # the MPC with full state on each upset; a fresh controller each time
    for key, sc in scen.items():
        ctl = CircuitMPC(circ, x_ss, u_ctl, rc.cut, spec, 0.995, rec, MPCSettings())
        traj, log = run_closed_loop(plant, x0, ctl, sc, T_END, plant_k_factor=kf)
        out[f"full_{key}"] = traj.x
        out[f"full_{key}_u"] = traj.u
    # the MPC with an estimator and two sensor sets, on the feed switch
    n_ext, n_scrub = rc["circuit"]["n_ext"], rc["circuit"]["n_scrub"]
    sets = {"prodraff_feed": [Sensor("composition_aq", stage=circ.cmap.product_stage), Sensor("composition_aq", stage=0),
                              Sensor("feed_analyser")],
            "ecjtu": [Sensor("composition_aq", stage=n_ext // 2), Sensor("composition_aq", stage=n_ext + n_scrub // 2)]}
    for key, sensors in sets.items():
        ctl = CircuitMPC(circ, x_ss, u_ctl, rc.cut, spec, 0.995, rec, MPCSettings())
        # seed 1 fixes the sensor noise, so the run repeats exactly
        ss = SensorSet(sensors, plant, seed=1)
        traj, log = run_closed_loop(plant, x0, ctl, scen["feed_switch"], T_END, sensors=ss,
                                    estimator=Estimator(circ, ss, x_ss, u_ctl, 0.5), plant_k_factor=kf)
        out[f"sensors_{key}"] = traj.x
    # the MPC allowed to move only the organic flow (jacklegs fixed), on the feed switch
    ctl = CircuitMPC(circ, x_ss, u_ctl, rc.cut, spec, 0.995, rec, MPCSettings(handles=("Q_org",), jacklegs=False))
    traj, _ = run_closed_loop(plant, x0, ctl, scen["feed_switch"], T_END, plant_k_factor=kf)
    out["actuators_org_only"] = traj.x
    # the design optimizer on ndpr_small, purity 0.995, reflux scrub with 300 L/h net product
    rc_s = load_config("ndpr_small")
    c_s = build_circuit(rc_s)
    # bounds: flows L/h (organic capped at 85 % of the 4700 L/h pump), scrub acid mol/L, fraction 0..1
    free = {"Q_org": (600.0, 0.85 * 4700.0), "Q_scrub": (50.0, 2000.0), "h_scrub": (0.05, 2.0), "saponified_fraction": (0.0, 0.6)}
    r = optimize_operating_point(c_s, nominal_inputs(rc_s, c_s), rc_s.cut, 0.995, free, link=reflux_link(300.0))
    out["design_small_x"] = r.x
    out["design_small_v"] = np.array([r.u.Q_org, r.u.Q_scrub, r.u.h_scrub, r.u.saponified_fraction, r.recovery])
    return out


def main():
    """Record (`write`) or compare (`check`, the default; add `--with-runs` for runs.execute too)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    t0 = time.time()
    now = results()
    print(f"computed {len(now)} arrays in {time.time()-t0:.0f}s")
    if mode == "write":
        np.savez(FILE, **now)
        print("wrote", FILE)
        return
    ref = dict(np.load(FILE))
    if len(sys.argv) > 2 and sys.argv[2] == "--with-runs":     # the new path: runs.execute end to end
        from src.runs import RunSpec, execute
        r = execute(RunSpec(config="ndpr_plant", plant="perturbed", controller="mpc", upset="feed_switch",
                            sensors="full_state", hours=T_END))
        # compared against the hand-wired full-state feed-switch run recorded above
        now["runs_full_feed_switch"] = r.x
        ref["runs_full_feed_switch"] = ref["full_feed_switch"]
    # worst relative difference over every recorded array; inf if any array changed shape
    worst = 0.0
    for k in ref:
        a, b = ref[k], now[k]
        if a.shape != b.shape:
            print(f"{k:24s} SHAPE {a.shape} vs {b.shape}")
            worst = np.inf
            continue
        # largest element-wise relative difference; the 1e-9 keeps zeros from dividing by zero
        rel = float(np.max(np.abs(a - b) / (np.abs(a) + 1e-9)))
        worst = max(worst, rel)
        print(f"{k:24s} max rel diff {rel:.2e}")
    print("SAME" if worst < 1e-7 else f"DIFFERENT (worst {worst:.2e})")


if __name__ == "__main__":
    main()
