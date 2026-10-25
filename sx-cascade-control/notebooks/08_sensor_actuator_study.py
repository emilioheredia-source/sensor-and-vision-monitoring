"""What to measure, what to move: the controller with less and less sight, and
with fewer and fewer handles, scored on transition cost.

The question: which instruments and which handles is the controller actually
using? Take them away one set at a time and see what each upset costs.

Sensor sets (the controller sees only readings through the estimator):
  full state                    the reference: the controller sees the true state
  ECJTU pair + feed analyser    two on-stream analysers (extraction, scrub) + feed analyser
  ECJTU pair                    the same, no feed analyser (the estimator infers the feed)
  product + raffinate           analysers on the two outlet streams
  product + raffinate + feed    the same plus a feed analyser
  pair + interfaces             ECJTU pair + feed analyser + interface level on every settler
  lab assay only                a product assay every 2 h, 4 h late
  nothing                       model only, no feedback

Actuator sets (full state seen):
  organic flow only; organic + scrub flow (ECJTU); + saponification; + scrub acidity;
  all of those + jacklegs (the full set)

Scenarios: feed switch, organic flowmeter bias, feed interruption.

Every run starts from the same commissioned and trimmed plant (ARCHITECTURE.md
decision 21) and gets one open-loop run per scenario as the baseline.
REPORT.md section 6 discusses the results.

Resumable: finished runs are written to the CSV at once and skipped on a rerun.

Run:  pixi run python notebooks/08_sensor_actuator_study.py [config_name] [t_end]
      config_name defaults to ndpr_plant, t_end to 40 h.
(The placement study's best pair, stages 24+26, is left out: under control it didn't finish.)
Writes runs/sensor_actuator_<config>.csv and figures/08_sensors_<config>.png, 08_actuators_<config>.png
"""
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.commission import commission, trim_under_control               # noqa: E402
from src.config import build_circuit, build_plant, load_config, nominal_inputs   # noqa: E402
from src.disturbances import FeedSwitch, FlowStep, FlowmeterBias, Scenario   # noqa: E402
from src.estimator import Estimator                                     # noqa: E402
from src.metrics import transition_cost                                 # noqa: E402
from src.mpc import CircuitMPC, MPCSettings, run_closed_loop            # noqa: E402
from src.plotting import FIGURES, INK, SLOT, style                      # noqa: E402
from src.sensors import Sensor, SensorSet                               # noqa: E402
from src.simulate import simulate, steady_state                         # noqa: E402

# ---- 1. the command line and the run file ----
HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs"
name = sys.argv[1] if len(sys.argv) > 1 else "ndpr_plant"
T_END = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
T0, T_STEP = 5.0, 0.5                    # upset time and control step, h

rc = load_config(name)
p = rc.params
SPEC, REC = rc.targets()                 # purity spec and recovery target, fractions

# ---- 2. commission and trim: the start of every run ----
# circ is the controller's model, plant what it drives, kf the plant's per-stage transfer-rate
# multipliers. The trim uses every handle and full information, like a start-up crew taking hand
# samples. x0, u (the plant's view) and u_ctl (the model's view) are where every run starts.
circ = build_circuit(rc)                                     # the controller's model
plant, kf = build_plant(rc)                                  # what it drives
t0 = time.time()
ones = np.ones(circ.N)
x_rest, u_design = commission(plant, nominal_inputs(rc, plant).replace(k_factor=kf))   # start-up on the design handles
x_ss = steady_state(circ, u_design.replace(k_factor=ones), x0=x_rest)               # the model's own rest point there
trim = CircuitMPC(circ, x_ss, u_design.replace(k_factor=ones), rc.cut, SPEC, 0.995, REC, MPCSettings())
x0, u_ctl = trim_under_control(plant, x_rest, kf, trim)      # commissioning: every handle, full information
u = u_ctl.replace(k_factor=kf)                               # the plant's view of the trimmed inputs
x_ss = steady_state(circ, u_ctl, x0=x0)                      # the model's rest point under the trimmed handles
print(f"commissioned and trimmed in {time.time()-t0:.0f}s", flush=True)
n_ext, n_scrub, N = rc["circuit"]["n_ext"], rc["circuit"]["n_scrub"], circ.N
ecjtu = (n_ext // 2, n_ext + n_scrub // 2)               # the patent's layout: mid-extraction, mid-scrub

# ---- 3. what is compared ----
# SENSOR_SETS: what the controller sees (None = the true state, no estimator).
# ACTUATOR_SETS: which handles the controller may move; it sees the true state in all of these.
# SCENARIOS: functions that build each upset fresh for every run.
product, raffinate = circ.cmap.product_stage, 0                  # the two streams that leave the circuit
SENSOR_SETS = {
    "full state": None,
    "product + raffinate analysers": [Sensor("composition_aq", stage=product), Sensor("composition_aq", stage=raffinate)],
    "product + raffinate + feed analysers": [Sensor("composition_aq", stage=product), Sensor("composition_aq", stage=raffinate),
                                             Sensor("feed_analyser")],
    "ECJTU pair + feed analyser": [Sensor("composition_aq", stage=ecjtu[0]), Sensor("composition_aq", stage=ecjtu[1]),
                                   Sensor("feed_analyser")],
    "ECJTU pair": [Sensor("composition_aq", stage=ecjtu[0]), Sensor("composition_aq", stage=ecjtu[1])],
    # the model's own best pair (stages 24 + 26 on the plant circuit, from 07) is left out of the closed
    # loop: on the perturbed plant the estimator fed by analysers that late in the scrub is the fragile
    # case of ARCHITECTURE.md decision 23, and the run did not finish in the time the others took
    "pair + interfaces": [Sensor("composition_aq", stage=ecjtu[0]), Sensor("composition_aq", stage=ecjtu[1]),
                          Sensor("feed_analyser")] + [Sensor("interface", stage=j) for j in range(N)],
    "lab assay only": [Sensor("product_assay")],
    "nothing": [],
}
ACTUATOR_SETS = {
    "organic flow": MPCSettings(handles=("Q_org",), jacklegs=False),
    "organic + scrub flow (ECJTU)": MPCSettings(handles=("Q_org", "Q_scrub"), jacklegs=False),
    "+ saponification": MPCSettings(handles=("Q_org", "Q_scrub", "saponified_fraction"), jacklegs=False),
    "+ scrub acidity": MPCSettings(handles=("Q_org", "Q_scrub", "saponified_fraction", "h_scrub"), jacklegs=False),
    "+ jacklegs (all)": MPCSettings(),
}
SCENARIOS = {
    "feed_switch": lambda: Scenario([FeedSwitch(t=T0, ramp_h=1.0, to="yangibana", params=p,
                                                removed_upstream=tuple(rc["feed"]["removed_upstream"]))]),
    "flowmeter_bias": lambda: Scenario([FlowmeterBias(t=T0, stream="organic", bias=0.05)]),
    "feed_interrupt": lambda: Scenario([FlowStep(t=T0, stream="feed", factor=0.0, duration_h=2.0)]),
}
# the transition costs kept for each run (src/metrics.py), and the CSV's columns
KEYS = ("off_spec_mol", "lost_excess_mol", "interface_excursion_m_h", "reagent_mol")
FIELDS = ["study", "set", "scenario", *KEYS, "purity_min", "qp_failures"]
CSV = RUNS / f"sensor_actuator_{name}.csv"


# ---- 4. helpers: score a run, save it, print it ----
def score(traj):
    """The transition costs in KEYS for one run, plus its lowest product purity (fraction)."""
    c = transition_cost(plant, traj, rc.cut, SPEC)
    return {k: c[k] for k in KEYS} | {"purity_min": float(c["purity_series"].min())}


# Resume: read what an earlier, interrupted run already finished. done holds the
# (study, set, scenario) keys to skip below.
rows = []
if CSV.exists():                                   # resume: keep what is done, skip it below
    with open(CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (v if k in ("study", "set", "scenario") else (float(v) if v != "" else np.nan))
                         for k, v in r.items()})
    print(f"resuming: {len(rows)} runs already in {CSV.name}", flush=True)
done = {(r["study"], r["set"], r["scenario"]) for r in rows}


def record(r):
    """Add one finished run to rows and rewrite the whole CSV, so nothing is lost if the script stops."""
    rows.append(r)
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows([{k: rr.get(k, "") for k in FIELDS} for rr in rows])


def report(r, dt):
    """Print one line for a finished run r (its scenario, study and set, then its scores) and how long
    it took (dt, s)."""
    print(f"{r['scenario']:15s} {r['study']:9s} {r['set']:28s} {dt:4.0f}s  lost {r['lost_excess_mol']:7.0f}  "
          f"iface {r['interface_excursion_m_h']:.3f}  reagent {r['reagent_mol']:.0f}  purity min {r['purity_min']:.4f}  "
          f"qp fails {int(r.get('qp_failures', 0))}", flush=True)


# ---- 5. the runs ----
# For each upset: one open-loop run, then the controller with each sensor set (all handles), then the
# controller with each handle set (true state). Each controller is new, so no run inherits another's
# state. qp_failures counts the steps where the controller's optimization didn't solve.
for sc_name, make in SCENARIOS.items():
    sc = make()
    if ("open loop", "-", sc_name) not in done:
        t0 = time.time()
        r = {"study": "open loop", "set": "-", "scenario": sc_name, "qp_failures": 0}
        r.update(score(simulate(plant, x0, sc.u_of_t(u), T_END, dt_out=T_STEP, breakpoints=sc.breakpoints)))
        record(r)
        report(r, time.time() - t0)

    # the sensor study: with a sensor set, the controller sees readings through an estimator
    # (sensor noise drawn with seed 1)
    for set_name, sensors in SENSOR_SETS.items():
        if ("sensors", set_name, sc_name) in done:
            continue
        t0 = time.time()
        ctl = CircuitMPC(circ, x_ss, u_ctl, rc.cut, SPEC, 0.995, REC, MPCSettings())
        if sensors is None:
            traj, log = run_closed_loop(plant, x0, ctl, sc, T_END, plant_k_factor=kf)
        else:
            ss = SensorSet(sensors, plant, seed=1)
            traj, log = run_closed_loop(plant, x0, ctl, sc, T_END, sensors=ss,
                                        estimator=Estimator(circ, ss, x_ss, u_ctl, T_STEP), plant_k_factor=kf)
        r = {"study": "sensors", "set": set_name, "scenario": sc_name}
        r.update(score(traj))
        r["qp_failures"] = sum(not h["qp_ok"] for h in log)
        record(r)
        report(r, time.time() - t0)

    # the handle study: the same controller, allowed fewer handles
    for set_name, settings in ACTUATOR_SETS.items():
        if ("actuators", set_name, sc_name) in done:
            continue
        t0 = time.time()
        ctl = CircuitMPC(circ, x_ss, u_ctl, rc.cut, SPEC, 0.995, REC, settings)
        traj, log = run_closed_loop(plant, x0, ctl, sc, T_END, plant_k_factor=kf)
        r = {"study": "actuators", "set": set_name, "scenario": sc_name}
        r.update(score(traj))
        r["qp_failures"] = sum(not h["qp_ok"] for h in log)
        record(r)
        report(r, time.time() - t0)

# ---- 6. figures: one per study; product metal lost beyond nominal and the lowest purity, side by side ----
style()


def pick(study, lab, sc_name):
    """The row for one run: study ("sensors" or "actuators"), set label, scenario. None if it isn't there.

    The label "open loop" picks that scenario's open-loop row, whatever the study.
    """
    if lab == "open loop":
        return next((r for r in rows if r["study"] == "open loop" and r["scenario"] == sc_name), None)
    return next((r for r in rows if r["study"] == study and r["set"] == lab and r["scenario"] == sc_name), None)


# One horizontal bar per scenario for each set, open loop at the top. Left panel: product metal lost
# beyond nominal, mol. Right panel: the lowest purity, in points above the spec (negative = below).
for study, sets in (("sensors", SENSOR_SETS), ("actuators", ACTUATOR_SETS)):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"wspace": 0.06, "width_ratios": [1.3, 1]},
                             sharey=True)
    labels = ["open loop"] + list(sets)
    y = np.arange(len(labels))
    width = 0.8 / len(SCENARIOS)
    for k, sc_name in enumerate(SCENARIOS):
        lost = [(pick(study, lab, sc_name) or {}).get("lost_excess_mol", np.nan) for lab in labels]
        pmin = [(pick(study, lab, sc_name) or {}).get("purity_min", np.nan) for lab in labels]
        axes[0].barh(y + (k - 1) * width, lost, height=width * 0.9, color=SLOT[k], label=sc_name.replace("_", " "))
        axes[1].barh(y + (k - 1) * width, 100 * (np.array(pmin, float) - SPEC), height=width * 0.9, color=SLOT[k])
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel(f"{'Nd' if rc.cut == 'Pr' else 'NdPr'} lost beyond nominal over the run, mol")
    axes[1].set_xlabel(f"lowest purity, points above the {SPEC:.0%} spec")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3, frameon=False)
    fig.suptitle(f"{name}: {'what each sensor set buys' if study == 'sensors' else 'what each handle set buys'} "
                 f"({T_END:.0f} h runs)", x=0.01, ha="left", color=INK)
    fig.savefig(FIGURES / f"08_{study}_{name}.png")
print("wrote", CSV)
