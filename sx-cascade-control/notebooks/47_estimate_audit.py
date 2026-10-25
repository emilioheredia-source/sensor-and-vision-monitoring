"""Does the estimate reach the controller, and why do the closed-loop results look the same? (Part A)

The question (Oct 23). In notebook 43's Tests 1 and 2 every estimator gave about the same closed-loop
result at 120 h, including a fit that was 0.29-0.44 off on Pr/(La+Ce). Before reading that as
robustness: an audit of the code path and the records.

  factors   rebuild each Test 1 controller exactly as closed_loop() does (the estimate's model, its
            rest point at the design handles, mpc_over_level_loops) and read the separation factors
            off the controller's own circuit
  bias0     the output bias each controller would start from: what the plant reports at its rest
            point minus what the controller's model gives at its own (purity and recovery)
  traces    plants 1 and 3 of Test 1 rerun 120 h with the true model and with the fit as built, the
            handles recorded every step, to compare the trajectories

Run:  pixi run python notebooks/47_estimate_audit.py factors | bias0 | traces
Writes runs/estimate_audit_<part>_ndpr_main.csv (and .npz for traces).
"""
import csv
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "notebooks"))
from src.circuit import Inputs                                          # noqa: E402
from src.config import build_circuit                                    # noqa: E402
from src.loop import run_closed_loop                                    # noqa: E402
from src.registry import mpc_over_level_loops                            # noqa: E402
from study_tools import (HOURS, PlantCase, load_notebook, notebook_35, purity_recovery, run_jobs,   # noqa: E402
                         separation_factors, settle, workers, write_rows)

CONFIG = "ndpr_main"
SEEDS = tuple(range(1, 11))
HANDLES = ("Q_org", "Q_scrub", "h_scrub", "saponified_fraction")
ESTIMATES = ("true", "fit", "net", "ridge")


@dataclass
class Audit:
    """One Test 1 plant rebuilt as notebook 43 builds it, with each estimate's three log numbers.

    case: the plant at rest. m: notebook 31 (its model builder). thetas: {estimate: (log Pr/(La+Ce),
    log Nd/Pr, log shared rate)}. row: the plant's row of runs/fair_fit_test1_ndpr_main.csv.
    """
    case: PlantCase
    m: object
    thetas: dict
    row: dict


def setup(seed: int) -> Audit:
    """Test 1's plant for this seed (notebook 43's sanity_case) and the estimates read back from its row
    (the net's and the ridge's as logs of their factors, shared rate at 1)."""
    warnings.filterwarnings("ignore")
    n43 = load_notebook("nb43", "43_fair_fit.py", ("--config", CONFIG))
    n35 = notebook_35(CONFIG)
    m = n35.nb31()
    case, theta_true, _ = n43.sanity_case(n35, m, seed)
    row = next(r for r in csv.DictReader(open(HERE / "runs" / f"fair_fit_test1_{CONFIG}.csv")) if int(r["seed"]) == seed)
    b0 = separation_factors(build_circuit(case.rc))
    thetas = {"true": theta_true}
    for name in ("fit", "net", "ridge"):
        k = np.log(float(row["k_fit"])) if name == "fit" else 0.0
        thetas[name] = np.array([np.log(float(row[f"beta_pl_{name}"]) / b0[0]), np.log(float(row[f"beta_np_{name}"]) / b0[1]), k])
    return Audit(case, m, thetas, row)


def controller_for(audit: Audit, name: str):
    """The controller closed_loop() would build on this estimate: (controller, its model)."""
    model, _, _ = audit.m.model_with(audit.case.rc, audit.thetas[name])
    x_model = settle(model, audit.case.u_design, x0=audit.case.plant.observable(audit.case.x_rest))
    return mpc_over_level_loops(audit.case.rc, model, x_model, audit.case.u_design), model


def factors_one(seed: int) -> dict:
    """The Pr/(La+Ce) factor and shared rate inside each estimate's controller, next to the stored estimate."""
    audit = setup(seed)
    k_nominal = build_circuit(audit.case.rc).p.k_transfer
    out = {"seed": seed}
    for name in ESTIMATES:
        ctl, _ = controller_for(audit, name)
        out[f"{name}_beta_pl_in_controller"] = separation_factors(ctl.circuit)[0]
        out[f"{name}_beta_pl_in_csv"] = float(audit.row[f"beta_pl_{name}"])
        out[f"{name}_k_transfer_ratio"] = float(ctl.circuit.p.k_transfer / k_nominal)
    return out


def bias0_one(seed: int) -> dict:
    """The gap the bias has to carry at the start: plant at its rest point minus each model at its own."""
    audit = setup(seed)
    case = audit.case
    yp = purity_recovery(case.plant, case.x_rest, case.u_design.numpy(), case.rc.cut)
    out = {"seed": seed, "plant_purity": yp[0], "plant_recovery": yp[1]}
    for name in ESTIMATES:
        model, _, _ = audit.m.model_with(case.rc, audit.thetas[name])
        xm = settle(model, case.u_design, x0=case.plant.observable(case.x_rest))
        ym = purity_recovery(model, xm, case.u_design.numpy(), case.rc.cut)
        out[f"{name}_model_purity"], out[f"{name}_model_recovery"] = ym
        out[f"{name}_bias0_purity"], out[f"{name}_bias0_recovery"] = yp[0] - ym[0], yp[1] - ym[1]
    return out


def traces_one(job):
    """One plant and estimate rerun 120 h: (seed, estimate, times, handles over time, the bias over time)."""
    seed, name = job
    audit = setup(seed)
    ctl, _ = controller_for(audit, name)
    plant = audit.case.plant
    traj, log = run_closed_loop(plant, audit.case.x_rest, ctl, None, HOURS, plant_k_factor=np.ones(plant.N))
    lay = Inputs.layout(plant.n, plant.N)
    U = np.asarray(traj.u)
    bias = np.array([r.get("bias_purity", np.nan) for r in log]), np.array([r.get("bias_recovery", np.nan) for r in log])
    return seed, name, np.asarray(traj.t), {k: U[:, lay[k].start] for k in HANDLES}, bias


def traces():
    """Plants 1 and 3 with the true model and the fit as built; the trajectories to one .npz."""
    data = {}
    for _, (seed, name, t, handles, (bp, br)) in run_jobs(traces_one, [(s, n) for s in (1, 3) for n in ("true", "fit")], workers()):
        data[f"{seed}_{name}_t"] = t
        data.update({f"{seed}_{name}_{k}": v for k, v in handles.items()})
        data[f"{seed}_{name}_bias_purity"], data[f"{seed}_{name}_bias_recovery"] = bp, br
    np.savez(HERE / "runs" / f"estimate_audit_traces_{CONFIG}.npz", **data)
    print("wrote estimate_audit_traces")


def table(fn, part: str):
    """fn on every plant, the rows to runs/estimate_audit_<part>_ndpr_main.csv in plant order."""
    rows = sorted((r for _, r in run_jobs(fn, SEEDS, workers())), key=lambda r: r["seed"])
    write_rows(HERE / "runs" / f"estimate_audit_{part}_{CONFIG}.csv", rows, first=("seed",))
    print("wrote", part)


if __name__ == "__main__":
    part = [a for a in sys.argv[1:] if a in ("factors", "bias0", "traces")][0]
    if part == "traces":
        traces()
    else:
        table({"factors": factors_one, "bias0": bias0_one}[part], part)
