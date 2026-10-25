"""Does the model's accuracy change what the plant does? Parts B, C and D of the Oct 23 check.

The question. In notebook 43 every estimator gave about the same closed-loop result at 120 h. The
audit (notebook 47) found why that wasn't a test: the badly wrong estimates all landed on easy
plants, end values converge to the targets by design (the output bias is integral action), and
reagent wasn't measured. So, on every test plant, hard ones included:

  Part B  measure what wasn't measured: purity shortfall below spec (points x hours), time to reach
          spec and stay there, reagent (saponification base, and acid to scrub and strip) over the
          run, the final handles and the full handle trajectories, runs of 400 h, and whether the
          handles have stopped moving by the end.
  Part C  dose-response: the controller on the plant's own model (its numbers, its stage
          efficiencies) with only the Pr/(La+Ce) factor wrong by 0, +-0.1, +-0.2, +-0.4 in log
          (+ means the model thinks the plant separates better than it does). A flat line all the
          way out is robustness; identical outputs would be a bug.
  Part D  the crutches taken away one at a time: the output bias off (filter 0) or slow (0.1);
          the feed switch to the rich deposit (80 h); the edge design (ndpr_plant_sized: 99.5 % /
          95 % targets, no margin) instead of the main case.
          (Part D1, starting from handles set on the model, is already how every run starts: the
          plant is commissioned on the design's flows, which were optimized on the nominal model.)

Every run starts with the plant at rest on the design handles (commissioned, then settled for
800 h), the controller's model at its own rest point there, the MPC over level loops. Test plants
1 to 10, generator "ratios", as notebooks 35 and 43.

Workers run one thread each (every run here starts from a plain steady-state solve, no trim, so the
thread count doesn't move its starting point), and study_tools.run_jobs restarts a stuck pool.

Run:  pixi run python notebooks/48_does_the_estimate_matter.py C | D_bias | D_switch | D_edge
Writes runs/estimate_matters_<part>.csv (one row per run) and runs/estimate_matters_<part>.npz
(handles, purity and recovery over time, keyed <seed>_<offset>_<setting>_<series>).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"                       # one thread per worker process (see the docstring)

import dataclasses                                                        # noqa: E402
import sys                                                                # noqa: E402
import time                                                               # noqa: E402
import warnings                                                           # noqa: E402
from pathlib import Path                                                  # noqa: E402

import numpy as np                                                        # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "notebooks"))
from src.circuit import Circuit, Inputs                                   # noqa: E402
from src.commission import commission                                    # noqa: E402
from src.config import load_config, nominal_inputs                       # noqa: E402
from src.loop import run_closed_loop                                     # noqa: E402
from src.metrics import point_outputs                                    # noqa: E402
from src.registry import PLANTS, mpc_over_level_loops                    # noqa: E402
from src.scenarios import catalogue                                      # noqa: E402
from study_tools import run_jobs, settle, workers, write_rows            # noqa: E402

SEEDS = tuple(range(1, 11))
WORKERS = workers()
HANDLES = ("Q_org", "Q_scrub", "h_scrub", "saponified_fraction")
PARTS = {   # part: (run file, upset, hours, Pr/(La+Ce) offsets in log, controller settings to vary)
    "C": ("ndpr_main", None, 400.0, (0.0, 0.1, -0.1, 0.2, -0.2, 0.4, -0.4), ({},)),
    "D_bias": ("ndpr_main", None, 400.0, (0.0, 0.2, -0.2, 0.4, -0.4), ({"bias_filter": 0.0}, {"bias_filter": 0.1})),
    "D_switch": ("ndpr_main", "feed_switch", 80.0, (0.0, 0.1, -0.1, 0.2, -0.2), ({},)),
    "D_edge": ("ndpr_plant_sized", None, 400.0, (0.0, 0.2, -0.2, 0.4, -0.4), ({},)),
}


# ---- the plant, the model, the run -----------------------------------------------------------------
def plant_and_model(config: str, seed: int, offset: float):
    """The perturbed plant for this seed, and its own model with the Pr/(La+Ce) factor moved by
    `offset` in log (the La+Ce constant changed; everything else, stage efficiencies included, true).

    Returns (rc, plant, plant stage efficiencies, model)."""
    rc = load_config(config, overrides={"plant": {"mismatch": "perturbed", "seed": seed, "generator": "ratios"}})
    plant, pk = PLANTS["perturbed"].build(rc)
    p = plant.p
    K = np.asarray(p.K, float).copy()
    K[list(p.names).index("LaCe")] *= np.exp(-offset)          # beta = K_Pr / K_LaCe goes up by exp(offset)
    model = Circuit(plant.cmap, dataclasses.replace(p, K=K), model=plant.model, scrub_reflux=plant.scrub_reflux)
    return rc, plant, np.asarray(pk, float), model


def start_point(rc, plant, pk):
    """The plant at rest on the design handles, and those inputs as the controller knows them (k = 1)."""
    x, u = commission(plant, nominal_inputs(rc, plant).replace(k_factor=pk))
    return settle(plant, u, x0=x), u.replace(k_factor=np.ones(plant.N))


def measures(rc, plant, traj, hours) -> tuple:
    """Part B's measures from a closed-loop trajectory (purity and recovery as notebook 41 reads them).
    Returns (the row's measures, the traces: t, purity, recovery and each handle)."""
    t, U = np.asarray(traj.t), np.asarray(traj.u)
    out = [point_outputs(plant, x, uv, rc.cut) for x, uv in zip(traj.x, U)]
    pur, rec = np.array([o["purity"] for o in out]), np.array([o["recovery"] for o in out])
    lay = Inputs.layout(plant.n, plant.N)
    col = {k: U[:, lay[k].start] for k in HANDLES + ("Q_strip", "h_strip")}
    rest = point_outputs(plant, settle(plant, U[-1], x0=traj.x[-1]), U[-1], rc.cut)
    row = {"purity_end": pur[-1], "recovery_end": rec[-1], "rest_purity": rest["purity"], "rest_recovery": rest["recovery"],
           **quality_measures(rc, t, pur, rec), **reagent_measures(plant, t, col),
           **{f"{k}_end": float(col[k][-1]) for k in HANDLES}, "handles_moving_last_50h": handles_moving(t, col, hours)}
    return row, {"t": t, "purity": pur, "recovery": rec, **{k: col[k] for k in HANDLES}}


def quality_measures(rc, t, pur, rec) -> dict:
    """Minimum purity, hours below spec, purity shortfall below spec (points x hours), time to reach
    spec and stay there (nan if it ends below), minimum recovery and its shortfall below target."""
    spec, rec_target = rc.targets()
    dt = np.diff(t)
    off = np.nonzero(pur < spec)[0]                      # time to spec: just after the last time below it
    to_spec = float("nan") if pur[-1] < spec else (0.0 if len(off) == 0 else float(t[off[-1] + 1]))
    return {"purity_min": pur[1:].min(), "hours_below_spec": float(dt[pur[1:] < spec].sum()),
            "purity_shortfall_pt_h": float(np.sum(np.maximum(spec - pur[1:], 0.0) * dt) * 100),
            "hours_to_spec": to_spec, "recovery_min": rec[1:].min(),
            "recovery_shortfall_pt_h": float(np.sum(np.maximum(rec_target - rec[1:], 0.0) * dt) * 100)}


def reagent_measures(plant, t, col: dict) -> dict:
    """Base (saponification) and acid (strip and scrub) over the run, mol, and their rates at the end, mol/h."""
    dt = np.diff(t)
    base = col["saponified_fraction"] * plant.p.raw["extractant"]["concentration_monomer"] * col["Q_org"]
    acid = col["Q_strip"] * col["h_strip"] + col["Q_scrub"] * col["h_scrub"]
    return {"base_mol": float(np.sum(base[:-1] * dt)), "acid_mol": float(np.sum(acid[:-1] * dt)),
            "base_end_mol_h": float(base[-1]), "acid_end_mol_h": float(acid[-1])}


def handles_moving(t, col: dict, hours: float) -> float:
    """The largest relative range of any handle over the last 50 h: how far from settled the controller is."""
    late = t >= hours - 50.0
    return float(max(np.ptp(col[k][late]) / max(abs(np.mean(col[k][late])), 1e-9) for k in HANDLES))


def run(job):
    """One closed-loop run: (part, seed, offset, settings index). Returns (row, traces)."""
    part, seed, offset, si = job
    warnings.filterwarnings("ignore")
    config, upset, hours, _, settings_list = PARTS[part]
    settings = dict(settings_list[si])
    rc, plant, pk, model = plant_and_model(config, seed, offset)
    x0, u0 = start_point(rc, plant, pk)
    u_model = u0.replace(k_factor=pk)                       # the model carries the plant's own stage efficiencies
    x_model = settle(model, u_model, x0=plant.observable(x0))
    if config != "ndpr_main":
        settings.setdefault("acid_floor", 1e-3)             # the main case's acidity floor, kept on the edge design
    ctl = mpc_over_level_loops(rc, model, x_model, u_model, **settings)
    scenario = catalogue(rc, u_model)[upset].scenario() if upset else None
    t0 = time.time()
    base = {"part": part, "seed": seed, "offset": offset, "setting": "default" if not settings_list[si] else
            " ".join(f"{k}={v}" for k, v in settings_list[si].items())}
    try:
        traj, log = run_closed_loop(plant, x0, ctl, scenario, hours, plant_k_factor=None)
    except RuntimeError as err:                         # the plant's integration stalled (simulate.MAX_RHS_CALLS)
        return {**base, "failed": 1, "error": str(err)[:200], "seconds": round(time.time() - t0)}, {}
    row, traces = measures(rc, plant, traj, hours)
    row = {**base, **row,
           "solver_failures": int(sum(1 for r in log if r.get("qp_ok") is False)), "seconds": round(time.time() - t0)}
    return row, traces


# ---- the study -----------------------------------------------------------------------------------
def report(job, result, rows: list, traces: dict) -> None:
    """Add one run's result to the rows and traces, and log it (a hang or a stalled plant as failed)."""
    if result is None:
        print(f"{job}: hung twice, recorded as failed", flush=True)
        rows.append({"part": job[0], "seed": job[1], "offset": job[2], "setting": str(job[3]), "failed": 1})
        return
    row, tr = result
    rows.append(row)
    if row.get("failed"):
        print(f"{job}: failed, {row['error']}", flush=True)
        return
    traces.update({f"{job[1]}_{job[2]:+.1f}_{job[3]}_{k}": v for k, v in tr.items()})
    print(f"seed {row['seed']:2d} offset {row['offset']:+.1f} {row['setting']:16s}: purity end {100 * row['purity_end']:.2f} "
          f"(rest {100 * row['rest_purity']:.2f}, min {100 * row['purity_min']:.2f}, short {row['purity_shortfall_pt_h']:.1f} pt.h) "
          f"recovery end {100 * row['recovery_end']:.2f} (rest {100 * row['rest_recovery']:.2f}) base {row['base_mol']:.0f} "
          f"acid {row['acid_mol']:.0f} mol, moving {100 * row['handles_moving_last_50h']:.2f} %", flush=True)


def main(part: str) -> None:
    """Every plant, offset and setting of one part; rows and traces written as each run finishes."""
    _, _, _, offsets, settings_list = PARTS[part]
    jobs = [(part, s, o, si) for si in range(len(settings_list)) for o in offsets for s in SEEDS]
    out, out_npz = HERE / "runs" / f"estimate_matters_{part}.csv", HERE / "runs" / f"estimate_matters_{part}.npz"
    rows, traces = [], {}
    for job, result in run_jobs(run, jobs, WORKERS):
        report(job, result, rows, traces)
        write_rows(out, rows, first=("part", "seed", "offset", "setting"))
        np.savez(out_npz, **traces)
    print("wrote", out.name)


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if a in PARTS][0])
