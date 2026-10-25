"""What the fair-test studies share (notebooks 43 to 50), one copy of each.

Where this sits
---------------
Notebooks 43 to 50 test how the physics fit and the learned estimators compare, and whether the
model's accuracy changes what the plant does. They need the same few pieces: a model with chosen
numbers changed, a steady state solved all the way to rest, the closed loop with its measures, a
notebook loaded as a module, and a pool of worker processes that can't hang a study overnight.
Kept here so a fix lands in every study at once.

Nothing here starts any math threads by itself. Each notebook sets its own thread count at its very
top, before numpy loads, because the trimmed operating point depends on it (FINDINGS phase 9).
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import importlib.util
import os
import sys
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from src.circuit import Circuit, Inputs                                  # noqa: E402
from src.config import RunConfig, build_circuit                          # noqa: E402
from src.loop import run_closed_loop                                     # noqa: E402
from src.metrics import heavy_mask, product_purity                      # noqa: E402
from src.registry import mpc_over_level_loops                            # noqa: E402
from src.simulate import steady_state                                    # noqa: E402

HOURS = 120.0                                  # h, the closed-loop runs of notebooks 43 to 47
SPEC = 0.99                                    # the purity spec, as a fraction
# Plausible spread of each candidate number, from src/mismatch.py ("ratios" generator): separation
# factors half their drawn range in log, the common factor on every K half its range, the acid exponent
# the plant's offset, stage rate the mean of U(0.85, 1) in log, hold-up 5 %, organic density 2 %,
# salt molal volume 10 %. Log factors, except dn_acid (added to the exponent).
SPREAD = {"beta_pl": 0.196, "beta_np": 0.196, "common_K": 0.100, "dn_acid": 0.3, "k_ext": 0.078,
          "k_scrub": 0.078, "k_strip": 0.078, "V_mix": 0.05, "rho_org": 0.02, "phi_v": 0.10}
_SECTION_RATE = {"extraction": "k_ext", "scrub": "k_scrub", "strip": "k_strip"}


def workers(threads_each: int = 1) -> int:
    """Worker processes for a pool: every core but two, divided by each worker's threads, at least one."""
    return max(1, ((os.cpu_count() or 1) - 2) // threads_each)


# ---- loading another notebook ------------------------------------------------------------------
def load_notebook(name: str, filename: str, argv: tuple = ()):
    """Another notebook in this folder as a module, run with the given command-line arguments.

    Its study only runs under __main__, so loading it just defines its functions and settings.
    Example: n35 = load_notebook("nb35", "35_learned_estimator.py", ("--config", "ndpr_main"))
    """
    spec = importlib.util.spec_from_file_location(name, HERE / "notebooks" / filename)
    module = importlib.util.module_from_spec(spec)
    old, sys.argv = sys.argv, [sys.argv[0], *argv]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = old
    return module


def notebook_35(config: str):
    """Notebook 35 on this run file, with its pickled Scaler class findable.

    Notebook 35 ran as a script, so its stored ridge names the class __main__.Scaler; it's put there."""
    module = load_notebook("nb35", "35_learned_estimator.py", ("--config", config))
    import __main__
    __main__.Scaler = module.Scaler
    return module


# ---- the model with chosen numbers changed ------------------------------------------------------
def extended_model(rc: RunConfig, theta: dict) -> tuple[Circuit, np.ndarray]:
    """The nominal model with candidate numbers changed, and the stage rates that go with it.

    theta: {candidate: value}, each a log factor on the model's value except dn_acid, which is added
        to the acid exponent. Candidates are SPREAD's keys plus "k_all" (one rate factor for every
        stage, as notebook 31 fits). Missing ones stay nominal.
    Returns (circuit, k_factor per stage): the section rates go in the inputs' k_factor, everything
    else in the circuit's parameters.
    """
    base = build_circuit(rc)
    p = base.p
    raw = copy.deepcopy(p.raw)
    raw["stage_geometry"]["mixer_volume_L"] *= np.exp(theta.get("V_mix", 0.0))
    p2 = dataclasses.replace(p, K=_constants(p, theta), raw=raw, n_acid=p.n_acid + theta.get("dn_acid", 0.0),
                             rho_org0=p.rho_org0 * np.exp(theta.get("rho_org", 0.0)),
                             phi_v=p.phi_v * np.exp(theta.get("phi_v", 0.0)))
    circ = Circuit(base.cmap, p2, model=base.model, scrub_reflux=base.scrub_reflux)
    k = np.array([np.exp(theta.get(_SECTION_RATE[circ.cmap.section(j)], 0.0) + theta.get("k_all", 0.0))
                  for j in range(circ.N)])
    return circ, k


def _constants(p, theta: dict) -> np.ndarray:
    """The extraction constants with the common factor and the two separation factors changed.
    Pr moves with the common factor; La+Ce and Nd follow from the separation factors; Sm keeps its
    ratio to Nd."""
    names = list(p.names)
    la, pr, nd, sm = (names.index(c) for c in ("LaCe", "Pr", "Nd", "SmPlus"))
    K0 = np.asarray(p.K, float)
    K = K0.copy()
    K[pr] = K0[pr] * np.exp(theta.get("common_K", 0.0))
    K[la] = K[pr] / (K0[pr] / K0[la] * np.exp(theta.get("beta_pl", 0.0)))
    K[nd] = K[pr] * K0[nd] / K0[pr] * np.exp(theta.get("beta_np", 0.0))
    K[sm] = K[nd] * K0[sm] / K0[nd]
    return K


def separation_factors(circ: Circuit) -> np.ndarray:
    """(Pr/(La+Ce), Nd/Pr) of a circuit's extraction constants."""
    names = list(circ.p.names)
    K = np.asarray(circ.p.K, float)
    return np.array([K[names.index("Pr")] / K[names.index("LaCe")], K[names.index("Nd")] / K[names.index("Pr")]])


# ---- steady states ----------------------------------------------------------------------------------
def settle(circ: Circuit, u, x0=None) -> np.ndarray:
    """A steady state solved all the way to rest.

    Newton from x0 first; if that fails, 800 h of integration before the polish. Why: the main case's
    slowest mode outlives simulate.steady_state's default 250 h (at the model's own design point that
    ladder stops at |dx/dt| 7.9e-4, this one at 7e-12; FINDINGS phase 8).
    u: Inputs or an input vector. Returns the state.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            return steady_state(circ, u, x0=x0)
        except (RuntimeWarning, RuntimeError):
            pass
    return steady_state(circ, u, x0=x0, t_settle=800.0)


def purity_recovery(circ: Circuit, x, u_vec, cut: str) -> tuple[float, float]:
    """Product purity and recovery at state x on inputs u_vec (fractions); recovery is the product's
    NdPr over the NdPr in the feed u_vec carries."""
    o = circ.outlets(x, u_vec)
    cp = np.atleast_1d(o["c_prod"])
    heavy = np.array(heavy_mask(circ.p, cut))
    return product_purity(cp, circ.p, cut), float(o["Q_prod"]) * cp[heavy].sum() / nd_pr_fed(circ, u_vec, cut)


def nd_pr_fed(circ: Circuit, u_vec, cut: str) -> float:
    """The NdPr (everything heavier than the cut) fed, mol/h, on inputs u_vec."""
    u = Inputs.from_vector(np.asarray(u_vec, float), circ.n, circ.N)
    return float(u.Q_feed) * float(np.asarray(u.c_feed, float)[np.array(heavy_mask(circ.p, cut))].sum())


# ---- the closed loop ----------------------------------------------------------------------------
@dataclass
class PlantCase:
    """A plant at rest and what the controller is told about it.

    rc: the run file. plant: the circuit that plays the plant. plant_k: its own stage rates.
    x_rest: its state at rest. u_design: the design inputs as the controller knows them (k = 1).
    """
    rc: RunConfig
    plant: Circuit
    plant_k: np.ndarray
    x_rest: np.ndarray
    u_design: Inputs



def closed_loop(case: PlantCase, model: Circuit, model_k) -> dict:
    """The main case's MPC on `model` running the plant HOURS h from rest, and the result.

    model_k: the stage rates the model carries in its inputs; the loop multiplies the controller's
    k_factor by plant_k / model_k so the plant sees its own. Returns end, at-rest (on the final
    handles; recovery above 100 % at the end means the inventory is still draining), minimum purity,
    hours below spec and failed QP solves.
    """
    u_model = case.u_design.replace(k_factor=model_k)
    x_model = settle(model, u_model, x0=case.plant.observable(case.x_rest))
    ctl = mpc_over_level_loops(case.rc, model, x_model, u_model)
    traj, log = run_closed_loop(case.plant, case.x_rest, ctl, None, HOURS,
                                plant_k_factor=np.asarray(case.plant_k) / np.asarray(model_k))
    cut = case.rc.cut
    pr = np.array([purity_recovery(case.plant, x, uv, cut) for x, uv in zip(traj.x, traj.u)])
    rest = purity_recovery(case.plant, settle(case.plant, traj.u[-1], x0=traj.x[-1]), traj.u[-1], cut)
    dt = np.diff(traj.t)
    return {"purity_end": pr[-1, 0], "recovery_end": pr[-1, 1], "rest_purity": rest[0], "rest_recovery": rest[1],
            "purity_min": pr[1:, 0].min(), "hours_below_spec": float(np.sum(dt[pr[1:, 0] < SPEC])),
            "solver_failures": int(sum(1 for r in log if r.get("qp_ok") is False))}


# ---- writing results and running jobs -----------------------------------------------------------
def write_rows(path: Path, rows: list, first: tuple = ()) -> None:
    """Rows (dicts) to a CSV, with every column any row has; the `first` columns lead, the rest in the
    order they first appear (so a rerun writes the same header as the run before it)."""
    seen = dict.fromkeys(k for r in rows for k in r)
    fields = [k for k in first if k in seen] + [k for k in seen if k not in first]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def run_jobs(fn, jobs, n_workers: int, timeout_s: float = 3600.0):
    """Run fn(job) for every job in worker processes; yield (job, result) as each finishes.

    If no job finishes for timeout_s, the workers still running are killed and their jobs tried once
    more in a fresh pool; a job that hangs twice is yielded as (job, None). Why: two studies hung
    overnight on one stuck run (Oct 23); this stops one run from stalling a study.
    """
    pending, tries = list(jobs), {}
    while pending:
        stuck = yield from _one_pool(fn, pending, n_workers, timeout_s)
        for job in stuck:
            tries[job] = tries.get(job, 0) + 1
        for job in [j for j in stuck if tries[j] > 1]:
            yield job, None
        pending = [j for j in stuck if tries[j] <= 1]


def _one_pool(fn, jobs, n_workers: int, timeout_s: float):
    """One pool over the jobs: yields (job, result) as they finish; returns the jobs left stuck."""
    ex = ProcessPoolExecutor(n_workers)
    futures = {ex.submit(fn, job): job for job in jobs}
    not_done = yield from _results_as_they_finish(futures, timeout_s)
    stuck = [futures[f] for f in not_done]
    _close_pool(ex, stuck, timeout_s)
    return stuck


def _results_as_they_finish(futures: dict, timeout_s: float):
    """Yield (job, result) as each future finishes; stop when all have, or when none finishes for
    timeout_s. Returns the futures still running."""
    not_done = set(futures)
    while not_done:
        done, not_done = wait(not_done, timeout=timeout_s, return_when=FIRST_COMPLETED)
        if not done:
            return not_done
        for f in done:
            yield futures[f], f.result()
    return not_done


def _close_pool(ex, stuck: list, timeout_s: float) -> None:
    """Shut the pool down; if jobs are stuck, kill its workers first and say so."""
    if not stuck:
        ex.shutdown()
        return
    print(f"watchdog: {len(stuck)} run(s) stuck for {timeout_s:.0f} s, restarting: {stuck}", flush=True)
    for p in list(ex._processes.values()):
        p.kill()
    ex.shutdown(wait=False, cancel_futures=True)
