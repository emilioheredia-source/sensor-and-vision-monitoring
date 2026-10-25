"""Is the feed-switch dip a controller limit or a plant limit? The best any controller could do (Part E2).

The question (Oct 23). When the feed switches from the lean deposit to the rich one, can any
controller keep the product at or above 99.5 % purity the whole way through without losing more than
5 % of the NdPr fed to the raffinate? The MPC on the plant's true model kept purity above the 99 %
spec but not above 99.5 % (99.22-99.47 % on the five plants that dipped, notebook 49).

The method. One optimization over the whole switch on the plant's own equations and numbers (stage
efficiencies included), from notebook 41's starting point (the plant trimmed to rest at its operating
point, 5 h before the switch, the same feed profile):
  grid       1 h steps over HORIZON_H, implicit Euler (stable on the stiff chemistry)
  handles    organic flow, scrub flow (the strip flow tied to it, as in the plant), scrub acid and
             saponified fraction, one value per hour; the MPC's bounds; its per-step rate limits
             doubled for the 1 h step (or none)
  limits     every extraction stage's acid at or above 1e-3 mol/L (the design's acidity floor); NdPr
             lost to the raffinate at most LOSS_CAP of the NdPr fed, summed from the switch to the end
             (metal in transit isn't lost: notebook 49)
  objective  the lowest purity at any hour, as high as possible
If that best lowest purity is at or above 99.5 %, a trajectory exists and the dip is the controller's
or its model's; if not, this plant can't hold 99.5 % through this switch with these handles. Each
answer is then replayed on the plant with the full integrator (the 1 h grid is coarse), and that
replay is what's reported as achieved. The binding limits are read off where each handle sits.
Variants: the switch known in advance (handles free from the start) or only when it happens (held
until the switch); with or without rate limits; the 1 h switch or a 6 h ramp.

Two math threads per worker, the project's setting (pixi.toml), so the operating points trimmed here
are notebook 41's: the 120 h trim lands on a different, equally valid point with one thread (plant 3:
2,385 against 2,134 L/h of organic), from rounding order alone (FINDINGS phase 9).

Run:  pixi run python notebooks/50_switch_bound.py [seed ...]   (default: 3 10 14 16 17 and 2 6 9)
Writes runs/switch_bound_ndpr_main.csv (one row per plant and variant) and .npz (the trajectories).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"

import sys                                                                # noqa: E402
import time                                                               # noqa: E402
import warnings                                                           # noqa: E402
from dataclasses import dataclass                                         # noqa: E402
from pathlib import Path                                                  # noqa: E402

import casadi as ca                                                       # noqa: E402
import numpy as np                                                        # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "notebooks"))
from src.circuit import Inputs                                            # noqa: E402
from src.metrics import heavy_mask, point_outputs                        # noqa: E402
from src.registry import PLANTS                                          # noqa: E402
from src.runs import RunSpec, start                                      # noqa: E402
from src.simulate import simulate                                        # noqa: E402
from study_tools import run_jobs, workers, write_rows                    # noqa: E402

HORIZON_H, DT = 60.0, 1.0
T_SWITCH = 5.0
LOSS_CAP = 0.05
ACID_FLOOR = 1e-3
HANDLES = ("Q_org", "Q_scrub", "h_scrub", "saponified_fraction")
VARIANTS = {   # name: (handles free before the switch, rate limits, switch ramp in hours)
    "known_ahead": (True, True, None),
    "at_the_switch": (False, True, None),
    "no_rate_limits": (True, False, None),
    "slow_ramp_6h": (True, True, 6.0),
}
# Why the limited-memory Hessian and the time cap: with the exact Hessian one case ran two hours without
# finishing (Oct 24). A caller can change these for a test.
IPOPT_OPTS = {"ipopt.print_level": 0, "ipopt.max_iter": 1500, "ipopt.tol": 1e-6, "ipopt.acceptable_tol": 1e-4,
              "ipopt.mu_strategy": "adaptive", "ipopt.hessian_approximation": "limited-memory",
              "ipopt.max_cpu_time": 3600.0}
SEEDS = tuple(int(a) for a in sys.argv[1:] if a.isdigit()) or (3, 10, 14, 16, 17, 2, 6, 9)
OUT = HERE / "runs" / "switch_bound_ndpr_main.csv"


@dataclass
class SwitchCase:
    """One plant through the switch, as the optimization and the replay need it.

    model: the plant's own equations and numbers. x0: its state at the operating point. base: the
    inputs each hour (feed as the scenario has it at mid-hour, handles at the operating point). fed:
    the NdPr fed each hour, mol/h. times: each hour's start, h. h_start: the handles at the operating
    point. net: the net product flow (strip minus scrub), L/h. bounds, rates: the MPC's handle bounds
    and per-step rate limits. cut: the last component meant for the raffinate.
    """
    model: object
    x0: np.ndarray
    base: list
    fed: np.ndarray
    times: np.ndarray
    h_start: np.ndarray
    net: float
    bounds: dict
    rates: dict
    cut: str

    @property
    def fed_after(self) -> float:
        """The NdPr fed from the switch to the end, mol."""
        return float(np.sum(self.fed[self.times >= T_SWITCH]) * DT)


# ---- the case -------------------------------------------------------------------------------------
def switch_case(seed: int, ramp_h) -> SwitchCase:
    """Notebook 41's run for this seed: its operating point, the feed profile hour by hour (stretched
    over ramp_h hours if given), the plant's own model and stage efficiencies, the MPC's limits."""
    session = start(RunSpec(config="ndpr_main", plant="perturbed", controller="mpc_level_loops", upset="feed_switch",
                            sensors="full_state", hours=HORIZON_H, run_file={"plant": {"seed": seed, "generator": "ratios"}}))
    op = session.op
    model, pk = PLANTS["perturbed"].build(op.rc)
    u0 = op.u_model.replace(k_factor=np.asarray(pk, float))
    u_of_t = session.upset.scenario().u_of_t(u0)
    times = np.arange(int(HORIZON_H / DT)) * DT
    base = [feed_at(u_of_t, u0, t + DT / 2, ramp_h) for t in times]
    heavy = np.array(heavy_mask(model.p, op.rc.cut))
    s = session.controller.s
    return SwitchCase(model, np.asarray(op.x_plant, float), base,
                      np.array([float(b.Q_feed) * np.asarray(b.c_feed, float)[heavy].sum() for b in base]), times,
                      np.array([float(getattr(u0, h)) for h in HANDLES]), float(u0.Q_strip) - float(u0.Q_scrub),
                      s.bounds, s.rate_per_step, op.rc.cut)


def feed_at(u_of_t, u0, t: float, ramp_h):
    """The inputs the plant gets at time t: the scenario's feed (stretched over ramp_h hours if given)."""
    if ramp_h is None:
        return u_of_t(t)
    frac = float(np.clip((t - T_SWITCH) / ramp_h, 0.0, 1.0))
    late = u_of_t(T_SWITCH + 10.0)                   # the new feed, fully switched
    return u0.replace(c_feed=(1 - frac) * np.asarray(u0.c_feed) + frac * np.asarray(late.c_feed),
                      h_feed=(1 - frac) * float(u0.h_feed) + frac * float(late.h_feed),
                      Q_feed=(1 - frac) * float(u0.Q_feed) + frac * float(late.Q_feed))


# ---- the optimization ---------------------------------------------------------------------------
def inputs_over_time(case: SwitchCase, U):
    """Each hour's full input vector with the handles U (4 x K) written in and the strip flow tied to the
    scrub: a constant matrix plus a selection of U, so the problem stays linear in the handles."""
    lay = Inputs.layout(case.model.n, case.model.N)
    rows = [lay[h].start for h in HANDLES]
    B = np.column_stack([b.numpy() for b in case.base])
    B[rows, :] = 0.0
    B[lay["Q_strip"].start, :] = case.net
    S = np.zeros((Inputs.size(case.model.n, case.model.N), len(HANDLES)))
    S[rows, range(len(HANDLES))] = 1.0
    S[lay["Q_strip"].start, 1] = 1.0
    return ca.DM(B) + ca.DM(S) @ U


def constraints(case: SwitchCase, variables: tuple, variant: str) -> tuple:
    """The problem's constraints (g, lower, upper): implicit Euler every hour, purity at least s every
    hour, the loss cap, and per variant the rate limits and the handles held before the switch.
    variables: (X states x hours, U handles x hours, s the lowest purity), CasADi symbols."""
    X, U, s = variables
    free_before, rate_on, _ = VARIANTS[variant]
    model, K = case.model, len(case.times)
    UV = inputs_over_time(case, U)
    dyn = X - ca.horzcat(ca.DM(case.x0), X[:, :-1]) - DT * model.casadi_rhs().map(K)(X, UV)
    out = model._out.map(K)(X, UV)
    hv = ca.DM(np.array(heavy_mask(model.p, case.cut), float)).T
    purity = (hv @ out[4]) / (ca.DM.ones(1, model.n) @ out[4] + 1e-12)
    lost = ca.sum2(ca.DM((case.times >= T_SWITCH).astype(float)).T * out[0] * (hv @ out[1])) * DT
    g = [ca.vec(dyn), ca.vec(purity - s), lost - LOSS_CAP * case.fed_after]
    lbg = [np.zeros(model.nx * K), np.zeros(K), [-np.inf]]
    ubg = [np.zeros(model.nx * K), np.full(K, np.inf), [0.0]]
    if rate_on:
        lim = np.array([2.0 * case.rates[h] for h in HANDLES])            # the MPC's per 0.5 h step, doubled
        g.append(ca.vec(U - ca.horzcat(ca.DM(case.h_start), U[:, :-1])))
        lbg.append(np.tile(-lim, K))
        ubg.append(np.tile(lim, K))
    if not free_before:
        k_before = int(np.sum(case.times < T_SWITCH))
        g.append(ca.vec(U[:, :k_before] - ca.repmat(ca.DM(case.h_start), 1, k_before)))
        lbg.append(np.zeros(len(HANDLES) * k_before))
        ubg.append(np.zeros(len(HANDLES) * k_before))
    return ca.vertcat(*g), np.concatenate(lbg), np.concatenate(ubg)


def variable_bounds(case: SwitchCase) -> tuple:
    """Bounds on (states, handles, s) stacked as the problem's variables: the extraction stages' acid
    at or above ACID_FLOOR, the handles within the MPC's bounds, the lowest purity s in [0.5, 1]."""
    model, K = case.model, len(case.times)
    lbx = np.full((model.nx, K), -np.inf)
    acid0 = 2 * model.N * model.n
    lbx[acid0: acid0 + model.cmap.n_ext, :] = ACID_FLOOR
    lbu = np.array([[case.bounds[h][0]] * K for h in HANDLES])
    ubu = np.array([[case.bounds[h][1]] * K for h in HANDLES])
    return (np.concatenate([lbx.flatten(order="F"), lbu.flatten(order="F"), [0.5]]),
            np.concatenate([np.full(model.nx * K, np.inf), ubu.flatten(order="F"), [1.0]]))


def warm_start(case: SwitchCase) -> np.ndarray:
    """A starting guess: the plant held on its starting handles through the switch, s at 0.98."""
    X = [case.x0]
    for b in case.base:
        X.append(simulate(case.model, X[-1], lambda t, uv=b.numpy(): uv, DT, dt_out=DT).x[-1])
    return np.concatenate([np.column_stack(X[1:]).flatten(order="F"), np.tile(case.h_start, len(case.times)), [0.98]])


def solve(case: SwitchCase, variant: str) -> tuple:
    """The optimization for one case and variant: (IPOPT status, seconds, lowest purity on the grid,
    the optimal handles as 4 x K)."""
    model, K = case.model, len(case.times)
    X, U, s = ca.MX.sym("X", model.nx, K), ca.MX.sym("U", len(HANDLES), K), ca.MX.sym("s")
    g, lbg, ubg = constraints(case, (X, U, s), variant)
    lbw, ubw = variable_bounds(case)
    w = ca.vertcat(ca.vec(X), ca.vec(U), s)
    solver = ca.nlpsol("bound", "ipopt", {"x": w, "f": -1000.0 * s, "g": g}, {"print_time": 0, **IPOPT_OPTS})
    t0 = time.time()
    sol = solver(x0=np.clip(warm_start(case), lbw, ubw), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    wopt = np.asarray(sol["x"]).ravel()
    U_opt = wopt[model.nx * K: model.nx * K + len(HANDLES) * K].reshape((len(HANDLES), K), order="F")
    return solver.stats()["return_status"], time.time() - t0, float(wopt[-1]), U_opt


# ---- checking the answer --------------------------------------------------------------------------
def replay(case: SwitchCase, U_opt) -> dict:
    """The optimized handles applied to the plant with the full integrator, four readings an hour:
    the lowest purity and the NdPr lost after the switch as a share of what was fed."""
    x, pur, lost = case.x0, [], []
    for k, b in enumerate(case.base):
        change = {h: float(U_opt[i, k]) for i, h in enumerate(HANDLES)}
        change["Q_strip"] = case.net + change["Q_scrub"]
        uv = b.replace(**change).numpy()
        traj = simulate(case.model, x, lambda t, uv=uv: uv, DT, dt_out=DT / 4)
        outs = [point_outputs(case.model, xx, uv, case.cut) for xx in traj.x[1:]]
        pur += [o["purity"] for o in outs]
        lost += [o["lost_rate"] if case.times[k] >= T_SWITCH else 0.0 for o in outs]
        x = traj.x[-1]
    return {"replay_min_purity": float(min(pur)), "replay_lost_pct": float(100 * np.sum(lost) * DT / 4 / case.fed_after)}


def binding(U_opt, case: SwitchCase, rate_on: bool) -> dict:
    """How many hours each handle sits on a bound or moves by its full rate limit, and its range."""
    out = {}
    for i, h in enumerate(HANDLES):
        lo, hi = case.bounds[h]
        on_bound = (U_opt[i] <= lo + 1e-6 * abs(lo) + 1e-9) | (U_opt[i] >= hi - 1e-6 * abs(hi) - 1e-9)
        out[f"{h}_hours_on_bound"] = int(np.sum(on_bound))
        out[f"{h}_hours_at_rate_limit"] = int(np.sum(np.abs(np.diff(U_opt[i])) >= 0.99 * 2 * case.rates[h])) if rate_on else 0
        out[f"{h}_range"] = f"{U_opt[i].min():.3g}-{U_opt[i].max():.3g}"
    return out


def job(seed_variant) -> tuple:
    """One plant and variant: (row, trajectories). An error becomes a row with its message, so one failed
    optimization doesn't stop the rest."""
    seed, variant = seed_variant
    warnings.filterwarnings("ignore")
    try:
        case = switch_case(seed, VARIANTS[variant][2])
        status, secs, grid_min, U_opt = solve(case, variant)
    except Exception as err:                                              # noqa: BLE001 (any failure is recorded)
        return {"seed": seed, "variant": variant, "status": f"error: {str(err)[:150]}"}, {}
    row = {"seed": seed, "variant": variant, "status": status, "solve_s": round(secs), "grid_min_purity": grid_min,
           **replay(case, U_opt), **binding(U_opt, case, VARIANTS[variant][1])}
    return row, {"U": U_opt, "times": case.times}


def main():
    """Every plant and variant; rows and trajectories written as each finishes."""
    rows, traces = [], {}
    for _, (row, tr) in run_jobs(job, [(s, v) for v in VARIANTS for s in SEEDS], workers(threads_each=2), 3 * 3600.0):
        rows.append(row)
        traces.update({f"{row['seed']}_{row['variant']}_{k}": v for k, v in tr.items()})
        print(f"seed {row['seed']:2d} {row['variant']:15s} {row.get('status')}: best lowest purity on the grid "
              f"{100 * row.get('grid_min_purity', np.nan):.2f} %, replayed {100 * row.get('replay_min_purity', np.nan):.2f} % "
              f"with {row.get('replay_lost_pct', np.nan):.2f} % lost ({row.get('solve_s', '-')} s)", flush=True)
        write_rows(OUT, rows, first=("seed", "variant", "status"))
        np.savez(OUT.with_suffix(".npz"), **traces)
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
