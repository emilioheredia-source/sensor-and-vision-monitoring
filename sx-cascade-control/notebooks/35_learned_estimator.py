"""A neural net that reads the instruments and returns the separation factors, trained on the physics.

The question. Notebook 31 identifies a plant by fitting the physics model to a few noisy readings:
about 30 steady-state solves per fit, a minute or more, and on the three easy plants it needed
about 200 solves and still came out 0.25 to 0.37 low on the Pr/(La+Ce) factor. Can a small neural
net, trained only on simulated plants, do the same in a millisecond? And does the controller
built on its answer bring the hard plants back to spec, as the fitted model did?

The method:
  1. Training data from the physics. 1,200 plants drawn the same way as the test plants (the
     perturbed plant, "ratios" generator) but with other seeds (1000 and up), so no test plant is
     ever seen in training. Each plant at its design flows and at seven random handle settings
     around them (organic and scrub flow +-8 %, scrub acid +-10 %, saponification +-5 %, feed flow
     +-10 %), solved to steady state. For each: the instruments' true readings (XRF on product and
     raffinate, UV-Vis Pr and Nd at stages 2, 4, 6, as notebook 31), the handles, and the label,
     the plant's true Pr/(La+Ce) and Nd/Pr separation factors.
  2. Two estimators, trained on the same data with fresh instrument noise drawn every batch:
       ridge   a linear regression (scikit-learn), the baseline any learned estimator must beat
       net     a small multilayer perceptron (PyTorch): 17 inputs, two hidden layers of 64
     Inputs: each reading through asinh(reading / its noise floor), log-like where the reading is
     large and smooth through zero where noise swamps it; the five handles relative to design.
     Outputs: the log of each separation factor relative to the model's.
  3. The test: notebook 31's ten plants, with exactly the readings (same noise draws) that
     notebook 31's least-squares fit saw. Then the MPC built on the net's model runs each hard
     plant for 120 h, as in notebook 31, and the net's answer is also tried as the starting point
     for the least-squares fit (the net proposes, the physics checks).

Run:
  pixi run python notebooks/35_learned_estimator.py data      # training data, about an hour on 11 cores (all but two)
  pixi run python notebooks/35_learned_estimator.py train     # fit ridge and net, score on held-out plants
  pixi run python notebooks/35_learned_estimator.py test      # the ten test plants, closed loop on the hard ones
Writes runs/learned_estimator/ (data chunks, the trained net) and
runs/learned_estimator_ndpr_plant.csv (one row per test plant).
"""
import csv
import importlib.util
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from src.config import build_circuit, load_config, nominal_inputs       # noqa: E402
from src.registry import PLANTS                                         # noqa: E402
from src.simulate import steady_state                                   # noqa: E402

# notebook 31's readings, noise and model rebuild, so both see exactly the same instruments
_spec = importlib.util.spec_from_file_location("nb31", HERE / "notebooks" / "31_identify_then_control.py")

CONFIG = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "ndpr_plant"
MAIN = CONFIG == "ndpr_main"            # the main case: 600 training plants, test plants 1 to 10, MPC over level loops
DATA_DIR = HERE / "runs" / ("learned_estimator" if not MAIN else f"learned_estimator_{CONFIG}")
# --hours H and --seeds a,b,c rerun chosen test plants for longer, into their own table
LONG = "--hours" in sys.argv
# Worker processes and torch threads: every core but two, at least one.
WORKERS = max(1, (os.cpu_count() or 1) - 2)
OUT = HERE / "runs" / (f"learned_estimator_{CONFIG}.csv" if not LONG else f"learned_estimator_{CONFIG}_long.csv")
TRAIN_SEEDS = range(1000, 1600) if MAIN else range(1000, 2200)
CHUNK = 50
SETTINGS_PER_PLANT = 8                       # the design flows and seven random settings around them
SPREAD = {"Q_org": 0.08, "Q_scrub": 0.08, "h_scrub": 0.10, "saponified_fraction": 0.05, "Q_feed": 0.10}
HANDLES = tuple(SPREAD)
HARD, EASY = ((tuple(range(1, 11)), ()) if MAIN else ((2, 3, 6, 10, 14, 16, 20), (1, 7, 12)))
HOURS = float(sys.argv[sys.argv.index("--hours") + 1]) if LONG else 120.0
XRF_SD, UV_SD = 0.02, 0.01                   # as notebook 31
FLOOR = np.array([1e-4] * 6 + [1e-5] * 6)    # each reading's noise floor (notebook 31's noise_sd)


def nb31():
    """Notebook 31 as a module, loaded with no arguments so it doesn't run its study."""
    argv, sys.argv = sys.argv, [sys.argv[0], "--sensors", "xrf_uv", "0"]
    module = importlib.util.module_from_spec(_spec)
    source = _spec.loader.get_source("nb31").split("\ndone = set()")[0]      # the definitions only
    exec(compile(source, str(_spec.origin), "exec"), module.__dict__)
    sys.argv = argv
    return module


def plant_for(seed):
    rc = load_config(CONFIG, overrides={"plant": {"mismatch": "perturbed", "seed": seed, "generator": "ratios"}})
    plant, plant_k = PLANTS["perturbed"].build(rc)
    return rc, plant, plant_k


def true_log_betas(plant, model):
    """log of the plant's Pr/(La+Ce) and Nd/Pr separation factors over the model's."""
    def betas(p):
        names = list(p.names)
        K = np.asarray(p.K, float)
        la, pr, nd = names.index("LaCe"), names.index("Pr"), names.index("Nd")
        return np.array([K[pr] / K[la], K[nd] / K[pr]])
    return np.log(betas(plant.p) / betas(model.p))


def handles_of(u, u0):
    """The five handles relative to design (0 = at design)."""
    return np.array([float(getattr(u, h)) / float(getattr(u0, h)) - 1.0 for h in HANDLES])


# ---- 1. training data ---------------------------------------------------------------------------
def make_chunk(start):
    """Training rows for plants start .. start+CHUNK-1, saved as one .npz."""
    path = DATA_DIR / f"chunk_{start}.npz"
    if path.exists():
        return str(path)
    m = nb31()
    rows_y, rows_h, rows_lab, rows_seed = [], [], [], []
    for seed in range(start, start + CHUNK):
        rc, plant, plant_k = plant_for(seed)
        model = build_circuit(rc)
        names = list(plant.p.names)
        label = true_log_betas(plant, model)
        u0 = nominal_inputs(rc, plant).replace(k_factor=plant_k)
        rng = np.random.default_rng(seed)
        x_prev = None
        for k in range(SETTINGS_PER_PLANT):
            changes = {} if k == 0 else {h: float(getattr(u0, h)) * (1 + rng.uniform(-s, s)) for h, s in SPREAD.items()}
            u = u0.replace(**changes)
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)          # an unconverged state isn't a sample
                try:
                    x = steady_state(plant, u, x0=x_prev)
                except (RuntimeWarning, RuntimeError):
                    continue
            x_prev = x if k == 0 else x_prev
            rows_y.append(m.readings(plant, x, u, names))
            rows_h.append(handles_of(u, u0))
            rows_lab.append(label)
            rows_seed.append(seed)
    np.savez(path, y=np.array(rows_y), h=np.array(rows_h), label=np.array(rows_lab), seed=np.array(rows_seed))
    return str(path)


def make_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    starts = list(range(TRAIN_SEEDS.start, TRAIN_SEEDS.stop, CHUNK))
    t0 = time.time()
    with Pool(WORKERS) as pool:
        for n, path in enumerate(pool.imap_unordered(make_chunk, starts), start=1):
            print(f"{n}/{len(starts)} chunks ({time.time() - t0:.0f} s): {Path(path).name}", flush=True)


def load_data(dirs=None):
    """All chunks, with each row's plant seed so held-out plants stay whole. dirs: the folders to read
    (default this config's DATA_DIR; notebook 44 adds the plants built with other physics)."""
    chunks = [np.load(path) for d in (dirs or (DATA_DIR,)) for path in sorted(Path(d).glob("chunk_*.npz"))]
    return (np.vstack([d["y"] for d in chunks]), np.vstack([d["h"] for d in chunks]),
            np.vstack([d["label"] for d in chunks]), np.concatenate([d["seed"] for d in chunks]))


# ---- 2. the estimators ----------------------------------------------------------------------------
def noise_sd(y):
    """notebook 31's instrument noise, row by row."""
    return np.concatenate([np.maximum(XRF_SD * np.abs(y[:, :6]), 1e-4), np.maximum(UV_SD * np.abs(y[:, 6:]), 1e-5)], axis=1)


def features(y, h):
    return np.hstack([np.arcsinh(y / FLOOR), h])


class Scaler:
    def __init__(self, a):
        self.mean, self.sd = a.mean(axis=0), a.std(axis=0) + 1e-12

    def __call__(self, a):
        return (a - self.mean) / self.sd

    def back(self, a):
        return a * self.sd + self.mean


def train(dirs=None, out_dir=None):
    """Fit the ridge and the net on the chunks in dirs (default DATA_DIR) and save them to out_dir
    (default DATA_DIR)."""
    out_dir = Path(out_dir or DATA_DIR)
    import torch
    from sklearn.linear_model import Ridge
    torch.manual_seed(0)
    torch.set_num_threads(WORKERS)
    y, h, lab, pid = load_data(dirs)
    rng = np.random.default_rng(0)
    plants = np.unique(pid)
    held_out = set(rng.choice(plants, size=len(plants) // 7, replace=False))
    val = np.array([p in held_out for p in pid])
    print(f"{len(y)} rows from {len(plants)} plants; {val.sum()} rows from {len(held_out)} plants held out")

    def noisy(rows):
        return y[rows] + rng.normal(0.0, noise_sd(y[rows]))

    # fixed noisy copies for validation and for the ridge (five noise draws per row)
    X_val = features(noisy(val), h[val])
    tr = np.where(~val)[0]
    X_ridge = np.vstack([features(noisy(tr), h[tr]) for _ in range(5)])
    L_ridge = np.vstack([lab[tr]] * 5)
    sx, sl = Scaler(X_ridge), Scaler(L_ridge)

    ridge = Ridge(alpha=1.0).fit(sx(X_ridge), sl(L_ridge))
    ridge_rmse = np.sqrt(((sl.back(ridge.predict(sx(X_val))) - lab[val]) ** 2).mean(axis=0))

    net = torch.nn.Sequential(torch.nn.Linear(X_val.shape[1], 64), torch.nn.SiLU(),
                              torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
    Xv = torch.tensor(sx(X_val), dtype=torch.float32)
    best, best_state = np.inf, None
    for epoch in range(400):
        order = rng.permutation(tr)
        for b in range(0, len(order), 256):
            rows = order[b:b + 256]
            X = torch.tensor(sx(features(noisy(rows), h[rows])), dtype=torch.float32)
            L = torch.tensor(sl(lab[rows]), dtype=torch.float32)
            loss = ((net(X) - L) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        with torch.no_grad():
            pred = sl.back(net(Xv).numpy())
        rmse = np.sqrt(((pred - lab[val]) ** 2).mean(axis=0))
        if rmse.sum() < best:
            best, best_state = rmse.sum(), {k: v.clone() for k, v in net.state_dict().items()}
            best_rmse = rmse
        if epoch % 50 == 0:
            print(f"epoch {epoch}: held-out rmse of log beta {rmse.round(4)}", flush=True)
    net.load_state_dict(best_state)
    torch.save({"net": best_state, "sx": (sx.mean, sx.sd), "sl": (sl.mean, sl.sd)}, out_dir / "net.pt")
    import pickle
    pickle.dump({"ridge": ridge, "sx": sx, "sl": sl}, open(out_dir / "ridge.pkl", "wb"))
    print(f"held-out plants, rmse of log beta [Pr/(La+Ce), Nd/Pr]: ridge {ridge_rmse.round(4)}, net {best_rmse.round(4)}")


def load_estimators(model_dir=None):
    """The trained net and ridge from model_dir (default DATA_DIR), as functions of (readings, handles)."""
    model_dir = Path(model_dir or DATA_DIR)
    import pickle
    import torch
    saved = torch.load(model_dir / "net.pt", weights_only=False)
    net = torch.nn.Sequential(torch.nn.Linear(17, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(),
                              torch.nn.Linear(64, 2))
    net.load_state_dict(saved["net"])
    sx_m, sx_s = saved["sx"]
    sl_m, sl_s = saved["sl"]

    def net_estimate(y, h):
        X = (features(y[None], h[None]) - sx_m) / sx_s
        with torch.no_grad():
            return net(torch.tensor(X, dtype=torch.float32)).numpy()[0] * sl_s + sl_m

    r = pickle.load(open(model_dir / "ridge.pkl", "rb"))

    def ridge_estimate(y, h):
        return r["sl"].back(r["ridge"].predict(r["sx"](features(y[None], h[None]))))[0]

    return net_estimate, ridge_estimate


# ---- 3. the test plants ---------------------------------------------------------------------------
def test_one(seed):
    """One of notebook 31's plants: the same noisy readings, both estimators, the net's answer as
    the least-squares starting point, and (hard plants) the MPC on the net's model for 120 h."""
    from scipy.optimize import least_squares
    from src.commission import commission
    from src.loop import run_closed_loop
    from src.metrics import heavy_mask, product_purity
    from src.mpc import CircuitMPC
    from src.operating_point import controller_settings
    m = nb31()
    net_estimate, ridge_estimate = load_estimators()
    t0 = time.time()
    rc, plant, plant_k = plant_for(seed)
    model = build_circuit(rc)
    names = list(plant.p.names)
    u0 = nominal_inputs(rc, plant).replace(k_factor=plant_k)
    x_rest, u_design = commission(plant, u0)
    rng = np.random.default_rng(seed + 5000)                   # notebook 31's noise draw for this plant
    y_true = m.readings(plant, x_rest, u_design, names)
    sd = m.noise_sd(y_true)
    y = y_true + rng.normal(0.0, sd)
    h = handles_of(u_design, u0)
    truth = true_log_betas(plant, model)
    t1 = time.perf_counter()
    est_net = net_estimate(y, h)
    net_ms = 1000 * (time.perf_counter() - t1)
    est_ridge = ridge_estimate(y, h)

    # the net proposes, the physics checks: least squares from the net's answer (k starts at 1)
    u_model = u_design.replace(k_factor=np.ones(plant.N))
    state = {"x": plant.observable(x_rest), "evals": 0}

    def residuals(theta):
        mod, _, _ = m.model_with(rc, theta)
        x = steady_state(mod, u_model, x0=state["x"])
        state["x"], state["evals"] = x, state["evals"] + 1
        return (m.readings(mod, x, u_model, names) - y) / sd

    t2 = time.time()
    polish = least_squares(residuals, np.clip([est_net[0], est_net[1], 0.0], -0.49, 0.49), bounds=(-0.5, 0.5),
                           diff_step=0.02, max_nfev=60)
    polish_s = time.time() - t2

    b_true = np.exp(truth)
    row = {"seed": seed, "group": "main" if MAIN else ("hard" if seed in HARD else "easy")}
    K = np.asarray(model.p.K, float)
    la, pr, nd = names.index("LaCe"), names.index("Pr"), names.index("Nd")
    base = np.array([K[pr] / K[la], K[nd] / K[pr]])
    row.update({"beta_pr_lace_true": base[0] * b_true[0], "beta_nd_pr_true": base[1] * b_true[1],
                "beta_pr_lace_net": base[0] * np.exp(est_net[0]), "beta_nd_pr_net": base[1] * np.exp(est_net[1]),
                "beta_pr_lace_ridge": base[0] * np.exp(est_ridge[0]), "beta_nd_pr_ridge": base[1] * np.exp(est_ridge[1]),
                "beta_pr_lace_polished": base[0] * np.exp(polish.x[0]), "beta_nd_pr_polished": base[1] * np.exp(polish.x[1]),
                "net_ms": round(net_ms, 3), "polish_evals": state["evals"], "polish_seconds": round(polish_s)})

    if seed in HARD:
        mod, _, _ = m.model_with(rc, [est_net[0], est_net[1], 0.0])
        x_model = steady_state(mod, u_model, x0=plant.observable(x_rest))
        if MAIN:
            from src.registry import mpc_over_level_loops
            ctl = mpc_over_level_loops(rc, mod, x_model, u_model)
        else:
            ctl = CircuitMPC(mod, x_model, u_model, rc.cut, rc.targets()[0], rc.design_purity, rc.targets()[1],
                             controller_settings(rc, acid_floor=1e-3))
        traj, log = run_closed_loop(plant, x_rest, ctl, None, HOURS, plant_k_factor=plant_k)
        heavy = np.array(heavy_mask(plant.p, rc.cut))
        fed = float(u_design.Q_feed) * np.asarray(u_design.c_feed, float)[heavy].sum()
        o = plant.outlets(traj.x[-1], traj.u[-1])
        cp = np.atleast_1d(o["c_prod"])
        x_rest_end = steady_state(plant, traj.u[-1], x0=traj.x[-1], t_settle=500.0)   # where the final handles settle
        o_r = plant.outlets(x_rest_end, traj.u[-1])
        cp_r = np.atleast_1d(o_r["c_prod"])
        row.update({"rest_purity": product_purity(cp_r, plant.p, rc.cut),
                    "rest_recovery": float(o_r["Q_prod"]) * cp_r[heavy].sum() / fed})
        row.update({"purity_end": product_purity(cp, plant.p, rc.cut), "recovery_end": float(o["Q_prod"]) * cp[heavy].sum() / fed,
                    "solver_failures": int(sum(1 for r in log if r.get("qp_ok") is False))})
    row["seconds"] = round(time.time() - t0)
    return row


FIELDS = ["seed", "group", "beta_pr_lace_true", "beta_nd_pr_true", "beta_pr_lace_net", "beta_nd_pr_net",
          "beta_pr_lace_ridge", "beta_nd_pr_ridge", "beta_pr_lace_polished", "beta_nd_pr_polished", "net_ms",
          "polish_evals", "polish_seconds", "purity_end", "recovery_end", "rest_purity", "rest_recovery",
          "solver_failures", "seconds"]


def _test_seeds():
    """The test plants: notebook 31's, or the ones named with --seeds a,b,c."""
    if "--seeds" in sys.argv:
        return tuple(int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(","))
    return HARD + EASY


def test():
    with open(OUT, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    with Pool(WORKERS) as pool:
        for row in pool.imap_unordered(test_one, _test_seeds()):
            with open(OUT, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            print(f"seed {row['seed']:2d} ({row['group']}): Pr/LaCe true {row['beta_pr_lace_true']:.3f} net "
                  f"{row['beta_pr_lace_net']:.3f} ridge {row['beta_pr_lace_ridge']:.3f} polished {row['beta_pr_lace_polished']:.3f} "
                  f"({row['polish_evals']} evals) | net {row['net_ms']} ms"
                  + (f" | end {100 * row['purity_end']:.2f}/{100 * row['recovery_end']:.2f} %" if "purity_end" in row else ""),
                  flush=True)


if __name__ == "__main__":
    {"data": make_data, "train": train, "test": test}[[a for a in sys.argv[1:] if a in ("data", "train", "test")][0]]()
