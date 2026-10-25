"""Test 6 of the fair test: the smallest network that does the job.

The question. Notebook 35's net is 17 -> 64 -> 64 -> 2 (about 5,400 weights), chosen without a size
sweep. For edge deployment and simpler validation, how small can it be?

The method. Everything as notebook 35 on the main case (same features, scaler, training plants,
held-out plants, fresh instrument noise every batch, Adam with weight decay, cosine schedule, 400
epochs, the best held-out epoch kept), only the hidden layers change: width 4, 8, 16, 32, 64 and
depth 1, 2, 3, each from three training seeds. The ridge (notebook 35's) is the zero-hidden-layer
reference. Per size: weights, held-out rms error on each log factor (mean over seeds), error on the
ten test plants with notebook 35's readings, the time for one estimate on this PC. Then the
smallest size within about 10 % of the best held-out error, and the closed loop (MPC on the
estimate, 120 h, end and rest) for each size's middle seed on plants 1 to 10.

Device: GPU if PyTorch has one, else CPU. The installed PyTorch is a CPU build, so this runs on CPU.

Run:  pixi run python notebooks/45_net_size.py train     # the sweep (writes runs/net_size_ndpr_main/)
      pixi run python notebooks/45_net_size.py test      # test plants, timing, closed loop
Writes runs/net_size_ndpr_main.csv (one row per size and seed), runs/net_size_test_ndpr_main.csv and
runs/net_size_loop_ndpr_main.csv.
"""
import csv
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "notebooks"))
from src.commission import commission                                   # noqa: E402
from src.config import build_circuit, nominal_inputs                    # noqa: E402
from study_tools import (PlantCase, closed_loop, extended_model, notebook_35, run_jobs, settle,   # noqa: E402
                         workers, write_rows)

CONFIG = "ndpr_main"
WIDTHS, DEPTHS, SEEDS = (4, 8, 16, 32, 64), (1, 2, 3), (0, 1, 2)
EPOCHS, BATCH = 400, 256                      # notebook 35's
OUT_DIR = HERE / "runs" / f"net_size_{CONFIG}"
OUT = HERE / "runs" / f"net_size_{CONFIG}.csv"
OUT_TEST = HERE / "runs" / f"net_size_test_{CONFIG}.csv"
OUT_LOOP = HERE / "runs" / f"net_size_loop_{CONFIG}.csv"
TEST_PLANTS = tuple(range(1, 11))
WORKERS = workers()


def device():
    """The GPU if PyTorch has one, else the CPU."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_net(width: int, depth: int, n_in: int = 17):
    """A multilayer perceptron: `depth` hidden layers of `width` SiLU units, two outputs."""
    import torch
    layers, k = [], n_in
    for _ in range(depth):
        layers += [torch.nn.Linear(k, width), torch.nn.SiLU()]
        k = width
    layers.append(torch.nn.Linear(k, 2))
    return torch.nn.Sequential(*layers)


# ---- training -------------------------------------------------------------------------------------
def split_and_noise(n35, data, seed: int):
    """Notebook 35's held-out plants (always its own draw) and the noise stream for this seed (seed 0
    keeps notebook 35's). Returns (held-out row mask, training rows, noisy-readings function, the
    random generator the noise and the batch order draw from)."""
    y, h, lab, pid = data
    rng = np.random.default_rng(0)
    plants = np.unique(pid)
    held_out = set(rng.choice(plants, size=len(plants) // 7, replace=False))
    if seed != 0:
        rng = np.random.default_rng(1000 + seed)
    val = np.array([p in held_out for p in pid])
    return val, np.where(~val)[0], lambda rows: y[rows] + rng.normal(0.0, n35.noise_sd(y[rows])), rng


def train_one(n35, size: tuple, seed: int, data):
    """Notebook 35's training with these hidden layers (size = (width, depth)) and this seed.

    Returns (best state, its held-out rmse per log factor, input scaler, label scaler, seconds)."""
    import torch
    y, h, lab, _ = data
    torch.manual_seed(seed)
    torch.set_num_threads(WORKERS)
    val, tr, noisy, rng = split_and_noise(n35, data, seed)
    X_val = n35.features(noisy(val), h[val])
    X_ridge = np.vstack([n35.features(noisy(tr), h[tr]) for _ in range(5)])
    sx, sl = n35.Scaler(X_ridge), n35.Scaler(np.vstack([lab[tr]] * 5))
    net = make_net(*size, X_val.shape[1]).to(device())

    def batches():
        """One epoch's batches of (features, labels), the training rows in a fresh random order."""
        order = rng.permutation(tr)
        for b in range(0, len(order), BATCH):
            rows = order[b:b + BATCH]
            yield n35.features(noisy(rows), h[rows]), lab[rows]
    t0 = time.time()
    state, rmse = fit_epochs(net, (sx, sl), (X_val, lab[val]), batches)
    return state, rmse, sx, sl, time.time() - t0


def fit_epochs(net, scalers, validation, epoch_batches):
    """Adam with weight decay and a cosine schedule for EPOCHS epochs; the epoch with the lowest
    held-out rmse (summed over the two factors) kept. Returns (its state, its rmse)."""
    import torch
    sx, sl = scalers
    X_val, lab_val = validation
    dev = device()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    Xv = torch.tensor(sx(X_val), dtype=torch.float32, device=dev)
    best, best_state, best_rmse = np.inf, None, None
    for _ in range(EPOCHS):
        for X, L in epoch_batches():
            loss = ((net(torch.tensor(sx(X), dtype=torch.float32, device=dev))
                     - torch.tensor(sl(L), dtype=torch.float32, device=dev)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        with torch.no_grad():
            rmse = np.sqrt(((sl.back(net(Xv).cpu().numpy()) - lab_val) ** 2).mean(axis=0))
        if rmse.sum() < best:
            best, best_rmse = rmse.sum(), rmse
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return best_state, best_rmse


def train():
    """Every size and seed, saved, with its held-out error; written row by row."""
    import torch
    n35 = notebook_35(CONFIG)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = n35.load_data()
    print(f"device {device()}; {len(data[0])} rows", flush=True)
    rows = []
    for depth, width, seed in itertools.product(DEPTHS, WIDTHS, SEEDS):
        state, rmse, sx, sl, secs = train_one(n35, (width, depth), seed, data)
        torch.save({"net": state, "sx": (sx.mean, sx.sd), "sl": (sl.mean, sl.sd), "width": width, "depth": depth},
                   OUT_DIR / f"net_w{width}_d{depth}_s{seed}.pt")
        n_w = sum(v.numel() for v in state.values())
        rows.append({"width": width, "depth": depth, "seed": seed, "weights": n_w,
                     "rmse_pl": rmse[0], "rmse_np": rmse[1], "train_s": round(secs, 1)})
        print(f"width {width:2d} depth {depth} seed {seed}: {n_w:5d} weights, held-out rmse log factor "
              f"{rmse.round(4)}, {secs:.0f} s", flush=True)
        write_rows(OUT, rows, first=("width", "depth", "seed"))


# ---- the test plants ------------------------------------------------------------------------------
def plant_case(seed: int):
    """A test plant at rest on the design handles, notebook 35's noisy readings of it, the handles
    relative to design, and its true log factors. Returns (PlantCase, readings, handles, truth)."""
    warnings.filterwarnings("ignore")
    n35 = notebook_35(CONFIG)
    m = n35.nb31()
    rc, plant, plant_k = n35.plant_for(seed)
    u0 = nominal_inputs(rc, plant).replace(k_factor=plant_k)
    x_rest, u_design = commission(plant, u0)
    y_true = m.readings(plant, x_rest, u_design, list(plant.p.names))
    y = y_true + np.random.default_rng(seed + 5000).normal(0.0, m.noise_sd(y_true))
    case = PlantCase(rc, plant, plant_k, x_rest, u_design.replace(k_factor=np.ones(plant.N)))
    return case, y, n35.handles_of(u_design, u0), n35.true_log_betas(plant, build_circuit(rc))


def plant_readings(seed: int):
    """(readings, handles, truth) for one test plant: what the estimators need (circuits don't travel
    between processes)."""
    _, y, h, truth = plant_case(seed)
    return y, h, truth


def estimator(path):
    """A trained size as a function of (readings, handles) -> log factors."""
    import torch
    saved = torch.load(path, weights_only=False)
    net = make_net(saved["width"], saved["depth"])
    net.load_state_dict(saved["net"])
    net.eval()
    sx_m, sx_s = saved["sx"]
    sl_m, sl_s = saved["sl"]
    features = notebook_35(CONFIG).features

    def est(y, h):
        """The net's log factors from one set of readings."""
        X = (features(y[None], h[None]) - sx_m) / sx_s
        with torch.no_grad():
            return net(torch.tensor(X, dtype=torch.float32)).numpy()[0] * sl_s + sl_m
    return est


def score(est, plants: dict) -> dict:
    """An estimator on the test plants: median and worst absolute error per log factor, and ms per call
    (1,000 calls on plant 1)."""
    errs = np.abs(np.array([est(y, h) - truth for y, h, truth in plants.values()]))
    y1, h1, _ = plants[1]
    t0 = time.perf_counter()
    for _ in range(1000):
        est(y1, h1)
    return {"test_abs_err_pl_median": float(np.median(errs[:, 0])), "test_abs_err_pl_worst": float(errs[:, 0].max()),
            "test_abs_err_np_median": float(np.median(errs[:, 1])), "test_abs_err_np_worst": float(errs[:, 1].max()),
            "ms_per_estimate": time.perf_counter() - t0}            # seconds for 1,000 calls = ms per call


def loop_one(job) -> dict:
    """The MPC on one size's estimate (or the ridge's) for one plant, 120 h."""
    label, path, seed = job
    warnings.filterwarnings("ignore")
    case, y, h, truth = plant_case(seed)
    e = notebook_35(CONFIG).load_estimators()[1](y, h) if path == "ridge" else estimator(path)(y, h)
    case.x_rest = settle(case.plant, case.u_design.replace(k_factor=case.plant_k), x0=case.x_rest)
    circ, kf = extended_model(case.rc, {"beta_pl": float(e[0]), "beta_np": float(e[1])})
    return {"size": label, "seed": seed, "err_pl_log": float(e[0] - truth[0]), "err_np_log": float(e[1] - truth[1]),
            **closed_loop(case, circ, kf)}


def middle_seed_jobs(sizes: list) -> list:
    """Closed-loop jobs: the ridge, and each size's middle seed by held-out error, on every test plant."""
    jobs = [("ridge", "ridge", p) for p in TEST_PLANTS]
    for depth, width in itertools.product(DEPTHS, WIDTHS):
        group = sorted([s for s in sizes if int(s["width"]) == width and int(s["depth"]) == depth],
                       key=lambda s: float(s["rmse_pl"]) + float(s["rmse_np"]))
        path = str(OUT_DIR / f"net_w{width}_d{depth}_s{group[len(group) // 2]['seed']}.pt")
        jobs += [(f"w{width}_d{depth}", path, p) for p in TEST_PLANTS]
    return jobs


def test():
    """Test-plant errors and timing for every trained size and the ridge, then the closed loops."""
    import torch
    torch.set_num_threads(1)
    plants = {seed: r for seed, r in run_jobs(plant_readings, TEST_PLANTS, WORKERS)}
    sizes = list(csv.DictReader(open(OUT)))
    rows = [{"size": "ridge", "width": 0, "depth": 0, "seed": 0, **score(notebook_35(CONFIG).load_estimators()[1], plants)}]
    for s in sizes:
        path = OUT_DIR / f"net_w{s['width']}_d{s['depth']}_s{s['seed']}.pt"
        rows.append({"size": f"w{s['width']}_d{s['depth']}", "width": int(s["width"]), "depth": int(s["depth"]),
                     "seed": int(s["seed"]), **score(estimator(path), plants)})
    write_rows(OUT_TEST, rows, first=("size", "width", "depth", "seed"))
    print("test-plant errors and timing written", flush=True)
    out_rows = []
    for _, r in run_jobs(loop_one, middle_seed_jobs(sizes), WORKERS):
        out_rows.append(r)
        print(f"{r['size']:8s} plant {r['seed']:2d}: end {100 * r['purity_end']:.2f}/{100 * r['recovery_end']:.2f} "
              f"rest {100 * r['rest_purity']:.2f}/{100 * r['rest_recovery']:.2f}", flush=True)
        write_rows(OUT_LOOP, out_rows, first=("size", "seed"))
    print("wrote", OUT_LOOP.name)


if __name__ == "__main__":
    {"train": train, "test": test}[[a for a in sys.argv[1:] if a in ("train", "test")][0]]()
