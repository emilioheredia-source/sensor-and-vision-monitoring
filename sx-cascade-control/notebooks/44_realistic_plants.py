"""Tests 4 and 5 of the fair test: plants with physics the model doesn't have, and the ridge head to head.

The question. Tests 1 to 3 (notebook 43) ran on the perturbed plant: the model's own equations with
other numbers. There the physics fit, once it adjusts the acid exponent instead of a shared stage
rate, matches or beats the net. But the perturbed plant suits the fit: the fit adjusts the
generator's main knobs. How do the two do on plants built differently from the model?

The plants (main case, test plants 1 to 10, each on that seed's perturbed numbers and stage
efficiencies, so every one has known separation factors on top of the physics the model lacks):
  entrainment  each phase carries droplets of the other into the next stage (plant_models)
  settler      a mixer then three settler tanks per stage; same steady state as the perturbed
               plant, different dynamics (a check: at rest the readings are the perturbed plant's)
  book         the book plant (plant_models.BookCircuit): Pr/(La+Ce) 2.1 mid-cascade, about 20 %
               weaker toward the ends and 8 % weaker in the scrub, acid slope 2.5, slower transfer
               in acid, slow drift. It has no single true Pr/(La+Ce) factor: 2.1 is shown, and it's
               judged on closed-loop purity and recovery.
  drifting     the perturbed plant whose chemistry drifts over days. Readings are taken after 72 h
               on the design handles with the drift running (not at rest); the true factors are the
               drifted ones at 72 h, and the controller then runs on with the drift continuing.
  perturbed    the perturbed plant itself (Test 5: the ridge, the net and both fits on it)
The estimators (the twelve readings of notebooks 31 and 35, the same noise draws, seed + 5000):
  fit_old      notebook 31's fit as published: both factors and one shared stage rate, one start
  fit_new      both factors and the acid exponent, MAP, four starts (notebook 43, Tests 2 and 3)
  ridge, net   notebook 35's, trained on perturbed plants only (out of distribution here)
  ridge_mix, net_mix   the same, retrained with book and entrainment plants added (seeds 1000+;
               the settler and drifting plants at rest are perturbed plants, so they add nothing)
For each: the error on each factor, and the MPC (main case, over level loops) on the estimate,
120 h from the snapshot, with end, minimum and rest purity and recovery.

Run:  pixi run python notebooks/44_realistic_plants.py data     # book + entrainment training plants
      pixi run python notebooks/44_realistic_plants.py train    # ridge_mix and net_mix
      pixi run python notebooks/44_realistic_plants.py test [--kinds perturbed] [--no-loop]
Writes runs/learned_estimator_mix_ndpr_main/ and runs/realistic_plants_ndpr_main[_<kinds>].csv.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "notebooks"))
from src.commission import commission                                   # noqa: E402
from src.config import build_circuit, load_config, nominal_inputs       # noqa: E402
from src.mismatch import Mismatch, perturbed_params, stage_efficiency_factors   # noqa: E402
from src.plant_models import EntrainmentCircuit, SettlerCircuit         # noqa: E402
from src.registry import PLANTS                                         # noqa: E402
from src.simulate import simulate, steady_state                        # noqa: E402
from study_tools import (SPREAD, PlantCase, closed_loop, extended_model, notebook_35, run_jobs,   # noqa: E402
                         separation_factors, settle, workers, write_rows)

CONFIG = "ndpr_main"
SEEDS = tuple(range(1, 11))
KINDS = tuple(sys.argv[sys.argv.index("--kinds") + 1].split(",")) if "--kinds" in sys.argv else \
    ("entrainment", "book", "drifting", "settler")
TRAIN_KINDS = {"book": 10000, "entrainment": 20000}   # kind: offset on the seed in the training rows
TRAIN_SEEDS = range(1000, 1150)
CHUNK = 25
T_SNAP = 72.0                                   # h of drift before the drifting plant is read
MIX_DIR = HERE / "runs" / f"learned_estimator_mix_{CONFIG}"
OUT = HERE / "runs" / (f"realistic_plants_{CONFIG}.csv" if "--kinds" not in sys.argv
                       else f"realistic_plants_{CONFIG}_{'-'.join(KINDS)}.csv")
ESTIMATORS = ("fit_old", "fit_new", "ridge", "net", "ridge_mix", "net_mix")
NEW_SET = ("beta_pl", "beta_np", "dn_acid")     # the corrected fitted set (notebook 43, Test 2)


def modules():
    """Notebook 35 (estimators, plants) and notebook 31 (readings, noise, notebook 31's model) for the main case."""
    n35 = notebook_35(CONFIG)
    return n35, n35.nb31()


# ---- the plants -----------------------------------------------------------------------------------
def build_plant(kind: str, seed: int):
    """(rc, plant, stage efficiencies, model) for this kind on seed's perturbed numbers."""
    rc = load_config(CONFIG, overrides={"plant": {"mismatch": "perturbed", "seed": seed, "generator": "ratios"}})
    model = build_circuit(rc)
    if kind in ("book", "drifting", "perturbed"):
        plant, pk = PLANTS[kind].build(rc)
        return rc, plant, np.asarray(pk, float), model
    mm = Mismatch(seed=seed, strength=1.0, generator="ratios")
    cls = {"entrainment": EntrainmentCircuit, "settler": SettlerCircuit}[kind]
    plant = cls(model.cmap, perturbed_params(rc.params, mm), model=model.model, scrub_reflux=model.scrub_reflux,
                **({"cells": 3} if kind == "settler" else {}))
    return rc, plant, np.asarray(stage_efficiency_factors(model.N, mm), float), model


def truth(kind: str, plant, model) -> np.ndarray:
    """log (Pr/(La+Ce), Nd/Pr) against the model's: the plant's own numbers, drifted to T_SNAP for the
    drifting plant, Pr/(La+Ce) at its mid-cascade value for the book plant."""
    lb = np.log(separation_factors(plant) / separation_factors(model))
    names = list(plant.p.names)
    if kind == "drifting":
        d = plant.log_multipliers(T_SNAP) - plant.log_multipliers(0.0)
        lb = lb + np.array([d[names.index("Pr")] - d[names.index("LaCe")], d[names.index("Nd")] - d[names.index("Pr")]])
    if kind == "book":
        lb[0] = np.log(plant.beta_mid / separation_factors(model)[0])
    return lb


def snapshot(kind: str, rc, plant, pk) -> PlantCase:
    """The plant when it's read: at rest on the design handles, or for the drifting plant after T_SNAP
    hours there with the drift running (its clock then shifted so the closed loop continues the drift)."""
    x, u = commission(plant, nominal_inputs(rc, plant).replace(k_factor=pk))
    x = settle(plant, u, x0=x)
    if kind == "drifting":
        x = drift_for_hours(plant, x, u.numpy(), T_SNAP)
        base_hidden = plant.hidden_at
        plant.hidden_at = lambda xx, t_h, _f=base_hidden: _f(xx, t_h + T_SNAP)
    return PlantCase(rc, plant, pk, x, u.replace(k_factor=np.ones(plant.N)))


def drift_for_hours(plant, x, uv, hours: float):
    """The drifting plant run open loop for `hours` on inputs uv, the drift written in every half
    hour as the closed loop does. Returns the state at the end, drift included."""
    for k in range(int(hours / 0.5)):
        x = plant.hidden_at(x, k * 0.5)
        x = simulate(plant, x, lambda t: uv, 0.5, dt_out=0.5).x[-1]
    return plant.hidden_at(x, hours)


# ---- training data with the other physics -------------------------------------------------------
def make_chunk(job) -> str:
    """Training rows for one kind and CHUNK plants from `start`: each plant at its design handles and
    up to seven random handle settings, at rest (settled), with its label. Saved as one .npz."""
    kind, start = job
    path = MIX_DIR / f"chunk_{kind}_{start}.npz"
    if path.exists():
        return str(path)
    warnings.filterwarnings("ignore")
    n35, m = modules()
    rows = {"y": [], "h": [], "label": [], "seed": []}
    for seed in range(start, start + CHUNK):
        plant_rows((kind, seed), (n35, m), rows)
    np.savez(path, **{k: np.array(v) for k, v in rows.items()})
    return str(path)


def plant_rows(kind_seed: tuple, notebooks: tuple, rows: dict) -> None:
    """One training plant's rows, appended to rows (readings, handles, label, plant id).
    kind_seed: (plant kind, seed). notebooks: (notebook 35, notebook 31)."""
    kind, seed = kind_seed
    n35, m = notebooks
    rc, plant, pk, model = build_plant(kind, seed)
    label = truth(kind, plant, model)
    u0 = nominal_inputs(rc, plant).replace(k_factor=pk)
    rng = np.random.default_rng(seed)
    x_prev = None
    for k in range(n35.SETTINGS_PER_PLANT):
        changes = {} if k == 0 else {h: float(getattr(u0, h)) * (1 + rng.uniform(-s, s)) for h, s in n35.SPREAD.items()}
        u = u0.replace(**changes)
        x = settled_or_none(plant, u, x_prev)
        if x is None:
            continue
        x_prev = x if k == 0 else x_prev
        rows["y"].append(m.readings(plant, x, u, list(plant.p.names)))
        rows["h"].append(n35.handles_of(u, u0))
        rows["label"].append(label)
        rows["seed"].append(TRAIN_KINDS[kind] + seed)


def settled_or_none(plant, u, x0):
    """The plant at rest on u, or None if the solve fails or ends anywhere not at rest (not a sample)."""
    try:
        x = settle(plant, u, x0=x0)
    except RuntimeError:
        return None
    return x if np.abs(plant.rhs(x, u.numpy())).max() <= 1e-6 else None


def data():
    """All training chunks for the book and entrainment plants."""
    MIX_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(k, s) for k in TRAIN_KINDS for s in range(TRAIN_SEEDS.start, TRAIN_SEEDS.stop, CHUNK)]
    t0 = time.time()
    for n, (_, path) in enumerate(run_jobs(make_chunk, jobs, workers()), start=1):
        print(f"{n}/{len(jobs)} chunks ({time.time() - t0:.0f} s): {Path(path).name}", flush=True)


def train():
    """The mixed ridge and net: notebook 35's training on its own data plus these chunks."""
    n35, _ = modules()
    # notebook 35 runs as a script, so its pickles name the class __main__.Scaler and the loaders look
    # for it there; pickle these the same way
    import __main__
    n35.Scaler.__module__ = "__main__"
    __main__.Scaler = n35.Scaler
    n35.train(dirs=[n35.DATA_DIR, MIX_DIR], out_dir=MIX_DIR)


# ---- the estimators -----------------------------------------------------------------------------
def fit_old(m, case: PlantCase, y, sd) -> tuple:
    """Notebook 31's fit as published: both factors and one shared stage rate, one start from the
    nominal, the default steady-state ladder. Returns ((model, k, log factors), seconds)."""
    from scipy.optimize import least_squares
    names = list(case.plant.p.names)
    state = {"x": case.plant.observable(case.x_rest)}

    def residuals(theta):
        """Noise-scaled reading errors of notebook 31's model with these three log numbers."""
        mod, _, _ = m.model_with(case.rc, theta)
        x = steady_state(mod, case.u_design, x0=state["x"])
        state["x"] = x
        return (m.readings(mod, x, case.u_design, names) - y) / sd
    t0 = time.time()
    f = least_squares(residuals, np.zeros(3), bounds=(-0.5, 0.5), diff_step=0.02, max_nfev=60)
    return (m.model_with(case.rc, f.x)[0], np.ones(case.plant.N), f.x[:2]), time.time() - t0


def fit_new(m, case: PlantCase, y, sd) -> tuple:
    """The corrected fit: both factors and the acid exponent, a prior at each one's plausible spread,
    four starts on Pr/(La+Ce), settled solves. Returns ((model, k, log factors), seconds, acid offset)."""
    from scipy.optimize import least_squares
    names = list(case.plant.p.names)
    prior = np.array([SPREAD[c] for c in NEW_SET])
    state = {"x": case.plant.observable(case.x_rest)}

    def residuals(theta):
        """Noise-scaled reading errors and theta over its prior spread."""
        circ, kf = extended_model(case.rc, dict(zip(NEW_SET, theta)))
        u = case.u_design.replace(k_factor=kf)
        x = settle(circ, u, x0=state["x"])
        state["x"] = x
        return np.concatenate([(m.readings(circ, x, u, names) - y) / sd, np.asarray(theta) / prior])
    t0 = time.time()
    best = None
    for s0 in (0.0, 0.15, 0.3, -0.15):
        f = least_squares(residuals, np.array([s0, 0.0, 0.0]), bounds=([-0.5, -0.5, -1.0], [0.5, 0.5, 1.0]),
                          diff_step=0.02, max_nfev=100)
        best = f if best is None or f.cost < best.cost else best
    circ, kf = extended_model(case.rc, dict(zip(NEW_SET, best.x)))
    return (circ, kf, best.x[:2]), time.time() - t0, float(best.x[2])


def learned(n35, rc, y, handles) -> dict:
    """The ridge and net estimates, perturbed-only and (if trained) mixed: {name: (model, k, log factors)}."""
    out = {}
    for tag, folder in (("", None), ("_mix", MIX_DIR)):
        if folder is not None and not (folder / "net.pt").exists():
            continue
        net_est, ridge_est = n35.load_estimators(folder)
        for name, est in (("net", net_est), ("ridge", ridge_est)):
            e = est(y, handles)
            circ, kf = extended_model(rc, {"beta_pl": float(e[0]), "beta_np": float(e[1])})
            out[name + tag] = (circ, kf, np.asarray(e[:2]))
    return out


# ---- the test -------------------------------------------------------------------------------------
def test_one(job) -> dict:
    """One plant of one kind: every estimator's factors and (unless --no-loop) its closed loop."""
    kind, seed = job
    warnings.filterwarnings("ignore")
    n35, m = modules()
    rc, plant, pk, model = build_plant(kind, seed)
    case = snapshot(kind, rc, plant, pk)
    y_true = m.readings(plant, case.x_rest, case.u_design.replace(k_factor=pk), list(model.p.names))
    sd = m.noise_sd(y_true)
    y = y_true + np.random.default_rng(seed + 5000).normal(0.0, sd)
    b0, lb = separation_factors(model), truth(kind, plant, model)
    row = {"kind": kind, "seed": seed, "beta_pl_true": b0[0] * np.exp(lb[0]), "beta_np_true": b0[1] * np.exp(lb[1])}
    models = {}
    models["fit_old"], t_old = fit_old(m, case, y, sd)
    models["fit_new"], t_new, row["fit_new_dn_acid"] = fit_new(m, case, y, sd)
    row.update({"fit_old_s": round(t_old, 1), "fit_new_s": round(t_new, 1)})
    models.update(learned(n35, rc, y, np.zeros(len(n35.HANDLES))))     # read at the design handles
    for name, (mod, mk, log_b) in models.items():
        row[f"{name}_beta_pl"], row[f"{name}_beta_np"] = b0 * np.exp(log_b)
        if "--no-loop" not in sys.argv:
            row.update({f"{name}_{k}": v for k, v in closed_loop(case, mod, mk).items()})
    return row


def summary_line(r: dict) -> str:
    """One plant's result as a log line."""
    line = f"{r['kind']:11s} seed {r['seed']:2d}: Pr/LaCe true {r['beta_pl_true']:.3f} |"
    for e in (e for e in ESTIMATORS if f"{e}_beta_pl" in r):
        line += f" {e} {r[f'{e}_beta_pl']:.3f}"
        if f"{e}_purity_end" in r:
            line += f" ({100 * r[f'{e}_purity_end']:.2f}/{100 * r[f'{e}_recovery_end']:.2f}, rest {100 * r[f'{e}_rest_purity']:.2f})"
    return line


def test():
    """test_one for every kind and plant; written row by row."""
    rows = []
    for _, r in run_jobs(test_one, [(k, s) for k in KINDS for s in SEEDS], workers()):
        rows.append(r)
        print(summary_line(r), flush=True)
        write_rows(OUT, rows, first=("kind", "seed"))
    print("wrote", OUT.name)


if __name__ == "__main__":
    {"data": data, "train": train, "test": test}[[a for a in sys.argv[1:] if a in ("data", "train", "test")][0]]()
