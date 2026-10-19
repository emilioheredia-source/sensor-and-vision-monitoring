"""Why are some plants hard to control: the controller's model, or the plant itself?

The question. On the seed study's realistic plants (notebook 14, generator "ratios"), twelve of
twenty settle on spec after the 30 h trim and the controller handles the upsets well. Seven are
left resting off spec, and a longer trim made it worse (seed 2: scrub pushed from 239 to 1346 L/h,
purity 96 %). Notebook 17 says each of those plants can make spec with the right handles. So
either the controller's model leads it the wrong way on them, or something about the plant
defeats any controller of this kind.

The method: the same MPC (settings, horizon, limits, full information) started on each plant at
the design handles, commissioned, and run 120 h with no upset, two ways:
  nominal model   the controller plans on the model, as in every study (the trim does this)
  true model      the controller plans on the plant's own equations and numbers, the stage
                  efficiencies included: no model error at all
  one number      the nominal model with only the Pr/(La+Ce) separation factor set to the plant's
                  (what one lab shake-out test on the plant's feed and extractant would give)
If the true-model controller brings the hard plants to spec and the nominal one doesn't, the
cause is model error, and the cure is identification or adaptation. If both fail, it's the plant
(or the design's margin), and no better model fixes it.

Plants: the seven hard ones and three easy ones as a check, all generator "ratios".
Scored at the end of the run and along it: purity, recovery, the scrub and organic flows where the
controller left them, hours to reach spec and stay there, and solver failures.

With --acid-floor, both controllers also keep every extraction stage at or above 1e-3 mol/L of
acid (MPCSettings.acid_floor, decision 33), the floor the design step uses.

Run:  pixi run python notebooks/21_model_vs_plant.py [--acid-floor] [--one-number] [seed ...]
Writes runs/model_vs_plant_ndpr_plant.csv, or runs/model_vs_plant_acid_floor_ndpr_plant.csv (row by row)
"""
import csv
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.circuit import Circuit                                          # noqa: E402
from src.commission import commission                                   # noqa: E402
from src.config import build_circuit, load_config, nominal_inputs       # noqa: E402
from src.loop import run_closed_loop                                    # noqa: E402
from src.metrics import heavy_mask, product_purity                      # noqa: E402
from src.mpc import CircuitMPC                                          # noqa: E402
from src.operating_point import controller_settings                     # noqa: E402
from src.registry import PLANTS                                         # noqa: E402
from src.simulate import steady_state                                   # noqa: E402

HERE = Path(__file__).resolve().parent.parent
CONFIG = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "ndpr_plant"
HOURS = 120.0
HARD, EASY = (2, 3, 6, 10, 14, 16, 20), (1, 7, 12)
ACID = "--acid-floor" in sys.argv
MODELS = ("nominal", "true", "one number") if "--one-number" in sys.argv else ("nominal", "true")
EXTRA = {"acid_floor": 1e-3} if ACID else {}
# --margin: the controller aims at 99.8 % / 97 % (operating with margin, notebook 26) instead of the
# design's 99.5 % / 95 %; scoring stays against the spec (99 % purity, 95 % recovery)
MARGIN = "--margin" in sys.argv
SEEDS = tuple(int(a) for a in sys.argv[1:] if a.isdigit()) or HARD + EASY
OUT = HERE / "runs" / ("model_vs_plant" + ("_acid_floor" if ACID else "") + ("_margin" if MARGIN else "") + f"_{CONFIG}.csv")


def targets(rc) -> tuple:
    """(purity target, recovery target) the controller aims at."""
    return (0.998, 0.97) if MARGIN else (rc.design_purity, rc.targets()[1])
FIELDS = ["seed", "group", "model", "purity_end", "recovery_end", "rest_purity", "rest_recovery",
          "Q_scrub_end", "Q_org_end", "hours_to_spec", "solver_failures", "seconds"]


def run(seed: int, which: str) -> dict:
    """Commission seed's plant on the design handles and let the MPC run it for HOURS."""
    rc = load_config(CONFIG, overrides={"plant": {"mismatch": "perturbed", "seed": seed, "generator": "ratios"}})
    plant, plant_k = PLANTS["perturbed"].build(rc)
    x_rest, u_design = commission(plant, nominal_inputs(rc, plant).replace(k_factor=plant_k))
    settings = controller_settings(rc, **EXTRA)
    if which == "true":
        # The plant's own equations, numbers and stage efficiencies: a second copy of the plant,
        # its efficiencies carried in the controller's inputs (so the loop must not apply them again).
        model, _ = PLANTS["perturbed"].build(rc)
        ctl = CircuitMPC(model, x_rest, u_design, rc.cut, rc.targets()[0], *targets(rc), settings)
        traj, log = run_closed_loop(plant, x_rest, ctl, None, HOURS, plant_k_factor=None)
    else:
        model, nominal_k = build_circuit(rc), np.ones(plant.N)
        if which == "one number":
            # only K(La+Ce) moves, so that K(Pr) / K(La+Ce) is the plant's; everything else nominal
            names = list(model.p.names)
            i_la, i_pr = names.index("LaCe"), names.index("Pr")
            K = np.array(model.p.K, float).copy()
            K[i_la] = K[i_pr] / (plant.p.K[i_pr] / plant.p.K[i_la])
            model = Circuit(model.cmap, dataclasses.replace(model.p, K=K), model=model.model,
                            scrub_reflux=model.scrub_reflux)
        x_model = steady_state(model, u_design.replace(k_factor=nominal_k), x0=plant.observable(x_rest))
        ctl = CircuitMPC(model, x_model, u_design.replace(k_factor=nominal_k), rc.cut, rc.targets()[0],
                         *targets(rc), settings)
        traj, log = run_closed_loop(plant, x_rest, ctl, None, HOURS, plant_k_factor=plant_k)
    heavy = np.array(heavy_mask(plant.p, rc.cut))
    c_feed = np.asarray(u_design.c_feed, float)

    def pur_rec(x, uv):
        o = plant.outlets(x, uv)
        cp = np.atleast_1d(o["c_prod"])
        fed = float(u_design.Q_feed) * c_feed[heavy].sum()
        return product_purity(cp, plant.p, rc.cut), float(o["Q_prod"]) * cp[heavy].sum() / fed

    pr = np.array([pur_rec(x, uv) for x, uv in zip(traj.x, traj.u)])
    on = (pr[:, 0] >= rc.targets()[0]) & (pr[:, 1] >= rc.targets()[1] - 5e-4)     # spec, interpolation slack
    # hours to spec: the first time after which both targets hold to the end
    off_idx = np.where(~on)[0]
    hours_to_spec = 0.0 if len(off_idx) == 0 else (np.nan if off_idx[-1] == len(on) - 1 else traj.t[off_idx[-1] + 1])
    # where the plant would come to rest on the handles the controller ended with
    u_end = traj.u[-1]
    x_end = steady_state(plant, u_end, x0=traj.x[-1])
    rest = pur_rec(x_end, u_end)
    from src.circuit import Inputs
    lay = Inputs.layout(plant.n, plant.N)
    return {"purity_end": pr[-1, 0], "recovery_end": pr[-1, 1], "rest_purity": rest[0], "rest_recovery": rest[1],
            "Q_scrub_end": float(u_end[lay["Q_scrub"].start]), "Q_org_end": float(u_end[lay["Q_org"].start]),
            "hours_to_spec": hours_to_spec,
            "solver_failures": int(sum(1 for r in log if r.get("qp_ok") is False))}


done = set()
if OUT.exists():
    done = {(int(r["seed"]), r["model"]) for r in csv.DictReader(open(OUT))}
else:
    with open(OUT, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
for seed in SEEDS:
    for which in MODELS:
        if (seed, which) in done:
            continue
        t0 = time.time()
        try:
            row = run(seed, which)
        except Exception as e:                       # keep going; say what broke
            print(f"seed {seed} {which}: failed: {e!r}", flush=True)
            continue
        row.update(seed=seed, group="hard" if seed in HARD else "easy", model=which,
                   seconds=round(time.time() - t0))
        with open(OUT, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"seed {seed:2d} ({row['group']}) {which:7s} end purity {100 * row['purity_end']:6.2f} % recovery "
              f"{100 * row['recovery_end']:6.2f} %   at rest {100 * row['rest_purity']:6.2f} / "
              f"{100 * row['rest_recovery']:6.2f} %   scrub {row['Q_scrub_end']:6.0f}  organic {row['Q_org_end']:5.0f}  "
              f"to spec {row['hours_to_spec']} h  ({row['seconds']} s)", flush=True)
