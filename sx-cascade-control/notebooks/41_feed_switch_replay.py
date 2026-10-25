"""Why do some plants dip below spec on the feed switch when the controller is told the feed changed?

The question (review session, Oct 20). On the feed switch to the richer deposit, the controller is
handed the feed's composition, acidity and flow as they are (full state, as the seed study runs
it), so it knows the switch is coming through. Notebook 40 --full-state still has 5 of 20 plants
dipping below the 99 % spec (seeds 3, 10, 14, 16, 17). Is that the model being wrong about the
plant, the move limits, the weights, or something else? And what did the controller do with each
handle?

The method. The five plants that dipped, the run of notebook 40 --full-state (main case, MPC over
level loops, feed switch, 80 h), four ways:
  as_run        the run as notebook 40 has it (a check: the minimum purity should match)
  true_model    the controller plans on the plant's own equations and numbers, stage efficiencies
                included: no model error at all
  rates_x3      the move limits per step three times wider (organic 150 L/h, scrub 30 L/h, ...)
  purity_x10    the purity weights ten times heavier (target and spec slacks)
Each run keeps the handles at every step (organic, scrub, scrub acid, saponified fraction), the
purity and the recovery, and how many steps each handle moved by its full limit.

Run:  pixi run python notebooks/41_feed_switch_replay.py [seed ...]   (default: 3 10 14 16 17)
Writes runs/feed_switch_replay_ndpr_main.csv (one row per run) and
runs/feed_switch_replay_ndpr_main.npz (the traces, keyed <variant>_<seed>_<series>).
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.loop import ClosedLoop                                          # noqa: E402
from src.operating_point import controller_settings                     # noqa: E402
from src.registry import PLANTS, mpc_over_level_loops                   # noqa: E402
from src.runs import RunSpec, start                                      # noqa: E402

HERE = Path(__file__).resolve().parent.parent
SEEDS = tuple(int(a) for a in sys.argv[1:] if a.isdigit()) or (3, 10, 14, 16, 17)
HOURS = 80.0
HANDLES = ("Q_org", "Q_scrub", "h_scrub", "saponified_fraction")
OUT_CSV = HERE / "runs" / "feed_switch_replay_ndpr_main.csv"
OUT_NPZ = HERE / "runs" / "feed_switch_replay_ndpr_main.npz"
FIELDS = (["variant", "seed", "min_purity", "hours_below_spec", "recovery_end", "purity_end"]
          + [f"{h}_{w}" for h in HANDLES for w in ("start", "end", "steps_at_limit")] + ["solver_failures"])


def spec_for(seed: int, variant: str) -> RunSpec:
    """Notebook 40 --full-state's run for this seed, with the variant's controller settings."""
    settings = {}
    if variant in ("rates_x3", "purity_x10"):
        base = controller_settings(start(RunSpec(hours=0.5, run_file=_plant(seed))).op.rc)
        if variant == "rates_x3":
            settings["rate_per_step"] = {k: 3 * v for k, v in base.rate_per_step.items()}
        else:
            settings["w_purity"] = 10 * base.w_purity
            settings["w_purity_floor"] = 10 * base.w_purity_floor
    return RunSpec(config="ndpr_main", plant="perturbed", controller="mpc_level_loops", upset="feed_switch",
                   sensors="full_state", hours=HOURS, run_file=_plant(seed), controller_settings=settings)


def _plant(seed: int) -> dict:
    """The run-file edit that picks test plant `seed` from the "ratios" generator."""
    return {"plant": {"seed": seed, "generator": "ratios"}}


def run(seed: int, variant: str):
    """One run; returns (the RunResult, the controller's move limits)."""
    session = start(spec_for(seed, variant))
    if variant == "true_model":
        # The plant's own equations and numbers as the controller's model, at the plant's rest
        # point; its stage efficiencies go in the model's inputs, so the loop doesn't add them again.
        op = session.op
        model, plant_k = PLANTS["perturbed"].build(op.rc)
        u_true = op.u_model.replace(k_factor=plant_k)
        ctl = mpc_over_level_loops(op.rc, model, op.x_plant, u_true)
        session.controller = ctl
        session.loop = ClosedLoop(op.plant, op.x_plant, ctl, session.upset.scenario(), None, None, None)
    for _ in session.steps():
        pass
    return session.result(), session.controller.s.rate_per_step


rows, traces = [], {}
for seed in SEEDS:
    for variant in ("as_run", "true_model", "rates_x3", "purity_x10"):
        r, rate = run(seed, variant)
        pur, rec = np.asarray(r.series["purity"]), np.asarray(r.series["recovery"])
        dt = np.diff(r.t)
        row = {"variant": variant, "seed": seed, "min_purity": pur.min(), "purity_end": pur[-1],
               "recovery_end": rec[-1], "hours_below_spec": float(np.sum(dt[pur[1:] < 0.99])),
               "solver_failures": r.scorecard.get("solver_failures", 0)}
        key = f"{variant}_{seed}"
        traces[f"{key}_t"], traces[f"{key}_purity"], traces[f"{key}_recovery"] = r.t, pur, rec
        for h in HANDLES:
            v = np.asarray(r.handles[h], float)
            traces[f"{key}_{h}"] = v
            moves = np.abs(np.diff(v))
            row[f"{h}_start"], row[f"{h}_end"] = v[0], v[-1]
            row[f"{h}_steps_at_limit"] = int(np.sum(moves >= 0.98 * rate[h]))
        rows.append(row)
        print(f"seed {seed:2d} {variant:10s} min purity {100 * row['min_purity']:.2f} %  "
              f"below spec {row['hours_below_spec']:4.1f} h  recovery end {100 * row['recovery_end']:.1f} %  "
              f"organic {row['Q_org_start']:.0f}->{row['Q_org_end']:.0f} (at limit {row['Q_org_steps_at_limit']})  "
              f"scrub {row['Q_scrub_start']:.0f}->{row['Q_scrub_end']:.0f} (at limit {row['Q_scrub_steps_at_limit']})  "
              f"acid {row['h_scrub_start']:.2f}->{row['h_scrub_end']:.2f}  "
              f"sapon {row['saponified_fraction_start']:.3f}->{row['saponified_fraction_end']:.3f}", flush=True)
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        np.savez(OUT_NPZ, **traces)
print("wrote", OUT_CSV.name, OUT_NPZ.name)
