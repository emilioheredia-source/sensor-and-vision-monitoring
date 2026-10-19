"""Does ranking the purity spec above recovery help the plants that can't make both?

The question. The seed study (notebook 14) found five perturbed plants that can't reach 95 %
recovery at spec with any handles, and on them the controller ended far below spec. One reading
was that purity and recovery are both soft floors with fixed squared weights, so purity could be
traded for recovery a little at a time; the industrial answer is ranked constraints. This script
tries two forms of it against the controller without:

  plain         the controller as built (OSQP deterministic, qpOASES as a backup: decision 30)
  two-stage     a first QP finds the smallest shortfall below spec the handles can reach; the
                full QP may not exceed it (MPCSettings.ranked)
  exact 1000    one QP, with a straight-line cost of 1000 per thousandth below spec on top of the
                squared one (MPCSettings.w_purity_floor_linear)
  exact 100     the same with 100

Each variant is used for the commissioning trim as well as the run, so each plant is trimmed by
the controller being tested. Cases: seeds 1 and 2 (plants that can make spec) through a feed
switch, seeds 3 and 13 (plants that can't) with no upset, 40 h, full state.

What it showed (Oct 14): ranking doesn't help. The two-stage form made plants that could make spec
worse (seed 2 collapsed to 61 % purity on the feed switch); the exact penalty changed little either
way. What helped the hard plants was the solver backup: without it OSQP often called the problem
infeasible and the controller held still. The rest of the gap is the model: on those plants it
underestimates how much purity the scrub flow buys by about fifteen times, so the controller moves
too timidly. The fix for that is adapting the model's gains, not ranking.

Run:  pixi run python notebooks/16_ranked_constraints.py
Writes runs/ranked_constraints_ndpr_plant.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.operating_point as operating_point_module      # noqa: E402
import src.runs as runs_module                             # noqa: E402
from src.runs import RunSpec, execute                      # noqa: E402

HERE = Path(__file__).resolve().parent.parent
VARIANTS = {"plain": {"ranked": False},
            "two-stage": {"ranked": True},
            "exact 1000": {"ranked": False, "w_purity_floor_linear": 1000.0},
            "exact 100": {"ranked": False, "w_purity_floor_linear": 100.0}}
CASES = ((1, "feed_switch"), (2, "feed_switch"), (3, "none"), (13, "none"))

# The trim builds its controller from operating_point.controller_settings, and the cached operating
# point is keyed without controller settings. So for each variant both are wrapped: the trim uses
# the variant's settings, and its cache entry gets the variant's name.
_settings = operating_point_module.controller_settings
_key = runs_module._operating_point_key


def use_variant(name: str, fields: dict) -> None:
    """Make the trim and the cache key follow this variant."""
    operating_point_module.controller_settings = lambda rc, **f: _settings(rc, **{**fields, **f})
    runs_module._operating_point_key = lambda spec: _key(spec) + "_" + name.replace(" ", "")


rows = []
for name, fields in VARIANTS.items():
    use_variant(name, fields)
    for seed, upset in CASES:
        r = execute(RunSpec(config="ndpr_plant", plant="perturbed", controller="mpc", upset=upset, hours=40.0,
                            run_file={"plant": {"seed": seed}}, controller_settings=fields))
        sc, purity = r.scorecard, r.series["purity"]
        solvers = [i.get("qp_solver") for i in r.info]
        row = {"variant": name, "seed": seed, "upset": upset, "purity_start": purity[0],
               "purity_min": min(purity), "purity_end": purity[-1], "recovery_end": r.series["recovery"][-1],
               "off_spec_mol": sc["off_spec_mol"], "lost_excess_mol": sc["lost_excess_mol"],
               "solver_failures": sc["solver_failures"], "fallback_solves": solvers.count("qpoases")}
        rows.append(row)
        print(f"{name:10s} seed {seed:2d} {upset:11s}: purity {row['purity_start']:.4f} -> {row['purity_end']:.4f} "
              f"(min {row['purity_min']:.4f}), recovery end {row['recovery_end']:.3f}, off spec "
              f"{row['off_spec_mol']:.0f}, lost excess {row['lost_excess_mol']:.0f}, failures "
              f"{row['solver_failures']}, backup solves {row['fallback_solves']}", flush=True)

out = HERE / "runs" / "ranked_constraints_ndpr_plant.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("wrote", out)
