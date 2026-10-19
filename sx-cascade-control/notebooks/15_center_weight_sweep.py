"""How strongly should the controller pull interfaces back to their setpoint?

The question. The controller has a small cost on each interface's distance from its setpoint
(MPCSettings.w_center), so after an upset the jacklegs bring interfaces back to the middle of
their window instead of leaving them anywhere inside it. Until Oct 13 a units slip made that term
1e8 too weak, so the weight had never been tried (ARCHITECTURE.md, decision 26). This sweep
tries five weights, from 0 (no pull) to 0.02 per cm^2 (the intended value).

The cases: the pump stop on the matched, perturbed and entrainment plants (the upset where the
interfaces move most), and a feed switch, a jackleg drift and a flowmeter bias on the perturbed
plant. 40 h each, full state, ndpr_plant.

What it showed: the feed switch, the drift and the flowmeter bias come out about the same at
every weight (the drift slightly better with the pull on). The pump stop jumps around with no
pattern: the same plant rides through at one weight and runs away at the next. So the pump stop
is fragile whatever the weight, and the weight stays at 0.02.

Run:  pixi run python notebooks/15_center_weight_sweep.py
Writes runs/center_weight_sweep_ndpr_plant.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runs import RunSpec, execute     # noqa: E402

HERE = Path(__file__).resolve().parent.parent
WEIGHTS = (0.0, 0.0002, 0.001, 0.005, 0.02)          # per cm^2 of interface offset
CASES = [("matched", "feed_interrupt"), ("perturbed", "feed_interrupt"), ("entrainment", "feed_interrupt"),
         ("perturbed", "feed_switch"), ("perturbed", "interface_drift"), ("perturbed", "flowmeter_bias")]
KEYS = ("off_spec_mol", "lost_excess_mol", "interface_excursion_m_h", "reagent_mol", "solver_failures")

rows = []
for w in WEIGHTS:
    for plant, upset in CASES:
        sc = execute(RunSpec(config="ndpr_plant", plant=plant, controller="mpc", upset=upset, hours=40.0,
                             controller_settings={"w_center": w})).scorecard
        rows.append({"w": w, "plant": plant, "upset": upset, **{k: round(float(sc[k]), 2) for k in KEYS}})
        print(rows[-1], flush=True)

out = HERE / "runs" / "center_weight_sweep_ndpr_plant.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("wrote", out)
