"""The controller on twenty plants through one upset: the purity and recovery traces, for the README.

The seed study (notebook 14) keeps only a score per run. This keeps the traces: the main case, the
MPC over level loops, the feed switch to the richer deposit, on the twenty plants drawn inside the
published separation-factor ranges (generator "ratios", seeds 1-20), 80 h each, the controller seeing
only the product, raffinate and feed analysers (sensors product_raffinate_feed, as in the workbench
run the README shows).

Run:  pixi run python notebooks/40_twenty_plants_traces.py [--full-state]
Writes runs/twenty_plants_traces_ndpr_main.npz (t, purity and recovery per seed). With --full-state the
controller sees the plant's state instead (sensors full_state, as in the seed study), written to
runs/twenty_plants_traces_ndpr_main_full_state.npz: the same runs with nothing between the plant and the
controller, to see how much of what happens comes from what the controller can see.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runs import RunSpec, execute                                   # noqa: E402

HERE = Path(__file__).resolve().parent.parent
FULL_STATE = "--full-state" in sys.argv
SENSORS = "full_state" if FULL_STATE else "product_raffinate_feed"
OUT = HERE / "runs" / f"twenty_plants_traces_ndpr_main{'_full_state' if FULL_STATE else ''}.npz"
SEEDS = range(1, 21)

traces = {}
for seed in SEEDS:
    r = execute(RunSpec(config="ndpr_main", plant="perturbed", controller="mpc_level_loops", upset="feed_switch",
                        sensors=SENSORS, hours=80.0, run_file={"plant": {"seed": seed, "generator": "ratios"}}))
    traces[f"t_{seed}"] = r.t
    traces[f"purity_{seed}"] = np.asarray(r.series["purity"])
    traces[f"recovery_{seed}"] = np.asarray(r.series["recovery"])
    print(f"seed {seed}: min purity {100 * traces[f'purity_{seed}'].min():.2f} %, "
          f"recovery at 80 h {100 * traces[f'recovery_{seed}'][-1]:.1f} %", flush=True)
np.savez(OUT, **traces)
print(f"wrote {OUT.name}")
