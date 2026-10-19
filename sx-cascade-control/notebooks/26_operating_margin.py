"""Phase 2, second pass: margin is where you choose to operate. What does it cost, by design?

Why a second pass. Notebook 25 compared designs by how the plants do on the design's own flows
("as built"), then retuned to the cheapest flows that meet 99.5 % / 95 %. Two things came out:
  - as built, almost no plant was on spec in any design. Every perturbed plant has the same
    structural error (acid exponent 3.3 against the model's 3.0, src/mismatch.py dn_acid) and
    stage transfer rates drawn below nominal, so they all miss in the same direction whatever the
    design assumed. Real plants are commissioned and retuned anyway, so "as built" says little.
  - retuned to the cheapest point, every plant sits on its own edge again, whatever the design:
    the retuned flows were identical across the three design bases. The margin isn't in the
    design flows; it's in the operating targets. The design (stages, stage size) decides what
    that margin costs.

So here each plant is retuned to an operating policy, and scored against the spec:
  edge      least reagent at 99.5 % purity, 95 % recovery (what we've done so far)
  margin    least reagent at 99.8 %, 97 %
  wide      least reagent at 99.9 %, 98 %
For each stage layout (8+20 ... 14+38) and stage size (130 L, 195 L mixers), and each of the
twenty realistic plants:
  feasible   can the plant reach the policy's targets with the organic pump up to 4700 L/h?
  reagent    mol/h at that point; organic flow needed
  window     the band of scrub flows (strip tied, everything else held) that keeps purity >= 99 %
             and recovery >= 95 % (the spec), as a percentage of the scrub flow: how big a flow
             error or upset the plant can take before going off spec
Cost: total stage volume and reagent. Margin should show as a wider window for more reagent;
more stages should make that margin cheaper.

Run:  pixi run python notebooks/26_operating_margin.py [case ...]   (case = <stages>_<size>_<policy>)
      pixi run python notebooks/26_operating_margin.py summary
Writes runs/operating_margin/<case>.csv and runs/operating_margin_summary.csv
"""
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("robust", HERE / "notebooks" / "25_robust_design.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)                          # notebook 25's plants, design step and window

from src.circuit import Circuit, build_map           # noqa: E402
from src.config import nominal_inputs                 # noqa: E402
from src.design import optimize_operating_point, reflux_link   # noqa: E402
from src.metrics import heavy_mask                    # noqa: E402
from src.mismatch import Mismatch, perturbed_params, stage_efficiency_factors   # noqa: E402
from src.simulate import steady_state                 # noqa: E402

OUT_DIR = HERE / "runs" / "operating_margin"
POLICIES = {"edge": (0.995, 0.95), "margin": (0.998, 0.97), "wide": (0.999, 0.98)}
CASES = [f"{s}_{v}_{p}" for s in R.STAGES for v in R.SIZES for p in POLICIES]


def evaluate(case: str):
    out_file = OUT_DIR / f"{case}.csv"
    if out_file.exists():
        return
    t0 = time.time()
    st, size, policy = case.split("_")
    purity_t, recovery_t = POLICIES[policy]
    n_ext, n_scrub = R.STAGES[st]
    cmap = build_map(n_ext, n_scrub, 4)
    p_true = R.sized_params(R.SIZES[size])
    heavy = np.array(heavy_mask(p_true, R.BASE.cut))
    # a starting point: the model's own edge design for this layout and size (notebook 25's step)
    _, d = R.design(f"{st}_corner_{size}")
    fed = float(d.u.Q_feed) * np.asarray(d.u.c_feed, float)[heavy].sum()
    rows = []
    for seed in R.SEEDS:
        m = Mismatch(seed=seed, strength=1.0, generator="ratios")
        plant = Circuit(cmap, perturbed_params(p_true, m), scrub_reflux=True)
        u0 = d.u.replace(k_factor=stage_efficiency_factors(cmap.N, m))
        x0 = steady_state(plant, u0)
        r = optimize_operating_point(plant, u0, R.BASE.cut, purity_t, R.OPERATE_FREE, link=reflux_link(R.NET),
                                     x_guess=x0, objective="reagent", min_recovery=recovery_t)
        win = R.window(plant, r.u, r.x, heavy, fed) if r.success else np.nan
        rows.append({"seed": seed, "feasible": int(r.success),
                     "reagent": r.reagent_mol_per_h if r.success else np.nan,
                     "Q_org": float(r.u.Q_org) if r.success else np.nan,
                     "Q_scrub": float(r.u.Q_scrub) if r.success else np.nan,
                     "window_pct": win, "stages": cmap.N, "stage_volume_L": cmap.N * 5.0 * R.SIZES[size]})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{case}: feasible {sum(r['feasible'] for r in rows)}/20, median reagent "
          f"{np.nanmedian([r['reagent'] for r in rows]):.0f}, median window "
          f"{np.nanmedian([r['window_pct'] for r in rows]):.1f} % ({time.time() - t0:.0f} s)", flush=True)


def summary():
    rows = []
    for case in CASES:
        f = OUT_DIR / f"{case}.csv"
        if not f.exists():
            continue
        d = list(csv.DictReader(open(f)))
        num = lambda k: np.array([float(r[k]) for r in d])      # noqa: E731
        ok = num("feasible") > 0
        rows.append({"case": case, "stage_volume_m3": float(d[0]["stage_volume_L"]) / 1000.0,
                     "feasible": int(ok.sum()), "median_reagent": float(np.nanmedian(num("reagent"))),
                     "max_Q_org": float(np.nanmax(num("Q_org"))),
                     "median_window_pct": float(np.nanmedian(num("window_pct"))),
                     "worst_window_pct": float(np.nanmin(num("window_pct")))})
    out = HERE / "runs" / "operating_margin_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['case']:22s} volume {r['stage_volume_m3']:5.1f} m3  feasible {r['feasible']:2d}/20  reagent "
              f"{r['median_reagent']:5.0f}  max organic {r['max_Q_org']:5.0f}  window median "
              f"{r['median_window_pct']:5.1f} % worst {r['worst_window_pct']:5.1f} %")
    print("wrote", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["summary"]:
        summary()
        sys.exit(0)
    for case in (args or CASES):
        evaluate(case)
    if not args:
        summary()
