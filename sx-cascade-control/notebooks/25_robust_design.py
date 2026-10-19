"""Phase 2: which plant design copes with not knowing the separation factors?

The question. Our design (notebook 02) is the cheapest set of flows that just meets 99.5 %
purity and 95 % recovery at the model's separation factors (Pr/La+Ce 2.0, Nd/Pr 1.51). Plants
that separate a little worse start off spec and need 15-25 % more organic; the window of flows
that meets both targets is a few percent wide; the Chinese design method puts the scrub ~60 %
above its minimum instead (FINDINGS_why_hard.md). What do stages, stage size and the way the
flows are chosen buy, against the same twenty realistic plants?

What is fixed when the plant is built, and what isn't:
  built      stages in each section, the stage size (mixer and settler volume), the organic pump
  operated   flows, scrub acid, saponification: the operator (or controller) can retune them

The designs (a grid):
  stages     8+20, 10+26, 12+32, 14+38 (extraction + scrub; strip 4)
  basis      corner       least reagent at 99.5 % / 95 %, model separation factors (what we did)
             pessimistic  the same, designed on the low end of the published separation factors
                          (Pr/La+Ce 1.8, Nd/Pr 1.3): flows that would still do if the plant is weak
             backoff      least reagent at 99.8 % / 97 %, model separation factors, run at 99.5 / 95
  size       mixer 130 L (three minutes at the sized plant's design flows) or 195 L (+50 %);
             settler four times the mixer, as in the lab rig

Each design against each of the twenty realistic plants (generator "ratios", the same plants
whatever the design assumed), at steady state:
  as built    the plant on the design's flows: purity and recovery out of the box
  retuned     the plant's own least-reagent flows that meet 99.5 % / 95 %, with the organic pump
              up to its 4700 L/h maximum (the oracle: what an operator who knew the plant could do).
              Feasible or not, reagent, and the organic flow needed
  window      at the retuned point, the band of scrub flows (strip tied, organic held) where purity
              >= 99 % and recovery >= 95 %, as a percentage of the scrub flow: how much room a
              controller has
Cost proxy: total stage volume (stages x (mixer + settler)), and reagent.

Run:  pixi run python notebooks/25_robust_design.py [design ...]   (no args: all designs, then the summary)
      pixi run python notebooks/25_robust_design.py summary
Writes runs/robust_design/<design>.csv (one row per plant) and runs/robust_design_summary.csv
"""
import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.circuit import Circuit, build_map                               # noqa: E402
from src.config import load_config, nominal_inputs                       # noqa: E402
from src.design import optimize_operating_point, reflux_link             # noqa: E402
from src.metrics import heavy_mask                                       # noqa: E402
from src.mismatch import Mismatch, perturbed_params, stage_efficiency_factors   # noqa: E402
from src.params import params_from_dict                                  # noqa: E402
from src.simulate import steady_state                                    # noqa: E402

HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "runs" / "robust_design"
BASE = load_config("ndpr_sweep_base")             # cut, feed, reagents (as notebook 02)
NET = 300.0                                       # L/h net product (reflux mode)
PUMP_MAX = 4700.0
DESIGN_FREE = {"Q_org": (600.0, 0.85 * PUMP_MAX), "Q_scrub": (50.0, 2000.0), "h_scrub": (0.05, 2.0),
               "saponified_fraction": (0.0, 0.6)}
OPERATE_FREE = dict(DESIGN_FREE, Q_org=(600.0, PUMP_MAX))     # in operation the pump may go to its maximum
SPEC, RUN_PURITY, RUN_RECOVERY = 0.99, 0.995, 0.95
STAGES = {"8+20": (8, 20), "10+26": (10, 26), "12+32": (12, 32), "14+38": (14, 38)}
BASES = {"corner": dict(beta_pr_lace=None, beta_nd_pr=None, purity=0.995, recovery=0.95),
         "pessimistic": dict(beta_pr_lace=1.8, beta_nd_pr=1.3, purity=0.995, recovery=0.95),
         "backoff": dict(beta_pr_lace=None, beta_nd_pr=None, purity=0.998, recovery=0.97)}
SIZES = {"130L": 130.0, "195L": 195.0}
DESIGNS = [f"{s}_{b}_{v}" for s in STAGES for b in BASES for v in SIZES]
SEEDS = range(1, 21)


def sized_params(mixer_L: float, beta_pr_lace=None, beta_nd_pr=None):
    """parameters.toml with the stage size, and (for the design only) other separation factors."""
    raw = copy.deepcopy(BASE.params.raw)
    raw["stage_geometry"]["mixer_volume_L"] = mixer_L
    raw["stage_geometry"]["settler_volume_L"] = 4.0 * mixer_L
    K = raw["equilibrium"]["K"]
    if beta_nd_pr is not None:
        K["Nd"] = K["Pr"] * beta_nd_pr
    if beta_pr_lace is not None:
        K["LaCe"] = K["Pr"] / beta_pr_lace
    return params_from_dict(raw)


def design(name: str):
    """The design's circuit (built on the true-size, model-K parameters) and its flows."""
    st, basis, size = name.split("_")
    n_ext, n_scrub = STAGES[st]
    b = BASES[basis]
    cmap = build_map(n_ext, n_scrub, 4)
    p_design = sized_params(SIZES[size], b["beta_pr_lace"], b["beta_nd_pr"])
    circ_d = Circuit(cmap, p_design, scrub_reflux=True)
    u0 = nominal_inputs(BASE, circ_d).replace(Q_strip=NET + 300.0)
    r1 = optimize_operating_point(circ_d, u0, BASE.cut, b["purity"], DESIGN_FREE, link=reflux_link(NET))
    r2 = optimize_operating_point(circ_d, r1.u, BASE.cut, b["purity"], DESIGN_FREE, link=reflux_link(NET),
                                  x_guess=r1.x, objective="reagent", min_recovery=b["recovery"])
    return cmap, r2


def window(plant, u, x, heavy, fed) -> float:
    """Width of the scrub band (strip tied, everything else held) meeting purity >= spec and
    recovery >= 95 %, as a percentage of the scrub flow at u. NaN if it can't be bracketed."""
    q0 = float(u.Q_scrub)
    cache = {}

    def out(q):
        if q not in cache:
            uu = u.replace(Q_scrub=q, Q_strip=NET + q)
            xx = steady_state(plant, uu, x0=x)
            o = plant.outlets(xx, uu.numpy())
            cp = np.atleast_1d(o["c_prod"])
            cache[q] = (cp[heavy].sum() / cp.sum(), float(o["Q_prod"]) * cp[heavy].sum() / fed)
        return cache[q]

    def cross(which, level, lo, hi, rising):
        f = lambda q: (out(q)[which] - level) * (1 if rising else -1)      # noqa: E731
        if f(lo) > 0 or f(hi) < 0:
            return np.nan
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        return 0.5 * (lo + hi)

    s_pur = cross(0, SPEC, 0.7 * q0, q0 * 1.001, rising=True)          # purity rises with scrub
    s_rec = cross(1, RUN_RECOVERY, q0 * 0.999, 1.5 * q0, rising=False)  # recovery falls with scrub
    return 100.0 * (s_rec - s_pur) / q0


def evaluate(name: str):
    out_file = OUT_DIR / f"{name}.csv"
    if out_file.exists():
        return
    t0 = time.time()
    cmap, d = design(name)
    if not d.success:
        print(f"{name}: design failed ({d.status})", flush=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_file.write_text("seed,design_failed\n0,1\n")
        return
    st, basis, size = name.split("_")
    p_true = sized_params(SIZES[size])                        # the real plant's starting numbers
    heavy = np.array(heavy_mask(p_true, BASE.cut))
    fed = float(d.u.Q_feed) * np.asarray(d.u.c_feed, float)[heavy].sum()
    stage_volume = cmap.N * 5.0 * SIZES[size]                  # mixer + settler (4 x mixer), every stage
    rows = []
    for seed in SEEDS:
        m = Mismatch(seed=seed, strength=1.0, generator="ratios")
        plant = Circuit(cmap, perturbed_params(p_true, m), scrub_reflux=True)
        k = stage_efficiency_factors(cmap.N, m)
        u_built = d.u.replace(k_factor=k)
        x_b = steady_state(plant, u_built)
        o = plant.outlets(x_b, u_built.numpy())
        cp = np.atleast_1d(o["c_prod"])
        pur_b, rec_b = cp[heavy].sum() / cp.sum(), float(o["Q_prod"]) * cp[heavy].sum() / fed
        r = optimize_operating_point(plant, u_built, BASE.cut, RUN_PURITY, OPERATE_FREE, link=reflux_link(NET),
                                     x_guess=x_b, objective="reagent", min_recovery=RUN_RECOVERY)
        win = window(plant, r.u, r.x, heavy, fed) if r.success else np.nan
        rows.append({"seed": seed, "built_purity": pur_b, "built_recovery": rec_b,
                     "built_on_spec": int(pur_b >= SPEC and rec_b >= RUN_RECOVERY - 5e-4),
                     "retune_ok": int(r.success), "retune_reagent": r.reagent_mol_per_h if r.success else np.nan,
                     "retune_Q_org": float(r.u.Q_org) if r.success else np.nan,
                     "retune_Q_scrub": float(r.u.Q_scrub) if r.success else np.nan,
                     "window_pct": win})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"design_Q_org": float(d.u.Q_org), "design_Q_scrub": float(d.u.Q_scrub),
            "design_sapon": float(d.u.saponified_fraction), "design_h_scrub": float(d.u.h_scrub),
            "design_reagent": d.reagent_mol_per_h, "stages": cmap.N, "stage_volume_L": stage_volume}
    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + list(meta))
        w.writeheader()
        w.writerows([{**r, **meta} for r in rows])
    ok = sum(r["built_on_spec"] for r in rows)
    print(f"{name}: design Q_org {meta['design_Q_org']:.0f} scrub {meta['design_Q_scrub']:.0f} reagent "
          f"{meta['design_reagent']:.0f}; on spec as built {ok}/20; retune feasible "
          f"{sum(r['retune_ok'] for r in rows)}/20; median window "
          f"{np.nanmedian([r['window_pct'] for r in rows]):.1f} %  ({time.time() - t0:.0f} s)", flush=True)


def summary():
    rows = []
    for name in DESIGNS:
        f = OUT_DIR / f"{name}.csv"
        if not f.exists():
            continue
        d = list(csv.DictReader(open(f)))
        if "design_failed" in d[0]:
            rows.append({"design": name, "failed": 1})
            continue
        num = lambda k: np.array([float(r[k]) for r in d])      # noqa: E731
        rows.append({"design": name, "failed": 0, "stages": int(d[0]["stages"]),
                     "stage_volume_L": float(d[0]["stage_volume_L"]), "design_reagent": float(d[0]["design_reagent"]),
                     "design_Q_org": float(d[0]["design_Q_org"]),
                     "on_spec_as_built": int(num("built_on_spec").sum()),
                     "worst_built_purity": float(num("built_purity").min()),
                     "worst_built_recovery": float(num("built_recovery").min()),
                     "retune_feasible": int(num("retune_ok").sum()),
                     "median_retune_reagent": float(np.nanmedian(num("retune_reagent"))),
                     "max_retune_Q_org": float(np.nanmax(num("retune_Q_org"))),
                     "median_window_pct": float(np.nanmedian(num("window_pct"))),
                     "worst_window_pct": float(np.nanmin(num("window_pct")))})
    out = HERE / "runs" / "robust_design_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k != "design", k)))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["failed"]:
            print(f"  {r['design']:28s} design failed")
            continue
        print(f"  {r['design']:28s} volume {r['stage_volume_L'] / 1000:6.1f} m3  reagent {r['design_reagent']:5.0f}  "
              f"as built on spec {r['on_spec_as_built']:2d}/20 (worst {100 * r['worst_built_purity']:6.2f} / "
              f"{100 * r['worst_built_recovery']:6.2f} %)  retune {r['retune_feasible']:2d}/20  window median "
              f"{r['median_window_pct']:5.1f} % worst {r['worst_window_pct']:5.1f} %")
    print("wrote", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["summary"]:
        summary()
        sys.exit(0)
    for name in (args or DESIGNS):
        evaluate(name)
    if not args:
        summary()
