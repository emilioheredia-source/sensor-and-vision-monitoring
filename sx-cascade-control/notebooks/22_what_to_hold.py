"""What should the plant measure and hold, so it stays on spec without a correct model?

The question. The design sits where both targets are just met, and the band of scrub flows that
meets both is a few L/h wide at any organic flow (notebook 19): the flows have to match the plant
and the feed closely. A controller that works this out from a model is only as good as the model
(notebooks 14, 21). Plants that run well do something simpler: they hold a measurement inside the
cascade at a fixed value with a handle, the way a distillation column holds the temperature of a
sensitive tray with the reflux. Which measurements, where, keep purity and recovery on spec across
everything the model could be wrong about?

The method (self-optimizing control, Skogestad 2000; sensitive-tray selection in distillation):
  1. Cases. The nominal plant under feed changes a plant sees (part of a deposit switch, stronger
     feed, more acid in the feed, 5 % more organic than the meter says), and the twenty realistic
     perturbed plants of the seed study (generator "ratios"). The controller is told none of them.
  2. For each case, at twelve organic flows (70 % to 150 % of the design's, saponified fraction
     unchanged so reagent moves with it), sweep the scrub flow from 100 to 900 L/h (strip tied:
     strip = net product + scrub), refined every 0.5 L/h where purity crosses spec, and solve the
     steady state at each point: purity, recovery, every stage's aqueous composition.
  3. Strategies, each scored on all cases:
       one handle   scrub holds one measurement at its nominal-design value, organic at design
       two handles  scrub holds one measurement, organic holds a second one (two-point control)
     Reference strategies: nothing held; a perfect product purity analyser (one handle); perfect
     purity and recovery analysers (two handles); and the best the handles can do for that case
     knowing everything (oracle).
     Candidate measurements, each at every stage:
        lace_frac   La+Ce share of the aqueous metal                  (XRF)
        pr          Pr in the aqueous, mol/L                           (UV-Vis or XRF)
        pr_ratio    Pr / (Pr + Nd) in the aqueous                      (UV-Vis band ratio: no gain error)
        ndpr_frac   (Pr + Nd) share of the aqueous metal               (XRF)
        metal       total metal in the aqueous, mol/L                  (density meter, roughly)
  4. On spec: purity >= 99 % and recovery >= 94.95 % (the design is at 95.00 % and interpolation
     shouldn't fail it). Reported: cases on spec, worst purity, worst recovery.

All steady state: where holding a measurement takes the plant in the long run, not how fast.

Run:  pixi run python notebooks/22_what_to_hold.py sweep CASE ...   (sweeps only; for parallel batches)
      pixi run python notebooks/22_what_to_hold.py                  (sweeps what's missing, then scores)
Writes runs/what_to_hold/*.npz (one per case) and runs/what_to_hold_ndpr_plant.csv
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.commission import commission                                   # noqa: E402
from src.config import build_circuit, load_config, nominal_inputs       # noqa: E402
from src.disturbances import FeedSwitch                                 # noqa: E402
from src.metrics import heavy_mask                                      # noqa: E402
from src.registry import PLANTS                                         # noqa: E402
from src.simulate import steady_state                                   # noqa: E402

HERE = Path(__file__).resolve().parent.parent
CONFIG = "ndpr_plant"
SWEEP_DIR = HERE / "runs" / "what_to_hold"
SCRUB = np.arange(100.0, 901.0, 10.0)             # L/h, coarse; refined every 0.5 L/h at the purity edge
ORGANIC = (0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50)   # times the design's organic flow
RECOVERY_OK = 0.9495                              # recovery counted as on target (design: 0.9500)
rc0 = load_config(CONFIG)
nominal = build_circuit(rc0)
x_design, u_design = commission(nominal, nominal_inputs(rc0, nominal))
NET = float(u_design.Q_strip) - float(u_design.Q_scrub)
Q_ORG = float(u_design.Q_org)
SPEC = rc0.targets()[0]
names = list(nominal.p.names)
I_LACE, I_PR, I_ND = names.index("LaCe"), names.index("Pr"), names.index("Nd")


def _feed_partly_switched(share: float):
    """The design's inputs with the feed `share` of the way to the Yangibana deposit."""
    new = FeedSwitch(t=0.0, ramp_h=0.0, to="yangibana", params=rc0.params,
                     removed_upstream=tuple(rc0["feed"]["removed_upstream"])).apply(1.0, u_design)
    c = (1 - share) * np.asarray(u_design.c_feed, float) + share * np.asarray(new.c_feed, float)
    return u_design.replace(c_feed=c)


def cases() -> dict:
    """name -> (plant circuit, inputs at the design handles). The organic meter bias is a plant
    that gets 5 % more organic than every setting says, so its organic grid is scaled by 1.05."""
    out = {
        "nominal": (nominal, u_design, 1.0),
        "feed 10 % to Yangibana": (nominal, _feed_partly_switched(0.10), 1.0),
        "feed 25 % to Yangibana": (nominal, _feed_partly_switched(0.25), 1.0),
        "feed +10 % stronger": (nominal, u_design.replace(c_feed=1.1 * np.asarray(u_design.c_feed, float)), 1.0),
        "feed acid 0.05 to 0.15": (nominal, u_design.replace(h_feed=0.15), 1.0),
        "organic meter 5 % low": (nominal, u_design, 1.05),
    }
    for seed in range(1, 21):
        rc = load_config(CONFIG, overrides={"plant": {"mismatch": "perturbed", "seed": seed, "generator": "ratios"}})
        plant, k = PLANTS["perturbed"].build(rc)
        out[f"plant seed {seed}"] = (plant, nominal_inputs(rc, plant).replace(k_factor=k), 1.0)
    return out


def _file(name: str) -> Path:
    """The .npz file a sweep is cached in, named after the case."""
    return SWEEP_DIR / (name.replace(" ", "_").replace("%", "pc").replace("+", "plus") + ".npz")


def sweep_level(plant, u0, q_org: float, x_from) -> dict:
    """Steady states along the scrub flow at one organic flow: purity, recovery, aqueous profile."""
    heavy = np.array(heavy_mask(plant.p, rc0.cut))
    fed = float(u0.Q_feed) * np.asarray(u0.c_feed, float)[heavy].sum()
    pts, states = {}, {}

    def solve(q, x0):
        """The steady state at scrub flow q (strip tied, organic held), warm-started from x0; records purity,
        recovery and the aqueous profile there, and returns the state."""
        u = u0.replace(Q_org=q_org, Q_scrub=q, Q_strip=NET + q)
        x, pts[round(q, 3)] = _point(plant, u, x0, (heavy, fed))
        states[round(q, 3)] = x
        return x

    x = x_from
    for q in SCRUB:
        x = solve(q, x)
    for q in _edge_points(pts):                       # refine where purity first reaches spec
        solve(q, states[min(states, key=lambda p: abs(p - q))])
    return _level_arrays(pts, states)


def _point(plant, u, x0, heavy_fed: tuple) -> tuple:
    """The steady state at inputs u, warm-started from x0, and (purity, recovery, aqueous profile) there.
    heavy_fed: (the product side's mask, the product metal fed in mol/h)."""
    heavy, fed = heavy_fed
    x = steady_state(plant, u, x0=x0)
    o = plant.outlets(x, u.numpy())
    cp = np.atleast_1d(o["c_prod"])
    c_aq = np.array([np.asarray(v, float).ravel() for v in plant.unpack(plant.observable(x))[0]])
    tot = cp.sum()
    return x, (cp[heavy].sum() / tot if tot > 0 else np.nan, float(o["Q_prod"]) * cp[heavy].sum() / fed, c_aq)


def _edge_points(pts: dict) -> list:
    """The scrub flows to add every 0.5 L/h over 20 L/h from just before purity first reaches spec
    (none if it never does), skipping ones already solved."""
    qs = sorted(pts)
    above = np.where(np.nan_to_num(np.array([pts[q][0] for q in qs])) >= SPEC)[0]
    if not len(above):
        return []
    q_edge = qs[max(above[0] - 1, 0)]
    return [q for q in np.arange(q_edge, q_edge + 20.01, 0.5) if round(q, 3) not in pts]


def _level_arrays(pts: dict, states: dict) -> dict:
    """One organic flow's sweep as arrays in scrub-flow order, with the first state (the next sweep's start)."""
    qs = np.array(sorted(pts))
    return {"q": qs, "purity": np.array([pts[q][0] for q in qs]), "recovery": np.array([pts[q][1] for q in qs]),
            "c_aq": np.array([pts[q][2] for q in qs]), "x_first": states[min(states)]}


def sweep_case(plant, u0, org_scale: float, wanted) -> dict:
    """One sweep per organic flow in `wanted`, from the design's outward so each starts near a
    solved state. Returns {factor: sweep}."""
    levels = {}
    x_start = steady_state(plant, u0.replace(Q_scrub=SCRUB[0], Q_strip=NET + SCRUB[0]))
    for f in sorted(wanted, key=lambda f: abs(f - 1.0)):
        prev = [g for g in levels if abs(g - 1.0) < abs(f - 1.0)]
        x0 = levels[min(prev, key=lambda g: abs(g - f))]["x_first"] if prev else x_start
        levels[f] = sweep_level(plant, u0, f * Q_ORG * org_scale, x0)
    return levels


def candidates(c_aq: np.ndarray) -> dict:
    """Every candidate measurement at every stage, from aqueous profiles (..., N, n)."""
    tot = np.maximum(c_aq.sum(axis=-1), 1e-12)
    return {"lace_frac": c_aq[..., I_LACE] / tot,
            "pr": c_aq[..., I_PR],
            "pr_ratio": c_aq[..., I_PR] / np.maximum(c_aq[..., I_PR] + c_aq[..., I_ND], 1e-12),
            "ndpr_frac": (c_aq[..., I_PR] + c_aq[..., I_ND]) / tot,
            "metal": tot}


def crossing(q: np.ndarray, v: np.ndarray, q_ref: float):
    """(q, fraction a, index i) where v crosses 0 nearest q_ref, by linear interpolation; or None."""
    v0, v1 = v[:-1], v[1:]
    idx = np.where(np.isfinite(v0) & np.isfinite(v1) & ((v0 == 0) | (v0 * v1 < 0)))[0]
    if len(idx) == 0:
        return None
    d = v0[idx] - v1[idx]
    a = np.where(d == 0, 0.0, v0[idx] / np.where(d == 0, 1.0, d))
    qq = q[idx] + a * (q[idx + 1] - q[idx])
    k = int(np.argmin(np.abs(qq - q_ref)))
    return float(qq[k]), float(a[k]), int(idx[k])


def at(arr: np.ndarray, hit) -> np.ndarray:
    """arr interpolated at a crossing (q, a, i)."""
    _, a, i = hit
    return arr[i] + a * (arr[i + 1] - arr[i])


# ---- 1. the sweeps ----
SWEEP_DIR.mkdir(parents=True, exist_ok=True)
ALL = cases()
only_sweep = len(sys.argv) > 1 and sys.argv[1] == "sweep"
for name in (sys.argv[2:] if only_sweep else list(ALL)):
    have = dict(np.load(_file(name))) if _file(name).exists() else {}
    missing = [f for f in ORGANIC if f"q_{f:.2f}" not in have]      # only the organic flows not yet swept
    if not missing:
        continue
    t0 = time.time()
    plant, u0, scale = ALL[name]
    levels = sweep_case(plant, u0, scale, missing)
    have.update({f"{k}_{f:.2f}": v for f, lv in levels.items() for k, v in lv.items() if k != "x_first"})
    np.savez(_file(name), **have)
    print(f"swept {name}: {sum(len(lv['q']) for lv in levels.values())} points, {time.time() - t0:.0f} s", flush=True)
if only_sweep:
    sys.exit(0)


# ---- 2. strategies ----
def load(name: str) -> list:
    """A cached sweep: for each organic flow, the scrub flows with their purity, recovery and aqueous profile."""
    d = np.load(_file(name))
    return [{k: d[f"{k}_{f:.2f}"] for k in ("q", "purity", "recovery", "c_aq")} for f in ORGANIC]


sweeps = {name: load(name) for name in ALL}
I_DESIGN = ORGANIC.index(1.00)
Q_DESIGN = float(u_design.Q_scrub)
nom = sweeps["nominal"][I_DESIGN]
i_d = int(np.argmin(np.abs(nom["q"] - Q_DESIGN)))
DESIGN = {k: v[i_d] for k, v in candidates(nom["c_aq"]).items()}       # (N,) per kind
N = nominal.N


def one_handle(level: dict, values: np.ndarray, target: float):
    """Scrub holds values == target at this organic flow: (purity, recovery) there, or None."""
    hit = crossing(level["q"], values - target, Q_DESIGN)
    return None if hit is None else (at(level["purity"], hit), at(level["recovery"], hit))


def two_handles(levels: list, held_a: tuple, held_b: tuple):
    """Scrub holds A == ta at every organic flow; the organic flow is then where B == tb.
    held_a, held_b: (va, ta) and (vb, tb), va and vb functions level -> values along that level's
    sweep. (purity, recovery) or None."""
    (va, ta), (vb, tb) = held_a, held_b
    curve = []                                      # per organic level: (B, purity, recovery) with A held
    for lv in levels:
        hit = crossing(lv["q"], va(lv) - ta, Q_DESIGN)
        curve.append(None if hit is None else (at(vb(lv), hit) - tb, at(lv["purity"], hit), at(lv["recovery"], hit)))
    org = np.array(ORGANIC)
    ok = [i for i, c in enumerate(curve) if c is not None]
    if len(ok) < 2:
        return None
    b = np.array([curve[i][0] for i in ok])
    hit = crossing(org[ok], b, 1.0)
    if hit is None:
        return None
    pur = np.array([curve[i][1] for i in ok])
    rec = np.array([curve[i][2] for i in ok])
    return at(pur, hit), at(rec, hit)


def oracle(levels: list):
    """The best recovery with purity >= spec anywhere on the grid; or None."""
    best = None
    for found in (_best_on_level(lv) for lv in levels):
        if found is not None and (best is None or found[1] > best[1]):
            best = found
    return best


def _best_on_level(lv: dict):
    """(purity, recovery) with the best recovery at purity >= spec at one organic flow; or None."""
    ok = np.nan_to_num(lv["purity"]) >= SPEC
    if not ok.any():
        return None
    i = int(np.argmax(np.where(ok, lv["recovery"], -1)))
    return lv["purity"][i], lv["recovery"][i]


def score(label: str, results: dict) -> dict:
    """One strategy's results over the cases: how many on spec, the worst purity and recovery, how many with
    no solution, and which failed."""
    vals = list(results.values())
    ok = {n: r is not None and r[0] >= SPEC and r[1] >= RECOVERY_OK for n, r in results.items()}
    pur = [r[0] for r in vals if r is not None]
    rec = [r[1] for r in vals if r is not None]
    return {"strategy": label, "cases_on_spec": int(sum(ok.values())), "cases": len(vals),
            "worst_purity": float(min(pur)) if pur else np.nan, "worst_recovery": float(min(rec)) if rec else np.nan,
            "no_solution": int(sum(r is None for r in vals)), "failed": ";".join(n for n, o in ok.items() if not o)}


cand = {name: [candidates(lv["c_aq"]) for lv in levels] for name, levels in sweeps.items()}
for name, levels in sweeps.items():                 # each level carries its candidates, for the pair search
    for lv, c in zip(levels, cand[name]):
        lv["cand"] = c
rows = []
rows.append(score("nothing held (design handles)",
                  {n: one_handle(lv[I_DESIGN], lv[I_DESIGN]["q"], Q_DESIGN) for n, lv in sweeps.items()}))
rows.append(score("one handle: product purity at 99.5 % (perfect analyser)",
                  {n: one_handle(lv[I_DESIGN], lv[I_DESIGN]["purity"], rc0.design_purity) for n, lv in sweeps.items()}))
rows.append(score("two handles: purity 99.5 % and recovery 95 % (perfect analysers)",
                  {n: two_handles(lv, (lambda l: l["purity"], rc0.design_purity), (lambda l: l["recovery"], 0.95))
                   for n, lv in sweeps.items()}))
rows.append(score("oracle: best the two handles can do", {n: oracle(lv) for n, lv in sweeps.items()}))
N_REF = len(rows)
KINDS = ("lace_frac", "pr", "pr_ratio", "ndpr_frac", "metal")
for kind in KINDS:
    for j in range(N):
        rows.append(score(f"one handle: {kind} at stage {j}",
                          {n: one_handle(sweeps[n][I_DESIGN], cand[n][I_DESIGN][kind][:, j], DESIGN[kind][j])
                           for n in sweeps}))
N_ONE = len(rows)
# two handles: scrub holds A (scrub section or product end), organic holds B (extraction section)
n_ext = rc0["circuit"]["n_ext"]
A_STAGES = range(n_ext, N)
B_STAGES = range(0, n_ext + 2)
for ka in ("lace_frac", "pr", "pr_ratio"):
    for ja in A_STAGES:
        for kb in ("ndpr_frac", "pr", "metal", "pr_ratio"):
            for jb in B_STAGES:
                rows.append(score(f"two handles: scrub holds {ka} at {ja}, organic holds {kb} at {jb}",
                                  {n: two_handles(sweeps[n],
                                                  (lambda l, k=ka, j=ja: l["cand"][k][:, j], DESIGN[ka][ja]),
                                                  (lambda l, k=kb, j=jb: l["cand"][k][:, j], DESIGN[kb][jb]))
                                   for n in sweeps}))

out = HERE / "runs" / f"what_to_hold_{CONFIG}.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def show(r):
    """One strategy's line on the console."""
    print(f"  {r['strategy']:78s} {r['cases_on_spec']:2d}/{r['cases']}  worst purity {100 * r['worst_purity']:6.2f} %"
          f"  worst recovery {100 * r['worst_recovery']:6.2f} %  no solution {r['no_solution']}")


def rank(r):
    """Sort key: most cases on spec first, then the least shortfall in the worst case."""
    return (-r["cases_on_spec"], -np.nan_to_num(min(r["worst_purity"] / SPEC, r["worst_recovery"] / 0.95)))


print(f"{len(sweeps)} cases. Reference strategies:")
for r in rows[:N_REF]:
    show(r)
print("Best single measurements (scrub holds it, organic at design):")
for r in sorted(rows[N_REF:N_ONE], key=rank)[:10]:
    show(r)
print("Best pairs (scrub holds the first, organic the second):")
for r in sorted(rows[N_ONE:], key=rank)[:15]:
    show(r)
print("wrote", out)
