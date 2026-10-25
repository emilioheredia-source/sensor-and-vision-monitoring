"""How wide is the operating window around the design point?

The question. The design (notebook 02) is the cheapest point that meets purity >= 99.5 % and
recovery >= 95 %, so both are exactly met there. Along the scrub flow (strip tied to it, as the
controller moves them) the band where purity >= 99 % (spec) and recovery >= 95 % is about 1 L/h
wide. Is that the circuit, or only the corner the optimizer chose? If more organic opens the
window, the design can buy margin; if not, the stage count is what's short.

The method, on the nominal model at steady state (no controller):
  - For each organic flow on a grid around the design's, with everything else at the design's
    (saponified fraction included, so reagent moves with the organic flow), find by bisection
      s_pur(Q_org)   the scrub flow where long-run purity crosses the line (99 % spec, 99.5 % design)
      s_rec(Q_org)   the scrub flow where long-run recovery crosses 95 %
    Purity rises with scrub and recovery falls, so the window is [s_pur, s_rec], width s_rec - s_pur
    (negative: no window at that organic flow).
  - Strip = net product (300 L/h) + scrub throughout.

Run:  pixi run python notebooks/19_operating_window.py [run_file]      (default ndpr_plant)
Writes runs/operating_window_<run_file>.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.commission import commission                                   # noqa: E402
from src.config import build_circuit, load_config, nominal_inputs       # noqa: E402
from src.metrics import heavy_mask                                      # noqa: E402
from src.simulate import steady_state                                   # noqa: E402

HERE = Path(__file__).resolve().parent.parent
CONFIG = sys.argv[1] if len(sys.argv) > 1 else "ndpr_plant"
rc = load_config(CONFIG)
circ = build_circuit(rc)
x0, u0 = commission(circ, nominal_inputs(rc, circ))
heavy = np.array(heavy_mask(circ.p, rc.cut))
NET = float(u0.Q_strip) - float(u0.Q_scrub)
NDPR_IN = float(u0.Q_feed) * np.asarray(u0.c_feed, float)[heavy].sum()
SPEC, DESIGN_PURITY = rc.targets()[0], rc.design_purity
RECOVERY = rc.targets()[1]
TOL = 0.02                                   # bisection stops when the scrub bracket is this narrow, L/h
_cache = {}


def outcome(q_org: float, q_scrub: float) -> tuple[float, float]:
    """(purity, recovery) at steady state, starting from the nearest point already solved."""
    key = (round(q_org, 3), round(q_scrub, 4))
    if key in _cache:
        return _cache[key][:2]
    u = u0.replace(Q_org=q_org, Q_scrub=q_scrub, Q_strip=NET + q_scrub)
    near = min(_cache.items(), key=lambda kv: abs(kv[0][0] - q_org) / 50 + abs(kv[0][1] - q_scrub),
               default=(None, (None, None, x0)))[1][2]
    x = steady_state(circ, u, x0=near)
    o = circ.outlets(x, u.numpy())
    cp = np.atleast_1d(o["c_prod"])
    result = (cp[heavy].sum() / cp.sum(), float(o["Q_prod"]) * cp[heavy].sum() / NDPR_IN)
    _cache[key] = (*result, x)
    return result


def crossing(q_org: float, which: int, level: float, bracket: tuple) -> float:
    """Scrub flow where purity (which=0, rising) or recovery (which=1, falling) crosses `level`,
    searched by bisection inside bracket = (lo, hi) scrub flows, to within TOL L/h."""
    lo, hi = bracket
    sign = 1 if which == 0 else -1
    f_lo, f_hi = sign * (outcome(q_org, lo)[which] - level), sign * (outcome(q_org, hi)[which] - level)
    if f_lo > 0 or f_hi < 0:
        return np.nan                                   # not bracketed in [lo, hi]
    while hi - lo > TOL:
        mid = 0.5 * (lo + hi)
        if sign * (outcome(q_org, mid)[which] - level) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


q_design = float(u0.Q_org)
rows = []
print(f"design: Q_org {q_design:.0f}, Q_scrub {float(u0.Q_scrub):.1f}, net product {NET:.0f} L/h")
print(" Q_org   scrub at purity 99 %   at 99.5 %   at recovery 95 %   window (spec)   window (design)")
for f in (0.90, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10, 1.15, 1.20):
    q = f * q_design
    lo, hi = 0.6 * f * float(u0.Q_scrub), 1.6 * f * float(u0.Q_scrub)    # brackets the window for this design
    s_spec = crossing(q, 0, SPEC, (lo, hi))
    s_des = crossing(q, 0, DESIGN_PURITY, (lo, hi))
    s_rec = crossing(q, 1, RECOVERY, (lo, hi))
    rows.append((q, s_spec, s_des, s_rec, s_rec - s_spec, s_rec - s_des))
    print(f"  {q:5.0f}   {s_spec:8.2f}            {s_des:8.2f}    {s_rec:8.2f}          "
          f"{s_rec - s_spec:+7.2f}         {s_rec - s_des:+7.2f}", flush=True)

out = HERE / "runs" / f"operating_window_{CONFIG}.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["Q_org", "scrub_purity_spec", "scrub_purity_design", "scrub_recovery_target",
                "window_spec_L_per_h", "window_design_L_per_h"])
    w.writerows(rows)
print("wrote", out)
