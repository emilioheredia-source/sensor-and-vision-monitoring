"""Stage-count study: how many stages does a cut need?

The cut is the base run file's: by default ndpr_sweep_base, the Ce | Pr cut that
makes the NdPr product; prnd_sweep_base is the Pr | Nd split used to check the
model against the published circuits.

For each (extraction, scrub) stage pair and each scrub mode, the operating
point is found by optimization on the steady-state model (src/design.py):
maximize product recovery subject to product purity >= spec, over organic flow,
scrub flow, scrub acidity and saponification. A few starting points per
circuit, best kept. The smallest circuit whose recovery clears the target is
the answer.

The two scrub modes: "none" is an acid-only scrub liquor; "reflux" sends part
of the strip product back as scrub liquor, so metal pushes metal off the organic.

Anchors to check against: Idaho 18 stages (12 scrub; 99.7 % purity, low
recovery); ECJTU 60 stages (26 + 34). REPORT.md sections 3 and 8 have the results.

Writes runs/stage_count<tag>.csv (every solve) and figures/04_stage_count<tag>.png, where the
tag is _ndpr or _prnd from the base, plus _ideal or _cap100 (e.g. stage_count_ndpr_ideal.csv).
Run:  pixi run python notebooks/04_stage_count.py [--base ndpr_sweep_base] [--quick] [--plot-only]
                                                   [--ideal] [--loading-cap 1.0]
      --base         the run file that sets the cut, the feed and the reagents (default ndpr_sweep_base)
      --quick        three circuit sizes and one starting point, to try the script out.
      --plot-only    redraws the figure from the CSV without solving.
      --ideal        near-ideal stages: the transfer rate x 20 (about 99 % stage efficiency at
                     the design flows, against about 80-85 % on the NdPr cut and about 70 % on Pr|Nd at the nominal rate).
      --loading-cap  organic loading cap as a fraction of capacity (default 0.85; 1.0 = none).
"""
import csv
import sys
import tomllib
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.circuit import Circuit, build_map                     # noqa: E402
from src.config import deep_update, load_config, nominal_inputs  # noqa: E402
from src.params import params_from_dict                        # noqa: E402
from src.design import optimize_operating_point, reflux_link   # noqa: E402
from src.plotting import FIGURES, INK, INK2, MUTED, SLOT, style   # noqa: E402

# ---- 1. the command line ----
# TAG goes into the output file names, so each variant of the study keeps its own CSV and figure.
HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs"
RUNS.mkdir(exist_ok=True)
BASE = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "ndpr_sweep_base"
QUICK = "--quick" in sys.argv
PLOT_ONLY = "--plot-only" in sys.argv
IDEAL = "--ideal" in sys.argv          # near-ideal stages: the transfer rate k x 20 (see the docstring);
                                       # k x 1000 is too stiff for the optimizer past ~20 stages
CAP = float(sys.argv[sys.argv.index("--loading-cap") + 1]) if "--loading-cap" in sys.argv else None
#   --loading-cap 1.0 lifts the 85 % organic loading cap of parameters.toml (the optimizer then runs the
#   organic as full as it likes); None keeps the parameter file's cap
TAG = (f"_{BASE.replace('_sweep_base', '')}" + ("_ideal" if IDEAL else "")
       + (f"_cap{int(round(100 * CAP))}" if CAP is not None else ""))

# ---- 2. the base run file and the study's settings ----
# p is the parameter set; with --ideal its transfer rate is raised 20 times.
rc = load_config(BASE)
p = rc.params
# --parameter-edits FILE (optional): changes to params/parameters.toml for this sweep, e.g. the
# stages sized to the design flows (params/edits_sized_stages.toml). The file's stem goes in TAG.
EDITS_FILE = Path(sys.argv[sys.argv.index("--parameter-edits") + 1]) if "--parameter-edits" in sys.argv else None
if EDITS_FILE is not None:
    p = params_from_dict(deep_update(p.raw, tomllib.loads(EDITS_FILE.read_text(encoding="utf-8"))))
    TAG += "_" + EDITS_FILE.stem.replace("edits_", "")
if IDEAL:
    from dataclasses import replace
    p = replace(p, k_transfer=p.k_transfer * 20)
SPEC = p.raw["targets"]["product_purity_fraction"]          # purity floor, fraction
RECOVERY_TARGET = p.raw["targets"]["recovery_fraction"]     # the line drawn on the figure, fraction
N_STRIP = 4                       # strip stages, the same in every circuit
NET_PRODUCT_FLOW = 300.0          # L/h of strip product leaving the circuit (reflux mode)

# circuit sizes to try: the Pr|Nd cut (separation factor 1.5) needs far more than the Ce|Pr cut (about 2)
PAIRS = ([(2, 6), (2, 12), (4, 12), (6, 18), (8, 24), (12, 30), (18, 36), (26, 34), (30, 44)] if rc.cut == "Pr"
         else [(2, 4), (2, 6), (4, 8), (4, 12), (6, 12), (6, 16), (8, 20), (10, 24), (14, 30)])
MODES = ["none", "reflux"]
# the handles the optimizer may move, with bounds: flows in L/h, scrub acid in mol/L, saponified fraction
FREE = {"Q_org": (600.0, 4000.0), "Q_scrub": (50.0, 2000.0), "h_scrub": (0.05, 2.0),
        "saponified_fraction": (0.0, 0.6)}
# starting points: (Q_org, Q_scrub, h_scrub, sapon)
STARTS = [(1200.0, 300.0, 0.5, 0.4), (2000.0, 400.0, 1.0, 0.5), (900.0, 600.0, 0.3, 0.5)]
if QUICK:
    PAIRS = [(2, 12), (6, 18), (12, 30)]
    STARTS = STARTS[:1]

# The base run file's own circuit, built only to get its nominal inputs (feed, strip acid, jacklegs).
c0 = rc["circuit"]
base = Circuit(build_map(c0["n_ext"], c0["n_scrub"], c0["n_strip"]), p)
u_nom = nominal_inputs(rc, base)

# ---- 3. with --plot-only: read the earlier solves back ----
# rows holds every solve; best holds, for each (scrub mode, total stages), the solve with the most
# recovery among those that converged and met the purity spec.
rows, best = [], {}
t_all = time.time()
if PLOT_ONLY:
    with open(RUNS / f"stage_count{TAG}.csv", newline="") as f:
        for r in csv.DictReader(f):
            # CSV values come back as text: keep mode and status, turn True/False into booleans,
            # everything else into numbers
            r = {k: (v if k in ("mode", "status")
                     else float(v) if v not in ("True", "False") else v == "True") for k, v in r.items()}
            rows.append(r)
            key = (r["mode"], int(r["N"]))
            if (r["success"] and r["purity"] >= SPEC - 1e-6
                    and (key not in best or r["recovery"] > best[key]["recovery"])):
                best[key] = r

# ---- 4. the sweep: every scrub mode, every circuit size, every starting point ----
# For each circuit: build it, then run the optimizer from each starting point. A solve that throws
# is kept as a failed row, so the CSV shows every attempt. In reflux mode the strip flow is tied to
# the scrub flow (strip = net product + scrub).
for mode in ([] if PLOT_ONLY else MODES):
    for n_ext, n_scrub in PAIRS:
        cmap = build_map(n_ext, n_scrub, N_STRIP)
        circ = Circuit(cmap, p, scrub_reflux=(mode == "reflux"))
        link = reflux_link(NET_PRODUCT_FLOW) if mode == "reflux" else None
        t0 = time.time()
        for Q_org, Q_scrub, h_scrub, sapon in STARTS:
            # the starting inputs: this start's four handles, every jackleg at the base file's first
            # one, and every stage at the nominal transfer rate
            u0 = u_nom.replace(Q_org=Q_org, Q_scrub=Q_scrub, h_scrub=h_scrub, saponified_fraction=sapon,
                               Q_strip=NET_PRODUCT_FLOW + (Q_scrub if mode == "reflux" else 0.0),
                               H_jackleg=np.full(cmap.N, u_nom.H_jackleg[0]), k_factor=np.ones(cmap.N))
            try:
                r = optimize_operating_point(circ, u0, rc.cut, SPEC, FREE, link=link, max_loading=CAP)
                row = dict(mode=mode, n_ext=n_ext, n_scrub=n_scrub, N=cmap.N,
                           start=STARTS.index((Q_org, Q_scrub, h_scrub, sapon)),
                           success=r.success, status=r.status, iterations=r.iterations,
                           recovery=r.recovery, purity=r.purity, Q_org=r.u.Q_org, Q_scrub=r.u.Q_scrub,
                           h_scrub=r.u.h_scrub, sapon=r.u.saponified_fraction, reagent=r.reagent_mol_per_h)
            except Exception as err:
                row = dict(mode=mode, n_ext=n_ext, n_scrub=n_scrub, N=cmap.N,
                           start=STARTS.index((Q_org, Q_scrub, h_scrub, sapon)),
                           success=False, status=f"error: {err}", iterations=0, recovery=np.nan, purity=np.nan,
                           Q_org=np.nan, Q_scrub=np.nan, h_scrub=np.nan, sapon=np.nan, reagent=np.nan)
            rows.append(row)
            key = (mode, cmap.N)
            if (row["success"] and row["purity"] >= SPEC - 1e-6
                    and (key not in best or row["recovery"] > best[key]["recovery"])):
                best[key] = row
        # one line per circuit: the best point found, or a note that none met spec.
        # O/A is the organic-to-aqueous flow ratio, organic flow over feed flow.
        b = best.get((mode, cmap.N))
        print(f"{mode:7s} ext={n_ext:2d} scrub={n_scrub:2d} N={cmap.N:2d}: {time.time()-t0:5.1f}s  "
              + (f"recovery {b['recovery']:.3f} at purity {b['purity']:.4f} | O/A {b['Q_org']/u_nom.Q_feed:.2f} "
                 f"Q_scrub {b['Q_scrub']:.0f} h_scrub {b['h_scrub']:.2f} sapon {b['sapon']:.2f}"
                 if b else "no converged point at spec"),
              flush=True)

# ---- 5. save every solve ----
if not PLOT_ONLY:
    print(f"total {len(rows)} solves in {time.time()-t_all:.0f}s")
    with open(RUNS / f"stage_count{TAG}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

# ---- 6. the figure ----
# Best recovery at spec against total stages, one line per scrub mode, with the recovery target as a
# dotted line. On the Pr|Nd cut, the two published circuits are marked as dashed lines.
# The title says which stage model was used: the nominal transfer rate (stages reach about 80-85 % on the NdPr cut, about 70 % on Pr|Nd, of
# equilibrium at the design flows; ARCHITECTURE.md, the stage equations) or the near-ideal one (rate x 20).
style()
fig, ax = plt.subplots(figsize=(7.2, 3.6))
Ns = [n_ext + n_scrub + N_STRIP for n_ext, n_scrub in PAIRS]
for k, mode in enumerate(MODES):
    rec = [best[(mode, N)]["recovery"] if (mode, N) in best else np.nan for N in Ns]
    ax.plot(Ns, rec, "-", color=SLOT[k])
    ax.plot(Ns, rec, "o", color=SLOT[k], markersize=4)
    i = 2                                                # label the third point, where the lines are apart and clear of the target line
    dy = -14 if mode == "none" else 8
    ax.annotate({"none": "acid-only scrub", "reflux": "reflux scrub"}[mode], (Ns[i], rec[i]), xytext=(10, dy),
                textcoords="offset points", va="center", color=INK2, fontsize=8)
ax.axhline(RECOVERY_TARGET, color=MUTED, lw=0.8, ls=":")
ax.text(0.99, RECOVERY_TARGET, f"recovery target {RECOVERY_TARGET:.0%} (chosen) ", va="top", ha="right",
        color=MUTED, fontsize=8, transform=ax.get_yaxis_transform())
if rc.cut == "Pr":        # the published Pr|Nd circuits, for the validation case only
    for N, lab in ((18, "Idaho 18 stages"), (60, "ECJTU 60 stages")):
        ax.axvline(N, color=MUTED, lw=0.8, ls="--")
        ax.text(N + 0.6, 0.03, lab, ha="left", va="bottom", color=MUTED, fontsize=8)
ax.set_xlabel(f"total stages (extraction + scrub + {N_STRIP} strip)")
ax.set_ylabel(f"{'Nd' if rc.cut == 'Pr' else 'NdPr'} recovery at >= {SPEC:.0%} purity")
ax.set_ylim(0, 1.02)
ax.set_title(("Pr|Nd cut, didymium feed" if rc.cut == "Pr" else "Ce|Pr cut, monazite feed, NdPr product")
             + ": best recovery at spec purity, by circuit size"
             + (" — near-ideal stages (transfer rate x 20)" if IDEAL else " — nominal transfer rate")
             + (f", organic loading up to {CAP:.0%}" if CAP is not None else ""), loc="left", color=INK, pad=10)
fig.savefig(FIGURES / f"04_stage_count{TAG}.png")
print("wrote", FIGURES / f"04_stage_count{TAG}.png")
