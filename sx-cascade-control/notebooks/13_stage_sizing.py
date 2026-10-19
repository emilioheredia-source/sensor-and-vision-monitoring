"""What changes when the stages are sized to the flows that go through them?

The question. The mixers are the Idaho lab rig's scaled 250 times: 30 L, sized for 600 L/h in
all at the lab's three-minute residence time. The ndpr_plant design puts up to 2,594 L/h through
a mixer, so its stages reach only 81-85 % of equilibrium (ARCHITECTURE.md, the stage equations).
params/edits_sized_stages.toml gives every stage at least three minutes at the design flows:
130 L mixers and 520 L settlers. The design step then gives configs/ndpr_plant_sized.toml.

This script compares the two plants on four things:
  1. stage count: recovery against circuit size (from notebook 04's two sweeps)
  2. how linear the plant is over the moves the controller makes: the change in purity and
     recovery 12 h after a step of one rate limit, against a step four times bigger divided by
     four. If the plant were linear the two would agree; the gap is the nonlinearity
     (the question behind the 85 % loading cap, ARCHITECTURE.md decision 22)
  3. how slow it is: the slowest time constant of the model at its design point
  4. the controller on both, on four upsets, 40 h, and on the bigger plant again at 120 h
     (it's slower, so 40 h shows less of each response)

Run (after notebook 04 has run with and without --parameter-edits params/edits_sized_stages.toml):
      pixi run python notebooks/13_stage_sizing.py
Writes runs/stage_sizing_ndpr.csv and figures/13_stage_sizing.png
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.commission import commission                                    # noqa: E402
from src.config import build_circuit, load_config, nominal_inputs        # noqa: E402
from src.metrics import point_outputs                                    # noqa: E402
from src.plotting import FIGURES, INK, INK2, MUTED, SLOT, style          # noqa: E402
from src.runs import RunSpec, execute                                    # noqa: E402
from src.simulate import simulate                                        # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs"
PLANTS = {"30 L mixers (ndpr_plant)": "ndpr_plant", "130 L mixers (ndpr_plant_sized)": "ndpr_plant_sized"}
UPSETS = ("feed_switch", "feed_interrupt", "flowmeter_bias", "interface_drift")
# one move of each handle, the size of the controller's rate limit per half-hour step (mpc.MPCSettings)
MOVES = {"Q_org": 50.0, "Q_scrub": 10.0, "h_scrub": 0.05, "saponified_fraction": 0.01}
HORIZON_H = 12.0          # the controller's horizon: 24 steps of 0.5 h


# ---- 1. the stage-count curves from notebook 04 ----
def best_recovery_by_size(path: Path) -> dict:
    """{total stages: best recovery at spec over both scrub modes} from a notebook-04 table."""
    best = {}
    for r in csv.DictReader(open(path)):
        if r["success"] != "True":
            continue
        n, rec = int(r["N"]), float(r["recovery"])
        best[n] = max(best.get(n, 0.0), min(rec, 1.0))
    return dict(sorted(best.items()))


sweeps = {"30 L mixers (ndpr_plant)": best_recovery_by_size(RUNS / "stage_count_ndpr.csv"),
          "130 L mixers (ndpr_plant_sized)": best_recovery_by_size(RUNS / "stage_count_ndpr_sized_stages.csv")}


# ---- 2 and 3. linearity and speed, on the model at its design point ----
def outputs_after(circ, x0, u, t_h: float) -> tuple[float, float]:
    """(purity, recovery) after t_h hours on constant inputs u, from state x0."""
    x = simulate(circ, x0, lambda _t: u, t_h, dt_out=t_h).x[-1]
    out = point_outputs(circ, x, u.numpy(), circ_cut)
    return out["purity"], out["recovery"]


def nonlinearity(circ, x0, u0, handle: str, move: float) -> dict:
    """Change in purity and recovery after HORIZON_H for a one-move step, and how far a
    four-move step divided by four is from it (as a fraction of the one-move change)."""
    p0, r0 = outputs_after(circ, x0, u0, HORIZON_H)
    one = outputs_after(circ, x0, u0.replace(**{handle: float(getattr(u0, handle)) + move}), HORIZON_H)
    four = outputs_after(circ, x0, u0.replace(**{handle: float(getattr(u0, handle)) + 4 * move}), HORIZON_H)
    d1 = np.array([one[0] - p0, one[1] - r0])
    d4 = np.array([four[0] - p0, four[1] - r0]) / 4
    gap = np.abs(d4 - d1) / np.maximum(np.abs(d1), 1e-12)
    return {"d_purity": d1[0], "d_recovery": d1[1], "nonlin_purity": gap[0], "nonlin_recovery": gap[1]}


def slowest_time_constant_h(circ, x, u) -> float:
    """1 / |slowest eigenvalue| of the model's Jacobian at (x, u), in hours."""
    eig = np.linalg.eigvals(circ.jac(x, u.numpy()))
    return float(1.0 / np.min(np.abs(eig.real[eig.real < 0])))


rows = []
for label, cfg in PLANTS.items():
    rc = load_config(cfg)
    circ = build_circuit(rc)
    circ_cut = rc.cut
    x0, u0 = commission(circ, nominal_inputs(rc, circ))
    tau = slowest_time_constant_h(circ, x0, u0)
    rows.append({"plant": label, "what": "slowest time constant", "value": tau})
    print(f"{label}: slowest time constant {tau:.0f} h", flush=True)
    for handle, move in MOVES.items():
        r = nonlinearity(circ, x0, u0, handle, move)
        print(f"  {handle:20s} 12 h after one move: purity {1e3 * r['d_purity']:+.3f} thousandths, recovery "
              f"{1e2 * r['d_recovery']:+.3f} hundredths; nonlinearity {r['nonlin_purity']:.0%} / "
              f"{r['nonlin_recovery']:.0%}", flush=True)
        rows += [{"plant": label, "what": f"{k} {handle}", "value": v} for k, v in r.items()]


# ---- 4. the controller on both plants ----
def run(cfg: str, controller: str, upset: str, hours: float) -> dict:
    """The scorecard of one run on the perturbed version of a run file's plant."""
    return execute(RunSpec(config=cfg, plant="perturbed", controller=controller, upset=upset,
                           hours=hours)).scorecard


cases = [(label, cfg, 40.0) for label, cfg in PLANTS.items()] + \
        [("130 L mixers (ndpr_plant_sized), 120 h", "ndpr_plant_sized", 120.0)]
for label, cfg, hours in cases:
    for upset in UPSETS:
        for controller in ("open_loop", "mpc"):
            sc = run(cfg, controller, upset, hours)
            print(f"{label} {upset} {controller}: off spec {sc['off_spec_mol']:.0f} mol, lost excess "
                  f"{sc['lost_excess_mol']:.0f} mol, interface {sc['interface_excursion_m_h']:.2f} m·h, "
                  f"reagent {sc['reagent_mol']:.0f} mol", flush=True)
            for k in ("off_spec_mol", "lost_excess_mol", "interface_excursion_m_h", "reagent_mol", "time_to_settle_h"):
                rows.append({"plant": label, "what": f"{upset} {controller} {k}", "value": sc[k]})

with open(RUNS / "stage_sizing_ndpr.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["plant", "what", "value"])
    w.writeheader()
    w.writerows(rows)
print("wrote", RUNS / "stage_sizing_ndpr.csv")


# ---- 5. the figure: stage count, and what the controller saves on each plant ----
style()
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
for k, (label, best) in enumerate(sweeps.items()):
    ax.plot(list(best), list(best.values()), "o-", color=SLOT[k], markersize=4, label=label)
ax.axhline(0.95, color=MUTED, lw=0.8, ls=":")
ax.set_xlabel("total stages (extraction + scrub + 4 strip)")
ax.set_ylabel("NdPr recovery at 99 % purity")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Stage count", loc="left", color=INK)

value = {(r["plant"], r["what"]): r["value"] for r in rows}
x = np.arange(len(UPSETS))
width = 0.2
for k, (label, cfg) in enumerate(PLANTS.items()):
    for j, controller in enumerate(("open_loop", "mpc")):
        v = [max(value[(label, f"{u} {controller} lost_excess_mol")], 0) + value[(label, f"{u} {controller} off_spec_mol")]
             for u in UPSETS]
        ax2.bar(x + (2 * k + j - 1.5) * width, v, width, color=SLOT[k], alpha=1.0 if controller == "mpc" else 0.4,
                label=f"{label.split(' (')[0]}, {'controller' if controller == 'mpc' else 'open loop'}")
ax2.set_xticks(x, [u.replace("_", " ") for u in UPSETS])
ax2.set_ylabel("NdPr lost beyond nominal + off spec, mol")
ax2.legend(frameon=False, fontsize=7.5)
ax2.set_title("40 h runs on the perturbed plant", loc="left", color=INK)
fig.suptitle("Stages sized to the design flows: 130 L mixers against 30 L", x=0.01, ha="left", color=INK)
fig.savefig(FIGURES / "13_stage_sizing.png")
print("wrote", FIGURES / "13_stage_sizing.png")
