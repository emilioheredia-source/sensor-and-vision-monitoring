"""Should the jacklegs be MPC handles, or local level loops under it?

The question. The MPC carries every settler's jackleg as a handle (decision 11 era). In the model
the interface moves with the densities but feeds nothing back into the chemistry (the hold-ups
are fixed at the setpoint), so purity and recovery gain nothing from the MPC owning the jacklegs.
Plants hold levels with a local loop per settler and put any APC above them. Does that cascade
do as well as the MPC owning the jacklegs, on the eight upsets, on a plant where the interface
does matter (entrainment), and on one with a different settler (settler)?

The method. Four controllers, full information, 40 h per upset:
  open_loop               nothing moves
  open_loop_level_loops   a PI loop per settler on its jackleg (loop.LevelLoops), nothing else
  mpc                     the MPC as built, jacklegs included
  mpc_level_loops         the MPC without the jacklegs and with its interface terms off, over the
                          same PI loops (a cascade with fixed interface setpoints; an MPC sending
                          setpoints would send these, since its model gains nothing from moving them)
Scored on NdPr lost beyond nominal + off spec (mol), interface time outside the window (m·h), the
largest distance of any interface from its setpoint (cm, at the control steps), and the time
per control step.

Run:  pixi run python notebooks/20_level_loop_cascade.py [plant ...]   (default: all three)
Writes runs/level_loop_cascade_ndpr_plant.csv (row by row; a stopped run picks up where it left off)
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config                                       # noqa: E402
from src.runs import RunSpec, execute                                    # noqa: E402

HERE = Path(__file__).resolve().parent.parent
CONFIG = "ndpr_plant"
HOURS = 40.0
PLANTS = tuple(sys.argv[1:]) or ("perturbed", "entrainment", "settler")
CONTROLLERS = ("open_loop", "open_loop_level_loops", "mpc", "mpc_level_loops")
UPSETS = ("feed_switch", "feed_step", "feed_acid", "feed_interrupt", "flowmeter_bias",
          "interface_drift", "reflux_acid", "efficiency_loss")
_geo = load_config(CONFIG).params.raw["stage_geometry"]
Z_SET = _geo["interface_setpoint_fraction"] * _geo["settler_height_m"]      # m
OUT = HERE / "runs" / f"level_loop_cascade_{CONFIG}.csv"
FIELDS = ["plant", "upset", "controller", "lost_plus_off_spec_mol", "off_spec_mol", "lost_excess_mol",
          "interface_excursion_m_h", "max_interface_dev_cm", "reagent_mol", "solver_failures", "s_per_step"]

done = set()
if OUT.exists():
    done = {(r["plant"], r["upset"], r["controller"]) for r in csv.DictReader(open(OUT))}
else:
    with open(OUT, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()

for plant in PLANTS:
    for upset in UPSETS:
        for controller in CONTROLLERS:
            if (plant, upset, controller) in done:
                continue
            t0 = time.time()
            r = execute(RunSpec(config=CONFIG, plant=plant, controller=controller, upset=upset, hours=HOURS))
            sc = r.scorecard
            z = np.asarray(r.series["interfaces"], float)
            row = {"plant": plant, "upset": upset, "controller": controller,
                   "lost_plus_off_spec_mol": max(sc["lost_excess_mol"], 0.0) + sc["off_spec_mol"],
                   "off_spec_mol": sc["off_spec_mol"], "lost_excess_mol": sc["lost_excess_mol"],
                   "interface_excursion_m_h": sc["interface_excursion_m_h"],
                   "max_interface_dev_cm": 100.0 * float(np.max(np.abs(z - Z_SET))),
                   "reagent_mol": sc["reagent_mol"], "solver_failures": sc.get("solver_failures", 0),
                   "s_per_step": (time.time() - t0) / (HOURS / 0.5)}
            with open(OUT, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            print(f"{plant:11s} {upset:16s} {controller:22s} lost+off {row['lost_plus_off_spec_mol']:7.0f} mol  "
                  f"outside {row['interface_excursion_m_h']:6.3f} m·h  max dev {row['max_interface_dev_cm']:5.1f} cm  "
                  f"{row['s_per_step']:.2f} s/step", flush=True)

# ---- summary: one table per plant ----
rows = list(csv.DictReader(open(OUT)))
for plant in PLANTS:
    print(f"\n{plant}: lost + off spec (mol) | interface outside window (m·h) | max interface distance (cm)")
    print(f"  {'upset':16s}" + "".join(f"{c:>34s}" for c in CONTROLLERS))
    for upset in UPSETS:
        cells = []
        for c in CONTROLLERS:
            r = next((r for r in rows if (r["plant"], r["upset"], r["controller"]) == (plant, upset, c)), None)
            cells.append("-" if r is None else f"{float(r['lost_plus_off_spec_mol']):6.0f} | "
                         f"{float(r['interface_excursion_m_h']):5.2f} | {float(r['max_interface_dev_cm']):5.1f}")
        print(f"  {upset:16s}" + "".join(f"{c:>34s}" for c in cells))
