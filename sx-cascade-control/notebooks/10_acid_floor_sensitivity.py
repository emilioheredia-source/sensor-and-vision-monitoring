"""How much does the acid floor inside D (parameters.toml, chosen 1e-3 M) shape
the plant operating point? The optimizer runs the extraction section right
where the floor starts to act, so this has to be measured before any flow is
quoted. Redo the design step at three floor values and compare.

The floor: D has the acid to the power n in its denominator, so as the acid goes
to zero D would blow up. The model keeps it finite by adding the floor to the
power n (h_min_D ** n) to that denominator (src/chemistry.py). The floor is a
number I chose, so it shouldn't decide the answer.

Method: the same two-step design as notebook 02 (most recovery at 99.5 % purity,
then least reagent at the recovery target), at floors of 3e-4, 1e-3 and 3e-3 M.
One line per floor: recovery, purity, O/A, scrub flow and acid, saponification,
reagent, and the lowest stage acid in the circuit. If the second step fails, the
line shows the first step's result. ARCHITECTURE.md ("Open", acid floor) has
what came out: the floor doesn't move the operating point.

Assumes a reflux scrub with 300 L/h net product (the ndpr_* run files).

Run:  pixi run python notebooks/10_acid_floor_sensitivity.py [config_name]
      config_name defaults to ndpr_plant.
Prints the table; writes nothing.
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.circuit import Circuit, build_map                       # noqa: E402
from src.config import load_config, nominal_inputs               # noqa: E402
from src.design import optimize_operating_point, reflux_link     # noqa: E402

# ---- 1. the run file and the design settings ----
# SPEC_DESIGN is the design purity; REC the run file's recovery target. FREE: the handles the
# optimizer may move, with bounds (flows in L/h, scrub acid in mol/L, saponified fraction).
rc = load_config(sys.argv[1] if len(sys.argv) > 1 else "ndpr_plant")
p0 = rc.params
SPEC_DESIGN, REC = 0.995, rc.targets()[1]
FREE = {"Q_org": (600.0, 4000.0), "Q_scrub": (50.0, 2000.0), "h_scrub": (0.05, 2.0),
        "saponified_fraction": (0.0, 0.6)}
c = rc["circuit"]

# ---- 2. the design step at each floor ----
# For each floor: a copy of the parameters with the new floor, the circuit built on it, then the two
# optimizations. h is the aqueous acid in every stage at the answer, mol/L; its minimum is the last
# column. Note: the O/A column divides the organic flow by 600 L/h, the feed flow of the run files.
print(f"{'floor M':>8s} {'recovery':>9s} {'purity':>7s} {'O/A':>5s} {'Q_scrub':>8s} {'h_scrub':>8s} "
      f"{'sapon':>6s} {'reagent':>8s} {'min acid':>9s}")
for floor in (3e-4, 1e-3, 3e-3):
    p = replace(p0, h_min_D=floor)
    circ = Circuit(build_map(c["n_ext"], c["n_scrub"], c["n_strip"]), p, scrub_reflux=True)
    u0 = nominal_inputs(rc, circ)
    t0 = time.time()
    r1 = optimize_operating_point(circ, u0, rc.cut, SPEC_DESIGN, FREE, link=reflux_link(300.0))
    r2 = optimize_operating_point(circ, r1.u, rc.cut, SPEC_DESIGN, FREE, link=reflux_link(300.0), x_guess=r1.x,
                                  objective="reagent", min_recovery=REC)
    r = r2 if r2.success else r1
    h = np.array([float(v) for v in circ.unpack(r.x)[2]])
    print(f"{floor:8.0e} {r.recovery:9.4f} {r.purity:7.4f} {r.u.Q_org/600:5.2f} {r.u.Q_scrub:8.0f} {r.u.h_scrub:8.3f} "
          f"{r.u.saponified_fraction:6.3f} {r.reagent_mol_per_h:8.0f} {h.min():9.2e}   ({r.status}, {time.time()-t0:.0f}s)",
          flush=True)
