"""Steady-state concentration profile down the train for one run file.

The question: once the circuit has settled on its designed handles, how much Pr
and Nd sits in each stage, in both liquids, how acid each stage is, and what
purity and recovery does the product come out at?

Method: build the circuit from the run file (configs/<name>.toml), solve its
steady state on the run file's flows and acids, set the jacklegs so every
interface sits at its setpoint (src/commission.py), and draw the profile.
This is the model itself, not the perturbed plant.

Run:  pixi run python notebooks/01_steady_profile.py [config_name]
      config_name defaults to ndpr_plant, the 32-stage NdPr circuit of the report.
Writes figures/01_profile_<name>.png and prints purity / recovery.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.commission import commission                     # noqa: E402
from src.config import build_circuit, load_config, nominal_inputs   # noqa: E402
from src.metrics import summarize_steady                  # noqa: E402
from src.plotting import FIGURES, profile_figure           # noqa: E402

# ---- 1. build the circuit from the run file ----
# rc is the run file, read and checked; circ is the circuit's equations for that many stages.
name = sys.argv[1] if len(sys.argv) > 1 else "ndpr_plant"
rc = load_config(name)
circ = build_circuit(rc)

# ---- 2. settle it on the design handles ----
# commission() solves the steady state on the run file's inputs and moves each jackleg so its
# interface sits at the setpoint. x is the full state (concentrations, acid, interfaces per stage),
# u the inputs it settled under.
x, u = commission(circ, nominal_inputs(rc, circ))

# ---- 3. score it ----
# summarize_steady gives product purity and recovery (fractions, in moles of metal), the
# raffinate's purity, and the total metal in the product and raffinate streams (mol/L).
s = summarize_steady(circ, x, u, rc.cut)
print(name, {k: round(v, 4) for k, v in s.items()})

# ---- 4. draw the profile ----
# One figure, three panels against stage number (0 at the raffinate end): each component the feed
# carries in the organic, then in the aqueous, and the aqueous acid on a log scale. For the NdPr
# circuit that is La+Ce (what the circuit removes) as well as Pr and Nd (the product).
c = rc["circuit"]
title = (f"{name}: {c['n_ext']} extraction + {c['n_scrub']} scrub + {c['n_strip']} strip stages, "
         f"purity {s['product_purity']:.1%}, recovery {s['recovery']:.1%}")
in_feed = tuple(comp for comp in circ.p.names if comp not in rc["feed"]["removed_upstream"])
profile_figure(circ, x, title, FIGURES / f"01_profile_{name}.png", components=in_feed)
print("wrote", FIGURES / f"01_profile_{name}.png")
