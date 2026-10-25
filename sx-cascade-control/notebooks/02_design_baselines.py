"""Generate run files from the design step, so their numbers are measured on the
model rather than typed.

The question: for a circuit of a given size, what flows, acid and saponification
should it run at? The answer becomes a run file in configs/, which every other
script and the workbench start from.

    pixi run python notebooks/02_design_baselines.py <base_config> <name>=<n_ext>,<n_scrub>,<n_strip>[,<min_recovery>] ...

Method, for each name: max recovery at the design purity (spec + margin), then minimize
reagent keeping recovery >= min_recovery (if given) or 90 % of the maximum (if
not). The base config supplies the cut, the feed and the reagents; the organic
flow is capped at 85 % of the pump maximum so a controller has headroom.
Both problems are solved by src/design.py (IPOPT on the steady-state model).
ARCHITECTURE.md decisions 15, 19 and 22 have the reasons.

Writes configs/<name>.toml for each name given, overwriting any that exist.

Examples:
    ... ndpr_sweep_base ndpr_plant=8,20,4,0.95 ndpr_small=4,8,4
    ... prnd_sweep_base prnd_plant=26,34,4,0.95 prnd_small=2,12,4
"""
import sys
import tomllib
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.circuit import Circuit, build_map                       # noqa: E402
from src.config import CONFIG_DIR, deep_update, load_config, nominal_inputs   # noqa: E402
from src.design import optimize_operating_point, reflux_link
from src.params import params_from_dict     # noqa: E402

# ---- 1. read the command line and the base run file ----
# base is the run file that sets the cut, the feed and the reagents (e.g. ndpr_sweep_base);
# specs are the "name=n_ext,n_scrub,n_strip[,min_recovery]" strings, one per circuit to design.
# --parameter-edits FILE (optional): a small TOML of changes to params/parameters.toml for this
# design, e.g. params/edits_sized_stages.toml. They're applied here and written into each run file
# as [parameter_edits], so loading the run file later gives the same plant.
args = list(sys.argv[1:])
EDITS_FILE = None
if "--parameter-edits" in args:
    i = args.index("--parameter-edits")
    EDITS_FILE = Path(args[i + 1])
    del args[i:i + 2]
base = args[0]
specs = args[1:]
rc = load_config(base)
p = rc.params
if EDITS_FILE is not None:
    p = params_from_dict(deep_update(p.raw, tomllib.loads(EDITS_FILE.read_text(encoding="utf-8"))))

# ---- 2. the design settings ----
# SPEC is the product purity the plant must meet (0.99); the design aims half a point above it.
# The free handles and their bounds, in FREE: organic and scrub flow in L/h, scrub acid in mol/L,
# and the saponified fraction of the extractant (0 to 0.6).
SPEC = p.raw["targets"]["product_purity_fraction"]
DESIGN_PURITY = 0.995                       # chosen: run half a point above spec, as plants run
NET_PRODUCT = 300.0                         # L/h of strip product that leaves the circuit (reflux mode)
PUMP_MAX = 4700.0                           # L/h, the organic pump's ceiling (chosen)
HEADROOM = 0.85                             # chosen: design at 85 % of it, so a controller can add organic
FREE = {"Q_org": (600.0, HEADROOM * PUMP_MAX), "Q_scrub": (50.0, 2000.0), "h_scrub": (0.05, 2.0),
        "saponified_fraction": (0.0, 0.6)}
reflux = rc["scrub"]["metal"] == "reflux"   # True when the scrub liquor is strip product sent back


# ---- 3. the design step, for one circuit size ----
def design(n_ext, n_scrub, n_strip, min_recovery):
    """Find the operating point of one circuit size, in two optimizations.

    n_ext, n_scrub, n_strip  stages in each section.
    min_recovery             the recovery floor for the second problem, a fraction;
                             None means 90 % of the best recovery found first.

    First problem: the most recovery possible at DESIGN_PURITY. Second: the least
    reagent (mol/h of caustic and acid) that keeps recovery at or above the floor,
    started from the first answer. Prints both.
    Returns (the second result, the recovery floor used). Stops the script if the
    second problem fails, which usually means the floor can't be reached.
    """
    cmap = build_map(n_ext, n_scrub, n_strip)
    circ = Circuit(cmap, p, scrub_reflux=reflux)
    # starting inputs: the base file's, with the strip flow set to the net product plus 300 L/h
    u0 = nominal_inputs(rc, circ).replace(Q_strip=NET_PRODUCT + 300.0)
    # in reflux mode the strip flow is tied to the scrub flow: strip = net product + scrub
    link = reflux_link(NET_PRODUCT) if reflux else None
    t0 = time.time()
    r1 = optimize_operating_point(circ, u0, rc.cut, DESIGN_PURITY, FREE, link=link)
    print(f"  max recovery: {r1.recovery:.4f} at purity {r1.purity:.4f} ({r1.status}, {time.time()-t0:.0f}s)")
    target = min_recovery if min_recovery is not None else round(0.9 * r1.recovery, 3)
    t0 = time.time()
    r2 = optimize_operating_point(circ, r1.u, rc.cut, DESIGN_PURITY, FREE, link=link, x_guess=r1.x,
                                  objective="reagent", min_recovery=target)
    print(f"  min reagent at recovery >= {target:.3f}: recovery {r2.recovery:.4f}, purity {r2.purity:.4f}, "
          f"reagent {r2.reagent_mol_per_h:.0f} mol/h ({r2.status}, {time.time()-t0:.0f}s)")
    if not r2.success:
        raise SystemExit(f"design failed for {n_ext},{n_scrub},{n_strip}: {r2.status} (is min_recovery feasible?)")
    return r2, target


# ---- 4. write the result as a run file ----
# The run file is a template filled in with the design's numbers. Its header comment records how it
# was made and what the model measured at that point, so nobody types those numbers by hand. The
# [plant] section names the perturbed plant (src/mismatch.py, seed 1), the one every result runs on.
def _edits_section() -> str:
    """The --parameter-edits file as a [parameter_edits] table for the run file ('' if none).
    Section headers [x] become [parameter_edits.x]; values and comments are copied as they are."""
    if EDITS_FILE is None:
        return ""
    lines = [f"\n# Parameter edits from {EDITS_FILE.as_posix()}, applied on top of params/parameters.toml"]
    for line in EDITS_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            line = f"[parameter_edits.{stripped[1:-1]}]"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _header(r, rec_target) -> str:
    """The run file's header comment: how it was made and what the model measured at the design point."""
    return f'''# Generated by notebooks/02_design_baselines.py from {base} on {time.strftime("%Y-%m-%d")}.
# Design step: minimize reagent subject to steady state, product purity >= {DESIGN_PURITY},
# recovery >= {rec_target:.3f}, input bounds; organic flow capped at {HEADROOM:.0%} of the {PUMP_MAX:.0f} L/h
# pump maximum so a controller has headroom (both chosen); organic loading in every stage at most
# {p.raw["extractant"]["max_loading_fraction"]:.0%} of its capacity (parameters.toml, chosen).
# Measured on the model at this point: purity {r.purity:.4f}, recovery {r.recovery:.4f}, reagent {r.reagent_mol_per_h:.0f} mol/h.
# Regenerate rather than hand-edit the numbers.
'''


def _circuit_and_feed(name, stages) -> str:
    """The name, [circuit] and [feed] sections. stages: (n_ext, n_scrub, n_strip)."""
    n_ext, n_scrub, n_strip = stages
    f = rc["feed"]
    removed = ", ".join(f'"{c}"' for c in f["removed_upstream"])
    return f'''name = "{name}"

[circuit]
n_ext = {n_ext}
n_scrub = {n_scrub}
n_strip = {n_strip}
organic_recycle = true
model = "fast"
cut = "{rc.cut}"

[feed]
deposit = "{f["deposit"]}"
total_RE_g_per_L = {f["total_RE_g_per_L"]}
acid = {f["acid"]}
removed_upstream = [{removed}]

'''


def _flows_and_acids(u) -> str:
    """The [flows], [scrub], [strip] and [saponification] sections, from the design's inputs u."""
    return f'''[flows]                         # L/h
organic = {u.Q_org:.1f}
feed = {u.Q_feed:.1f}
scrub = {u.Q_scrub:.1f}                  # reflux: strip product sent back as scrub liquor
strip = {u.Q_strip:.1f}                  # = {NET_PRODUCT:.0f} net product + scrub reflux

[scrub]
acid = {u.h_scrub:.4f}
metal = "{rc["scrub"]["metal"]}"

[strip]
acid = {u.h_strip}

[saponification]
mode = "simple"
fraction = {u.saponified_fraction:.4f}

'''


def _targets_plant_sim(rec_target) -> str:
    """The [targets], [plant] and [sim] sections."""
    return f'''[targets]
purity = {SPEC}                   # spec; the design point sits at {DESIGN_PURITY}
recovery = {rec_target:.3f}

[plant]
mismatch = "perturbed"          # the plant differs from the model: src/mismatch.py, seed 1
seed = 1
strength = 1.0

[sim]
t_end = 60.0
dt_out = 0.25
'''


def write(name, stages, r, rec_target):
    """Write configs/<name>.toml from one design result.

    name        the run file's name.
    stages      (n_ext, n_scrub, n_strip), stages in each section.
    r           the design result (its inputs u, purity, recovery, reagent).
    rec_target  the recovery floor the design used, written as the run's target.

    The feed, scrub metal and cut come from the base run file; flows, acids and
    saponification from r.
    """
    text = (_header(r, rec_target) + _circuit_and_feed(name, stages) + _flows_and_acids(r.u)
            + _targets_plant_sim(rec_target) + _edits_section())
    (CONFIG_DIR / f"{name}.toml").write_text(text, encoding="utf-8", newline="\n")
    print("  wrote", CONFIG_DIR / f"{name}.toml")


# ---- 5. design and write each circuit asked for ----
# Each spec is "name=n_ext,n_scrub,n_strip" with an optional fourth number, the recovery floor.
for spec in specs:
    name, sizes = spec.split("=")
    parts = sizes.split(",")
    n_ext, n_scrub, n_strip = (int(v) for v in parts[:3])
    min_rec = float(parts[3]) if len(parts) > 3 else None
    print(f"{name} ({n_ext} + {n_scrub} + {n_strip}) from {base}, cut {rc.cut}")
    r, tgt = design(n_ext, n_scrub, n_strip, min_rec)
    write(name, (n_ext, n_scrub, n_strip), r, tgt)
