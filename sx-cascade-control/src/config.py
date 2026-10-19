"""Read a run file (configs/*.toml), check it, fill defaults, and build the
circuit and its nominal inputs from it. One place that knows every option.

What this file does
-------------------
A run file describes a circuit: how many stages in each section, the cut,
the feed, the flows, the scrub and strip liquors, saponification, the
targets, and which plant to drive (ARCHITECTURE.md, decision 7). Chemistry
and geometry don't live there; they come from params/parameters.toml.

Reading a run file goes in four steps:

  1. read     the TOML file, by path or by its name in configs/ (read_run_file)
  2. edit     any overrides are written in, nested dicts merged key by key (deep_update)
  3. default  every key the file leaves out takes its value from DEFAULTS, and a
              key DEFAULTS doesn't know stops the run with KeyError (_merge)
  4. check    every option in BUILT must be one that exists today; anything else
              stops the run with NotImplementedError naming the option, never a
              silent fallback (decision 7's rule)

The result is a RunConfig. From it, build_circuit makes the circuit model,
nominal_inputs makes the steady inputs the run file asks for, and build_plant
makes the plant the controller drives.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .circuit import Circuit, CircuitMap, Inputs, build_map
from .params import Params, feed_concentrations, load_params, params_from_dict

# The folder of run files. A run file can be named by its stem, e.g. "ndpr_plant".
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


# ---------------------------------------------------------------------------
# Every option, with its default
# ---------------------------------------------------------------------------
# Units: flows in L/h, acid in mol/L, feed strength in g/L of rare-earth metal, times in hours.

DEFAULTS = {
    "circuit": {"n_ext": 2, "n_scrub": 12, "n_strip": 4, "organic_recycle": True,
                "model": "fast", "cut": "Pr"},
    "feed": {"deposit": "eneabba", "total_RE_g_per_L": 50.0, "acid": 0.05,
             "removed_upstream": ["LaCe", "SmPlus"]},
    "flows": {"organic": 1200.0, "feed": 600.0, "scrub": 300.0, "strip": 300.0},
    "scrub": {"acid": 0.5, "metal": "none"},     # metal: "none" (acid only) or "reflux" (strip product sent back)
    "strip": {"acid": 2.0},
    "saponification": {"mode": "simple", "fraction": 0.4},
    "sim": {"t_end": 60.0, "dt_out": 0.1},
    "disturbances": [],
    "sensors": [],
    "actuators": [],
    "controller": {"type": "none"},
    "targets": {"purity": None, "recovery": None},   # None: use params/parameters.toml [targets]
    # How long the controller trims the plant onto spec after start-up (operating_point.prepare). The
    # stage-sized plant is ~3.5x slower than the original one and needs longer.
    "commissioning": {"trim_hours": 30.0},
    # The plant the notebooks drive (build_plant): "none" = the model itself; "perturbed" = src/mismatch.py;
    # "settler", "entrainment" = src/plant_models.py. Runs through runs.py (CLI, workbench, matrix) take the
    # plant from RunSpec.plant instead, so for them only seed, strength and settler_cells matter here.
    "plant": {"mismatch": "none", "seed": 1, "strength": 1.0, "settler_cells": 3, "generator": "independent",
              "drift_split": 0.03, "drift_trend": 0.0},     # drift_split / drift_trend: the drifting plant's La+Ce-against-the-rest amplitude, and slide per week
}
# The defaults above are the Pr | Nd validation case (cut "Pr", La+Ce and Sm+ removed upstream). Every run
# file sets its own cut and feed; the main case, the NdPr product, is cut "LaCe" (configs/ndpr_*.toml).
# "sim", "disturbances", "sensors", "actuators" and "controller" are accepted so older run files load, but
# nothing reads them: upsets, sensors and controllers are chosen per run (runs.RunSpec, registry.py).

# Options that exist today, as "section.key": {allowed values}. Anything else in these keys raises.
BUILT = {
    "circuit.model": {"fast"},
    "scrub.metal": {"none", "reflux"},
    "saponification.mode": {"simple"},
    "controller.type": {"none"},
    "plant.mismatch": {"none", "perturbed", "settler", "entrainment"},
    "plant.generator": {"independent", "ratios"},
}


# ---------------------------------------------------------------------------
# A checked run file
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """A run file, checked and filled with defaults, plus the parameter set it runs with.

    name     the run file's name (its `name` key, else the file's stem)
    cfg      the full settings, every key of DEFAULTS present
    params   the chemistry and geometry (params/parameters.toml)

    rc["flows"]["feed"] reads a setting, the same as rc.cfg["flows"]["feed"].
    """
    name: str
    cfg: dict
    params: Params

    def __getitem__(self, key):
        """One section of the settings, e.g. rc["flows"]."""
        return self.cfg[key]

    @property
    def cut(self) -> str:
        """The last component meant for the raffinate: "LaCe" for the NdPr product, "Pr" for validation."""
        return self.cfg["circuit"]["cut"]

    def targets(self) -> tuple[float, float]:
        """(purity spec, recovery target) for this run: the run file's, else the project's."""
        t, tp = self.cfg["targets"], self.params.raw["targets"]
        return (t["purity"] or tp["product_purity_fraction"], t["recovery"] or tp["recovery_fraction"])

    @property
    def design_purity(self) -> float:
        """The purity the design point and the controller aim for (above spec, as plants run)."""
        return float(self.params.raw["targets"]["design_purity_fraction"])

    @property
    def design(self) -> dict:
        """The parameter file's [design] rules: net product flow, pump ceiling and headroom."""
        return self.params.raw["design"]

    @property
    def reflux(self) -> bool:
        """True when the scrub liquor is strip product sent back (scrub.metal = "reflux")."""
        return self.cfg["scrub"]["metal"] == "reflux"


# ---------------------------------------------------------------------------
# Reading, editing, defaulting and checking
# ---------------------------------------------------------------------------

# Run-file sections nothing reads any more. They stay in DEFAULTS so old run files still load, but
# they're left out of a run file saved from the workbench and out of its settings tree.
UNUSED_SECTIONS = ("sim", "disturbances", "sensors", "actuators", "controller")


def _merge(defaults: dict, given: dict, prefix: str = "") -> dict:
    """`given` with every missing key filled from `defaults`, nested sections merged key by key.

    Raises KeyError naming any key in `given` that `defaults` doesn't have (a typo in a
    run file stops the run instead of being ignored), and TypeError naming any value whose type
    isn't the default's (see _check_type).
    """
    unknown = set(given) - set(defaults)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return {k: _merged_value(k, v, given, prefix) for k, v in defaults.items()}


def _merged_value(key: str, default, given: dict, prefix: str):
    """One key of _merge: a nested section merged, or the given value (checked) or the default."""
    if isinstance(default, dict):
        return _merge(default, given.get(key, {}), f"{prefix}{key}.")
    if key not in given:
        return default
    _check_type(f"{prefix}{key}", default, given[key])
    return given[key]


# Settings whose default is empty (None: "use the parameter file's"), with an example of the type
# they must have when a run file sets them.
EMPTY_DEFAULT_TYPES = {"targets.purity": 0.99, "targets.recovery": 0.95}

# What a value must be, by the type of its default. Order matters: a bool is an int in Python, so
# bool comes first, and an int is fine where a decimal is expected.
_TYPE_RULES = (
    (bool, lambda v: isinstance(v, bool)),
    (int, lambda v: isinstance(v, int) and not isinstance(v, bool)),
    (float, lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)),
    ((list, tuple), lambda v: isinstance(v, (list, tuple))),
    (str, lambda v: isinstance(v, str)),
)


def _check_type(name: str, default, value):
    """A run-file value must have its default's type: a number where the default is a number (a
    whole number is fine for a decimal), true/false for a yes/no, text for text, a list for a list.
    Settings whose default is empty (None) take any type unless EMPTY_DEFAULT_TYPES names one.
    Raises TypeError naming the setting.
    Why: text such as "0.99" got into a run as the purity spec through the workbench (Oct 18) and
    only failed deep inside the controller; a wrong type now stops the run at loading, by name."""
    default = EMPTY_DEFAULT_TYPES.get(name) if default is None else default
    if value is None or default is None:
        return
    if not _type_rule(default)(value):
        raise TypeError(f"run file setting {name} = {value!r} is {type(value).__name__}; "
                        f"it should be {type(default).__name__} like its default {default!r}")


def _type_rule(default):
    """The check from _TYPE_RULES that fits the default's type; one that passes anything when no
    rule fits (a default of some other type isn't checked)."""
    return next((ok for kind, ok in _TYPE_RULES if isinstance(default, kind)), lambda v: True)


def deep_update(base: dict, changes: dict) -> dict:
    """A copy of `base` with `changes` written in, nested dicts merged key by key.

    Example
    -------
        deep_update({"flows": {"feed": 600.0, "scrub": 300.0}}, {"flows": {"feed": 650.0}})
        # {"flows": {"feed": 650.0, "scrub": 300.0}}
    """
    out = dict(base)
    for k, v in changes.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def read_run_file(path_or_name: str | Path) -> tuple[str, dict]:
    """(name, contents) of a run file, by path or by its name in configs/.

    If `path_or_name` isn't an existing file, it's taken as a name: configs/<name>.toml.
    The `name` key is taken out of the contents; without one, the file's stem is the name. The
    `description` key (what the run file is, in a few sentences) is taken out too; read it with
    run_file_description. Raises FileNotFoundError if neither exists.
    """
    path = _run_file_path(path_or_name)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    raw.pop("description", None)
    return raw.pop("name", path.stem), raw


def run_file_description(path_or_name: str | Path) -> str:
    """A run file's `description` (what it is, which study it belongs to, whether it's current),
    or "" if it has none."""
    with open(_run_file_path(path_or_name), "rb") as f:
        return str(tomllib.load(f).get("description", "")).strip()


def _run_file_path(path_or_name: str | Path) -> Path:
    """The file itself if it exists, otherwise configs/<name>.toml."""
    path = Path(path_or_name)
    return path if path.exists() else CONFIG_DIR / f"{path_or_name}.toml"


def load_config(path_or_name: str | Path, params: Params | None = None,
                overrides: dict | None = None) -> RunConfig:
    """A run file, checked and filled with defaults.

    Parameters
    ----------
    path_or_name   a path to a run file, or its name in configs/
    params         the parameter set; None loads params/parameters.toml
    overrides      a nested dict in the run file's own shape, e.g. {"flows": {"feed": 650.0}},
                   written in before checking

    A run file may carry a [parameter_edits] table: changes to params/parameters.toml for this
    run only, in the parameter file's own shape (e.g. [parameter_edits.stage_geometry]
    mixer_volume_L = 130.0). They're applied on top of `params`. This is how a variant plant,
    such as the one with stages sized to the design flows, gets its own run file without a
    second copy of the whole parameter file.

    A run file may also carry a [controller_settings] table: MPCSettings fields for this run file
    (e.g. acid_floor = 1e-3), applied by operating_point.controller_settings before any per-run
    edits. Field names are checked there.

    Returns
    -------
    RunConfig

    Raises
    ------
    KeyError             a key the run file has but DEFAULTS doesn't
    NotImplementedError  an option that is named but not built yet (see BUILT)

    Example
    -------
        rc = load_config("ndpr_plant", overrides={"flows": {"feed": 650.0}})
    """
    name, raw = read_run_file(path_or_name)
    raw = deep_update(raw, overrides or {})
    edits = raw.pop("parameter_edits", {})               # this run's changes to the parameter file
    settings = raw.pop("controller_settings", {})        # this run file's controller settings
    params = params or load_params()
    if edits:
        params = params_from_dict(deep_update(params.raw, edits))
    cfg = _merge(DEFAULTS, raw)
    _check_built(cfg)
    if settings:
        cfg["controller_settings"] = settings
    return RunConfig(name, cfg, params)


def _check_built(cfg: dict) -> None:
    """Stop with NotImplementedError if any option in BUILT is set to a value not built yet."""
    for dotted, allowed in BUILT.items():
        a, b = dotted.split(".")
        if cfg[a][b] not in allowed:
            raise NotImplementedError(
                f"{dotted} = {cfg[a][b]!r} is a setting but not built yet; built: {sorted(allowed)}")


# ---------------------------------------------------------------------------
# Building the circuit, the plant and the inputs
# ---------------------------------------------------------------------------

def build_circuit(rc: RunConfig) -> Circuit:
    """The circuit model the run file describes: its stage counts, organic recycle, model and scrub."""
    c = rc["circuit"]
    cmap: CircuitMap = build_map(c["n_ext"], c["n_scrub"], c["n_strip"], c["organic_recycle"])
    return Circuit(cmap, rc.params, model=c["model"], scrub_reflux=(rc["scrub"]["metal"] == "reflux"))


def build_plant(rc: RunConfig):
    """The plant the controller drives, as the run file's [plant] mismatch says.

    none (the model itself), perturbed (same equations, different numbers), settler or
    entrainment (mechanisms the model lacks). Returns (plant_circuit, per-stage
    transfer-rate multipliers). The plants themselves are in registry.py, where the
    run file's "none" is called "matched".
    """
    from .registry import PLANTS
    key = {"none": "matched", "perturbed": "perturbed", "settler": "settler",
           "entrainment": "entrainment"}[rc["plant"]["mismatch"]]
    return PLANTS[key].build(rc)


def nominal_inputs(rc: RunConfig, circuit: Circuit, H_jackleg: np.ndarray | None = None) -> Inputs:
    """The run file's steady inputs.

    Parameters
    ----------
    rc          the run file
    circuit     the circuit built from it (for the stage and component counts)
    H_jackleg   jackleg height per stage, m. None: every stage gets the height that puts
                the interface at its setpoint with empty phases; commission.commission()
                refines them.

    Returns
    -------
    Inputs: flows (L/h), feed composition (mol/L, from the deposit and its strength in g/L),
    acidities (mol/L), saponified fraction and jackleg heights (m). The scrub liquor carries
    no metal here (c_scrub = 0).
    """
    p = rc.params
    f = rc["feed"]
    c_feed = feed_concentrations(f["deposit"], f["total_RE_g_per_L"], p, tuple(f["removed_upstream"]))
    if H_jackleg is None:
        H_jackleg = _default_jacklegs(p, circuit)
    fl = rc["flows"]
    return Inputs(
        Q_org=fl["organic"], Q_feed=fl["feed"], Q_scrub=fl["scrub"], Q_strip=fl["strip"],
        c_feed=c_feed, h_feed=f["acid"],
        c_scrub=np.zeros(circuit.n), h_scrub=rc["scrub"]["acid"],
        h_strip=rc["strip"]["acid"],
        saponified_fraction=_saponified_fraction(rc),
        H_jackleg=np.asarray(H_jackleg, float),
    )


def _default_jacklegs(p: Params, circuit: Circuit) -> np.ndarray:
    """One jackleg height (m) for every stage: the one that puts the interface at its setpoint
    (interface_setpoint_fraction of the settler height) when both phases hold no metal."""
    geo = p.raw["stage_geometry"]
    z_set = geo["interface_setpoint_fraction"] * geo["settler_height_m"]
    no_metal_aq, no_metal_org = np.zeros(circuit.n), np.zeros(circuit.n)      # mol/L, every component
    return np.full(circuit.N, circuit.jackleg_for_interface(no_metal_aq, no_metal_org, z_set))


def _saponified_fraction(rc: RunConfig) -> float:
    """The saponified fraction for mode "simple"; 0 for any other mode."""
    sap = rc["saponification"]
    return sap["fraction"] if sap["mode"] == "simple" else 0.0
