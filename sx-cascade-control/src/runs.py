"""A run, from its description to its saved result.

This file is the one way into a simulation. The command line (cli.py), the matrix of studies
and the workbench GUI all come through here, so they all run the same thing the same way
(ARCHITECTURE.md, decision 24).

A run goes through four stages, and the file is laid out in the same order:

1. The description. A `RunSpec` names everything a run depends on: the run file, the plant,
   the controller, the upset, the sensor set, how many hours, and any edits to the run file,
   the parameter file or the controller's settings. Because nothing else feeds into a run, a
   saved run says exactly how to make it again.

2. The operating point. Every run starts from the plant commissioned and trimmed onto spec
   (operating_point.py). That takes a while, so it's made once and cached, in memory and on
   disk, keyed by the inputs and the source code.

3. The session. `start(spec)` builds the plant, the controller, the upset and the instruments
   and hands back a `RunSession`. Its `steps()` is a generator: it advances one control
   interval at a time and yields a `LiveStep`, so a front end can draw while the run goes.
   `execute(spec)` does the same thing all at once.

4. The result. `session.result()` packs what has run into a `RunResult` (time series, the
   commanded handles, a scorecard and the settings needed to read it). `save()` writes it to
   a folder under runs/saved/, and `load()` reads it back.

Usage:

    spec = RunSpec(config="ndpr_plant", plant="perturbed", controller="mpc",
                   upset="feed_switch", sensors="full_state", hours=40)
    result = execute(spec)                       # all at once
    session = start(spec)                        # or step by step (a front end draws as it goes)
    for step in session.steps(): ...
    result = session.result()
    save(result); load(folder)

Edits to the run file, the parameter file or the controller's settings travel as nested dicts
in the shape of the thing they edit (`run_file`, `parameters`, `controller_settings`). For
instance `run_file={"flows": {"feed": 650}}` changes the feed flow in the run file.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from .circuit import Inputs
from .config import UNUSED_SECTIONS, deep_update, load_config
from .estimator import Estimator
from .loop import ClosedLoop, StepRecord
from .metrics import interface_window, point_outputs, stage_profile, transition_cost
from .report import run_report
from .operating_point import OperatingPoint, controller_settings, prepare
from .params import load_params, params_from_dict
from .registry import CONTROLLERS, PLANTS, SENSOR_SETS
from .scenarios import catalogue
from .sensors import SensorSet

# The project folder, and the two folders runs write to. runs/saved holds finished runs (one
# folder each); runs/cache holds operating points, so commissioning isn't repeated.
HERE = Path(__file__).resolve().parent.parent
SAVED_DIR = HERE / "runs" / "saved"
CACHE_DIR = HERE / "runs" / "cache"


# ==============================================================================================
# 1. The description of a run
# ==============================================================================================
# A RunSpec is plain data: strings, numbers and nested dicts. That's on purpose. It can be
# written to JSON, shown in a settings tree and read back, and two runs that differ only in
# what their specs say are a fair comparison (ARCHITECTURE.md, decision 7).

# The five choices a run is made of, in the order the workbench shows them. Each is a field of
# RunSpec, so getattr(spec, choice) is the current one.
CHOICES = ("config", "plant", "controller", "upset", "sensors")


@dataclass
class RunSpec:
    """Everything a run depends on.

    Attributes:
        config: the name of a run file in configs/ (without .toml).
        plant: a key in registry.PLANTS: matched, perturbed, settler or entrainment.
        controller: a key in registry.CONTROLLERS: open_loop, mpc_fixed or mpc.
        upset: a key in scenarios.catalogue, such as feed_switch.
        sensors: a key in registry.SENSOR_SETS; full_state means the controller sees the truth.
        hours: how long the run lasts, in hours of plant time.
        sensor_seed: the seed for the instruments' noise.
        run_file: edits to the run file, as a nested dict.
        parameters: edits to params/parameters.toml, as a nested dict.
        controller_settings: MPCSettings fields to change, by name.
    """
    config: str = "ndpr_main"                 # a run file in configs/ (the main case, decision 34)
    plant: str = "perturbed"                  # registry.PLANTS
    controller: str = "mpc_level_loops"       # registry.CONTROLLERS (the main case runs level loops under the MPC)
    upset: str = "feed_switch"                # scenarios.catalogue
    sensors: str = "full_state"               # registry.SENSOR_SETS
    hours: float = 40.0                       # h of plant time
    sensor_seed: int = 1                      # seed for the instruments' noise
    run_file: dict = field(default_factory=dict)              # edits to the run file
    parameters: dict = field(default_factory=dict)            # edits to params/parameters.toml
    controller_settings: dict = field(default_factory=dict)   # MPCSettings fields

    def label(self) -> str:
        """A one-line name for the run, for printouts and window titles.

        Example: "ndpr_plant | perturbed | mpc | feed_switch | full_state | 40 h"
        """
        return (f"{self.config} | {self.plant} | {self.controller} | {self.upset} | {self.sensors}"
                f" | {self.hours:g} h")


# ---- what a run is made of, as plain values (what a front end shows and edits) -----------------
# The run file and the parameter file are read from disk, and the spec's edits are written in
# on top. Nothing here simulates anything, so it's quick.

def run_config(spec: RunSpec):
    """The run file for this spec, with the spec's edits to it and to the parameter file applied.

    Args:
        spec: the run's description.

    Returns:
        A config.RunConfig: the checked run file, with its Params attached.
    """
    params = params_from_dict(deep_update(load_params().raw, spec.parameters))
    return load_config(spec.config, params=params, overrides=spec.run_file)


def editable_values(spec: RunSpec) -> dict:
    """Every setting of the run, as nested plain values.

    This is what the workbench's settings tree shows: the run file (after edits), the parameter
    file (after edits) and the controller's settings. The three top-level keys are the same as
    the RunSpec's edit fields, so an edit made in the tree goes straight back into the spec.

    Args:
        spec: the run's description.

    Returns:
        {"run_file": {...}, "parameters": {...}, "controller_settings": {...}}, JSON-safe.
    """
    rc = run_config(spec)
    settings = controller_settings(rc, **spec.controller_settings)
    # the sections nothing reads any more are left out, so the tree shows only what matters
    run_file = {k: v for k, v in rc.cfg.items() if k not in UNUSED_SECTIONS}
    return {"run_file": json_safe(run_file), "parameters": json_safe(rc.params.raw),
            "controller_settings": json_safe(dataclasses.asdict(settings))}


# ==============================================================================================
# 2. The operating point, cached
# ==============================================================================================
# Every run starts from the plant commissioned on its design handles and then trimmed onto spec
# by the controller (operating_point.prepare, decision 21). That takes a while, and many runs
# share it: a matrix of four controllers on one plant needs it once. So it's kept in two places:
#
#   - in memory, in _OPERATING_POINTS, for the rest of this Python session;
#   - on disk, in runs/cache/op_<key>.npz, for the next session.
#
# The key is a hash of everything the operating point could depend on: every source file in
# src/, every parameter file in params/, and the parts of the spec that pick the plant and the
# run file. Change any of them and the key changes, so a stale operating point is never used.
# The controller, the upset and the sensors aren't in the key: the operating point doesn't
# depend on them.

_OPERATING_POINTS: dict[str, OperatingPoint] = {}       # key -> operating point, this session


def operating_point(spec: RunSpec) -> OperatingPoint:
    """The plant commissioned and trimmed at the spec's run file, parameters and plant.

    Looked up in memory first, then on disk; made (and saved) only if neither has it.

    Args:
        spec: the run's description. Only config, plant, run_file and parameters matter here.

    Returns:
        An OperatingPoint: the model, the plant, the plant at rest on spec, and the handles.
    """
    key = _operating_point_key(spec)
    if key in _OPERATING_POINTS:
        return _OPERATING_POINTS[key]

    # Not in memory. Build the plant (fast), then try the disk cache before commissioning.
    rc = run_config(spec)
    plant, plant_k = PLANTS[spec.plant].build(rc)
    op = _load_cached(CACHE_DIR / f"op_{key}.npz", rc, plant, plant_k)
    if op is None:
        op = prepare(rc, plant, plant_k, trim_hours=float(rc["commissioning"]["trim_hours"]))   # the slow part
        _save_cached(CACHE_DIR / f"op_{key}.npz", op)
    _OPERATING_POINTS[key] = op
    return op


def _operating_point_key(spec: RunSpec) -> str:
    """A 16-character hash of the source code, the parameter files and the spec's plant inputs.

    Any change to a file in src/ or params/, or to the run file itself, gives a new key, even a
    comment. That costs a fresh commissioning now and then, and it means an old cache entry can't
    be picked up by mistake.
    """
    code = b"".join(p.read_bytes() for p in sorted((HERE / "src").glob("*.py")))
    run_file = HERE / "configs" / f"{spec.config}.toml"
    code += run_file.read_bytes() if run_file.exists() else str(spec.config).encode()
    inputs = json.dumps([spec.config, spec.plant, spec.run_file, spec.parameters], sort_keys=True, default=str)
    files = b"".join(p.read_bytes() for p in sorted((HERE / "params").glob("*.toml")))
    return hashlib.sha1(code + files + inputs.encode()).hexdigest()[:16]


def _save_cached(path: Path, op: OperatingPoint):
    """Write the four arrays that can't be rebuilt quickly: the two rest states and the two
    sets of inputs (as vectors). The rest of an OperatingPoint is rebuilt from the run file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x_plant=op.x_plant, u_model=op.u_model.numpy(), x_model=op.x_model,
             u_design=op.u_design.numpy())


def _load_cached(path: Path, rc, plant, plant_k) -> OperatingPoint | None:
    """Rebuild an OperatingPoint from a cache file, or return None if there isn't one.

    Args:
        path: the cache file, runs/cache/op_<key>.npz.
        rc: the run config (the model circuit is rebuilt from it).
        plant: the plant circuit, already built from the registry.
        plant_k: the plant's stage-efficiency multipliers, one per stage.
    """
    if not path.exists():
        return None
    from .config import build_circuit
    saved = np.load(path)
    model = build_circuit(rc)
    # The inputs were saved as flat vectors; turn them back into Inputs for this circuit's size.
    as_inputs = lambda v: Inputs.from_vector(v, model.n, model.N)
    return OperatingPoint(rc, model, plant, np.asarray(plant_k, float), saved["x_plant"],
                          as_inputs(saved["u_model"]), saved["x_model"], as_inputs(saved["u_design"]))


# ==============================================================================================
# 3. A run, step by step
# ==============================================================================================
# A session is a closed loop (loop.ClosedLoop: plant, controller, upset, instruments) plus the
# list of what has happened so far. steps() is a generator, so whoever calls it decides the
# pace: the command line just runs it to the end; the GUI draws after every step and can stop
# it partway.

@dataclass
class LiveStep:
    """One control interval as a front end needs it: when, what the plant did, what was commanded."""
    t: float               # h, the end of the interval
    outputs: dict          # metrics.point_outputs at the end of the interval
    handles: dict          # what the controller commanded (scalars; jacklegs as an array)
    info: dict             # the controller's record (QP status, bias, plan ...)
    profile: dict          # metrics.stage_profile of the plant at the end of the interval


def circuit_layout(spec: RunSpec) -> dict:
    """What a drawing of the circuit needs. Needs no simulation.

    Args:
        spec: the run's description (the run file and the sensor set are what matter).

    Returns:
        A dict with: stage counts per section ("extraction", "scrub", "strip") and in all
        ("stages"); where the feed enters, where the product and the raffinate leave (stage
        numbers, counted from 0 at the raffinate end); whether the scrub is a reflux; the
        component names and the cut; and the sensor set as a list of {"kind", "stage"}
        ("full_state" is True when there are no instruments and the controller sees the truth).
    """
    from .config import build_circuit
    rc = run_config(spec)
    circuit = build_circuit(rc)
    sensors = SENSOR_SETS[spec.sensors].build(circuit, rc)        # None means full state
    stage_counts = rc["circuit"]
    return {"extraction": stage_counts["n_ext"], "scrub": stage_counts["n_scrub"],
            "strip": stage_counts["n_strip"], "stages": circuit.N,
            "feed_stage": stage_counts["n_ext"] - 1, "product_stage": circuit.cmap.product_stage,
            "raffinate_stage": 0,
            "reflux": rc.reflux, "components": list(rc.params.names), "cut": rc.cut,
            "sensors": [] if sensors is None else [{"kind": s.kind, "stage": s.stage} for s in sensors],
            "full_state": sensors is None}


class RunSession:
    """One run in progress.

    Building a session does all the setup: it gets the operating point (cached), picks the
    upset from the catalogue, builds the controller and the instruments, and puts them in a
    closed loop. Nothing has run yet; call steps() to advance.

    Attributes:
        spec: the run's description.
        op: the operating point the run starts from.
        upset: the scenarios.UpsetEntry picked by spec.upset.
        controller: the controller, built from registry.CONTROLLERS.
        loop: the loop.ClosedLoop that does the stepping.
        stopped: set by stop(); steps() ends after the step it's on.

    Example:
        session = RunSession(RunSpec(hours=10))
        for live in session.steps():
            print(live.t, live.outputs["purity"])
        result = session.result()
    """

    def __init__(self, spec: RunSpec):
        """Set the run up from its spec (see the class docstring). Commissions the plant if no
        cached operating point fits the spec, which takes a while."""
        self.spec = spec
        self.op = operating_point(spec)
        rc = self.op.rc
        # The upset is set relative to the operating point's handles (the scrub-acid upset is).
        self.upset = catalogue(rc, self.op.u_model)[spec.upset]
        self.controller = CONTROLLERS[spec.controller].build(self.op, **spec.controller_settings)
        sensors, estimator = self._instruments()
        self.loop = ClosedLoop(self.op.plant, self.op.x_plant, self.controller, self.upset.scenario(),
                               sensors, estimator, self.op.plant_k)
        self._records: list[StepRecord] = []           # one loop.StepRecord per interval run so far
        self.stopped = False

    def _instruments(self):
        """The sensor set and the estimator that reads it, or (None, None) for the full state.

        The estimator starts from the model's own rest point and works on the controller's
        interval (decision 8: with a sensor set the controller only sees readings).
        """
        sensor_list = SENSOR_SETS[self.spec.sensors].build(self.op.model, self.op.rc)
        if sensor_list is None:
            return None, None
        sensor_set = SensorSet(sensor_list, self.op.plant, seed=self.spec.sensor_seed)
        estimator = Estimator(self.op.model, sensor_set, self.op.x_model, self.op.u_model, self.controller.t_step)
        return sensor_set, estimator

    def start_point(self) -> LiveStep:
        """The operating point, before the first interval: what a live plot draws at t = 0."""
        # The inputs at t = 0: the controller's nominal inputs with the scenario applied at
        # t = 0, and the plant's own stage efficiencies written in.
        u_start = self.loop.u_of_t(0.0).replace(k_factor=self.op.plant_k).numpy()
        x_start = self.op.x_plant
        return LiveStep(0.0, point_outputs(self.op.plant, x_start, u_start, self.op.rc.cut), self._start_handles(),
                        {}, stage_profile(self.op.plant, x_start))

    def _start_handles(self) -> dict:
        """The handles at the operating point, in the same shape the controller reports them."""
        names = ("Q_org", "Q_scrub", "h_scrub", "saponified_fraction", "H_jackleg")
        return {h: np.asarray(getattr(self.op.u_model, h), float) if h == "H_jackleg"
                else float(getattr(self.op.u_model, h)) for h in names}

    def steps(self) -> Iterator[LiveStep]:
        """Advance to the spec's end, one control interval at a time.

        Yields:
            A LiveStep after every interval. stop() ends the run after the current one.
        """
        for record in self.loop.steps(self.spec.hours):
            self._records.append(record)
            outputs = point_outputs(self.op.plant, record.x, record.u, self.op.rc.cut)
            yield LiveStep(record.t, outputs, record.handles, record.info, stage_profile(self.op.plant, record.x))
            if self.stopped:
                return

    def stop(self):
        """Ask steps() to end after the interval it's on. What has run is kept."""
        self.stopped = True

    def result(self) -> "RunResult":
        """Whatever has run so far (all of it, if steps() was run to the end) as a RunResult."""
        return _result_from(self.spec, self.op, self._records, self.loop.t_step, self.upset.title)


def start(spec: RunSpec) -> RunSession:
    """Set up a run and return the session, ready to step. See RunSession."""
    return RunSession(spec)


def execute(spec: RunSpec) -> "RunResult":
    """Run a spec to its end and return the result. The same as stepping a session through."""
    session = start(spec)
    for _ in session.steps():
        pass
    return session.result()


# ==============================================================================================
# 4. The result
# ==============================================================================================
# A result holds the arrays (time, states, inputs, output series, handles), the scorecard and
# a "meta" dict with what is needed to read it later without rerunning anything: the cut, the
# spec, the interface window, the section sizes, the titles of what was chosen, and the
# operating point.

@dataclass
class RunResult:
    """A finished (or stopped) run."""
    spec: RunSpec
    t: np.ndarray                  # h, the start and the end of every interval
    series: dict                   # point outputs over t; "interfaces" is (len(t), N)
    handles: dict                  # commanded handles at the start of every interval (len(t) - 1)
    info: list                     # the controller's record per interval
    scorecard: dict                # metrics.transition_cost (without the series) + solver failures
    meta: dict                     # what is needed to read it: cut, spec, window, sections, titles, settings
    x: np.ndarray                  # plant state over t
    u: np.ndarray                  # plant inputs over t (the last repeated)


def _result_from(spec, op: OperatingPoint, records: list, t_step: float, upset_title: str) -> RunResult:
    """Pack the step records into a RunResult.

    Works on a partial run too: with no records at all the result is the operating point alone,
    with an empty scorecard (apart from the solver failures, 0).

    Args:
        spec: the run's description.
        op: the operating point the run started from.
        records: one loop.StepRecord per interval run.
        t_step: the control interval, h.
        upset_title: the upset's title, for the meta.
    """
    traj = _trajectory_from(op, records)
    series = _series_from(op, traj)
    scorecard = _scorecard_from(op, traj, records)
    return RunResult(spec, traj.t, series, _handle_series(records), [r.info for r in records], scorecard,
                     _meta(spec, op, upset_title, t_step), traj.x, traj.u)


def _trajectory_from(op: OperatingPoint, records: list):
    """The run as a simulate.Trajectory: times, plant states and plant inputs, one row per time.

    The first row is the operating point at t = 0. Inputs are applied over an interval, so there
    is one fewer of them than of times; the last one is repeated to make the rows line up.
    """
    from .simulate import Trajectory
    states = np.array([op.x_plant] + [r.x for r in records])
    inputs = np.array([r.u for r in records] + [_last_inputs(op, records)])
    return Trajectory(np.array([0.0] + [r.t for r in records]), states, inputs)


def _last_inputs(op: OperatingPoint, records: list) -> np.ndarray:
    """The input row repeated at the end: the last interval's, or the operating point's if
    nothing has run yet."""
    return records[-1].u if records else op.u_plant.numpy()


def _series_from(op: OperatingPoint, traj) -> dict:
    """metrics.point_outputs at every time, turned into one array per output (purity, ...)."""
    points = [point_outputs(op.plant, xk, uk, op.rc.cut) for xk, uk in zip(traj.x, traj.u)]
    return {k: np.array([p[k] for p in points]) for k in points[0]}


def _scorecard_from(op: OperatingPoint, traj, records: list) -> dict:
    """The transition cost of the run (mol, m*h, h; see metrics.transition_cost), plus how many
    intervals the controller's QP failed. The purity series is dropped: it's in the series."""
    cost = transition_cost(op.plant, traj, op.rc.cut, op.spec) if len(records) else {}
    cost.pop("purity_series", None)
    cost["solver_failures"] = _solver_failures(records)
    return cost


def _solver_failures(records: list) -> int:
    """How many intervals the controller reported its QP as not solved (info["qp_ok"] False)."""
    return sum(1 for r in records if r.info and not r.info.get("qp_ok", True))


def _handle_series(records: list) -> dict:
    """The commanded handles, one array per handle name, one row per interval."""
    names = records[0].handles.keys() if records else ()
    return {k: np.array([np.asarray(r.handles[k], float) for r in records]) for k in names}


def _meta(spec: RunSpec, op: OperatingPoint, upset_title: str, t_step: float) -> dict:
    """What a reader of the result needs besides the arrays: the cut, the spec and targets, the
    interface window (m), the section sizes, the titles of what was chosen, the control interval
    (h), and the handles at the operating point and at the design point before the trim."""
    rc = op.rc
    lo, hi = interface_window(rc.params)
    stage_counts = rc["circuit"]
    return {"cut": rc.cut, "spec": op.spec, "recovery_target": op.recovery_target,
            "design_purity": rc.design_purity, "interface_window": [lo, hi],
            "sections": {"extraction": stage_counts["n_ext"], "scrub": stage_counts["n_scrub"],
                         "strip": stage_counts["n_strip"]},
            "upset_title": upset_title, "t_step": t_step,
            "plant_title": PLANTS[spec.plant].title, "controller_title": CONTROLLERS[spec.controller].title,
            "sensors_title": SENSOR_SETS[spec.sensors].title,
            "operating_point": {h: float(getattr(op.u_model, h)) for h in
                                ("Q_org", "Q_feed", "Q_scrub", "Q_strip", "h_scrub", "saponified_fraction")},
            "design_point": {h: float(getattr(op.u_design, h))
                             for h in ("Q_org", "Q_scrub", "h_scrub", "saponified_fraction")}}


# ==============================================================================================
# Saving and loading
# ==============================================================================================
# A saved run is a folder with two files:
#   run.json    the spec, the scorecard, the meta and the controller's record per interval
#   arrays.npz  t, x, u, and the series and handles (stored as series__<name>, handles__<name>)
# The spec is in run.json in full, so the run can be made again from the folder alone.

def save(result: RunResult, folder: Path | None = None) -> Path:
    """Write a result to a folder and return the folder: run.json (the spec, scorecard, meta and the
    controller's record), arrays.npz (every time series) and report.md (the run in words, report.py).

    Args:
        result: the run to save.
        folder: where to write it. By default a new folder under runs/saved/, named
            <date-time>_<config>_<plant>_<controller>_<upset>_<sensors>.

    Returns:
        The folder written.
    """
    s = result.spec
    name = f"{time.strftime('%Y%m%d-%H%M%S')}_{s.config}_{s.plant}_{s.controller}_{s.upset}_{s.sensors}"
    folder = folder or SAVED_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    doc = {"spec": dataclasses.asdict(s), "scorecard": result.scorecard, "meta": result.meta, "info": result.info}
    (folder / "run.json").write_text(json.dumps(json_safe(doc), indent=1), encoding="utf-8")
    arrays = {"t": result.t, "x": result.x, "u": result.u}
    arrays.update({f"series__{k}": v for k, v in result.series.items()})
    arrays.update({f"handles__{k}": v for k, v in result.handles.items()})
    np.savez_compressed(folder / "arrays.npz", **arrays)
    (folder / "report.md").write_text(run_report(result), encoding="utf-8")
    return folder


def load(folder: Path) -> RunResult:
    """Read a folder written by save() back into a RunResult."""
    doc = json.loads((Path(folder) / "run.json").read_text(encoding="utf-8"))
    arrays = np.load(Path(folder) / "arrays.npz")
    # every array whose name starts with the prefix, with the prefix taken off
    pick = lambda prefix: {k[len(prefix):]: arrays[k] for k in arrays.files if k.startswith(prefix)}
    return RunResult(RunSpec(**doc["spec"]), arrays["t"], pick("series__"), pick("handles__"), doc["info"],
                     doc["scorecard"], doc["meta"], arrays["x"], arrays["u"])


def saved_runs() -> list[Path]:
    """The run.json of every saved run, newest first (the folder names start with the time)."""
    return sorted(SAVED_DIR.glob("*/run.json"), reverse=True) if SAVED_DIR.exists() else []


def json_safe(value):
    """The value as plain JSON types: numpy numbers and arrays to Python, tuples to lists,
    NaN and infinity to None. Works through nested dicts and lists; dict keys become strings.

    Example: json_safe({"a": np.array([1.0, np.nan])}) == {"a": [1.0, None]}
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return _json_safe_scalar(value)


def _json_safe_scalar(value):
    """One number as a Python number (None for NaN or infinity); anything else as it is."""
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
