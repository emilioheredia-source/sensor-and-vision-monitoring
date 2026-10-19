"""Command line: the same runs a front end makes, without one.

Three commands:

  list    prints what can be chosen: the plants, controllers and sensor sets (registry.py) and
          the upsets (scenarios.py), each with its key and title.
  run     makes one run, prints its scorecard and saves it under runs/saved/.
  matrix  runs every combination of the choices given and writes one row per run to
          runs/matrix_<time>.csv. Each run is saved too.

Every run goes through runs.py, exactly as the workbench's do (ARCHITECTURE.md, decision 24),
so a run made here and one made in the GUI with the same choices give the same numbers.

    pixi run python -m src.cli list
    pixi run python -m src.cli run --plant perturbed --controller mpc --upset feed_switch --hours 40
    pixi run python -m src.cli run --set run_file.flows.feed=650 --set controller_settings.w_reagent=0.02
    pixi run python -m src.cli matrix --plant matched perturbed --controller open_loop mpc \
                                       --upset feed_switch feed_interrupt --hours 40

`--set PATH=VALUE` edits one setting. PATH starts with run_file, parameters or
controller_settings (the three kinds of edit a RunSpec carries) and goes down the nested keys.
VALUE is read as JSON when it can be (650, 0.02, true, [1, 2]) and as plain text otherwise.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time

from .config import load_config
from .registry import CONTROLLERS, PLANTS, SENSOR_SETS, options
from .runs import HERE, RunSpec, execute, save
from .scenarios import catalogue

# The scorecard numbers printed by `run` and written by `matrix`, in this order. Units:
# off_spec_mol, lost_excess_mol, reagent_mol in mol; interface_excursion_m_h in m*h;
# time_to_settle_h in h; solver_failures is a count of intervals (see metrics.transition_cost).
SCORE_KEYS = ("off_spec_mol", "lost_excess_mol", "interface_excursion_m_h", "reagent_mol",
              "time_to_settle_h", "solver_failures")

# The places a --set edit may go: the RunSpec's three edit fields.
EDIT_ROOTS = ("run_file", "parameters", "controller_settings")


# ==============================================================================================
# The entry point and the argument parser
# ==============================================================================================

def main(argv=None):
    """Parse the command line and run the command it names.

    Args:
        argv: the arguments, without the program name. None reads them from sys.argv.

    Example:
        main(["run", "--plant", "matched", "--hours", "10"])
    """
    args = _parser().parse_args(argv)
    {"list": _list, "run": _run, "matrix": _matrix}[args.command](args)


def _parser() -> argparse.ArgumentParser:
    """The parser for the three commands: list, run and matrix."""
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="Solvent extraction cascade runs")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="what can be chosen").add_argument("--config", default="ndpr_main")

    run = sub.add_parser("run", help="one run")
    _add_choices(run, many=False)
    run.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                     help="edit a setting, e.g. run_file.flows.feed=650 or controller_settings.w_reagent=0.02")

    matrix = sub.add_parser("matrix", help="every combination of the choices")
    _add_choices(matrix, many=True)
    return parser


def _add_choices(parser, many: bool):
    """Add the run choices (config, plant, controller, upset, sensors, hours) to a sub-parser.

    The defaults are the same as RunSpec's. With many=True, plant, controller, upset and
    sensors each take one or more values (for the matrix) and default to a one-item list.

    Args:
        parser: the sub-parser for `run` or `matrix`.
        many: True for the matrix, False for a single run.
    """
    nargs = "+" if many else None
    parser.add_argument("--config", default="ndpr_main")
    parser.add_argument("--plant", nargs=nargs, default=_default("perturbed", many), choices=list(PLANTS))
    parser.add_argument("--controller", nargs=nargs, default=_default("mpc", many), choices=list(CONTROLLERS))
    # the upsets depend on the run file, so their names are checked when the run starts
    parser.add_argument("--upset", nargs=nargs, default=_default("feed_switch", many))
    parser.add_argument("--sensors", nargs=nargs, default=_default("full_state", many), choices=list(SENSOR_SETS))
    parser.add_argument("--hours", type=float, default=40.0)


def _default(value: str, many: bool):
    """A choice's default: the value itself for a single run, a one-item list for the matrix."""
    return [value] if many else value


# ==============================================================================================
# list: what can be chosen
# ==============================================================================================

def _list(args):
    """Print the key and title of every plant, controller, sensor set and upset."""
    rc = load_config(args.config)
    upsets = [(e.key, e.title, e.explanation) for e in catalogue(rc, _nominal(rc)).values()]
    for heading, rows in (("Plants", options(PLANTS)), ("Controllers", options(CONTROLLERS)),
                          ("Sensor sets", options(SENSOR_SETS)), ("Upsets", upsets)):
        print(f"\n{heading}")
        for key, title, _ in rows:
            print(f"  {key:24s} {title}")


def _nominal(rc):
    """The run file's nominal inputs. The upset catalogue needs inputs to be set against, and
    for listing the names any will do, so this skips commissioning."""
    from .config import build_circuit, nominal_inputs
    return nominal_inputs(rc, build_circuit(rc))


# ==============================================================================================
# run: one run, printed and saved
# ==============================================================================================

def _run(args):
    """Build a RunSpec from the arguments, apply any --set edits, run it, print and save it."""
    spec = RunSpec(config=args.config, plant=args.plant, controller=args.controller, upset=args.upset,
                   sensors=args.sensors, hours=args.hours)
    for item in args.set:
        _apply_edit(spec, item)
    t0 = time.time()                         # wall-clock seconds, for the printout
    result = execute(spec)
    print(spec.label(), f"({time.time() - t0:.0f} s)")
    for k in SCORE_KEYS:
        print(f"  {k:26s} {result.scorecard.get(k, float('nan')):12.3f}")
    print("saved", save(result))


def _apply_edit(spec: RunSpec, item: str):
    """Write one --set edit into the spec.

    Args:
        spec: the spec to edit, in place.
        item: "PATH=VALUE", where PATH starts with run_file, parameters or controller_settings.
            Missing levels of the nested dict are created.

    Raises:
        SystemExit: if PATH doesn't start with one of the three, or names nothing below it.

    Example:
        _apply_edit(spec, "run_file.flows.feed=650")    # spec.run_file == {"flows": {"feed": 650}}
    """
    path, _, text = item.partition("=")
    root, *keys = path.split(".")
    if root not in EDIT_ROOTS or not keys:
        raise SystemExit(f"--set {item}: the path must start with run_file., parameters. or controller_settings.")
    node = getattr(spec, root)
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = _parse_value(text)


def _parse_value(text: str):
    """The text as a JSON value (number, true/false, list ...) if it is one, else the text itself."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# ==============================================================================================
# matrix: every combination, one CSV row each
# ==============================================================================================
# The CSV is flushed after every row, so a long matrix can be watched (and a crash partway
# keeps the rows already done).

def _matrix(args):
    """Run every combination of plant, controller, upset and sensor set; write runs/matrix_<time>.csv."""
    combos = list(itertools.product(args.plant, args.controller, args.upset, args.sensors))
    out = HERE / "runs" / f"matrix_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plant", "controller", "upset", "sensors", "hours", *SCORE_KEYS, "seconds", "saved"])
        for combo in combos:
            row, summary = _matrix_row(args, combo)
            writer.writerow(row)
            f.flush()
            print(*summary, flush=True)
    print("wrote", out)


def _matrix_row(args, combo: tuple) -> tuple[list, tuple]:
    """Make and save one run of the matrix.

    Args:
        args: the parsed arguments (for config and hours).
        combo: this combination's keys, (plant, controller, upset, sensors).

    Returns:
        The CSV row: [plant, controller, upset, sensors, hours, the SCORE_KEYS values, seconds,
        saved folder name]; and what to print for it: the run's label and its first four scores.
    """
    plant, controller, upset, sensors = combo
    spec = RunSpec(config=args.config, plant=plant, controller=controller, upset=upset,
                   sensors=sensors, hours=args.hours)
    t0 = time.time()                         # wall-clock seconds, for the "seconds" column
    result = execute(spec)
    folder = save(result)
    row = [plant, controller, upset, sensors, args.hours,
           *[round(result.scorecard.get(k, float("nan")), 4) for k in SCORE_KEYS],
           round(time.time() - t0, 1), folder.name]
    summary = (spec.label(), {k: round(result.scorecard[k], 1) for k in SCORE_KEYS[:4]})
    return row, summary


if __name__ == "__main__":
    main()
