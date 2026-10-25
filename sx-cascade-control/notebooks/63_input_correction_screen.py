"""Screening: would correcting the model at its inputs beat the output bias on this plant? (v1.3, last test)

The question (Emilio, Oct 25). The MPC corrects model error with an output bias: a filtered gap
added to its predicted purity and recovery. Pannocchia and Rawlings (TWMCC 2001) found that on an
ill-conditioned plant with errors in the model's gains, a correction at the inputs is more robust and
follows slow disturbances faster. Is it worth building here? A screen before building anything: their
kind of experiment, on this plant's own numbers. Decision rule, set before running: build it only if
the input correction is clearly better on most of the plants; otherwise mention it and stop.

The method. For each of the 19 plants of notebook 24 (--settled), the plant's and the model's 2x2
long-run gains at the plant's own design point, from (ln scrub, ln organic) to (log10 impurity,
log10 loss). Each made a first-order system with time constant TAU (the same for every channel),
stepped every 0.5 h for 600 h. The controller plans with the model's gains: each step it computes the
handles that would put the outputs on target at steady state, after its correction, and moves a
fraction BETA of the way there. Two corrections, everything else the same:
  output   the model runs on its own; a filtered gap (plant - model) is added to its outputs (this MPC)
  input    the gap is treated as an offset on the handles, fed through the model, filtered
Three cases: a target change (impurity 0.2 decades lower), an unmeasured 2 % organic-flow offset at
the plant's input, and a step of 0.1 decades on the plant's measured impurity. Each correction gets
its best filter from FILTERS (the lowest median error over the plants), so neither is handicapped.
Linear, 2x2, no floors or constraints: a screen, not a proof.

A second table separates the two things that differ between the plants and the model here: the
correction, and how wrong the model is. The plant's own gains, each multiplied by a random factor
within +-ERROR_SIZES (seed 0), stand in for models of known accuracy; the input-offset case, each
correction at its best filter.

Run:  pixi run python notebooks/63_input_correction_screen.py
Writes runs/input_correction_screen_ndpr_main.csv (one row per plant, case, correction and filter).
"""
import ast
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "notebooks"))
from study_tools import write_rows                                     # noqa: E402

TAU, STEP_H, HOURS = 100.0, 0.5, 600.0          # h: chosen; the slow inventory mode is of this order or slower
BETA = 0.1                                      # share of the way to the target handles moved each step
FILTERS = (0.05, 0.2, 0.5)                      # correction filters tried (this MPC uses 0.5)
CASES = {"target": ([-0.2, 0.0], [0.0, 0.0], [0.0, 0.0]),        # (target, input offset, output step)
         "input_offset": ([0.0, 0.0], [0.0, 0.02], [0.0, 0.0]),
         "output_step": ([0.0, 0.0], [0.0, 0.0], [0.1, 0.0])}
ERROR_SIZES = (0.0, 0.1, 0.3, 0.5)              # how far each of the model's gains is off, at most
GAINS = HERE / "runs" / "conditioning_ndpr_plant_sized_settled.csv"
OUT = HERE / "runs" / "input_correction_screen_ndpr_main.csv"
OUT_SIZES = HERE / "runs" / "input_correction_error_sizes_ndpr_main.csv"


def plant_pairs() -> list:
    """(seed, plant gains, model gains) at each plant's own design point (notebook 24, settled)."""
    out = []
    for r in csv.DictReader(open(GAINS)):
        if r["case"].startswith("seed") and r.get("G_spec"):
            out.append((int(r["case"].split()[1]), np.array(ast.literal_eval(r["G_spec"])),
                        np.array(ast.literal_eval(r["G_model_spec"]))))
    return out


def simulate(Gp, Gm, case: tuple, how: tuple) -> dict:
    """One closed loop: integral of |error| over both outputs (h), the final error, and whether it
    stayed bounded. case: (target, input offset, output step); how: (correction, filter)."""
    target, w_in, w_out = (np.array(v, float) for v in case)
    correction, alpha = how
    a = np.exp(-STEP_H / TAU)
    xp, xm, u, gap = np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)
    iae = 0.0
    for _ in range(int(HOURS / STEP_H)):
        y = xp + w_out
        u_s, gap, xm = correction_step(Gm, (y, xm, u, gap), (target, alpha, a), correction)
        u = u + BETA * (u_s - u)
        if not np.all(np.isfinite(u)) or np.abs(u).max() > 1e3:
            return {"iae": np.inf, "final_error": np.inf, "bounded": False}
        xp = a * xp + (1 - a) * Gp @ (u + w_in)
        iae += np.abs(y - target).sum() * STEP_H
    return {"iae": iae, "final_error": float(np.abs(xp + w_out - target).sum()), "bounded": True}


def correction_step(Gm, signals: tuple, settings: tuple, correction: str) -> tuple:
    """One step of the estimator and target: (target handles, correction, model state).
    signals: (measured outputs, model state, handles, correction); settings: (target, filter, a)."""
    y, xm, u, gap = signals
    target, alpha, a = settings
    if correction == "output":
        gap = gap + alpha * (y - xm - gap)                   # a gap on the outputs
        return np.linalg.solve(Gm, target - gap), gap, a * xm + (1 - a) * Gm @ u
    gap = gap + alpha * np.linalg.solve(Gm, y - xm)          # the gap read as an offset on the handles
    return np.linalg.solve(Gm, target) - gap, gap, a * xm + (1 - a) * Gm @ (u + gap)


def screen() -> list:
    """Every plant, case, correction and filter."""
    combos = itertools.product(plant_pairs(), CASES.items(), ("output", "input"), FILTERS)
    return [{"seed": seed, "case": name, "correction": correction, "filter": alpha,
             **simulate(Gp, Gm, case, (correction, alpha))}
            for (seed, Gp, Gm), (name, case), correction, alpha in combos]


def best_filter(rows: list, case: str, correction: str) -> float:
    """The filter with the lowest median error over the plants for one case and correction."""
    def median_iae(alpha):
        """Median error over the plants with this filter."""
        return np.median([r["iae"] for r in rows if r["case"] == case and r["correction"] == correction
                          and r["filter"] == alpha])
    return min(FILTERS, key=median_iae)


def at_best_filter(rows: list, case: str, correction: str) -> dict:
    """One case and correction at its best filter: {seed: row}."""
    alpha = best_filter(rows, case, correction)
    return {r["seed"]: r for r in rows if r["case"] == case and r["correction"] == correction and r["filter"] == alpha}


def case_line(case: str, out: dict, inp: dict) -> str:
    """One case's line: each correction's median error and unbounded runs, and where input wins."""
    def part(name, pick):
        """One correction's part of the line."""
        iae = np.median([r["iae"] for r in pick.values()])
        return f"{name} (filter {next(iter(pick.values()))['filter']}): median error {iae:8.2f}, "                f"unbounded {sum(not r['bounded'] for r in pick.values())}"
    wins = sum(inp[s]["iae"] < out[s]["iae"] for s in out)
    return f"{case:13s} {part('output', out)} | {part('input', inp)} | input better on {wins} of {len(out)} plants"


def summary(rows: list) -> None:
    """Per case: each correction at its best filter; plants where input beats output; unbounded runs."""
    for case in CASES:
        print(case_line(case, at_best_filter(rows, case, "output"), at_best_filter(rows, case, "input")), flush=True)


def size_row(size: float, correction: str, factors: list) -> dict:
    """One model accuracy and correction on the input-offset case, at the correction's best filter."""
    def result(alpha):
        """Median error and unbounded runs over the plants with this filter."""
        res = [simulate(Gp, Gp * f, CASES["input_offset"], (correction, alpha)) for (_, Gp, _), f in zip(plant_pairs(), factors)]
        return {"filter": alpha, "median_iae": float(np.median([r["iae"] for r in res])),
                "unbounded": sum(not r["bounded"] for r in res), "plants": len(res)}
    return {"error_size": size, "correction": correction, **min((result(a) for a in FILTERS), key=lambda r: r["median_iae"])}


def error_sizes() -> list:
    """Both corrections against models of known accuracy (the plant's gains times random factors)."""
    rng, n = np.random.default_rng(0), len(plant_pairs())
    rows = []
    for size in ERROR_SIZES:
        factors = [1 + size * rng.uniform(-1, 1, (2, 2)) for _ in range(n)]
        rows += [size_row(size, c, factors) for c in ("output", "input")]
    return rows


def main():
    """The screen on every plant and the model-accuracy table; both written, with a summary."""
    rows = screen()
    write_rows(OUT, rows, first=("seed", "case", "correction", "filter"))
    summary(rows)
    sizes = error_sizes()
    write_rows(OUT_SIZES, sizes, first=("error_size", "correction"))
    for r in sizes:
        print(f"gains off by up to {int(100 * r['error_size']):2d} %  {r['correction']:6s} filter {r['filter']}: "
              f"median error {r['median_iae']:8.2f}, unbounded {r['unbounded']} of {r['plants']}", flush=True)
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
