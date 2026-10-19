"""A plain-language report of one run: what was run, what happened, what the numbers mean.

Why this file exists
--------------------
A run leaves behind a scorecard of six numbers and a few hundred points of time series. Both are
complete, but neither says in words what happened. This file reads a finished run (a
runs.RunResult) and writes a short report that a person new to the project can follow:

    1. What was run: the run file and what it's for, the plant, the controller, the upset, the
       instruments, the run length, and any setting changed from the files.
    2. What happened, in a few sentences: how purity and recovery moved and when, how far the
       controller moved each handle, whether the interfaces stayed in their window, whether any
       solve failed, and whether a safety layer (the interlock or the deployment guard) stepped in.
    3. The scorecard, each number with a line on what it measures.
    4. Things worth a second look: anything below spec, failed solves, large handle moves, a plant
       still moving at the end of the run.

The report is Markdown text, so it reads as plain text in any editor, shows formatted in the
workbench's Report panel, and is saved as report.md next to each saved run (runs.save).

This file only reads numbers and writes text; it runs nothing and knows nothing about any window.
"""
from __future__ import annotations

import numpy as np

from .config import run_file_description
from .scenarios import UPSET_TIME_H

# What each scorecard number measures, in one line. The keys are metrics.score's.
SCORE_MEANINGS = {
    "off_spec_mol": ("Product made off spec", "mol", "metal sent to the product while its purity was below the spec"),
    "lost_mol": ("Product metal lost", "mol", "Pr and Nd that left with the raffinate over the run"),
    "lost_excess_mol": ("Lost beyond the starting point", "mol",
                        "the same, minus what the plant was already losing at the start; negative means it lost less"),
    "time_to_settle_h": ("Time for purity to settle", "h", "when purity last left a narrow band around its final value"),
    "interface_excursion_m_h": ("Interfaces outside their window", "m·h",
                                "how far and how long any settler's interface sat outside its allowed band"),
    "reagent_mol": ("Reagent used", "mol", "base for saponification plus scrub and strip acid, over the run"),
    "solver_failures": ("Controller solves that failed", "", "intervals where the controller's optimization had no answer and it held the handles"),
}
HANDLE_NAMES = {"Q_org": "organic flow", "Q_scrub": "scrub flow", "h_scrub": "scrub acid",
                "saponified_fraction": "saponification"}
LARGE_MOVE = 0.30              # a handle moved more than 30 % from its start is pointed out (chosen)


def run_report(result) -> str:
    """The whole report for one finished run, as Markdown text.

    Args:
        result: a runs.RunResult (from a run, or read back with runs.load).
    Returns:
        The report text. Nothing is written to disk here; runs.save does that.

    Example:
        text = run_report(runs.execute(RunSpec(config="ndpr_main", upset="feed_switch", hours=80)))
    """
    parts = [f"# Run report\n\n{result.spec.label()}",
             _what_was_run(result),
             _what_happened(result),
             _scorecard_table(result.scorecard),
             _worth_a_look(result)]
    return "\n\n".join(parts) + "\n"


# ---- 1. what was run --------------------------------------------------------------------------------
def _what_was_run(result) -> str:
    """The choices behind the run, with the run file's own description."""
    spec, meta = result.spec, result.meta
    description = run_file_description(spec.config) or "(this run file has no description)"
    lines = ["## What was run", "",
             f"- **Run file:** {spec.config}. {description}",
             f"- **Plant:** {meta.get('plant_title', spec.plant)}",
             f"- **Controller:** {meta.get('controller_title', spec.controller)}",
             f"- **Upset:** {meta.get('upset_title', spec.upset)}" + (f", starting at {UPSET_TIME_H:g} h" if spec.upset != "none" else ""),
             f"- **Instruments:** {meta.get('sensors_title', spec.sensors)}",
             f"- **Length:** {spec.hours:g} h, the controller acting every {meta.get('t_step', 0.5):g} h",
             f"- **Settings changed from the files:** {_edits_text(spec)}"]
    return "\n".join(lines)


def _edits_text(spec) -> str:
    """The run's edits to the run file, the parameters and the controller settings, or "none"."""
    edits = [f"{section} {values}" for section, values in
             (("run file", spec.run_file), ("parameters", spec.parameters), ("controller", spec.controller_settings))
             if values]
    return "; ".join(edits) if edits else "none"


# ---- 2. what happened -------------------------------------------------------------------------------
def _what_happened(result) -> str:
    """A few sentences on purity, recovery, metal lost, the handles, the interfaces and the controller."""
    sentences = [_purity_sentence(result), _recovery_sentence(result), _lost_share_sentence(result),
                 _lost_sentence(result.scorecard),
                 _handles_sentence(result.handles), _interfaces_sentence(result.scorecard),
                 _controller_sentence(result)]
    return "## What happened\n\n" + "\n\n".join(s for s in sentences if s)


def _purity_sentence(result) -> str:
    """Start, lowest (and when), end; and how long it was below spec, if it was."""
    t, purity, spec = result.t, result.series["purity"], result.meta["spec"]
    low = int(np.argmin(purity))
    text = (f"Product purity started at {_pct(purity[0])}, was lowest at {_pct(purity[low])} at {t[low]:.1f} h, "
            f"and ended at {_pct(purity[-1])}. The spec is {_pct(spec)}.")
    hours_below = _hours_where(t, purity < spec)
    if hours_below > 0:
        return text + f" It was below spec for about {hours_below:.1f} h."
    return text + " It stayed above spec throughout."


def _recovery_sentence(result) -> str:
    """Start, lowest (and when), end, against the recovery target."""
    t, recovery = result.t, result.series["recovery"]
    low = int(np.argmin(recovery))
    target = result.meta.get("recovery_target")
    text = (f"Recovery (the share of the fed Pr and Nd that reached the product) started at {_pct(recovery[0])}, "
            f"was lowest at {_pct(recovery[low])} at {t[low]:.1f} h, and ended at {_pct(recovery[-1])}")
    return text + (f", against a target of {_pct(target)}." if target else ".")


def _lost_share_sentence(result) -> str:
    """The NdPr lost to the raffinate as a share of what was being fed: on average, and at worst.
    Runs saved before this measure existed don't have it; they get no sentence."""
    share = result.series.get("lost_pct_of_feed")
    if share is None or not np.isfinite(share).any():
        return ""
    worst = int(np.nanargmax(share))
    return (f"Of the Pr and Nd being fed, {np.nanmean(share):.2f} % on average left with the raffinate; "
            f"at worst {share[worst]:.2f} %, at {result.t[worst]:.1f} h.")


def _lost_sentence(scorecard: dict) -> str:
    """Metal lost beyond the starting point, in words."""
    excess = scorecard.get("lost_excess_mol")
    if excess is None:
        return ""
    if excess <= 0:
        return f"The plant lost {abs(excess):.0f} mol less Pr and Nd to the raffinate than it was losing at the start."
    return f"The plant lost {excess:.0f} mol more Pr and Nd to the raffinate than it was losing at the start."


def _handles_sentence(handles: dict) -> str:
    """How far each handle ended from where it started."""
    moves = [f"{HANDLE_NAMES.get(name, name)} {_signed_pct(_relative_change(values))}"
             for name, values in handles.items() if _is_series(values)]
    if not moves:
        return "The controller moved no handles (open loop)."
    return "Handles at the end, against their start: " + ", ".join(moves) + "."


def _interfaces_sentence(scorecard: dict) -> str:
    """Whether every interface stayed in its window, and if not, the excursion in m·h."""
    excursion = scorecard.get("interface_excursion_m_h", 0.0) or 0.0
    if excursion <= 1e-6:
        return "Every settler's interface stayed inside its window."
    return f"Some interfaces left their window: {excursion:.3f} m·h in all (how far, times how long)."


def _controller_sentence(result) -> str:
    """Failed solves, and anything a safety layer recorded (interlock trips, guard alarms)."""
    return _solves_text(result) + _interlock_text(result.info) + _guard_text(result.info)


def _solves_text(result) -> str:
    """How many controller solves failed; "" for a run with no intervals."""
    info = result.info
    failures = result.scorecard.get("solver_failures", 0) or 0
    if failures:
        return f"The controller's optimization failed in {failures} of {len(info)} intervals (it held the handles each time)."
    return "Every controller solve succeeded." if info else ""


def _interlock_text(info: list) -> str:
    """How many intervals the feed-loss interlock held the handles, or "" if it never tripped."""
    tripped = sum(1 for i in info if i.get("interlock") == "tripped")
    return f" The feed-loss interlock held the handles for {tripped} intervals." if tripped else ""


def _guard_text(info: list) -> str:
    """How many alarms the deployment guard raised and the first one, or "" if none."""
    alarms = [i["guard_alarm"] for i in info if i.get("guard_alarm")]
    return f" The deployment guard raised {len(alarms)} alarms; the first: {alarms[0]}" if alarms else ""


# ---- 3. the scorecard -------------------------------------------------------------------------------
def _scorecard_table(scorecard: dict) -> str:
    """Each number with its unit and a line on what it measures."""
    rows = ["## The numbers", "", "| | Value | What it measures |", "| --- | --- | --- |"]
    for key, (title, unit, meaning) in SCORE_MEANINGS.items():
        if key in scorecard:
            rows.append(f"| {title} | {_number(scorecard[key])} {unit} | {meaning} |")
    return "\n".join(rows)


# ---- 4. things worth a second look ------------------------------------------------------------------
def _worth_a_look(result) -> str:
    """Short warnings; "nothing unusual" if there are none."""
    flags = [f for f in (_flag_off_spec(result), _flag_failures(result), _flag_large_moves(result),
                         _flag_still_moving(result)) if f]
    body = "\n".join(f"- {f}" for f in flags) if flags else "Nothing unusual."
    return "## Worth a second look\n\n" + body


def _flag_off_spec(result) -> str:
    """Product made below spec, if more than half a mole of it."""
    off = result.scorecard.get("off_spec_mol", 0.0) or 0.0
    return f"{off:.0f} mol of product was made below spec." if off > 0.5 else ""


def _flag_failures(result) -> str:
    """Failed controller solves, if any."""
    n = result.scorecard.get("solver_failures", 0) or 0
    return f"{n} controller solves failed; see the log for their status." if n else ""


def _flag_large_moves(result) -> str:
    """Handles that ended more than LARGE_MOVE from where they started, if any."""
    big = [HANDLE_NAMES.get(k, k) for k, v in result.handles.items() if _is_large_move(v)]
    return f"Large handle moves (over {LARGE_MOVE:.0%}): {', '.join(big)}. Check the plant can deliver them." if big else ""


def _is_large_move(values) -> bool:
    """True for a handle series that ended more than LARGE_MOVE (a fraction) from its start."""
    return _is_series(values) and abs(_relative_change(values)) > LARGE_MOVE


def _flag_still_moving(result) -> str:
    """Purity still settling in the last tenth of the run: a longer run would show where it ends."""
    settle, hours = result.scorecard.get("time_to_settle_h"), result.spec.hours
    if settle is not None and settle > 0.9 * hours:
        return f"Purity was still moving at {settle:.1f} h of {hours:g} h; run longer to see where it settles."
    return ""


# ---- small helpers ----------------------------------------------------------------------------------
def _hours_where(t: np.ndarray, mask: np.ndarray) -> float:
    """Hours during which `mask` is true, counting each interval by its start."""
    return float(np.sum(np.diff(t)[mask[:-1]])) if len(t) > 1 else 0.0


def _is_series(values) -> bool:
    """True for a handle recorded as one number per interval, with at least one interval. The
    jacklegs are a row per interval and are left out of the handle sentences."""
    return np.ndim(values) == 1 and len(values) > 0


def _relative_change(values) -> float:
    """The last value against the first, as a fraction (0.1 = 10 % up); 0 if the first is 0."""
    values = np.asarray(values, float)
    return float(values[-1] / values[0] - 1.0) if values[0] else 0.0


def _pct(fraction) -> str:
    return f"{100 * float(fraction):.2f} %"


def _signed_pct(fraction: float) -> str:
    return f"{100 * fraction:+.1f} %"


def _number(value) -> str:
    """A scorecard value for the table: none, a whole number, or one decimal."""
    if value is None:
        return "-"
    if float(value).is_integer() or abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:,.1f}"
