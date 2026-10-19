"""Checks that the numbers the documents quote still match the result tables.

Every result in the write-ups comes from a file in runs/: the 60 h upset runs,
the 40 h controller catalogue, the commissioning trim, the sensor and handle
study, the structural plants, the learned policy, the stage-count sweeps, the
sensor placement, the twenty perturbed plants, the stage-sizing study, the
centering-weight sweep and the ranked-constraints trial. When those files are regenerated, every sentence that
quotes them has to be found and updated. This test does the finding.

The documents' numbers are from the code after decision 30 (the backup solver,
Oct 14 evening). The structural matrix of that code is the timestamped file the
matrix command wrote (MATRIX below); runs/structural_matrix_ndpr_plant.csv is
the copy from before decision 30.

Each claim below names a document, the table and row a number comes from,
how the text rounds it, and a short piece of the sentence with the number in
it. The test reads the number from the table, rounds it the way the text
does, fills in the sentence and checks the document contains it. Line breaks
and runs of spaces are ignored, since sentences wrap. When a claim fails, the
message gives the document, the sentence, what the document says now and
what the table gives.

Out of scope: numbers that don't come from a table in runs/. Parameters and
chosen settings, literature values, the first-run failures kept as history
(1042 mol against 156, the 2240 and 56 mol of the scrub acid read-back, the
0.365 to 0.006 m·h of the drift fix, 845 to 715 mol on the Pr | Nd circuit),
the 96.9 % and 99.75 % of the commissioning trim (the trim's log, not a
table), the exact-equilibrium check (58 % to 91 %, four thousandths) and the
Idaho rig's 99.7 % purity. The plant's own optimum, 2,340 mol/h, is from
runs/diag_oracle.log; it is kept below as a constant so the claims that
compare it with the trim can still follow the trim. Also out of scope: the
results from before decision 30 that the documents quote as history, from
earlier versions of the tables in git: the pump stop's 190 mol off spec on
seed 1, the matched plant's 1,300 mol and the sweep's 1,400 and 1,435 mol,
the seed study's "better on 1, worse on 7, worst 1,680", seed 3's 47 %
trim, "every step of every run failed" on seed 3, and the 0 to 195 mol of
the matrix after decision 26.

Checked by hand (the text rounds loosely, or the rule isn't simple enough to
encode). After a regeneration, read these sentences against the tables:

- "95 % takes about 24" (README), "95 % recovery needs about 24" (REPORT 3),
  "95 % needs ~24 stages" (PROGRESS): linear interpolation of runs/stage_count_ndpr.csv gives 23.4 on the best of the
  two scrub modes and 24.2 on the acid-only scrub.
- "95 % recovery needs about eighteen stages" (REPORT 3), "~18 with ~99 %
  stages" (PROGRESS): the same on stage_count_ndpr_ideal.csv gives 18.1 (best) and 18.3 (acid-only).
- "the two meet by thirty stages" (README, REPORT 3): reflux and acid-only
  scrub within half a point of each other from 32 stages on.
- "95 % needs about seventy stages" (README, REPORT 3 and 8): interpolation on stage_count_prnd.csv gives 70.3.
- "about 52 with the organic loaded as full as the optimizer likes"
  (REPORT 3): interpolation on stage_count_prnd_cap100.csv gives 52.6.
- "the same curve sits forty times higher" (README, REPORT 6, decision 23), "numbers 40× larger" (B.10 #24): err_single over sd_single in
  the placement file is about 46 at the best stages.
- "between stage 23 and the product stream" (README, REPORT 6, decision 23): where the estimator's purity error is about five thousandths.
- "uses a fifth less reagent" (README, REPORT 7, PROGRESS): the policy's mean reagent is 22 % below the controller's.
- "lets the interfaces stray a little" (README): the policy's
  interface metre-hours, 0 to 0.14 (the number is a claim in REPORT 7).
- "the uncontrolled circuit stays within a point of spec" (REPORT 5 and 6): open-loop purity at worst on the pump stop.
- "the interfaces stay in" on the pump stop (REPORT 5): 0.019 m·h controlled,
  0.020 open loop.
- "next to nothing at the others" (REPORT 5, decision 26) and "lost next to
  nothing at some weights" (README, APPB #6): in
  runs/center_weight_sweep_ndpr_plant.csv, the matched plant's pump stop at
  every weight but 0.005 (34 mol lost at most), the entrainment plant's at
  0.001 and 0.02.
- "at about half the scrub acidity" (REPORT 5): the saved feed-switch run's
  h_scrub, 1.21 M at the start against 0.58 M at the end.
- "a quarter to a half" (README, REPORT 5.3, decision 29, PROGRESS): on the five plants that can't make spec, the
  controller's median over open loop's, per upset, is 0.53 to 0.76.
- "+4.6 thousandths ... +64", "−9.7 points ... −1.5", "about fifteen times"
  (README, REPORT 5.3 and 10, decision 30, APPB #29, PROGRESS): seed 3, a 10 L/h scrub step run out long on the model and on the
  plant; a diagnostic, not a table.
- "both solvers still fail on most steps of four upsets" on seed 17: its
  solver_failures in seed_spread_ndpr_plant.csv, 61 to 68 of 80 on the feed
  switch, the flowmeter bias, the strong scrub acid and the slow mixers.
- "No variant brought a hard plant to spec": purity_end below 0.99 for seeds
  3 and 13 in runs/ranked_constraints_ndpr_plant.csv. "The straight-line cost
  changed little on seeds 1 and 2 and gave mixed results on 3 and 13": the
  same table.
- "each handle added after those brings it down" (README, REPORT 6): reagent on the feed switch, 18 → 10 → 5 → −0.5 %.
- "with one bad run each" (REPORT 5.3): the stronger feed on seed 2 and the
  strong scrub acid on seed 18 are the only plants the controller doesn't
  beat open loop on.
- "a little less organic" (REPORT 3): Q_org in the commissioning table.
- "every other set with one instrument ... nothing off spec", "the lab assay
  alone does it too", "no readings does no better than none" (README,
  REPORT 6).
- "On the slow mixers neither does anything" (README), "The slow mixers do
  almost nothing" (REPORT 4, 5 and 5.3).
- "95 % recovery needs about 19 stages" with mixers sized to the flows
  (README, REPORT 3 and 5.4, decision 28, PROGRESS, Appendix A):
  interpolation on stage_count_ndpr_sized_stages.csv gives 19.3.
- "stages get 95 to 96 %" on the sized plant: kτ/(1+kτ) with the 130 L mixer
  and the flows in configs/ndpr_plant_sized.toml. "1,634 L/h" and
  "2,172 mol/h": that run file's flows and the reagent in its header (a run
  file, not a table). "about 3.5 times slower": the ratio of the two slowest
  time constants in runs/stage_sizing_ndpr.csv.
- "all five sit below spec for every run": open-loop off spec on the drift
  and the slow mixers for those seeds in seed_spread_ndpr_plant.csv.
- "4 of 80 steps fell back" (decision 27, Appendix B, PROGRESS):
  from the run that stalled, not a table.
- The 96.9 %, 99.75 % and ten hours of the commissioning trim come from
  runs/diag_trim.log, which dates from before decision 26. The trimmed point
  moved a little since (2,712 → 2,717 mol/h); every run still starts at
  99.75 % (claim REPORT:trim_settles).
"""
import functools
import re
from dataclasses import dataclass
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parent.parent
RUNS = PROJECT / "runs"

# The plant's own optimum, reagent in mol/h, from runs/diag_oracle.log (the
# optimizer told the true parameters). Not in a table; see the docstring.
PLANT_OPTIMUM_MOL_PER_H = 2340

# Short names for the documents, used in the test ids.
DOCS = {
    "README": "README.md",
    "README_v1": "history/README_first_version.md",   # the first README, kept as history
    "REPORT": "REPORT.md",
    "ARCH": "ARCHITECTURE.md",
    "APPB": "APPENDIX_B_control.md",
    "PROGRESS": "PROGRESS.md",
}

# Short names for the result files.
DIST = "disturbances_ndpr_plant.csv"            # 60 h, open loop, perturbed plant
CAT = "mpc_catalogue_ndpr_plant.csv"            # 40 h, open loop against MPC, perturbed
CAT_MATCHED = "mpc_catalogue_ndpr_plant_matched.csv"
TRIM = "commissioning_ndpr_plant.csv"           # design handles against trimmed ones
TRIM_MATCHED = "commissioning_ndpr_plant_matched.csv"
STUDY = "sensor_actuator_ndpr_plant.csv"        # sensor sets and handle sets
# plant x controller x upset. The run after decision 30, as the matrix command wrote it; when it is
# copied to structural_matrix_ndpr_plant.csv, point this back there.
MATRIX = "structural_matrix_ndpr_plant.csv"
RL = "rl_eval_ndpr_small.csv"                   # learned policy against the MPC
STAGES = "stage_count_ndpr.csv"
STAGES_IDEAL = "stage_count_ndpr_ideal.csv"
STAGES_PRND = "stage_count_prnd.csv"            # Pr | Nd with the 85 % loading cap
STAGES_PRND_FREE = "stage_count_prnd_cap100.csv"  # Pr | Nd, organic loaded freely
PLACEMENT = "sensor_placement_ndpr_plant.npz"
# The first seed study (independent generator, 30 L mixers), kept as history: the sentences that quote it
# are labelled as the first version. The study on realistic plants replaced it (notebook 14, batch tables).
SEEDS = "history/seed_spread_ndpr_plant_v1.csv"        # 20 perturbed plants x 8 upsets x open loop / MPC
SEED_ORACLE = "history/seed_oracle_ndpr_plant_v1.csv"  # each seed's best recovery at spec
SIZING = "stage_sizing_ndpr.csv"                # 30 L against 130 L mixers: speed, linearity, runs
STAGES_SIZED = "stage_count_ndpr_sized_stages.csv"  # the stage sweep with the 130 L mixers
SWEEP = "center_weight_sweep_ndpr_plant.csv"    # the interface pull's weight, 0 to 0.02
RANKED = "ranked_constraints_ndpr_plant.csv"    # ranked constraints tried: variant x seed (decision 30)


# ---------------------------------------------------------------- reading results

@functools.lru_cache(maxsize=None)
def table(name):
    """One result file from runs/: a DataFrame for a .csv, a dict of arrays for the .npz."""
    path = RUNS / name
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    return pd.read_csv(path)


def row(name, **where):
    """The single row of a table where every named column has the given value."""
    frame = table(name)
    for column, value in where.items():
        frame = frame[frame[column] == value]
    assert len(frame) == 1, f"{name}: {where} picks {len(frame)} rows, not one"
    return frame.iloc[0]


def cell(name, column, **where):
    """One number from a table: the column of the single row picked by `where`."""
    return float(row(name, **where)[column])


def study(set_name, scenario, column):
    """One number from the sensor and handle study. 'open loop' is its own set."""
    if set_name == "open loop":
        return cell(STUDY, column, study="open loop", scenario=scenario)
    return cell(STUDY, column, set=set_name, scenario=scenario)


def best_recovery(name, stages):
    """Highest recovery of any successful solve at this stage count, either scrub mode."""
    frame = table(name)
    picked = frame[(frame["N"] == stages) & frame["success"]]
    return float(picked["recovery"].max())


def recovery_at(name, stages):
    """Best recovery at a stage count that wasn't run, by straight-line interpolation."""
    counts = sorted(set(table(name)["N"]))
    below = max(n for n in counts if n <= stages)
    above = min(n for n in counts if n >= stages)
    if below == above:
        return best_recovery(name, below)
    low, high = best_recovery(name, below), best_recovery(name, above)
    return low + (high - low) * (stages - below) / (above - below)


def percent(fraction):
    """A fraction as a percent."""
    return 100 * fraction


def percent_more(new, old):
    """How much bigger `new` is than `old`, in percent (negative when smaller)."""
    return 100 * (new / old - 1)


def reagent_change(scenario, set_name):
    """Extra reagent of a handle or sensor set on an upset, in percent of the open-loop run."""
    return percent_more(study(set_name, scenario, "reagent_mol"),
                        study("open loop", scenario, "reagent_mol"))


def trim_reagent_increase():
    """Extra reagent the commissioning trim needs over the model's design, in percent."""
    return percent_more(cell(TRIM, "trimmed", handle="reagent_mol_per_h"),
                        cell(TRIM, "design", handle="reagent_mol_per_h"))


def optimum_reagent_increase():
    """Extra reagent of the plant's own optimum over the model's design, in percent."""
    return percent_more(PLANT_OPTIMUM_MOL_PER_H, cell(TRIM, "design", handle="reagent_mol_per_h"))


# Episodes (numbered from 0 in the table) that have a feed switch. Episode 0 has only a
# flowmeter bias, and there the controller loses nothing, so a ratio means nothing.
RL_FEED_SWITCH_EPISODES = [1, 2, 3, 4]


def rl_feed_switch_episodes():
    """The learned-policy table, only the episodes with a feed switch."""
    frame = table(RL)
    return frame[frame["episode"].isin(RL_FEED_SWITCH_EPISODES)]


def rl_mean_ratio():
    """NdPr lost by the policy over NdPr lost by the MPC, averaged over the feed-switch episodes."""
    frame = rl_feed_switch_episodes()
    return float((frame["lost_excess_mol_rl"] / frame["lost_excess_mol_mpc"]).mean())


def placement(key):
    """One array (or number) from the sensor placement file."""
    return table(PLACEMENT)[key]


def best_single_sd():
    """Purity standard deviation, trusting the model, with one analyser at the best stage."""
    return float(placement("sd_single")[int(placement("best1"))])


# ---------------------------------------------------------------- how the text rounds

def rounded(value, decimals=0, mode=ROUND_HALF_UP):
    """A number rounded the way a person does it (halves go up), as a Decimal."""
    return Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-decimals), rounding=mode)


def whole(value):
    """Nearest whole number, with thousands commas: 1754.5 -> '1,755'."""
    return f"{rounded(value):,}"


def plain(value):
    """Nearest whole number, no commas: 2308."""
    return f"{rounded(value)}"


def decimals(places):
    """Formatter: this many decimal places."""
    return lambda value: f"{rounded(value, places):,}"


def nearest(step):
    """Formatter: rounded to the nearest 10 or 100, with thousands commas."""
    return lambda value: f"{rounded(value / step) * step:,}"


def floored(value):
    """Whole number, rounded down: 17.5 -> '17'."""
    return f"{rounded(value, 0, ROUND_FLOOR)}"


def floored_to(places):
    """Formatter: rounded down to this many decimals, for 'above 99.3 %'."""
    return lambda value: f"{rounded(value, places, ROUND_FLOOR)}"


def nothing_below_zero(value):
    """Nearest whole number, with anything negative read as 0 ('945 -> 0 mol lost')."""
    return whole(max(value, 0.0))


def with_minus_sign(value):
    """Nearest whole number, written with a typographic minus: '−12'."""
    return whole(value).replace("-", "−")


NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen", 20: "twenty", 26: "twenty-six",
}


def in_words(value):
    """Nearest whole number, spelled out: 'five', 'eleven', 'twenty-six'."""
    number = int(rounded(value))
    return NUMBER_WORDS.get(number, str(number))


def floored_in_words(value):
    """Whole number rounded down, spelled out: 3.05 -> 'three'."""
    number = int(rounded(value, 0, ROUND_FLOOR))
    return NUMBER_WORDS.get(number, str(number))


def times_in_words(value):
    """A ratio as the text says it: 2 -> 'twice'."""
    return "twice" if rounded(value) == 2 else in_words(value)


def tenth_in_words(value):
    """Thousandths of purity as the text says them: 0.1 -> 'a tenth'."""
    return "a tenth" if rounded(value, 1) == Decimal("0.1") else f"{rounded(value, 2)}"


def nothing_if_zero(value):
    """'nothing' when the value rounds to zero, else the whole number: 'nothing off spec'."""
    return "nothing" if rounded(value) == 0 else whole(value)


# ---------------------------------------------------------------- the claims

@dataclass(frozen=True)
class Claim:
    """One number a document quotes, and where in runs/ it comes from.

    Either `table`, `where` and `column` pick the number(s) from one row, and
    `transform` (if given) is applied to each, or `compute` works them out
    from any table. `fmt` rounds each value the way the text does (one
    formatter, or one per value). `template` is the piece of the sentence the
    document must contain, with {} where each number goes.
    """

    doc: str
    name: str
    template: str
    fmt: Callable | tuple = whole
    table: str = ""
    where: tuple = ()
    column: str | tuple = ""
    transform: Callable | None = None
    compute: Callable | None = None
    xfail: str = ""

    def values(self):
        """The numbers this claim quotes, straight from the results."""
        if self.compute is not None:
            found = self.compute()
            return list(found) if isinstance(found, (tuple, list)) else [found]
        picked = row(self.table, **dict(self.where))
        columns = (self.column,) if isinstance(self.column, str) else self.column
        values = [float(picked[c]) for c in columns]
        return [self.transform(v) for v in values] if self.transform else values

    def expected_text(self):
        """The template filled with the numbers, rounded the way the text rounds them."""
        values = self.values()
        formats = self.fmt if isinstance(self.fmt, tuple) else (self.fmt,) * len(values)
        return self.template.format(*(f(v) for f, v in zip(formats, values)))


def C(doc, name, template, fmt=whole, table="", where=None, column="", transform=None,
      compute=None, xfail=""):
    """Shorthand for a Claim, with `where` given as a plain dict."""
    return Claim(doc, name, template, fmt, table, tuple((where or {}).items()), column,
                 transform, compute, xfail)


def disturbance(scenario):
    """Row picker for the 60 h open-loop upsets."""
    return dict(table=DIST, where={"scenario": scenario})


def catalogue(scenario, matched=False):
    """Row picker for the 40 h controller catalogue."""
    return dict(table=CAT_MATCHED if matched else CAT, where={"scenario": scenario})


def matrix(plant, controller, upset):
    """Row picker for the structural matrix."""
    return dict(table=MATRIX, where={"plant": plant, "controller": controller, "upset": upset})


def episode(n):
    """Row picker for the learned policy's episodes (numbered from 0 in the table)."""
    return dict(table=RL, where={"episode": n})


def matrix_row_claims(doc):
    """The off-spec table of the structural plants, one claim per row."""
    rows = [("matched", "mpc", "MPC"), ("perturbed", "mpc", "MPC"), ("settler", "mpc", "MPC"),
            ("entrainment", "mpc", "MPC"), ("matched", "mpc_fixed", "MPC fixed at one point"),
            ("settler", "mpc_fixed", "MPC fixed at one point"),
            ("entrainment", "mpc_fixed", "MPC fixed at one point")]
    upsets = ("feed_switch", "feed_interrupt", "flowmeter_bias", "interface_drift")
    claims = []
    for plant, controller, label in rows:
        def offspec(plant=plant, controller=controller):
            return [cell(MATRIX, "off_spec_mol", plant=plant, controller=controller, upset=u)
                    for u in upsets]
        claims.append(C(doc, f"matrix_{plant}_{controller}",
                        f"| {plant} | {label} | {{}} | {{}} | {{}} | {{}} |", compute=offspec))
    return claims


def rl_table_claims(doc):
    """REPORT 7's episode table, one claim per row."""
    claims = []
    for n in range(5):
        claims.append(C(
            doc, f"rl_episode_{n + 1}",
            f"| {n + 1} | {{}} % | {{}} % | {{}} | {{}} | {{}} | {{}} |",
            fmt=(decimals(2), decimals(2), whole, whole, nearest(100), nearest(100)),
            compute=lambda n=n: [percent(cell(RL, "purity_min_rl", episode=n)),
                                 percent(cell(RL, "purity_min_mpc", episode=n)),
                                 cell(RL, "lost_excess_mol_rl", episode=n),
                                 cell(RL, "lost_excess_mol_mpc", episode=n),
                                 cell(RL, "reagent_mol_rl", episode=n),
                                 cell(RL, "reagent_mol_mpc", episode=n)]))
    return claims


def rl_purity_ranges():
    """Lowest purity per episode, as ranges: policy low and high, MPC low and high."""
    rl, mpc = percent(table(RL)["purity_min_rl"]), percent(table(RL)["purity_min_mpc"])
    return [rl.min(), rl.max(), mpc.min(), mpc.max()]


def rl_loss_ranges():
    """NdPr lost beyond nominal on the feed-switch episodes, as ranges: policy low and high, MPC
    low and high."""
    frame = rl_feed_switch_episodes()
    rl, mpc = frame["lost_excess_mol_rl"], frame["lost_excess_mol_mpc"]
    return [rl.min(), rl.max(), mpc.min(), mpc.max()]


def reflux_advantage():
    """Most points of recovery the reflux scrub gains over the acid-only scrub, any size."""
    frame = table(STAGES)
    ok = frame[frame["success"]]
    best = ok.groupby(["mode", "N"])["recovery"].max().unstack("mode")
    return float((best["reflux"] - best["none"]).max() * 100)


def largest_reagent_move():
    """Largest change in reagent, MPC against open loop, over the whole catalogue, in percent."""
    frame = table(CAT)
    return float((100 * (frame["reagent_mol_mpc"] / frame["reagent_mol_open"] - 1)).abs().max())


def fixed_mpc_feed_switch_range():
    """Fixed linear MPC off spec on a feed switch, lowest and highest over the matched, settler and
    entrainment plants (the three in the documents' table)."""
    values = [cell(MATRIX, "off_spec_mol", plant=p, controller="mpc_fixed", upset="feed_switch")
              for p in ("matched", "settler", "entrainment")]
    return [min(values), max(values)]


def lowest_purity_with_an_instrument():
    """Lowest purity on the flowmeter bias over every sensor set that reads something, the
    mid-cascade pair alone and with interface readings apart (the documents quote those on their
    own), in percent."""
    frame = table(STUDY)
    picked = frame[(frame["study"] == "sensors") & (frame["scenario"] == "flowmeter_bias")
                   & (frame["set"] != "nothing") & (frame["set"] != "ECJTU pair")
                   & (frame["set"] != "pair + interfaces")]
    return percent(float(picked["purity_min"].min()))


def pump_stop_sets_with_nothing_off_spec(full_state=True):
    """Sensor sets that put nothing off spec on the pump stop (under 1 mol): how many, and the
    lowest and highest NdPr lost beyond nominal among them. 'nothing' (no readings) is left out;
    so is the full state unless asked for."""
    frame = table(STUDY)
    picked = frame[(frame["study"] == "sensors") & (frame["scenario"] == "feed_interrupt")
                   & (frame["set"] != "nothing") & (frame["off_spec_mol"] < 1.0)]
    if not full_state:
        picked = picked[picked["set"] != "full state"]
    return [len(picked), picked["lost_excess_mol"].min(), picked["lost_excess_mol"].max()]


def matrix_pump_stop_worst_loss():
    """Most NdPr lost beyond nominal by the MPC on the pump stop, over the four plants."""
    frame = table(MATRIX)
    picked = frame[(frame["controller"] == "mpc") & (frame["upset"] == "feed_interrupt")]
    return float(picked["lost_excess_mol"].max())


def ranked(variant, seed, column):
    """One number from the ranked-constraints table."""
    return cell(RANKED, column, variant=variant, seed=seed)


# ---- the twenty perturbed plants (notebooks/14_seed_spread.py, its summary logic)

def seeds_that_can_make_spec():
    """Seeds whose plant can reach 95 % recovery at 99.5 % purity with some handles."""
    oracle = table(SEED_ORACLE)
    return set(oracle.loc[oracle["best_recovery_at_spec"] >= 0.95, "seed"])


def best_recovery_of_seeds_that_cannot():
    """Lowest and highest best recovery at spec, in percent, over the seeds that can't make it."""
    oracle = table(SEED_ORACLE)
    best = oracle.loc[oracle["best_recovery_at_spec"] < 0.95, "best_recovery_at_spec"]
    return [percent(best.min()), percent(best.max())]


def seed_cost(upset, controller, can_make_spec=True):
    """NdPr lost beyond nominal (a gain counts as zero) plus product off spec, mol, per seed,
    on the seeds that can make spec (or, with can_make_spec=False, on the ones that can't)."""
    frame = table(SEEDS)
    chosen = frame["seed"].isin(seeds_that_can_make_spec())
    frame = frame[(frame["upset"] == upset) & (frame["controller"] == controller)
                  & (chosen if can_make_spec else ~chosen)].set_index("seed")
    return frame["lost_excess_mol"].clip(lower=0) + frame["off_spec_mol"]


def seed_better(upset, can_make_spec=True):
    """On how many of the plants that can make spec (or can't) the controller beats open loop
    (by > 1 mol)."""
    return int((seed_cost(upset, "mpc", can_make_spec)
                < seed_cost(upset, "open_loop", can_make_spec) - 1.0).sum())


SEED_UPSETS = ("feed_switch", "feed_step", "feed_acid", "feed_interrupt", "flowmeter_bias",
               "interface_drift", "reflux_acid", "efficiency_loss")


def seed_hard_medians():
    """On the plants that can't make spec, the per-upset medians: lowest and highest open loop,
    lowest and highest controller, mol."""
    open_loop = [seed_cost(u, "open_loop", False).median() for u in SEED_UPSETS]
    mpc = [seed_cost(u, "mpc", False).median() for u in SEED_UPSETS]
    return [min(open_loop), max(open_loop), min(mpc), max(mpc)]


def seed_hard_better():
    """On the plants that can't make spec, the fewest and most of them the controller beats open
    loop on, over the eight upsets."""
    counts = [seed_better(u, False) for u in SEED_UPSETS]
    return [min(counts), max(counts)]


def seed_hard_failures():
    """Controller steps whose QP failed (both solvers) on the plants that can't make spec, all
    upsets: in total, and on seed 17."""
    frame = table(SEEDS)
    picked = frame[(frame["controller"] == "mpc") & ~frame["seed"].isin(seeds_that_can_make_spec())]
    return [int(picked["solver_failures"].sum()),
            int(picked.loc[picked["seed"] == 17, "solver_failures"].sum())]


def seed_worse(upset):
    """On how many of the plants that can make spec the controller loses to open loop (by > 1 mol)."""
    return int((seed_cost(upset, "mpc") > seed_cost(upset, "open_loop") + 1.0).sum())


def seed_open_loop_clean(upset):
    """On how many of the plants that can make spec open loop costs nothing (1 mol or less)."""
    return int((seed_cost(upset, "open_loop") <= 1.0).sum())


def seed_row(upset):
    """One row of REPORT 5.3's table: open-loop median, MPC median, MPC worst, MPC better on."""
    return [float(seed_cost(upset, "open_loop").median()), float(seed_cost(upset, "mpc").median()),
            float(seed_cost(upset, "mpc").max()), seed_better(upset)]


def seed_drift_open_loop_interfaces():
    """Lowest and highest open-loop interface metre-hours on the drift, plants that can make spec."""
    frame = table(SEEDS)
    picked = frame[(frame["upset"] == "interface_drift") & (frame["controller"] == "open_loop")
                   & frame["seed"].isin(seeds_that_can_make_spec())]["interface_excursion_m_h"]
    return [picked.min(), picked.max()]


# ---- stages sized to the flows (notebooks/13_stage_sizing.py)

def sizing(plant, what):
    """One number from the stage-sizing table. plant: '30 L', '130 L' or '130 L, 120 h'."""
    labels = {"30 L": "30 L mixers (ndpr_plant)", "130 L": "130 L mixers (ndpr_plant_sized)",
              "130 L, 120 h": "130 L mixers (ndpr_plant_sized), 120 h"}
    return cell(SIZING, "value", plant=labels[plant], what=what)


def purity_nonlinearity_range(plant):
    """Lowest and highest purity nonlinearity over the four handles, in percent."""
    values = [sizing(plant, f"nonlin_purity {h}")
              for h in ("Q_org", "Q_scrub", "h_scrub", "saponified_fraction")]
    return [percent(min(values)), percent(max(values))]


def sweep(w, plant, upset, column):
    """One number from the centering-weight sweep."""
    return cell(SWEEP, column, w=w, plant=plant, upset=upset)


def placement_error(stage):
    """Purity error on the perturbed plant, one analyser at this stage, in thousandths."""
    return 1000 * float(placement("err_single")[stage])


def best_placement_on_plant():
    """Stage where one analyser gives the smallest purity error on the perturbed plant."""
    return int(np.nanargmin(placement("err_single")))


def error_from_late_scrub_to_product():
    """Mean purity error, thousandths, one analyser anywhere from stage 23 to the product stream."""
    return 1000 * float(np.mean(placement("err_single")[23:29]))


def error_in_extraction():
    """Mean purity error, thousandths, one analyser in the extraction section (stages 0 to 7)."""
    return 1000 * float(np.mean(placement("err_single")[0:8]))


# The product stream is read at stage 28, the first strip stage (8 + 20 + 4 stages).
PRODUCT_STREAM_STAGE = 28


# ---- the main case (configs/ndpr_main.toml), quoted in the README ----
@functools.lru_cache(maxsize=None)
def main_seed_costs():
    """NdPr lost beyond nominal (a gain counts as zero) plus off spec, per (seed, upset, controller),
    from the main case's seed-study batch tables."""
    files = sorted(RUNS.glob("seed_spread_ndpr_main_s*-*.csv"))
    d = pd.concat([pd.read_csv(f) for f in files])
    d["cost"] = d["lost_excess_mol"].clip(lower=0) + d["off_spec_mol"]
    return d.pivot_table(index=["upset", "seed"], columns="controller", values="cost")


def main_better(upset):
    """On how many plants the controller loses less than open loop (by more than 1 mol)."""
    t = main_seed_costs().loc[upset]
    return int((t["mpc_level_loops"] < t["open_loop_level_loops"] - 1.0).sum())


def main_median(upset, controller):
    return float(main_seed_costs().loc[upset][controller].median())


def margin_row(case):
    return row("operating_margin_summary.csv", case=case)


CLAIMS = [
    # ------------------------------------------------------------ README
    *[C("README_v1", f"stages_{n}", template, compute=lambda n=n: percent(best_recovery(STAGES, n)))
      for n, template in [(12, "12 stages give {} %"), (16, "16 give {} %"), (20, "20 give {} %"),
                          (22, "22 give {} %"), (26, "26 give {} %"), (32, "32 give {} %")]],
    C("README_v1", "reflux_advantage", "beats an acid-only scrub by up to {} points",
      fmt=in_words, compute=reflux_advantage),
    C("README_v1", "design_reagent", "the loading cap: {} mol/h of caustic and acid",
      table=TRIM, where={"handle": "reagent_mol_per_h"}, column="design"),
    C("README_v1", "design_organic", "{} L/h of organic against 600 L/h of feed",
      table=TRIM, where={"handle": "Q_org"}, column="design"),
    C("README_v1", "trim_reagent", "at lower acidity, at {} mol/h",
      table=TRIM, where={"handle": "reagent_mol_per_h"}, column="trimmed"),
    C("README_v1", "trim_cost", "most of that {} % is what the model's error costs",
      fmt=floored, compute=trim_reagent_increase),
    C("README_v1", "60h_feed_switch", "the feed switch sends {} mol of NdPr beyond the nominal 5 %",
      fmt=nearest(10), **disturbance("feed_switch"), column="lost_excess_mol"),
    C("README_v1", "60h_scrub_acid", "the strong scrub acid {}, the feed acid",
      fmt=nearest(10), **disturbance("reflux_acid"), column="lost_excess_mol"),
    C("README_v1", "60h_feed_acid", "the feed acid {}, the stronger feed",
      fmt=nearest(10), **disturbance("feed_acid"), column="lost_excess_mol"),
    C("README_v1", "60h_feed_step", "the stronger feed {}; the flowmeter bias",
      fmt=nearest(10), **disturbance("feed_step"), column="lost_excess_mol"),
    C("README_v1", "60h_flowmeter", "the flowmeter bias takes purity to {} % and {} mol of product off spec",
      fmt=(decimals(1), nearest(10)), compute=lambda: [
          percent(cell(DIST, "purity_min", scenario="flowmeter_bias")),
          cell(DIST, "off_spec_mol", scenario="flowmeter_bias")]),
    C("README_v1", "60h_pump_stop", "the pump stop takes purity to {} %", fmt=decimals(1),
      **disturbance("feed_interrupt"), column="purity_min", transform=percent),
    C("README_v1", "40h_feed_switch", "on the feed switch {} → {} mol",
      fmt=(whole, nothing_below_zero), **catalogue("feed_switch"),
      column=("lost_excess_mol_open", "lost_excess_mol_mpc")),
    C("README_v1", "40h_scrub_acid", "on the strong scrub acid {} → {}",
      fmt=(nearest(10), nothing_below_zero), **catalogue("reflux_acid"),
      column=("lost_excess_mol_open", "lost_excess_mol_mpc")),
    C("README_v1", "40h_feed_acid", "on the feed acid {} → {}",
      fmt=(nearest(10), whole), **catalogue("feed_acid"),
      column=("lost_excess_mol_open", "lost_excess_mol_mpc")),
    C("README_v1", "40h_feed_step", "on the stronger feed {} → {};", **catalogue("feed_step"),
      column=("lost_excess_mol_open", "lost_excess_mol_mpc")),
    C("README_v1", "40h_flowmeter", "flowmeter bias purity at worst {} % → {} % with {} mol off spec → none",
      fmt=(decimals(1), decimals(1), nearest(100)), compute=lambda: [
          percent(cell(CAT, "purity_min_open", scenario="flowmeter_bias")),
          percent(cell(CAT, "purity_min_mpc", scenario="flowmeter_bias")),
          cell(CAT, "off_spec_mol_open", scenario="flowmeter_bias")]),
    C("README_v1", "40h_drift", "drifting hydraulics {} → {} metre-hours",
      fmt=(whole, decimals(2)), **catalogue("interface_drift"),
      column=("interface_excursion_m_h_open", "interface_excursion_m_h_mpc")),
    C("README_v1", "40h_reagent", "Reagent moves by at most {} % either way",
      compute=largest_reagent_move),
    C("README_v1", "40h_pump_stop",
      "this plant: {} off spec either way, and {} mol of NdPr lost where the uncontrolled circuit loses none",
      fmt=(nothing_if_zero, whole), compute=lambda: [
          max(cell(CAT, "off_spec_mol_open", scenario="feed_interrupt"),
              cell(CAT, "off_spec_mol_mpc", scenario="feed_interrupt")),
          cell(CAT, "lost_excess_mol_mpc", scenario="feed_interrupt")]),
    C("README_v1", "sweep_entrainment", "and {} mol of NdPr at another", fmt=nearest(10),
      compute=lambda: max(sweep(w, "entrainment", "feed_interrupt", "lost_excess_mol")
                          for w in (0.0, 0.0002, 0.001, 0.005, 0.02))),
    C("README_v1", "seed_pump_stop_worse", "the controller is worse than doing nothing on {} of 15",
      fmt=plain, compute=lambda: seed_worse("feed_interrupt")),
    C("README_v1", "40h_feed_switch_dip", "the controller lets purity dip to {} % for a few hours",
      fmt=decimals(1), **catalogue("feed_switch"), column="purity_min_mpc", transform=percent),
    *matrix_row_claims("README_v1"),
    C("README_v1", "matrix_pump_stop_loss", "loses at most {} mol of NdPr",
      compute=matrix_pump_stop_worst_loss),
    C("README_v1", "fixed_mpc_feed_switch", "{} to {} mol off spec on a feed switch",
      compute=fixed_mpc_feed_switch_range),
    C("README_v1", "offspec_215", "so {} against {} between two similar runs", compute=lambda: [
        cell(MATRIX, "off_spec_mol", plant="perturbed", controller="mpc", upset="feed_switch"),
        cell(MATRIX, "off_spec_mol", plant="matched", controller="mpc", upset="feed_switch")]),
    C("README_v1", "placement_stage", "one on-stream analyser at stage {}",
      plain, compute=lambda: int(placement("best1"))),
    C("README_v1", "placement_sd", "pins product purity to {} of a thousandth",
      fmt=tenth_in_words, compute=lambda: 1000 * best_single_sd()),
    C("README_v1", "placement_second", "a second improves that by {} %", compute=lambda: 100 * (
        1 - float(placement("sd_pair")[int(placement("best2"))]) / best_single_sd())),
    C("README_v1", "placement_ecjtu", "the patent's two-analyser layout is {} % worse",
      compute=lambda: percent_more(float(placement("sd_ecjtu")), best_single_sd())),
    C("README_v1", "placement_lab", "a lab assay of the product alone is {} as bad",
      fmt=times_in_words, compute=lambda: float(placement("sd_lab")) / best_single_sd()),
    C("README_v1", "plant_placement_late", "about {} thousandths from any analyser",
      fmt=in_words, compute=error_from_late_scrub_to_product),
    C("README_v1", "plant_placement_extraction", "{} from one in the extraction section",
      fmt=in_words, compute=error_in_extraction),
    C("README_v1", "plant_placement_ecjtu", "{} from the patent's pair",
      fmt=in_words, compute=lambda: 1000 * float(placement("err_ecjtu"))),
    C("README_v1", "plant_placement_lab", "{} from the lab assay alone",
      fmt=in_words, compute=lambda: 1000 * float(placement("err_lab"))),
    C("README_v1", "sensors_feed_switch_open", "the controller loses nothing ({} mol uncontrolled)",
      compute=lambda: study("open loop", "feed_switch", "lost_excess_mol")),
    C("README_v1", "sensors_pair_feed", "when a feed analyser is added ({} mol off spec)",
      fmt=nearest(100), compute=lambda: study("ECJTU pair + feed analyser", "feed_switch", "off_spec_mol")),
    C("README_v1", "sensors_product_late", "alone see it late ({} mol)", fmt=nearest(10),
      compute=lambda: study("product + raffinate analysers", "feed_switch", "lost_excess_mol")),
    C("README_v1", "sensors_lab_nothing", "a lab assay alone loses {}, no readings {}.",
      fmt=nearest(10), compute=lambda: [study("lab assay only", "feed_switch", "lost_excess_mol"),
                                        study("nothing", "feed_switch", "lost_excess_mol")]),
    C("README_v1", "sensors_pair_flowmeter",
      "the mid-cascade pair alone lets {} mol go off spec, {} with interface readings added",
      fmt=nearest(10), compute=lambda: [study("ECJTU pair", "flowmeter_bias", "off_spec_mol"),
                                        study("pair + interfaces", "flowmeter_bias", "off_spec_mol")]),
    C("README_v1", "sensors_flowmeter", "every other set with one holds purity above {} %",
      fmt=floored_to(1), compute=lowest_purity_with_an_instrument),
    C("README_v1", "sensors_pump_stop_clean", "On the pump stop most sets put nothing off spec and lose {} to {} mol of NdPr",
      compute=lambda: pump_stop_sets_with_nothing_off_spec()[1:]),
    C("README_v1", "sensors_product_pump_stop", "take purity to {} % with {} mol off spec",
      fmt=(whole, nearest(10)), compute=lambda: [
          percent(study("product + raffinate analysers", "feed_interrupt", "purity_min")),
          study("product + raffinate analysers", "feed_interrupt", "off_spec_mol")]),
    C("README_v1", "sensors_nothing_pump_stop", "no readings at all puts {} mol off spec",
      fmt=nearest(10), compute=lambda: study("nothing", "feed_interrupt", "off_spec_mol")),
    C("README_v1", "handles_reagent",
      "organic flow alone holds the metal at {} % more reagent and the patent's two flows at {} %",
      compute=lambda: [reagent_change("feed_switch", "organic flow"),
                       reagent_change("feed_switch", "organic + scrub flow (ECJTU)")]),
    C("README_v1", "handles_full", "the full set uses {} % less than doing nothing", fmt=decimals(1),
      compute=lambda: -reagent_change("feed_switch", "+ jacklegs (all)")),
    C("README_v1", "handles_pump_stop", "({} mol, the patent's two flows plus saponification)",
      fmt=nearest(10), compute=lambda: study("+ saponification", "feed_interrupt", "off_spec_mol")),
    C("README_v1", "rl_purity", "Both hold purity on all five ({} % at worst)", fmt=decimals(1),
      compute=lambda: min(rl_purity_ranges())),
    C("README_v1", "rl_ratio", "loses about {} times more NdPr", fmt=in_words, compute=rl_mean_ratio),
    C("README_v1", "rl_losses", "({}–{} mol against {}–{})", fmt=nearest(10), compute=rl_loss_ranges),
    C("README_v1", "rl_no_feed_switch", "the policy still loses {} mol there against {} for the controller",
      fmt=(nearest(10), nothing_below_zero), **episode(0),
      column=("lost_excess_mol_rl", "lost_excess_mol_mpc")),
    C("README_v1", "seed_cannot", "optimizer told their true numbers, is {} to {} %",
      compute=best_recovery_of_seeds_that_cannot),
    C("README_v1", "seed3_trim", "(seed 3 at {} % purity)",
      compute=lambda: percent(ranked("plain", 3, "purity_start"))),
    C("README_v1", "seed_hard_medians", "(medians {} to {} mol uncontrolled, {} to {} controlled)",
      fmt=nearest(10), compute=seed_hard_medians),
    C("README_v1", "seed_hard_better", "it's better on {} or {} of the 5 each time", fmt=plain,
      compute=seed_hard_better),
    C("README_v1", "seed_hard_failures", "Now {} of the 3,200 steps on those plants fail, {} of them on seed 17",
      fmt=plain, compute=seed_hard_failures),
    C("README_v1", "seed_feed_switch", "on the feed switch on all {} (median {} → {} mol)",
      fmt=(plain, whole, whole), compute=lambda: [
          seed_better("feed_switch"), seed_cost("feed_switch", "open_loop").median(),
          seed_cost("feed_switch", "mpc").median()]),
    C("README_v1", "seed_others",
      "on the feed acid on all {}, on the stronger feed on {} and on the strong scrub acid on {}",
      fmt=plain, compute=lambda: [seed_better(u) for u in ("feed_acid", "feed_step", "reflux_acid")]),
    C("README_v1", "seed_flowmeter",
      "On the flowmeter bias it's better on {}; on the other {} the uncontrolled plant puts nothing off spec",
      fmt=plain, compute=lambda: [seed_better("flowmeter_bias"), seed_open_loop_clean("flowmeter_bias")]),
    C("README_v1", "seed_pump_stop", "On the pump stop it's better on {} and worse on {}, up to {} mol",
      fmt=(plain, plain, whole), compute=lambda: [
          seed_better("feed_interrupt"), seed_worse("feed_interrupt"),
          seed_cost("feed_interrupt", "mpc").max()]),
    C("README_v1", "sizing_time_constant", "slowest time constant {} h against {}", compute=lambda: [
        sizing("130 L", "slowest time constant"), sizing("30 L", "slowest time constant")]),
    C("README_v1", "sizing_feed_switch", "the feed switch loses {} mol in 40 h and {} in 120 h",
      compute=lambda: [sizing("130 L", "feed_switch open_loop lost_excess_mol"),
                       sizing("130 L, 120 h", "feed_switch open_loop lost_excess_mol")]),
    C("README_v1", "prnd_free", "(the model: {} % at 58 stages, {} % at 64)", fmt=decimals(1),
      compute=lambda: [percent(best_recovery(STAGES_PRND_FREE, 58)),
                       percent(best_recovery(STAGES_PRND_FREE, 64))]),
    C("README_v1", "prnd_capped", "the same plant sits at {}–{} %", compute=lambda: [
        percent(best_recovery(STAGES_PRND, 58)), percent(best_recovery(STAGES_PRND, 64))]),
    C("README_v1", "drift_interface_top", "it climbs to {} m", fmt=decimals(2),
      **disturbance("interface_drift"), column="interface_max_m"),

    # ------------------------------------------------------------ REPORT
    *[C("REPORT", f"stages_{n}", template, compute=lambda n=n: percent(best_recovery(STAGES, n)))
      for n, template in [(12, "Twelve stages give {} %"), (16, "sixteen give {} %"),
                          (20, "twenty give {} %"), (22, "twenty-two give {} %"),
                          (26, "twenty-six give {} %"), (32, "thirty-two give {} %")]],
    C("REPORT", "reflux_advantage", "beats an acid-only scrub by up to {} points",
      fmt=in_words, compute=reflux_advantage),
    C("REPORT", "stages_ideal",
      "the same sweep gives {} % at twelve stages, {} % at sixteen, {} % at twenty, {} % at thirty-two",
      fmt=(whole, whole, whole, decimals(1)),
      compute=lambda: [percent(best_recovery(STAGES_IDEAL, n)) for n in (12, 16, 20, 32)]),
    C("REPORT", "design_point", "loading cap — {} mol/h of caustic and acid, with {} L/h of organic",
      compute=lambda: [cell(TRIM, "design", handle="reagent_mol_per_h"),
                                             cell(TRIM, "design", handle="Q_org")]),
    C("REPORT", "trim_scrub", "more scrub flow ({} L/h against {}) at lower acidity ({} M against {})",
      fmt=(whole, whole, decimals(1), decimals(1)), compute=lambda: [
          cell(TRIM, "trimmed", handle="Q_scrub"), cell(TRIM, "design", handle="Q_scrub"),
          cell(TRIM, "trimmed", handle="h_scrub"), cell(TRIM, "design", handle="h_scrub")]),
    C("REPORT", "trim_reagent", "a little less organic, {} mol/h of reagent",
      table=TRIM, where={"handle": "reagent_mol_per_h"}, column="trimmed"),
    C("REPORT", "60h_feed_switch", "the feed switch sends {} mol of NdPr beyond the nominal 5 %",
      fmt=nearest(10), **disturbance("feed_switch"), column="lost_excess_mol"),
    C("REPORT", "60h_others", "the strong scrub acid {}, the feed acid {}, the stronger feed {}.",
      fmt=nearest(10), compute=lambda: [cell(DIST, "lost_excess_mol", scenario=s)
                                        for s in ("reflux_acid", "feed_acid", "feed_step")]),
    C("REPORT", "60h_flowmeter", "takes purity down to {} % and {} mol of product off spec",
      fmt=(decimals(1), nearest(10)), compute=lambda: [
          percent(cell(DIST, "purity_min", scenario="flowmeter_bias")),
          cell(DIST, "off_spec_mol", scenario="flowmeter_bias")]),
    C("REPORT", "60h_pump_stop", "stop takes purity to {} % and one interface", fmt=decimals(1),
      **disturbance("feed_interrupt"), column="purity_min", transform=percent),
    C("REPORT", "60h_drift", "about {} metre-hours of it",
      **disturbance("interface_drift"), column="interface_excursion_m_h"),
    C("REPORT", "40h_losses",
      "feed switch {} to {} mol, strong scrub acid {} to {}, feed acid {} to {}, stronger feed {} to {}.",
      fmt=(whole, nothing_below_zero, nearest(10), nothing_below_zero, nearest(10), whole,
           whole, whole),
      compute=lambda: [cell(CAT, column, scenario=s)
                       for s in ("feed_switch", "reflux_acid", "feed_acid", "feed_step")
                       for column in ("lost_excess_mol_open", "lost_excess_mol_mpc")]),
    C("REPORT", "40h_flowmeter",
      "costs purity {} % at worst and {} mol of off-spec product uncontrolled, {} % and nothing off spec",
      fmt=(decimals(1), nearest(100), decimals(1)), compute=lambda: [
          percent(cell(CAT, "purity_min_open", scenario="flowmeter_bias")),
          cell(CAT, "off_spec_mol_open", scenario="flowmeter_bias"),
          percent(cell(CAT, "purity_min_mpc", scenario="flowmeter_bias"))]),
    C("REPORT", "40h_drift", "{} metre-hours outside its window uncontrolled, {} controlled",
      fmt=(whole, decimals(2)), **catalogue("interface_drift"),
      column=("interface_excursion_m_h_open", "interface_excursion_m_h_mpc")),
    C("REPORT", "40h_reagent", "Reagent moves by at most {} % either way on all eight",
      compute=largest_reagent_move),
    C("REPORT", "40h_pump_stop_purity", "purity dips to {} % against {} % uncontrolled",
      fmt=decimals(1), **catalogue("feed_interrupt"), column=("purity_min_mpc", "purity_min_open"),
      transform=percent),
    C("REPORT", "40h_pump_stop_offspec", "{} goes off spec either way", fmt=nothing_if_zero,
      compute=lambda: max(cell(CAT, "off_spec_mol_open", scenario="feed_interrupt"),
                          cell(CAT, "off_spec_mol_mpc", scenario="feed_interrupt"))),
    C("REPORT", "40h_pump_stop_loss", "and it loses {} mol of NdPr where the uncontrolled circuit loses none",
      **catalogue("feed_interrupt"), column="lost_excess_mol_mpc"),
    C("REPORT", "40h_feed_switch_dip", "lets purity dip to {} % for a few hours", fmt=decimals(2),
      **catalogue("feed_switch"), column="purity_min_mpc", transform=percent),
    C("REPORT", "40h_feed_switch_reagent", "with {} % less reagent; it ends", fmt=decimals(1),
      compute=lambda: -percent_more(cell(CAT, "reagent_mol_mpc", scenario="feed_switch"),
                                    cell(CAT, "reagent_mol_open", scenario="feed_switch"))),
    C("REPORT", "sweep_entrainment",
      "On the entrainment plant the controller lost about {} mol of NdPr at 0.0002 and {} at 0.005, and at most {} at the other weights",
      fmt=nearest(10), compute=lambda: [
          sweep(0.0002, "entrainment", "feed_interrupt", "lost_excess_mol"),
          sweep(0.005, "entrainment", "feed_interrupt", "lost_excess_mol"),
          max(sweep(w, "entrainment", "feed_interrupt", "lost_excess_mol")
              for w in (0.0, 0.001, 0.02))]),
    C("REPORT", "sweep_matched", "on the matched plant it put {} mol off spec at 0.005",
      fmt=nearest(10), compute=lambda: sweep(0.005, "matched", "feed_interrupt", "off_spec_mol")),
    C("REPORT", "seed_pump_stop_5", "it is better than doing nothing on {} of 15 and worse on {}",
      fmt=plain, compute=lambda: [seed_better("feed_interrupt"), seed_worse("feed_interrupt")]),
    C("REPORT", "40h_feed_switch_price", "a few hours of {} % product cost against {} mol of NdPr",
      fmt=(decimals(1), whole), compute=lambda: [
          percent(cell(CAT, "purity_min_mpc", scenario="feed_switch")),
          cell(CAT, "lost_excess_mol_open", scenario="feed_switch")]),
    C("REPORT", "trim_settles", "(every run starts from {} %)", fmt=decimals(2),
      **catalogue("interface_drift"), column="purity_min_open", transform=percent),
    C("REPORT", "trim_cost", "at {} % more reagent than the design",
      fmt=floored, compute=trim_reagent_increase),
    C("REPORT", "trim_against_optimum", "is {} % more; the difference, about {} % of the reagent bill",
      fmt=(decimals(1), whole), compute=lambda: [
          optimum_reagent_increase(), trim_reagent_increase() - optimum_reagent_increase()]),
    C("REPORT", "trim_matched", "the trim moves nothing ({} to {} mol/h)", fmt=plain,
      table=TRIM_MATCHED, where={"handle": "reagent_mol_per_h"}, column=("design", "trimmed")),
    C("REPORT", "matched_start", "with purity at {} % and no margin", fmt=decimals(2),
      **catalogue("interface_drift", matched=True), column="purity_min_open", transform=percent),
    C("REPORT", "matched_feed_switch", "dip to {} %, {} mol off spec, and spends {} % more reagent",
      fmt=(decimals(2), whole, whole), compute=lambda: [
          percent(cell(CAT_MATCHED, "purity_min_mpc", scenario="feed_switch")),
          cell(CAT_MATCHED, "off_spec_mol_mpc", scenario="feed_switch"),
          percent_more(cell(CAT_MATCHED, "reagent_mol_mpc", scenario="feed_switch"),
                       cell(CAT_MATCHED, "reagent_mol_open", scenario="feed_switch"))]),
    C("REPORT", "perturbed_start", "the trimmed perturbed plant starts at {} %", fmt=decimals(2),
      **catalogue("interface_drift"), column="purity_min_open", transform=percent),
    C("REPORT", "perturbed_feed_switch_reagent", "stays on spec with {} % less reagent",
      fmt=decimals(1),
      compute=lambda: -percent_more(cell(CAT, "reagent_mol_mpc", scenario="feed_switch"),
                                    cell(CAT, "reagent_mol_open", scenario="feed_switch"))),
    *matrix_row_claims("REPORT"),
    C("REPORT", "matrix_pump_stop_loss", "loses at most {} mol of NdPr",
      compute=matrix_pump_stop_worst_loss),
    C("REPORT", "fixed_mpc_feed_switch", "{} to {} mol off spec on a feed switch",
      compute=fixed_mpc_feed_switch_range),
    C("REPORT", "seed_cannot", "optimizer told their true numbers, is {} to {} %",
      compute=best_recovery_of_seeds_that_cannot),
    C("REPORT", "seed3_trim", "(seed 3 at {} % purity)",
      compute=lambda: percent(ranked("plain", 3, "purity_start"))),
    C("REPORT", "seed_hard_medians", "(medians {} to {} mol uncontrolled, {} to {} controlled)",
      fmt=nearest(10), compute=seed_hard_medians),
    C("REPORT", "seed_hard_better", "it's better on {} or {} of the 5 each time", fmt=plain,
      compute=seed_hard_better),
    C("REPORT", "seed_hard_failures", "Now {} of the 3,200 steps on the five plants fail, {} of them on seed 17",
      fmt=plain, compute=seed_hard_failures),
    C("REPORT", "ranked_seed1_trim", "seed 1's trim ended at {} %, short of the 99.5 %",
      fmt=decimals(2), compute=lambda: percent(ranked("two-stage", 1, "purity_start"))),
    C("REPORT", "ranked_seed2",
      "seed 2 fell to {} % purity with {} mol off spec, against {} % and {} mol without it",
      fmt=(whole, whole, decimals(1), whole), compute=lambda: [
          percent(ranked("two-stage", 2, "purity_min")), ranked("two-stage", 2, "off_spec_mol"),
          percent(ranked("plain", 2, "purity_min")), ranked("plain", 2, "off_spec_mol")]),
    C("REPORT", "ranked_hard",
      "helped seed 13 ({} % purity at the end against {} %) and hurt seed 3 ({} % against {} %)",
      fmt=(decimals(1), decimals(1), whole, whole), compute=lambda: [
          percent(ranked("two-stage", 13, "purity_end")), percent(ranked("plain", 13, "purity_end")),
          percent(ranked("two-stage", 3, "purity_end")), percent(ranked("plain", 3, "purity_end"))]),
    *[C("REPORT", f"seed_row_{upset}", f"| {label} | {{}} | {{}} | {{}} | {{}} of 15 |",
        fmt=(whole, whole, whole, plain), compute=lambda upset=upset: seed_row(upset))
      for upset, label in [("feed_switch", "feed switch"), ("feed_acid", "feed acid"),
                           ("feed_step", "stronger feed"), ("reflux_acid", "strong scrub acid"),
                           ("flowmeter_bias", "flowmeter bias"), ("feed_interrupt", "pump stop"),
                           ("interface_drift", "interface drift"), ("efficiency_loss", "slow mixers")]],
    C("REPORT", "seed_flowmeter_clean", "the uncontrolled plant puts nothing off spec on {} of the 15",
      fmt=plain, compute=lambda: seed_open_loop_clean("flowmeter_bias")),
    C("REPORT", "seed_drift", "against {} to {} metre-hours uncontrolled",
      compute=seed_drift_open_loop_interfaces),
    C("REPORT", "seed_pump_stop", "better on {} of 15, worse on {}, up to {} mol",
      fmt=(plain, plain, whole), compute=lambda: [
          seed_better("feed_interrupt"), seed_worse("feed_interrupt"),
          seed_cost("feed_interrupt", "mpc").max()]),
    C("REPORT", "sizing_best_at_32", "the best recovery at 99 % purity is {} % against {} %",
      fmt=decimals(1), compute=lambda: [percent(best_recovery(STAGES_SIZED, 32)),
                                        percent(best_recovery(STAGES, 32))]),
    C("REPORT", "sizing_nonlinearity",
      "purity differs by {} to {} % across the four handles, against {} to {} %",
      compute=lambda: purity_nonlinearity_range("130 L") + purity_nonlinearity_range("30 L")),
    C("REPORT", "sizing_time_constant", "slowest time constant is {} hours against {}",
      compute=lambda: [sizing("130 L", "slowest time constant"),
                       sizing("30 L", "slowest time constant")]),
    C("REPORT", "sizing_feed_switch", "feed switch loses {} mol in 40 hours and {} in 120",
      compute=lambda: [sizing("130 L", "feed_switch open_loop lost_excess_mol"),
                       sizing("130 L, 120 h", "feed_switch open_loop lost_excess_mol")]),
    C("REPORT", "offspec_215", "so {} against {} between two similar runs", compute=lambda: [
        cell(MATRIX, "off_spec_mol", plant="perturbed", controller="mpc", upset="feed_switch"),
        cell(MATRIX, "off_spec_mol", plant="matched", controller="mpc", upset="feed_switch")]),
    C("REPORT", "placement_stage", "reading the aqueous at stage {}, late in",
      plain, compute=lambda: int(placement("best1"))),
    C("REPORT", "placement_sd", "pins product purity to {} of a thousandth",
      fmt=tenth_in_words, compute=lambda: 1000 * best_single_sd()),
    C("REPORT", "placement_second", "A second analyser improves that by {} %", compute=lambda: 100 * (
        1 - float(placement("sd_pair")[int(placement("best2"))]) / best_single_sd())),
    C("REPORT", "placement_ecjtu", "is {} % worse than the best single analyser",
      compute=lambda: percent_more(float(placement("sd_ecjtu")), best_single_sd())),
    C("REPORT", "placement_lab", "the product alone is {} as bad",
      fmt=times_in_words, compute=lambda: float(placement("sd_lab")) / best_single_sd()),
    C("REPORT", "plant_placement_late", "is off by about {} thousandths rms",
      fmt=in_words, compute=error_from_late_scrub_to_product),
    C("REPORT", "plant_placement_detail",
      "(best at stage {} with {}, the model's choice at {} with {}, the product stream itself {})",
      fmt=(plain, decimals(1), plain, decimals(1), decimals(1)), compute=lambda: [
          best_placement_on_plant(), placement_error(best_placement_on_plant()),
          int(placement("best1")), placement_error(int(placement("best1"))),
          placement_error(PRODUCT_STREAM_STAGE)]),
    C("REPORT", "plant_placement_others",
      "{} thousandths from an analyser in the extraction section, {} with the patent's pair, and {} with the lab assay alone",
      fmt=(in_words, decimals(1), whole), compute=lambda: [
          error_in_extraction(), 1000 * float(placement("err_ecjtu")),
          1000 * float(placement("err_lab"))]),
    C("REPORT", "sensors_feed_set", "analysers: {} mol against {} uncontrolled",
      fmt=(with_minus_sign, whole), compute=lambda: [
          study("product + raffinate + feed analysers", "feed_switch", "lost_excess_mol"),
          study("open loop", "feed_switch", "lost_excess_mol")]),
    C("REPORT", "sensors_pair", "without a feed analyser ({} mol)",
      compute=lambda: study("ECJTU pair", "feed_switch", "lost_excess_mol")),
    C("REPORT", "sensors_pair_feed", "{} mol lost and {} mol off spec, purity down to {} %",
      fmt=(whole, nearest(100), decimals(1)), compute=lambda: [
          study("ECJTU pair + feed analyser", "feed_switch", c)
          if c != "purity_min" else percent(study("ECJTU pair + feed analyser", "feed_switch", c))
          for c in ("lost_excess_mol", "off_spec_mol", "purity_min")]),
    C("REPORT", "sensors_product_late", "see the switch late, {} mol", fmt=nearest(10),
      compute=lambda: study("product + raffinate analysers", "feed_switch", "lost_excess_mol")),
    C("REPORT", "sensors_lab_nothing", "loses {}; the controller with no readings at all, {},",
      fmt=nearest(10), compute=lambda: [study("lab assay only", "feed_switch", "lost_excess_mol"),
                                        study("nothing", "feed_switch", "lost_excess_mol")]),
    C("REPORT", "sensors_flowmeter_open", "purity falls to {} % and about {} mol go off spec",
      fmt=(decimals(1), nearest(100)), compute=lambda: [
          percent(study("open loop", "flowmeter_bias", "purity_min")),
          study("open loop", "flowmeter_bias", "off_spec_mol")]),
    C("REPORT", "sensors_pair_flowmeter", "The mid-cascade pair alone lets purity fall to {} % with {} mol off spec",
      fmt=(decimals(1), nearest(10)), compute=lambda: [
          percent(study("ECJTU pair", "flowmeter_bias", "purity_min")),
          study("ECJTU pair", "flowmeter_bias", "off_spec_mol")]),
    C("REPORT", "sensors_interfaces_flowmeter", "with interface readings added, to {} % with {} mol",
      fmt=(decimals(1), nearest(10)), compute=lambda: [
          percent(study("pair + interfaces", "flowmeter_bias", "purity_min")),
          study("pair + interfaces", "flowmeter_bias", "off_spec_mol")]),
    C("REPORT", "sensors_flowmeter", "any kind holds purity above {} %",
      fmt=floored_to(1), compute=lowest_purity_with_an_instrument),
    C("REPORT", "sensors_full_pump_stop", "With the full state the controller puts {} off spec either",
      fmt=nothing_if_zero, compute=lambda: study("full state", "feed_interrupt", "off_spec_mol")),
    C("REPORT", "sensors_pump_stop_clean",
      "So do {} of the six sensor sets that read anything, losing {} to {} mol of NdPr",
      fmt=(in_words, whole, whole),
      compute=lambda: pump_stop_sets_with_nothing_off_spec(full_state=False)),
    C("REPORT", "sensors_product_pump_stop",
      "purity down to {} % with {} mol off spec and interfaces out for about {} metre-hours",
      fmt=(whole, nearest(10), in_words), compute=lambda: [
          percent(study("product + raffinate analysers", "feed_interrupt", "purity_min")),
          study("product + raffinate analysers", "feed_interrupt", "off_spec_mol"),
          study("product + raffinate analysers", "feed_interrupt", "interface_excursion_m_h")]),
    C("REPORT", "sensors_nothing_pump_stop", "No readings at all puts {} mol off spec",
      fmt=nearest(10), compute=lambda: study("nothing", "feed_interrupt", "off_spec_mol")),
    C("REPORT", "sensors_interfaces_feed_switch",
      "the interfaces spend about {} metre-hours outside their window and {} of the 80 solves fail",
      fmt=(in_words, plain), compute=lambda: [
          study("pair + interfaces", "feed_switch", "interface_excursion_m_h"),
          study("pair + interfaces", "feed_switch", "qp_failures")]),
    C("REPORT", "handles_organic", "organic flow alone holds recovery at {} % more reagent",
      compute=lambda: reagent_change("feed_switch", "organic flow")),
    C("REPORT", "handles_ecjtu", "the patent's two, at {} %",
      compute=lambda: reagent_change("feed_switch", "organic + scrub flow (ECJTU)")),
    C("REPORT", "handles_sapon", "add saponification and it drops to {} %, with {} failed solves",
      fmt=(whole, in_words), compute=lambda: [
          reagent_change("feed_switch", "+ saponification"),
          study("+ saponification", "feed_switch", "qp_failures")]),
    C("REPORT", "handles_acidity", "add scrub acidity, {} %;",
      compute=lambda: reagent_change("feed_switch", "+ scrub acidity")),
    C("REPORT", "handles_full", "it uses {} % less reagent than doing nothing", fmt=decimals(1),
      compute=lambda: -reagent_change("feed_switch", "+ jacklegs (all)")),
    C("REPORT", "handles_organic_pump_stop", "organic flow alone holds purity at {} %",
      fmt=decimals(1), compute=lambda: percent(study("organic flow", "feed_interrupt", "purity_min"))),
    C("REPORT", "handles_more_pump_stop", "add saponification and {} mol go off spec, add scrub acidity {},",
      fmt=nearest(10), compute=lambda: [
          study("+ saponification", "feed_interrupt", "off_spec_mol"),
          study("+ scrub acidity", "feed_interrupt", "off_spec_mol")]),
    C("REPORT", "handles_full_pump_stop", "add scrub acidity {}, the full set {}.", fmt=nearest(10),
      compute=lambda: [study("+ scrub acidity", "feed_interrupt", "off_spec_mol"),
                       study("+ jacklegs (all)", "feed_interrupt", "off_spec_mol")]),
    *rl_table_claims("REPORT"),
    C("REPORT", "rl_interfaces", "(at most {} metre-hours outside the window)", fmt=decimals(2),
      compute=lambda: float(table(RL)["interface_excursion_m_h_rl"].max())),
    C("REPORT", "rl_offspec", "holds purity on all five episodes, with {} off spec",
      fmt=nothing_if_zero, compute=lambda: float(table(RL)["off_spec_mol_rl"].max())),
    C("REPORT", "rl_ratio", "loses about {} times more NdPr", fmt=in_words, compute=rl_mean_ratio),
    C("REPORT", "rl_no_feed_switch", "it loses {} mol where the controller loses none ({})",
      fmt=(whole, with_minus_sign), **episode(0),
      column=("lost_excess_mol_rl", "lost_excess_mol_mpc")),
    C("REPORT", "prnd_free", "the model gives {} % at 58 stages and {} % at 64", fmt=decimals(1),
      compute=lambda: [percent(best_recovery(STAGES_PRND_FREE, 58)),
                       percent(best_recovery(STAGES_PRND_FREE, 64))]),
    C("REPORT", "prnd_capped", "the same sweep moves: {} % at 58 stages, {} % at 64, {} % at 78",
      compute=lambda: [percent(best_recovery(STAGES_PRND, n)) for n in (58, 64, 78)]),
    C("REPORT", "drift_interface_top", "it climbs to {} m", fmt=decimals(2),
      **disturbance("interface_drift"), column="interface_max_m"),

    # ------------------------------------------------------------ ARCHITECTURE, decisions 19-29
    C("ARCH", "d23_pair_feed_switch",
      "the ECJTU pair holds the feed switch nearly as well as the product, raffinate and feed analysers do ({} mol lost",
      compute=lambda: study("ECJTU pair", "feed_switch", "lost_excess_mol")),
    C("ARCH", "d23_open_loop", "mol lost against {} open loop)",
      compute=lambda: study("open loop", "feed_switch", "lost_excess_mol")),
    C("ARCH", "d23_pair_flowmeter", "but not a flowmeter bias ({} %, {} mol off spec)",
      fmt=(decimals(1), nearest(10)), compute=lambda: [
          percent(study("ECJTU pair", "flowmeter_bias", "purity_min")),
          study("ECJTU pair", "flowmeter_bias", "off_spec_mol")]),
    C("ARCH", "d23_product_late", "switch late without a feed analyser ({} mol)", fmt=nearest(10),
      compute=lambda: study("product + raffinate analysers", "feed_switch", "lost_excess_mol")),
    C("ARCH", "d23_lab", "a lab assay alone loses {}.", fmt=nearest(10),
      compute=lambda: study("lab assay only", "feed_switch", "lost_excess_mol")),
    C("ARCH", "d23_placement_model", "(stage {} counted from 0, {} thousandths)",
      fmt=(plain, decimals(2)), compute=lambda: [int(placement("best1")), 1000 * best_single_sd()]),
    C("ARCH", "d23_placement_plant", "about {} thousandths anywhere from stage",
      fmt=in_words, compute=error_from_late_scrub_to_product),
    C("ARCH", "d23_placement_detail",
      "(best {} at {}, the model's choice {}, the product stream {}), the ECJTU pair {}, the lab assay alone {}.",
      fmt=(plain, decimals(1), decimals(1), decimals(1), decimals(1), whole), compute=lambda: [
          best_placement_on_plant(), placement_error(best_placement_on_plant()),
          placement_error(int(placement("best1"))), placement_error(PRODUCT_STREAM_STAGE),
          1000 * float(placement("err_ecjtu")), 1000 * float(placement("err_lab"))]),
    C("ARCH", "d25_pump_stop_loss", "lost at most {} mol of NdPr",
      compute=matrix_pump_stop_worst_loss),
    C("ARCH", "d25_fixed_mpc_feed_switch", "{} to {} mol off spec on a feed switch",
      compute=fixed_mpc_feed_switch_range),
    C("ARCH", "d26_trim_reagent", "(reagent 2,712 → {} mol/h)",
      table=TRIM, where={"handle": "reagent_mol_per_h"}, column="trimmed"),
    C("ARCH", "d26_sweep_matched", "the matched plant puts {} mol off spec at 0.005",
      fmt=nearest(10), compute=lambda: sweep(0.005, "matched", "feed_interrupt", "off_spec_mol")),
    C("ARCH", "d26_sweep_entrainment",
      "the entrainment plant loses {} mol of NdPr at 0.0002 and {} at 0.005, and at most {} at the others",
      fmt=(whole, nearest(10), nearest(10)), compute=lambda: [
          sweep(0.0002, "entrainment", "feed_interrupt", "lost_excess_mol"),
          sweep(0.005, "entrainment", "feed_interrupt", "lost_excess_mol"),
          max(sweep(w, "entrainment", "feed_interrupt", "lost_excess_mol")
              for w in (0.0, 0.001, 0.02))]),
    C("ARCH", "d28_best_at_32", "with 32 stages is {} % against {} %", fmt=decimals(1),
      compute=lambda: [percent(best_recovery(STAGES_SIZED, 32)), percent(best_recovery(STAGES, 32))]),
    C("ARCH", "d28_nonlinearity", "purity nonlinearity {}–{} % against {}–{} %",
      compute=lambda: purity_nonlinearity_range("130 L") + purity_nonlinearity_range("30 L")),
    C("ARCH", "d28_time_constant", "slowest time constant {} h against {} h", compute=lambda: [
        sizing("130 L", "slowest time constant"), sizing("30 L", "slowest time constant")]),
    C("ARCH", "d28_feed_switch", "the feed switch loses {} mol in 40 h and {} in 120 h",
      compute=lambda: [sizing("130 L", "feed_switch open_loop lost_excess_mol"),
                       sizing("130 L, 120 h", "feed_switch open_loop lost_excess_mol")]),
    C("ARCH", "d29_cannot", "(best {}–{} %)", compute=best_recovery_of_seeds_that_cannot),
    C("ARCH", "d29_feed_switch", "on the feed switch on {} of 15 (median {} → {} mol)",
      fmt=(plain, whole, whole), compute=lambda: [
          seed_better("feed_switch"), seed_cost("feed_switch", "open_loop").median(),
          seed_cost("feed_switch", "mpc").median()]),
    C("ARCH", "d29_others",
      "the feed acid {}, the stronger feed {}, the strong scrub acid {} and the flowmeter bias {} (open loop puts nothing off spec on the other {})",
      fmt=plain, compute=lambda: [seed_better(u) for u in
                                  ("feed_acid", "feed_step", "reflux_acid", "flowmeter_bias")]
      + [seed_open_loop_clean("flowmeter_bias")]),
    C("ARCH", "d29_pump_stop", "better on {} and worse on {}, worst {} mol",
      fmt=(plain, plain, whole), compute=lambda: [
          seed_better("feed_interrupt"), seed_worse("feed_interrupt"),
          seed_cost("feed_interrupt", "mpc").max()]),
    C("ARCH", "d29_drift", "against {}–{} m·h open loop", compute=seed_drift_open_loop_interfaces),
    C("ARCH", "d29_seed3_trim", "(seed 3 at {} % purity)",
      compute=lambda: percent(ranked("plain", 3, "purity_start"))),
    C("ARCH", "d29_hard_medians", "(medians {}–{} mol open loop, {}–{} controlled)",
      fmt=nearest(10), compute=seed_hard_medians),
    C("ARCH", "d29_hard_better", "better on {} or {} of the 5 each time", fmt=plain,
      compute=seed_hard_better),
    C("ARCH", "d30_failures", "On the five plants {} of 3,200 steps still fail, {} of them on seed 17",
      fmt=plain, compute=seed_hard_failures),
    C("ARCH", "d30_pump_stop", "on seed 1 the pump stop now puts {} off spec",
      fmt=nothing_if_zero, **catalogue("feed_interrupt"), column="off_spec_mol_mpc"),
    C("ARCH", "d30_seed_pump_stop", "the controller is better on it on {} of 15 and worse on {}",
      fmt=plain, compute=lambda: [seed_better("feed_interrupt"), seed_worse("feed_interrupt")]),
    C("ARCH", "d30_pair_feed", "the ECJTU pair with a feed analyser now puts {} mol off spec on the feed switch",
      fmt=nearest(100), compute=lambda: study("ECJTU pair + feed analyser", "feed_switch", "off_spec_mol")),
    C("ARCH", "d30_product_pump_stop", "run away on the pump stop ({} mol off spec)",
      fmt=nearest(10), compute=lambda: study("product + raffinate analysers", "feed_interrupt", "off_spec_mol")),
    C("ARCH", "d30_seed1_trim", "seed 1's trim ended at {} %, short of the 99.5 % target",
      fmt=decimals(2), compute=lambda: percent(ranked("two-stage", 1, "purity_start"))),
    C("ARCH", "d30_seed2",
      "seed 2 fell to {} % purity on the feed switch with {} mol off spec, against {} % and {} without it",
      fmt=(whole, whole, decimals(1), whole), compute=lambda: [
          percent(ranked("two-stage", 2, "purity_min")), ranked("two-stage", 2, "off_spec_mol"),
          percent(ranked("plain", 2, "purity_min")), ranked("plain", 2, "off_spec_mol")]),
    C("ARCH", "d30_hard",
      "helped seed 13 ({} % purity at the end against {} %) and hurt seed 3 ({} % against {} %)",
      fmt=(decimals(1), decimals(1), whole, whole), compute=lambda: [
          percent(ranked("two-stage", 13, "purity_end")), percent(ranked("plain", 13, "purity_end")),
          percent(ranked("two-stage", 3, "purity_end")), percent(ranked("plain", 3, "purity_end"))]),

    # ------------------------------------------------------------ APPENDIX B
    C("APPB", "b9_trim_cost", "at {} % more reagent than the design",
      fmt=floored, compute=trim_reagent_increase),
    C("APPB", "b9_optimum", "is {} % more, so the rest", fmt=decimals(1),
      compute=optimum_reagent_increase),
    C("APPB", "b10_offspec_215", "{} against 0 mol can be a small purity difference",
      **matrix("matched", "mpc", "feed_switch"), column="off_spec_mol"),
    C("APPB", "b10_placement_shift", "the same shape, best {} stage later", fmt=in_words,
      compute=lambda: best_placement_on_plant() - int(placement("best1"))),
    C("APPB", "b10_1_seeds",
      "the strong scrub acid on {}, the flowmeter bias on {} (open loop clean on the other {}); the pump stop is worse on {}",
      fmt=plain, compute=lambda: [seed_better("reflux_acid"), seed_better("flowmeter_bias"),
                                  seed_open_loop_clean("flowmeter_bias"), seed_worse("feed_interrupt")]),
    C("APPB", "b10_6_seed1", "({} off spec, {} mol lost; REPORT.md 5)",
      fmt=(nothing_if_zero, whole), **catalogue("feed_interrupt"),
      column=("off_spec_mol_mpc", "lost_excess_mol_mpc")),
    C("APPB", "b10_6_sweep", "about {} mol at another", fmt=nearest(10),
      compute=lambda: max(sweep(w, "entrainment", "feed_interrupt", "lost_excess_mol")
                          for w in (0.0, 0.0002, 0.001, 0.005, 0.02))),
    C("APPB", "b10_6_seeds", "the controller is worse than open loop on {} of 15", fmt=plain,
      compute=lambda: seed_worse("feed_interrupt")),
    C("APPB", "b10_16_seed2", "seed 2 down to {} % on the feed switch",
      compute=lambda: percent(ranked("two-stage", 2, "purity_min"))),
    C("APPB", "b10_16_seed3", "(seed 3 trims to {} %)",
      compute=lambda: percent(ranked("plain", 3, "purity_start"))),

    # ------------------------------------------------------------ PROGRESS, status table
    C("PROGRESS", "stages_32", "32 give {} %", compute=lambda: percent(best_recovery(STAGES, 32))),
    C("PROGRESS", "prnd_free_60", "(60 stages → {} %)",
      compute=lambda: percent(recovery_at(STAGES_PRND_FREE, 60))),
    C("PROGRESS", "prnd_capped_60", "the same plant sits at {} %",
      compute=lambda: percent(recovery_at(STAGES_PRND, 60))),
    C("PROGRESS", "mpc_summary",
      "feed switch {} → {} mol lost, strong scrub acid {} → {}, flowmeter bias {} mol off spec → {}, drift {} → {} m·h",
      fmt=(whole, nothing_below_zero, nearest(10), nothing_below_zero, nearest(100), whole,
           whole, decimals(2)), compute=lambda: [
          cell(CAT, "lost_excess_mol_open", scenario="feed_switch"),
          cell(CAT, "lost_excess_mol_mpc", scenario="feed_switch"),
          cell(CAT, "lost_excess_mol_open", scenario="reflux_acid"),
          cell(CAT, "lost_excess_mol_mpc", scenario="reflux_acid"),
          cell(CAT, "off_spec_mol_open", scenario="flowmeter_bias"),
          cell(CAT, "off_spec_mol_mpc", scenario="flowmeter_bias"),
          cell(CAT, "interface_excursion_m_h_open", scenario="interface_drift"),
          cell(CAT, "interface_excursion_m_h_mpc", scenario="interface_drift")]),
    C("PROGRESS", "mpc_pump_stop", "({} mol off spec against none, {} mol lost)",
      fmt=(nearest(10), whole), **catalogue("feed_interrupt"),
      column=("off_spec_mol_mpc", "lost_excess_mol_mpc")),
    C("PROGRESS", "rl_ratio", "~{}× more NdPr lost", compute=rl_mean_ratio),
    C("PROGRESS", "seed_status",
      "strong scrub acid {}, flowmeter bias {} (open loop clean on {}), pump stop better on {} and worse on {}",
      fmt=plain, compute=lambda: [seed_better("reflux_acid"), seed_better("flowmeter_bias"),
                                  seed_open_loop_clean("flowmeter_bias"),
                                  seed_better("feed_interrupt"), seed_worse("feed_interrupt")]),
    # ---- the Oct 14 evening log entry (decision 30)
    C("PROGRESS", "d30_failures", "On the five plants {} of 3,200 steps still fail, {} on seed 17",
      fmt=plain, compute=seed_hard_failures),
    C("PROGRESS", "d30_seed2", "(seed 2 down to {} % purity on the feed switch, {} mol off spec against {};",
      compute=lambda: [percent(ranked("two-stage", 2, "purity_min")),
                       ranked("two-stage", 2, "off_spec_mol"), ranked("plain", 2, "off_spec_mol")]),
    C("PROGRESS", "d30_seed1_trim", "seed 1's trim ended at {} %, short of target)", fmt=decimals(2),
      compute=lambda: percent(ranked("two-stage", 1, "purity_start"))),
    C("PROGRESS", "d30_hard_medians", "(medians {}–{} mol open loop, {}–{} controlled)",
      fmt=nearest(10), compute=seed_hard_medians),
    C("PROGRESS", "d30_easy", "(strong scrub acid better on {}, was 13; pump stop better on {} and worse on {}, worst {} mol",
      fmt=(plain, plain, plain, whole), compute=lambda: [
          seed_better("reflux_acid"), seed_better("feed_interrupt"), seed_worse("feed_interrupt"),
          seed_cost("feed_interrupt", "mpc").max()]),
    C("PROGRESS", "d30_pair_feed", "the ECJTU pair with a feed analyser now puts {} mol off spec on the feed switch",
      fmt=nearest(100), compute=lambda: study("ECJTU pair + feed analyser", "feed_switch", "off_spec_mol")),
    C("PROGRESS", "d30_product_pump_stop", "pump stop (purity {} %, {} mol off spec)",
      fmt=(whole, nearest(10)), compute=lambda: [
          percent(study("product + raffinate analysers", "feed_interrupt", "purity_min")),
          study("product + raffinate analysers", "feed_interrupt", "off_spec_mol")]),
    C("PROGRESS", "d30_reagent", "became \"at most {} %\"", compute=largest_reagent_move),
    C("PROGRESS", "d30_seed3_trim", "Seed 3's trim now ends at {} %",
      compute=lambda: percent(ranked("plain", 3, "purity_start"))),

    # the README, about the main case (from 2026-10-17)
    C("README", "main_window_edge", "the band that keeps the product on spec is about {} % wide",
      compute=lambda: margin_row("8+20_130L_edge")["median_window_pct"]),
    C("README", "main_window_ratio", "made the band about {} times wider", fmt=plain,
      compute=lambda: margin_row("8+20_130L_margin")["median_window_pct"]
      / margin_row("8+20_130L_edge")["median_window_pct"]),
    C("README", "main_margin_reagent", "for about {} % more reagent",
      compute=lambda: percent_more(margin_row("8+20_130L_margin")["median_reagent"],
                                   margin_row("8+20_130L_edge")["median_reagent"])),
]


# ---------------------------------------------------------------- the test

def normalized(text):
    """Text with every run of spaces and line breaks made one space."""
    return re.sub(r"\s+", " ", text)


def what_the_document_says(template, text):
    """The numbers the document has where the template expects them, or None if the sentence is gone."""
    pattern = r"([\-−]?[\w.,~]+?)".join(re.escape(part) for part in template.split("{}"))
    found = re.search(pattern, text)
    return found.groups() if found else None


def claim_id(claim):
    """Readable test id: document and claim name."""
    return f"{claim.doc}:{claim.name}"


def as_param(claim):
    """A claim as a pytest parameter, marked xfail when the document disagrees today."""
    marks = [pytest.mark.xfail(reason=claim.xfail, strict=True)] if claim.xfail else []
    return pytest.param(claim, id=claim_id(claim), marks=marks)


def test_claim_names_are_unique():
    """Two claims with the same id would hide one another in the report."""
    ids = [claim_id(c) for c in CLAIMS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("claim", [as_param(c) for c in CLAIMS])
def test_document_quotes_its_table(claim):
    """The document contains the sentence with the number its table gives today."""
    path = (PROJECT / DOCS[claim.doc]).resolve()
    if not path.exists():
        pytest.skip(f"{DOCS[claim.doc]} is not here (the project folder is published on its own)")
    text = normalized(path.read_text(encoding="utf-8"))
    expected = claim.expected_text()
    if expected in text:
        return
    old = what_the_document_says(claim.template, text)
    new = expected
    if old is None:
        pytest.fail(f"{DOCS[claim.doc]}: no sentence like '{claim.template}' any more; "
                    f"the table gives '{new}'")
    pytest.fail(f"{DOCS[claim.doc]}: '{claim.template}' says {', '.join(old)}; "
                f"the table gives '{new}'")
