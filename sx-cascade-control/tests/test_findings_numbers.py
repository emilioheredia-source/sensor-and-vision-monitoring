"""Checks that the numbers FINDINGS_why_hard.md quotes in phases 8 and 9 still match the result tables.

Phases 8 and 9 (Oct 22 to 24) are the fair test of the physics fit against the learned estimators
(notebooks 43 to 45), the bias at a model swap (notebook 46), and whether the model's accuracy
changes what the plant does (notebooks 47 to 50). Every number below is recomputed from the file in
runs/ the way the text defines it (which plants count as "within noise", which ones are "the four
highest", ...), rounded the way the text rounds it, and the sentence must contain it. Same idea as
test_doc_numbers.py, whose helpers this uses.

Out of scope, checked by hand when the files change: the solver finding's 7.9e-4, 2.3e-4 and 1e-11
(logs of one-off runs, not tables), the closed-loop "within 0.05 points" of Test 3 (an inequality
over three files), the swap test's "within 0.006 points of purity at any moment" and "up to about 1
point of recovery" (traces), and the drift-mechanism table and the thread-count trims of phase 9
(printed by one-off scripts, not stored).
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from test_doc_numbers import normalized, what_the_document_says

PROJECT = Path(__file__).resolve().parent.parent
RUNS = PROJECT / "runs"
FINDINGS = PROJECT / "FINDINGS_why_hard.md"
SPEC = 0.99


# ---- reading the tables --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def rows(name: str) -> tuple:
    """All rows of runs/<name> as dicts (cached)."""
    with open(RUNS / name, newline="") as f:
        return tuple(csv.DictReader(f))


def col(name: str, column: str, **where) -> np.ndarray:
    """One column of the rows matching `where` (string comparison), as floats."""
    return np.array([float(r[column]) for r in rows(name) if all(r[k] == str(v) for k, v in where.items())])


def err(name: str, est: str, truth="beta_pl_true", **where) -> np.ndarray:
    """|estimate - truth| for each matching row."""
    return np.abs(col(name, est, **where) - col(name, truth, **where))


# ---- formats: how the text rounds -----------------------------------------------------------------
def f1(v):
    return f"{v:.1f}"


def f2(v):
    return f"{v:.2f}"


def f3(v):
    return f"{v:.3f}"


def f4(v):
    return f"{v:.4f}"


def pct1(v):
    return f"{100 * v:.1f}"


def pct2(v):
    return f"{100 * v:.2f}"


def whole(v):
    return f"{v:,.0f}"


def tens(v):
    return f"{round(v, -1):,.0f}"


def signed2(v):
    return f"{v:+.2f}"


def signed3(v):
    return f"{v:+.3f}"


# ---- the claims: (name, sentence with {} for each number, numbers, format) -----------------------
def claims_test1():
    T1, T1M = "fair_fit_test1_ndpr_main.csv", "fair_fit_test1_ndpr_main_multistart_noloop.csv"
    good = [r for r in rows(T1) if float(r["cost_fit"]) <= float(r["cost_truth"])]
    bad = [r for r in rows(T1) if float(r["cost_fit"]) > float(r["cost_truth"])]
    low = [float(r["beta_pl_true"]) - float(r["beta_pl_fit"]) for r in bad]
    top4 = sorted(rows(T1), key=lambda r: -float(r["beta_pl_true"]))[:4]
    net_low = [float(r["beta_pl_true"]) - float(r["beta_pl_net"]) for r in top4]
    s1 = next(r for r in rows(T1) if r["seed"] == "1")
    on_bound = sum(float(r["k_fit"]) >= np.exp(0.5) - 1e-3 for r in rows(T1))
    zm = max(max(abs(float(r["z_pl"])), abs(float(r["z_np"]))) for r in rows(T1M))
    below = [r for r in rows(T1) if float(r["true_purity_min"]) < SPEC]
    hours = [float(r[f"{e}_hours_below_spec"]) for r in below for e in ("true", "fit", "net", "ridge")]
    return [
        ("test1 within noise", "recovers them within noise on {} of 10 plants", [len(good)], str),
        ("test1 z of the good fits", "(|error| / Fisher bound at most {})",
         [max(abs(float(r["z_pl"])) for r in good)], f1),
        ("test1 trapped fits", "{} to {} low at a cost of {} to {} against {} to {} at the truth",
         [min(low), max(low), min(float(r["cost_fit"]) for r in bad), max(float(r["cost_fit"]) for r in bad),
          min(float(r["cost_truth"]) for r in bad), max(float(r["cost_truth"]) for r in bad)],
         (f2, f2, tens, tens, whole, whole)),
        ("test1 multistart", "recover all ten within noise (|z| at most {})", [zm], f1),
        ("test1 rate bound", "(Fisher bound {}-{} % in log)", [col(T1, "bound_k_pct").min(), col(T1, "bound_k_pct").max()], whole),
        ("test1 rate on its bound", "ends on its upper bound on {} of 10", [on_bound], str),
        ("test1 net low", "the net is {} to {} low on the four highest factors", [min(net_low), max(net_low)], f2),
        ("test1 net seed 1", "(seed 1: {} against {})", [float(s1["beta_pl_net"]), float(s1["beta_pl_true"])], f3),
        ("test1 start below spec", "(minimum {} to {} % for {}-{} h",
         [min(float(r["true_purity_min"]) for r in below), max(float(r["true_purity_min"]) for r in below),
          min(hours), max(hours)], (pct1, pct1, whole, whole)),
    ]


def claims_test2():
    ID, G, CH = "fair_fit_identify_ndpr_main.csv", "fair_fit_gains_ndpr_main.csv", "fair_fit_check_ndpr_main.csv"
    alone = {r["param"]: float(r["bound_alone"]) / float(r["spread"]) for r in rows(ID)}
    g = lambda p: col(G, "rel_change_G", param=p)                         # noqa: E731
    new, old = "beta_pl+beta_np+dn_acid", "beta_pl+beta_np+k_all"
    return [
        ("test2 alone Pr/(La+Ce)", "Pr/(La+Ce) {}", [alone["beta_pl"]], f3),
        ("test2 alone Nd/Pr", "Nd/Pr {}", [alone["beta_np"]], f3),
        ("test2 alone acid", "acid exponent {}", [alone["dn_acid"]], f3),
        ("test2 alone common K", "common K factor {}", [alone["common_K"]], f2),
        ("test2 alone extraction rate", "shared/extr. rate {}", [alone["k_ext"]], f2),
        ("test2 alone hold-up", "mixer hold-up {}", [alone["V_mix"]], f2),
        ("test2 gains Pr/(La+Ce)", "- {}-{} harder side", [g("beta_pl").min(), g("beta_pl").max()], f2),
        ("test2 gains Nd/Pr", "- {}-{} no harm", [g("beta_np").min(), g("beta_np").max()], f2),
        ("test2 gains acid", "(-1.000) {}-{} no harm", [g("dn_acid").min(), g("dn_acid").max()], f2),
        ("test2 gains common K", "acid exponent {}-{} no harm", [g("common_K").min(), g("common_K").max()], f2),
        ("test2 check new", "factors + acid exponent {} {} {} {} {}",
         [np.median(err(CH, "beta_pl_fit", set=new)), err(CH, "beta_pl_fit", set=new).max(),
          np.median(err(CH, "beta_np_fit", "beta_np_true", set=new)), err(CH, "beta_np_fit", "beta_np_true", set=new).max(),
          np.median(col(CH, "cost_fit", set=new))], (f3, f3, f3, f3, whole)),
        ("test2 check old", "factors + shared rate {} {} {} {} {}",
         [np.median(err(CH, "beta_pl_fit", set=old)), err(CH, "beta_pl_fit", set=old).max(),
          np.median(err(CH, "beta_np_fit", "beta_np_true", set=old)), err(CH, "beta_np_fit", "beta_np_true", set=old).max(),
          np.median(col(CH, "cost_fit", set=old))], (f3, f3, f3, f3, whole)),
        ("test2 fit times", "{} {}-{} s", [np.median(col(CH, "cost_fit", set=new)), col(CH, "fit_s", set=new).min(),
                                           col(CH, "fit_s", set=new).max()], whole),
        ("test2 acid offsets", "The fitted acid offset is {} to {} against",
         [min(float(r["others"].split("=")[1]) for r in rows(CH) if r["set"] == new),
          max(float(r["others"].split("=")[1]) for r in rows(CH) if r["set"] == new)], signed2),
        ("test2 rest 7 and 8", "leaves plants 7 and 8 at {} and {} % purity (recovery {} and {} %)",
         [col(CH, "rest_purity", set=new, seed=7)[0], col(CH, "rest_purity", set=new, seed=8)[0],
          col(CH, "rest_recovery", set=new, seed=7)[0], col(CH, "rest_recovery", set=new, seed=8)[0]], pct1),
    ]


def claims_test3():
    A, BC = "fair_fit_check_ndpr_main_map.csv", "fair_fit_check_ndpr_main_uv2-4-6-12-16-20_map.csv"
    b, c = "beta_pl+beta_np+dn_acid_map_uv2-4-6-12-16-20", "beta_pl+beta_np+dn_acid+common_K_map_uv2-4-6-12-16-20"

    def line(name, where):
        return [np.median(err(name, "beta_pl_fit", **where)), err(name, "beta_pl_fit", **where).max(),
                np.median(err(name, "beta_np_fit", "beta_np_true", **where)), err(name, "beta_np_fit", "beta_np_true", **where).max(),
                col(name, "worst_corr", **where).min(), col(name, "worst_corr", **where).max(),
                np.median(col(name, "cost_fit", **where))]
    fmt = (f3, f3, f3, f3, f2, f2, whole)
    ck = [float(r["common_K_fit"]) for r in rows(BC) if r["set"] == c and r["seed"] != "1"]
    return [
        ("test3 A", "12 readings, three numbers {} {} {} {} {}-{} {}", line(A, {}), fmt),
        ("test3 B", "+ scrub 12, 16, 20 {} {} {} {} {}-{} {}", line(BC, {"set": b}), fmt),
        ("test3 C", "B + common K factor {} {} {} {} {}-{} {}", line(BC, {"set": c}), fmt),
        ("test3 common K fitted", "fitted {} to {} (plant 1 {})",
         [min(ck), max(ck), col(BC, "common_K_fit", set=c, seed=1)[0]], signed3),
        ("test3 plant 1 trapped", "(cost {} against about", [col(BC, "cost_fit", set=b, seed=1)[0]], whole),
    ]


def claims_test4():
    R = "realistic_plants_ndpr_main.csv"
    est = ("fit_old", "fit_new", "ridge", "net", "ridge_mix", "net_mix")
    out = []
    for kind, label in (("entrainment", "entrainment"), ("settler", "settler"), ("drifting", "drifting"), ("book", "book (vs 2.1)")):
        vals = []
        for e in est:
            d = err(R, f"{e}_beta_pl", kind=kind)
            vals += [np.median(d), d.max()]
        out.append((f"test4 {kind}", label + " " + " ".join(["{}/{}"] * len(est)), vals, f3))
    return out


def swap_claims():
    S = "model_swap_ndpr_main.csv"
    m = {(r["seed"], r["mode"]): r for r in rows(S)}
    return [
        ("swap fixed", "never refitted {} {} 0 {} / {}",
         [float(m[("4", "fixed")]["purity_end"]), float(m[("4", "fixed")]["purity_min"]),
          float(m[("4", "fixed")]["recovery_end"]), float(m[("11", "fixed")]["recovery_end"])], pct2),
        ("swap bias numbers", "({} against {} and {} against {}, purity plus recovery)",
         [float(m[("4", "match")]["bias_after_mean"]), float(m[("4", "match")]["bias_before_mean"]),
          float(m[("11", "match")]["bias_after_mean"]), float(m[("11", "match")]["bias_before_mean"])], f4),
    ]


def claims_test5_6():
    P, NS, NL = "realistic_plants_ndpr_main_perturbed.csv", "net_size_ndpr_main.csv", "net_size_loop_ndpr_main.csv"
    pairs = [(e, err(P, f"{e}_beta_pl")) for e in ("fit_old", "fit_new", "ridge", "net", "ridge_mix", "net_mix")]
    mean = lambda w, d: np.mean(col(NS, "rmse_pl", width=w, depth=d) + col(NS, "rmse_np", width=w, depth=d))   # noqa: E731
    return [
        ("test5 errors", "fit as published {} / {} new fit {} / {} ridge {} / {} net {} / {} ridge mixed {} / {} net mixed {} / {}",
         [v for _, d in pairs for v in (np.median(d), d.max())], f3),
        ("test6 table depth 2", "depth 2 {} {} {} {} {}", [mean(w, 2) for w in (4, 8, 16, 32, 64)],
         (f3, f3, f3, f4, f4)),
        ("test6 weights", "width 32, depth 3, {} weights", [col(NS, "weights", width=32, depth=3)[0]], whole),
        ("test6 loop", "end purity {}-{} % at worst",
         [min(col(NL, "purity_end", size=s).min() for s in {r["size"] for r in rows(NL)}),
          max(col(NL, "purity_end", size=s).min() for s in {r["size"] for r in rows(NL)})], pct2),
    ]


def phase9_claims():
    C, SL, SB = "estimate_matters_C.csv", "switch_losses_ndpr_main.csv", "switch_bound_ndpr_main.csv"
    ref = {r["seed"]: r for r in rows(C) if float(r["offset"]) == 0.0}

    def pct_vs_true(o, column):
        sel = [r for r in rows(C) if float(r["offset"]) == o]
        return 100 * np.mean([float(r[column]) - float(ref[r["seed"]][column]) for r in sel]) / np.mean(
            [float(ref[r["seed"]][column]) for r in sel])

    def c_line(o):
        sel = [r for r in rows(C) if float(r["offset"]) == o]
        return [min(float(r["purity_end"]) for r in sel), np.mean([float(r["hours_below_spec"]) for r in sel]),
                np.mean([float(r["purity_shortfall_pt_h"]) for r in sel])]
    after24 = col(SL, "lost_pct_24h")
    after235 = col(SL, "lost_pct_235h")
    tm = [r for r in rows(SL) if r["variant"] == "true_model" and r["seed"] in ("3", "10", "14", "16", "17")]
    e2 = lambda v: col(SB, "replay_min_purity", variant=v)                  # noqa: E731
    l2 = lambda v: col(SB, "replay_lost_pct", variant=v)                    # noqa: E731
    out = [(f"C {o:+.1f}", f"{o:+.1f} ".replace("+0.0", "0").replace("0.0 ", "0 ") + "{} % {} {}", c_line(o),
            (pct2, f1, f1)) for o in (-0.4, -0.2, -0.1, 0.1, 0.2)]
    out += [
        ("C reagent -0.1", "-0.1 99.88 % 10.3 21.2 +{} % +{} %",
         [pct_vs_true(-0.1, "base_mol"), pct_vs_true(-0.1, "acid_mol")], whole),
        ("C reagent -0.4", "-0.4 99.92 % 10.4 21.8 +{} % +{} %",
         [pct_vs_true(-0.4, "base_mol"), pct_vs_true(-0.4, "acid_mol")], whole),
        ("E1 windows", "(24 h: {}-{} %; 235 h: {}-{} %)", [after24.min(), after24.max(), after235.min(), after235.max()], f1),
        ("E1 true model purity", "The true model holds purity at {}-{} % on the plants that dipped",
         [min(float(r["purity_min"]) for r in tm), max(float(r["purity_min"]) for r in tm)], pct2),
        ("E2 known ahead", "switch known in advance {}-{} % {}-{} %",
         [e2("known_ahead").min(), e2("known_ahead").max(), l2("known_ahead").min(), l2("known_ahead").max()],
         (pct2, pct2, f2, f2)),
        ("E2 at the switch", "known only when it happens {}-{} % {}-{} %",
         [e2("at_the_switch").min(), e2("at_the_switch").max(), l2("at_the_switch").min(), l2("at_the_switch").max()],
         (pct2, pct2, f2, f2)),
        ("E2 no rate limits", "no rate limits {}-{} % {}-{} %",
         [e2("no_rate_limits").min(), e2("no_rate_limits").max(), l2("no_rate_limits").min(), l2("no_rate_limits").max()],
         (pct2, pct2, f2, f2)),
        ("E2 slow ramp", "6 h ramp instead of 1 h {}-{} % {}-{} %",
         [e2("slow_ramp_6h").min(), e2("slow_ramp_6h").max(), l2("slow_ramp_6h").min(), l2("slow_ramp_6h").max()],
         (pct2, pct2, f2, f2)),
    ]
    return out


def phase10_claims():
    O = "online_refit_ndpr_main.csv"
    r = {row["variant"]: row for row in rows(O)}
    labels = {"true": "true model", "fit_at_rest": "fit at rest (three numbers)", "online": "online, from the ridge",
              "ridge_factors_fit_acid": "ridge factors + fitted acid", "fit_factors_nominal_acid": "fitted factors + nominal acid"}
    out = []
    for v, label in labels.items():
        x = r[v]
        vals = [float(x[c]) for c in ("sapon_max", "scrub_acid_min", "acid_extraction_min_mM", "acid_scrub_min_mM",
                                       "hours_below_spec", "purity_end", "recovery_end")]
        out.append((f"phase10 {v}", label + " no {} {} {} / {} {} {} % / {} %", vals, (f3, f2, f2, f1, f1, pct2, pct2)))
    rd = r["ridge"]
    out.append(("phase10 ridge", "ridge {} h {} {} mol/L", [float(rd["stalled_at_h"]), float(rd["sapon_max"]),
                                                             float(rd["scrub_acid_min"])], (f1, f3, f2)))
    B = {row["variant"]: row for row in rows("saponification_balance_ndpr_main.csv")}
    for v, label, unit in (("ridge+balance", "ridge + balance", " mmol/L"), ("true+balance", "true model + balance", ""),
                           ("online+balance", "online + balance", "")):
        x = B[v]
        vals = [float(x[c]) for c in ("sapon_max", "acid_extraction_min_mM", "acid_scrub_min_mM", "hours_below_spec",
                                       "purity_end", "recovery_end")]
        out.append((f"phase10 {v}", label + " no {} {} / {}" + unit + " {} {} % / {} %", vals, (f3, f1, f1, f1, pct2, pct2)))
    out.append(("phase10 fit at rest acid", "The fit at rest found +{};", [float(r["fit_at_rest"]["dn_acid_end"])], f3))
    out.append(("phase10 online last acid", "(last +{})", [float(r["online"]["dn_acid_end"])], f3))
    return out


def notebook24_claims():
    """FINDINGS section 4's dated note of Oct 24: the stored run against the settled rerun (sized plant)."""
    def table(name):
        return {r["case"]: r for r in rows(name)}
    old, new = table("conditioning_ndpr_plant_sized.csv"), table("conditioning_ndpr_plant_sized_settled.csv")
    near_one = (2, 3, 6, 9, 10, 11, 14, 16, 17, 20)
    others = tuple(s for s in range(1, 21) if s not in near_one)

    def vals(t, seeds, col):
        v = [float(t[f"seed {s}"][col]) for s in seeds if t[f"seed {s}"].get(col, "") not in ("", "nan")]
        return [x for x in v if np.isfinite(x)]

    def span(t, seeds, col):
        v = vals(t, seeds, col)
        return [min(v), max(v)]

    def above_one(t):
        v = vals(t, range(1, 21), "spec_model_error_flows")
        return [sum(x > 1 for x in v), len(v)]
    settled_six = (3, 6, 9, 10, 17, 20)
    return [
        ("nb24 model at design", "condition no. {} {}",
         [float(old["model at design"]["flows_cond"]), float(new["model at design"]["flows_cond"])], f1),
        ("nb24 near one", "16, 17, 20) {}-{} {}-{}", span(old, near_one, "model_error_flows")
         + span(new, near_one, "model_error_flows"), f2),
        ("nb24 the others", "the other ten {}-{} {}-{}", span(old, others, "model_error_flows")
         + span(new, others, "model_error_flows"), f1),
        ("nb24 spec point count", "above 1 on {} of {}, above 1 on {} of {},", above_one(old) + above_one(new), str),
        ("nb24 spec point range", "{}-{} {}-{} condition number there", span(old, range(1, 21), "spec_model_error_flows")
         + span(new, range(1, 21), "spec_model_error_flows"), (f2, f1, f1, f1)),
        ("nb24 spec point cond", "condition number there (plant) {}-{} {}-{}",
         span(old, range(1, 21), "spec_flows_cond") + span(new, range(1, 21), "spec_flows_cond"), f1),
        ("nb24 settled six", "give {}-{} at their design points", span(new, settled_six, "spec_model_error_flows"), f1),
    ]


def ten_plant_claims():
    """FINDINGS phase 10, both fixes on the ten fair-test plants (notebook 53)."""
    T = "ten_plants_v13_ndpr_main.csv"
    labels = (("ridge", "ridge {} of 10 (to {})"), ("ridge+balance", "ridge + balance {} of 10 (lowest {} mmol/L)"),
              ("online", "online {} of 10 (lowest {})"), ("online+balance", "online + balance {} of 10 (lowest {})"),
              ("true", "true model {} of 10 (lowest {})"), ("true+balance", "true + balance {} of 10 (lowest {})"))
    out = []
    for v, head in labels:
        acid = col(T, "acid_extraction_min_mM", variant=v)
        stalled = sum(1 for r in rows(T) if r["variant"] == v and r["stalled_at_h"] != "")
        vals = [int((acid < 1.0).sum()), acid.min(), stalled, col(T, "sapon_max", variant=v).max(),
                col(T, "hours_below_spec", variant=v).mean(), col(T, "purity_end", variant=v).min()]
        out.append((f"ten plants {v}", head + " {} {} {} {} %", vals, (str, f2, str, f3, f1, pct2)))
    return out


def drift_claims():
    """FINDINGS phase 10, the slow drift measured on ten plants (notebook 54)."""
    D = "slow_drift_ndpr_main.csv"
    r = {row["seed"]: row for row in rows(D)}
    cols = ("purity_end", "purity_min_after_24h", "recovery_end", "recovery_max_after_24h", "recovery_loss_end",
            "stock_change_end_mol", "rest_purity_min", "rest_purity_end")
    out = [(f"drift plant {s}", f"plant {s} " + "{} / {} {} / {} {} {} {} / {}", [float(r[s][c]) for c in cols],
            (pct2, pct2, pct2, pct2, pct2, whole, pct2, pct2)) for s in ("7", "8")]
    others = [row for k, row in r.items() if k not in ("7", "8")]

    def span(c):
        v = [float(x[c]) for x in others]
        return [min(v), max(v)]
    vals = (span("purity_end") + span("recovery_end") + span("recovery_loss_end") + span("stock_change_end_mol")[::-1]
            + span("rest_purity_end") + span("purity_min_after_24h") + span("recovery_max_after_24h"))
    out.append(("drift the other eight", "the other {}-{} / {}-{} / {}-{} {} to {} end {}-{} eight {}-{} {}-{}", vals,
                (pct2, pct2, pct2, pct2, pct2, pct2, whole, whole, pct2, pct2, pct2, pct2, pct2, pct2)))
    return out


def rest_check_claims():
    """FINDINGS phase 10, the rest-point check against the drift (notebook 55 against notebook 54)."""
    A = {r["seed"]: r for r in rows("slow_drift_ndpr_main.csv")}
    B = {r["seed"]: r for r in rows("rest_point_check_ndpr_main.csv")}
    out = []
    for s in ("7", "8"):
        vals = [float(t[s][c]) for c in ("rest_purity_min", "recovery_max_after_24h", "recovery_loss_end",
                                          "stock_change_end_mol") for t in (A, B)]
        out.append((f"rest check plant {s}", f"plant {s} " + "{} {} {} {} {} {} {} {}", vals,
                    (pct2, pct2, pct2, pct2, pct2, pct2, whole, whole)))
    ends = [float(B[s]["rest_purity_end"]) for s in B if s not in ("7", "8")]
    out.append(("rest check other eight", "End settled purity on the other eight: {}-{} %", [min(ends), max(ends)], pct2))
    return out


def all_fixes_claims():
    """FINDINGS phase 10, all three fixes together from the ridge's model over 600 h (notebook 56)."""
    F = "all_fixes_ndpr_main.csv"
    traces = np.load(RUNS / "all_fixes_ndpr_main.npz")

    def late_settled(v):
        """Each plant's lowest settled purity from 120 h on (plants with checks that late)."""
        out = []
        for r in rows(F):
            t, p = (traces.get(f"{r['seed']}_{v}_rest_{k}") for k in ("t", "purity"))
            if r["variant"] == v and t is not None and (t >= 120).any():
                out.append(p[t >= 120].min())
        return np.array(out)
    fixed, ridge = "online+balance+rest", "ridge"
    late_f, late_r = late_settled(fixed), late_settled(ridge)
    acid = {v: col(F, "acid_extraction_min_mM", variant=v) for v in (fixed, ridge)}
    hours = {v: [float(r["hours_below_spec"]) for r in rows(F) if r["variant"] == v and not r["stalled_at_h"]]
             for v in (fixed, ridge)}
    return [
        ("all fixes settled", "settled purity under spec after 120 h {} of 10 plants 0 (lowest {} %)",
         [int((late_r < SPEC).sum()), late_f.min()], (str, pct2)),
        ("all fixes acid", "extraction acid under 1 mmol/L {} plants (to {}) 0 (lowest {} mmol/L)",
         [int((acid[ridge] < 1).sum()), acid[ridge].min(), acid[fixed].min()], (str, f2, f2)),
        ("all fixes recovery read", "recovery read, highest {} % {} %",
         [max(float(r["recovery_max_after_24h"]) for r in rows(F) if r["variant"] == v and r["recovery_max_after_24h"])
          for v in (ridge, fixed)], pct2),
        ("all fixes hours", "hours below spec (mean) {} ({} plants) {} ({} plants)",
         [np.mean(hours[ridge]), len(hours[ridge]), np.mean(hours[fixed]), len(hours[fixed])], (f1, str, f1, str)),
    ]


def margin_claims():
    """FINDINGS phase 10, the margin above the acid floor (notebook 57 against notebook 56)."""
    A = {r["seed"]: r for r in rows("all_fixes_ndpr_main.csv") if r["variant"] == "online+balance+rest"}
    B = {r["seed"]: r for r in rows("acid_margin_ndpr_main.csv")}
    out = []
    for s in ("3", "9", "10"):
        vals = [float(t[s][c]) for c in ("acid_extraction_min_mM", "hours_below_spec", "recovery_loss_end") for t in (A, B)]
        out.append((f"margin plant {s}", f"plant {s} " + "{} {} {} {} {} {}", vals, (f2, f2, f1, f1, pct2, pct2)))
    return out


def v13_claims() -> list:
    """FINDINGS phase 10 (v1.3). Each group is checked when its result table is in this version of the
    project; v1.2 has neither the tables nor phase 10, so there they're left out. Only these groups
    are optional: a missing table anywhere else is an error."""
    out = []
    for build in (phase10_claims, ten_plant_claims, drift_claims, rest_check_claims, all_fixes_claims, margin_claims):
        try:
            out += build()
        except FileNotFoundError:
            continue
    return out


CLAIMS = (notebook24_claims() + claims_test1() + claims_test2() + claims_test3() + claims_test4() + swap_claims()
          + claims_test5_6() + phase9_claims() + v13_claims())


# ---- the README ----------------------------------------------------------------------------------
README = PROJECT / "README.md"


def readme_claims() -> list:
    """Numbers the README quotes, computed as FINDINGS computes them: (name, sentence, values, format)."""
    CH, P = "fair_fit_check_ndpr_main.csv", "realistic_plants_ndpr_main_perturbed.csv"
    new = "beta_pl+beta_np+dn_acid"
    C = [r for r in rows("estimate_matters_C.csv") if abs(float(r["offset"]) - 0.4) < 1e-9]
    return [
        ("readme ml typical", "({} against {})", [np.median(err(P, "net_beta_pl")), np.median(err(CH, "beta_pl_fit", set=new))], f3),
        ("readme ml worst", "({} against {})", [err(P, "net_beta_pl").max(), err(CH, "beta_pl_fit", set=new).max()], (f2, f3)),
        ("readme seven of ten", "about 50 %, lost {} plants of 10", [sum(r["hours_to_spec"] == "nan" for r in C)], str),
    ]


def readme_conditions() -> list:
    """README statements that round or bound a table's number: (name, sentence, holds)."""
    C = rows("estimate_matters_C.csv")
    ref = {r["seed"]: float(r["base_mol"]) for r in C if float(r["offset"]) == 0.0}
    pess = [r for r in C if abs(float(r["offset"]) + 0.1) < 1e-9]
    base_up = 100 * np.mean([float(r["base_mol"]) - ref[r["seed"]] for r in pess]) / np.mean([ref[r["seed"]] for r in pess])
    off = [r for r in rows("estimate_matters_D_bias.csv") if r["setting"] == "bias_filter=0.0" and abs(float(r["offset"]) - 0.2) < 1e-9]
    SB = rows("switch_bound_ndpr_main.csv")
    weights = {(r["width"], r["depth"]): float(r["weights"]) for r in rows("net_size_ndpr_main.csv")}
    return [
        ("readme reagent per error", "about 10 % more for each 10 % of error", 8 <= base_up <= 12),
        ("readme bias off", "with it switched off, about 20 % was enough to lose a plant",
         sum(r["hours_to_spec"] == "nan" for r in off) >= 1),
        ("readme switch bound", "holds 99.5 % through the switch on every plant tested, losing under 5 % of the NdPr",
         min(float(r["replay_min_purity"]) for r in SB) >= 0.995 and max(float(r["replay_lost_pct"]) for r in SB) < 5),
        ("readme half the size", "A network half the size does as well", 0.45 <= weights[("32", "3")] / weights[("64", "2")] <= 0.55),
    ]


README_CLAIMS = readme_claims()
README_CONDITIONS = readme_conditions()


def expected(template: str, values, fmt) -> str:
    """The template with each value formatted the way the text rounds it."""
    fmts = fmt if isinstance(fmt, tuple) else (fmt,) * len(values)
    return normalized(template.format(*(f(v) for f, v in zip(fmts, values))))


@pytest.mark.parametrize("claim", CLAIMS, ids=[c[0] for c in CLAIMS])
def test_findings_quotes_its_table(claim):
    """FINDINGS contains the sentence with the numbers its table gives today."""
    name, template, values, fmt = claim
    text = normalized(FINDINGS.read_text(encoding="utf-8"))
    want = expected(template, values, fmt)
    if want in text:
        return
    old = what_the_document_says(template, text)
    pytest.fail(f"FINDINGS: '{template}' says {old}; the table gives '{want}'")


@pytest.mark.parametrize("claim", README_CLAIMS, ids=[c[0] for c in README_CLAIMS])
def test_readme_quotes_its_table(claim):
    """The README contains the sentence with the numbers its table gives today."""
    name, template, values, fmt = claim
    text = normalized(README.read_text(encoding="utf-8"))
    want = expected(template, values, fmt)
    assert want in text, f"README: '{template}' says {what_the_document_says(template, text)}; the table gives '{want}'"


@pytest.mark.parametrize("condition", README_CONDITIONS, ids=[c[0] for c in README_CONDITIONS])
def test_readme_statement_holds(condition):
    """The README says it, and the tables bear it out."""
    name, sentence, holds = condition
    assert normalized(sentence) in normalized(README.read_text(encoding="utf-8")), f"README no longer says: {sentence}"
    assert holds, f"the tables no longer bear out: {sentence}"
