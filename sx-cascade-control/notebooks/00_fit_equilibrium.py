"""Fit the extraction equilibrium constants K_Nd and K_Pr from the Idaho thesis
single-metal tables, and check the resulting separation factor against the
thesis's own mixed-metal measurements.

The question: what are K_Nd and K_Pr, the two constants the whole model's
chemistry hangs on? (The La+Ce and the heavy constants are set relative to
these in params/parameters.toml.)

The law the thesis uses, for each metal i:

    D_i = K_i * [H2L2_free]^3 / [H+]^n          (Idaho thesis Eq. 4)

D_i is the distribution ratio (metal in the organic over metal in the aqueous),
[H2L2_free] the free extractant dimer in mol/L, [H+] the acid in mol/L.
Free dimer = total dimer - 3 * (metal in organic), as the thesis does.

Method: take the logarithm, so the law becomes a straight line, and fit it two
ways: with the acid exponent n held at 3 (the law), and with n left free.
Then compare the separation factor beta = D_Nd / D_Pr that each fit gives
against what the thesis measured on mixed Nd + Pr solutions (its Table 6).

Prints the numbers that went into params/parameters.toml [equilibrium.K].
Writes nothing.

Run:  pixi run python notebooks/00_fit_equilibrium.py
"""
import tomllib
from pathlib import Path

import numpy as np

# ---- 1. load the thesis tables ----
# The single-metal shake-out tables from the Idaho thesis, typed into a TOML file.
# Each row is: feed metal (g/L), starting acid, acid at equilibrium (mol/L),
# percent extracted, and the measured D.
HERE = Path(__file__).resolve().parent.parent
tables = tomllib.load(open(HERE / "params" / "idaho_extraction_tables.toml", "rb"))

DIMER_TOTAL = 0.5          # mol/L: 1 M PC88A monomer, all as dimer
MIN_EXTRACTION_PCT = 3     # below this D is a ratio of two tiny numbers


# ---- 2. the fit, for one metal ----
def fit(name, n_fixed=3.0):
    """Fit K (and optionally n) for one metal from its single-metal table.

    name     the table to use: "Nd" or "Pr".
    n_fixed  the acid exponent for the first fit (3, the law).

    Two fits on the log form  ln D - 3 ln L_free = ln K - n ln H:
      (a) n held at n_fixed, so ln K is just an average;
      (b) n free, by least squares.
    Prints both, with the rms error in ln D.
    Returns (K with n fixed, (K free, n free)).
    """
    t = tables[name]
    M = t["molar_mass"]                                  # g/mol, to turn g/L into mol/L
    H, D, L = [], [], []                                 # acid, D, free dimer: one entry per usable row
    for feed_gL, _h0, h_eq, pct, d in t["rows"]:
        if pct <= MIN_EXTRACTION_PCT:
            continue
        metal_org = feed_gL * pct / 100 / M              # mol/L, O/A = 1
        L.append(DIMER_TOTAL - 3 * metal_org)            # each metal ion holds three dimers' worth of extractant
        H.append(h_eq)
        D.append(d)
    H, D, L = map(np.array, (H, D, L))

    # y = ln D - 3 ln L_free = ln K - n ln H
    y = np.log(D) - 3 * np.log(L)

    # (a) n fixed: ln K is the mean
    lnK = np.mean(y + n_fixed * np.log(H))
    rms_fixed = np.sqrt(np.mean((lnK - n_fixed * np.log(H) - y) ** 2))

    # (b) n free: least squares on [1, -ln H]
    A = np.c_[np.ones_like(H), -np.log(H)]
    (lnK_free, n_free), *_ = np.linalg.lstsq(A, y, rcond=None)
    rms_free = np.sqrt(np.mean((A @ np.r_[lnK_free, n_free] - y) ** 2))

    print(f"{name}: {len(H)} points used")
    print(f"  n = {n_fixed:.0f} fixed : K = {np.exp(lnK):.4f}   rms(ln D) = {rms_fixed:.2f}")
    print(f"  n free       : K = {np.exp(lnK_free):.4f}   n = {n_free:.2f}   rms(ln D) = {rms_free:.2f}")
    return np.exp(lnK), (np.exp(lnK_free), n_free)


# ---- 3. fit both metals ----
# K_nd and K_pr are the n = 3 constants; free_nd and free_pr hold (K, n) from the free fit.
K_nd, free_nd = fit("Nd")
K_pr, free_pr = fit("Pr")

# ---- 4. check the separation factor against the mixed-metal measurements ----
# With n fixed at 3 for both metals, the acid cancels and beta is just K_Nd / K_Pr.
# With free exponents the two n differ, so beta changes with acidity; print it at a few acid levels
# and set it beside what the thesis measured on mixed solutions. ARCHITECTURE.md ("Open", acid
# exponent) has why the model keeps n = 3.
print(f"\nbeta Nd/Pr with n = 3 fixed: {K_nd / K_pr:.2f}")
print("beta Nd/Pr with free exponents, by acidity:")
for h in (0.08, 0.13, 0.18, 0.30):
    b = (free_nd[0] * h ** -free_nd[1]) / (free_pr[0] * h ** -free_pr[1])
    print(f"  [H+] = {h:.2f} M : {b:.2f}")
print("Idaho Table 6, measured mixed metal:")
for _h0, h, _nd, _pr, beta in tables["mixed_table6"]["rows"]:
    print(f"  [H+] = {h:.2f} M : {beta:.2f}")
