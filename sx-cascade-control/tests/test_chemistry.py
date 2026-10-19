"""Checks on the chemistry: src/chemistry.py and the feed numbers in src/params.py.

Everything else in the project sits on these few functions. The simulator, the
optimizer and the MPC all get their distribution ratios D, free extractant and
phase densities from here. If they're wrong, every run is wrong in the same way
and nothing downstream would notice. So these tests hold them to outside facts:

- the Idaho thesis extraction tables the K values were fitted from,
- the Nd/Pr separation factor those K values give,
- simple bookkeeping (3 extractant dimers per metal, densities in a sane range),
- the published Eneabba feed composition.

There's also a check that the CasADi version of each formula gives the same
numbers as the numpy version, because the MPC uses the CasADi one and the
simulator uses the numpy one.
"""
import tomllib

import casadi as ca
import numpy as np
import pytest

from src.chemistry import distribution, free_dimer, rho_aqueous, rho_organic
from src.params import PARAMS_DIR, feed_concentrations, feed_mole_fractions, load_params

# The project's parameters (params/parameters.toml). Component order is LaCe, Pr, Nd, SmPlus.
p = load_params()
# Lyon's single-metal batch contacts (Idaho MSc thesis 2016, Tables 4 and 5). Each row is:
# feed metal g/L, feed H+ mol/L, equilibrium H+ mol/L, % extracted, measured D.
tables = tomllib.load(open(PARAMS_DIR / "idaho_extraction_tables.toml", "rb"))


def _c_org_single(name, feed_gL, pct):
    """Organic concentration, mol/L, after a batch contact of one metal at O/A = 1.

    feed_gL is the metal in the feed, g/L, and pct the % of it that was extracted.
    At O/A = 1 the extracted metal ends up in the same volume of organic, so
    c_org = feed_gL * pct / 100 / molar mass. Every other component is zero.
    """
    c = np.zeros(len(p.names))
    c[p.index(name)] = feed_gL * pct / 100 / p.M[p.index(name)]   # O/A = 1
    return c


@pytest.mark.parametrize("name", ["Nd", "Pr"])
def test_distribution_reproduces_idaho_tables_within_scatter(name):
    """D from the model matches the measured D in the Idaho tables, on average and within the fit scatter.

    The K values for Nd and Pr were fitted to these same rows, so the average
    error in ln D should be close to zero. The rms error should stay inside the
    scatter that parameters.toml reports for the fit (0.5 to 0.7 in ln D).
    It's a check that the fit was carried over into the code correctly: a
    wrong exponent, a wrong K or a units slip would show up as a bias.
    """
    # Rows with 3 % extraction or less are left out, the same as in the fit: D is too uncertain there.
    rows = [r for r in tables[name]["rows"] if r[3] > 3]
    ln_err = []
    for feed_gL, _h0, h_eq, pct, D_meas in rows:
        D = distribution(_c_org_single(name, feed_gL, pct), h_eq, p)[p.index(name)]
        ln_err.append(np.log(D) - np.log(D_meas))
    ln_err = np.array(ln_err)
    assert abs(ln_err.mean()) < 0.05          # unbiased: it is the fit
    assert np.sqrt(np.mean(ln_err**2)) < 0.8  # the scatter parameters.toml reports


def test_separation_factor_nd_pr():
    """The Nd/Pr separation factor (beta = D_Nd / D_Pr) is 1.51, as parameters.toml states.

    With an empty organic and the same acid, the ratio of the two D values is
    just K_Nd / K_Pr. Beta is the number that decides how many stages a Pr|Nd
    split needs. The measured mixed-metal value is 1.57 to 1.59 (Idaho Table 6)
    and the industrial range is 1.3 to 1.5, so 1.51 sits between the two.
    """
    # Empty organic (no loading), aqueous acid 0.15 mol/L.
    D = distribution(np.zeros(4), 0.15, p)
    beta = D[p.index("Nd")] / D[p.index("Pr")]
    assert beta == pytest.approx(1.51, abs=0.02)   # Idaho Table 6: 1.57-1.59; industrial 1.3-1.5


def test_loading_reduces_free_dimer_and_D():
    """Metal in the organic uses up extractant, so every D drops as the organic loads.

    Each metal ion holds 3 extractant dimers. The organic carries 0.5 mol/L of
    dimer, so 0.1 mol/L of Nd leaves 0.5 - 3 * 0.1 = 0.2 mol/L free. That's
    plain bookkeeping and it's checked exactly. With less free extractant, D
    for every component has to be lower than in an empty organic. This is the
    effect that limits how much metal a given organic flow can carry.
    """
    empty = distribution(np.zeros(4), 0.15, p)
    loaded = distribution(np.array([0, 0, 0.1, 0]), 0.15, p)   # 0.1 M Nd takes 0.3 of 0.5 M dimer
    assert free_dimer(np.array([0, 0, 0.1, 0]), p) == pytest.approx(0.2)
    assert np.all(loaded < empty)


def test_casadi_matches_numpy():
    """The same chemistry functions give the same numbers on CasADi symbols as on numpy arrays.

    The simulator calls these functions with numpy arrays. The MPC and the
    optimizer call them with CasADi symbols to get exact derivatives. The code
    is written once for both, so this checks that nothing in it behaves
    differently on the two kinds of input.
    """
    # Symbols: 4 organic concentrations (mol/L) and the aqueous acid (mol/L).
    c = ca.SX.sym("c", 4)
    h = ca.SX.sym("h")
    f = ca.Function("D", [c, h], [distribution(c, h, p), rho_organic(c, p), rho_aqueous(c, p)])
    # A test point: some of each of LaCe, Pr and Nd, mol/L.
    c0 = np.array([0.01, 0.02, 0.05, 0.0])
    D_sym, ro_sym, ra_sym = f(c0, 0.15)
    assert np.allclose(np.array(D_sym).ravel(), distribution(c0, 0.15, p))
    assert float(ro_sym) == pytest.approx(rho_organic(c0, p))
    assert float(ra_sym) == pytest.approx(rho_aqueous(c0, p))


def test_densities_sensible():
    """Phase densities, g/mL, land where they should and the aqueous stays the heavier phase.

    The unloaded organic is 0.85 g/mL (parameters.toml). Near full loading
    (0.16 mol/L of Nd, close to the 0.5 / 3 = 0.167 mol/L the extractant can
    hold) it gets heavier but stays under 0.90. A salty aqueous is a bit
    heavier than water. The aqueous has to stay heavier than even a fully
    loaded organic, or the settler wouldn't separate the phases the right way
    up and the interface model would make no sense.
    """
    assert rho_organic(np.zeros(4), p) == pytest.approx(0.85)
    assert 0.85 < rho_organic(np.array([0, 0, 0.16, 0]), p) < 0.90     # near full loading
    assert 0.997 < rho_aqueous(np.array([0.1, 0.1, 0.1, 0.0]), p) < 1.10
    assert rho_aqueous(np.zeros(4), p) > rho_organic(np.array([0, 0, 0.16, 0]), p)


def test_feed_composition_eneabba():
    """The Eneabba feed comes out of feeds.toml with the right shape and the right total.

    Mole fractions have to add up to 1. La+Ce is most of a monazite feed (over
    60 % here) and there's more Nd than Pr. Taking Sm and heavier out upstream
    has to zero that lump and renormalize the rest. Finally, converting to
    mol/L for a 50 g/L feed has to give back 50 g/L when multiplied by the
    molar masses: a mass balance on the unit conversion.
    """
    x = feed_mole_fractions("eneabba", p)
    assert x.sum() == pytest.approx(1.0)
    assert x[p.index("LaCe")] > 0.6 and x[p.index("Nd")] > x[p.index("Pr")]
    x2 = feed_mole_fractions("eneabba", p, removed=("SmPlus",))
    assert x2[p.index("SmPlus")] == 0 and x2.sum() == pytest.approx(1.0)
    # mol/L of each lump for 50 g/L of total rare-earth metal; c @ M is back in g/L.
    c = feed_concentrations("eneabba", 50.0, p, removed=("SmPlus",))
    assert float(c @ p.M) == pytest.approx(50.0)
