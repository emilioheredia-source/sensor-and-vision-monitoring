"""Phase equilibrium and phase densities: the chemistry of one stage.

Where this sits
---------------
circuit.py builds the stage equations and calls this module for two things: how far metal wants
to move between the phases (the distribution coefficient D), and how heavy each phase is (the
densities, which set where the settler interface sits). The numbers come from params.py.

Every function here works on numpy arrays and on CasADi symbols alike, so the same lines serve
the simulator and the MPC (ARCHITECTURE.md, decision 1). That's why the code only uses +, -, *,
/, ** and a sum: both numpy and CasADi understand those. There's no `if` on a value, only on
parameters, which are plain numbers.

The chemistry
-------------
The extractant (PC88A) sits in the organic as a dimer, H2L2. It gives up three H+ to take one
rare-earth ion (Idaho thesis Eq. 4, Carester eq. 1):

    RE3+ (aq) + 3 H2L2 (org)  <->  RE(HL2)3 (org) + 3 H+ (aq)

Mass action on that reaction gives the distribution coefficient of component i, the ratio of
its organic to its aqueous concentration at equilibrium:

    D_i = c_org_i / c_aq_i = K_i * L_free^3 / h^n

  L_free  free dimer, mol/L: the total minus what the metal already holds (3 per metal ion)
  h       aqueous H+, mol/L
  K_i     the one number that differs between components (fitted or chosen, parameters.toml)
  n       acid exponent, 3

Two things follow. A nearly full organic has little L_free, so the component with the bigger K
pushes the others out: that's what the scrub section works on. And extraction releases acid,
which lowers D: that's why plants saponify the extractant, and why acid is tracked per stage.

As implemented (APPENDIX_A_model.md, A.4), with a floor on h and a smooth cap on D, both
explained in parameters.toml:

    D_raw = K * L_free^3 / (h^n + h_min^n)
    D     = D_raw / (1 + D_raw / D_max)

The densities (A.9) are linear in loading:

    rho_org = rho_org0 + sum_i M_i * c_org_i / 1000
    rho_aq  = rho_water + sum_i (M_i + 3 M_Cl - rho_water * phi_v) * c_aq_i / 1000

Why the law has this shape: the reaction above, with the extractant as a dimer and three dimers
per RE3+, is the standard mass-action form for an acidic organophosphorus extractant in a
kerosene-type diluent (Idaho Eq. 4, Carester eq. 1; textbook).

Judgement call: no temperature anywhere. K is taken at the lab's temperature. A plant runs warmer
or colder with the season, and saponification gives off heat in the first extraction stage. How
much D moves per degree for PC88A with these elements wasn't looked up or measured. If it moves
a lot, the operating point drifts with the season. Batch tests at two temperatures would settle it.
"""
from __future__ import annotations

import numpy as np

from .params import Params


def _sum(x):
    """Sum over components, for a numpy array or a CasADi column alike.

    numpy's sum doesn't work on CasADi symbols, so a CasADi object is summed with ca.sum1
    (sum down the column). Anything else goes to np.sum. If CasADi isn't installed, everything
    goes to np.sum.
    """
    try:
        import casadi as ca
        if isinstance(x, (ca.SX, ca.MX, ca.DM)):
            return ca.sum1(x)
    except ImportError:
        pass
    return np.sum(x)


def free_dimer(c_org, p: Params):
    """Extractant dimer not yet holding metal, mol/L.

    Each metal ion in the organic holds `ligands_per_metal` (3) dimers, as RE(HL2)3. So the free
    dimer is the total minus three times the total metal in the organic (Idaho Eq. 4).

    c_org: organic concentrations, mol/L, one per component.
    p:     the Params (dimer_total, ligands_per_metal).
    Returns L_free, mol/L. It goes negative if the organic is loaded past its capacity; nothing
    here stops that.

    Why: three dimers per metal ion is the stoichiometry of the reaction in the module docstring.
    Judgement call: it assumes every metal ion sits as RE(HL2)3 right up to full loading. Near
    saturation acidic organophosphorus extractants can form other complexes that hold the metal
    with less extractant (textbook), so the cube law is least sure exactly where the scrub runs.
    The 85 % loading cap (parameters.toml) keeps the design off the steepest part. Loading
    isotherms up to saturation, on the real organic, would settle it.
    """
    return p.dimer_total - p.ligands_per_metal * _sum(c_org)


def distribution(c_org, h, p: Params):
    """Distribution coefficient D_i for every component, given the organic loading and acidity.

    D_i is the organic-to-aqueous concentration ratio component i would have at equilibrium in
    this stage. Bigger D means the metal goes to the organic more readily.

    c_org: organic concentrations, mol/L (array of n components)
    h:     aqueous H+ concentration, mol/L
    p:     the Params (K, n_acid, h_min_D, D_max, and what free_dimer needs)
    Returns D, dimensionless, one per component (numpy array or CasADi column).

    Example:
        D = distribution(np.zeros(4), 0.1, p)     # unloaded organic at 0.1 M acid
        beta_Nd_Pr = D[2] / D[1]                  # separation factor, about K_Nd / K_Pr = 1.51
    """
    L = free_dimer(c_org, p)

    # The acid floor: below about pH 3 the law stops applying and D would run to infinity.
    # Adding h_min^n to h^n keeps D finite. 0 = off (see parameters.toml).
    # Judgement call: pH 3 is where the fitted data stop, not where the chemistry changes. Light
    # lanthanide hydroxides come down nearer pH 6-7 (textbook). The floor only acts if a stage runs
    # that low; at the ndpr_plant point the least acid stage is about pH 1.75, so it isn't touched.
    floor = p.h_min_D ** p.n_acid if p.h_min_D > 0 else 0.0

    # Smooth positive part of h: about h when h > 0, about 0 when h < 0. An integrator step can
    # overshoot h below zero, and h^n needs h >= 0. The 1e-12 keeps it smooth at h = 0.
    h = 0.5 * (h + (h * h + 1e-12) ** 0.5)

    # The mass-action law: K times free dimer cubed, over acid to the n.
    # Judgement call: h is the H+ concentration from the acid balance, and K is a conditional
    # constant. It folds in the activity coefficients at the Idaho test conditions (50 g/L feeds,
    # 0.04-0.3 M acid). The scrub here holds about 0.8 M RECl3, so its ionic strength is more
    # than twice that of the fitted data. If K shifts there, D and the stage counts shift with
    # it. The free-exponent fit (n 3.9-4.5, parameters.toml) hints at this. Batch tests at
    # scrub-section strength would settle it.
    D = p.K * L ** 3 / (h ** p.n_acid + floor)

    # Smooth cap: D stays about the same while D << D_max and levels off at D_max. A nearly empty
    # organic at the acid floor would give D ~ 1e6, which only makes the equations stiff.
    # 0 = off (see parameters.toml).
    if p.D_max > 0:
        D = D / (1 + D / p.D_max)
    return D


def rho_organic(c_org, p: Params):
    """Density of the organic phase, g/mL.

    The unloaded organic plus the mass of the metal it carries, with its volume taken as
    unchanged. c_org is in mol/L and M in g/mol, so M * c_org is g/L; dividing by 1000 gives g/mL.

    c_org: organic concentrations, mol/L, one per component.
    p:     the Params (rho_org0, M).

    Why: 0.85 g/mL checks against a mix of 1 M PC88A (about a third of the volume, at about 0.95)
    in kerosene (about 0.80). Judgement call: the water and sodium a saponified organic picks up
    are left out. That's a few thousandths of g/mL at most, and commissioning sets the jacklegs
    on the model's densities, so it shows only in how the interface moves, not where it sits.
    """
    return p.rho_org0 + _sum(p.M * c_org) / 1000.0


def rho_aqueous(c_aq, p: Params):
    """Density of the aqueous phase, g/mL.

    Water plus the dissolved RECl3 salt. Each mol/L of salt adds its own mass (metal plus three
    chlorides) and pushes out the water its volume takes up (phi_v mL per mol, the apparent molal
    volume). The 1000 turns g/L into g/mL.

    c_aq: aqueous concentrations, mol/L, one per component.
    p:    the Params (rho_water, M, M_cl, phi_v).

    Judgement call: only the RECl3 counts. Free HCl and the NaCl from saponification are left
    out. At the ndpr_plant point the raffinate carries about 0.9 M sodium (752 mol/h of base in
    839 L/h), which adds roughly 0.03-0.04 g/mL, and the strip runs at 1.5-2 M HCl, which adds
    about 0.03 (2 M HCl is about 1.03 g/mL, textbook). A steady offset is absorbed when the
    jacklegs are set at commissioning. What it misses is the interface moving when the base dose
    or the strip acid changes. Adding the two salts to this sum would settle it.
    """
    salt_M = p.M + 3 * p.M_cl                       # g/mol of RECl3 per component
    return p.rho_water + _sum((salt_M - p.rho_water * p.phi_v) * c_aq) / 1000.0
