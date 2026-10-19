"""A plant that is not the controller's model.

What this file does
-------------------
The controller, the estimator and the optimizer plan on the nominal
parameters. If the plant they drive were built from the same numbers, the
controller would know the plant exactly and would always look good. So the
plant can be built from a perturbed copy of the parameters instead, drawn
from a seed. The gap between model and plant is then real and documented
instead of zero (ARCHITECTURE.md, decisions 2 and 20; APPENDIX_B_control.md, B.8).

It's the same equations with different numbers, and the controller is never told
the numbers. What is changed, and by how much (all "chosen", see Mismatch below):

  extraction constants K     each multiplied by exp(N(0, sigma_K)); the chosen
                             La+Ce constant gets twice the spread of the fitted ones
  acid exponent              n = 3 in the model, n + dn in the plant (a mild
                             difference in the law itself)
  stage transfer rate        per stage, a multiplier drawn between k_low and 1.0
                             (the plant's mixers aren't all the same)
  densities                  organic base density (spread sigma_rho, 2 %) and the
                             salt's molal volume (5 x sigma_rho, 10 %)
  mixer hold-up              spread sigma_volume, 5 % (slightly different dynamics)

The numbers are fixed for the whole run; nothing drifts. `strength` scales
every spread, and 0 gives the nominal plant back.

The random numbers
------------------
Two random generators are used, each started from a fixed seed, so the same
seed always gives the same plant:

  perturbed_params           seed m.seed. Draws, in this order: one normal per
                             extraction constant (in component order), then the
                             mixer volume, then the organic density, then the
                             salt's molal volume.
  stage_efficiency_factors   seed m.seed + 1000. Draws N uniforms, one per stage.

The draw order matters: changing it gives a different plant and different
reference results.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np

from .params import Params


# ---------------------------------------------------------------------------
# The settings of the perturbation
# ---------------------------------------------------------------------------

@dataclass
class Mismatch:
    """How far the plant is from the model. Every spread is a fraction (0.10 = 10 %).

    seed           the random seed; the same seed gives the same plant
    strength       scales every spread below; 0 = the nominal plant
    sigma_K        spread of the extraction constants (twice this for La+Ce)
    dn_acid        what is added to the acid exponent in the plant
    k_low          the lowest per-stage transfer-rate multiplier (at strength 1)
    sigma_rho      spread of the organic base density; the salt's molal volume gets 5 x this
    sigma_volume   spread of the mixer hold-up volume
    """
    seed: int = 1
    strength: float = 1.0
    sigma_K: float = 0.10            # chosen: fitted constants to +-10 %, the chosen La+Ce one to +-20 %
    # Judgement call on sigma_K: the Idaho fit scatter is 0.5-0.7 in ln D over about 10 points per
    # element, so the standard error on ln K alone is near 0.15-0.2, before any lab-to-plant shift.
    # 10 % is on the kind side. What matters for purity is the separation factor: two independent
    # 10 % draws move it by about 14 % (sqrt 2 x 10 %). A plant-feed batch test would settle it.
    dn_acid: float = 0.3             # chosen: the plant's acid exponent, 3.3 against the model's 3.0
    # Judgement call on dn_acid: the free-exponent fit gave 3.9 (Nd) and 4.5 (Pr), so a real plant
    # could sit further from 3 than 0.3. Same test settles it.
    k_low: float = 0.85              # chosen: worst stage at 85 % of nominal transfer rate
    sigma_rho: float = 0.02          # chosen: densities to +-2 %
    sigma_volume: float = 0.05       # chosen: mixer hold-up to +-5 %
    # How the extraction constants are drawn (decision 31):
    #   "independent"  each K gets its own random factor (sigma_K, 2 x for La+Ce). Used for the
    #                  report's plant (seed 1), whose draw lands inside every published range. But the
    #                  ratios between the K's, which are what the separation depends on, are not held
    #                  anywhere: across 20 seeds, five drew separation factors no published circuit has
    #                  (Pr/La+Ce about 1.0, or Nd extracting less than Pr). Notebook 17.
    #   "ratios"       the separation factors are drawn evenly within their published ranges, widened
    #                  10 % on each side, with no tails; a common factor moves all K together (extractant
    #                  strength, temperature). Used for the seed study.
    generator: str = "independent"
    beta_pr_lace: tuple = (1.8 / 1.1, 2.2 * 1.1)   # Pr/(La+Ce): YIN18 industrial 1.8-2.2, +-10 %
    beta_nd_pr: tuple = (1.3 / 1.1, 1.59 * 1.1)    # Nd/Pr: YIN18 industrial 1.3-1.5, IDAHO mixed 1.57-1.59, +-10 %
    beta_sm_nd: tuple = (6.0, 14.0)                # Sm/Nd: handbook ~6-7 to CARESTER Fig. 1 ~14 (parameters.toml)
    common_K: tuple = (0.9, 1.1)                   # chosen: all K together, extractant strength / temperature
    # Judgement call on what isn't perturbed: the extractant concentration (in a plant it drifts
    # with losses and make-up; D goes as its cube, so 5 % low is about 15 % off on D, and the
    # capacity drops 5 %), temperature, and impurities that load the organic over time (iron, for
    # instance). Those would be the next ones to add; a few months of organic titrations on a
    # plant would show how far they really move.


# ---------------------------------------------------------------------------
# The plant's parameters
# ---------------------------------------------------------------------------

def perturbed_params(p: Params, m: Mismatch) -> Params:
    """A new Params for the plant. Deterministic in the seed.

    Parameters
    ----------
    p   the nominal parameters (the model's); not changed
    m   the perturbation settings

    Returns
    -------
    A copy of p with K, the acid exponent, the organic density, the salt's
    molal volume and the mixer volume changed. Everything else is p's.

    Example
    -------
        plant_p = perturbed_params(load_params(), Mismatch(seed=1))
    """
    # One generator for all the draws below. Their order is fixed (see the module docstring).
    rng = np.random.default_rng(m.seed)
    strength = m.strength

    # Draws 1..n: the extraction constants.
    K = _perturbed_extraction_constants(p, m, rng)

    # Draw n+1: the mixer hold-up volume, L.   V_mix -> V_mix * (1 + N(0, sigma_volume))
    # It lives in the raw parameter dict, so work on a deep copy and leave p alone.
    raw = copy.deepcopy(p.raw)
    geo = raw["stage_geometry"]
    geo["mixer_volume_L"] = (p.raw["stage_geometry"]["mixer_volume_L"]
                             * (1 + rng.normal(0.0, m.sigma_volume * strength)))

    # The acid exponent has no draw: n -> n + dn.
    # Draw n+2: organic base density, g/mL.    rho -> rho * (1 + N(0, sigma_rho))
    # Draw n+3: salt's molal volume, mL/mol.   phi -> phi * (1 + N(0, 5 * sigma_rho))
    # (The keyword arguments are evaluated top to bottom, so rho_org0 is drawn before phi_v.)
    return replace(
        p, raw=raw, K=K,
        n_acid=p.n_acid + m.dn_acid * strength,
        rho_org0=p.rho_org0 * (1 + rng.normal(0.0, m.sigma_rho * strength)),
        phi_v=p.phi_v * (1 + rng.normal(0.0, 5 * m.sigma_rho * strength)),
    )


def _perturbed_extraction_constants(p: Params, m: Mismatch, rng: np.random.Generator) -> np.ndarray:
    """The plant's extraction constants, one normal draw per component, in component order.

        K_i -> K_i * exp(N(0, spread * strength))
        spread = 2 * sigma_K for "LaCe" (its constant is chosen, not fitted), sigma_K otherwise

    A multiplicative log-normal factor keeps every K positive. With m.generator "ratios" the
    constants come from drawn separation factors instead (_extraction_constants_from_ratios).

    Raises ValueError for a generator other than "independent" or "ratios".
    """
    if m.generator == "ratios":
        return _extraction_constants_from_ratios(p, m, rng)
    if m.generator != "independent":
        raise ValueError(f"unknown generator {m.generator!r}: 'independent' or 'ratios'")
    return _independent_extraction_constants(p, m, rng)


def _independent_extraction_constants(p: Params, m: Mismatch, rng: np.random.Generator) -> np.ndarray:
    """Generator "independent": each K gets its own log-normal factor, in component order."""
    K = p.K.copy()
    for i, name in enumerate(p.names):
        spread = m.sigma_K * (2.0 if name == "LaCe" else 1.0)
        K[i] = p.K[i] * np.exp(rng.normal(0.0, spread * m.strength))
    return K


def _extraction_constants_from_ratios(p: Params, m: Mismatch, rng: np.random.Generator) -> np.ndarray:
    """The plant's extraction constants built from drawn separation factors (generator "ratios").

        common ~ U(common_K)                   all K move together
        K_Pr   = K_Pr(model) x common
        K_LaCe = K_Pr / beta_pr_lace,   K_Nd = K_Pr x beta_nd_pr,   K_Sm = K_Nd x beta_sm_nd

    Each factor is drawn evenly in log space within its range, then moved toward the model's own
    value by `strength` (strength 0 gives the model's ratios back). Draw order: common, Pr/La+Ce,
    Nd/Pr, Sm/Nd."""
    names = list(p.names)
    i_la, i_pr, i_nd, i_sm = (names.index(c) for c in ("LaCe", "Pr", "Nd", "SmPlus"))
    model = {"pr_lace": p.K[i_pr] / p.K[i_la], "nd_pr": p.K[i_nd] / p.K[i_pr], "sm_nd": p.K[i_sm] / p.K[i_nd]}

    def draw(lo_hi: tuple, nominal: float) -> float:
        """One factor, drawn evenly in log space within lo_hi, pulled toward `nominal` by strength."""
        value = float(np.exp(rng.uniform(np.log(lo_hi[0]), np.log(lo_hi[1]))))
        return float(nominal * (value / nominal) ** m.strength)

    common = draw(m.common_K, 1.0)
    pr_lace, nd_pr, sm_nd = (draw(m.beta_pr_lace, model["pr_lace"]), draw(m.beta_nd_pr, model["nd_pr"]),
                             draw(m.beta_sm_nd, model["sm_nd"]))
    K = p.K.copy()
    K[i_pr] = p.K[i_pr] * common
    K[i_la] = K[i_pr] / pr_lace
    K[i_nd] = K[i_pr] * nd_pr
    K[i_sm] = K[i_nd] * sm_nd
    return K


# ---------------------------------------------------------------------------
# The plant's stages
# ---------------------------------------------------------------------------

def stage_efficiency_factors(N: int, m: Mismatch) -> np.ndarray:
    """Per-stage multipliers on the transfer rate, the plant's own and unknown to the controller.

    Drawn from their own generator, seeded m.seed + 1000, so these draws don't
    overlap the ones in perturbed_params.

        lowest = 1 - (1 - k_low) * strength        (0.85 at strength 1)
        factor_j ~ U(lowest, 1)                    for j = 0 .. N-1

    Parameters
    ----------
    N   number of stages
    m   the perturbation settings

    Returns
    -------
    An array of N multipliers, each between `lowest` and 1.
    """
    rng = np.random.default_rng(m.seed + 1000)
    lowest = 1.0 - (1.0 - m.k_low) * m.strength
    return rng.uniform(lowest, 1.0, size=N)
