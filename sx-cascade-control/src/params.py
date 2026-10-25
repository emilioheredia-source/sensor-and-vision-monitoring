"""Read the parameter files into plain Python objects.

Where this sits
---------------
Every physical number the model uses lives in two TOML files in params/, never in the code
(ARCHITECTURE.md, decision 4):

  parameters.toml   every physical number, each one labelled measured, chosen or derived,
                    with its source written next to it
  feeds.toml        Carester Table 1: the rare-earth make-up of real deposits, in % of total
                    rare-earth oxide by mass

This module loads them and hands the numbers to the rest of the code. chemistry.py and
circuit.py read everything they need from the Params object made here.

What happens here
-----------------
Nothing here does chemistry. The module loads the files, converts units, and works out the
few numbers that parameters.toml marks as "derived in code":

  dimer_total = (extractant monomer, mol/L) * (dimer fraction) / 2
      The extractant (PC88A) pairs up into dimers H2L2 in the kerosene, so 1 mol/L of
      monomer is 0.5 mol/L of dimer.

  k_transfer = (f / (1 - f)) / tau
      A stirred stage gets the fraction k*tau / (1 + k*tau) of the way to equilibrium, with
      tau the mixer residence time. The Idaho lab saw equilibrium in 3 minutes; that's read as
      f = 95 % at tau = 0.05 h, which gives k = 380 per hour (APPENDIX_A_model.md, A.5).

It also turns a deposit's oxide percentages into feed concentrations in mol/L, lump by lump
(the four lumps are La+Ce, Pr, Nd and Sm-and-heavier; ARCHITECTURE.md, decision 6).
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The folder that holds parameters.toml and feeds.toml.
PARAMS_DIR = Path(__file__).resolve().parent.parent / "params"

# Which columns of Carester Table 1 add up to each of the model's four lumps. The table lists
# oxides; a lump's share of the feed is the sum of its columns.
# Why Ce goes in as Ce(III): in a chloride liquor cerium is trivalent; Ce(IV) would oxidize the
# chloride (textbook). The CeO2 in the table is only how assays report it.
# Why Y goes with the heavy lump: with PC88A in chloride, yttrium extracts among the heavy
# lanthanides (textbook). The lump is removed upstream in the NdPr runs anyway.
# Judgement call: no cerium removal ahead of SX. Many monazite flowsheets oxidize and take out
# most of the cerium first. With that, the La+Ce load on the Ce|Pr cut drops a lot and the
# extraction section and organic flow change. A real feed assay settles which case applies.
_LUMP_COLUMNS = {
    "LaCe": ["La2O3_CeO2"],
    "Pr": ["Pr6O11"],
    "Nd": ["Nd2O3"],
    "SmPlus": ["Sm_Eu_Gd_2O3", "Tb4O7", "Dy2O3", "Ho_Er_Tm_Yb_Lu_2O3", "Y2O3"],
}


# --------------------------------------------------------------------------
# The parameter object
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    """Every physical number the model needs, in the units the equations use.

    Arrays have one entry per component, in the order of `names` (light to heavy). The whole
    parameters.toml is kept in `raw` for numbers that aren't lifted into their own field (the
    stage geometry, for instance, is read from raw["stage_geometry"]).

    Frozen, so a Params can be shared safely. To try a different number, make a new one with
    dataclasses.replace (the tests do this) or edit the dict and call params_from_dict.
    """
    raw: dict                    # the whole parameters.toml, for anything not lifted below
    names: tuple[str, ...]       # component (lump) names, light to heavy: LaCe, Pr, Nd, SmPlus
    M: np.ndarray                # g/mol of metal, per component
    oxide_per_metal: np.ndarray  # g of oxide per mol of metal, per component
    K: np.ndarray                # equilibrium constant K_i in the distribution law, per component
    n_acid: float                # acid exponent n in D = K L^3 / (h^n + h_min^n)
    h_min_D: float               # mol/L, acid floor inside D (see parameters.toml); 0 = off
    D_max: float                 # smooth cap on D (see parameters.toml); 0 = off
    dimer_total: float           # mol/L of extractant dimer (H2L2) in the organic, free or holding metal
    ligands_per_metal: int       # dimers taken up per metal ion: 3, from RE3+ + 3 H2L2
    k_transfer: float            # 1/h, rate constant of the fast stage model
    rho_org0: float              # g/mL, density of the organic with no metal in it
    rho_water: float             # g/mL, density of water
    phi_v: float                 # mL/mol, apparent molal volume of the RECl3 salt in water
    M_cl: float                  # g/mol, molar mass of chlorine

    def index(self, name: str) -> int:
        """Position of a component in the arrays above, e.g. p.index("Nd") == 2.

        Raises ValueError if there's no component of that name.
        """
        return self.names.index(name)


# --------------------------------------------------------------------------
# Loading parameters.toml
# --------------------------------------------------------------------------
def load_params(path: Path | None = None) -> Params:
    """Read parameters.toml and return it as Params.

    path: the file to read. Leave it out to read params/parameters.toml.
    Raises FileNotFoundError if the file isn't there, and KeyError if a needed entry is missing.

    Example:
        p = load_params()
        p.K[p.index("Nd")]      # 0.0321
    """
    with open(path or PARAMS_DIR / "parameters.toml", "rb") as f:
        return params_from_dict(tomllib.load(f))


def params_from_dict(p: dict) -> Params:
    """Params from the parameter file's contents, as a nested dict.

    The dict has the same layout as parameters.toml. It can be the file just read, or an edited
    copy of it, as a front end or a run spec may make (runs.py overlays a run's own parameters on
    the file this way).

    p: the nested dict, as tomllib.load returns it.
    Returns Params, with the derived numbers worked out (see the module docstring).
    Raises KeyError if a needed entry is missing.
    """
    # Per-component numbers, put in the component order the file gives.
    names = tuple(p["components"]["names"])
    M = np.array([p["components"]["molar_mass"][c] for c in names], float)
    oxide = np.array([p["components"]["oxide_mass_per_metal"][c] for c in names], float)
    K = np.array([p["equilibrium"]["K"][c] for c in names], float)

    # Extractant: monomer mol/L times the share that's dimerized, halved (two monomers per dimer).
    # Why all dimer: acidic organophosphorus extractants sit as dimers in kerosene-type diluents
    # (textbook), and the Idaho Eq. 4 is written that way.
    ext = p["extractant"]
    dimer_total = ext["concentration_monomer"] * ext["dimer_fraction"] / 2

    k_transfer = _transfer_rate_constant(p["kinetics"])

    d = p["densities"]
    return Params(
        raw=p, names=names, M=M, oxide_per_metal=oxide, K=K,
        n_acid=float(p["equilibrium"]["acid_exponent"]),
        h_min_D=float(p["equilibrium"]["acid_floor_mol_per_L"]),
        D_max=float(p["equilibrium"].get("D_max", 0.0)),
        dimer_total=dimer_total, ligands_per_metal=int(ext["ligands_per_metal"]),
        k_transfer=float(k_transfer),
        rho_org0=d["organic_unloaded_g_per_mL"], rho_water=d["aqueous_water_g_per_mL"],
        phi_v=d["apparent_molal_volume_RECl3_mL_per_mol"], M_cl=d["chloride_molar_mass"],
    )


def _transfer_rate_constant(kin: dict) -> float:
    """Rate constant k of the fast stage model, 1/h, from the [kinetics] table.

    A stirred stage approaches equilibrium to the fraction k*tau / (1 + k*tau) (its Murphree
    efficiency), with tau the mixer residence time in hours. Solving for k when the stage reaches
    the fraction f at tau gives k = (f / (1 - f)) / tau. With f = 0.95 and tau = 0.05 h (the
    Idaho lab's three minutes) that's 380 per hour.

    kin: the [kinetics] table, with approach_fraction (0..1) and mixer_time_to_equilibrium_h.

    Judgement call: one k for every element, for extraction and stripping alike, carried over
    from a 120 mL lab mixer. How fast a mixer transfers depends on how hard it's stirred and how
    fine the drops are, and that doesn't scale up by itself. If the plant mixers are slower,
    stage efficiency drops and the stage count rises (REPORT section 3 shows by how much). Stage
    samples on a pilot or plant mixer at design flow would settle it.
    """
    f = kin["approach_fraction"]
    return (f / (1 - f)) / kin["mixer_time_to_equilibrium_h"]


# --------------------------------------------------------------------------
# Feeds: from oxide percentages to mol/L
# --------------------------------------------------------------------------
def load_feeds(path: Path | None = None) -> dict:
    """Read feeds.toml: one table per deposit, % of total rare-earth oxide by column.

    path: the file to read. Leave it out to read params/feeds.toml.
    Returns the nested dict, keyed by deposit name ("eneabba", "yangibana", ...).
    """
    with open(path or PARAMS_DIR / "feeds.toml", "rb") as f:
        return tomllib.load(f)


def feed_mole_fractions(deposit: str, params: Params, removed: tuple[str, ...] = (),
                        feeds: dict | None = None) -> np.ndarray:
    """Mole fraction of each lump in the feed, after some lumps are taken out upstream.

    Carester gives % of oxide by mass. Moles of metal are proportional to
    (oxide %) / (oxide mass per mol metal), lump by lump. Lumps named in `removed` are set to
    zero before the fractions are normalized to sum to 1.

    deposit: a deposit name in feeds.toml, e.g. "eneabba".
    params:  the Params, for the component order and the oxide mass per mol of metal.
    removed: names of lumps taken out before solvent extraction, e.g. ("SmPlus",).
    feeds:   the feeds dict, if already loaded; otherwise feeds.toml is read.
    Returns an array of n mole fractions (dimensionless), summing to 1.
    Raises KeyError for an unknown deposit, ValueError for an unknown lump in `removed`.
    """
    feeds = feeds or load_feeds()
    col = feeds[deposit]
    # moles of metal per 100 g of oxide, for each lump
    moles = np.array([
        sum(col[k] for k in _LUMP_COLUMNS[c]) / params.oxide_per_metal[i]
        for i, c in enumerate(params.names)
    ])
    for c in removed:
        moles[params.index(c)] = 0.0
    return moles / moles.sum()


def feed_concentrations(deposit: str, total_RE_g_per_L: float, params: Params,
                        removed: tuple[str, ...] = (), feeds: dict | None = None) -> np.ndarray:
    """Feed concentration of each lump, mol/L, such that the metals together weigh total_RE_g_per_L.

    The mole fractions come from feed_mole_fractions. Dividing the total metal mass (g/L) by the
    mean molar mass of the mix (g/mol) gives the total mol/L; each lump gets its fraction of it.

    deposit:          a deposit name in feeds.toml.
    total_RE_g_per_L: g/L of rare-earth metal in the feed liquor (all lumps together).
    params, removed, feeds: as in feed_mole_fractions.
    Returns an array of n concentrations, mol/L.

    Example:
        c_feed = feed_concentrations("eneabba", 50.0, p, removed=("SmPlus",))

    Note the basis: grams of metal per litre, not oxide (50 g/L metal is about 60 g/L as oxide).
    Judgement call: the Carester table is a deposit's make-up, not a leach liquor's. Leaching
    doesn't take every element out in the same share, so the real liquor can differ. A leach
    liquor assay would settle it.

    Why five parameters: the deposit and the total, the parameter file, and two optional keywords (elements removed, the feed table).
    """
    x = feed_mole_fractions(deposit, params, removed, feeds)
    mean_M = float(x @ params.M)           # g per mol of mixed metal
    return x * total_RE_g_per_L / mean_M
