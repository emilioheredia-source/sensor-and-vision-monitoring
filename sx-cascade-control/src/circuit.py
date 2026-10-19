"""The circuit: a connection map (data) plus the equations of one stage, assembled into one ODE
right-hand side in CasADi.

Where this sits
---------------
This is the model at the centre of the project. The simulator integrates it, the design
optimizer (design.py) finds operating points on it, the MPC (mpc.py) linearizes it, and the
estimator predicts with it. All of them call the same CasADi function built here, so the
equations are written once (ARCHITECTURE.md, decision 1). plant_models.py builds plants with
extra physics out of the same pieces.

What a circuit is
-----------------
A row of N mixer-settler stages in three sections: extraction, scrub, strip. Stages are numbered
from 0 at the raffinate end. The organic flows from stage 0 to stage N-1 and is recycled back to
stage 0 after the strip. The aqueous flows the other way, along two paths that never meet:

  path A  scrub liquor enters the last scrub stage, the feed joins at the last extraction stage,
          and the mix leaves stage 0 as raffinate
  path B  strip acid enters stage N-1 and leaves the first strip stage as product

Which stream goes where is data, a list of edges (phase, from, to), built by build_map. The
equations only read the edges, so a hand-edited map runs without code changes (decision 3).

For the main case, the NdPr product, the cut is between Ce and Pr: La and Ce leave in the
raffinate, Pr and Nd go together to the product.

What one stage is, in equations
-------------------------------
Each phase in a stage is one well-mixed tank with a fixed hold-up (mixer and settler together,
decision 11). For component i in stage j (APPENDIX_A_model.md, A.5 to A.8):

  transfer, mol/h of metal from aqueous to organic, proportional to the distance from
  equilibrium:
      r_ij = k * kappa_j * V_mix * (1 - phi_j) * (D_ij * c_aq_ij - c_org_ij)

  metal balances (in - out -+ transfer):
      V_aq,j  * dc_aq_ij/dt  = sum of Q*c in - Q_aq,j * c_aq_ij  - r_ij
      V_org,j * dc_org_ij/dt = sum of Q*c in - Q_org  * c_org_ij + r_ij

  acid balance: three H+ released per metal ion extracted, minus the saponification base:
      V_aq,j * dh_j/dt = sum of Q*h in - Q_aq,j * h_j + 3 * sum_i r_ij - dose_j

  interface: relaxes toward where the jackleg's hydrostatic balance puts it:
      z_rest,j = (rho_aq * H_jackleg,j - rho_org * H_weir) / (rho_aq - rho_org)
      dz_j/dt  = (z_rest,j - z_j) / tau_z

  phi_j = Q_aq,j / (Q_aq,j + Q_org) is the aqueous share of the mixer. D comes from
  chemistry.distribution, the densities from chemistry.rho_aqueous and rho_organic.

State vector x, for N stages and n components:
    [ c_aq (N*n) | c_org (N*n) | h (N) | z (N) ]
    c_aq, c_org  mol/L, stage-major (stage 0's components first)
    h            aqueous H+, mol/L
    z            interface height in the settler, m

Input vector u: see Inputs. Units throughout: hours, mol/L, L, L/h, m, g/mL (decision 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import casadi as ca
import numpy as np

from . import chemistry
from .params import Params

# The aqueous streams that enter the circuit from outside, in the order of the columns of
# Circuit._aq_coef and of the source flows in Circuit.aqueous_flows.
SOURCES_AQ = ("feed", "scrub", "strip")


# --------------------------------------------------------------------------
# The map: which stream goes where
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Edge:
    """One stream in the flowsheet: a phase flowing from one place to another.

    A place is either a stage index (int) or a name (str). A named source is where a stream
    comes in from outside ("feed", "scrub", "strip", "organic"); a named sink is where one leaves
    ("raffinate", "product", "spent_organic").
    """
    phase: str          # "aq" or "org"
    src: int | str      # stage index, or a source name
    dst: int | str      # stage index, or a sink name


@dataclass
class CircuitMap:
    """The flowsheet: how many stages per section, and the list of streams between them."""
    n_ext: int                # extraction stages, numbered 0 .. n_ext-1
    n_scrub: int              # scrub stages, next after the extraction
    n_strip: int              # strip stages, the last ones, ending at N-1
    edges: list[Edge]         # every stream, see build_map
    organic_recycle: bool     # True: the organic leaving stage N-1 comes back into stage 0

    @property
    def N(self) -> int:
        """Total number of stages."""
        return self.n_ext + self.n_scrub + self.n_strip

    def section(self, j: int) -> str:
        """Name of the section stage j is in: "extraction", "scrub" or "strip"."""
        if j < self.n_ext:
            return "extraction"
        if j < self.n_ext + self.n_scrub:
            return "scrub"
        return "strip"

    @property
    def product_stage(self) -> int | None:
        """Stage whose aqueous outflow is the strip product (None if no strip section)."""
        return self.n_ext + self.n_scrub if self.n_strip > 0 else None


def build_map(n_ext: int, n_scrub: int, n_strip: int,
              organic_recycle: bool | None = None) -> CircuitMap:
    """The Idaho flowsheet layout as a list of edges (ARCHITECTURE.md, 'The circuit').

    Stages are numbered from the raffinate end. Organic goes 0 -> N-1.
    Aqueous path A: scrub liquor in at the last scrub stage, feed joins at
    the last extraction stage, raffinate out of stage 0.
    Aqueous path B: strip acid in at stage N-1, product out of the first
    strip stage.

    n_ext, n_scrub, n_strip: number of stages in each section.
    organic_recycle: send the spent organic back to stage 0. Leave it out to recycle whenever
        there's a strip section to clean the organic first.
    Returns the CircuitMap.
    Raises ValueError with no extraction stage (the feed has nowhere to go), or when asked to
    recycle the organic without a strip section.

    Example:
        cmap = build_map(2, 12, 4)       # 18 stages; product leaves stage 14

    Why: extraction, scrub and strip in one row with the organic recycled is the standard layout
    of a fractional-extraction cascade, and the one the Idaho rig ran (Figs. 17 and 21).
    Judgement call: nothing sits between the strip and stage 0. A plant has a stripped-organic
    tank, often saponification contactors, and sometimes a wash stage. Those add hold-up and delay
    on the recycle and slow the circuit's slow modes; they don't change the steady state. The
    plant's tank and contactor volumes would settle how much.
    """
    if n_ext < 1:
        raise ValueError("need at least one extraction stage (the feed enters there)")
    if organic_recycle is None:
        organic_recycle = n_strip > 0
    if organic_recycle and n_strip == 0:
        raise ValueError("cannot recycle the organic without a strip section")

    N = n_ext + n_scrub + n_strip
    edges = _organic_edges(N) + _aqueous_path_A(n_ext, n_scrub) + _aqueous_path_B(n_ext, n_scrub, n_strip)
    return CircuitMap(n_ext, n_scrub, n_strip, edges, organic_recycle)


def _organic_edges(N: int) -> list[Edge]:
    """The organic, end to end: in at stage 0, through every stage, out of stage N-1."""
    e = [Edge("org", "organic", 0)]
    e += [Edge("org", j, j + 1) for j in range(N - 1)]
    e.append(Edge("org", N - 1, "spent_organic"))
    return e


def _aqueous_path_A(n_ext: int, n_scrub: int) -> list[Edge]:
    """Aqueous path A, extraction plus scrub: scrub liquor in at the last scrub stage (if there's
    a scrub section), feed in at the last extraction stage, then stage by stage down to stage 0,
    which lets out the raffinate."""
    e: list[Edge] = []
    last_A = n_ext + n_scrub - 1
    if n_scrub > 0:
        e.append(Edge("aq", "scrub", last_A))
    e.append(Edge("aq", "feed", n_ext - 1))
    e += [Edge("aq", j, j - 1) for j in range(last_A, 0, -1)]
    e.append(Edge("aq", 0, "raffinate"))
    return e


def _aqueous_path_B(n_ext: int, n_scrub: int, n_strip: int) -> list[Edge]:
    """Aqueous path B, the strip: strip acid in at the last stage, down through the strip section
    only, product out of the first strip stage. No edges if there's no strip section."""
    if n_strip == 0:
        return []
    N = n_ext + n_scrub + n_strip
    first_B = n_ext + n_scrub
    e = [Edge("aq", "strip", N - 1)]
    e += [Edge("aq", j, j - 1) for j in range(N - 1, first_B, -1)]
    e.append(Edge("aq", first_B, "product"))
    return e


# --------------------------------------------------------------------------
# Inputs: everything the circuit is given from outside, in one fixed layout
# --------------------------------------------------------------------------
@dataclass
class Inputs:
    """Everything the circuit is given from outside. Flows in L/h, concentrations in mol/L,
    heights in m.

    Handles and disturbances share this one vector; which entries a
    controller may move is decided in mpc.MPCSettings (handles, jacklegs), not here.

    The fields hold numbers when the simulator runs, and CasADi symbols when the equations are
    built (from_vector). The vector order is the field order below; layout() gives each field's
    slice.
    """
    Q_org: float                # organic flow, L/h (the same through every stage)
    Q_feed: float               # feed liquor flow, L/h
    Q_scrub: float              # scrub liquor flow, L/h
    Q_strip: float              # strip acid flow, L/h
    c_feed: np.ndarray          # (n,) feed concentration of each component, mol/L
    h_feed: float               # feed acidity, mol/L H+
    c_scrub: np.ndarray         # (n,) scrub liquor metal, mol/L; zeros for an acid-only scrub
    h_scrub: float              # scrub liquor acidity, mol/L H+
    h_strip: float              # strip acid strength, mol/L H+
    saponified_fraction: float  # share of the extractant monomer saponified (pre-neutralized), 0..1
    H_jackleg: np.ndarray       # (N,) jackleg height per stage, m: the handle on each interface
    k_factor: np.ndarray | None = None   # (N,) multiplier on the transfer rate per stage: 1 = nominal
                                         # stage efficiency; <1 = a tired mixer; mixer speed acts here

    @staticmethod
    def size(n: int, N: int) -> int:
        """Length of the input vector for n components and N stages."""
        # 4 flows, c_feed, h_feed, c_scrub, h_scrub, h_strip, saponified_fraction, H_jackleg, k_factor
        return 4 + n + 1 + n + 1 + 1 + 1 + N + N

    @staticmethod
    def layout(n: int, N: int) -> dict:
        """Where each field sits in to_vector(), as slices, e.g. layout(n, N)["Q_org"] == slice(0, 1)."""
        i, lay = 0, {}
        for name, k in (("Q_org", 1), ("Q_feed", 1), ("Q_scrub", 1), ("Q_strip", 1),
                        ("c_feed", n), ("h_feed", 1), ("c_scrub", n), ("h_scrub", 1), ("h_strip", 1),
                        ("saponified_fraction", 1), ("H_jackleg", N), ("k_factor", N)):
            lay[name] = slice(i, i + k)
            i += k
        return lay

    def to_vector(self):
        """The inputs as one CasADi column, in the layout above. A missing k_factor counts as all
        ones (every stage at nominal efficiency)."""
        kf = self._default_k_factor() if self.k_factor is None else self.k_factor
        return ca.vertcat(self.Q_org, self.Q_feed, self.Q_scrub, self.Q_strip,
                          self.c_feed, self.h_feed, self.c_scrub, self.h_scrub,
                          self.h_strip, self.saponified_fraction, self.H_jackleg, kf)

    def _default_k_factor(self) -> np.ndarray:
        """All ones, one per stage (the number of stages read from H_jackleg)."""
        return np.ones(len(np.atleast_1d(np.asarray(ca.DM(self.H_jackleg)).ravel())))

    @classmethod
    def from_vector(cls, v, n: int, N: int) -> "Inputs":
        """Inputs read back from a vector in the to_vector layout.

        v: the vector, numeric or a CasADi symbol (that's how the equations get symbolic inputs).
        n, N: number of components and of stages.
        """
        i = 0                       # read position in v

        def scalar():
            """The next single entry of v."""
            nonlocal i
            out = v[i]
            i += 1
            return out

        def vec(k):
            """The next k entries of v."""
            nonlocal i
            out = v[i:i + k]
            i += k
            return out

        return cls(Q_org=scalar(), Q_feed=scalar(), Q_scrub=scalar(), Q_strip=scalar(),
                   c_feed=vec(n), h_feed=scalar(), c_scrub=vec(n), h_scrub=scalar(),
                   h_strip=scalar(), saponified_fraction=scalar(), H_jackleg=vec(N), k_factor=vec(N))

    def replace(self, **changes) -> "Inputs":
        """A copy with some fields changed (disturbances and controllers use this)."""
        from dataclasses import replace as _replace
        return _replace(self, **changes)

    def numpy(self) -> np.ndarray:
        """The input vector as a flat numpy array."""
        return np.array(ca.DM(self.to_vector())).ravel()


# --------------------------------------------------------------------------
# What leaves each stage, as the equations see it
# --------------------------------------------------------------------------
class _StageOutlets(NamedTuple):
    """The streams leaving each stage, one list entry per stage (symbolic).

    Each stage's outlet carries its contents: in the model that's the stage's own state; in the
    settler plant (plant_models.py) it's the last settler cell. Kept together because the inflows
    and the outlet function both need all four.
    """
    Q_aq: ca.SX      # (N,1) aqueous flow out of each stage, L/h
    c_aq: list       # aqueous concentrations leaving each stage, (n,) mol/L per stage
    h: list          # aqueous acid leaving each stage, mol/L
    c_org: list      # organic concentrations leaving each stage, (n,) mol/L per stage


# --------------------------------------------------------------------------
# The circuit model
# --------------------------------------------------------------------------
@dataclass
class Circuit:
    """The controller's model of the circuit: map plus parameters, built into CasADi functions.

    Building one (at construction) makes the right-hand side xdot = f(x, u), its Jacobians, and
    a function for the streams that leave the circuit. The numeric interface (rhs, jac, jac_u,
    outlets) evaluates them; casadi_rhs hands the symbolic function to anyone who wants it.

    Raises NotImplementedError for a stage model other than "fast", and ValueError for a reflux
    scrub without a strip section.

    Example:
        circ = Circuit(build_map(2, 12, 4), load_params())
        x0 = circ.initial_state(u)            # u: an Inputs, e.g. config.nominal_inputs
        xdot = circ.rhs(x0, u.numpy())
    """
    cmap: CircuitMap                  # the flowsheet
    p: Params                         # the physical numbers
    model: str = "fast"               # "fast" (rate) or "exact" (equilibrium) - exact not built yet
    scrub_reflux: bool = False        # scrub liquor carries the product's own metal (part of the
                                      # strip product sent back), on top of Inputs.c_scrub
    acid_floor: float = 1e-4          # mol/L; base dose fades out below this (smooth)
    _f: ca.Function = field(init=False, repr=False)      # xdot = f(x, u)
    _out: ca.Function = field(init=False, repr=False)    # the streams leaving the circuit

    def __post_init__(self):
        """Check the settings, work out which sources feed each stage, and build the functions.

        Raises NotImplementedError for a stage model other than "fast", and ValueError for a
        reflux scrub on a map without a strip section.
        """
        if self.model != "fast":
            raise NotImplementedError(f"stage model '{self.model}' not built yet; use 'fast'")
        if self.scrub_reflux and self.cmap.product_stage is None:
            raise ValueError("reflux scrub needs a strip section to take the product from")
        self._aq_coef = self._aqueous_flow_coefficients()
        self._build()

    # ---- sizes and packing -------------------------------------------------
    @property
    def n(self) -> int:
        """Number of components."""
        return len(self.p.names)

    @property
    def N(self) -> int:
        """Number of stages."""
        return self.cmap.N

    @property
    def nx(self) -> int:
        """Length of the state vector: c_aq and c_org (N*n each), then h and z (N each)."""
        return 2 * self.N * self.n + 2 * self.N

    @property
    def nx_model(self) -> int:
        """Length of the state in this model's layout (what `observable` returns). A richer plant
        has a longer state that starts with this layout."""
        return 2 * self.N * self.n + 2 * self.N

    def unpack(self, x):
        """Split a state vector into per-stage pieces.

        Returns (c_aq, c_org, h, z): lists of length N. c_aq[j] and c_org[j] are stage j's n
        concentrations (mol/L); h[j] is its acid (mol/L) and z[j] its interface height (m).
        Works on numpy arrays and CasADi symbols alike.
        """
        N, n = self.N, self.n
        c_aq = [x[j * n:(j + 1) * n] for j in range(N)]
        c_org = [x[N * n + j * n: N * n + (j + 1) * n] for j in range(N)]
        h = [x[2 * N * n + j] for j in range(N)]
        z = [x[2 * N * n + N + j] for j in range(N)]
        return c_aq, c_org, h, z

    def pack(self, c_aq, c_org, h, z) -> np.ndarray:
        """The opposite of unpack: one flat state vector from (N, n), (N, n), (N,), (N,) arrays."""
        return np.concatenate([np.ravel(c_aq), np.ravel(c_org), np.ravel(h), np.ravel(z)])

    # ---- flows: linear in the source flows, coefficients from the map ---------
    def _aqueous_flow_coefficients(self) -> np.ndarray:
        """coef[j, s] = 1 if aqueous source s reaches stage j (flows are conserved,
        so every stage's aqueous outflow is the sum of the sources upstream of it).

        Returns an (N, 3) array of 0 and 1, columns in SOURCES_AQ order.
        """
        coef = np.zeros((self.N, len(SOURCES_AQ)))
        aq = [e for e in self.cmap.edges if e.phase == "aq"]
        for s_i, s in enumerate(SOURCES_AQ):
            coef[sorted(_stages_reached(s, aq)), s_i] = 1.0
        return coef

    def aqueous_flows(self, u: Inputs):
        """Aqueous flow out of each stage, L/h, as an (N,1) column.

        Flows are conserved, so each stage passes on the sum of the source flows (feed, scrub,
        strip) upstream of it: coefficient matrix times source flows.

        Judgement call: no volume change between the phases. Water carried by a saponified
        organic and the small mutual solubility of the phases are left out. That's a few percent
        at most on a flow, and it doesn't touch the metal balance. A plant flow survey would
        settle it.
        """
        src = ca.vertcat(u.Q_feed, u.Q_scrub, u.Q_strip)
        return ca.mtimes(ca.DM(self._aq_coef), src)        # (N,1) outflow of each stage

    # ---- the equations -----------------------------------------------------
    def _build(self):
        """Build the right-hand side and the outlet function in CasADi.

        Each stage one well-mixed volume per phase (mixer and settler hold-up together); the
        streams leaving a stage carry its contents. plant_models.py builds a richer plant from the
        same pieces.
        """
        N, n = self.N, self.n

        # Symbols: the state, the input vector, and the input vector read as Inputs.
        x = ca.SX.sym("x", self.nx)
        uv = ca.SX.sym("u", Inputs.size(n, N))
        u = Inputs.from_vector(uv, n, N)
        state = self.unpack(x)                     # (c_aq, c_org, h, z), per stage
        self._read_hidden(x)                       # a plant subclass may keep hidden states (none here)

        # Flows. In this model what leaves a stage is the stage's own state.
        c_aq, c_org, h, _ = state
        Q_aq = self.aqueous_flows(u)
        outlets = _StageOutlets(Q_aq, c_aq, h, c_org)

        # mol/h of metal and acid into each stage, from the sources and the neighbours, plus any
        # extra transport a plant subclass adds (none here).
        inflows = self._inflows(u, outlets)
        self._carry_over(u, Q_aq, state, inflows)
        doses = self._saponification_doses(u, h)

        # Stack the derivatives in the state's own layout and make the functions.
        dc_aq, dc_org, dh, dz = self._stage_balances(u, Q_aq, state, inflows, doses)
        self._make_rhs_functions(x, uv, ca.vertcat(*dc_aq, *dc_org, *dh, *dz, *self._hidden_rates()))
        self._make_outlet_function(x, uv, u, outlets)

    def _stage_balances(self, u: Inputs, Q_aq, state, inflows, doses):
        """The balances of every stage: (in - out -+ transfer) / hold-up, mol/L/h (and m/h for z).

        u: the inputs. Q_aq: aqueous flow out of each stage, L/h. state: (c_aq, c_org, h, z) per
        stage. inflows: (aq_mol_in, aq_acid_in, org_mol_in) per stage, mol/h. doses: the
        saponification base spent in each stage, mol/h.
        Returns (dc_aq, dc_org, dh, dz): lists of N entries, in organic order 0 -> N-1.
        """
        c_aq, c_org, h, z = state
        aq_mol_in, aq_acid_in, org_mol_in = inflows
        Q_org = u.Q_org
        dc_aq, dc_org, dh, dz = [], [], [], []
        for j in range(self.N):
            phi_aq = Q_aq[j] / (Q_aq[j] + Q_org)                 # aqueous share of the mixer
            V_aq, V_org = self._holdups(phi_aq)
            r = self._transfer(u, j, (c_aq[j], c_org[j], h[j]), phi_aq)    # mol/h, aqueous -> organic
            dc_aq.append((aq_mol_in[j] - Q_aq[j] * c_aq[j] - r) / V_aq)
            dc_org.append((org_mol_in[j] - Q_org * c_org[j] + r) / V_org)
            # acid: 3 H+ released per metal ion extracted, minus the base the organic spends here
            dh.append((aq_acid_in[j] - Q_aq[j] * h[j] + self.p.ligands_per_metal * ca.sum1(r) - doses[j])
                      / V_aq)
            dz.append(self._interface_rate(u, j, c_aq[j], c_org[j], z[j]))
        return dc_aq, dc_org, dh, dz

    def _holdups(self, phi_aq):
        """Hold-up of each phase in one stage, L: (aqueous, organic).

        The mixer is split between the phases by their share of the flow through it (phi_aq is the
        aqueous share). The settler is split by the interface: the aqueous sits below it, the
        organic above. The interface is taken at its setpoint here, so the hold-ups don't move
        with z (fixed hold-ups, decision 11).

        Judgement call: the stage is the lab unit scaled 250 times, for 600 L/h in all. At the
        ndpr_plant flows the extraction settler takes about 2.6 m3/h on 0.24 m2, near 10 m3/h per
        m2 of settler. Settlers are usually designed for a few m3/h per m2 (copper SX practice
        is often quoted at 3-5; textbook, not checked against a project source). A settler sized
        for these flows would be two to three times bigger. The steady state doesn't depend on
        the settler volume, but every time constant does: a real plant of this throughput would
        be slower than the model. Sizing the stage from the design flows would settle it.
        """
        geo = self.p.raw["stage_geometry"]
        V_mix, V_set, H = geo["mixer_volume_L"], geo["settler_volume_L"], geo["settler_height_m"]
        z_set = geo["interface_setpoint_fraction"] * H
        V_aq = V_mix * phi_aq + V_set * z_set / H            # fixed hold-ups (v1)
        V_org = V_mix * (1 - phi_aq) + V_set * (1 - z_set / H)
        return V_aq, V_org

    # ---- the pieces every stage model shares ----------------------------------
    def _inflows(self, u: Inputs, outlets: _StageOutlets):
        """mol/h of metal (aqueous, organic) and acid into each stage, from the map.

        Each edge that ends at a stage brings flow times concentration: either a source stream
        from outside, or the stream leaving a neighbouring stage (outlets: what each stage's
        outlet carries).

        Returns (aq_mol_in, aq_acid_in, org_mol_in): lists of N entries. aq_mol_in[j] and
        org_mol_in[j] are (n,) mol/h of metal; aq_acid_in[j] is mol/h of H+. Plain lists, so a
        plant subclass can add to them in _carry_over.
        """
        sources = self._source_streams(u, outlets)
        inflows = self._no_inflows()
        for e in self.cmap.edges:
            if not isinstance(e.dst, int):
                continue                              # a stream leaving the circuit
            _add_stream(e, self._edge_stream(e, sources, u, outlets), inflows)
        return inflows

    def _no_inflows(self) -> tuple:
        """(aq_mol_in, aq_acid_in, org_mol_in) with every stage at zero, ready to be added to."""
        N, n = self.N, self.n
        aq_mol_in = [ca.SX.zeros(n) for _ in range(N)]
        aq_acid_in = [ca.SX(0) for _ in range(N)]
        org_mol_in = [ca.SX.zeros(n) for _ in range(N)]
        return aq_mol_in, aq_acid_in, org_mol_in

    def _source_streams(self, u: Inputs, outlets: _StageOutlets) -> dict:
        """The streams that enter from outside, by source name: (flow L/h, metal mol/L, acid mol/L).

        Two of them read the circuit's own state:
          - the recycled organic comes back into stage 0 with what left stage N-1 (it isn't
            stripped further in the model); with no recycle it's fresh, metal-free organic.
          - with a reflux scrub, the scrub liquor is strip product sent back: its metal is the
            product stage's aqueous, at the acidity the operator sets (decision 14).
        The organic carries no acid, so its acid entry is None.

        Judgement call on the reflux: the scrub takes the product's metal but the operator's
        acid. In a plant that is product plus added acid. At the ndpr_plant point the product
        leaves the strip at about 1.5 M acid and the scrub is set to 2.0 M, so only about 0.5 M
        is new acid. The reagent count (metrics.reagent_rate, design.py) charges the full 2.0 M,
        and the product's own acid in the reflux drops out of the books. The balance here stays
        closed; the reagent number is high by about Q_scrub * 1.5 M. A scrub acid set below the
        product's would need neutralizing, which nothing charges. Counting only the added acid
        would settle it.
        Judgement call on the strip: 2 M acid leaves the strip at about 1.5 M at that point. A
        plant would usually strip with less excess, for a stronger product and less acid to
        neutralize downstream. It shows in the reagent number, not in the separation.
        """
        n = self.n
        c_org_in0 = outlets.c_org[self.N - 1] if self.cmap.organic_recycle else ca.SX.zeros(n)
        ps = self.cmap.product_stage
        # reflux: the scrub liquor is strip product (its metal, at the acidity the operator sets)
        c_scrub = u.c_scrub + (outlets.c_aq[ps] if self.scrub_reflux else 0)
        return {
            "feed": (u.Q_feed, u.c_feed, u.h_feed),
            "scrub": (u.Q_scrub, c_scrub, u.h_scrub),
            "strip": (u.Q_strip, ca.SX.zeros(n), u.h_strip),
            "organic": (u.Q_org, c_org_in0, None),
        }

    @staticmethod
    def _edge_stream(e: Edge, sources: dict, u: Inputs, outlets: _StageOutlets):
        """(flow L/h, metal mol/L, acid mol/L) carried along edge e, which ends at a stage.

        A named source gives its own stream. An aqueous edge from stage k carries stage k's
        aqueous outflow; an organic edge from stage k carries the organic flow at stage k's
        organic concentrations (no acid in the organic, so None).
        """
        if isinstance(e.src, str):
            return sources[e.src]
        if e.phase == "aq":
            return outlets.Q_aq[e.src], outlets.c_aq[e.src], outlets.h[e.src]
        return u.Q_org, outlets.c_org[e.src], None

    def _carry_over(self, u: Inputs, Q_aq, state, inflows):
        """Extra transport between stages, added to the inflows in place. None in this model; a
        plant with entrainment (plant_models.py) puts it here.

        u: the inputs. Q_aq: aqueous flow out of each stage, L/h.
        state: (c_aq, c_org, h, z) per stage. inflows: (aq_mol_in, aq_acid_in, org_mol_in), mol/h.
        """

    def _saponification_doses(self, u: Inputs, h) -> list:
        """Saponification = base carried by the organic; it neutralizes acid where it finds it, and
        whatever finds no acid in a stage moves on with the organic to the next. mol OH-/h per stage.

        The base entering stage 0 with the organic is (saponified fraction) * (extractant monomer,
        mol/L) * Q_org. Stage j spends the share h_j / (h_j + acid_floor) of what's left: nearly
        all of it where there's acid, little where the aqueous is already near neutral. The rest
        goes on to stage j+1 (APPENDIX_A_model.md, A.7; decision 12).

        u: the inputs. h: aqueous acid per stage, mol/L.
        Returns a list of N doses, mol/h.

        Why: a sodium-form extractant trading Na+ for RE3+ has the same net proton balance as the
        acid form plus a base dose, so the metal and acid balances come out the same. Check: at
        the ndpr_plant point the saponified fraction (0.4285 of 1 M) is about what an organic
        loaded to the 85 % cap gives off (0.85 x 0.5 M of sites = 0.425 M of H+), which is how a
        plant sets saponification too.
        Judgement call: the base is spent where there's acid, starting at stage 0, so at that
        point 99 % of it goes in stage 0. In a plant the sodium soap does meet the aqueous first
        there, but the local pH in that mixer can run high enough for emulsions and precipitates.
        And 43 % saponified is on the high side of what plants commonly run (practice, not
        checked against a project source), where viscosity and gelling start to worry. The
        sodium itself, about 0.9 M in the raffinate at that point, isn't tracked: it reaches
        neither the densities nor the ionic strength. The "full" mode (sodium form tracked) and a
        lab check of the organic at 43 % would settle it. The acid_floor in the fraction below
        only makes the dose smooth (chosen).
        """
        base_left = u.saponified_fraction * self.p.raw["extractant"]["concentration_monomer"] * u.Q_org
        doses = []
        for j in range(self.N):
            dose = base_left * h[j] / (h[j] + self.acid_floor)
            base_left = base_left - dose
            doses.append(dose)
        return doses

    def _transfer(self, u: Inputs, j: int, mixer_j, phi_aq):
        """mol/h of each metal from aqueous to organic in stage j's mixer (rate model).

        r = k * kappa_j * V_org,mixer * (D * c_aq - c_org). The bracket is how far the organic is
        from equilibrium with the aqueous (zero at equilibrium, since D = c_org / c_aq there).
        V_mix * (1 - phi_aq) is the organic in the mixer, L. k is p.k_transfer (1/h) and kappa_j
        is u.k_factor[j] (1 = nominal stage efficiency).

        mixer_j: (c_aq (n,), c_org (n,), h) in stage j's mixer, mol/L.
        phi_aq: aqueous share of the mixer. Returns (n,) mol/h; negative means organic to aqueous
        (as in the strip).

        Why: the organic stays in the mixer V_mix * (1 - phi_aq) / Q_org = V_mix / (Q_aq + Q_org)
        = tau, so with the aqueous held fixed this gives the stirred-tank stage efficiency
        k tau / (1 + k tau) of A.5.
        Judgement call: the stage's whole hold-up is taken as the mixer's contents (decision 11),
        and one k serves every element in extraction, scrub and strip. Stripping with acid can be
        slower than loading for some systems; for Pr and Nd in PC88A that wasn't checked. If the
        strip is slower, more strip stages are needed. Strip kinetics on the real organic would
        settle it.
        """
        c_aq_j, c_org_j, h_j = mixer_j
        p = self.p
        V_mix = p.raw["stage_geometry"]["mixer_volume_L"]
        D = chemistry.distribution(c_org_j, h_j, p) * self._distribution_multiplier(j, mixer_j)
        rate = p.k_transfer * u.k_factor[j] * self._rate_multiplier(j, mixer_j)
        return rate * V_mix * (1 - phi_aq) * (D * c_aq_j - c_org_j)

    def _interface_rate(self, u: Inputs, j: int, c_aq_j, c_org_j, z_j):
        """dz/dt: the interface relaxes to where the jackleg's hydrostatic balance puts it.

        At rest the interface sits where the liquid columns balance. On the jackleg side there's
        aqueous up to the jackleg height; in the settler there's aqueous up to the interface and
        organic from there up to the weir. Equal pressure at the bottom gives

            z_rest = (rho_aq * H_jackleg - rho_org * H_weir) / (rho_aq - rho_org), m.

        The interface moves toward z_rest with the jackleg time constant (hours). The densities
        follow the loading, so a change in loading moves the interface.

        Why: it's the hydrostatic balance of a continuous gravity decanter (Seader, Henley and
        Roper). It gives the gain a plant sees: dz/dH_jackleg = rho_aq / (rho_aq - rho_org), about
        4 to 7 at the ndpr_plant point, so a 1 cm jackleg move puts the interface 4-7 cm.
        Judgement call: the balance is taken with no flow. In a plant the organic stands higher
        over its weir as the organic flow goes up, so a flow change moves the interface on its
        own; here only the densities move it. The time constant (6 min) is chosen; a small
        settler with high flow would answer faster. A step test on a jackleg would settle both.

        Returns dz/dt for stage j, m/h.
        """
        geo = self.p.raw["stage_geometry"]
        rho_a = chemistry.rho_aqueous(c_aq_j, self.p)
        rho_o = chemistry.rho_organic(c_org_j, self.p)
        z_rest = (rho_a * u.H_jackleg[j] - rho_o * geo["organic_weir_height_m"]) / (rho_a - rho_o)
        return (z_rest - z_j) / geo["jackleg_time_constant_h"]

    def _make_rhs_functions(self, x, uv, xdot):
        """The right-hand side xdot = f(x, u) as a CasADi function, and its Jacobians with respect
        to the state (for the integrator and the linearization) and to the inputs (for the MPC)."""
        self._f = ca.Function("rhs", [x, uv], [xdot], ["x", "u"], ["xdot"])
        self._jac = ca.Function("rhs_jac", [x, uv], [ca.jacobian(xdot, x)])
        self._jac_u = ca.Function("rhs_jac_u", [x, uv], [ca.jacobian(xdot, uv)])

    def _make_outlet_function(self, x, uv, u: Inputs, outlets: _StageOutlets):
        """The streams leaving the circuit, as a CasADi function of (x, u), for the metrics.

        Raffinate: the aqueous out of stage 0. Product: the aqueous out of the product stage, net
        of what goes back as reflux. Spent organic: the organic out of stage N-1. Each as
        (flow L/h, concentrations mol/L, and acid mol/L for the aqueous ones).
        """
        raff = (outlets.Q_aq[0], outlets.c_aq[0], outlets.h[0])
        prod = self._product_stream(u, outlets)
        self._out = ca.Function("out", [x, uv],
                                [raff[0], raff[1], raff[2], prod[0], prod[1], prod[2],
                                 u.Q_org, outlets.c_org[self.N - 1]],
                                ["x", "u"], ["Q_raff", "c_raff", "h_raff", "Q_prod", "c_prod", "h_prod",
                                             "Q_spent", "c_spent"])

    def _product_stream(self, u: Inputs, outlets: _StageOutlets) -> tuple:
        """(flow L/h, metal mol/L, acid mol/L) of the net product.

        With a reflux scrub the scrub liquor is taken from the product, so the net product flow is
        the product stage's outflow minus the scrub flow. With no strip section, all zeros.
        """
        ps = self.cmap.product_stage
        if ps is None:
            return ca.SX(0), ca.SX.zeros(self.n), ca.SX(0)
        reflux = u.Q_scrub if self.scrub_reflux else 0
        return outlets.Q_aq[ps] - reflux, outlets.c_aq[ps], outlets.h[ps]

    def observable(self, x) -> np.ndarray:
        """The state in the controller model's layout, as instruments would see it. Here the
        state itself; a richer plant (plant_models.py) maps its own state onto this layout."""
        return np.asarray(x, float)

    # ---- hooks for a plant with hidden states (plant_models.DriftingCircuit); no-ops here -----
    def _read_hidden(self, x):
        """Called once at the start of _build with the state symbol. Nothing to read here."""

    def _hidden_rates(self) -> list:
        """Derivatives of any hidden states, appended after the model's. None here."""
        return []

    def _distribution_multiplier(self, j=None, mixer_j=None):
        """A factor on stage j's distribution coefficients (1 here; plant_models moves it).
        mixer_j: (c_aq, c_org, h) in the stage's mixer."""
        return 1

    def _rate_multiplier(self, j=None, mixer_j=None):
        """A factor on stage j's transfer rate (1 here; plant_models.BookCircuit slows it with acid)."""
        return 1

    def hidden_at(self, x: np.ndarray, t_h: float) -> np.ndarray:
        """The state with any hidden states set for plant time t_h (h). The closed loop calls this
        once per control step on every plant. A plain circuit has no hidden states, so x comes back
        as it is; plant_models.DriftingCircuit (and BookCircuit, built on it) set their slow drift."""
        return x

    # ---- numeric interface -------------------------------------------------
    def rhs(self, x, u) -> np.ndarray:
        """xdot = f(x, u) at numbers: state x and input vector u (Inputs.numpy()). Flat array."""
        return np.array(self._f(x, u)).ravel()

    def jac(self, x, u) -> np.ndarray:
        """d(xdot)/dx, (nx, nx)."""
        return np.array(self._jac(x, u))

    def jac_u(self, x, u) -> np.ndarray:
        """d(xdot)/d(inputs), (nx, Inputs.size)."""
        return np.array(self._jac_u(x, u))

    def outlets(self, x, u) -> dict:
        """The streams leaving the circuit at (x, u): Q_raff, c_raff, h_raff, Q_prod, c_prod,
        h_prod, Q_spent, c_spent. Scalars as floats, concentrations as flat arrays."""
        r = self._out(x=x, u=u)
        return {k: np.array(v).ravel() if np.array(v).size > 1 else float(v) for k, v in r.items()}

    def casadi_rhs(self) -> ca.Function:
        """The symbolic right-hand side, for design.py, mpc.py and anyone else who wants it."""
        return self._f

    # ---- helpers -----------------------------------------------------------
    def jackleg_for_interface(self, c_aq, c_org, z_target: float) -> float:
        """Jackleg height that puts the interface at z_target for these phase
        compositions: how an operator sets it at commissioning.

        The hydrostatic balance in _interface_rate, solved for the jackleg height.
        c_aq, c_org: one stage's concentrations, mol/L. z_target: interface height, m.
        Returns the jackleg height, m.
        """
        geo = self.p.raw["stage_geometry"]
        rho_a = float(chemistry.rho_aqueous(np.asarray(c_aq), self.p))
        rho_o = float(chemistry.rho_organic(np.asarray(c_org), self.p))
        return (z_target * (rho_a - rho_o) + rho_o * geo["organic_weir_height_m"]) / rho_a

    def initial_state(self, u: Inputs) -> np.ndarray:
        """Circuit full of metal-free liquids at the feed acidity, interfaces at setpoint.

        The acid starts at the feed acidity but no lower than 0.01 mol/L.
        """
        geo = self.p.raw["stage_geometry"]
        z_set = geo["interface_setpoint_fraction"] * geo["settler_height_m"]
        N, n = self.N, self.n
        h0 = np.full(N, max(float(u.h_feed), 1e-2))
        return self.pack(np.zeros((N, n)), np.zeros((N, n)), h0, np.full(N, z_set))


def _add_stream(e: Edge, stream: tuple, inflows: tuple) -> None:
    """Add what edge e brings into its stage to the inflows, in place.

    stream: (flow L/h, metal mol/L, acid mol/L) along e. An aqueous edge brings metal and acid;
    an organic edge brings metal only. inflows: (aq_mol_in, aq_acid_in, org_mol_in) per stage.
    """
    Q, c, hh = stream
    aq_mol_in, aq_acid_in, org_mol_in = inflows
    if e.phase == "aq":
        aq_mol_in[e.dst] += Q * c
        aq_acid_in[e.dst] += Q * hh
    else:
        org_mol_in[e.dst] += Q * c


def _stages_reached(source: str, aq_edges: list[Edge]) -> set[int]:
    """Every stage the aqueous from `source` passes through, following the edges downstream.

    source: a source name ("feed", "scrub", "strip"). aq_edges: the map's aqueous edges.
    Returns the set of stage indices; sinks such as "raffinate" are left out.
    """
    frontier = _next_places(source, aq_edges)
    seen: set[int] = set()
    while frontier:
        j = frontier.pop()
        if isinstance(j, int) and j not in seen:       # a stage not counted yet (not a sink)
            seen.add(j)
            frontier += _next_places(j, aq_edges)
    return seen


def _next_places(place: int | str, edges: list[Edge]) -> list:
    """Where the streams leaving `place` go: a list of stage indices and sink names."""
    return [e.dst for e in edges if e.src == place]
