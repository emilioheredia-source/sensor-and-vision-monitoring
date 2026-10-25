"""Find the operating point of a circuit by optimization, on the same model.

What this file does
-------------------
A solvent-extraction circuit has a handful of inputs an operator can set: the
organic flow, the scrub flow, the scrub acidity and how much of the extractant
is saponified, among others. Before any controller runs, someone has to pick
values for them. This file picks them with a nonlinear optimizer that works
on the same model equations the simulator uses (ARCHITECTURE.md, decision 15;
APPENDIX_B_control.md, B.1).

The problem handed to the optimizer is

    maximize   recovery                                   (objective="recovery")
      or minimize reagent with recovery >= a floor        (objective="reagent")
    over       the free inputs (flows, scrub acidity, saponification)
               and the full state x of the circuit
    subject to steady state           rhs(x, u) = 0
               product purity         >= spec
               organic loading        <= max_loading of capacity, in every stage
               concentrations >= 0, acid >= floor, inputs inside their boxes

The state x is a decision variable too. That's how a steady state is found
and optimized in one go: the optimizer moves the inputs and the state
together and the constraint rhs(x, u) = 0 makes sure the state it ends on is
a steady state of the model under the inputs it ends on.

The two objectives are used one after the other (notebooks/02_design_baselines.py):
first the highest recovery possible at the design purity, then the least
reagent that still keeps recovery above a floor. Three more rules come in
from the caller or the parameter file:

  pump headroom   the caller caps the organic flow's upper bound at 85 % of
                  the pump maximum, so a controller can add organic later
                  (decision 19). It arrives here as the upper bound in `free`.
  reflux link     with a reflux scrub, the strip flow is the net product flow
                  plus the scrub flow. `reflux_link` below builds that tie.
  loading cap     no stage's organic may hold more than 85 % of the metal it
                  could hold in theory (decision 22, parameters.toml
                  [extractant] max_loading_fraction).

It's solved with IPOPT through CasADi, with exact derivatives from the model.
The stage-count study uses it for every circuit size. The plant's own
operating point is then found by commissioning and a trim under control
(operating_point.py). The ECJTU patent does the same step and calls it
"steady-state optimization".

A note on order: IPOPT's path depends on the order of the variables and of
the constraints. The helpers below build them in a fixed order (state, then
handles; steady state, purity, recovery floor, loading cap). Don't reorder
them; the reference runs in tests/baseline depend on it.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import casadi as ca
import numpy as np

from .circuit import Circuit, Inputs
from .metrics import heavy_mask
from .simulate import steady_state

# The names of every field of Inputs, in the order the dataclass declares them.
INPUT_FIELDS = [f.name for f in fields(Inputs)]


# ---------------------------------------------------------------------------
# What the optimizer hands back
# ---------------------------------------------------------------------------

@dataclass
class DesignResult:
    """The optimizer's answer and how it got there.

    u                  the full set of inputs at the optimum (free handles moved, the rest as given)
    x                  the steady state of the model under u (the circuit's state vector)
    recovery           fraction of the product-side metal fed that leaves in the product, 0..1
    purity             product purity on a mole-of-metal basis, 0..1
    reagent_mol_per_h  base for saponification plus strip and scrub acid, mol/h
    success            True when IPOPT says it converged
    status             IPOPT's return status, e.g. "Solve_Succeeded"
    iterations         how many IPOPT iterations it took
    """
    u: Inputs
    x: np.ndarray
    recovery: float
    purity: float
    reagent_mol_per_h: float
    success: bool
    status: str
    iterations: int = 0


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------

def optimize_operating_point(circuit: Circuit, u0: Inputs, cut: str, spec: float,
                             free: dict[str, tuple[float, float]], link=None,
                             x_guess: np.ndarray | None = None, reagent_weight: float = 0.0,
                             acid_floor: float = 1e-3, max_iter: int = 3000,
                             ipopt_opts: dict | None = None, objective: str = "recovery",
                             min_recovery: float | None = None,
                             max_loading: float | None = None) -> DesignResult:
    """Find the steady operating point that is best by `objective`, under purity and loading limits.

    Parameters
    ----------
    circuit        the circuit model (stages, chemistry, geometry)
    u0             the starting inputs. Every input not named in `free` stays at its value here.
    cut            the last component meant for the raffinate ("LaCe" for the NdPr product,
                   "Pr" for the Pr | Nd validation case)
    spec           the purity the product must reach, 0..1 (e.g. the design purity, 0.995)
    free           {input name: (low, high)}: the handles the optimizer may move, with their
                   bounds in the input's own units (L/h for flows, mol/L for acid, a fraction for
                   saponification). The organic flow's high bound is where the pump headroom
                   goes (decision 19).
    link           optional f(dict of input fields) -> dict, for tied inputs
                   (e.g. strip flow = net product flow + reflux flow; see reflux_link)
    x_guess        a steady state near the answer; computed at u0 if not given
    reagent_weight mol/h of acid + base per unit of recovery you would trade;
                   0 = recovery only. Used only with objective "recovery".
    acid_floor     the lowest aqueous acid allowed in any stage, mol/L
    max_iter       IPOPT's iteration limit
    ipopt_opts     extra CasADi/IPOPT options, written over the defaults below
    objective      "recovery" (maximize it) or "reagent" (minimize acid + base, mol/h)
    min_recovery   with objective "reagent", the recovery that must still be met, 0..1
    max_loading    the organic may hold at most this fraction of its theoretical metal
                   capacity in any stage; None = parameters.toml [extractant] max_loading_fraction
                   (a margin against saturation, where D goes as the cube of what little
                   free extractant is left and the model turns very steep)

    Returns
    -------
    DesignResult with the optimal inputs, the steady state and what they give.
    Check `success`: IPOPT can stop without converging, and the result is then
    just where it stopped.

    Raises
    ------
    ValueError     if `objective` is neither "recovery" nor "reagent"

    Example
    -------
        free = {"Q_org": (600.0, 0.85 * 4700.0), "Q_scrub": (50.0, 2000.0),
                "h_scrub": (0.05, 2.0), "saponified_fraction": (0.0, 0.6)}
        r1 = optimize_operating_point(circ, u0, "LaCe", 0.995, free, link=reflux_link(300.0))
        r2 = optimize_operating_point(circ, r1.u, "LaCe", 0.995, free, link=reflux_link(300.0),
                                      x_guess=r1.x, objective="reagent", min_recovery=0.95)

    Why fourteen parameters: the first five say what to design; the rest are keyword options with defaults (limits, weights, solver settings) that most callers leave alone. Grouping them would change every call in the notebooks and tests for no gain in clarity.
    """
    # The decision variables. `handles` are the free inputs, in the order of `free`.
    # `x` is the circuit's full state: aqueous metal, organic metal, aqueous acid
    # and interface height in every stage.
    handle_names = list(free)
    handles = ca.SX.sym("v", len(handle_names))
    x = ca.SX.sym("x", circuit.nx)

    # The inputs as symbols: the fixed ones are numbers from u0, the free ones are
    # the symbols above, and the link (if any) fills in the tied ones.
    symbolic_handles = {name: handles[k] for k, name in enumerate(handle_names)}
    u_sym = _inputs_with_handles(u0, symbolic_handles, link, circuit.N)

    # The model's equations and what we score, all as CasADi expressions in (x, handles).
    xdot, recovery, purity, reagent = _model_expressions(circuit, x, u_sym, cut)

    # The objective, then the constraints in their fixed order.
    f = _objective(objective, recovery, reagent, reagent_weight)
    g, lbg, ubg = _constraints(circuit, x, (xdot, purity, recovery), (spec, min_recovery, max_loading))

    # The NLP. The variable vector is the state first, then the handles.
    nlp = {"x": ca.vertcat(x, handles), "f": f, "g": g}
    solver = _make_solver(nlp, max_iter, ipopt_opts)

    # Bounds on the variables, the starting point, and the solve.
    lbx, ubx = _variable_bounds(circuit, free, handle_names, acid_floor)
    z0 = _starting_point(circuit, u0, handle_names, x_guess)
    sol = solver(x0=z0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    stats = solver.stats()

    # Split the answer back into state and handles and report what they give.
    z = np.array(sol["x"]).ravel()
    xs, vs = z[: circuit.nx], z[circuit.nx:]
    numeric_handles = {name: float(vs[k]) for k, name in enumerate(handle_names)}
    u_opt = _inputs_with_handles(u0, numeric_handles, link, circuit.N)
    report = ca.Function("report", [x, handles], [recovery, purity, reagent])
    rec, pur, rea = (float(a) for a in report(xs, vs))
    return DesignResult(u_opt, xs, rec, pur, rea, bool(stats["success"]),
                        str(stats["return_status"]), int(stats["iter_count"]))


# ---------------------------------------------------------------------------
# Building the inputs from the handles
# ---------------------------------------------------------------------------

def _inputs_with_handles(u0: Inputs, handle_values: dict, link, n_stages: int) -> Inputs:
    """The inputs u0 with the free handles written in and the tied inputs filled by `link`.

    The same function serves twice: once with CasADi symbols as the handle values
    (to build the problem) and once with plain floats (to report the answer).

    Parameters
    ----------
    u0             the starting inputs; every field not in `handle_values` keeps its value
    handle_values  {input name: value} for each free handle (a symbol or a float)
    link           optional f(dict of input fields) -> dict for tied inputs, or None
    n_stages       number of stages N, for the default transfer-rate multipliers

    Returns
    -------
    Inputs with the handles in place.
    """
    values = {k: getattr(u0, k) for k in INPUT_FIELDS}
    # k_factor is the per-stage multiplier on the transfer rate. None means
    # "all stages nominal", which is a row of ones.
    if values["k_factor"] is None:
        values["k_factor"] = np.ones(n_stages)
    values.update(handle_values)
    if link:
        values = link(values)
    return Inputs(**values)


# ---------------------------------------------------------------------------
# The model's equations, and what is scored
# ---------------------------------------------------------------------------

def _model_expressions(circuit: Circuit, x: ca.SX, u_sym: Inputs, cut: str):
    """The steady-state residual, recovery, purity and reagent rate, as CasADi expressions.

    Parameters
    ----------
    circuit   the circuit model
    x         the state as a symbol (length circuit.nx)
    u_sym     the inputs, with the free handles as symbols
    cut       the last component meant for the raffinate

    Returns
    -------
    (xdot, recovery, purity, reagent):
        xdot      d(state)/dt from the model; zero at steady state
        recovery  product-side metal in the product / product-side metal in the feed
        purity    product-side metal in the product / all metal in the product
        reagent   base + acid fed, mol/h
    """
    p = circuit.p
    uv = u_sym.to_vector()

    # The model's right-hand side and its outlet streams.
    xdot = circuit.casadi_rhs()(x, uv)
    out = circuit._out(x=x, u=uv)

    # The components heavier than the cut are the product side (for cut "LaCe": Pr and Nd,
    # and Sm+ if it wasn't removed upstream).
    heavy = [i for i, m in enumerate(heavy_mask(p, cut)) if m]
    c_prod, c_feed = out["c_prod"], u_sym.c_feed

    # recovery = Q_prod * sum(c_prod[heavy]) / (Q_feed * sum(c_feed[heavy]))     (both mol/h)
    prod_heavy = out["Q_prod"] * ca.sum1(c_prod[heavy])
    fed_heavy = u_sym.Q_feed * ca.sum1(c_feed[heavy])
    recovery = prod_heavy / fed_heavy

    # purity = sum(c_prod[heavy]) / sum(c_prod)                                  (mol/L over mol/L)
    purity = ca.sum1(c_prod[heavy]) / ca.sum1(c_prod)

    reagent = _reagent_expression(u_sym, p)
    return xdot, recovery, purity, reagent


def _reagent_expression(u_sym: Inputs, p) -> ca.SX:
    """Base and acid fed per hour, mol/h. The same formula as metrics.reagent_rate.

    reagent = s * L_mono * Q_org        base for saponification
            + Q_strip * h_strip         strip acid
            + Q_scrub * h_scrub         scrub acid

    s is the saponified fraction, L_mono the extractant concentration as monomer (mol/L),
    flows in L/h and acidities in mol/L.
    """
    return (u_sym.saponified_fraction * p.raw["extractant"]["concentration_monomer"] * u_sym.Q_org
            + u_sym.Q_strip * u_sym.h_strip + u_sym.Q_scrub * u_sym.h_scrub)


def _objective(objective: str, recovery, reagent, reagent_weight: float):
    """The quantity IPOPT minimizes.

    "recovery":  f = -recovery + reagent_weight * reagent      (maximize recovery)
    "reagent":   f = reagent / 1000                            (minimize reagent; the 1000 only
                                                                rescales mol/h to a size IPOPT likes)

    Raises ValueError for any other name.
    """
    if objective == "recovery":
        return -recovery + reagent_weight * reagent
    if objective == "reagent":
        return reagent / 1000.0
    raise ValueError(f"objective {objective!r}: use 'recovery' or 'reagent'")


# ---------------------------------------------------------------------------
# Constraints. Each block is (expression, lower bounds, upper bounds).
# ---------------------------------------------------------------------------

def _constraints(circuit: Circuit, x: ca.SX, exprs: tuple, limits: tuple):
    """All the constraints, stacked in their fixed order: steady state, purity floor, recovery
    floor (if asked), loading cap (if any).

    exprs: (xdot, purity, recovery) as CasADi expressions.
    limits: (spec, min_recovery, max_loading), as passed to optimize_operating_point.
    Returns (g, lbg, ubg) for the NLP.
    """
    xdot, purity, recovery = exprs
    spec, min_recovery, max_loading = limits
    blocks = [_steady_state_constraint(xdot, circuit.nx), _purity_constraint(purity, spec)]
    if min_recovery is not None:
        blocks.append(_recovery_floor_constraint(recovery, min_recovery))
    loading_cap = _loading_cap_fraction(circuit, max_loading)
    if loading_cap is not None:
        blocks.append(_loading_cap_constraint(circuit, x, loading_cap))
    return _stack_constraints(blocks)


def _steady_state_constraint(xdot, n_states: int):
    """rhs(x, u) = 0 for every one of the n_states states: the circuit is at rest."""
    return xdot, np.zeros(n_states), np.zeros(n_states)


def _purity_constraint(purity, spec: float):
    """purity - spec >= 0: the product is on spec."""
    return purity - spec, [0.0], [np.inf]


def _recovery_floor_constraint(recovery, min_recovery: float):
    """recovery - min_recovery >= 0: used with the reagent objective (decision 15)."""
    return recovery - min_recovery, [0.0], [np.inf]


def _loading_cap_fraction(circuit: Circuit, max_loading: float | None) -> float | None:
    """The loading cap to use: the caller's, else parameters.toml [extractant] max_loading_fraction.

    Returns None when neither is set, and then no loading constraint is added.
    """
    if max_loading is None:
        return circuit.p.raw["extractant"].get("max_loading_fraction")
    return max_loading


def _loading_cap_constraint(circuit: Circuit, x: ca.SX, max_loading: float):
    """The metal held in the organic of every stage stays under the cap (decision 22).

    For stage j:   sum_i c_org[i, j]  <=  max_loading * L_tot / 3          (mol/L)

    L_tot is the extractant as dimer (p.dimer_total, mol/L) and 3 is the number of
    dimers one metal ion takes (p.ligands_per_metal), so L_tot / 3 is the most metal
    the organic could hold. The sum is linear in the organic states, which sit in x
    after the N * n aqueous ones.
    """
    p, n, N = circuit.p, circuit.n, circuit.N
    c_org_sum = ca.vertcat(*[ca.sum1(x[N * n + j * n: N * n + (j + 1) * n]) for j in range(N)])
    upper = max_loading * p.dimer_total / p.ligands_per_metal
    return c_org_sum, [-np.inf] * N, [upper] * N


def _stack_constraints(blocks):
    """Stack the constraint blocks, in the order given, into g and its bounds lbg <= g <= ubg."""
    g = ca.vertcat(*[expr for expr, _, _ in blocks])
    lbg = np.concatenate([lower for _, lower, _ in blocks])
    ubg = np.concatenate([upper for _, _, upper in blocks])
    return g, lbg, ubg


# ---------------------------------------------------------------------------
# Bounds, starting point, solver
# ---------------------------------------------------------------------------

def _variable_bounds(circuit: Circuit, free: dict, handle_names: list[str], acid_floor: float):
    """Lower and upper bounds on [x, handles], in that order.

    The state is laid out as aqueous metal (N * n), organic metal (N * n),
    aqueous acid (N), interface height (N). So:

        concentrations   >= 0                 (mol/L)
        acid             >= acid_floor        (mol/L)
        interfaces       unbounded            (m)
        handles          inside their boxes   (from `free`)
        hidden states    pinned at 0, for a plant that carries some after the model's layout
                         (plant_models.DriftingCircuit: no drift in a steady-state design)
    """
    N, n = circuit.N, circuit.n
    n_conc = 2 * N * n
    hidden = np.zeros(circuit.nx - circuit.nx_model)
    lbx = np.concatenate([np.zeros(n_conc), np.full(N, acid_floor), np.full(N, -np.inf), hidden,
                          [free[k][0] for k in handle_names]])
    ubx = np.concatenate([np.full(n_conc + N, np.inf), np.full(N, np.inf), hidden,
                          [free[k][1] for k in handle_names]])
    return lbx, ubx


def _starting_point(circuit: Circuit, u0: Inputs, handle_names: list[str], x_guess):
    """Where IPOPT starts: a steady state (x_guess, or one computed at u0), then the handles at u0."""
    x0 = steady_state(circuit, u0) if x_guess is None else np.asarray(x_guess, float)
    v0 = np.array([float(getattr(u0, k)) for k in handle_names])
    return np.concatenate([x0, v0])


def _make_solver(nlp: dict, max_iter: int, ipopt_opts: dict | None):
    """An IPOPT solver for the NLP. Quiet, tight tolerance (1e-8, acceptable 1e-6),
    adaptive barrier update; `ipopt_opts` is written over these."""
    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": max_iter,
            "ipopt.tol": 1e-8, "ipopt.acceptable_tol": 1e-6, "ipopt.mu_strategy": "adaptive"}
    opts.update(ipopt_opts or {})
    return ca.nlpsol("design", "ipopt", nlp, opts)


# ---------------------------------------------------------------------------
# Tied inputs
# ---------------------------------------------------------------------------

def reflux_link(net_product_flow: float):
    """Tie the strip flow to the reflux: strip = net product + what goes back as scrub.

    With a reflux scrub, part of the strip product is sent back as scrub liquor.
    So the strip flow (L/h) must be the product that leaves the circuit plus the
    scrub flow. Pass the result as `link` to optimize_operating_point.

    Parameters
    ----------
    net_product_flow   strip product leaving the circuit, L/h (parameters.toml [design], 300)

    Returns
    -------
    A function that takes a dict of input fields and returns a copy with Q_strip set.

    Example
    -------
        link = reflux_link(300.0)
        link({"Q_scrub": 250.0, "Q_strip": 0.0})["Q_strip"]     # 550.0
    """
    def link(fields):
        """A copy of the input fields with Q_strip = net product + Q_scrub."""
        fields = dict(fields)
        fields["Q_strip"] = net_product_flow + fields["Q_scrub"]
        return fields
    return link
