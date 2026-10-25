"""Run the circuit in time, and find its steady state.

The circuit is a set of ordinary differential equations, dx/dt = rhs(x, u),
with x the state and u the input vector (flows, feed, acids, jacklegs). The
equations are stiff: some things settle within minutes (the interface,
mass transfer in the mixer) and others over days (the metal held up in a
reflux circuit). An explicit integrator would need tiny steps to stay
stable, so the integrators here are ones built for stiff equations.

Three tools:

  simulate        the plant in time. scipy's BDF with the CasADi Jacobian,
                  adaptive step size, tight tolerances. If BDF stalls it falls
                  back to Radau, then LSODA, then the segment in ten pieces.
  implicit_euler  a model state inside the controller or an estimator. Fixed
                  cost every step (a few Newton solves), stable on the stiff
                  modes. Less accurate, but always back on time.
  steady_state    where the circuit settles under constant inputs. Integrate
                  until the state stops moving, then polish with a root solve.
                  Integrating first keeps the root solver out of the region
                  where acid goes negative and D blows up.

Time is in hours throughout. State layout (stages numbered from 0 at the
raffinate end, N stages, n components):

    x = [ c_aq (N*n) | c_org (N*n) | h (N) | z (N) ]
          mol/L        mol/L         mol/L   m
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from .circuit import Circuit, Inputs

# Why: measured, Oct 23 (notebooks 44, 45, 48). On a plant driven into the stiff corner of the
# chemistry, BDF and Radau gave up on a 0.025 h segment and LSODA, with no Jacobian and no step
# limit, then crawled for hours on it: two studies stalled overnight. A normal segment takes a few
# hundred evaluations; past this many the attempt counts as failed and the ladder moves on.
MAX_RHS_CALLS = 50_000                         # right-hand-side evaluations per integration attempt


class _OverBudget(Exception):
    """An integration attempt used up MAX_RHS_CALLS without finishing."""


@dataclass
class Trajectory:
    """A run in time: the state and the inputs at each output time."""
    t: np.ndarray          # hours
    x: np.ndarray          # (len(t), nx)
    u: np.ndarray          # (len(t), nu) input applied at each time


def _u_vec(u) -> np.ndarray:
    """The inputs as a plain float vector, whether given as Inputs or already as a vector."""
    return u.numpy() if isinstance(u, Inputs) else np.asarray(u, float)


# --------------------------------------------------------------------------
# The adaptive integrator, one segment of constant inputs at a time
# --------------------------------------------------------------------------
@dataclass
class _Segment:
    """One stretch of time with the inputs held constant: the problem handed to scipy.

    Fields:
        circuit: the circuit to run.
        u_seg: the input vector, constant over the segment.
        rtol, atol: relative and absolute tolerances for the integrator.
    """
    circuit: Circuit
    u_seg: np.ndarray
    rtol: float
    atol: float
    calls: int = 0                              # rhs evaluations in the current attempt

    def rhs(self, t, y):
        """dx/dt at state y (the inputs don't change inside a segment, so t isn't used).
        Raises _OverBudget past MAX_RHS_CALLS evaluations in one attempt."""
        self.calls += 1
        if self.calls > MAX_RHS_CALLS:
            raise _OverBudget
        return self.circuit.rhs(y, self.u_seg)

    def jac(self, t, y):
        """d(rhs)/dx at state y, from CasADi: the stiff solvers need it."""
        return self.circuit.jac(y, self.u_seg)


@dataclass
class _StitchedSolution:
    """The result of the ten-piece fallback, in the same shape scipy returns (t, y, success)."""
    t: np.ndarray
    y: np.ndarray
    success: bool = True


def _integrate(seg: _Segment, x: np.ndarray, t_span: tuple, t_eval: np.ndarray):
    """Integrate one segment from state x over t_span = (a, b) hours, output at the times t_eval.

    BDF first; if it stalls (a stage driven to the stiff corner of the chemistry), Radau, then
    LSODA, then the segment in ten pieces. One bad half hour must not kill a long study.

    Returns an object with .t (output times) and .y (states, one column per time).
    Raises RuntimeError if x isn't finite, or if even the ten-piece fallback fails.
    """
    if not np.all(np.isfinite(x)):
        raise RuntimeError("integration asked to start from a non-finite state")
    sol = _solve_with_fallbacks(seg, x, t_span, t_eval)
    if sol is not None:
        return sol
    return _solve_in_pieces(seg, x, t_span, t_eval)


def _solve_with_fallbacks(seg: _Segment, x: np.ndarray, t_span: tuple, t_eval: np.ndarray):
    """Try BDF, Radau and LSODA in turn on the whole segment. Returns the first solution that
    succeeds with finite values, or None if all three fail. (LSODA works out its own Jacobian.)"""
    for method, kw in (("BDF", dict(jac=seg.jac)), ("Radau", dict(jac=seg.jac)), ("LSODA", {})):
        seg.calls = 0
        try:
            sol = solve_ivp(seg.rhs, t_span, x, method=method, t_eval=t_eval, rtol=seg.rtol, atol=seg.atol, **kw)
        except _OverBudget:
            continue                                     # stalled: the next method, then the pieces
        if sol.success and np.all(np.isfinite(sol.y)):
            return sol
    return None


def _solve_in_pieces(seg: _Segment, x: np.ndarray, t_span: tuple, t_eval: np.ndarray) -> _StitchedSolution:
    """Last resort: the segment in ten equal pieces with Radau and a looser tolerance
    (rtol at least 1e-4, atol at least 1e-7), stitched back onto the requested output times.

    Each output time is produced by the first piece that contains it. Raises RuntimeError
    naming the piece if one fails.
    """
    a, b = t_span
    x_piece = np.asarray(x, float)                     # the state at the start of the current piece
    edges = np.linspace(a, b, 11)
    out_t, out_y = [], []
    for a2, b2 in zip(edges[:-1], edges[1:]):
        te = _times_in_piece(t_eval, a2, b2, out_t)
        sol = _solve_piece(seg, x_piece, (a2, b2), te)
        if te:
            out_t += list(sol.t)
            out_y += list(sol.y.T)
        x_piece = sol.y[:, -1]
    return _stitched(out_t, out_y, x_piece, b)


def _solve_piece(seg: _Segment, x_piece: np.ndarray, piece: tuple, te: list):
    """One piece of the ten-piece fallback: Radau from x_piece over piece = (a2, b2) hours, output
    at the times te (or wherever Radau stops, if te is empty), with the looser tolerances.

    Returns scipy's solution. Raises RuntimeError naming the piece if it fails or isn't finite.
    """
    seg.calls = 0
    try:
        sol = solve_ivp(seg.rhs, piece, x_piece, method="Radau", jac=seg.jac, t_eval=te or None,
                        rtol=max(seg.rtol, 1e-4), atol=max(seg.atol, 1e-7))
    except _OverBudget:
        raise RuntimeError(f"integration stalled on [{piece[0]}, {piece[1]}]: over {MAX_RHS_CALLS} evaluations")
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise RuntimeError(f"integration failed on [{piece[0]}, {piece[1]}]: {sol.message}")
    return sol


def _times_in_piece(t_eval: np.ndarray, a2: float, b2: float, out_t: list) -> list:
    """The output times inside the piece [a2, b2] that an earlier piece has not already produced
    (out_t is the list of times produced so far)."""
    return [t for t in t_eval if a2 <= t <= b2 and (not out_t or t > out_t[-1])]


def _stitched(out_t: list, out_y: list, x_end: np.ndarray, b: float) -> _StitchedSolution:
    """Put the pieces' outputs together, and make sure the end time b is the last output
    (with the final state x_end) even when no requested output time landed exactly on it."""
    s = _StitchedSolution(np.array(out_t), np.array(out_y).T)
    if len(out_t) == 0 or out_t[-1] < b - 1e-12:
        s.t, s.y = (np.append(s.t, b),
                    np.column_stack([s.y, x_end]) if len(out_t) else np.array([[v] for v in x_end]))
    return s


# --------------------------------------------------------------------------
# Implicit Euler: the fixed-cost integrator for model states
# --------------------------------------------------------------------------
# One implicit Euler step of size h solves for x_new in
#     x_new = x_old + h * rhs(x_new)
# Written as g(x_new) = x_new - x_old - h * rhs(x_new) = 0, Newton's method repeats
#     (I - h J) delta = -g,    x_new = x_new + delta
# with J the circuit's Jacobian at x_new, until delta is tiny or `iters` tries are used up.
# Because rhs is evaluated at the END of the step, the stiff fast modes are damped
# instead of blowing up, whatever h is.
def implicit_euler(circuit: Circuit, x: np.ndarray, u_vec: np.ndarray, dt: float,
                   n_sub: int = 6, iters: int = 8) -> np.ndarray:
    """The circuit over `dt` by implicit Euler in `n_sub` sub-steps, Newton on each with the
    circuit's own Jacobian. For a model state inside a controller or an estimator: a fixed cost
    every step, stable on the stiff modes, concentrations and acid kept non-negative. The adaptive
    integrator above is for the plant; on a model state fed by estimates it could crawl for
    minutes in its retry ladder, and a controller has to come back on time. (Decision 23.)

    Parameters:
        circuit: the circuit (usually the controller's model).
        x: the state at the start (state layout above).
        u_vec: the input vector, held constant over dt.
        dt: the time to cover, hours.
        n_sub: number of equal sub-steps.
        iters: most Newton iterations per sub-step.
    Returns:
        the state after dt. If a sub-step's Newton goes non-finite, the state at the start of that
        sub-step is returned instead.

    Usage:
        x_next = implicit_euler(model, x_now, u.numpy(), 0.5)

    Why six parameters: circuit, state, inputs and time step, plus two solver settings with defaults.
    """
    h = dt / n_sub                                     # sub-step size, hours
    x = np.asarray(x, float).copy()
    for _ in range(n_sub):
        x_next = _implicit_euler_substep(circuit, x, u_vec, h, iters)
        if x_next is None:
            return x
        x = x_next
    return x


def _implicit_euler_substep(circuit: Circuit, x_old: np.ndarray, u_vec: np.ndarray, h: float, iters: int):
    """One implicit Euler step of h hours from x_old, solved by Newton (see the comment above).

    After each Newton update, concentrations and acid are clipped at zero. The loop stops early when
    |delta| <= 1e-9 * (1 + |x_new|), or if the Newton matrix is singular (the current x_new is kept).
    Returns x_new, or None if it went non-finite.

    Why five parameters: as implicit_euler, for one sub-step.
    """
    N, n, nx = circuit.N, circuit.n, circuit.nx
    n_conc = 2 * N * n + N                             # concentrations and acid: everything but z
    eye = np.eye(nx)
    x_new = x_old.copy()
    for _ in range(iters):
        f = np.asarray(circuit.rhs(x_new, u_vec)).ravel()
        J = np.asarray(circuit.jac(x_new, u_vec))
        g = x_new - x_old - h * f                      # the residual the step must zero
        try:
            delta = np.linalg.solve(eye - h * J, -g)
        except np.linalg.LinAlgError:
            break
        x_new = x_new + delta
        x_new[:n_conc] = np.maximum(x_new[:n_conc], 0.0)
        if not np.all(np.isfinite(x_new)):
            return None
        if np.linalg.norm(delta) <= 1e-9 * (1.0 + np.linalg.norm(x_new)):
            break
    return x_new


# --------------------------------------------------------------------------
# simulate: the plant in time
# --------------------------------------------------------------------------
def simulate(circuit: Circuit, x0: np.ndarray, u_of_t, t_end: float,
             dt_out: float = 0.1, breakpoints=(), rtol=1e-6, atol=1e-9) -> Trajectory:
    """Run the circuit from x0 for t_end hours.

    u_of_t(t) -> Inputs or vector. Put the times of any step changes in `breakpoints` so the
    integrator restarts there instead of stumbling over them. Between breakpoints the inputs are
    taken as constant: each segment uses u_of_t just after its start.

    Parameters:
        circuit: the circuit (plant or model).
        x0: the starting state.
        u_of_t: function of time (h) giving the inputs.
        t_end: how long to run, hours.
        dt_out: spacing of the output times, hours (each segment's end is always an output too).
        breakpoints: times (h) where the inputs change; ones outside (0, t_end) are ignored.
        rtol, atol: integrator tolerances.
        (Eight parameters, more than the guidelines like. The last four have defaults and most
        callers leave them alone; the signature is used across the project, so it stays.)
    Returns:
        a Trajectory. A segment's end time appears twice: as the last output of one segment and the
        first of the next.
    Raises:
        RuntimeError if a segment can't be integrated even by the fallbacks.

    Usage:
        traj = simulate(plant, x0, scenario.u_of_t(u), 40.0, dt_out=0.5, breakpoints=scenario.breakpoints)

    Why eight parameters: circuit, start, inputs over time and end time, plus four integrator settings with defaults.
    """
    times = sorted({0.0, float(t_end), *[float(b) for b in breakpoints if 0 < b < t_end]})
    ts, xs, us = [], [], []
    x = np.asarray(x0, float)
    for a, b in zip(times[:-1], times[1:]):
        u_seg = _u_vec(u_of_t(a + 1e-9))               # the inputs just after the segment starts
        sol = _integrate(_Segment(circuit, u_seg, rtol, atol), x, (a, b), _output_times(a, b, dt_out))
        ts.append(sol.t)
        xs.append(sol.y.T)
        us.append(np.tile(u_seg, (len(sol.t), 1)))
        x = sol.y[:, -1]
    return Trajectory(np.concatenate(ts), np.vstack(xs), np.vstack(us))


def _output_times(a: float, b: float, dt_out: float) -> np.ndarray:
    """Output times for the segment [a, b]: every dt_out hours from a, with b added at the end."""
    t_eval = np.arange(a, b, dt_out)
    if len(t_eval) == 0 or t_eval[-1] < b:
        t_eval = np.append(t_eval, b)
    return t_eval


# --------------------------------------------------------------------------
# steady_state: where the circuit settles under constant inputs
# --------------------------------------------------------------------------
def _root(circuit: Circuit, x: np.ndarray, uv: np.ndarray, tol: float):
    """Newton polish: solve rhs(x, uv) = 0 starting from x (scipy's hybrid method, CasADi Jacobian).

    Returns the root, or None if it fails, goes negative (a concentration or acid below -1e-9),
    or does not actually zero the residual (max |rhs| must be under 1e-7).
    """
    sol = root(lambda y: circuit.rhs(y, uv), x, jac=lambda y: circuit.jac(y, uv), method="hybr", tol=tol)
    n_conc = 2 * circuit.N * circuit.n + circuit.N
    if sol.success and np.all(sol.x[:n_conc] >= -1e-9) and np.abs(circuit.rhs(sol.x, uv)).max() < 1e-7:
        return sol.x
    return None


def steady_state(circuit: Circuit, u, x0: np.ndarray | None = None,
                 t_settle: float = 50.0, tol: float = 1e-8) -> np.ndarray:
    """Steady state for constant inputs u.

    With a warm start x0 (a nearby solution) the root solve is tried first;
    otherwise, and whenever the root solve is not trusted, the circuit is
    integrated into the basin first. Integrating keeps Newton out of the
    region where acid goes negative and D blows up.

    The ladder:
      1. x0 given: Newton from x0. Done if it's trusted.
      2. Integrate t_settle hours (from x0, or from a metal-free circuit), then Newton.
      3. Integrate 4 * t_settle hours more, then Newton; if Newton still fails, return the
         integrated state as it is.

    Parameters:
        circuit: the circuit.
        u: the inputs (Inputs or vector), held constant.
        x0: a nearby state to start from, or None.
        t_settle: hours to integrate before the first polish.
        tol: tolerance handed to the root solver.
    Returns:
        the steady state.

    Why five parameters: circuit, inputs and a warm start, plus two solver settings with defaults.
    """
    uv = _u_vec(u)
    if x0 is not None:
        x = _root(circuit, np.asarray(x0, float), uv, tol)
        if x is not None:
            return x
    x = circuit.initial_state(u) if x0 is None else np.asarray(x0, float)
    x = simulate(circuit, x, lambda t: uv, t_settle, dt_out=t_settle).x[-1]
    polished = _root(circuit, x, uv, tol)
    if polished is not None:
        return polished
    x = simulate(circuit, x, lambda t: uv, 4 * t_settle, dt_out=4 * t_settle).x[-1]
    # the slowest mode of a reflux circuit outlives the integration; Newton from this close usually finishes it
    polished = _root(circuit, x, uv, tol)
    if polished is None:
        # Why the warning: the integrated state is returned as it is, and it may not be at rest (a
        # book-plant search found "steady states" with recovery above 100 %). Callers that need a
        # true steady state should check the residual; this makes sure it never passes silently.
        warnings.warn(f"steady_state: not converged, returning the state after {5 * t_settle:g} h "
                      f"(largest |dx/dt| {np.abs(circuit.rhs(x, uv)).max():.2e})", RuntimeWarning, stacklevel=2)
        return x
    return polished
