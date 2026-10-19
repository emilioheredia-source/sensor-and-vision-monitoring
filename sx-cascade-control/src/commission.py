"""Bring a circuit to its operating point the way a plant is commissioned.

Two steps, as on a real plant:

  commission          Run the circuit up on the nominal inputs until it
                      settles, then set each jackleg so its interface sits at
                      the setpoint for the phases that stage actually holds.
                      Returns the steady state and the inputs with the
                      jacklegs set.

  trim_under_control  After start-up, run the controller on the plant with
                      full information until the product is on spec and the
                      handles stop moving. Where they end is the plant's own
                      operating point, and every study starts there
                      (ARCHITECTURE.md decision 21, APPENDIX_B_control.md B.9).

On a plant that is the model, the trim moves nothing. On a plant that differs
from the model, the extra reagent the trim needs is what the model's error costs.

The interface setpoint is a fraction of the settler height (parameters.toml
[stage_geometry]: 0.5 of 0.5 m, both chosen).
"""
from __future__ import annotations

import numpy as np

from .circuit import Circuit, Inputs
from .simulate import steady_state


# --------------------------------------------------------------------------
# Start-up: settle on the nominal inputs, then set the jacklegs
# --------------------------------------------------------------------------
def commission(circuit: Circuit, u: Inputs) -> tuple[np.ndarray, Inputs]:
    """Settle the circuit on the inputs u, then set every jackleg for the interface setpoint.

    The jackleg height that puts an interface at a given height depends on the two phase
    densities, and those depend on the loading. So the circuit is first run to its steady
    state with the jacklegs as given, each jackleg is then computed from that stage's
    compositions, and the circuit is settled again with the new jacklegs.

    Parameters:
        circuit: the circuit (plant or model).
        u: the starting inputs (flows L/h, concentrations mol/L, jackleg heights m).
    Returns:
        (x, u_set): the steady state with the interfaces at setpoint, and u with the jacklegs set.

    Usage:
        x_rest, u_design = commission(plant, nominal_inputs(rc, plant).replace(k_factor=plant_k))
    """
    geo = circuit.p.raw["stage_geometry"]
    z_set = geo["interface_setpoint_fraction"] * geo["settler_height_m"]     # interface setpoint, m

    x = steady_state(circuit, u)
    u2 = u.replace(H_jackleg=_jacklegs_for_setpoint(circuit, x, z_set))
    # concentrations do not depend on the interface (one-way coupling), so only z moves
    x2 = steady_state(circuit, u2, x0=x, t_settle=5.0)
    return x2, u2


def _jacklegs_for_setpoint(circuit: Circuit, x: np.ndarray, z_set: float) -> np.ndarray:
    """Jackleg height (m) for each stage that puts its interface at z_set (m), given the stage's
    aqueous and organic compositions in state x (Circuit.jackleg_for_interface does the balance)."""
    c_aq, c_org, _, _ = circuit.unpack(x)
    return np.array([circuit.jackleg_for_interface(np.asarray(c_aq[j], float).ravel(),
                                                   np.asarray(c_org[j], float).ravel(), z_set)
                     for j in range(circuit.N)])


# --------------------------------------------------------------------------
# The trim: the controller finds the plant's own operating point
# --------------------------------------------------------------------------
def trim_under_control(plant: Circuit, x_rest: np.ndarray, k_factor: np.ndarray, controller,
                       t_settle: float = 30.0) -> tuple[np.ndarray, Inputs]:
    """What the commissioning crew does after start-up: run the controller on the real plant,
    with full information (hand samples at every stage), until the product is on spec and the
    handles have stopped moving. The model's design point is where the handles start; where they
    end is the plant's own operating point. On a plant that is the model, nothing moves.

    Parameters:
        plant: the plant circuit.
        x_rest: the plant at rest after commission.
        k_factor: the plant's own stage efficiencies (multipliers on the transfer rate, one per stage).
        controller: a CircuitMPC built on the model at the design point.
        t_settle: hours the controller runs (no upset during that time).
    Returns:
        the plant at rest under the trimmed inputs, and those inputs (k_factor as the
        controller's model sees it, i.e. nominal).
    """
    from .loop import run_closed_loop
    traj, _ = run_closed_loop(plant, x_rest, controller, None, t_settle, plant_k_factor=k_factor,
                              plant_drift=False)      # a drifting plant is trimmed before its drift starts
    u = controller.apply(controller.handles())                # the handles where the controller left them
    # bring the plant to rest on those handles, with its own efficiencies
    x = steady_state(plant, u.replace(k_factor=k_factor), x0=traj.x[-1], t_settle=5.0)
    return x, u
