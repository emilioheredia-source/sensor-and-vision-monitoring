"""Checks on the circuit model (src/circuit.py) against things that are known exactly.

The circuit is the ODE model every other part of the project runs on. These
tests hold it to results that don't depend on any fitted number:

1. Kremser. One component, constant D, N ideal countercurrent stages. The
   fraction left in the raffinate is a textbook result:
       fraction left in raffinate = (E - 1) / (E^(N+1) - 1),  E = D * (O/A)
   If the stage equations, the flow directions or the stage count were off,
   the model wouldn't land on it.
2. Mass balance. At steady state, what comes in has to go out, metal by
   metal and for acid too.
3. Layout. The map builder puts each stream into and out of the stage the
   Idaho flowsheet says.
4. Reflux and the interface: the scrub reflux closes its own balance, and the
   interface sits where the jackleg puts it.

Stages are numbered from 0 at the raffinate end. Organic flows 0 -> N-1.
Component order is LaCe, Pr, Nd, SmPlus.
"""
from dataclasses import replace

import numpy as np
import pytest

from src.circuit import Circuit, Inputs, build_map
from src.params import load_params
from src.simulate import steady_state

p = load_params()
n = len(p.names)       # number of components (4)


def _inputs(cmap, Q_org=300.0, Q_feed=300.0, Q_scrub=100.0, Q_strip=100.0,
            c_feed=None, h_feed=0.05, h_scrub=0.2, h_strip=2.0, sapon=0.0, H_jackleg=0.41):
    """An Inputs for the given map, with plain test values that can be overridden.

    Flows are L/h, concentrations and acid mol/L, the jackleg height m. The
    feed is metal-free unless c_feed is given. The scrub carries acid only
    (no metal), saponification is off, and every jackleg is set the same.

    Why eleven parameters: one keyword per Inputs field, each with a default, so a test overrides only what it needs.
    """
    c = np.zeros(n) if c_feed is None else np.asarray(c_feed, float)
    return Inputs(Q_org, Q_feed, Q_scrub, Q_strip, c, h_feed, np.zeros(n), h_scrub, h_strip,
                  sapon, np.full(cmap.N, H_jackleg))


def test_map_layout():
    """Each stream enters and leaves the stage the Idaho layout says.

    2 extraction, 12 scrub and 4 strip stages give 18 stages. Feed joins at
    the last extraction stage (1), scrub liquor at the last scrub stage (13),
    strip acid at the last stage (17), and fresh organic at stage 0. Raffinate
    leaves stage 0 and product leaves the first strip stage (14). The strip
    aqueous must not run on into the scrub section: it leaves as product.
    Getting one of these wrong would give a different flowsheet that still runs.
    """
    m = build_map(2, 12, 4)
    assert (m.N, m.product_stage) == (18, 14)
    assert m.organic_recycle
    assert {("aq", "feed", 1), ("aq", "scrub", 13), ("aq", "strip", 17), ("org", "organic", 0)} <= _streams_in(m)
    assert {("aq", 0, "raffinate"), ("aq", 14, "product")} <= _streams_out(m)
    # strip aqueous does not flow into the scrub section
    assert ("aq", 14, 13) not in {(e.phase, e.src, e.dst) for e in m.edges}


def _streams_in(m) -> set:
    """(phase, source name, stage) for every stream that comes from outside the circuit."""
    return {(e.phase, e.src, e.dst) for e in m.edges if isinstance(e.src, str)}


def _streams_out(m) -> set:
    """(phase, stage, sink name) for every stream that leaves the circuit."""
    return {(e.phase, e.src, e.dst) for e in m.edges if isinstance(e.dst, str)}


def test_aqueous_flow_coefficients():
    """Each stage's aqueous flow is the sum of the right source streams.

    Flows are conserved, so a stage's aqueous outflow is the sum of the
    sources upstream of it. The circuit works this out once from the map as a
    table of 0s and 1s: one row per stage, columns feed, scrub, strip. With 2
    extraction, 3 scrub and 2 strip stages: the extraction stages carry feed
    plus scrub, the scrub stages scrub only, the strip stages strip acid only.
    """
    c = Circuit(build_map(2, 3, 2), p)
    # coef[stage] = [feed, scrub, strip], 1 where that source's flow passes through the stage
    coef = c._aq_coef
    assert np.array_equal(coef[0], [1, 1, 0])      # raffinate end sees feed + scrub
    assert np.array_equal(coef[1], [1, 1, 0])
    assert np.array_equal(coef[2], [0, 1, 0])      # scrub section above the feed: scrub only
    assert np.array_equal(coef[5], [0, 0, 1])      # strip section: strip acid only


@pytest.mark.parametrize("N,E_target", [(3, 2.0), (6, 1.5), (4, 0.8)])
def test_kremser_single_component(N, E_target):
    """An extraction-only circuit leaves the fraction in the raffinate that the Kremser equation gives.

    Kremser is the textbook answer for N ideal countercurrent stages with a
    constant D. The model's D normally depends on acid and loading, and its
    stages are not quite ideal, so the test switches those off: no acid
    dependence, a feed dilute enough not to load the organic, and a transfer
    rate 2000 times the normal one. What's left is the bare stage-to-stage
    bookkeeping, which has to match Kremser to 0.1 %. The cases cover more
    and fewer stages, and an extraction factor below 1 (E = 0.8) as well as above.
    """
    # constant D: no acid dependence, negligible loading, ideal stages (fast transfer)
    OA = 1.0                                     # organic to aqueous flow ratio
    D_target = E_target / OA                     # the D that gives this extraction factor E = D * O/A
    L3 = p.dimer_total ** 3                      # free dimer cubed, (mol/L)^3; D = K * L^3 with acid off
    K = np.array([D_target / L3, 0, 0, 0])              # only LaCe present
    # no acid dependence at all: exponent 0 and the acid floor off (h^0 + floor^0 would be 2)
    pk = replace(p, n_acid=0.0, h_min_D=0.0, D_max=0.0, K=K, k_transfer=p.k_transfer * 2000)   # efficiency 0.99997
    # N extraction stages, no scrub, no strip, fresh organic in (no recycle)
    cmap = build_map(N, 0, 0, organic_recycle=False)
    circ = Circuit(cmap, pk)
    c_feed = np.array([1e-5, 0, 0, 0])                  # dilute, so loading does not eat the extractant
    u = _inputs(cmap, Q_org=300.0 * OA, Q_feed=300.0, Q_scrub=0.0, Q_strip=0.0, c_feed=c_feed)
    x = steady_state(circ, u)
    out = circ.outlets(x, u.numpy())
    left = out["c_raff"][0] / c_feed[0]        # fraction of the feed's LaCe that leaves in the raffinate
    E = E_target
    kremser = (E - 1) / (E ** (N + 1) - 1)
    assert left == pytest.approx(kremser, rel=1e-3)


def test_mass_balance_closes_full_circuit():
    """At steady state, metal in equals metal out for every component, and acid balances too.

    Metal: what comes in with the feed leaves in the raffinate or the product.
    The organic is recycled, so it carries nothing out of the circuit net.
    Acid: acid in (feed, scrub, strip) minus acid out (raffinate, product) has
    to equal the base the saponified organic actually neutralized. The test
    rebuilds that amount stage by stage the same way the circuit spends it.
    A balance that doesn't close means a missing or doubled term in the equations.
    Last, no concentration or acid may go negative.
    """
    # 2 extraction, 6 scrub, 3 strip stages
    cmap = build_map(2, 6, 3)
    circ = Circuit(cmap, p)
    # feed, mol/L of LaCe, Pr, Nd, SmPlus
    c_feed = np.array([0.20, 0.03, 0.10, 0.0])
    # 30 % of the extractant saponified, so there is base to account for in the acid balance
    u = _inputs(cmap, c_feed=c_feed, sapon=0.3)
    x = steady_state(circ, u)
    uv = u.numpy()
    out = circ.outlets(x, uv)
    # metal: feed in = raffinate + product + (spent - recycled organic) ; recycle means spent == entering
    metal_in = u.Q_feed * c_feed                                     # mol/h per component
    metal_out = out["Q_raff"] * out["c_raff"] + out["Q_prod"] * out["c_prod"]
    assert np.allclose(metal_in, metal_out, rtol=1e-6, atol=1e-9)
    # acid: in (feed, scrub, strip) + 3 * metal extracted net ... net extraction across the whole
    # circuit is zero at steady state with recycle (what loads must strip), so:
    # acid in - acid out = base dose actually consumed
    acid_in = u.Q_feed * u.h_feed + u.Q_scrub * u.h_scrub + u.Q_strip * u.h_strip     # mol/h
    acid_out = out["Q_raff"] * out["h_raff"] + out["Q_prod"] * out["h_prod"]          # mol/h
    # the base carried by the organic is spent stage by stage where there is acid (circuit.acid_floor
    # makes that smooth); whatever is left after the last stage leaves with the spent organic
    _, _, h, _ = circ.unpack(x)
    # base fed with the organic, mol/h: saponified fraction * extractant monomer mol/L * organic L/h
    dose = u.saponified_fraction * p.raw["extractant"]["concentration_monomer"] * u.Q_org
    left = dose
    for j in range(circ.N):
        hj = float(h[j])
        left -= left * hj / (hj + circ.acid_floor)
    assert acid_in - acid_out == pytest.approx(dose - left, rel=1e-6)
    assert left < 1e-3 * dose              # at normal acidity it is all spent in the first stages
    # the first 2*N*n + N entries of the state are c_aq, c_org and h: none may be negative
    assert np.all(x[: 2 * circ.N * n + circ.N] >= -1e-12)


def test_reflux_scrub_balances_and_sharpens():
    """With part of the product sent back as scrub, the balance still closes and the product gets purer.

    This is the Pr|Nd validation split on a didymium feed (Pr and Nd only),
    on the Idaho-size circuit. In reflux mode the scrub liquor is strip
    product. So the net product flow must be strip minus scrub, the metal
    balance must close with that loop in place, and a scrub rich in Nd should
    push Pr out of the organic and give a purer Nd product at the same flows.
    That's the reason plants use a reflux scrub.
    """
    cmap = build_map(2, 12, 4)
    c_feed = np.array([0.0, 0.0885, 0.2602, 0.0])        # didymium
    # flows L/h, scrub acid 0.5 mol/L, 40 % saponified
    u = _inputs(cmap, Q_org=1200.0, Q_scrub=300.0, Q_strip=300.0, c_feed=c_feed, h_scrub=0.5, sapon=0.4)
    uv = u.numpy()
    plain = Circuit(cmap, p, scrub_reflux=False)        # acid-only scrub
    reflux = Circuit(cmap, p, scrub_reflux=True)        # scrub liquor is strip product
    xp = steady_state(plain, u)
    xr = steady_state(reflux, u)
    op, orf = plain.outlets(xp, uv), reflux.outlets(xr, uv)
    # net product flow is what is left after the reflux is taken back
    assert orf["Q_prod"] == pytest.approx(u.Q_strip - u.Q_scrub)
    # metal balance closes with the reflux loop in place
    for out in (op, orf):
        metal_out = out["Q_raff"] * out["c_raff"] + out["Q_prod"] * out["c_prod"]
        assert np.allclose(u.Q_feed * c_feed, metal_out, rtol=1e-6, atol=1e-9)
    # a Nd-rich scrub crowds Pr out of the organic: product purer, at the same flows
    pur = lambda o: o["c_prod"][2] / o["c_prod"][1:3].sum()     # Nd / (Pr + Nd) in the product
    assert pur(orf) > pur(op)


def test_interface_sits_where_the_jackleg_puts_it():
    """Set the jackleg for a target interface height, and the steady state puts the interface there.

    The jackleg height that gives a target interface comes from a hydrostatic
    balance between the two phase densities. That's how an operator sets it at
    commissioning. The interface ODE has to settle exactly there. A heavier
    (loaded) organic needs a higher jackleg for the same interface height,
    which is the direction the physics says.
    """
    # a single extraction stage, no recycle
    cmap = build_map(1, 0, 0, organic_recycle=False)
    circ = Circuit(cmap, p)
    geo = p.raw["stage_geometry"]
    # interface setpoint, m above the settler floor: a fraction of the settler height
    z_set = geo["interface_setpoint_fraction"] * geo["settler_height_m"]
    # jackleg height, m, for metal-free phases
    Hj = circ.jackleg_for_interface(np.zeros(n), np.zeros(n), z_set)
    u = _inputs(cmap, Q_scrub=0.0, Q_strip=0.0, H_jackleg=Hj)
    x = steady_state(circ, u)
    _, _, _, z = circ.unpack(x)
    assert float(z[0]) == pytest.approx(z_set, abs=1e-6)
    # a heavier (loaded) organic lowers the interface for the same jackleg
    Hj2 = circ.jackleg_for_interface(np.zeros(n), np.array([0, 0, 0.15, 0]), z_set)
    assert Hj2 > Hj
