"""Checks on the structural plants (src/plant_models.py): plants built differently from the model.

The controller's model treats each stage as one well-mixed pool per phase.
These plants add things the model doesn't have, to test whether the
controller copes with dynamics and losses it can't see:

- SettlerCircuit: transfer happens in a small mixer only, then the phases
  flow through a settler split into tanks in series. The total hold-up is the
  model's, so the steady state must be the model's; only the dynamics differ.
- EntrainmentCircuit: each phase carries droplets of the other into the next
  stage, more of them as the interface strays toward a weir. That pushes
  metal back the wrong way and costs separation.

The tests check that the settler plant agrees with the model at steady state
and lags it in a transient, that the controller runs on it, and that
entrainment has the size and the effect it's meant to have.

All runs use ndpr_small: the NdPr product, cut between La+Ce and Pr.
"""
import numpy as np
import pytest

from src.commission import commission
from src.config import build_circuit, load_config, nominal_inputs
from src.plant_models import SettlerCircuit
from src.runs import RunSpec, execute
from src.simulate import simulate


@pytest.fixture(scope="module")
def pair():
    """ndpr_small's model and a settler plant built on the same map and parameters.

    The settler in each stage is split into 3 tanks in series per phase.
    Returns (run config, model, plant).
    """
    rc = load_config("ndpr_small")
    model = build_circuit(rc)
    plant = SettlerCircuit(model.cmap, model.p, model=model.model, scrub_reflux=model.scrub_reflux, cells=3)
    return rc, model, plant


def test_state_starts_with_the_model_layout(pair):
    """The settler plant's state is the model's state followed by the extra settler cells.

    The first part of the plant's state has exactly the model's layout and
    holds what leaves each stage. That's what lets sensors, metrics and
    commissioning read the plant as if it were the model. After it come 3
    more blocks (the mixers and the inner settler cells), each holding the
    aqueous and organic concentrations and the acid of every stage.
    """
    _, model, plant = pair
    assert plant.nx_model == model.nx
    # 3 extra blocks, each: aqueous + organic concentrations (N stages x n components each) + acid (N)
    assert plant.nx == model.nx + 3 * (2 * model.N * model.n + model.N)


def test_same_steady_state_as_the_model(pair):
    """Commissioned on the same inputs, the settler plant and the model reach the same steady state.

    At steady state a settler just passes on what its mixer makes, so the
    two have to agree, to 1e-9, in everything the model can see. The
    jacklegs commissioning sets have to agree too. This is what makes the
    settler plant a clean test: any difference in a run comes from the
    dynamics alone.
    """
    rc, model, plant = pair
    u = nominal_inputs(rc, model)
    x_model, u_model = commission(model, u)
    x_plant, u_plant = commission(plant, u)
    # observable() is the part of the plant's state in the model's layout
    assert np.allclose(plant.observable(x_plant), x_model, atol=1e-9)
    assert np.allclose(u_plant.H_jackleg, u_model.H_jackleg)


def test_settler_delays_what_leaves_the_stage(pair):
    """Right after a feed step the plant's raffinate has moved less than the model's: the change
    has to come through the settler first.

    Both start at the same steady state and get a 30 % stronger feed. After
    0.05 h (3 minutes) the total metal in the raffinate has to have risen in
    both, but by less in the settler plant. That's the extra lag the
    controller's model doesn't know about.
    """
    rc, model, plant = pair
    x_model, u0 = commission(model, nominal_inputs(rc, model))
    x_plant, _ = commission(plant, u0)
    # the same inputs with the feed 30 % stronger
    u1 = u0.replace(c_feed=np.asarray(u0.c_feed, float) * 1.3)
    moved = []         # rise in total raffinate metal, mol/L: [model, plant]
    for circuit, x0 in ((model, x_model), (plant, x_plant)):
        x = simulate(circuit, x0, lambda t: u1, 0.05, dt_out=0.05).x[-1]
        before = circuit.outlets(x0, u1.numpy())["c_raff"].sum()
        moved.append(circuit.outlets(x, u1.numpy())["c_raff"].sum() - before)
    assert 0 < moved[1] < moved[0]


@pytest.mark.parametrize("sensors", ["full_state", "product_raffinate_feed"])
def test_controller_runs_on_the_settler_plant(sensors):
    """The MPC runs a 4 h feed switch on the settler plant, with full state and with analysers.

    Purity has to stay a finite number all the way through (no blow-up in the
    plant or the controller). The run's stored state has to be the settler
    plant's longer one, more than 10 times the number of interfaces, which
    shows the run really used the settler plant and not the model.
    """
    r = execute(RunSpec(config="ndpr_small", plant="settler", controller="mpc", upset="feed_switch",
                        sensors=sensors, hours=4.0))
    assert np.all(np.isfinite(r.series["purity"]))
    # state length per time point against the number of stages (one interface each)
    assert r.x.shape[1] > r.series["interfaces"].shape[1] * 10


def test_entrainment_rises_past_the_window():
    """Entrainment sits at its base value mid-window and climbs well past it outside the window.

    With the interface mid-settler (0.25 m) the aqueous carried in the
    organic is the 500 ppm base. With the interface 5 cm above the top of the
    window (0.40 m) aqueous carried in the organic has to be over 1 %. With
    it 5 cm below the bottom (0.10 m) organic carried in the aqueous has to be
    over 1 %. Most neighbouring stage pairs exchange droplets (there's no
    exchange where the aqueous doesn't flow back, at the scrub/strip boundary).
    """
    from src.plant_models import EntrainmentCircuit
    rc = load_config("ndpr_small")
    model = build_circuit(rc)
    plant = EntrainmentCircuit(model.cmap, model.p, model=model.model, scrub_reflux=model.scrub_reflux)
    # entrained_fractions(z) = (aqueous in organic, organic in aqueous) at interface height z, m
    mid, high, low = (float(v) for v in (plant.entrained_fractions(0.25)[0], plant.entrained_fractions(0.40)[0],
                                          plant.entrained_fractions(0.10)[1]))
    assert mid == pytest.approx(5e-4, rel=0.01)
    assert high > 0.01 and low > 0.01
    assert len(plant._neighbours()) >= model.N - 3


def test_entrainment_costs_separation():
    """Same inputs, entrainment on: more of the product side leaves in the raffinate.

    The plant here has 1 % entrainment everywhere (base = 0.01). Droplets
    carry metal back toward the raffinate end, so at steady state there has
    to be more Pr, Nd and Sm+ in the raffinate than in the model. That's the
    loss that makes keeping interfaces in their window worth something.
    """
    from src.plant_models import EntrainmentCircuit
    rc = load_config("ndpr_small")
    model = build_circuit(rc)
    plant = EntrainmentCircuit(model.cmap, model.p, model=model.model, scrub_reflux=model.scrub_reflux, base=0.01)
    x_m, u = commission(model, nominal_inputs(rc, model))
    x_p, _ = commission(plant, u)
    # c_raff[-3:] is Pr, Nd, SmPlus: everything on the product side of the La+Ce | Pr cut, mol/L
    lost = [c.outlets(x, u.numpy())["c_raff"][-3:].sum() for c, x in ((model, x_m), (plant, x_p))]
    assert lost[1] > lost[0]


def test_every_plant_answers_hidden_at():
    """The closed loop calls hidden_at on every plant: a plain circuit hands the state back as it
    is; a drifting plant (the "drifting" registry entry) sets its hidden drift states for the time."""
    from src.registry import PLANTS
    rc = load_config("ndpr_small")
    plain = build_circuit(rc)
    x = np.zeros(plain.nx)
    assert plain.hidden_at(x, 50.0) is x
    drifting, _ = PLANTS["drifting"].build(rc)
    xd = np.zeros(drifting.nx)
    assert not np.array_equal(drifting.hidden_at(xd.copy(), 50.0), drifting.hidden_at(xd.copy(), 0.0))
