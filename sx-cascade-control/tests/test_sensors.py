"""Checks on the sensors (src/sensors.py) and the state estimator (src/estimator.py).

With sensors, the controller doesn't see the plant's true state. It sees
readings: some quantities at some stages, with noise, a sample interval and
a delay. The estimator turns those readings into a state the controller can
use. These tests check each piece:

- each sensor reads the right entry of the state,
- slow, delayed sensors (a lab assay) give a reading only on their tick, and
  the reading is from the right time in the past,
- the noise is the size parameters.toml states,
- the estimator, fed readings, tracks a plant the model is wrong about better
  than the model running alone.

These run on prnd_small, the Pr|Nd validation split (cut = "Pr") on the
Idaho-size circuit. The plant's product is NdPr; Pr|Nd is for checking.
"""
import numpy as np
import pytest

from src.commission import commission
from src.config import build_circuit, load_config, nominal_inputs
from src.disturbances import FeedStep, Scenario
from src.estimator import Estimator
from src.sensors import Sensor, SensorSet
from src.simulate import simulate


@pytest.fixture(scope="module")
def plant():
    """prnd_small commissioned at its design point.

    Returns (run config, circuit, steady state x0, inputs u with the jacklegs set).
    """
    rc = load_config("prnd_small")
    circ = build_circuit(rc)
    x0, u = commission(circ, nominal_inputs(rc, circ))
    return rc, circ, x0, u


def test_rows_pick_the_right_state(plant):
    """Each kind of sensor reads exactly the quantity it's named for, at its stage.

    Without noise, an aqueous Nd analyser at stage 3 has to read the Nd in
    stage 3's aqueous, an interface sensor at stage 5 that stage's interface
    height, a pH probe at stage 2 that stage's acid, and an organic density
    meter at stage 1 the density the chemistry gives for that organic. An
    index off by one here would feed the controller the wrong stage.
    """
    rc, circ, x0, u = plant
    ss = SensorSet([Sensor("composition_aq", stage=3, components=("Nd",)), Sensor("interface", stage=5),
                    Sensor("acid", stage=2), Sensor("density_org", stage=1)], circ)
    # per stage: aqueous and organic concentrations (mol/L), acid (mol/L), interface height (m)
    c_aq, c_org, h, z = circ.unpack(x0)
    # the noise-free value each sensor would read, keyed by label ("kind@stage")
    v = {s.label(): ss.true_value(s, x0, u.numpy()) for s in ss.sensors}
    assert v["composition_aq@3"][0] == pytest.approx(float(c_aq[3][circ.p.index("Nd")]))
    assert v["interface@5"][0] == pytest.approx(float(z[5]))
    assert v["acid@2"][0] == pytest.approx(float(h[2]))
    from src.chemistry import rho_organic
    assert v["density_org@1"][0] == pytest.approx(rho_organic(np.asarray(c_org[1], float).ravel(), circ.p))


def test_sampling_and_delay(plant):
    """A product assay gives a reading only every 2 h, and that reading is from 4 h earlier.

    That's the lab-sample case: a shift takes a sample every two hours and
    the result comes back four hours later (parameters.toml). At 1 h there's
    no reading (off the 2 h tick). At 6 h there is one, and it's the sample
    taken at 2 h. At 2 h there's nothing yet, because the first sample isn't
    back. The controller has to live with exactly this timing.
    """
    rc, circ, x0, u = plant
    ss = SensorSet([Sensor("product_assay")], circ, seed=0)      # 2 h interval, 4 h delay by default
    # a fake history every 0.5 h from 0 to 10 h: (time h, state, inputs), the state scaled a little
    # with time so samples from different times differ
    hist = [(t, x0 * (1 + 0.01 * t), u.numpy()) for t in np.arange(0, 10.5, 0.5)]
    assert ss.readings(1.0, hist) == {}                             # off the 2 h tick
    r = ss.readings(6.0, hist)
    label = ss.sensors[0].label()                                   # pinned to the product stage: "product_assay@<j>"
    assert label in r
    _, t_meas = r[label]
    assert t_meas == pytest.approx(2.0)                             # 6 h minus 4 h delay
    assert ss.readings(2.0, hist) == {}                             # 2 - 4 < 0: nothing yet


def test_noise_is_the_stated_size(plant):
    """The interface sensor's noise has the standard deviation parameters.toml gives (2 mm).

    400 readings of the same steady interface are taken. Their spread has to
    be within 20 % of the stated value, which is loose enough for 400 samples
    and tight enough to catch noise that's off by a factor.
    """
    rc, circ, x0, u = plant
    ss = SensorSet([Sensor("interface", stage=0)], circ, seed=3)
    hist = [(0.0, x0, u.numpy())]                 # one point of history: the steady state at t = 0
    # 400 readings of stage 0's interface height, m
    vals = np.array([ss.readings(0.0, hist)["interface@0"][0][0] for _ in range(400)])
    sd = circ.p.raw["sensors"]["interface_noise_m"]
    assert abs(vals.std() - sd) < 0.2 * sd


def test_estimator_closes_on_the_plant(plant):
    """With readings, the estimator tracks a plant whose feed it doesn't know about; without, it doesn't.

    The plant gets a 30 % stronger feed at t = 0. Both estimators are stepped
    with the nominal inputs, so they're never told about it. One gets readings
    from two aqueous analysers (the feed stage and the last scrub stage) and a
    feed analyser; the other gets none. After 12 h the one with readings has
    to be within 10 % of the true state, and less than half as far off as the
    blind one. An input change nobody tells the estimator about can leave a
    steady bias, so the bar is 10 %, not zero.
    """
    rc, circ, x0, u = plant
    n_ext, n_scrub = rc["circuit"]["n_ext"], rc["circuit"]["n_scrub"]
    # aqueous analysers at the last extraction stage (where the feed joins) and the last scrub stage
    ss = SensorSet([Sensor("composition_aq", stage=n_ext - 1), Sensor("composition_aq", stage=n_ext + n_scrub - 1),
                    Sensor("feed_analyser")], circ, seed=2)
    # estimators stepping every 0.5 h, starting from the nominal steady state
    est = Estimator(circ, ss, x0, u, 0.5)
    blind = Estimator(circ, SensorSet([], circ), x0, u, 0.5)       # model only, no readings
    sc = Scenario([FeedStep(t=0.0, factor=1.3)])
    u_true = sc.u_of_t(u)(0.5)                    # the plant sees the step; the estimators do not know it
    # x: the true plant state; hist: (time h, state, inputs) at every step, what the sensors sample from
    x, hist = x0.copy(), [(0.0, x0.copy(), u.numpy())]
    for k in range(1, 25):                        # 24 steps of 0.5 h: 12 h
        t = 0.5 * k
        x = simulate(circ, x, lambda _t: u_true.numpy(), 0.5, dt_out=0.5).x[-1]
        hist.append((t, x.copy(), u_true.numpy()))
        x_hat = est.step(t, u, ss.readings(t, hist))      # predicted with the nominal (wrong) feed
        x_blind = blind.step(t, u, {})
    # relative error of each estimate against the true state at 12 h
    err = np.linalg.norm(x_hat - x) / np.linalg.norm(x)
    err_blind = np.linalg.norm(x_blind - x) / np.linalg.norm(x)
    # a persistent unmodelled input leaves a steady bias; readings must still beat the blind model
    assert err < 0.1
    assert err < 0.5 * err_blind
