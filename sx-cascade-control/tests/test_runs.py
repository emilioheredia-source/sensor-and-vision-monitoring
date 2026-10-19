"""Checks on runs from a description (src/runs.py and src/registry.py).

A run is described by a RunSpec: which run file, which plant, which
controller, which upset, which sensors, how long. The command line, the
studies and the workbench all go through this one path, so it has to hold up:

- every registered plant, controller and sensor set says what it is,
- each controller runs and gets scored,
- a run can be stepped one control interval at a time and stopped early,
- edits to the run file and the controller settings actually reach the model,
- a saved run reads back exactly as it was.

The runs use ndpr_small (the NdPr product, 16 stages) on the perturbed plant
with the Eneabba -> Yangibana feed switch, for 8 h. The control interval is
0.5 h, so a full run has 17 time points: the start and the end of 16 intervals.
"""
import numpy as np
import pytest

from src.registry import CONTROLLERS, PLANTS, SENSOR_SETS
from src.runs import RunSpec, editable_values, execute, load, operating_point, save, start

# What every run here shares: ndpr_small on the perturbed plant, feed switch, 8 h.
SMALL = dict(config="ndpr_small", plant="perturbed", upset="feed_switch", hours=8.0)


def test_every_entry_is_explained():
    """Every plant, controller and sensor set in the registry has a title and a real explanation.

    Front ends list these entries and show the explanation to whoever picks
    one. An entry with no text there would be a choice nobody can make sense of.
    """
    for table in (PLANTS, CONTROLLERS, SENSOR_SETS):
        for e in table.values():
            assert e.title and len(e.explanation) > 20


@pytest.mark.parametrize("controller", ["open_loop", "mpc_fixed", "mpc"])
def test_each_controller_runs_and_scores(controller):
    """Each controller runs the full 8 h and comes back with a scorecard and series of the right shape.

    17 time points ending at 8 h, the four main scorecard entries, and an
    interface series for all 16 stages at every time point. Open loop moves
    no handles, so it has no handle series.
    """
    r = execute(RunSpec(controller=controller, **SMALL))
    assert len(r.t) == 17 and r.t[-1] == pytest.approx(8.0)
    assert set(r.scorecard) >= {"off_spec_mol", "lost_excess_mol", "interface_excursion_m_h", "reagent_mol"}
    # (time points, stages)
    assert r.series["interfaces"].shape == (17, 16)
    if controller == "open_loop":
        assert r.handles == {}


def test_steps_arrive_one_interval_at_a_time_and_can_stop():
    """A session hands back one control interval at a time and can be stopped partway.

    This is how the workbench draws a run as it goes. The first three steps
    end at 0.5, 1.0 and 1.5 h. Stopping after the third must end the run
    there, and the result then holds 4 time points: the start plus three intervals.
    """
    session = start(RunSpec(controller="mpc", **SMALL))
    seen = []          # the end time, h, of each step handed back
    for step in session.steps():
        seen.append(step.t)
        if len(seen) == 3:
            session.stop()
    assert seen == [0.5, 1.0, 1.5]
    assert len(session.result().t) == 4


def test_sensor_set_runs():
    """The MPC runs with analysers on the product, raffinate and feed in place of the full state.

    With a sensor set, the controller works from an estimator's state, not
    the true one. Purity has to stay a finite number all the way through.
    """
    r = execute(RunSpec(controller="mpc", sensors="product_raffinate_feed", **SMALL))
    assert np.all(np.isfinite(r.series["purity"]))


def test_edits_reach_the_model():
    """An edit in the RunSpec changes what the model is built from, and leaves the rest alone.

    Raising the feed flow from 600 to 660 L/h in the run file has to show up
    in the editable values and in the model's inputs at the operating point,
    while the unedited spec keeps 600 L/h. Editing one controller bound (the
    scrub flow, L/h) has to keep the other bounds at their defaults, for
    instance the organic flow at 600 to 4700 L/h.
    """
    base = RunSpec(controller="open_loop", **SMALL)
    edited = RunSpec(controller="open_loop", run_file={"flows": {"feed": 660.0}}, **SMALL)
    assert editable_values(edited)["run_file"]["flows"]["feed"] == 660.0
    assert operating_point(edited).u_model.Q_feed == 660.0
    assert operating_point(base).u_model.Q_feed == 600.0
    settings = editable_values(RunSpec(controller_settings={"bounds": {"Q_scrub": [40.0, 900.0]}}, **SMALL))
    assert settings["controller_settings"]["bounds"]["Q_scrub"] == [40.0, 900.0]
    assert settings["controller_settings"]["bounds"]["Q_org"] == [600.0, 4700.0]      # the other bounds kept


def test_save_and_load_round_trip(tmp_path):
    """A run saved to disk and loaded back is the same run.

    The spec, the full state history and the purity series have to come back
    exactly; the scorecard and the set of handles have to match too. A saved
    run is how results get compared later, so it can't lose or change anything.
    """
    r = execute(RunSpec(controller="mpc", **SMALL))
    back = load(save(r, tmp_path / "run"))
    assert back.spec == r.spec
    assert np.array_equal(back.x, r.x) and np.array_equal(back.series["purity"], r.series["purity"])
    assert back.scorecard["off_spec_mol"] == pytest.approx(r.scorecard["off_spec_mol"])
    assert set(back.handles) == set(r.handles)
