"""The deployment guard (src/deployment.py), on a stub controller: what each mode lets through, the
move limits, the watchdog, and the journal. No plant is simulated."""
import pytest

from src.deployment import DeploymentGuard, Limits


class Stub:
    """A controller that proposes a fixed organic flow, or fails, as told."""
    t_step, cut, u_nominal, level_loops = 0.5, "LaCe", None, None

    def __init__(self):
        self.want, self.fail, self.raise_ = 100.0, False, False
        self.last = {}

    def handles(self):
        return {"Q_org": 100.0, "Q_scrub": 20.0}

    def step(self, obs):
        if self.raise_:
            raise RuntimeError("solver crashed")
        self.last = {"qp_ok": not self.fail}
        return {"Q_org": self.want, "Q_scrub": 20.0}

    def apply(self, handles):
        return handles


def test_shadow_never_reaches_the_plant_but_journals_the_advice():
    stub = Stub()
    guard = DeploymentGuard(stub, mode="shadow")
    stub.want = 110.0
    for _ in range(4):
        out = guard.step(None)
    assert out["Q_org"] == 100.0
    assert guard.journal[-1]["proposed"]["Q_org"] == 110.0 and guard.journal[-1]["applied"]["Q_org"] == 100.0


def test_closed_loop_moves_at_most_two_percent_an_interval_and_twenty_in_all():
    stub = Stub()
    guard = DeploymentGuard(stub, mode="closed")
    stub.want = 150.0
    assert guard.step(None)["Q_org"] == pytest.approx(102.0)
    assert "Q_org" in guard.journal[-1]["clipped"]
    for _ in range(20):
        out = guard.step(None)
    assert out["Q_org"] == pytest.approx(120.0)


def test_advisory_applies_only_at_each_review():
    stub = Stub()
    guard = DeploymentGuard(stub, mode="advisory", limits=Limits(review_every_h=2.0))
    stub.want = 101.0
    applied = [guard.step(None)["Q_org"] for _ in range(4)]          # 0.5 h steps: a review at 2 h
    assert applied == [100.0, 100.0, 100.0, 101.0]


def test_watchdog_holds_after_three_failed_solves_and_after_a_crash_until_rearmed():
    stub = Stub()
    guard = DeploymentGuard(stub, mode="closed")
    stub.want, stub.fail = 101.0, True
    for _ in range(3):
        guard.step(None)
    assert guard.tripped and "watchdog" in guard.journal[-1]["alarm"]
    held = guard.applied["Q_org"]
    stub.fail, stub.want = False, 102.0
    assert guard.step(None)["Q_org"] == held                       # still out
    guard.rearm()
    assert guard.step(None)["Q_org"] == pytest.approx(102.0)
    stub.raise_ = True
    out = guard.step(None)
    assert guard.tripped and "raised" in guard.journal[-1]["alarm"] and out["Q_org"] == pytest.approx(102.0)


def test_mode_change_is_journalled():
    guard = DeploymentGuard(Stub(), mode="shadow")
    guard.set_mode("advisory")
    assert guard.mode == "advisory" and guard.journal[-1]["event"] == "mode shadow -> advisory"
    with pytest.raises(ValueError):
        guard.set_mode("autopilot")


def test_a_review_may_move_as_far_as_the_intervals_it_stands_for():
    stub = Stub()
    guard = DeploymentGuard(stub, mode="advisory", limits=Limits(review_every_h=2.0))
    stub.want = 150.0
    applied = [guard.step(None)["Q_org"] for _ in range(4)]          # four 0.5 h intervals: up to 8 %
    assert applied[-1] == pytest.approx(108.0)
