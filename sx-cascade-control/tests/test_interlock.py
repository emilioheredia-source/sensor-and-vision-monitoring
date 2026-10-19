"""The feed-loss interlock (src/interlock.py) on a stub controller: trip, hold, wait for the feed
to be back rearm_h hours (a new drop restarts the wait), hand back."""
from types import SimpleNamespace

from src.interlock import FeedLossInterlock


class Stub:
    t_step, cut, level_loops = 0.5, "LaCe", None
    u_nominal = SimpleNamespace(Q_feed=600.0)

    def __init__(self):
        self.q_org, self.last, self.resyncs = 100.0, {}, 0

    def handles(self):
        return {"Q_org": self.q_org, "saponified_fraction": 0.4}

    def step(self, obs):
        self.q_org += 1.0
        self.last = {"qp_ok": True}
        return self.handles()

    def resync(self):
        self.resyncs += 1


def feed(q):
    return SimpleNamespace(Q_feed=q)


def test_trip_hold_and_hand_back_after_two_hours_of_feed():
    stub = Stub()
    lock = FeedLossInterlock(stub, rearm_h=2.0, stop_base=True)
    assert lock.step(feed(600.0))["Q_org"] == 101.0                   # armed: the controller steps
    held = lock.step(feed(0.0))                                       # pump stops: trip
    assert lock.tripped and held["Q_org"] == 101.0 and held["saponified_fraction"] == 0.0
    for _ in range(4):                                                # back 0, 0.5, 1.0, 1.5 h: still held
        assert lock.step(feed(600.0))["Q_org"] == 101.0
    lock.step(feed(0.0))                                              # a second drop restarts the count
    for _ in range(4):
        lock.step(feed(600.0))
    assert lock.tripped and stub.resyncs == 0
    out = lock.step(feed(600.0))                                      # 2 h of feed back: re-arm
    assert not lock.tripped and stub.resyncs == 1 and out["Q_org"] == 102.0 and lock.trips == 1
