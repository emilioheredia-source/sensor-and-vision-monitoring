"""The simulated plant behind OPC UA, and the controller as a client that only sees tags.

On a plant the advanced controller doesn't live inside the plant model. It runs on its own
server and talks to the control system (DCS or PLC/SCADA) over an industrial protocol, today
usually OPC UA: it reads process values and writes setpoints, like any other client. This file
splits the closed loop the same way:

    plant side (PlantServer, its own process)             controller side (ControllerClient)
    ------------------------------------------            -------------------------------------
    the plant, the upsets, the instruments                 the estimator and the MPC (any
    the settlers' level loops (local, in the DCS)          registry controller), seeing only tags
    the strip flow following the scrub flow
    publishes:  FI-101 feed flow, FI-401 product flow,  -> reads them, folds the readings into its
                AI-302 scrub acid, each analyser,          view of the plant (loop.Observer), steps
                LT-xx interfaces, SIM.T and SIM.STEP       the controller
    takes:      FIC-201 organic, FIC-301 scrub,         <- writes the setpoints, then asks for the
                AIC-302 scrub acid, XIC-202 saponif.       next interval (SIM.REQUEST)

The two step in lockstep: the plant runs one control interval when the client asks for it, so a
run over OPC UA can be checked against the same run in one process, number for number (notebook
38). A real plant runs on its own clock; lockstep is what makes the check possible.

Every value carries a source timestamp in simulated time (SIM_EPOCH plus the plant's hours), and an
analyser reading carries the time its sample was taken, so a historian subscribed to the tags
(historian.py) records the plant as a plant historian would.

Tag names follow ISA-5.1 habits (FI flow indicator, FIC flow controller, AI/AIC analyser, LT level
transmitter, XIC the saponification controller). They're made up for this simulation.
"""
from __future__ import annotations

import dataclasses
import time
from datetime import datetime, timedelta, timezone

import numpy as np

SIM_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
NAMESPACE = "urn:sx-cascade-control:plant"
SETPOINTS = {"Q_org": "FIC-201.SP", "Q_scrub": "FIC-301.SP", "h_scrub": "AIC-302.SP",
             "saponified_fraction": "XIC-202.SP"}
FEED_FLOW, PRODUCT_FLOW, SCRUB_ACID = "FI-101.PV", "FI-401.PV", "AI-302.PV"
SIM_T, SIM_STEP, SIM_REQUEST = "SIM.T", "SIM.STEP", "SIM.REQUEST"
STOP = -1


def analyser_tag(label: str) -> str:
    """The tag of one analyser: "AI-composition_aq@28" and so on, with .PV, .T (sample time, h) and
    .STEP (the interval it was last updated in)."""
    return f"AI-{label}"


def level_tag(stage: int) -> str:
    """The interface level transmitter of one settler (stage counted from 0): "LT-05.PV"."""
    return f"LT-{stage:02d}.PV"


def stamp(t_h: float) -> datetime:
    """Plant time (h) as a source timestamp: SIM_EPOCH plus that many hours."""
    return SIM_EPOCH + timedelta(hours=float(t_h))


# ==================================================================================================
# Plant side
# ==================================================================================================
class _Setpoints:
    """The plant side's stand-in for a controller: it holds the setpoints written to it and turns
    them into the plant's inputs the way the DCS would (the strip flow follows the scrub flow when
    they're tied). The settlers' level loops ride along, as in the in-process loop."""

    def __init__(self, op, t_step: float, strip_net: float | None, level_loops):
        """Start with the operating point's handles as the setpoints.

        Args:
            op: the operating point (operating_point.OperatingPoint).
            t_step: the control interval, h.
            strip_net: the strip flow above the scrub flow, L/h, when the two are tied; else None.
            level_loops: the settlers' level loops (loop.LevelLoops), or None.
        """
        self.u_nominal, self.cut, self._t_step = op.u_model, op.rc.cut, t_step
        self.strip_net, self.level_loops, self.last = strip_net, level_loops, {}
        self.held = {name: float(getattr(op.u_model, name)) for name in SETPOINTS}

    @property
    def t_step(self):
        return self._t_step

    def step(self, obs):
        return dict(self.held)

    def apply(self, handles):
        """The plant's inputs with the setpoints written in, and the strip flow following the scrub."""
        u = self.u_nominal.replace(**handles)
        if self.strip_net is not None:
            u = u.replace(Q_strip=self.strip_net + u.Q_scrub)
        return u


class _Capture:
    """Takes the place of loop.Observer on the plant side: keeps the readings to publish, and
    leaves the estimating to the controller side. (Same arguments as loop.Observer.update.)"""

    readings: dict = {}
    q_prod = None

    def update(self, t, u_cmd, q_feed, readings, q_prod):
        self.readings, self.q_prod = readings, q_prod


def serve(spec, endpoint: str, out_path: str | None = None, ready=None):
    """Run the plant side until the client writes SIM.REQUEST = STOP.

    Args:
        spec: the run (runs.RunSpec). Its plant, upset and sensors are used here; its controller
            only for its settings (the strip-to-scrub tie, the level loops).
        endpoint: where to listen, e.g. "opc.tcp://127.0.0.1:4840/sx/".
        out_path: where to save the plant's true trajectory (npz) for scoring, or None.
        ready: a multiprocessing Event, set once the server is listening, or None.
    """
    PlantSide(spec, endpoint).run(ready, out_path)


class PlantSide:
    """The plant, its upsets, instruments and level loops, behind an OPC UA server (see serve)."""

    def __init__(self, spec, endpoint: str):
        """Build the plant's closed loop and the server with every tag (nothing is served yet).

        Args:
            spec: the run (runs.RunSpec); see serve().
            endpoint: where to listen.
        """
        from asyncua import ua
        from asyncua.sync import Server
        self.ua = ua
        self.loop, self.op = _plant_loop(spec)
        self.N = self.op.plant.N
        self.server = Server()
        self.server.set_endpoint(endpoint)
        self.server.set_server_name("SX cascade (simulated)")
        idx = self.server.register_namespace(NAMESPACE)
        self.plant_obj, self.idx, self.nodes = self.server.nodes.objects.add_object(idx, "SXPlant"), idx, {}
        self._add_tags()
        self.ts, self.xs, self.us = [0.0], [self.loop.x.copy()], []

    def run(self, ready=None, out_path: str | None = None):
        """Serve one control interval per client request until it asks to stop."""
        self.server.start()
        try:
            self._publish_state(0)
            if ready is not None:
                ready.set()
            step_no = 0
            while self._next_request(step_no) != STOP:
                self.loop.controller.held = self._read_setpoints()
                self._record(self.loop.step())
                step_no += 1
                self._publish_state(step_no)
        finally:
            self.server.stop()
            self._save(out_path)

    def _read_setpoints(self) -> dict:
        """The setpoints the client wrote, as {handle name: value}."""
        return {name: float(self.nodes[tag].read_value()) for name, tag in SETPOINTS.items()}

    # ---- the tags -------------------------------------------------------------------------------
    def _add_tags(self):
        """Setpoints (writable), flows, the scrub acid, the clock, each analyser, each interface."""
        int64 = self.ua.VariantType.Int64
        for name, tag in SETPOINTS.items():
            self._add(tag, self.loop.controller.held[name], writable=True)
        for tag in (FEED_FLOW, PRODUCT_FLOW, SCRUB_ACID, SIM_T):
            self._add(tag, 0.0)
        self._add(SIM_STEP, 0, vtype=int64)
        self._add(SIM_REQUEST, 0, writable=True, vtype=int64)
        self._add_instrument_tags()

    def _add_instrument_tags(self):
        """The tags of each analyser and each settler's interface."""
        for sensor in self.loop.sensors.sensors:
            self._add_analyser(sensor)
        for j in range(self.N):
            self._add(level_tag(j), 0.0)

    def _add_analyser(self, sensor):
        """An analyser's reading (.PV), its sample time (.T) and the interval it came in (.STEP)."""
        base = analyser_tag(sensor.label())
        self._add(base + ".PV", [0.0] * max(len(sensor.components), 1))
        self._add(base + ".T", 0.0)
        self._add(base + ".STEP", -1, vtype=self.ua.VariantType.Int64)

    def _add(self, name, value, writable=False, vtype=None):
        """One variable under SXPlant, its first value stamped at plant time 0 (not wall time)."""
        vtype = vtype or self.ua.VariantType.Double
        node = self.plant_obj.add_variable(self.idx, name, self.ua.Variant(value, vtype))
        node.write_value(self.ua.DataValue(self.ua.Variant(value, vtype), SourceTimestamp=stamp(0.0)))
        if writable:
            node.set_writable()
        self.nodes[name] = node

    def _publish(self, name, value, t_h, int64=False):
        """Write a new value to a tag, stamped with plant time t_h (h)."""
        vtype = self.ua.VariantType.Int64 if int64 else self.ua.VariantType.Double
        self.nodes[name].write_value(self.ua.DataValue(self.ua.Variant(value, vtype), SourceTimestamp=stamp(t_h)))

    def _publish_state(self, step_no: int):
        """Everything the client reads after an interval; SIM.STEP last, since the client waits on it."""
        t = self.loop.t
        self._publish_flows_and_levels(t)
        self._publish_readings(step_no, t)
        self._publish(SIM_T, float(t), t)
        self._publish(SIM_STEP, step_no, t, int64=True)

    def _publish_flows_and_levels(self, t: float):
        """The feed flow, the scrub acid as the plant got it, and every settler's interface."""
        loop = self.loop
        self._publish(FEED_FLOW, float(loop.u_of_t(t).Q_feed), t)              # the feed for the next interval
        self._publish(SCRUB_ACID, float(loop.obs.h_scrub) if loop.obs.h_scrub is not None else 0.0, t)
        z = self.op.plant.observable(loop.x)[-self.N:]
        for j in range(self.N):
            self._publish(level_tag(j), float(z[j]), t)

    def _publish_readings(self, step_no: int, t: float):
        """The analysers with a fresh reading this interval, each stamped with its sample time."""
        captured = self.loop.observer
        for label, (value, t_meas) in captured.readings.items():
            base = analyser_tag(label)
            self._publish(base + ".PV", [float(v) for v in np.atleast_1d(value)], t_meas)
            self._publish(base + ".T", float(t_meas), t_meas)
            self._publish(base + ".STEP", step_no, t, int64=True)
        if captured.q_prod is not None:
            self._publish(PRODUCT_FLOW, float(captured.q_prod), t)

    # ---- the clock and the record ----------------------------------------------------------------
    def _next_request(self, step_no: int) -> int:
        """Wait until the client asks for the next interval or to stop; return what it asked."""
        while True:
            request = self.nodes[SIM_REQUEST].read_value()
            if request in (STOP, step_no + 1):
                return request
            time.sleep(0.002)

    def _record(self, rec):
        """Keep the interval's true plant state and inputs, for scoring the run afterwards."""
        self.ts.append(rec.t)
        self.xs.append(rec.x.copy())
        self.us.append(rec.u)

    def _save(self, out_path):
        """Write the true trajectory (t, x, u, the last input repeated) to out_path, if one was given."""
        if out_path is not None and self.us:
            np.savez(out_path, t=np.array(self.ts), x=np.array(self.xs), u=np.array(self.us + [self.us[-1]]))


def _plant_loop(spec):
    """The closed loop on the plant side: the run's plant, upsets and instruments, with the setpoint
    holder in the controller's place and _Capture in the Observer's. Returns (loop, operating point)."""
    from .operating_point import controller_settings
    from .runs import RunSession
    session = RunSession(dataclasses.replace(spec, controller="open_loop_level_loops"))
    loop, op = session.loop, session.op
    strip_net = controller_settings(op.rc, **spec.controller_settings).strip_tied_to_scrub
    loop.controller = _Setpoints(op, loop.t_step, strip_net, loop.controller.level_loops)
    loop.observer = _Capture()
    return loop, op


# ==================================================================================================
# Controller side
# ==================================================================================================
class ControllerClient:
    """The controller on its own, talking to the plant only through OPC UA tags.

    Usage:
        client = ControllerClient(spec, "opc.tcp://127.0.0.1:4840/sx/")
        handles = client.run(hours)            # one dict of handles per interval
        client.close(stop_plant=True)
    """

    def __init__(self, spec, endpoint: str):
        """Build the controller side of the run and connect to the plant server.

        Args:
            spec: the run (runs.RunSpec); its controller, sensors and settings are used here.
            endpoint: the plant server's address, e.g. "opc.tcp://127.0.0.1:4840/sx/".
        """
        from asyncua.sync import Client

        from .runs import RunSession

        session = RunSession(spec)                     # the controller, its estimator and Observer
        self.controller, self.loop = session.controller, session.loop
        self.observer, self.obs = session.loop.observer, session.loop.obs
        self.labels = [s.label() for s in session.loop.sensors.sensors]
        self.client = Client(endpoint)
        self.client.connect()
        idx = self.client.get_namespace_index(NAMESPACE)
        plant_obj = self.client.nodes.objects.get_child([f"{idx}:SXPlant"])
        self.node = lambda name: plant_obj.get_child([f"{idx}:{name}"])
        self.nodes = {name: self.node(name) for name in self._tag_names()}
        self.t, self.step_no = 0.0, 0
        self.seconds_waiting = 0.0

    def _tag_names(self) -> list[str]:
        """Every tag the client reads or writes: setpoints, flows, the clock, and each analyser's
        .PV, .T and .STEP."""
        names = [*SETPOINTS.values(), FEED_FLOW, PRODUCT_FLOW, SCRUB_ACID, SIM_T, SIM_STEP, SIM_REQUEST]
        return names + [analyser_tag(label) + suffix for label in self.labels for suffix in (".PV", ".T", ".STEP")]

    def read(self, name):
        return self.nodes[name].read_value()

    def write(self, name, value, int64=False):
        """Write a value to a tag (a double, or an integer with int64)."""
        from asyncua import ua
        vtype = ua.VariantType.Int64 if int64 else ua.VariantType.Double
        # stamped with plant time, so the historian files a setpoint with the interval it was set for
        self.nodes[name].write_value(ua.DataValue(ua.Variant(value, vtype), SourceTimestamp=stamp(self.t)))

    def interval(self) -> dict:
        """One control interval: read the feed flow, step the controller, write its setpoints, ask
        the plant for the interval, wait, read the instruments, update the controller's view."""
        q_feed = float(self.read(FEED_FLOW))
        self.obs.Q_feed = q_feed
        handles = self.controller.step(self.obs)
        u_cmd = self.controller.apply(handles)
        for name, tag in SETPOINTS.items():
            self.write(tag, float(getattr(u_cmd, name)))
        self.write(SIM_REQUEST, self.step_no + 1, int64=True)
        self._wait_for_plant()
        self.step_no += 1
        self.t = float(self.read(SIM_T))
        self.obs.h_scrub = float(self.read(SCRUB_ACID))
        readings = self._fresh_readings()
        self.observer.update(self.t, u_cmd, q_feed, readings, self._product_flow(readings))
        return handles

    def _wait_for_plant(self):
        """Wait until the plant has run the interval asked for (SIM.STEP moves on), and count the
        time spent waiting."""
        t0 = time.perf_counter()
        while self.read(SIM_STEP) != self.step_no + 1:
            time.sleep(0.002)
        self.seconds_waiting += time.perf_counter() - t0

    def _fresh_readings(self) -> dict:
        """The analysers updated in this interval, as {label: (value, sample time h)}, the shape
        loop.Observer.update takes. An analyser not updated this interval is left out."""
        readings = {}
        for label in self.labels:
            base = analyser_tag(label)
            if self.read(base + ".STEP") == self.step_no:
                readings[label] = (np.array(self.read(base + ".PV"), float), float(self.read(base + ".T")))
        return readings

    def _product_flow(self, readings: dict) -> float | None:
        """The product flowmeter, read only when the product analyser reported this interval."""
        product = self.observer.product_label
        return float(self.read(PRODUCT_FLOW)) if product is not None and product in readings else None

    def run(self, hours: float) -> list[dict]:
        """Run control intervals until plant time reaches `hours`; one dict of handles per interval."""
        out = []
        while self.t < hours - 1e-9:
            out.append(self.interval())
        return out

    def close(self, stop_plant: bool = True):
        """Disconnect; with stop_plant, tell the plant side to stop first (it then closes its end)."""
        if stop_plant:
            self.write(SIM_REQUEST, STOP, int64=True)
        try:
            self.client.disconnect()
        except ConnectionError:                     # the plant side closed first: nothing left to close
            pass
