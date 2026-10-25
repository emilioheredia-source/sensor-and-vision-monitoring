"""The controller on the far side of an OPC UA interface, with a historian recording the plant.

The question. Every run so far had the plant and the controller in one Python process. On a plant
the controller is a client of the control system: it reads tags and writes setpoints over OPC UA.
Does the controller behave the same through that interface, and what does it cost?

The method. src/plant_server.py runs the plant (main case, first realistic plant, the feed switch,
the instruments of "product_raffinate_feed") as an OPC UA server in its own process, with the
settlers' level loops on the plant side as in a DCS. The guarded MPC (mpc_guarded: limits and
watchdog, deployment.py) runs as a client that sees only tags. src/historian.py subscribes to
every tag and stores each change in SQLite. The run is 80 h. Then the same run in one process,
and the two compared: every handle the controller set, and the plant's true state at the end.
The figure is drawn from the historian alone.

Run:  pixi run python notebooks/38_opcua.py
Writes runs/opcua_historian.sqlite, runs/opcua_ndpr_main.txt (the summary) and figures/opcua_historian.png.
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from src import plant_server as ps                                      # noqa: E402
from src.historian import Historian, series                             # noqa: E402
from src.runs import RunSession, RunSpec                                # noqa: E402

SPEC = RunSpec(config="ndpr_main", plant="perturbed", controller="mpc_guarded", upset="feed_switch",
               sensors="product_raffinate_feed", hours=80.0, run_file={"plant": {"seed": 1, "generator": "ratios"}})
ENDPOINT = "opc.tcp://127.0.0.1:48410/sx/"
DB = HERE / "runs" / "opcua_historian.sqlite"
PLANT_NPZ = HERE / "runs" / "opcua_plant_truth.npz"
SUMMARY = HERE / "runs" / "opcua_ndpr_main.txt"


def main():
    """The run over OPC UA with the historian, then the same run in one process; compare the handles and
    the final state, and write the summary."""
    ready = mp.Event()
    server = mp.Process(target=ps.serve, args=(SPEC, ENDPOINT, str(PLANT_NPZ), ready))
    server.start()
    ready.wait(600)
    historian = Historian(ENDPOINT, DB)
    t0 = time.time()
    client = ps.ControllerClient(SPEC, ENDPOINT)
    remote = []
    while client.t < SPEC.hours - 1e-9:
        remote.append(client.interval())
        historian.flush()
    seconds = time.time() - t0
    product_label = client.observer.product_label
    time.sleep(0.5)                                   # let the last data changes arrive
    historian.close()
    client.close()
    server.join(120)

    t1 = time.time()
    session = RunSession(SPEC)
    local = [rec.handles for rec in session.steps()]
    local_seconds = time.time() - t1
    handle_diff = max(abs(float(a[k]) - float(b[k])) for a, b in zip(remote, local) for k in a)
    truth = np.load(PLANT_NPZ)
    state_diff = float(np.max(np.abs(truth["x"][-1] - session.loop.x)))
    rows = series(DB, ps.SETPOINTS["Q_org"])[0].size
    import sqlite3
    with sqlite3.connect(str(DB)) as db:
        n_rows, n_tags = db.execute("SELECT COUNT(*), COUNT(DISTINCT tag) FROM samples").fetchone()
    lines = [f"intervals: {len(remote)} over OPC UA, {len(local)} in one process",
             f"largest difference in any handle: {handle_diff:.3g}; in the plant's final state: {state_diff:.3g}",
             f"wall time: {seconds:.0f} s over OPC UA ({client.seconds_waiting:.0f} s of it waiting for the plant), "
             f"{local_seconds:.0f} s in one process",
             f"historian: {n_rows} samples of {n_tags} tags ({rows} changes of the organic setpoint)"]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    figure()


def figure():
    """From the historian only: product purity as the analyser read it, the feed as its analyser
    read it (the switch to a richer feed at 5 h), and the organic flow setpoint the controller wrote."""
    import matplotlib.pyplot as plt
    from src.plotting import FIGURES, INK2, SLOT, style
    style()
    from src.metrics import heavy_mask
    session = RunSession(SPEC)                         # only for the analysers' labels and components
    observer = session.loop.observer
    heavy = np.array(heavy_mask(session.op.plant.p, session.op.rc.cut))

    def share_of_product(label):
        """An analyser's history as the share of Pr + Nd in what it reads, %, from the historian."""
        t, values = series(DB, ps.analyser_tag(label) + ".PV")
        keep = [k for k, v in enumerate(values) if np.sum(v) > 0]   # the empty value before the first reading
        sensor = next(s for s in session.loop.sensors.sensors if s.label() == label)
        idx = np.array([heavy[session.op.plant.p.index(n)] for n in sensor.components])
        return t[keep], np.array([100 * np.sum(np.array(values[k])[idx]) / np.sum(values[k]) for k in keep])

    t_p, purity = share_of_product(observer.product_label)
    t_f, feed_share = share_of_product(observer.feed_label)
    t_o, v_o = series(DB, ps.SETPOINTS["Q_org"])
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.4), sharex=True)
    axes[0].plot(t_p, purity, color=SLOT[0], lw=1.6)
    axes[0].axhline(99.0, color=INK2, lw=0.8, ls="--")
    axes[0].set_ylabel("product purity, %\n(analyser)")
    axes[1].plot(t_f, feed_share, color=SLOT[1], lw=1.6)
    axes[1].set_ylabel("Pr + Nd in the feed,\n% (analyser)")
    axes[2].step(t_o, v_o, where="post", color=SLOT[2], lw=1.6)
    axes[2].set_ylabel("organic setpoint,\nL/h (FIC-201)")
    axes[2].set_xlabel("plant time, h")
    axes[0].set_title("Read back from the historian: the plant over OPC UA, the controller as a client", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "opcua_historian.png", dpi=150)
    print("wrote", FIGURES / "opcua_historian.png")


if __name__ == "__main__":
    figure() if sys.argv[1:] == ["figure"] else main()     # "figure": redraw from the stored historian
