"""The plant over OPC UA (src/plant_server.py): a short run with the controller as a client gives
exactly the handles of the same run in one process, and the historian records the tags."""
import multiprocessing as mp

import pytest

from src import plant_server as ps
from src.historian import Historian, series
from src.runs import RunSession, RunSpec

SPEC = RunSpec(config="ndpr_main", plant="perturbed", controller="mpc_level_loops", upset="feed_switch",
               sensors="product_raffinate_feed", hours=3.0, run_file={"plant": {"seed": 1, "generator": "ratios"}})
ENDPOINT = "opc.tcp://127.0.0.1:48411/sx-test/"


def test_the_same_run_over_opc_ua_and_in_one_process(tmp_path):
    """A run over OPC UA gives the same moves as the same run in one process."""
    ready = mp.Event()
    server = mp.Process(target=ps.serve, args=(SPEC, ENDPOINT, None, ready))
    server.start()
    try:
        assert ready.wait(600)
        historian = Historian(ENDPOINT, tmp_path / "hist.sqlite")
        client = ps.ControllerClient(SPEC, ENDPOINT)
        remote = client.run(SPEC.hours)
        historian.close()
        client.close()
    finally:
        server.join(60)
        if server.is_alive():
            server.terminate()
    local = [rec.handles for rec in RunSession(SPEC).steps()]
    assert len(remote) == len(local) == 6
    assert all(float(a[k]) == float(b[k]) for a, b in zip(remote, local) for k in a)
    t, values = series(tmp_path / "hist.sqlite", ps.SIM_T)
    assert t.size >= 1 and max(values) == pytest.approx(3.0)
