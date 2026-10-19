"""A small process historian: subscribes to the plant's OPC UA tags and keeps every change in SQLite.

A plant historian (PI, Canary, and the like) records each tag's value when it changes, with the
time the value belongs to. This one does the same for the simulated plant (plant_server.py): an
OPC UA subscription on every tag, each data change stored as (tag, time, value), the time being
the value's source timestamp turned back into plant hours. Arrays (an analyser's reading of several
elements) are stored as JSON text.

Usage:
    hist = Historian("opc.tcp://127.0.0.1:4840/sx/", "runs/historian.sqlite")
    ...                                   # the plant runs
    hist.close()
    t, v = series("runs/historian.sqlite", "FI-101.PV")
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from .plant_server import NAMESPACE, SIM_EPOCH

SCHEMA = "CREATE TABLE IF NOT EXISTS samples (tag TEXT, t_h REAL, value TEXT)"


class _Handler:
    """Called by the subscription for each data change; queues rows for the writer."""

    def __init__(self, names: dict):
        """names: {node id: tag name}, so each change is stored under its tag."""
        self.names, self.rows, self.lock = names, [], threading.Lock()

    def datachange_notification(self, node, value, data):
        """Queue one change as (tag, plant hours, value as JSON). asyncua calls this by name, on its
        own thread; hence the lock."""
        t_h = _plant_hours(data.monitored_item.Value.SourceTimestamp)
        stored = json.dumps([float(v) for v in value]) if isinstance(value, (list, tuple)) else json.dumps(value)
        with self.lock:
            self.rows.append((self.names[node.nodeid], t_h, stored))


def _plant_hours(ts) -> float:
    """A source timestamp turned back into plant hours since SIM_EPOCH; NaN if there's none."""
    if not ts:
        return float("nan")
    return (ts.replace(tzinfo=SIM_EPOCH.tzinfo) - SIM_EPOCH).total_seconds() / 3600.0


class Historian:
    """Subscribes to every variable under SXPlant and writes each change to a SQLite file."""

    def __init__(self, endpoint: str, db_path: str | Path, period_ms: int = 50):
        """Start a fresh database file and subscribe to every tag.

        Args:
            endpoint: the plant server's OPC UA address.
            db_path: the SQLite file; a samples table already in it is dropped.
            period_ms: the subscription's publishing interval, ms.
        """
        from asyncua.sync import Client
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("DROP TABLE IF EXISTS samples")
        self.db.execute(SCHEMA)
        self.client = Client(endpoint)
        self.client.connect()
        idx = self.client.get_namespace_index(NAMESPACE)
        plant = self.client.nodes.objects.get_child([f"{idx}:SXPlant"])
        variables = plant.get_variables()
        names = {v.nodeid: v.read_browse_name().Name for v in variables}
        self.handler = _Handler(names)
        self.period_s = period_ms / 1000.0
        self.sub = self.client.create_subscription(period_ms, self.handler)
        self.sub.subscribe_data_change(variables, queuesize=1000)

    def flush(self) -> int:
        """Write the queued changes to the database; returns how many were written."""
        with self.handler.lock:
            rows, self.handler.rows = self.handler.rows, []
        self.db.executemany("INSERT INTO samples VALUES (?, ?, ?)", rows)
        self.db.commit()
        return len(rows)

    def close(self):
        """Stop the subscription, write what's left and close the file. The plant server may have
        gone already; that's fine, the queued changes are still written.

        Why the wait: the server sends changes in batches, one per publishing interval. Closing at
        once can lose the last batch (the final interval's values); two intervals is enough."""
        time.sleep(2 * self.period_s)
        try:
            self.sub.delete()
            self.client.disconnect()
        except ConnectionError:
            pass
        self.flush()
        self.db.close()


def series(db_path: str | Path, tag: str) -> tuple[np.ndarray, list]:
    """One tag's history: times (h) and values, in time order."""
    with sqlite3.connect(str(db_path)) as db:
        rows = db.execute("SELECT t_h, value FROM samples WHERE tag = ? ORDER BY t_h, rowid", (tag,)).fetchall()
    return np.array([r[0] for r in rows]), [json.loads(r[1]) for r in rows]
