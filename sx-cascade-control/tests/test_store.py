"""The workbench's file access (src/store.py): run files, where a setting is written, saved runs and
their reports, and the session logs, all as plain values."""

import pytest

import src.runs as runs
import src.store as store
from src.runs import RunSpec, execute


def test_run_files_and_the_parameter_file_as_text():
    """Run files and the parameter file are listed and read as text."""
    assert "ndpr_main" in store.run_file_names()
    path, text = store.editable_file("run_file", "ndpr_main")
    assert path.endswith("ndpr_main.toml") and 'name = "ndpr_main"' in text
    path, text = store.editable_file("parameters", "ndpr_main")
    assert path.endswith("parameters.toml") and "[equilibrium.K]" in text


def test_where_a_setting_is_written_and_who_reads_it():
    """A setting's file and line are found, whether it can be edited there, and the code that reads it."""
    spec = RunSpec(config="ndpr_main")
    path, text, line, editable = store.setting_in_file(spec, "run_file", ("flows", "feed"))
    assert editable and text.splitlines()[line - 1].startswith("feed")
    path, _, line, editable = store.setting_in_file(spec, "controller_settings", ("w_reagent",))
    assert path.endswith("mpc.py") and not editable
    assert store.setting_file_line(spec, "controller_settings", ("w_reagent",)) == (path, line)
    assert any(p.endswith("circuit.py") for p, _, _ in store.setting_uses("mixer_volume_L"))


def test_a_run_saved_listed_read_back_and_its_report(tmp_path, monkeypatch):
    """Save a short run, find it in the list, read it back; a run whose report is missing (saved
    before reports existed) gets one written when asked for."""
    monkeypatch.setattr(runs, "SAVED_DIR", tmp_path)
    run_id, report = store.save_run(execute(RunSpec(config="ndpr_small", upset="feed_step", hours=1.0)))
    assert report.startswith("# Run report")
    ids = [rid for rid, _ in store.saved_run_list()]
    assert ids == [run_id]
    assert store.load_run(run_id).spec.config == "ndpr_small"
    (tmp_path / run_id / "report.md").unlink()
    assert store.saved_report(run_id).startswith("# Run report")
    assert (tmp_path / run_id / "report.md").exists()


def test_session_logs_newest_first(tmp_path, monkeypatch):
    """Session logs are listed newest first and read back as text."""
    monkeypatch.setattr(store, "LOG_DIR", tmp_path)
    for stamp in ("20261018_100000", "20261018_120000"):
        (tmp_path / f"workbench_{stamp}.log").write_text(f"line from {stamp}\n", encoding="utf-8")
    assert store.log_names() == ["workbench_20261018_120000.log", "workbench_20261018_100000.log"]
    assert "120000" in store.log_text("workbench_20261018_120000.log")
