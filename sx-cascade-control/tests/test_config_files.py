"""Saving edited settings as a run file, finding a setting's line, and checking a file edited as text."""
import tomllib

import pytest

from src.config import CONFIG_DIR, load_config
from src.config_files import (check_and_save_text, code_uses, save_run_file, setting_location, to_toml)
from src.runs import RunSpec


def test_to_toml_round_trips_nested_tables():
    """Nested tables, lists and booleans written as TOML read back the same."""
    data = {"a": 1, "b": {"c": 2.5, "d": [1, 2], "e": {"f": "x"}}, "g": True}
    assert tomllib.loads(to_toml(data)) == data


def test_saved_run_file_carries_every_edit(tmp_path):
    """A saved run file keeps every edit (run file, parameter, controller setting) and the base file's own settings."""
    spec = RunSpec(config="ndpr_main", run_file={"flows": {"feed": 640.0}},
                   parameters={"stage_geometry": {"mixer_volume_L": 150.0}},
                   controller_settings={"w_reagent": 2.0})
    name = "test_saved_by_workbench"
    path = save_run_file(spec, name, note="a test", overwrite=True)
    try:
        rc = load_config(name)
        assert rc["flows"]["feed"] == 640.0
        assert rc.params.raw["stage_geometry"]["mixer_volume_L"] == 150.0
        assert rc.cfg["controller_settings"]["w_reagent"] == 2.0
        assert rc.cfg["controller_settings"]["acid_floor"] == 1e-3          # the run file's own, kept
        assert rc.design_purity == 0.998                                   # its own parameter edit, kept
    finally:
        path.unlink()
        path.with_suffix(".toml.bak").unlink(missing_ok=True)


def test_save_refuses_to_overwrite_and_odd_names():
    """Saving refuses to overwrite an existing run file and refuses a name with a path in it."""
    spec = RunSpec(config="ndpr_main")
    with pytest.raises(FileExistsError):
        save_run_file(spec, "ndpr_main")
    with pytest.raises(ValueError):
        save_run_file(spec, "../escape")


def test_setting_location_finds_the_line():
    """A setting is found on the right line of the file that writes it (run file, parameter file or mpc.py)."""
    spec = RunSpec(config="ndpr_main")
    f, line = setting_location(spec, "run_file", ("flows", "feed"))
    assert f.name == "ndpr_main.toml" and f.read_text(encoding="utf-8").splitlines()[line - 1].startswith("feed")
    f, line = setting_location(spec, "parameters", ("equilibrium", "K", "Pr"))
    assert f.name == "parameters.toml" and f.read_text(encoding="utf-8").splitlines()[line - 1].startswith("Pr")
    f, line = setting_location(spec, "parameters", ("stage_geometry", "mixer_volume_L"))
    assert f.name == "ndpr_main.toml"                                       # edited by the run file
    f, line = setting_location(spec, "controller_settings", ("w_reagent",))
    assert f.name == "mpc.py" and "w_reagent" in f.read_text(encoding="utf-8").splitlines()[line - 1]


def test_code_uses_finds_the_name():
    """The code that reads a parameter is found by its name."""
    assert any(p.name == "circuit.py" for p, _, _ in code_uses("mixer_volume_L"))


def test_text_edit_is_checked_before_saving(tmp_path):
    """A run file edited as text is checked before it's saved, and a bad edit is not saved."""
    target = CONFIG_DIR / "test_text_edit.toml"
    good = (CONFIG_DIR / "ndpr_main.toml").read_text(encoding="utf-8").replace('name = "ndpr_main"',
                                                                               'name = "test_text_edit"')
    try:
        check_and_save_text(target, good)
        assert load_config("test_text_edit")["circuit"]["n_ext"] == 8
        with pytest.raises(Exception):
            check_and_save_text(target, good.replace("[flows]", "[flows]\nnot_a_setting = 1"))
        assert "not_a_setting" not in target.read_text(encoding="utf-8")      # the bad edit wasn't saved
    finally:
        target.unlink(missing_ok=True)
        target.with_suffix(".toml.bak").unlink(missing_ok=True)


def test_a_run_file_value_of_the_wrong_type_stops_the_load():
    """A run file value of the wrong type stops the load; a whole number where a decimal is expected doesn't."""
    with pytest.raises(TypeError, match="targets.purity"):
        load_config("ndpr_main", overrides={"targets": {"purity": "0.99"}})
    with pytest.raises(TypeError, match="circuit.n_ext"):
        load_config("ndpr_main", overrides={"circuit": {"n_ext": 8.5}})
    load_config("ndpr_main", overrides={"flows": {"feed": 600}})            # a whole number for a decimal is fine


def test_every_run_file_says_what_it_is_and_a_saved_one_carries_the_note():
    """Every run file has a description, and a saved one carries the note it was saved with."""
    from pathlib import Path
    from src.config import run_file_description
    assert all(run_file_description(p.stem) for p in sorted(CONFIG_DIR.glob("*.toml")))
    path = save_run_file(RunSpec(config="ndpr_main"), "test_described", note="A test run file.", overwrite=True)
    try:
        assert run_file_description("test_described") == "A test run file."
        load_config("test_described")                        # the description doesn't trip the key check
    finally:
        path.unlink()
        path.with_suffix(".toml.bak").unlink(missing_ok=True)
