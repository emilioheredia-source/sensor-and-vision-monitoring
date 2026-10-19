"""Every file the workbench reads or writes, behind plain functions.

Why this file exists
--------------------
The workbench's presenter decides what to show; it shouldn't also know where files live or how
they're read. This module is the one place that does: run files and the parameter file, the
place a setting is written and the code that reads it, saved runs and their reports, and the
session logs. The presenter calls these functions with names and gets back plain values (text,
names, line numbers), so it has no file handling of its own, and another front end could use the
same functions.

    run_file_names()                  the run files in configs/, by name
    editable_file(which, config)      (path, text) of the run file or of parameters.toml
    save_settings_as_run_file(...)    the run's settings, edits included, as a new run file
    save_edited_file(path, text)      check and save a file edited as text; returns its name
    setting_in_file(spec, s, path)    (path, text, line, editable) of where one setting is written
    setting_file_line(spec, s, path)  (path, line) of the same, to open in an outside editor
    setting_uses(name)                the lines of src/ that mention a setting's name
    save_run(result)                  save a finished run; returns (run id, its report)
    load_run(run_id)                  a saved run, read back
    saved_report(run_id)              a saved run's report (written now if it predates reports)
    saved_run_list()                  (run id, readable label) for every saved run, newest first
    log_names(), log_text(name)       the session logs, newest first, and one log's text

Paths travel as text, which is what the view shows and hands back.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import CONFIG_DIR
from .config_files import PARAMETERS_FILE, check_and_save_text, code_uses, save_run_file, setting_location
from .report import run_report

PROJECT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT / "logs"            # the workbench's session logs (workbench/logs.py writes them)


# ---- run files and the parameter file ---------------------------------------------------------------
def run_file_names() -> list[str]:
    """The run files in configs/, by name (without .toml), sorted."""
    return [path.stem for path in sorted(CONFIG_DIR.glob("*.toml"))]


def editable_file(which: str, config: str) -> tuple[str, str]:
    """(path, text) of the run file `config` (which = "run_file") or of params/parameters.toml
    (which = "parameters")."""
    path = CONFIG_DIR / f"{config}.toml" if which == "run_file" else PARAMETERS_FILE
    return str(path), path.read_text(encoding="utf-8")


def save_settings_as_run_file(spec, name: str, note: str = "", overwrite: bool = False) -> str:
    """Write the run's settings, edits included, as configs/<name>.toml (config_files.save_run_file:
    checked to load, `note` as its description). Returns the file's name. Raises if the name is
    taken (unless overwrite), not a plain word, or the file doesn't load."""
    return save_run_file(spec, name, note=note, overwrite=overwrite).name


def save_edited_file(path: str, text: str) -> str:
    """Save a TOML file edited as text, if it loads (config_files.check_and_save_text keeps the old
    one as .toml.bak). Returns the file's name. Raises with the reason if it doesn't load."""
    check_and_save_text(Path(path), text)
    return Path(path).name


# ---- where a setting is written, and the code that reads it ----------------------------------------
def setting_in_file(spec, section: str, path: tuple) -> tuple[str, str, int | None, bool]:
    """(path, text, line, editable) of the file a setting comes from. Only TOML files are editable;
    a setting whose default is in the code shows that source file, read only."""
    file, line = setting_location(spec, section, tuple(path))
    return str(file), file.read_text(encoding="utf-8"), line, file.suffix == ".toml"


def setting_file_line(spec, section: str, path: tuple) -> tuple[str, int | None]:
    """(path, line) of the file a setting comes from, to open in an outside editor."""
    file, line = setting_location(spec, section, tuple(path))
    return str(file), line


def setting_uses(name: str) -> list[tuple[str, int, str]]:
    """The lines of src/ that mention a setting's name, as (path, line, the line's text)."""
    return [(str(f), n, text) for f, n, text in code_uses(name)]


# ---- saved runs -------------------------------------------------------------------------------------
def save_run(result) -> tuple[str, str]:
    """Save a finished run (runs.save writes run.json, arrays.npz and report.md). Returns
    (run id, the report's text); the run id is the folder's name."""
    from .runs import save
    folder = save(result)
    return folder.name, (folder / "report.md").read_text(encoding="utf-8")


def load_run(run_id: str):
    """A saved run, read back as a runs.RunResult."""
    from .runs import load
    return load(_saved_folder(run_id))


def saved_report(run_id: str) -> str:
    """A saved run's report. A run saved before reports existed gets one written now, from its data."""
    report = _saved_folder(run_id) / "report.md"
    if not report.exists():
        report.write_text(run_report(load_run(run_id)), encoding="utf-8")
    return report.read_text(encoding="utf-8")


def saved_run_list() -> list[tuple[str, str]]:
    """(run id, label) for every saved run, newest first. The label is the run's date-time stamp
    and its five choices; a run.json that can't be read shows its folder name instead."""
    from .runs import saved_runs
    return [(run_json.parent.name, _saved_label(run_json)) for run_json in saved_runs()]


def _saved_folder(run_id: str) -> Path:
    """The folder of a saved run. SAVED_DIR is looked up at call time, so the tests can point it
    at a temporary folder."""
    from .runs import SAVED_DIR
    return SAVED_DIR / run_id


def _saved_label(run_json: Path) -> str:
    """A saved run's line in the list: its date-time stamp and what it ran."""
    try:
        spec = json.loads(run_json.read_text(encoding="utf-8"))["spec"]
    except (OSError, ValueError, KeyError):
        return run_json.parent.name
    from .runs import CHOICES
    stamp = run_json.parent.name.split("_")[0]
    return f"{stamp}  " + " / ".join(spec[choice] for choice in CHOICES)


# ---- session logs -----------------------------------------------------------------------------------
def log_files(log_dir: Path | None = None) -> list[Path]:
    """The workbench's log files, newest first (their names start with the date and time)."""
    log_dir = log_dir or LOG_DIR
    return sorted(log_dir.glob("workbench_*.log"), reverse=True) if log_dir.exists() else []


def log_names() -> list[str]:
    """The session logs' file names, newest first."""
    return [path.name for path in log_files()]


def log_text(name: str, max_bytes: int = 2_000_000) -> str:
    """One log's text (its last `max_bytes` if it's longer)."""
    data = (LOG_DIR / name).read_bytes()
    if len(data) > max_bytes:
        data = b"[... earlier lines cut ...]\n" + data[-max_bytes:]
    return data.decode("utf-8", errors="replace")
