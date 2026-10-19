"""Run files and the parameter file as files: save edited settings, find a setting's line, find
where the code uses it, and save a file edited as text. No toolkit here: the workbench's presenter
calls these, the view only shows what they return.

  save_run_file(spec, name)        the run's settings, edits included, as configs/<name>.toml
  setting_location(spec, s, path)  (file, line) of one setting: the run file, parameters.toml, or
                                   the MPCSettings field in src/mpc.py
  code_uses(key)                   where src/ mentions a setting's name, as (file, line, text)
  check_and_save_text(path, text)  write a TOML file edited as text, after checking it loads; the
                                   old version is kept next to it as <name>.toml.bak
"""
from __future__ import annotations

import copy
import re
import shutil
import tomllib
from pathlib import Path

from .config import CONFIG_DIR, UNUSED_SECTIONS, deep_update, load_config, read_run_file
from .params import PARAMS_DIR, load_params, params_from_dict

PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "src"
PARAMETERS_FILE = PARAMS_DIR / "parameters.toml"


# ---- writing TOML --------------------------------------------------------------------------------
def _value(v) -> str:
    """One TOML value: number, bool, string or list of them. Raises TypeError for anything else."""
    writer = next((write for kind, write in _TOML_WRITERS if isinstance(v, kind)), None)
    if writer is None:
        raise TypeError(f"can't write {type(v).__name__} to TOML: {v!r}")
    return writer(v)


def _quoted(text: str) -> str:
    """A TOML string: in double quotes, with backslashes and quotes escaped."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


# How each kind of value is written, by its type. Order matters: a bool is an int in Python, so
# bool comes first.
_TOML_WRITERS = (
    (bool, lambda v: "true" if v else "false"),
    ((list, tuple), lambda v: "[" + ", ".join(_value(x) for x in v) + "]"),
    (str, _quoted),
    ((int, float), repr),
)


def _is_plain(v) -> bool:
    """A value written as `key = value` (not a table, not an unset setting)."""
    return not isinstance(v, dict) and v is not None


def to_toml(data: dict, prefix: str = "") -> str:
    """A nested dict as TOML text: plain keys first, then one [table] per nested dict. None values
    (an unset setting) are left out.

    Example: to_toml({"a": 1, "b": {"c": 2.5}}) == 'a = 1\n\n[b]\nc = 2.5\n'
    """
    lines = [f"{k} = {_value(v)}" for k, v in data.items() if _is_plain(v)]
    out = "\n".join(lines) + ("\n" if lines else "")
    for k, v in data.items():
        if isinstance(v, dict):
            out += _table(f"{prefix}.{k}" if prefix else k, v)
    return out


def _table(name: str, values: dict) -> str:
    """One [name] table; a table with only sub-tables gets no header of its own."""
    body = to_toml(values, name)
    has_plain = any(_is_plain(x) for x in values.values())
    return f"\n[{name}]\n" + body if has_plain else body


# ---- saving the run's settings as a run file -----------------------------------------------------
def run_file_contents(spec) -> dict:
    """The run file the spec describes, edits included: its sections, plus [parameter_edits] and
    [controller_settings] carrying the file's own and the spec's edits."""
    _, raw = read_run_file(spec.config)
    own_param_edits = raw.get("parameter_edits", {})
    own_settings = raw.get("controller_settings", {})
    rc = load_config(spec.config, overrides=spec.run_file)
    body = {k: copy.deepcopy(v) for k, v in rc.cfg.items()
            if k not in UNUSED_SECTIONS and k != "controller_settings"}
    edits = deep_update(own_param_edits, spec.parameters)
    if edits:
        body["parameter_edits"] = edits
    settings = {**own_settings, **spec.controller_settings}
    if settings:
        body["controller_settings"] = settings
    return body


def save_run_file(spec, name: str, note: str = "", overwrite: bool = False) -> Path:
    """Write the spec's settings as configs/<name>.toml and check it loads. Returns the path.
    Raises FileExistsError unless overwrite, ValueError for a name that isn't a plain word."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise ValueError("a run file name is letters, digits, _ and - only")
    path = CONFIG_DIR / f"{name}.toml"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path.name} already exists")
    header = f"# Saved from the workbench, starting from run file {spec.config!r}.\n"
    description = note or f"Saved from the workbench: {spec.config} with edits."
    text = header + f'name = "{name}"\ndescription = {_value(description)}\n' + to_toml(run_file_contents(spec))
    tomllib.loads(text)                                         # it must at least parse
    if path.exists():
        shutil.copyfile(path, path.with_suffix(".toml.bak"))
    path.write_text(text, encoding="utf-8")
    try:
        load_config(name)
    except Exception:
        path.unlink()
        raise
    return path


# ---- where a setting lives -----------------------------------------------------------------------
def _toml_line(path: Path, table: tuple, key: str) -> int | None:
    """1-based line of `key = ...` under the TOML table `table` (dotted), or None."""
    sets_key = re.compile(rf"{re.escape(key)}\s*=")
    return next((n for n, s in _lines_in_table(path, ".".join(table)) if sets_key.match(s)), None)


def _lines_in_table(path: Path, want: str):
    """(1-based line number, stripped text) of each line under the table [want] of a TOML file.
    Keys before the first header are under the table "". Header lines themselves are skipped."""
    current = ""
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        header = _table_header(line)
        if header is not None:
            current = header
            continue
        if current == want:
            yield n, line.strip()


def _table_header(line: str) -> str | None:
    """The table name if the line is a TOML table header ("[flows]", "[a.b]  # note"), else None."""
    m = re.fullmatch(r"\[([^\]]+)\]\s*(#.*)?", line.strip())
    return m.group(1).strip() if m else None


def setting_location(spec, section: str, path: tuple) -> tuple[Path, int | None]:
    """The file and line a setting comes from. A run-file setting the run file doesn't set comes
    from config.DEFAULTS; a parameter from parameters.toml unless the run file edits it; a
    controller setting from the MPCSettings field (or the run file's [controller_settings])."""
    path = tuple(path)
    run_file = CONFIG_DIR / f"{spec.config}.toml"
    if section == "run_file":
        line = _toml_line(run_file, path[:-1], path[-1])
        return (run_file, line) if line else (SRC / "config.py", _first_line(SRC / "config.py", f'"{path[-1]}"'))
    if section == "parameters":
        line = _toml_line(run_file, ("parameter_edits",) + path[:-1], path[-1])
        if line:
            return run_file, line
        return PARAMETERS_FILE, _toml_line(PARAMETERS_FILE, path[:-1], path[-1])
    if section == "controller_settings":
        line = _toml_line(run_file, ("controller_settings",), path[0])
        if line:
            return run_file, line
        return SRC / "mpc.py", _first_line(SRC / "mpc.py", rf"^\s+{re.escape(path[0])}\s*:", regex=True)
    raise ValueError(f"unknown section {section!r}")


def _first_line(path: Path, needle: str, regex: bool = False) -> int | None:
    """1-based number of the first line of the file that contains `needle` (a regular expression
    if regex), or None."""
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if (re.search(needle, line) if regex else needle in line):
            return n
    return None


def code_uses(key: str, limit: int = 40) -> list[tuple[Path, int, str]]:
    """Where src/ mentions `key` (a setting's name) as a whole word, as (file, line, the line's
    text), at most `limit` of them."""
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])")
    hits = []
    for f in sorted(SRC.glob("*.py")):
        hits += _matching_lines(f, pattern)
    return hits[:limit]


def _matching_lines(path: Path, pattern) -> list[tuple[Path, int, str]]:
    """(file, line number, stripped text) of every line of the file that `pattern` matches."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [(path, n, line.strip()) for n, line in enumerate(lines, start=1) if pattern.search(line)]


# ---- a file edited as text -----------------------------------------------------------------------
def check_and_save_text(path: Path, text: str) -> None:
    """Save a run file or parameters.toml edited as text, if it's valid: it must parse, and a run
    file must load (with the parameter file as it is), a parameter file must build a parameter set.
    The previous version is kept as <name>.toml.bak. Raises with the reason if it isn't valid."""
    path = Path(path)
    data = tomllib.loads(text)                                  # TOML syntax
    if path.resolve() == PARAMETERS_FILE.resolve():
        params_from_dict(data)                                  # every field the model needs
    else:
        trial = path.with_suffix(".check.toml")
        trial.write_text(text, encoding="utf-8")
        try:
            load_config(trial, params=load_params())
        finally:
            trial.unlink()
    if path.exists():
        shutil.copyfile(path, path.with_suffix(".toml.bak"))
    path.write_text(text, encoding="utf-8")
