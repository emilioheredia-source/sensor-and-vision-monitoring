"""The workbench's interaction logic, with no GUI toolkit in it.

What this file is
-----------------
The workbench is built in three layers (the Passive-View MVP pattern, ARCHITECTURE.md decision 24):

    src/            the core. It builds the plant, runs the controller, scores and saves a run.
                    It knows nothing about any window.
    presenter.py    this file. It decides what happens when the user does something, and what
                    the screen should show. It knows nothing about Qt.
    view.py,        the PyQt shell. It draws what it's told and passes clicks back here.
    panels.py

The presenter owns the run being set up, a `RunSpec`: run file, plant, controller, upset, sensor
set, run length, and any edits to the settings. When the view reports that the user did something
("chose this plant", "edited that setting", "pressed Run"), the presenter updates the spec, asks
src/ for what's needed, and tells the view what to show.

Why the presenter has no Qt in it
---------------------------------
Everything the presenter sends to the view is a plain Python value: strings, lists, dicts and
numpy arrays. It talks to the view only through `WorkbenchView`, an interface (a typing.Protocol)
listed below. Anything that has those methods can be the front end:

    - the PyQt window in view.py, which is what you see when you run `python -m workbench`;
    - a fake view in tests/test_workbench.py, which just records the calls, so the presenter is
      tested with no display at all;
    - a web front end later, without touching this file or anything in src/.

The presenter also does no physics. Every number about the plant comes from src/; the presenter
only arranges it for display (picks the columns, turns the handles into fractions of their start
value, formats the scorecard).

How a run works, in short
-------------------------
`run()` hands the view a generator (`_produce`) and asks it to run that in a background thread.
The generator yields one item per control interval. The view passes each item back to `_on_item`
on the GUI thread, which appends it to the live series and tells the view to redraw. When the
generator ends, `_on_done` saves the run and shows the scorecard. Details are in the "Running"
section below.

A note on the product: in the main run files the cut is "LaCe", so the product is NdPr, taken as
one product. The "Pr" cut (product Nd) is only there to check the model against the published
Pr|Nd data.
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Callable, Iterator, Protocol

import numpy as np

from src.config import build_circuit, load_config, nominal_inputs, run_file_description
from src.registry import CONTROLLERS, PLANTS, SENSOR_SETS, options
from src import store
from src.runs import CHOICES, RunSpec, circuit_layout, editable_values, start
from src.scenarios import UPSET_TIME_H, catalogue

from .logs import log


# ==================================================================================================
# Fixed lists the presenter works from
# ==================================================================================================
# The five choices on the Run panel, in the order they're shown (runs.CHOICES). Each is also the
# name of a field of RunSpec, so `getattr(spec, kind)` is the current choice.
KINDS = CHOICES

# The three sections of settings the user can edit. Each is also a field of RunSpec holding the
# edits as a nested dict in the shape of what it edits.
SETTING_SECTIONS = ("run_file", "parameters", "controller_settings")

# The scorecard, one row per entry: (key in the run's scorecard dict, row label, unit).
SCORE_ROWS = (("off_spec_mol", "Product off spec", "mol"),
              ("lost_excess_mol", "Metal lost beyond the starting point", "mol"),
              ("interface_excursion_m_h", "Interfaces outside their window", "m·h"),
              ("reagent_mol", "Reagent used", "mol"),
              ("time_to_settle_h", "Time for purity to settle", "h"),
              ("solver_failures", "Controller solves that failed", ""))

# The four handles the controller can move, as (name in the step's handles, readable name).
# The plots show each one as a fraction of its starting value.
HANDLE_NAMES = (("Q_org", "organic flow"), ("Q_scrub", "scrub flow"), ("h_scrub", "scrub acid"),
                ("saponified_fraction", "saponification"))

# What the plant reports each interval that goes straight into the live series.
PLANT_OUTPUT_KEYS = ("purity", "recovery", "lost_rate", "reagent_rate", "product_rate", "lost_pct_of_feed",
                     "reagent_per_product")

# What the controller records each interval about its own view of purity: what it was told
# (after the sensors), what it plans for the end of its horizon, and the bias it carries.
CONTROLLER_RECORD_KEYS = ("purity_seen", "purity_plan", "bias_purity")


# ==================================================================================================
# The view interface: everything the presenter may ask of a front end
# ==================================================================================================
# This is the whole contract between the presenter and whatever draws the screen. The presenter
# calls these methods and nothing else on the view. A front end doesn't need to inherit from this
# class; it just needs methods with these names (that's what a Protocol means).
class WorkbenchView(Protocol):
    """What the presenter needs from a front end. All arguments are plain Python values."""

    def show_options(self, kind: str, options: list[tuple[str, str]], selected: str):
        """Fill one of the five choice lists. `kind` is one of KINDS; `options` is a list of
        (key, title) pairs; `selected` is the key to show as chosen."""

    def show_explanation(self, kind: str, text: str):
        """Show the short explanation of the current choice for `kind`."""

    def show_hours(self, hours: float):
        """Show the run length in hours."""

    def show_settings(self, values: dict, edited: list[tuple]):
        """Show every setting. `values` maps each section name to nested dicts of plain values;
        `edited` lists the paths that differ from the files, each as (section, key, key, ...)."""

    def show_layout(self, layout: dict):
        """Draw the circuit: stage counts per section, feed and product stages, stream names and
        where each instrument sits. See `_layout` below and `src.runs.circuit_layout`."""

    def begin_plots(self, meta: dict):
        """Clear the plots for a new run and draw its reference lines (spec, targets, interface
        window, upset time). See `_plot_meta` for the keys."""

    def update_plots(self, series: dict):
        """Redraw the time plots from the live series: a dict of 1-D numpy arrays keyed by name,
        all the same length, with "t" (hours) among them."""

    def show_profile(self, t: float, profile: dict):
        """Draw every stage at time `t`: the aqueous concentrations and the interface heights."""

    def show_scorecard(self, columns: list[str], rows: list[tuple[str, list[str]]]):
        """Fill the scorecard table. `columns` are the run labels; each row is
        (row label, one formatted value per column)."""

    def show_saved_runs(self, runs: list[tuple[str, str]]):
        """List the saved runs as (run id, readable label) pairs. The run id is the folder name."""

    def show_comparison(self, runs: list[dict]):
        """Overlay saved runs on the plots. Each is {"label": str, "series": dict like
        update_plots}. An empty list clears the overlay."""

    def set_running(self, running: bool):
        """Enable or disable the controls that only make sense while a run is or isn't going."""

    def set_status(self, text: str):
        """Show a one-line status message."""

    def show_error(self, text: str):
        """Tell the user something went wrong. `text` may be a full traceback."""

    def show_text(self, path: str, text: str, line: int | None, editable: bool):
        """Show a file's text in the text editor, at `line` (1-based) if given. Only TOML files
        are `editable`; source files are shown read-only."""

    def show_code_uses(self, name: str, hits: list[tuple[str, int, str]]):
        """List where the code mentions a setting's `name`, as (file, line, the line's text)."""

    def open_in_code(self, path: str, line: int | None):
        """Open a file in an outside editor (VS Code) at a line."""

    def show_report(self, text: str):
        """Show a run's report (Markdown text, see src/report.py)."""

    def show_progress(self, fraction: float | None, text: str):
        """How far the run is: a fraction from 0 to 1, or None while that isn't known yet."""

    def show_log_list(self, logs: list[tuple[str, str]]):
        """List the log files as (name, label) pairs, newest first."""

    def show_log_text(self, name: str, text: str):
        """Show one log file's text."""

    def run_in_background(self, produce: Callable[[], Iterator], on_item: Callable, on_done: Callable,
                          on_error: Callable):
        """Call `produce()` off the GUI thread and walk the iterator it returns.

        Each item must reach `on_item(item)` on the GUI thread, in order. When the iterator ends,
        call `on_done()`; if it raises, call `on_error(text)` with the error text. The PyQt view
        does this with a QThread; the fake view in the tests just runs it inline.
        """


class _LoggedView:
    """The view, with every status line and error also written to the session log (logs.py).
    Everything else passes straight through."""

    def __init__(self, view):
        self._view = view

    def __getattr__(self, name):
        return getattr(self._view, name)

    def set_status(self, text: str):
        log.info(text)
        self._view.set_status(text)

    def show_error(self, text: str):
        log.error(text)
        self._view.show_error(text)


# ==================================================================================================
# The presenter
# ==================================================================================================
class WorkbenchPresenter:
    """All the workbench's interaction logic. Holds the run being set up and the run going on.

    Args:
        view: anything that implements WorkbenchView.
        spec: the run to start with. Defaults to RunSpec() (see src/runs.py for its defaults).

    Usage (this is what workbench/__main__.py does):

        view = QtWorkbenchView()
        presenter = WorkbenchPresenter(view, RunSpec(config="ndpr_small", hours=40.0))
        view.bind(presenter)          # the view connects its buttons to the presenter's methods
        presenter.start()             # the presenter fills the view for the first time
    """

    def __init__(self, view: WorkbenchView, spec: RunSpec | None = None):
        """Keep the view and the run to start with. Nothing is shown until start()."""
        self.view = _LoggedView(view)          # every status line and error also goes to the log
        self.spec = spec or RunSpec()

        # The run going on, if any. It's a src.runs.RunSession; stop() asks it to end early.
        self._session = None

        # The live run so far, one list per series name, one entry per interval. Lists are cheap
        # to append to; they're turned into numpy arrays each time the view is told to redraw.
        self._live: dict[str, list] = {}

        # The finished run (a src.runs.RunResult), once the background work hands it back.
        self._result = None

        # The saved runs the user chose to overlay, each {"label", "series", "result"}.
        self._compared: list[dict] = []

    # ----------------------------------------------------------------------------------------------
    # Setting up a run
    # ----------------------------------------------------------------------------------------------
    # The user picks a run file, a plant, a controller, an upset and a sensor set, sets the run
    # length, and may edit any setting. Each of these replaces `self.spec` with an updated copy
    # (RunSpec is a dataclass; `replace` makes the copy), then tells the view what changed.

    def start(self):
        """Fill the view for the first time: every choice list, the run length, the settings,
        the circuit drawing and the saved runs."""
        for kind in KINDS:
            self._show_choice(kind)
        self.view.show_hours(self.spec.hours)
        self._refresh_settings()
        self.view.show_layout(_layout(self.spec))
        self._refresh_saved()
        self.view.set_running(False)
        self.view.set_status("Ready. Choose a plant, a controller, an upset and a sensor set, then Run.")

    def choose(self, kind: str, key: str):
        """The user picked `key` in the choice list for `kind` (one of KINDS)."""
        changing_run_file = kind == "config" and key != self.spec.config
        if changing_run_file:
            # Edits to the run file were made against the old file, so they're dropped.
            self.spec = replace(self.spec, config=key, run_file={})
        else:
            self.spec = replace(self.spec, **{kind: key})
        self._show_choice(kind)

        # The upsets on offer depend on the run file, so a new run file refreshes that list too.
        if kind in ("config", "upset"):
            self._show_choice("upset")
        self._refresh_settings()
        self.view.show_layout(_layout(self.spec))

    def set_hours(self, hours: float):
        """The user changed the run length."""
        self.spec = replace(self.spec, hours=float(hours))

    # Editing a setting. The view sends the section ("run_file", "parameters" or
    # "controller_settings"), the path of keys down to the value, and the new value. The edit is
    # stored in the spec as a nested dict, so the spec only holds what the user changed. If the
    # model can't take the new value (loading the edited run raises), the edit is undone and the
    # user is told why.

    def edit_setting(self, section: str, path: tuple, value):
        """Store one edited setting in the spec and redraw what depends on it.

        Args:
            section: "run_file", "parameters" or "controller_settings".
            path: the keys from the section down to the value, e.g. ("flows", "feed").
            value: the new value. A value the view shows as text (a list, or an empty setting)
                comes back as text and is read as JSON where it can be (see `_parse`).

        Errors: an edit the model can't take is undone and reported with view.show_error;
        nothing is raised.

        Example:
            presenter.edit_setting("run_file", ("flows", "feed"), 640.0)
        """
        edits = copy.deepcopy(getattr(self.spec, section))
        parent = _parent_node(edits, path)
        try:
            new_value = _parse(value, self._current_value(section, path))
        except ValueError as exc:                     # the wrong type: refuse it, show the value as it was
            self._refresh_settings()
            self.view.show_error(f"{'.'.join(path)}: {exc}")
            return
        parent[path[-1]] = new_value
        self.spec = replace(self.spec, **{section: edits})
        try:
            # Both of these load the run from the edited spec, so a bad edit raises here.
            self._refresh_settings()
            self.view.show_layout(_layout(self.spec))
            dotted_path = ".".join(path)
            self.view.set_status(f"{section}.{dotted_path} = {new_value!r} (takes effect on the next run)")
        except Exception as exc:                      # an edit the model can't take: undo it and say why
            self.reset_settings(section, path)
            self.view.show_error(f"{'.'.join(path)}: {exc}")

    def reset_settings(self, section: str | None = None, path: tuple | None = None):
        """Forget one edit (give its section and path), or all of them (give nothing)."""
        if section is None:
            self.spec = replace(self.spec, run_file={}, parameters={}, controller_settings={})
        else:
            edits = copy.deepcopy(getattr(self.spec, section))
            _remove(edits, path)
            self.spec = replace(self.spec, **{section: edits})
        self._refresh_settings()
        self.view.show_layout(_layout(self.spec))

    # ----------------------------------------------------------------------------------------------
    # The settings as files
    # ----------------------------------------------------------------------------------------------
    # The edits above live only in the spec. These methods write them to a new run file, open the
    # run file or parameters.toml as text, and find where a setting is written and where the code
    # reads it. The file work is in src/config_files.py; here it's only arranged for the view.

    def save_settings_as(self, name: str, note: str = "", overwrite: bool = False):
        """Write the current settings, edits included, as configs/<name>.toml and switch to it.
        The edits are now in that file, so the spec's edits are cleared."""
        try:
            file_name = store.save_settings_as_run_file(self.spec, name, note=note, overwrite=overwrite)
        except Exception as exc:                      # a taken name, an odd name, or a file that won't load
            self.view.show_error(f"Not saved: {exc}")
            return
        self.spec = replace(self.spec, run_file={}, parameters={}, controller_settings={})
        self.choose("config", name)
        self.view.set_status(f"Saved {file_name}; it's now the run file.")

    def open_text(self, which: str):
        """Open the run file ("run_file") or the parameter file ("parameters") in the text editor."""
        path, text = store.editable_file(which, self.spec.config)
        self.view.show_text(path, text, None, True)

    def save_text(self, path: str, text: str):
        """Save a TOML file edited as text, if it loads. An invalid file isn't saved; the user is
        told why and the text stays in the editor to fix."""
        try:
            name = store.save_edited_file(path, text)
        except Exception as exc:
            self.view.show_error(f"Not saved, the file doesn't load: {exc}")
            return
        self._show_choice("config")                   # the explanations come from the files
        self._show_choice("upset")
        self._refresh_settings()
        self.view.show_layout(_layout(self.spec))
        self.view.set_status(f"Saved {name} (the old version is kept as {name}.bak).")

    def locate_setting(self, section: str, path: tuple, how: str):
        """Find one setting. `how` is "show" (its line, in the text editor), "uses" (the lines in
        src/ that mention its name) or "code" (its line, opened in VS Code)."""
        if how == "show":
            self.view.show_text(*store.setting_in_file(self.spec, section, path))
        elif how == "uses":
            self.view.show_code_uses(str(path[-1]), store.setting_uses(str(path[-1])))
        else:
            self.view.open_in_code(*store.setting_file_line(self.spec, section, path))

    def open_in_code(self, path: str, line: int | None):
        """Open a file in VS Code at a line."""
        self.view.open_in_code(path, line)

    # ----------------------------------------------------------------------------------------------
    # Running
    # ----------------------------------------------------------------------------------------------
    # A run takes from seconds to minutes, so it can't run on the GUI thread or the window would
    # freeze. The presenter doesn't start threads itself (that's toolkit business). Instead it
    # hands the view a generator function, `_produce`, and three callbacks:
    #
    #     background thread                         GUI thread
    #     -----------------                         ----------
    #     _produce() yields ("ready", first step) -> _on_item: set up the plots, draw the start
    #                yields ("step", step)        -> _on_item: append, redraw   (once per interval)
    #                yields ("result", result)    -> _on_item: keep the result
    #     generator ends                          -> _on_done: save, score, list it
    #     generator raises                        -> _on_error: show the error
    #
    # Everything slow (commissioning the operating point, every controller solve, every plant
    # step) happens inside the generator, on the background thread. The callbacks only touch
    # small lists and tell the view to draw.

    def run(self):
        """Start a run of the current spec in the background."""
        # A copy, so edits the user makes while the run goes on can't reach the running spec.
        spec_for_this_run = copy.deepcopy(self.spec)
        log.info("run started: %r", spec_for_this_run)
        self._live, self._result = {}, None
        self.view.set_running(True)
        self.view.show_progress(None, "preparing the operating point")
        self.view.set_status("Preparing the operating point: commissioning and trim. The first time for a "
                             "given plant and settings this can take a few minutes; after that it's cached.")
        self.view.run_in_background(lambda: self._produce(spec_for_this_run), self._on_item, self._on_done,
                                    self._on_error)

    def stop(self):
        """Ask the run going on, if any, to end after the interval it's working on."""
        if self._session is not None:
            self._session.stop()
            self.view.set_status("Stopping after this interval...")

    def _produce(self, spec: RunSpec) -> Iterator:
        """The background work. Yields (kind, payload) pairs, kind being "ready", "step" or
        "result". Runs on the background thread: don't call the view from here."""
        self._session = start(spec)                   # builds the operating point: the slow part
        yield "ready", self._session.start_point()
        for step in self._session.steps():           # one src.runs.LiveStep per control interval
            yield "step", step
        yield "result", self._session.result()       # what ran, even if it was stopped early

    def _on_item(self, item):
        """One item from `_produce`, now on the GUI thread."""
        kind, payload = item
        if kind == "ready":
            self._begin_live_plots()
        if kind in ("ready", "step"):
            self._show_step(payload)
        if kind == "result":
            self._result = payload

    def _begin_live_plots(self):
        """The operating point is ready: clear the plots and draw this run's reference lines."""
        self.view.begin_plots(self._plot_meta())
        self.view.set_status(f"Running: {self.spec.label()}")

    def _show_step(self, step):
        """Add one interval to the live series and redraw the plots and the stage profile."""
        self._append(step)
        self.view.update_plots(self._series())
        self.view.show_profile(step.t, step.profile)
        self.view.show_progress(step.t / self.spec.hours, f"{step.t:.1f} of {self.spec.hours:g} h")

    def _on_done(self):
        """The background work finished. Save the run, unless it stopped before the first interval."""
        self.view.set_running(False)
        self._session = None
        finished = self._result is not None
        self.view.show_progress(1.0 if finished else 0.0, "finished" if finished else "stopped")
        if not finished or len(self._result.t) < 2:
            self.view.set_status("Stopped before the first interval; nothing to keep.")
            return
        run_id, report = store.save_run(self._result)
        log.info("run finished, saved as %s; scorecard %s", run_id, self._result.scorecard)
        self.view.show_report(report)
        self._refresh_saved()
        self._show_scorecard()
        self.view.set_status(f"Done: {self._result.spec.label()}. Saved as {run_id}.")

    def _on_error(self, text: str):
        """The background work raised. `text` is the error, usually a traceback."""
        self.view.set_running(False)
        self._session = None
        self.view.show_progress(0.0, "failed: see the error, and File > Show the logs")
        self.view.show_error(text)

    # ----------------------------------------------------------------------------------------------
    # Comparing with saved runs
    # ----------------------------------------------------------------------------------------------
    # Every finished run is saved in its own folder (src.runs.save). The user can tick any of them
    # and overlay them on the plots; they also get a column each on the scorecard.

    def compare(self, run_ids: list[str]):
        """Overlay saved runs on the plots and add them to the scorecard.

        Args:
            run_ids: folder names of saved runs, as listed by show_saved_runs. An empty list
                clears the overlay.
        """
        self._compared = []
        for run_id in run_ids:
            saved_run = store.load_run(run_id)
            self._compared.append({"label": _short_label(saved_run.spec),
                                   "series": self._series_of(saved_run), "result": saved_run})
        overlays = [{"label": c["label"], "series": c["series"]} for c in self._compared]
        self.view.show_comparison(overlays)
        self._show_scorecard()

    def show_logs(self):
        """List the session logs (newest first) in the Logs panel and show the newest."""
        names = store.log_names()
        self.view.show_log_list([(name, name) for name in names])
        if names:
            self.open_log(names[0])

    def open_log(self, name: str):
        """Show one log file's text."""
        self.view.show_log_text(name, store.log_text(name))

    def show_saved_report(self, run_id: str):
        """Show a saved run's report; one saved before reports existed gets one written now."""
        self.view.show_report(store.saved_report(run_id))

    def refresh_saved(self):
        """List the saved runs again (the user pressed Refresh)."""
        self._refresh_saved()

    # ----------------------------------------------------------------------------------------------
    # Arranging the choices and the settings for display
    # ----------------------------------------------------------------------------------------------

    def _show_choice(self, kind: str):
        """Send the view the options for `kind`, the current choice and its explanation.

        If the current choice isn't on offer any more (for instance, a run file that has no such
        upset), the first option is chosen instead and the spec is updated to match.
        """
        choices = self._options(kind)                 # (key, title, explanation) triples
        selected = getattr(self.spec, kind)
        if selected not in {key for key, _, _ in choices}:
            selected = choices[0][0]
            self.spec = replace(self.spec, **{kind: selected})
        self.view.show_options(kind, [(key, title) for key, title, _ in choices], selected)
        explanation = next(text for key, _, text in choices if key == selected)
        self.view.show_explanation(kind, explanation)

    def _options(self, kind: str) -> list[tuple[str, str, str]]:
        """The options for one choice list, as (key, title, explanation) triples."""
        if kind == "config":
            return _run_file_options()
        if kind == "upset":
            return self._upset_options()
        registry = {"plant": PLANTS, "controller": CONTROLLERS, "sensors": SENSOR_SETS}[kind]
        return options(registry)

    def _upset_options(self) -> list[tuple[str, str, str]]:
        """The upsets the current run file offers. They're built from its nominal inputs, so
        they depend on which run file is chosen and on any edits to it."""
        run_config = load_config(self.spec.config, overrides=self.spec.run_file)
        nominal = nominal_inputs(run_config, build_circuit(run_config))
        upsets = catalogue(run_config, nominal).values()
        return [(u.key, u.title, f"{u.explanation} Source: {u.source}. Starts at {UPSET_TIME_H:g} h.")
                for u in upsets]

    def _refresh_settings(self):
        """Send the view every setting of the run (after edits) and the paths that were edited."""
        edited_paths = [(section,) + path for section in SETTING_SECTIONS
                        for path in _paths(getattr(self.spec, section))]
        self.view.show_settings(editable_values(self.spec), edited_paths)

    def _current_value(self, section: str, path: tuple):
        """The value a setting has now (after any edits), found by walking `path`."""
        node = editable_values(self.spec)[section]
        for key in path:
            node = node[key]
        return node

    def _refresh_saved(self):
        """Send the view the list of saved runs, newest first."""
        self.view.show_saved_runs(store.saved_run_list())

    # ----------------------------------------------------------------------------------------------
    # Arranging a run for the plots
    # ----------------------------------------------------------------------------------------------

    def _plot_meta(self) -> dict:
        """What the plots need before the first point: reference levels, the upset time, the
        title, and which components to draw.

        Returns a dict with:
            spec, purity_target, recovery_target: reference levels for the purity and recovery plots;
            window: (low, high) settler interface heights the interfaces should stay between, in m;
            upset_time: when the upset starts, in h, or None for no upset;
            title, product: the run's label and the product's name (e.g. "NdPr");
            components: the components to draw on the stage profile (those still in the feed);
            all_components: every component, in the column order of the profile arrays.
        """
        run_config = load_config(self.spec.config, overrides=self.spec.run_file)
        geometry = run_config.params.raw["stage_geometry"]
        low, high = (fraction * geometry["settler_height_m"]
                     for fraction in geometry["interface_window_fraction"])
        purity_spec, recovery_target = run_config.targets()
        removed_upstream = run_config["feed"]["removed_upstream"]
        return {"spec": purity_spec, "purity_target": run_config.design_purity,
                "recovery_target": recovery_target, "window": (low, high),
                "upset_time": UPSET_TIME_H if self.spec.upset != "none" else None,
                "title": self.spec.label(), "product": _product_name(run_config.cut),
                "components": [c for c in run_config.params.names if c not in removed_upstream],
                "all_components": list(run_config.params.names)}

    # The live series. Each interval adds one entry to every list in self._live. The order the
    # helpers run in below is the order the series appear in the dict sent to the view.

    def _append(self, step):
        """Add one interval (a src.runs.LiveStep) to the live series."""
        self._append_plant_outputs(step.outputs)
        self._live_column("t").append(step.t)
        self._append_interface_spread(step.outputs["interfaces"])
        self._append_handles(step.handles)
        self._append_controller_record(step.info)

    def _live_column(self, key: str) -> list:
        """The list for one live series, created empty the first time it's asked for."""
        return self._live.setdefault(key, [])

    def _append_plant_outputs(self, outputs: dict):
        """Purity, recovery and the rates, as the plant reports them."""
        for key in PLANT_OUTPUT_KEYS:
            self._live_column(key).append(outputs[key])

    def _append_interface_spread(self, interface_heights):
        """One settler interface per stage, kept as lowest, mean and highest. The interface
        plot draws the band between lowest and highest, and the mean as a line."""
        heights = np.asarray(interface_heights, float)
        self._live_column("interfaces_min").append(float(heights.min()))
        self._live_column("interfaces_mean").append(float(heights.mean()))
        self._live_column("interfaces_max").append(float(heights.max()))

    def _append_handles(self, handles: dict):
        """What the controller commanded for each handle. A controller doesn't have to move every
        handle; one it leaves alone keeps its last value."""
        for name, _ in HANDLE_NAMES:
            column = self._live_column(f"handle_{name}")
            column.append(_handle_value(handles, name, column))

    def _append_controller_record(self, info: dict):
        """The controller's own record of the interval. The start point has an empty record,
        so it gets NaN (not drawn) and doesn't count as a failed solve."""
        for key in CONTROLLER_RECORD_KEYS:
            self._live_column(key).append(float(info.get(key, np.nan)))
        self._live_column("solver_failed").append(bool(info) and not info.get("qp_ok", True))

    def _series(self) -> dict:
        """The live run as numpy arrays, with each handle as a fraction of its starting value."""
        series = {key: np.asarray(values, float) for key, values in self._live.items()}
        for name, _ in HANDLE_NAMES:
            key = f"handle_{name}"
            if key in series:
                series[key] = _relative_to_first(series[key])
        return series

    def _series_of(self, saved_run) -> dict:
        """A saved run (a src.runs.RunResult) in the same shape as the live series, so the plots
        can draw both the same way."""
        interface_heights = saved_run.series["interfaces"]    # one row per interval, one column per stage
        series = {"t": saved_run.t, "purity": saved_run.series["purity"],
                  "recovery": saved_run.series["recovery"], "lost_rate": saved_run.series["lost_rate"],
                  "reagent_rate": saved_run.series["reagent_rate"],
                  "product_rate": saved_run.series["product_rate"],
                  "interfaces_min": interface_heights.min(axis=1),
                  "interfaces_mean": interface_heights.mean(axis=1),
                  "interfaces_max": interface_heights.max(axis=1)}

        # A saved run keeps the handles from the first interval on; the starting values are in its
        # metadata. Put the start in front so the series lines up with the times, then divide by it.
        start_values = saved_run.meta.get("operating_point", {})
        for name, _ in HANDLE_NAMES:
            if name in saved_run.handles and start_values.get(name):
                values = np.concatenate([[start_values[name]], saved_run.handles[name]])
                series[f"handle_{name}"] = values / start_values[name]
        return series

    def _show_scorecard(self):
        """Send the view the scorecard: one column for the run just finished (if any), then one
        for each overlaid saved run."""
        columns, scorecards = [], []
        if self._result is not None:
            columns.append(f"this run: {_short_label(self._result.spec)}")
            scorecards.append(self._result.scorecard)
        for compared in self._compared:
            columns.append(compared["label"])
            scorecards.append(compared["result"].scorecard)
        rows = [(_row_label(label, unit), [_fmt(card.get(key)) for card in scorecards])
                for key, label, unit in SCORE_ROWS]
        self.view.show_scorecard(columns, rows)


# ==================================================================================================
# Small helpers (plain functions, no state)
# ==================================================================================================

# ---- edited settings, as nested dicts -----------------------------------------------------------
def _parse(value, current):
    """A value from the view, read in the type of the value it replaces.

    The view shows lists, dicts and empty settings as text. When such a setting comes back, the
    text is read as JSON ('["Q_org", "Q_scrub"]' becomes a list); empty text means None; text
    that isn't JSON is kept as text. A number or a yes/no setting comes back as that type, or the
    edit is refused (ValueError): text such as "0.99" is never stored where a number was.
    Why the conversion: a box built as text for one run file can be handed a number by the next,
    and give it back as text; that text reached the controller as its purity spec (Oct 18).
    """
    if isinstance(current, bool):                       # before numbers: a bool is an int in Python
        return _as_bool(value)
    if isinstance(current, (int, float)):
        return _as_number(value, whole=isinstance(current, int))
    if not _shown_as_text(value, current):
        return value
    return _from_text(value)


def _as_bool(value) -> bool:
    """A yes/no setting: True/False, or the text "true"/"false". Raises ValueError otherwise."""
    if not isinstance(value, str):
        return bool(value)
    text = value.strip().lower()
    if text not in ("true", "false"):
        raise ValueError(f"expected true or false, got {value!r}")
    return text == "true"


def _as_number(value, whole: bool):
    """A number setting, as a float (or an int when `whole`). Raises ValueError for text that
    isn't a number, or a fraction where a whole number is needed."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"expected a number, got {value!r}") from None
    if not whole:
        return number
    if number != int(number):
        raise ValueError(f"expected a whole number, got {value!r}")
    return int(number)


def _from_text(value: str):
    """A list, dict or empty setting shown as text: JSON where it parses, None when empty, else the text."""
    text = value.strip()
    if text == "":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _shown_as_text(value, current) -> bool:
    """True when the view sent text for a setting it shows as text: a list, a dict, or None."""
    return isinstance(value, str) and (current is None or isinstance(current, (list, dict)))


def _parent_node(nested: dict, path: tuple) -> dict:
    """The dict that holds the last key of `path`, creating empty dicts on the way down."""
    node = nested
    for key in path[:-1]:
        node = node.setdefault(key, {})
    return node


def _paths(nested: dict, prefix: tuple = ()) -> list[tuple]:
    """Every path down to a value in a nested dict, e.g. {"a": {"b": 1}} gives [("a", "b")]."""
    found = []
    for key, value in nested.items():
        found += _paths(value, prefix + (key,)) if isinstance(value, dict) else [prefix + (key,)]
    return found


def _remove(nested: dict, path: tuple):
    """Delete the value at `path`, if it's there. Leaves the dicts above it in place."""
    node = nested
    for key in path[:-1]:
        node = node.get(key, {})
    node.pop(path[-1], None)


# ---- live series ----------------------------------------------------------------------------------
def _handle_value(handles: dict, name: str, column: list) -> float:
    """This interval's value of one handle: the commanded value if there is one, otherwise the
    last value in its column (NaN if the column is still empty)."""
    if name in handles:
        return float(handles[name])
    return column[-1] if column else np.nan


def _relative_to_first(values: np.ndarray) -> np.ndarray:
    """`values` divided by its first entry, or unchanged when that can't be done (empty, NaN or 0)."""
    if len(values) and np.isfinite(values[0]) and values[0] != 0:
        return values / values[0]
    return values


# ---- text for the scorecard, the circuit and the lists --------------------------------------------
def _fmt(value) -> str:
    """A scorecard number as text: "-" if missing, no decimals from 100 up, two below."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.2f}"


def _row_label(label: str, unit: str) -> str:
    """A scorecard row label with its unit in brackets, when it has one."""
    return label + (f" ({unit})" if unit else "")


def _product_name(cut: str) -> str:
    """The product stream's name for a cut: NdPr for the LaCe cut, Nd for the Pr check case."""
    return {"LaCe": "NdPr", "Pr": "Nd"}.get(cut, "product")


def _raffinate_name(cut: str) -> str:
    """The raffinate stream's name for a cut: the lighter elements that leave with it."""
    return {"LaCe": "La+Ce", "Pr": "Pr and lighter"}.get(cut, "lighter")


def _layout(spec: RunSpec) -> dict:
    """The circuit layout from src.runs, plus the stream names the drawing shows for this cut."""
    layout = circuit_layout(spec)
    return {**layout, "product_name": _product_name(layout["cut"]),
            "raffinate_name": _raffinate_name(layout["cut"])}


def _short_label(spec: RunSpec) -> str:
    """A run's label for a scorecard column: plant / controller / upset / sensors."""
    return f"{spec.plant} / {spec.controller} / {spec.upset} / {spec.sensors}"


# ---- the run files and the saved runs on disk ------------------------------------------------------
def _run_file_options() -> list[tuple[str, str, str]]:
    """Every run file in configs/, as (name, name, explanation), sorted by name."""
    return [(name, name, _config_explanation(name)) for name in store.run_file_names()]


def _config_explanation(name: str) -> str:
    """What a run file is: its own description (the `description` field in the file), then one
    line worked out from it: stages per section, where the cut is, the feed, the scrub."""
    description = run_file_description(name)
    summary = _config_summary(name)
    return f"{description}\n\n{summary}" if description else summary


def _config_summary(name: str) -> str:
    """One line about a run file: stages per section, where the cut is, the feed, the scrub.
    A run file that doesn't load says why, in place of the line."""
    try:
        run_config = load_config(name)
    except Exception as exc:                          # a run file that doesn't load says why
        return f"(does not load: {exc})"
    circuit = run_config["circuit"]
    stages = circuit["n_ext"] + circuit["n_scrub"] + circuit["n_strip"]
    scrub = "reflux" if run_config.reflux else "acid-only"
    return (f"{stages} stages ({circuit['n_ext']} extraction, {circuit['n_scrub']} scrub, "
            f"{circuit['n_strip']} strip), cut after {run_config.cut}, feed {run_config['feed']['deposit']}, "
            f"{scrub} scrub.")
