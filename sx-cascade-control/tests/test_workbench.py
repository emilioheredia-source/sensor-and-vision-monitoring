"""Checks on the workbench GUI (workbench/presenter.py and workbench/view.py).

The workbench is split so that all the logic sits in the presenter, which
has no GUI code in it. The view only shows what it's told and reports what
the user did. That split is what makes most of this testable without a
screen: the presenter is driven against a fake view that just records every
call, and the tests check what the presenter told it to show.

One last test builds the real PyQt view offscreen and runs a short run
through it, to catch wiring mistakes between the two.

The runs use ndpr_small (the NdPr product) with the MPC and a 20 % stronger
feed. The control interval is 0.5 h.
"""
import os

import pytest

from src.runs import RunSpec
from workbench.presenter import KINDS, WorkbenchPresenter


class FakeView:
    """Implements presenter.WorkbenchView by recording what it is told; runs 'background' work inline.

    Any method the presenter calls (show_options, update_plots, show_error and
    so on) is stored in `calls` as {method name: [args of each call, in order]}.
    """

    def __init__(self):
        self.calls = {}

    def __getattr__(self, name):
        # Called only for methods FakeView doesn't define: return a function that records its arguments.
        def record(*args):
            self.calls.setdefault(name, []).append(args)
        return record

    def run_in_background(self, produce, on_item, on_done, on_error):
        """Run the work right here, in order, so a test sees the whole run when run() returns."""
        try:
            for item in produce():
                on_item(item)
        except Exception as exc:
            on_error(str(exc))
        else:
            on_done()


@pytest.fixture
def presenter():
    """A presenter on a fake view, started on a 4 h ndpr_small run under the MPC with a feed step.

    Returns (presenter, view).
    """
    view = FakeView()
    p = WorkbenchPresenter(view, RunSpec(config="ndpr_small", controller="mpc", upset="feed_step", hours=4.0))
    p.start()
    return p, view


def test_start_offers_every_choice_with_its_explanation(presenter):
    """On start, the view is shown every kind of choice, an explanation for each, and the settings.

    The kinds are run file, plant, controller, upset and sensors. Each comes
    with a non-empty explanation. The settings shown are the run file, the
    parameter file and the controller settings, with nothing marked as edited yet.
    """
    _, view = presenter
    # the first argument of each show_options call is the kind of choice
    kinds = {args[0] for args in view.calls["show_options"]}
    assert kinds == set(KINDS)
    assert all(text for _, text in view.calls["show_explanation"])
    values, edited = view.calls["show_settings"][-1]
    assert set(values) == {"run_file", "parameters", "controller_settings"} and edited == []


def test_an_edit_is_kept_marked_and_forgotten(presenter):
    """An edited setting shows its new value, is marked as edited, and a reset clears every edit.

    The feed flow goes to 640 L/h and has to show up as such and be marked.
    A list typed as JSON text (the MPC's handles) has to be read back as a
    real list. After reset_settings the spec holds no edits at all.
    """
    p, view = presenter
    p.edit_setting("run_file", ("flows", "feed"), 640.0)
    values, edited = view.calls["show_settings"][-1]
    assert values["run_file"]["flows"]["feed"] == 640.0
    assert ("run_file", "flows", "feed") in edited
    p.edit_setting("controller_settings", ("handles",), '["Q_org", "Q_scrub"]')      # a list, typed as JSON
    assert p.spec.controller_settings["handles"] == ["Q_org", "Q_scrub"]
    p.reset_settings()
    assert p.spec.run_file == {} and p.spec.controller_settings == {}


def test_a_bad_edit_is_undone_and_reported(presenter):
    """An edit the model can't take is taken back out of the spec and the user is told.

    Asking for the "exact" stage model, which isn't built, makes loading the
    run fail. The edit must not stay in the spec, and the view must have been
    shown an error.
    """
    p, view = presenter
    p.edit_setting("run_file", ("circuit", "model"), "exact")        # a setting that is not built
    assert p.spec.run_file.get("circuit", {}).get("model") is None
    assert view.calls["show_error"]


def test_a_run_streams_plots_then_scores_and_saves(presenter, tmp_path, monkeypatch):
    """A run updates the plots once per interval, then shows the scorecard, saves, and can be compared.

    4 h at 0.5 h per interval is 8 intervals, so the plots are updated 9
    times counting the start point. Handle series are shown relative to
    their start, so the organic flow starts at 1.0. The scorecard has one
    column (this run) and 6 rows. The run is saved, and comparing against
    the saved copy gives a series that ends at 4 h.
    """
    p, view = presenter
    import src.runs as runs
    # save runs in a temporary folder, not in runs/saved
    monkeypatch.setattr(runs, "SAVED_DIR", tmp_path)
    p.run()
    assert len(view.calls["update_plots"]) == 9                      # the start point and 8 intervals
    series = view.calls["update_plots"][-1][0]
    assert len(series["t"]) == 9 and series["handle_Q_org"][0] == pytest.approx(1.0)
    columns, rows = view.calls["show_scorecard"][-1]
    assert len(columns) == 1 and len(rows) == 6
    assert list(tmp_path.glob("*/run.json"))
    p.compare([next(tmp_path.glob("*")).name])
    assert view.calls["show_comparison"][-1][0][0]["series"]["t"][-1] == pytest.approx(4.0)


def test_the_qt_view_wires_up_and_draws_a_run(tmp_path, monkeypatch):
    """The real PyQt view, run offscreen, takes a 2 h run from start to scorecard.

    This catches mistakes the fake view can't: a view method missing or
    taking the wrong arguments, or the background thread not handing results
    back. At the end the run thread is gone, the scorecard has its 6 rows,
    and the purity curve has 5 points (the start and 4 intervals of 0.5 h).
    """
    # no display needed: Qt draws to an offscreen buffer
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtWidgets
    from workbench.view import QtWorkbenchView               # imports pyqtgraph, pinned to PyQt5
    import src.runs as runs
    # save runs in a temporary folder, not in runs/saved
    monkeypatch.setattr(runs, "SAVED_DIR", tmp_path)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = QtWorkbenchView(remember_layout=False)
    p = WorkbenchPresenter(view, RunSpec(config="ndpr_small", controller="mpc", upset="feed_step", hours=2.0))
    view.bind(p)
    p.start()
    p.run()
    for _ in range(600):                                  # let the background run finish (60 s at most)
        app.processEvents()
        if view._thread is None:
            break
        QtCore.QThread.msleep(100)
    assert view._thread is None
    assert view.score.rowCount() == 6
    assert len(view.product._curves["purity"].getData()[0]) == 5
    view.close()


def test_settings_saved_as_a_run_file_become_the_run_file(presenter):
    """Save settings as a new run file writes configs/<name>.toml with the edits, makes it the run
    file and clears the edits (they're in the file now). A taken name is refused with an error."""
    from src.config import CONFIG_DIR, load_config
    p, view = presenter
    p.edit_setting("run_file", ("flows", "feed"), 123.0)
    name = "test_workbench_save_as"
    try:
        p.save_settings_as(name, "from the tests")
        assert p.spec.config == name and p.spec.run_file == {}
        assert load_config(name)["flows"]["feed"] == 123.0
        p.save_settings_as(name)
        assert "already exists" in view.calls["show_error"][-1][0]
    finally:
        (CONFIG_DIR / f"{name}.toml").unlink(missing_ok=True)


def test_a_setting_is_found_in_its_file_and_in_the_code(presenter):
    """Show in its file opens the run file at the setting's line; Where the code uses it lists
    lines in src/ that mention the name; a source file is shown read only."""
    p, view = presenter
    p.locate_setting("run_file", ("flows", "feed"), "show")
    path, text, line, editable = view.calls["show_text"][-1]
    assert path.endswith("ndpr_small.toml") and editable and text.splitlines()[line - 1].startswith("feed")
    p.locate_setting("controller_settings", ("w_reagent",), "show")
    path, _, line, editable = view.calls["show_text"][-1]
    assert path.endswith("mpc.py") and not editable and line
    p.locate_setting("parameters", ("stage_geometry", "mixer_volume_L"), "uses")
    name, hits = view.calls["show_code_uses"][-1]
    assert name == "mixer_volume_L" and hits


def test_a_text_edit_that_does_not_load_is_not_saved(presenter):
    """Saving a run file edited into something that doesn't load shows an error and leaves the
    file as it was."""
    from src.config import CONFIG_DIR
    p, view = presenter
    p.open_text("run_file")
    path, text, _, _ = view.calls["show_text"][-1]
    p.save_text(path, text + "\n[flows]\nfeed = 1.0\n")              # a table defined twice: bad TOML
    assert "Not saved" in view.calls["show_error"][-1][0]
    assert (CONFIG_DIR / "ndpr_small.toml").read_text(encoding="utf-8") == text


def test_the_qt_editor_panel_and_menu_build(monkeypatch):
    """The File menu, the File editor panel and the settings tree's right-click handler exist and
    take what the presenter sends, offscreen."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from workbench.view import QtWorkbenchView
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = QtWorkbenchView(remember_layout=False)
    p = WorkbenchPresenter(view, RunSpec(config="ndpr_small", hours=1.0))
    view.bind(p)
    p.start()
    titles = [action.text() for action in view.menuBar().actions()]
    assert titles[:2] == ["File", "View"]
    p.locate_setting("run_file", ("flows", "feed"), "show")
    assert view.editor.text.textCursor().block().text().startswith("feed")
    assert not view.editor.text.isReadOnly()
    p.locate_setting("controller_settings", ("w_reagent",), "show")
    assert view.editor.text.isReadOnly()
    app.processEvents()
    view.close()


def test_an_edit_keeps_the_type_of_the_setting_it_replaces():
    """Text that looks like a number comes in as the number; text that isn't one is refused.
    (Oct 18: "0.99" as text reached the controller as its purity spec.)"""
    from workbench.presenter import _parse
    assert _parse("0.99", 0.995) == 0.99 and isinstance(_parse("0.99", 0.995), float)
    assert _parse("8", 8) == 8 and isinstance(_parse("8", 8), int)
    assert _parse("false", True) is False
    for bad, current in (("high", 0.99), ("8.5", 8), ("maybe", True)):
        with pytest.raises(ValueError):
            _parse(bad, current)


def test_a_wrong_type_is_refused_with_a_message(presenter):
    p, view = presenter
    p.edit_setting("run_file", ("flows", "feed"), "lots")
    assert "expected a number" in view.calls["show_error"][-1][0]
    assert p.spec.run_file == {}


def test_switching_run_files_records_no_edits():
    """The tree updating its own values doesn't come back as edits (it used to, after a switch
    from ndpr_small or ndpr_plant to prnd_plant), and no box holds a number as text."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from workbench.view import QtWorkbenchView
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = QtWorkbenchView(remember_layout=False)
    p = WorkbenchPresenter(view, RunSpec(config="ndpr_small", hours=1.0))
    view.bind(p)
    p.start()
    p.choose("config", "prnd_plant")
    app.processEvents()
    assert p.spec.run_file == {} and p.spec.parameters == {} and p.spec.controller_settings == {}
    view.close()


def test_a_session_log_is_written_the_last_20_kept_and_shown(tmp_path, monkeypatch):
    """Each session opens a log; only the newest 20 stay; status lines and errors go in it; the
    Logs panel lists the files and shows one."""
    import logging
    from workbench import logs
    for k in range(25):
        (tmp_path / f"workbench_20260101_0000{k:02d}.log").write_text("old\n")
    path = logs.start_session_log(tmp_path, console=False)
    try:
        assert len(logs.log_files(tmp_path)) == 20 and path in logs.log_files(tmp_path)
        import src.store as store
        monkeypatch.setattr(store, "LOG_DIR", tmp_path)              # where the presenter's logs come from
        view = FakeView()
        p = WorkbenchPresenter(view, RunSpec(config="ndpr_small", hours=1.0))
        p.start()
        p.edit_setting("run_file", ("flows", "feed"), "lots")             # an error, logged
        for h in logging.getLogger().handlers:
            h.flush()
        p.show_logs()
        name, text = view.calls["show_log_text"][-1]
        assert name == path.name and "Ready." in text and "expected a number" in text
    finally:
        for h in list(logging.getLogger().handlers):
            if isinstance(h, logging.FileHandler) and h.baseFilename == str(path):
                logging.getLogger().removeHandler(h)
                h.close()


def test_the_run_controls_and_progress_stay_in_view_and_the_plots_can_hide_their_points():
    """The run controls are a toolbar and the progress sits in the status bar, outside every panel;
    progress goes from busy to a fraction; Run and Stop swap while a run goes on; the View menu's
    data-points switch reaches every plot."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from workbench.view import QtWorkbenchView
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = QtWorkbenchView(remember_layout=False)
    p = WorkbenchPresenter(view, RunSpec(config="ndpr_small", hours=4.0))
    view.bind(p)
    p.start()
    assert view.run_bar in view.findChildren(QtWidgets.QToolBar)
    bar = view.run_progress.progress
    assert view.run_progress.parent() is view.statusBar()
    view.set_running(True)
    assert view.run_bar.stop_action.isEnabled() and not view.run_bar.run_action.isEnabled()
    view.set_running(False)
    view.show_progress(None, "preparing")
    assert bar.maximum() == 0
    view.show_progress(0.5, "2.0 of 4 h")
    assert bar.maximum() == 1000 and bar.value() == 500
    view.set_markers(False)
    assert all(c.opts["symbol"] is None for c in view.product._curves.values())
    view.set_markers(True)
    assert all(c.opts["symbol"] == "o" for c in view.product._curves.values())
    view.close()


def test_a_finished_run_has_a_report_shown_and_saved(presenter, tmp_path, monkeypatch):
    """The 4 h feed-step run ends with its report: shown in the view and saved as report.md."""
    import src.runs as runs
    monkeypatch.setattr(runs, "SAVED_DIR", tmp_path)
    p, view = presenter
    p.run()
    text = view.calls["show_report"][-1][0]
    assert text.startswith("# Run report") and "## What happened" in text and "## The numbers" in text
    assert list(tmp_path.glob("*/report.md"))


def test_a_layout_saved_to_a_file_comes_back(tmp_path):
    """View > Save layout and Load layout: a hidden panel and the data-points switch come back as
    they were saved."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from workbench.view import QtWorkbenchView
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = QtWorkbenchView(remember_layout=False)
    view.bind(WorkbenchPresenter(view, RunSpec(config="ndpr_small", hours=1.0)))
    view.show()
    view._docks["logs"].hide()
    view._points_action.setChecked(False)
    view.write_layout(tmp_path / "mine.json")
    view._docks["logs"].show()
    view._points_action.setChecked(True)
    assert view.read_layout(tmp_path / "mine.json")
    app.processEvents()
    assert not view._docks["logs"].isVisible() and not view._points_action.isChecked()
    view.close()
