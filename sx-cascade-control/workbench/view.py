"""The PyQt front end: a passive shell that implements presenter.WorkbenchView.

What this file is
-----------------
This is the main window. It does three jobs and nothing else:

    1. It builds the panels (from panels.py) and connects each button and list to a method of the
       presenter, so "the user did X" becomes a call such as presenter.run().
    2. It implements every method of the WorkbenchView interface by passing the values on to the
       right panel. When the presenter says show_scorecard(columns, rows), this file hands them to
       the scorecard panel, which fills its table.
    3. It runs the presenter's background work in a QThread and brings each item back to the
       GUI thread (see "Running in the background" below).

There's no logic about the plant or the run in here. Decisions (what to show, when to save, what
a setting means) are in presenter.py. That's what makes this file and panels.py the only two that
would be rewritten to swap PyQt for another front end, such as a web page (ARCHITECTURE.md,
decision 24).

Floating docks
--------------
Every panel sits in a QDockWidget. A dock can sit in the main window (side by side, stacked, or
tabbed if you drop one on another) or float as its own window on any monitor. The arrangement is
saved when the window closes and restored next time (QSettings); View > Reset layout puts it back
to the default.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import traceback
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from src.plotting import COLOR, INK2, SLOT, STATUS_CRITICAL

from .panels import (MARKERS, CircuitPanel, InterfacePlot, ProfilePlot, RunPanel, SavedPanel, ScorePanel,
                     LogPanel, ReportPanel, RunProgress, RunToolBar, SettingsPanel, TextEditorPanel, TimePlots)

# Saved with the dock arrangement. Raising it makes Qt ignore an arrangement saved by an older
# layout of the docks, so a changed set of panels starts from the default instead.
LAYOUT_VERSION = 5

# Where Qt keeps the saved window geometry and dock arrangement: (organisation, application).
SETTINGS_SCOPE = ("sx-cascade-control", "workbench")

# Where View > Save layout... puts layout files (personal, not kept in git).
LAYOUT_DIR = Path(__file__).resolve().parent.parent / "layouts"


# ==================================================================================================
# What each time-plot panel shows
# ==================================================================================================
# A TimePlots panel is a stack of plots sharing the time axis. Each row is
#     (plot title, y-axis label, [(series key, legend label, colour), ...])
# and the series keys are the names in the dict the presenter sends to update_plots().

# Recovery is what's leaving in the product now over what's coming in now, so it can pass 1 for a
# while: when the front moves, the cascade gives up NdPr it was holding. The title says so.
PRODUCT_ROWS = [("Product purity", "fraction", [("purity", "plant", COLOR["Nd"])]),
                ("Recovery now: NdPr out / NdPr in (above 1 while the cascade releases metal it holds)",
                 "fraction", [("recovery", "plant", SLOT[0])])]

HANDLE_ROWS = [("Handles, as a fraction of their starting value", "× start",
                [("handle_Q_org", "organic flow", SLOT[0]), ("handle_Q_scrub", "scrub flow", SLOT[1]),
                 ("handle_h_scrub", "scrub acid", SLOT[3]),
                 ("handle_saponified_fraction", "saponification", SLOT[2])])]

# Each loss and reagent plot comes twice: against what's coming in or going out (the figure that
# means something to a person: % of the NdPr being fed now, mol of reagent per mol of NdPr made),
# then as the raw rate it comes from.
LOSS_ROWS = [("NdPr lost to the raffinate, % of the NdPr being fed now", "%", [("lost_pct_of_feed", "lost", SLOT[1])]),
             ("NdPr lost to the raffinate", "mol/h", [("lost_rate", "lost", SLOT[1])]),
             ("Reagent per mol of NdPr made", "mol/mol", [("reagent_per_product", "reagent", SLOT[3])]),
             ("Reagent (caustic and acid)", "mol/h", [("reagent_rate", "reagent", SLOT[3])])]

# What the controller sees and plans, next to what the plant actually does.
CONTROLLER_ROWS = [("Purity: plant, as the controller is told, as it plans", "fraction",
                    [("purity", "plant", COLOR["Nd"]), ("purity_seen", "told", SLOT[4]),
                     ("purity_plan", "plan (end of horizon)", INK2)]),
                   ("Output bias the controller carries (purity)", "fraction",
                    [("bias_purity", "bias", STATUS_CRITICAL)])]


# ==================================================================================================
# Running in the background
# ==================================================================================================
# The presenter gives the view a generator function. Walking that generator is slow (it runs the
# whole simulation), so it happens on a QThread. Qt widgets may only be touched from the GUI
# thread, though, so the worker never calls the presenter directly. It emits a signal for each
# item instead. A signal sent from one thread to an object living in another is queued by Qt and
# delivered on the receiver's thread, so the window's slots (_item, _done, _failed) run on the GUI
# thread, in order, and they call the presenter's callbacks from there.

class _Worker(QtCore.QObject):
    """Pulls items from a generator in a background thread and hands each one back as a signal.

    Signals:
        item(object): one item from the generator.
        done(): the generator ended normally.
        failed(str): the generator raised; the argument is the full traceback.
    """
    item = QtCore.pyqtSignal(object)
    done = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)

    def __init__(self, produce):
        """produce: a function that returns the iterator to walk (called on the worker's thread)."""
        super().__init__()
        self._produce = produce

    def run(self):
        """Walk the iterator, emitting each item, then `done` or `failed`."""
        try:
            for produced_item in self._produce():
                self.item.emit(produced_item)
        except Exception:                             # shown to the user, not swallowed
            self.failed.emit(traceback.format_exc())
        else:
            self.done.emit()


# ==================================================================================================
# The main window
# ==================================================================================================
class QtWorkbenchView(QtWidgets.QMainWindow):
    """The PyQt implementation of presenter.WorkbenchView.

    Args:
        remember_layout: save the dock arrangement on close and restore it on start. The tests
            turn it off so they don't read or write the user's saved arrangement.

    Usage:
        view = QtWorkbenchView()
        presenter = WorkbenchPresenter(view, spec)
        view.bind(presenter)          # builds the panels; must come before presenter.start()
        presenter.start()
        view.show()
    """

    def __init__(self, remember_layout: bool = True):
        """Make the empty main window. The panels are built later, by bind()."""
        super().__init__()
        self._remember = remember_layout
        self.setWindowTitle("SX cascade workbench")
        self.setDockOptions(self.AllowNestedDocks | self.AllowTabbedDocks | self.AnimatedDocks)
        self.presenter = None

        # The background run, while one is going: the QThread, the worker living in it, and the
        # presenter's three callbacks (on_item, on_done, on_error). All None when idle.
        self._thread = self._worker = None
        self._callbacks = None

        # Every dock by a short key ("run", "settings", "circuit", ...), filled by _add_docks.
        self._docks: dict[str, QtWidgets.QDockWidget] = {}

    # ----------------------------------------------------------------------------------------------
    # Wiring: build the panels and connect them to the presenter
    # ----------------------------------------------------------------------------------------------
    # The panels take plain callbacks. Passing the presenter's bound methods (presenter.choose,
    # presenter.run, ...) is all the wiring there is: a click calls the method, with the values
    # the panel read from its widgets.

    def bind(self, presenter):
        """Build the panels, connect what the user does to the presenter, and lay out the docks."""
        self.presenter = presenter
        # reset_settings goes through a lambda: Qt would pass the button's checked flag on as
        # its first argument, and it would be read as a section name.
        self.run_panel = RunPanel(presenter.choose)
        self.run_bar = RunToolBar(presenter.run, presenter.stop, presenter.set_hours, lambda: presenter.reset_settings())
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.run_bar)
        self.run_progress = RunProgress()
        self.statusBar().addPermanentWidget(self.run_progress)
        self.settings = SettingsPanel(presenter.edit_setting, presenter.locate_setting)
        self.editor = TextEditorPanel(presenter.save_text, lambda path: self._reload_text(path),
                                      presenter.open_in_code)
        self.logs = LogPanel(presenter.show_logs, presenter.open_log)
        self.circuit = CircuitPanel()
        self.profile = ProfilePlot()
        self.product = TimePlots(PRODUCT_ROWS)
        self.interfaces = InterfacePlot()
        self.handles = TimePlots(HANDLE_ROWS)
        self.losses = TimePlots(LOSS_ROWS)
        self.controller_view = TimePlots(CONTROLLER_ROWS)
        self.score = ScorePanel()
        self.saved = SavedPanel(presenter.compare, presenter.refresh_saved, presenter.show_saved_report)
        self.report = ReportPanel()
        self._add_docks()
        self._add_menu()
        self._restore_layout()

    def _add_docks(self):
        """Put each panel in its own dock, then arrange the docks in the default layout."""
        # (key, dock title, panel). The key names the dock in _docks and in the saved arrangement.
        panels = [("run", "Run", self.run_panel), ("settings", "Settings", self.settings),
                  ("circuit", "Circuit", self.circuit), ("profile", "Stages, now", self.profile),
                  ("product", "Product", self.product), ("interfaces", "Interfaces", self.interfaces),
                  ("handles", "Handles", self.handles), ("losses", "Losses and reagent", self.losses),
                  ("controller", "Controller's view", self.controller_view),
                  ("score", "Scorecard", self.score), ("saved", "Saved runs", self.saved),
                  ("editor", "File editor", self.editor), ("logs", "Logs", self.logs),
                  ("report", "Run report", self.report)]
        for key, title, widget in panels:
            dock = QtWidgets.QDockWidget(title, self)
            dock.setObjectName(f"dock_{key}")                  # needed to save and restore the arrangement
            dock.setWidget(widget)                             # the Run panel scrolls its own choices
            dock.setFeatures(dock.DockWidgetMovable | dock.DockWidgetFloatable | dock.DockWidgetClosable)
            self._docks[key] = dock
        self._default_layout()

    # The default layout is three columns:
    #     left:   Run, and under it Settings / Saved runs / Scorecard as tabs
    #     middle: Circuit, Product, Handles
    #     right:  Stages now, Interfaces, Losses and reagent / Controller's view as tabs
    # Two groups start as tabs so it all fits one screen; drag any tab out to float it.
    # The sizes are in pixels, for a 1800 x 1050 window.

    def _default_layout(self):
        """Dock every panel in the default arrangement (also View > Reset layout)."""
        docks = self._docks
        for dock in docks.values():
            dock.setFloating(False)
            dock.show()

        # Left column.
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks["run"])
        self.splitDockWidget(docks["run"], docks["settings"], QtCore.Qt.Vertical)
        self.tabifyDockWidget(docks["settings"], docks["saved"])
        self.tabifyDockWidget(docks["saved"], docks["score"])
        self.tabifyDockWidget(docks["score"], docks["editor"])
        self.tabifyDockWidget(docks["editor"], docks["logs"])
        self.tabifyDockWidget(docks["losses"], docks["report"])

        # Middle and right columns: the circuit, then the stage profile beside it, then each
        # column filled downwards.
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks["circuit"])
        self.splitDockWidget(docks["circuit"], docks["profile"], QtCore.Qt.Horizontal)
        self.splitDockWidget(docks["circuit"], docks["product"], QtCore.Qt.Vertical)
        self.splitDockWidget(docks["product"], docks["handles"], QtCore.Qt.Vertical)
        self.splitDockWidget(docks["profile"], docks["interfaces"], QtCore.Qt.Vertical)
        self.splitDockWidget(docks["interfaces"], docks["losses"], QtCore.Qt.Vertical)
        self.tabifyDockWidget(docks["losses"], docks["controller"])

        # Which tab is on top in each tabbed group.
        docks["settings"].raise_()
        docks["losses"].raise_()
        self._default_sizes()

    def _default_sizes(self):
        """Column widths, then the heights within each column, for an 1800 x 1050 window."""
        self.resize(1800, 1050)
        self._resize(["run", "circuit", "profile"], [430, 690, 680], QtCore.Qt.Horizontal)
        self._resize(["run", "settings"], [470, 560], QtCore.Qt.Vertical)
        self._resize(["circuit", "product", "handles"], [250, 390, 390], QtCore.Qt.Vertical)
        self._resize(["profile", "interfaces", "losses"], [330, 300, 400], QtCore.Qt.Vertical)

    def _resize(self, dock_keys, sizes, orientation):
        """Give the named docks these widths (Horizontal) or heights (Vertical), in pixels."""
        self.resizeDocks([self._docks[key] for key in dock_keys], sizes, orientation)

    def _add_menu(self):
        """The File menu (save the settings as a run file, edit the files as text) and the View
        menu (show or hide each panel, float or dock them all, reset the layout). A run file is
        loaded by choosing it in the Run panel's list."""
        files = self.menuBar().addMenu("File")
        files.addAction("Save settings as a new run file...", self._ask_save_as)
        files.addSeparator()
        files.addAction("Edit this run file as text", lambda: self.presenter.open_text("run_file"))
        files.addAction("Edit parameters.toml as text", lambda: self.presenter.open_text("parameters"))
        files.addSeparator()
        files.addAction("Show the logs", self._show_logs)

        menu = self.menuBar().addMenu("View")
        for dock in self._docks.values():
            menu.addAction(dock.toggleViewAction())
        menu.addSeparator()
        menu.addAction("Float all panels", lambda: self._set_all_floating(True))
        menu.addAction("Dock all panels", lambda: self._set_all_floating(False))
        menu.addAction("Reset layout", self._default_layout)
        menu.addAction("Save layout...", self._save_layout_file)
        menu.addAction("Load layout...", self._load_layout_file)
        menu.addSeparator()
        self._points_action = menu.addAction("Show data points on the plots")
        self._points_action.setCheckable(True)
        self._points_action.setChecked(True)
        self._points_action.toggled.connect(self.set_markers)

    def _ask_save_as(self):
        """Ask for a name and a one-line note, then hand them to the presenter."""
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save settings as a new run file",
            "Name (letters, digits, _ and -). It's saved as configs/<name>.toml:")
        if not ok or not name.strip():
            return
        note, _ = QtWidgets.QInputDialog.getText(self, "Save settings as a new run file",
                                                 "What this run file is, in a sentence or two (shown when it is chosen):")
        self.presenter.save_settings_as(name.strip(), note.strip())

    def _show_logs(self):
        """File > Show the logs: fill the Logs panel and bring it to the front."""
        self.presenter.show_logs()
        self._docks["logs"].show()
        self._docks["logs"].raise_()

    def _reload_text(self, path):
        """Reload: the file as it is on disk."""
        with open(path, encoding="utf-8") as handle:
            self.editor.show_text(path, handle.read(), None, path.endswith(".toml"))

    def _set_all_floating(self, floating: bool):
        """Float every panel as its own window, or dock them all back into the main window."""
        for dock in self._docks.values():
            dock.setFloating(floating)

    # ----------------------------------------------------------------------------------------------
    # Saving and restoring the arrangement
    # ----------------------------------------------------------------------------------------------

    def _restore_layout(self):
        """Put the window and docks back where they were last time, if remember_layout is on."""
        if not self._remember:
            return
        settings = QtCore.QSettings(*SETTINGS_SCOPE)
        geometry, state = settings.value("geometry"), settings.value("state")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None:
            self.restoreState(state, LAYOUT_VERSION)

    def _save_layout(self):
        """Remember the window's size and place and the dock arrangement for next time."""
        settings = QtCore.QSettings(*SETTINGS_SCOPE)
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("state", self.saveState(LAYOUT_VERSION))

    # ---- named layouts, kept as files ------------------------------------------------------------
    # The arrangement is also remembered between sessions (above); these keep several by name, for
    # instance one for working and one for screenshots. A file holds the window's size and place,
    # the docks, and whether the plots show their data points.

    def _save_layout_file(self):
        """Ask for a file name in layouts/ and write the current arrangement to it."""
        LAYOUT_DIR.mkdir(exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save layout", str(LAYOUT_DIR / "my_layout.json"),
                                                        "Workbench layouts (*.json)")
        if path:
            self.write_layout(Path(path))
            self.set_status(f"Layout saved as {Path(path).name}.")

    def _load_layout_file(self):
        """Ask for a layout file and put the window and docks the way it says."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load layout", str(LAYOUT_DIR),
                                                        "Workbench layouts (*.json)")
        if not path:
            return
        if self.read_layout(Path(path)):
            self.set_status(f"Layout {Path(path).name} loaded.")
        else:
            self.show_error(f"{Path(path).name} was saved when the workbench had a different set of panels; "
                            "the panels it doesn't know stay where they were.")

    def write_layout(self, path: Path):
        """Write the window's size and place, the docks and the data-points setting to a JSON file."""
        layout = {"version": LAYOUT_VERSION, "data_points": self._points_action.isChecked(),
                  "geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
                  "state": bytes(self.saveState(LAYOUT_VERSION).toBase64()).decode("ascii")}
        path.write_text(json.dumps(layout, indent=1), encoding="utf-8")

    def read_layout(self, path: Path) -> bool:
        """Put the window the way a layout file says. Returns False if the file was saved with a
        different set of panels (Qt then leaves the docks it can't place where they are)."""
        layout = json.loads(path.read_text(encoding="utf-8"))
        self.restoreGeometry(QtCore.QByteArray.fromBase64(layout["geometry"].encode("ascii")))
        restored = self.restoreState(QtCore.QByteArray.fromBase64(layout["state"].encode("ascii")),
                                     layout.get("version", LAYOUT_VERSION))
        self._points_action.setChecked(layout.get("data_points", True))
        return bool(restored) and layout.get("version") == LAYOUT_VERSION

    def closeEvent(self, event):
        """On close: stop any run, save the arrangement, and wait for the background thread.

        The wait is up to 30 s, so the interval being worked on can finish before the window
        goes away.
        """
        if self.presenter is not None:
            self.presenter.stop()
        if self._remember:
            self._save_layout()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(30000)
        super().closeEvent(event)

    # ----------------------------------------------------------------------------------------------
    # WorkbenchView: what the presenter tells the view
    # ----------------------------------------------------------------------------------------------
    # Each method passes the values to the panel that shows them. The docstrings for what the
    # arguments mean are on the WorkbenchView interface in presenter.py.

    def show_options(self, kind, options, selected):
        """Fill one choice list on the Run panel."""
        self.run_panel.show_options(kind, options, selected)

    def show_explanation(self, kind, text):
        """Show the explanation under one choice list."""
        self.run_panel.show_explanation(kind, text)

    def show_hours(self, hours):
        """Show the run length."""
        self.run_bar.show_hours(hours)

    def show_settings(self, values, edited):
        """Fill the settings tree."""
        self.settings.show_settings(values, edited)

    def show_layout(self, layout):
        """Redraw the circuit."""
        self.circuit.show_layout(layout)

    def begin_plots(self, meta):
        """Clear every plot for a new run and draw its reference lines.

        Each reference line is (plot row, "h" or "v", value, label): "h" is a horizontal level
        such as the purity spec, "v" a vertical time such as the start of the upset.
        """
        upset_time = meta["upset_time"]
        purity_spec = meta["spec"]
        window_low, window_high = meta["window"]
        self.product.begin([(0, "h", purity_spec, "spec"), (0, "h", meta["purity_target"], "target"),
                            (1, "h", meta["recovery_target"], "target")] + _upset_marks(upset_time, 0, 1))
        self.interfaces.begin([(0, "h", window_low, "window"), (0, "h", window_high, "")]
                              + _upset_marks(upset_time, 0))
        self.handles.begin([(0, "h", 1.0, "start")] + _upset_marks(upset_time, 0))
        self.losses.begin(_upset_marks(upset_time, 0, 1))
        self.controller_view.begin([(0, "h", purity_spec, "spec")] + _upset_marks(upset_time, 0, 1))
        self.profile.begin(meta["components"], meta["window"], meta["all_components"])
        self.setWindowTitle(f"SX cascade workbench: {meta['title']}")

    def update_plots(self, series):
        """Redraw every time plot from the live series."""
        for panel in (self.product, self.interfaces, self.handles, self.losses, self.controller_view):
            panel.update(series)

    def show_profile(self, t, profile):
        """Redraw the stage profile at time t."""
        self.profile.show(t, profile)

    def show_scorecard(self, columns, rows):
        """Fill the scorecard table."""
        self.score.show_scorecard(columns, rows)

    def show_saved_runs(self, runs):
        """Fill the saved-runs list."""
        self.saved.show_saved_runs(runs)

    def show_comparison(self, runs):
        """Overlay saved runs on the time plots. The controller's-view panel is left out; a saved
        run's series don't carry the controller's record (see _series_of in presenter.py)."""
        for panel in (self.product, self.interfaces, self.handles, self.losses):
            panel.compare(runs)

    def set_running(self, running):
        """Enable Run or Stop, lock the choices and the run length, and run the clock while a run goes on."""
        self.run_panel.set_running(running)
        self.run_bar.set_running(running)
        self.run_progress.set_running(running)

    def set_status(self, text):
        """Show the message on the window's status bar (it stays until the next one)."""
        self.statusBar().showMessage(text)

    def show_error(self, text):
        """Put the last line of the error on the status bar and the whole of it in a dialog."""
        self.set_status("Error: " + text.strip().splitlines()[-1])
        QtWidgets.QMessageBox.warning(self, "Workbench", text)

    def show_text(self, path, text, line, editable):
        """Show the file in the File editor panel and bring that panel to the front."""
        self.editor.show_text(path, text, line, editable)
        self._docks["editor"].show()
        self._docks["editor"].raise_()

    def show_code_uses(self, name, hits):
        """List the lines of code that mention a setting; double-click one to open it in VS Code."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Where the code uses '{name}'")
        listing = QtWidgets.QListWidget()
        for path, line, text in hits:
            item = QtWidgets.QListWidgetItem(f"{_path_inside_src(path)}:{line}    {text}")
            item.setData(QtCore.Qt.UserRole, (path, line))      # what a double-click opens
            listing.addItem(item)
        if not hits:
            listing.addItem("No line in src/ mentions this name.")
        listing.itemDoubleClicked.connect(self._open_code_use)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Double-click a line to open it in VS Code."))
        layout.addWidget(listing)
        dialog.resize(900, 400)
        dialog.show()

    def _open_code_use(self, item):
        """A line in the code-uses list was double-clicked: open it in VS Code. The "no line"
        message has no (path, line) attached, so it does nothing."""
        location = item.data(QtCore.Qt.UserRole)
        if location:
            self.presenter.open_in_code(*location)

    def show_report(self, text):
        """Show a run's report and bring its panel to the front."""
        self.report.show_report(text)
        self._docks["report"].show()
        self._docks["report"].raise_()

    def show_progress(self, fraction, text):
        """Move the progress bar on the Run panel."""
        self.run_progress.show_progress(fraction, text)

    def set_markers(self, on):
        """Markers on every computed point of every plot (the lines between them are drawn, not computed).
        The shared setting is changed too, so plots cleared for the next run keep the choice."""
        MARKERS["on"] = on
        for plot in (self.product, self.interfaces, self.handles, self.losses, self.controller_view, self.profile):
            plot.set_markers(on)

    def show_log_list(self, logs):
        """Fill the list of log files."""
        self.logs.show_list(logs)

    def show_log_text(self, name, text):
        """Show one log file in the Logs panel."""
        self.logs.show_text(name, text)

    def open_in_code(self, path, line):
        """Open the file in VS Code at the line (`code -g file:line`)."""
        code = shutil.which("code")
        if code is None:
            self.show_error("VS Code's 'code' command isn't on the PATH.")
            return
        subprocess.Popen([code, "-g", f"{path}:{line or 1}"])

    def run_in_background(self, produce, on_item, on_done, on_error):
        """Walk `produce()` on a QThread; each item, the end, or an error comes back through the
        slots below, on the GUI thread."""
        self._callbacks = (on_item, on_done, on_error)
        self._thread = QtCore.QThread(self)
        self._worker = _Worker(produce)
        self._worker.moveToThread(self._thread)       # the worker's run() now executes on that thread
        self._thread.started.connect(self._worker.run)
        self._worker.item.connect(self._item)
        self._worker.done.connect(self._done)
        self._worker.failed.connect(self._failed)
        self._thread.start()

    # Slots on the window (the GUI thread): the worker's signals arrive here, queued.

    @QtCore.pyqtSlot(object)
    def _item(self, item):
        """One item from the background run: pass it to the presenter's on_item."""
        on_item = self._callbacks[0]
        on_item(item)

    @QtCore.pyqtSlot()
    def _done(self):
        """The background run ended: close the thread, then call the presenter's on_done."""
        self._finish()
        on_done = self._callbacks[1]
        on_done()

    @QtCore.pyqtSlot(str)
    def _failed(self, text):
        """The background run raised: close the thread, then call the presenter's on_error."""
        self._finish()
        on_error = self._callbacks[2]
        on_error(text)

    def _finish(self):
        """Stop the thread's event loop, wait for it to end, and forget it. _thread being None
        is how the rest of the window (and the smoke test) knows nothing is running."""
        self._thread.quit()
        self._thread.wait()
        self._thread = self._worker = None


# ==================================================================================================
# Small helpers
# ==================================================================================================
def _upset_marks(upset_time: float | None, *rows: int) -> list[tuple]:
    """A vertical line at the upset's start on each given plot row, labelled on the first row
    only. No upset (upset_time None) gives no lines."""
    if upset_time is None:
        return []
    return [(row, "v", upset_time, "upset" if row == 0 else "") for row in rows]


def _path_inside_src(path: str) -> str:
    """A file's path from inside src/ (src\\mpc.py -> mpc.py), to keep the code-uses list short."""
    return path.split("src")[-1].lstrip("\\/")

