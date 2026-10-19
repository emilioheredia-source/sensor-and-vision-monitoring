"""The workbench's panels: passive PyQt widgets, one per dock.

What this file is
-----------------
Each class here is one panel of the workbench window. A panel does two things:

    - it shows what it's given: a list of options, a dict of settings, a dict of arrays to plot;
    - it forwards what the user does as a plain callback, such as on_run() or
      on_edit(section, path, value).

None of them computes anything about the plant, and none of them knows the presenter exists. They
get their callbacks from view.py, which passes in the presenter's methods. Together with view.py,
this is the part that would be rewritten to move the workbench to another front end
(ARCHITECTURE.md, decision 24).

The panels, in the order they appear below:

    RunPanel       the five choice lists, the run length, Run / Stop, and the status line
    SettingsPanel  every setting of the run as one editable tree
    CircuitPanel   a drawing of the circuit: stages, streams in and out, instruments
    TimePlots      stacked plots against time (product, handles, losses, controller's view)
    InterfacePlot  TimePlots with a band for the spread of interface heights
    ProfilePlot    every stage at the latest instant: concentrations and interface heights
    ScorePanel     the scorecard table
    SavedPanel     the list of saved runs, to overlay on the plots

The plots use pyqtgraph; the settings tree is a pyqtgraph ParameterTree.
"""
from __future__ import annotations

import json
import os
from typing import NamedTuple

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")         # pyqtgraph would otherwise pick PySide6 if present

import numpy as np                                          # noqa: E402
from PyQt5 import QtCore, QtWidgets                         # noqa: E402
import pyqtgraph as pg                                      # noqa: E402
from pyqtgraph.parametertree import Parameter, ParameterTree   # noqa: E402

from src.plotting import COLOR, GRID, INK, INK2, MUTED, SLOT, STATUS_CRITICAL, SURFACE

# The project's plot colours (src/plotting.py) for every pyqtgraph plot.
pg.setConfigOptions(antialias=True, background=SURFACE, foreground=INK2)

# The label for each of the five choice lists on the Run panel. The keys are presenter.KINDS.
KIND_LABELS = {"config": "Run file", "plant": "Plant", "controller": "Controller",
               "upset": "Upset", "sensors": "Sensor set"}

# Saved runs overlaid on the plots are dashed greys, one shade per run, repeating after four.
COMPARE_PENS = [pg.mkPen(c, width=1.2, style=QtCore.Qt.DashLine)
                for c in ("#898781", "#52514e", "#c3c2b7", "#0b0b0b")]


def _pen(color, width=1.8):
    """A solid pen for a data line."""
    return pg.mkPen(color, width=width)


# Markers on the computed points: the lines between them are drawn, not computed. Shared by every
# plot panel; the View menu turns them on and off (QtWorkbenchView.set_markers).
MARKERS = {"on": True}


def _drop_legend(plot):
    """Take a plot's legend off it. PlotItem.clear() removes the curves but leaves the legend, and
    addLegend() then makes a second one."""
    if plot.legend is not None:
        plot.legend.scene() and plot.legend.scene().removeItem(plot.legend)
        plot.legend = None


def _label_fill():
    """A light, mostly opaque background for legends and line labels, so a curve passing behind
    them doesn't make them unreadable."""
    brush = pg.mkBrush(SURFACE)
    color = brush.color()
    color.setAlpha(215)
    return pg.mkBrush(color)


def _show_points(curve, color, on: bool):
    """Put a small marker on each data point of a curve, or take them off."""
    if on:
        curve.setSymbol("o")
        curve.setSymbolSize(4)
        curve.setSymbolBrush(color)
        curve.setSymbolPen(None)
    else:
        curve.setSymbol(None)


# ==================================================================================================
# Choosing and running
# ==================================================================================================
class RunPanel(QtWidgets.QWidget):
    """The five choice lists (run file, plant, controller, upset, instruments), each with an
    explanation under it, in a part that scrolls. Running is done from the toolbar (RunToolBar)
    and followed in the status bar (RunProgress), so those stay in view whatever this panel shows.

    Args:
        on_choose(kind, key): the user picked an option in one of the lists.
    """

    def __init__(self, on_choose):
        """Build the choice lists in a scrolling part."""
        super().__init__()
        self._on_choose = on_choose
        self._combos, self._explanations = {}, {}          # by kind: the list, and the text under it
        choices = QtWidgets.QWidget()
        inner = QtWidgets.QVBoxLayout(choices)
        inner.addLayout(self._build_choice_form())
        inner.addStretch(1)
        scroll = QtWidgets.QScrollArea(widgetResizable=True, frameShape=QtWidgets.QFrame.NoFrame)
        scroll.setWidget(choices)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def _build_choice_form(self) -> QtWidgets.QFormLayout:
        """One combo box per kind of choice, each with a small grey explanation under it."""
        form = QtWidgets.QFormLayout()
        for kind, label in KIND_LABELS.items():
            combo = QtWidgets.QComboBox()
            # `activated` fires only when the user picks, not when show_options fills the list.
            # k=kind fixes the kind for this combo (a plain closure would see the loop's last kind).
            combo.activated.connect(lambda _index, k=kind: self._chosen(k))
            explanation = QtWidgets.QLabel(wordWrap=True)
            explanation.setStyleSheet(f"color: {INK2}; font-size: 9pt;")
            form.addRow(label, combo)
            form.addRow("", explanation)
            self._combos[kind], self._explanations[kind] = combo, explanation
        return form

    def _chosen(self, kind):
        """Pass the picked option's key (stored as the item's data) on to the presenter."""
        self._on_choose(kind, self._combos[kind].currentData())

    def show_options(self, kind, opts, selected):
        """Refill one list with (key, title) pairs and select `selected`.

        Signals are blocked while the list is refilled, so this doesn't look like a user choice.
        """
        combo = self._combos[kind]
        combo.blockSignals(True)
        combo.clear()
        for key, title in opts:
            combo.addItem(title, key)
        combo.setCurrentIndex(max(0, combo.findData(selected)))
        combo.blockSignals(False)

    def show_explanation(self, kind, text):
        """Show the explanation under one list."""
        self._explanations[kind].setText(text)

    def set_running(self, running):
        """The choices are locked while a run goes on."""
        for combo in self._combos.values():
            combo.setEnabled(not running)


class RunToolBar(QtWidgets.QToolBar):
    """The run controls, along the top of the window: Run, Stop, the run length, and "Forget all
    setting edits". A toolbar keeps them in view and out of the panels' way.

    Args:
        on_run(), on_stop(): the Run and Stop buttons.
        on_hours(hours): the user changed the run length.
        on_reset(): "Forget all setting edits".
    """

    def __init__(self, on_run, on_stop, on_hours, on_reset):
        """Build the buttons and the run-length box."""
        super().__init__("Run")
        self.setObjectName("run_toolbar")                  # needed to save and restore the window layout
        self.setMovable(False)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.run_action = self.addAction("  Run  ", on_run)
        self.stop_action = self.addAction("  Stop  ", on_stop)
        self.addSeparator()
        self.addWidget(QtWidgets.QLabel(" Run length "))
        self.hours = QtWidgets.QDoubleSpinBox(minimum=1.0, maximum=500.0, singleStep=5.0, suffix=" h", decimals=1)
        self.hours.valueChanged.connect(on_hours)
        self.addWidget(self.hours)
        self.addSeparator()
        self.addAction("Forget all setting edits", on_reset)
        bold = self.widgetForAction(self.run_action).font()
        bold.setBold(True)
        self.widgetForAction(self.run_action).setFont(bold)

    def show_hours(self, hours):
        """Show the run length without sending it back as a user edit."""
        self.hours.blockSignals(True)
        self.hours.setValue(hours)
        self.hours.blockSignals(False)

    def set_running(self, running):
        """While a run goes on, only Stop is enabled and the run length is locked."""
        self.run_action.setEnabled(not running)
        self.stop_action.setEnabled(running)
        self.hours.setEnabled(not running)


class RunProgress(QtWidgets.QWidget):
    """How the run is going, at the right of the status bar: a progress bar (busy while the
    operating point is prepared, then hours done) and a clock with the time since the last update,
    so a slow run can be told from a stuck one."""

    def __init__(self):
        """Build the bar and the clock; the clock ticks once a second while a run goes on."""
        super().__init__()
        self.progress = QtWidgets.QProgressBar(minimum=0, maximum=1000, textVisible=True)
        self.progress.setFormat("")
        self.progress.setFixedWidth(260)
        self.clock = QtWidgets.QLabel()
        self.clock.setStyleSheet(f"color: {INK2};")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.addWidget(self.clock)
        layout.addWidget(self.progress)
        self._started = self._last_update = None
        self._timer = QtCore.QTimer(self, interval=1000)
        self._timer.timeout.connect(self._tick)

    def show_progress(self, fraction, text):
        """fraction: 0 to 1 through the run, or None while the length of the work isn't known
        (preparing the operating point): the bar then shows a busy pattern."""
        if fraction is None:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(round(1000 * min(max(fraction, 0.0), 1.0))))
        self.progress.setFormat(text)
        self._last_update = QtCore.QElapsedTimer()
        self._last_update.start()
        self._tick()

    def set_running(self, running):
        """Start the clock with a run; stop it, and any busy pattern, when the run ends."""
        if running:
            self._started = QtCore.QElapsedTimer()
            self._started.start()
            self._timer.start()
        else:
            self._timer.stop()
            if self.progress.maximum() == 0:
                self.progress.setRange(0, 1000)
        self._tick()

    def _tick(self):
        """Elapsed time, and the time since the last update; a long silence is pointed out."""
        if self._started is None:
            self.clock.setText("")
            return
        text = f"elapsed {_mm_ss(self._started.elapsed() / 1000)}"
        self.clock.setText(text + self._since_last_update() if self._timer.isActive() else text)

    def _since_last_update(self) -> str:
        """", last update N s ago", with a note once a minute has gone by without news."""
        quiet = self._last_update.elapsed() / 1000 if self._last_update is not None else 0.0
        note = " (commissioning can take a few minutes; much longer may mean it's stuck)" if quiet > 60 else ""
        return f", last update {quiet:.0f} s ago{note}"


def _mm_ss(seconds: float) -> str:
    """Seconds as minutes:seconds, for the clock under the run bar."""
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


# ==================================================================================================
# Every setting, editable
# ==================================================================================================
# The settings come from the presenter as nested dicts of plain values, one dict per section:
#     {"run_file": {...}, "parameters": {...}, "controller_settings": {...}}
# They're shown as a pyqtgraph ParameterTree: a dict becomes a collapsible group, anything else an
# editable value. Each value's Parameter keeps its full path, ("run_file", "flows", "feed") for
# instance, so an edit can be sent back as (section, path, value).

class SettingsPanel(QtWidgets.QWidget):
    """The run file, the parameter file and the controller's settings as one tree. Lists and empty
    values are edited as JSON text. Edited values are marked with a dot.

    Right-click a value for its line in the file, the lines of code that use it, or the file
    opened in VS Code at that line.

    Args:
        on_edit(section, path, value): the user changed a value. `section` is the first key of
            its path, `path` the rest of the keys down to the value.
        on_locate(section, path, how): the user asked where a value is; `how` is "show",
            "uses" or "code".
    """
    SECTION_TITLES = {"run_file": "Run file", "parameters": "Parameters (params/parameters.toml)",
                      "controller_settings": "Controller settings"}

    def __init__(self, on_edit, on_locate=None):
        """Build an empty tree with a note above it. The tree is filled by show_settings."""
        super().__init__()
        self._on_edit = on_edit
        self._on_locate = on_locate
        self._params: dict[tuple, Parameter] = {}          # every value's Parameter, by full path
        self._root = None                                   # the tree's top Parameter, once built
        self._updating = False                              # True while show_settings sets values
        # Two resizable columns with a header: drag the divider to see long names. The name column
        # also fits itself to the longest visible name when the tree is built or a group is opened,
        # and every value has its full path as a tooltip.
        self.tree = ParameterTree(showHeader=True)
        self.tree.setHeaderLabels(["Setting", "Value"])
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
        self.tree.header().setStretchLastSection(True)
        self.tree.itemExpanded.connect(lambda _item: self._fit_name_column())
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel("Edits take effect on the next run. A dot marks an edited value. "
                                "Right-click a value to find it in its file or in the code. "
                                "File > Save settings as a new run file keeps the edits.", wordWrap=True)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        note.setStyleSheet(f"color: {INK2};")
        layout.addWidget(note)
        layout.addWidget(self.tree)

    # The tree is rebuilt only when the set of settings changes (for instance, another run file
    # or controller). Otherwise the values are updated in place, which keeps the groups the user
    # has opened open.

    def show_settings(self, values: dict, edited: list):
        """Show every value and mark the edited ones.

        Args:
            values: {section: nested dict of plain values}.
            edited: full paths of the edited values, each (section, key, key, ...).
        """
        leaves = dict(_leaves(values))                      # full path -> value
        edited = set(edited)
        if self._needs_rebuild(leaves):
            self._rebuild(values)

        # Setting values here mustn't look like the user editing them. Holding the tree's change
        # signal isn't enough: pyqtgraph queues the changes and sends them all when it's released,
        # and each would come back as an edit. So changes are ignored while this runs.
        self._updating = True
        self._root.blockTreeChangeSignal()
        try:
            for path, value in leaves.items():
                param = self._params[path]
                param.setValue(_display(value))
                param.setOpts(title=("● " if path in edited else "") + str(path[-1]))
        finally:
            self._root.unblockTreeChangeSignal()
            self._updating = False

    def _needs_rebuild(self, leaves: dict) -> bool:
        """True when the settings differ from the tree's, or one of them changed type. The type
        matters: a box built as text for an empty setting would hand a number back as text."""
        if set(leaves) != set(self._params):
            return True
        return any(self._params[path].type() != _parameter_type(value) for path, value in leaves.items())

    def _rebuild(self, values: dict):
        """Build the tree again from scratch, and index every value's Parameter by its path."""
        children = [_group(self.SECTION_TITLES.get(section, section), section_values, (section,))
                    for section, section_values in values.items()]
        self._root = Parameter.create(name="settings", type="group", children=children)
        self._params = {}
        for param in _walk(self._root):
            if param.opts.get("path"):                      # groups have no path; values do
                self._params[param.opts["path"]] = param
        self._root.sigTreeStateChanged.connect(self._changed)
        self.tree.setParameters(self._root, showTop=False)
        self._fit_name_column()

    def _fit_name_column(self):
        """Make the name column wide enough for the longest name now showing, but no more than 60 %
        of the panel, so the values stay in view; drag the divider for more."""
        self.tree.resizeColumnToContents(0)
        widest = self.tree.columnWidth(0) + 12
        self.tree.setColumnWidth(0, min(widest, int(0.6 * self.tree.viewport().width())))

    def _context_menu(self, pos):
        """The right-click menu on a value: where it's written, where the code uses it."""
        item = self.tree.itemAt(pos)
        path = getattr(getattr(item, "param", None), "opts", {}).get("path")
        if not path or self._on_locate is None:
            return
        section, rest = path[0], tuple(path[1:])
        menu = QtWidgets.QMenu(self)
        menu.addAction("Show in its file", lambda: self._on_locate(section, rest, "show"))
        menu.addAction(f"Where the code uses '{rest[-1]}'", lambda: self._on_locate(section, rest, "uses"))
        menu.addAction("Open its file in VS Code", lambda: self._on_locate(section, rest, "code"))
        menu.exec_(self.tree.viewport().mapToGlobal(pos))

    def _changed(self, _root, changes):
        """The tree changed. Pass each value the user changed on as (section, path, value)."""
        if self._updating:                                  # the panel setting its own values
            return
        for param, change, value in changes:
            path = param.opts.get("path")
            if change == "value" and path:
                self._on_edit(path[0], tuple(path[1:]), value)


class TextEditorPanel(QtWidgets.QWidget):
    """One file as text: the run file or parameters.toml to edit, or a source file to read.
    Save checks the file loads before writing it (the presenter does that); Reload throws the
    changes away.

    Args:
        on_save(path, text): the user pressed Save.
        on_reload(path): the user pressed Reload.
        on_open_in_code(path, line): the user pressed Open in VS Code.
    """

    def __init__(self, on_save, on_reload, on_open_in_code):
        """Build the file name label, the text box and the button row. Empty until show_text."""
        super().__init__()
        self._path = None                                   # the file shown; None until show_text
        self.label = QtWidgets.QLabel("File > Edit ... as text, or right-click a setting > Show in its file.",
                                      wordWrap=True)
        self.label.setStyleSheet(f"color: {INK2};")
        self.text = QtWidgets.QPlainTextEdit()
        _monospace(self.text)
        self.line = QtWidgets.QLabel("")
        self.text.cursorPositionChanged.connect(lambda: self.line.setText(f"line {self._cursor_line()}"))
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.text)
        layout.addLayout(self._build_buttons(on_save, on_reload, on_open_in_code))

    def _build_buttons(self, on_save, on_reload, on_open_in_code) -> QtWidgets.QHBoxLayout:
        """Save, Reload and Open in VS Code, with the cursor's line number on the right. The
        buttons do nothing while no file is shown."""
        self.save_button = QtWidgets.QPushButton("Save")
        reload_button = QtWidgets.QPushButton("Reload")
        code_button = QtWidgets.QPushButton("Open in VS Code")
        self.save_button.clicked.connect(lambda: self._path and on_save(self._path, self.text.toPlainText()))
        reload_button.clicked.connect(lambda: self._path and on_reload(self._path))
        code_button.clicked.connect(lambda: self._path and on_open_in_code(self._path, self._cursor_line()))
        buttons = QtWidgets.QHBoxLayout()
        for widget in (self.save_button, reload_button, code_button):
            buttons.addWidget(widget)
        buttons.addStretch()
        buttons.addWidget(self.line)
        return buttons

    def show_text(self, path: str, text: str, line, editable: bool):
        """Show the file, put the cursor on `line` (1-based) and mark that line."""
        self._path = path
        self.label.setText(path + ("" if editable else "   (read only: source code)"))
        self.text.setPlainText(text)
        self.text.setReadOnly(not editable)
        self.save_button.setEnabled(editable)
        if line:
            self._mark_line(line)
        else:
            self.text.setExtraSelections([])

    def _mark_line(self, line: int):
        """Move the cursor to the start of `line`, scroll it to the middle, and tint the line."""
        block = self.text.document().findBlockByNumber(line - 1)
        cursor = self.text.textCursor()
        cursor.setPosition(block.position())
        self.text.setTextCursor(cursor)
        self.text.centerCursor()
        mark = QtWidgets.QTextEdit.ExtraSelection()
        mark.format.setBackground(pg.mkColor(SLOT[1]).lighter(170))
        mark.format.setProperty(mark.format.FullWidthSelection, True)
        mark.cursor = cursor
        self.text.setExtraSelections([mark])

    def _cursor_line(self) -> int:
        """The line the cursor is on, counted from 1 as editors do."""
        return self.text.textCursor().blockNumber() + 1


class ReportPanel(QtWidgets.QTextBrowser):
    """A run's report (src/report.py), shown formatted: what was run, what happened, the numbers
    with what each means, and what's worth a second look. Written at the end of every run; older
    runs' reports open from the Saved runs panel."""

    def __init__(self):
        """Start with a note on where reports come from."""
        super().__init__()
        self.setOpenExternalLinks(False)
        self.setMarkdown("No run yet. The report appears here when a run finishes; "
                         "for an older run, pick it in Saved runs and press Show report.")

    def show_report(self, text: str):
        """Show a report, given as Markdown text."""
        self.setMarkdown(text)


class LogPanel(QtWidgets.QWidget):
    """The session logs: the list on top (newest first), the chosen one's text below, read only.

    Args:
        on_refresh(): the user pressed Refresh.
        on_open(name): the user picked a log in the list.
    """

    def __init__(self, on_refresh, on_open):
        """Build the list of logs with a Refresh button beside it, and the text box below."""
        super().__init__()
        self.files = QtWidgets.QComboBox()
        self.text = QtWidgets.QPlainTextEdit(readOnly=True)
        _monospace(self.text)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(on_refresh)
        self.files.activated.connect(lambda i: on_open(self.files.itemData(i)))
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.files, 1)
        top.addWidget(refresh)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.text)

    def show_list(self, logs):
        """Refill the list with (file name, label) pairs. The name is kept as the item's data."""
        self.files.clear()
        for name, label in logs:
            self.files.addItem(label, name)

    def show_text(self, name, text):
        """Show one log's text, scrolled to the end, and select it in the list."""
        i = self.files.findData(name)
        if i >= 0:
            self.files.setCurrentIndex(i)
        self.text.setPlainText(text)
        self.text.moveCursor(self.text.textCursor().End)          # the newest lines in view


def _monospace(text_box: QtWidgets.QPlainTextEdit):
    """Show a text box's contents in a fixed-width font without line wrapping, so files and logs
    keep their columns."""
    text_box.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
    font = text_box.font()
    font.setFamily("Consolas")
    font.setStyleHint(font.Monospace)
    text_box.setFont(font)


# ---- turning nested dicts into a parameter tree ------------------------------------------------------
def _leaves(nested, prefix=()):
    """Yield (full path, value) for every value in a nested dict. An empty dict counts as a value
    (it's shown as "{}" text), so it can be edited."""
    for key, value in nested.items():
        if isinstance(value, dict) and value:
            yield from _leaves(value, prefix + (key,))
        else:
            yield prefix + (key,), value


def _group(title, value, path):
    """The ParameterTree description of one setting: a group if it's a non-empty dict, a single
    editable value otherwise."""
    if isinstance(value, dict) and value:
        return {"name": str(path[-1]), "title": title, "type": "group", "expanded": False,
                "children": [_group(str(key), child, path + (key,)) for key, child in value.items()]}
    return _value_entry(title, value, path)


def _value_entry(title, value, path):
    """The ParameterTree description of one editable value. Its full path is kept in the options
    so an edit can be sent back to the presenter."""
    kind = _parameter_type(value)
    entry = {"name": str(path[-1]), "title": title, "type": kind, "value": _display(value),
             "default": _display(value), "path": path, "tip": ".".join(str(p) for p in path)}
    if kind == "float":
        entry.update(decimals=8, dec=True, step=0.05)       # dec=True: the step scales with the value
    return entry


def _parameter_type(value) -> str:
    """The ParameterTree type for a value. bool is checked before int (a bool is an int in Python).
    Anything else (text, a list, None) is edited as text."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _display(value):
    """A value as the tree shows it: lists and dicts as JSON text, None as empty text."""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return "" if value is None else value


def _walk(param):
    """Yield a Parameter and every Parameter under it."""
    yield param
    for child in param.children():
        yield from _walk(child)


# ==================================================================================================
# The circuit, drawn
# ==================================================================================================
# The drawing uses plot coordinates. Along x, stage i is a box from i + 0.08 to i + 0.92, so its
# centre is at i + 0.5; stage 0 is the raffinate end, on the left. Along y, the boxes run from 0
# to 1. Streams coming in (feed, strip acid) are arrows from above, streams going out (raffinate,
# product) arrows below. Text rows sit at fixed heights:
#
#      2.25  stream-in labels          1.4   section names        -0.45  "stages 0 to N-1"
#     -1.0   stream-out labels (hanging below their arrows)       -2.05  flow directions
# The instrument legend is the plot's title, above the drawing, so it can't run into the labels.

class _ArrowTrack(NamedTuple):
    """The heights of a stream arrow: it runs from y_start to y_end (head at y_end), and its
    label sits at label_y."""
    y_start: float
    y_end: float
    label_y: float


STREAM_IN = _ArrowTrack(2.0, 1.05, 2.25)       # from above, ending just over the box
STREAM_OUT = _ArrowTrack(-0.05, -0.95, -1.0)   # from just under the box, downwards

# How each kind of instrument is marked on the stages: (symbol, colour, legend label).
# Interface gauges sit just above the box, the rest in its middle.
SENSOR_MARKS = {"composition_aq": ("t", SLOT[4], "analyser"), "interface": ("s", SLOT[3], "interface level"),
                "product_assay": ("d", STATUS_CRITICAL, "lab assay")}


class CircuitPanel(pg.GraphicsLayoutWidget):
    """A drawing of the circuit: the stages by section, the streams in and out, and where each
    instrument of the sensor set sits. Redrawn whenever the presenter sends a new layout."""
    SECTION_TINT = {"extraction": "#dce9f9", "scrub": "#f2f1ec", "strip": "#fde7dc"}

    def __init__(self):
        """One plot with no axes, used as a canvas. Drawn by show_layout."""
        super().__init__()
        self.plot = self.addPlot()
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setMouseEnabled(False, False)              # a fixed drawing: no pan or zoom

    def show_layout(self, lay: dict):
        """Draw the circuit from scratch.

        Args:
            lay: presenter._layout(spec): stage counts ("extraction", "scrub", "strip", "stages"),
                "feed_stage", "product_stage", "reflux", "sensors", "full_state", and the stream
                names "product_name" and "raffinate_name".
        """
        plot = self.plot
        plot.clear()
        stage_count = lay["stages"]
        self._draw_sections(plot, lay)
        _text(plot, f"stages 0 to {stage_count - 1}", (stage_count / 2, -0.45), MUTED)
        _text(plot, "← aqueous flows this way", (stage_count * 0.3, -2.05), INK2)
        _text(plot, "organic flows this way →", (stage_count * 0.62, -2.05), INK2)
        self._draw_streams(plot, lay)
        self._draw_sensors(plot, lay)
        plot.setXRange(-1, stage_count + 1, padding=0)
        plot.setYRange(-2.5, 2.6, padding=0)

    def _draw_sections(self, plot, lay):
        """One tinted box per stage, and each section's name and stage count above it."""
        first_scrub = lay["extraction"]
        first_strip = lay["extraction"] + lay["scrub"]
        sections = [("extraction", 0, lay["extraction"]), ("scrub", first_scrub, lay["scrub"]),
                    ("strip", first_strip, lay["strip"])]
        for name, first_stage, count in sections:
            boxes = pg.BarGraphItem(x0=np.arange(first_stage, first_stage + count) + 0.08, width=0.84,
                                    y0=0, height=1.0, brush=self.SECTION_TINT[name], pen=pg.mkPen(GRID))
            plot.addItem(boxes)
            _text(plot, f"{name} ({count})", (first_stage + count / 2, 1.4), INK2)

    def _draw_streams(self, plot, lay):
        """Feed and strip acid in from above; raffinate and product out below.

        The feed enters the last extraction stage and the strip acid the last stage. A diamond
        on the feed label means the sensor set has a feed analyser.
        """
        stage_count = lay["stages"]
        feed_analysed = any(s["kind"] == "feed_analyser" for s in lay["sensors"])
        feed_label = "feed" + (" ◆ analysed" if feed_analysed else "")
        reflux_note = ", part returned as scrub" if lay["reflux"] else ""
        product_label = f"{lay['product_name']} product" + reflux_note
        _arrow(plot, lay["feed_stage"] + 0.5, feed_label, STREAM_IN)
        _arrow(plot, stage_count - 0.5, "strip acid", STREAM_IN)
        _arrow(plot, 0.5, f"raffinate ({lay['raffinate_name']})", STREAM_OUT, anchor=(0, 0.5))
        # right-aligned at the product stage, so it reaches left under the strip, not into the text row
        _arrow(plot, lay["product_stage"] + 0.5, product_label, STREAM_OUT, anchor=(1, 0.5))

    def _draw_sensors(self, plot, lay):
        """Mark each instrument on its stage and add a legend for the kinds shown."""
        if lay["full_state"]:
            plot.setTitle("no instruments: the controller sees the full state", color=MUTED, size="9pt")
            return
        shown = _draw_sensor_marks(plot, lay["sensors"])
        legend = "&nbsp;&nbsp;&nbsp;".join(f'<span style="color:{color}">■ {label}</span>' for label, color in sorted(shown))
        plot.setTitle(legend, size="9pt")


def _draw_sensor_marks(plot, sensors) -> set:
    """Draw one marker per instrument. Returns the (legend label, colour) pairs used.

    The feed analyser isn't drawn here: it's marked on the feed arrow's label instead. A kind
    without an entry in SENSOR_MARKS gets a plain circle, labelled with its kind.
    """
    shown = set()
    for sensor in sensors:
        if sensor["kind"] == "feed_analyser":
            continue
        symbol, color, label = SENSOR_MARKS.get(sensor["kind"], ("o", INK2, sensor["kind"]))
        x = sensor["stage"] + 0.5                                   # the middle of its stage's box
        y = 1.15 if sensor["kind"] == "interface" else 0.5
        marker = pg.ScatterPlotItem([x], [y], symbol=symbol, size=11, brush=color, pen=None)
        plot.addItem(marker)
        shown.add((label, color))
    return shown


def _text(plot, text, position, color, anchor=(0.5, 0.5)):
    """Put text on the plot at position (x, y). The anchor (0.5, 0.5) centres it there; (0, 0.5)
    starts it there."""
    x, y = position
    item = pg.TextItem(text, color=color, anchor=anchor)
    item.setPos(x, y)
    plot.addItem(item)


def _arrow(plot, x, label, track: _ArrowTrack, anchor=(0.5, 0.5)):
    """A stream at x: a line from track.y_start to track.y_end with the head at y_end, and its
    label at track.label_y. A left-anchored label starts half a stage to the left of the arrow."""
    plot.plot([x, x], [track.y_start, track.y_end], pen=_pen(INK, 1.2))
    pointing_down = track.y_end < track.y_start
    plot.addItem(pg.ArrowItem(pos=(x, track.y_end), angle=90 if pointing_down else -90, headLen=10,
                              brush=INK, pen=None))
    label_x = x + {0: -0.5, 1: 0.5}.get(anchor[0], 0.0)     # left- or right-aligned labels start at the stage edge
    # a label under a downward arrow hangs from its top edge, so the arrow never runs through it
    label_anchor = (anchor[0], 0.0) if pointing_down else anchor
    _text(plot, label, (label_x, track.label_y), INK, anchor=label_anchor)


# ==================================================================================================
# Time series
# ==================================================================================================
# A TimePlots panel is a stack of plots that share the time axis: zoom or pan one in time and the
# others follow. Each plot shows one quantity, so there's never a second y scale to read.
#
# A run goes through three steps on this panel:
#     begin(refs)       clear the curves and draw the reference lines (spec, target, upset time)
#     update(series)    redraw every curve from the series so far (once per interval)
#     compare(runs)     overlay saved runs as dashed lines (any time)
# After update and compare, each plot's y range is set to cover its data and its reference levels.

class TimePlots(pg.GraphicsLayoutWidget):
    """Stacked plots sharing the time axis, one quantity per plot (never two y-scales).

    Args:
        rows: one entry per plot, top to bottom: (title, y label, [(series key, label, colour)]).
            The series keys are names in the dict passed to update().

    Example:
        plots = TimePlots([("Recovery", "fraction", [("recovery", "plant", "#2a78d6")])])
        plots.begin([(0, "h", 0.95, "target")])
        plots.update({"t": np.array([0.0, 0.5]), "recovery": np.array([0.97, 0.96])})
    """

    def __init__(self, rows):
        """Build one plot per row, each with an empty curve per series."""
        super().__init__()
        self._rows = rows
        self._plots = []                  # one pyqtgraph PlotItem per row
        self._curves = {}                 # series key -> its curve
        self._compare_items = []          # (plot, item) pairs of the overlaid saved runs
        self._legend_rows = set()           # rows with a legend: they get headroom above the data
        for row_index, row in enumerate(rows):
            self._plots.append(self._add_row(row_index, row))
        self._refs = []                   # (plot, line) pairs of the reference lines
        self._levels = {}                 # row -> its horizontal reference levels, for the y range

    def _add_row(self, row_index, row):
        """Add one plot with its curves. Only the bottom plot has a time label; every plot below
        the first follows the first one's time axis."""
        title, ylabel, lines = row
        plot = self.addPlot(row=row_index, col=0, title=title)
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("left", ylabel)
        if row_index == len(self._rows) - 1:
            plot.setLabel("bottom", "time", units="h")
        if row_index:
            plot.setXLink(self._plots[0])
        self._add_curves(plot, lines)
        return plot

    def _add_curves(self, plot, lines):
        """An empty curve for each series of one plot, with a legend if there's more than one."""
        if len(lines) > 1:
            # one row, on a light background, in the headroom _fit_y leaves above the data
            plot.addLegend(offset=(10, 5), labelTextColor=INK2, brush=_label_fill(), colCount=len(lines))
            self._legend_rows.add(len(self._plots))
        for key, label, color in lines:
            self._curves[key] = plot.plot([], [], pen=_pen(color), name=label)
            _show_points(self._curves[key], color, MARKERS["on"])

    def begin(self, refs: list):
        """Clear the curves and draw a new run's reference lines.

        Args:
            refs: (row, "h" or "v", value, label) for each line: spec, targets, window, upset time.
                Horizontal ones stay in view; their labels sit at different places along the line.
        """
        self._remove(self._refs)
        self._refs, self._levels = [], {}
        for curve in self._curves.values():
            curve.setData([], [])
        for ref in refs:
            self._add_reference_line(ref)

    def _add_reference_line(self, ref):
        """Draw one dotted reference line. Labels of horizontal lines on the same plot are spread
        along the line (from the right) so they don't sit on top of each other."""
        row, orient, value, label = ref
        if orient == "h":
            self._levels.setdefault(row, []).append(value)
        lines_on_row = len(self._levels.get(row, []))
        # vertical labels (the upset) at the foot of the line, where the curves rarely are
        label_position = 0.06 if orient == "v" else 0.96 - 0.22 * (lines_on_row - 1)
        line = pg.InfiniteLine(pos=value, angle=0 if orient == "h" else 90,
                               pen=pg.mkPen(MUTED, width=1, style=QtCore.Qt.DotLine),
                               label=label, labelOpts={"color": MUTED, "position": label_position,
                                                       "fill": _label_fill()})
        self._plots[row].addItem(line)
        self._refs.append((self._plots[row], line))

    def _fit_y(self):
        """Each plot's y range covers its data and its reference levels."""
        for row, plot in enumerate(self._plots):
            y_range = self._y_range(row, plot)
            if y_range is not None:
                plot.setYRange(*y_range, padding=0)

    def _y_range(self, row, plot):
        """(low, high) covering one plot's data and reference levels, with room for its legend on
        top if it has one; None while the plot has no data."""
        values = [v for v in list(self._levels.get(row, [])) + _data_extremes(plot) if np.isfinite(v)]
        if not values:
            return None
        low, high = _padded_range(values)
        headroom = 0.3 * (high - low) if row in self._legend_rows else 0.0   # the legend sits above the curves
        return low, high + headroom

    @staticmethod
    def _remove(pairs):
        """Take each (plot, item) pair's item off its plot."""
        for plot, item in pairs:
            plot.removeItem(item)

    def update(self, series: dict):
        """Redraw every curve whose key is in `series`, against series["t"]. Does nothing
        without a "t"."""
        t = series.get("t")
        if t is None:
            return
        for key, curve in self._curves.items():
            if key in series:
                curve.setData(t, series[key])
        self._fit_y()

    def set_markers(self, on: bool):
        """A marker on every computed point (each control interval), or lines only."""
        for curve in self._curves.values():
            _show_points(curve, curve.opts["pen"].color(), on)

    def compare(self, runs: list):
        """Overlay saved runs as dashed lines, replacing any earlier overlay.

        Args:
            runs: [{"label": str, "series": dict like update()}]. An empty list clears the overlay.
        """
        self._remove(self._compare_items)
        self._compare_items = []
        for run_index, run in enumerate(runs):
            pen = COMPARE_PENS[run_index % len(COMPARE_PENS)]
            self._overlay(run["series"], pen)
        self._fit_y()

    def _overlay(self, series: dict, pen):
        """Draw one saved run's series on whichever plots show them."""
        for row, key in self._series_keys_by_row():
            if key in series:
                item = self._plots[row].plot(series["t"], series[key], pen=pen)
                self._compare_items.append((self._plots[row], item))

    def _series_keys_by_row(self) -> list[tuple[int, str]]:
        """(row, series key) for every curve, top plot first."""
        return [(row, key) for row, (_, _, lines) in enumerate(self._rows) for key, _, _ in lines]


def _data_extremes(plot) -> list[float]:
    """The lowest and highest finite y of every data item on a plot, as one flat list."""
    extremes = []
    for item in plot.listDataItems():
        y = item.getData()[1]
        y = np.asarray([] if y is None else y, float)
        y = y[np.isfinite(y)]
        if len(y):
            extremes += [float(y.min()), float(y.max())]
    return extremes


def _padded_range(values: list[float]) -> tuple[float, float]:
    """The range of `values` with 8 % added at each end. A flat range gets a small fixed margin
    (1 % of the value, at least 1e-6) so the plot doesn't collapse to a line."""
    low, high = min(values), max(values)
    pad = 0.08 * (high - low) if high > low else max(abs(high) * 0.01, 1e-6)
    return low - pad, high + pad


class InterfacePlot(TimePlots):
    """Every interface as a band (lowest to highest) and the mean, against the window."""

    def __init__(self):
        """One plot: the mean as a line, with a shaded band from the lowest to the highest."""
        super().__init__([("Settler interfaces (all stages)", "height, m",
                           [("interfaces_mean", "mean", SLOT[0])])])
        # Two invisible curves for the edges of the band, and the fill between them.
        self._band_lo = self._plots[0].plot([], [], pen=None)
        self._band_hi = self._plots[0].plot([], [], pen=None)
        fill = pg.FillBetweenItem(self._band_lo, self._band_hi, brush=pg.mkBrush(42, 120, 214, 50))
        self._plots[0].addItem(fill)

    def update(self, series):
        """Redraw the mean line, then the band from the lowest and highest interface."""
        super().update(series)
        if "t" in series:
            self._band_lo.setData(series["t"], series["interfaces_min"])
            self._band_hi.setData(series["t"], series["interfaces_max"])


# ==================================================================================================
# Every stage, now
# ==================================================================================================
class ProfilePlot(pg.GraphicsLayoutWidget):
    """Every stage at the latest instant: aqueous concentration of each component (log scale) and
    the interface heights. The separation front, moving."""

    def __init__(self):
        """Two plots sharing the stage axis: concentrations on top, interface heights below."""
        super().__init__()
        self.conc = self.addPlot(row=0, col=0, title="Aqueous concentration by stage")
        self.conc.setLogMode(y=True)
        self.conc.setLabel("left", "mol/L")
        self.conc.showGrid(x=True, y=True, alpha=0.25)
        self.iface = self.addPlot(row=1, col=0, title="Interface height by stage")
        self.iface.setLabel("left", "m")
        self.iface.setLabel("bottom", "stage (0 = raffinate end)")
        self.iface.setXLink(self.conc)
        self._curves = {}                 # component -> its concentration curve
        self._bars = None                 # the interface bars now drawn
        self._title = None

    def begin(self, components: list, window, all_components: list | None = None):
        """Clear both plots for a new run.

        Args:
            components: the components to draw, one curve each.
            window: (low, high) interface heights, drawn as dotted lines on the lower plot.
            all_components: every component, in the column order of the profile arrays. Defaults
                to `components`.
        """
        self.conc.clear()
        self.iface.clear()
        _drop_legend(self.conc)                          # clear() leaves the old legend in place
        # bottom-left: on the log scale that corner is empty (low concentrations, raffinate end)
        self.conc.addLegend(offset=(10, -10), labelTextColor=INK2, brush=_label_fill(), colCount=len(components))
        self._columns = list(all_components or components)       # column order of the profile arrays
        self._curves = {c: self.conc.plot([], [], pen=_pen(COLOR.get(c, INK2)), name=c) for c in components}
        self.set_markers(MARKERS["on"])
        for value in window:
            window_edge = pg.InfiniteLine(pos=value, angle=0, pen=pg.mkPen(MUTED, style=QtCore.Qt.DotLine))
            self.iface.addItem(window_edge)
        self._bars = None

    def set_markers(self, on: bool):
        """A marker on each stage (the computed points), or lines only."""
        for curve in getattr(self, "_curves", {}).values():
            _show_points(curve, curve.opts["pen"].color(), on)

    def show(self, t, profile):
        """Draw the stages at time t.

        Args:
            t: hours since the start of the run, shown in the title.
            profile: {"aqueous": array (stages x components, mol/L), "interfaces": array (stages, m)}.
        """
        stages = np.arange(profile["aqueous"].shape[0])
        for name, curve in self._curves.items():
            column = self._columns.index(name)
            # Floor at 1e-8 mol/L: the log axis can't show zero.
            curve.setData(stages, np.maximum(profile["aqueous"][:, column], 1e-8))
        if self._bars is not None:
            self.iface.removeItem(self._bars)
        self._bars = pg.BarGraphItem(x=stages, height=profile["interfaces"], width=0.7, brush=SLOT[0])
        self.iface.addItem(self._bars)
        self.conc.setTitle(f"Aqueous concentration by stage, t = {t:.1f} h")


# ==================================================================================================
# Results
# ==================================================================================================
class ScorePanel(QtWidgets.QTableWidget):
    """The scorecard: one row per measure, one column per run. Filled by the presenter."""

    def show_scorecard(self, columns, rows):
        """Fill the table.

        Args:
            columns: one label per run.
            rows: (row label, one formatted value per column) for each row.
        """
        self.clear()
        self.setColumnCount(len(columns))
        self.setRowCount(len(rows))
        self.setHorizontalHeaderLabels([_wrapped_header(column) for column in columns])
        self.setVerticalHeaderLabels([label for label, _ in rows])
        for row_index, (_, values) in enumerate(rows):
            for column_index, text in enumerate(values):
                self.setItem(row_index, column_index, _number_cell(text))
        self.resizeColumnsToContents()


def _wrapped_header(column: str) -> str:
    """A column label over several lines, broken at " / " and after ": ", so columns stay narrow."""
    return column.replace(" / ", "\n").replace(": ", ":\n")


def _number_cell(text: str) -> QtWidgets.QTableWidgetItem:
    """A table cell with its text right-aligned, as numbers are."""
    item = QtWidgets.QTableWidgetItem(text)
    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    return item


class SavedPanel(QtWidgets.QWidget):
    """The saved runs as a checkable list, with buttons to overlay the checked ones, clear the
    overlay, and read the list again.

    Args:
        on_compare(run_ids): overlay these runs (an empty list clears the overlay).
        on_refresh(): list the saved runs again.
        on_report(run_id): show the report of the run selected in the list.
    """

    def __init__(self, on_compare, on_refresh, on_report=None):
        """Build the list with a row of four buttons under it."""
        super().__init__()
        self._on_report = on_report
        self.list = QtWidgets.QListWidget()
        compare_button = QtWidgets.QPushButton("Overlay checked runs")
        clear_button = QtWidgets.QPushButton("Clear overlay")
        report_button = QtWidgets.QPushButton("Show report")
        refresh_button = QtWidgets.QPushButton("Refresh")
        compare_button.clicked.connect(lambda: on_compare(self.checked()))
        clear_button.clicked.connect(lambda: on_compare([]))
        report_button.clicked.connect(self._report_selected)
        refresh_button.clicked.connect(on_refresh)
        buttons = QtWidgets.QHBoxLayout()
        for button in (compare_button, clear_button, report_button, refresh_button):
            buttons.addWidget(button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.list)
        layout.addLayout(buttons)

    def _report_selected(self):
        """Ask for the report of the run selected in the list (the highlighted one, not the
        checked ones). Does nothing if no run is selected."""
        item = self.list.currentItem()
        if self._on_report is not None and item is not None:
            self._on_report(item.data(QtCore.Qt.UserRole))

    def show_saved_runs(self, runs):
        """Refill the list with (run id, label) pairs. Runs that were checked stay checked."""
        checked = set(self.checked())
        self.list.clear()
        for run_id, label in runs:
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, run_id)             # the run id travels with the item
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if run_id in checked else QtCore.Qt.Unchecked)
            self.list.addItem(item)

    def checked(self) -> list:
        """The run ids of the checked runs, top to bottom."""
        items = [self.list.item(i) for i in range(self.list.count())]
        return [item.data(QtCore.Qt.UserRole) for item in items if item.checkState() == QtCore.Qt.Checked]
