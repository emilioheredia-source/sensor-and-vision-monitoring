"""The workbench GUI (presenter + PyQt view) on top of the core in src/. See __main__.py.

The package is split in the Passive-View MVP way (ARCHITECTURE.md, decision 24):

    presenter.py   the interaction logic; no Qt. Talks to the screen only through the
                   WorkbenchView interface, in plain Python values.
    view.py        the PyQt main window: implements WorkbenchView, runs a run on a QThread.
    panels.py      the PyQt panels the window is made of.
    __main__.py    wires the three together and starts the app.

Everything that computes a number about the plant is in src/, which knows nothing of any GUI.
"""
