"""The workbench: choose a plant, a controller, an upset and a sensor set, edit any setting, run,
and watch the circuit respond.

    pixi run python -m workbench                  # from projects/sx-cascade-control
    pixi run python -m workbench --config ndpr_plant      # another run file

Layers (Passive-View MVP, ARCHITECTURE.md decision 24): src/ is the core and knows nothing of any GUI; presenter.py holds
the interaction logic and knows nothing of Qt; view.py and panels.py are the PyQt shell. This file
only wires them together.

The order of the wiring matters:

    1. make the Qt application (every widget needs one to exist first);
    2. make the view, an empty main window;
    3. make the presenter, giving it the view and the run to start with;
    4. view.bind(presenter): the view builds its panels and connects their buttons to the
       presenter's methods;
    5. presenter.start(): the presenter fills the panels for the first time;
    6. show the window and hand control to Qt's event loop until the window closes.

A web front end would replace steps 1, 2, 4 and 6 with its own; the presenter and src/ would
stay as they are.
"""
from __future__ import annotations

import argparse
import sys

from PyQt5 import QtWidgets

from src.runs import RunSpec

from .logs import start_session_log
from .presenter import WorkbenchPresenter
from .view import QtWorkbenchView


def main(argv=None):
    """Start the workbench and return Qt's exit code when the window closes.

    Args:
        argv: command-line arguments (without the program name). None reads them from sys.argv.
    """
    parser = argparse.ArgumentParser(prog="python -m workbench")
    parser.add_argument("--config", default="ndpr_main", help="run file to start with (configs/)")
    args = parser.parse_args(argv)

    start_session_log()                   # logs/workbench_<date>_<time>.log, the last 20 kept, and the console
    app = QtWidgets.QApplication(sys.argv)
    view = QtWorkbenchView()
    # The workbench opens on the main case's showcase run: a realistic plant, the MPC over level
    # loops, the feed switch, with analysers on the product, the raffinate and the feed.
    presenter = WorkbenchPresenter(view, RunSpec(config=args.config, plant="perturbed", controller="mpc_level_loops",
                                                 upset="feed_switch", sensors="product_raffinate_feed",
                                                 hours=80.0))
    view.bind(presenter)
    presenter.start()
    view.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
