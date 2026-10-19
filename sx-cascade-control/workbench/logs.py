"""The workbench's log files: one per session, the last 20 kept, also printed to the console.

Each time the workbench starts, a new file logs/workbench_<date>_<time>.log is opened and the
oldest are deleted so at most KEEP remain. Everything the workbench says goes in it: each status
line, each error with its whole traceback, each run started (what was chosen and every edit) and
finished (its scorecard), each setting edited or file saved, Python warnings (an unconverged
steady state, for instance) and any exception nothing else caught. The same lines go to the
console at INFO and above. The Logs panel lists the files and shows one.

No toolkit here: the presenter uses `log`, and the view only shows what the presenter reads.
"""
from __future__ import annotations

import logging
import sys
import threading
from datetime import datetime
from pathlib import Path

from src.store import LOG_DIR, log_files                # where the logs live is the store's to know
KEEP = 20
FORMAT = "%(asctime)s  %(levelname)-7s  %(name)s: %(message)s"
log = logging.getLogger("workbench")


def start_session_log(log_dir: Path = LOG_DIR, keep: int = KEEP, console: bool = True) -> Path:
    """Open this session's log file (and the console copy), delete all but the newest `keep`
    files, and route warnings and uncaught exceptions into the log. Returns the file's path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    _delete_oldest(log_dir, keep - 1)                     # room for the new one
    path = log_dir / f"workbench_{datetime.now():%Y%m%d_%H%M%S}.log"
    _add_handlers(path, console)
    logging.captureWarnings(True)                         # warnings.warn(...) goes to the log too
    for noisy in ("matplotlib", "PIL", "asyncua"):        # their debug lines would bury ours
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _log_uncaught_exceptions()
    log.info("log file %s", path)
    return path


def _delete_oldest(log_dir: Path, keep: int):
    """Delete all but the newest `keep` log files. A file that can't be deleted (open in another
    program, for instance) is left; it'll go next time."""
    for old in log_files(log_dir)[keep:]:
        try:
            old.unlink()
        except OSError:
            pass


def _add_handlers(path: Path, console: bool):
    """Send every log line to the file at `path`, and lines at INFO and above to the console."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(file_handler)
    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(logging.INFO)
        stream.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(stream)


def _log_uncaught_exceptions():
    """Write an exception that nothing caught, on the main thread or any other, to the log with
    its traceback. Python's default only prints it to the console."""
    def uncaught(kind, value, tb):
        log.critical("uncaught exception", exc_info=(kind, value, tb))
    sys.excepthook = uncaught
    threading.excepthook = lambda a: log.critical("uncaught exception in a thread",
                                                  exc_info=(a.exc_type, a.exc_value, a.exc_traceback))
