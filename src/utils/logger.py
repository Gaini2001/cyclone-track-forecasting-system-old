"""
logger.py

Centralized logging for the Cyclone Track Forecasting project.

Design
------
Handlers are attached once, to a single package-level logger. Module loggers
are plain `logging.getLogger(__name__)` objects that propagate upward and own
no handlers of their own.

This matters for three reasons:

1. pytest's `caplog` captures by attaching a handler to the root logger. A
   module logger with `propagate = False` is invisible to it, so log assertions
   silently never fire.
2. Adding a file handler, changing the format, or silencing output becomes a
   single call instead of an edit to every module.
3. One handler writes to the console instead of one per module.

Usage
-----
    # at module scope, anywhere
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

    # once, at a CLI entry point
    from src.utils.logger import configure_logging
    configure_logging(level="DEBUG", log_file=True)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from src.utils.config import LOG_LEVEL, REPORT_DIR

# ==========================================================
# Constants
# ==========================================================

PACKAGE_LOGGER_NAME = "src"

LOG_DIR = REPORT_DIR / "logs"

CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-38s | %(message)s"
CONSOLE_DATEFMT = "%H:%M:%S"

# Full timestamp in files: a saved log is read days later, and "14:02:11" with
# no date is ambiguous the moment you have two runs.
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-38s | %(funcName)s:%(lineno)d | %(message)s"
FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"

VALID_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# Libraries that flood the output once the level drops to DEBUG.
NOISY_LIBRARIES = (
    "matplotlib",
    "matplotlib.font_manager",
    "PIL",
    "urllib3",
    "optuna",
    "numexpr",
    "fsspec",
)

_configured = False


# ==========================================================
# Level Resolution
# ==========================================================

def _resolve_level(level: str | int) -> int:
    """
    Convert a level name to its numeric value, failing loudly on nonsense.

    The previous implementation used `getattr(logging, LOG_LEVEL, logging.INFO)`.
    That returns the *function* `logging.warning` for a lowercase "warning",
    and `setLevel()` then raises an opaque TypeError.
    """

    if isinstance(level, int):
        return level

    normalized = str(level).strip().upper()

    if normalized not in VALID_LEVELS:
        raise ValueError(
            f"Invalid log level {level!r}. Expected one of {VALID_LEVELS}."
        )

    return getattr(logging, normalized)


# ==========================================================
# Handler Construction
# ==========================================================

def _build_console_handler(level: int) -> logging.Handler:
    """
    Console handler with UTF-8 forced on.

    Windows consoles default to cp1252, which raises UnicodeEncodeError on the
    emoji used in timer.py. Reconfiguring the stream (Python 3.7+) makes those
    characters safe everywhere; `errors="replace"` is a final backstop for
    streams that cannot be reconfigured at all.
    """

    stream = sys.stdout

    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass  # not a reconfigurable TextIOWrapper (pytest capture, pipes)

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt=CONSOLE_DATEFMT))

    return handler


def _build_file_handler(path: Path, level: int) -> logging.Handler:
    """
    File handler writing a full-detail log of the run.

    Always records at DEBUG or the requested level, whichever is more verbose,
    so the saved log is more useful than what scrolled past on the console.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(min(level, logging.DEBUG) if level <= logging.INFO else level)
    handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=FILE_DATEFMT))

    return handler


def _default_log_path() -> Path:
    """
    Timestamped log path for the current run.
    """

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"run_{stamp}.log"


# ==========================================================
# Configuration
# ==========================================================

def configure_logging(
    level: str | int = None,
    log_file: bool | str | Path = False,
    capture_warnings: bool = True,
    quiet_libraries: bool = True,
    force: bool = False,
) -> logging.Logger:
    """
    Configure project logging. Idempotent unless `force=True`.

    Parameters
    ----------
    level : str or int, optional
        Console verbosity. Defaults to `LOG_LEVEL` from config, which honours
        the CYCLONE_LOG_LEVEL environment variable.
    log_file : bool, str, or Path
        `True` writes a timestamped log under reports/logs/. A str/Path writes
        to that exact location. `False` disables file logging.
    capture_warnings : bool
        Route `warnings.warn` (pandas SettingWithCopy, sklearn deprecations)
        into the log instead of letting them print to stderr unrecorded.
    quiet_libraries : bool
        Hold third-party loggers at WARNING so DEBUG runs stay readable.
    force : bool
        Tear down existing handlers and reconfigure.

    Returns
    -------
    logging.Logger
        The configured package logger.
    """

    global _configured

    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    if _configured and not force:
        return package_logger

    resolved = _resolve_level(level if level is not None else LOG_LEVEL)

    # Remove prior handlers so repeated configuration cannot duplicate output.
    for handler in list(package_logger.handlers):
        package_logger.removeHandler(handler)
        handler.close()

    package_logger.setLevel(logging.DEBUG)  # handlers do the real filtering
    package_logger.addHandler(_build_console_handler(resolved))

    # Propagate to root so pytest's caplog and any external configuration can
    # see these records. Set to False only if root duplicates output for you.
    package_logger.propagate = True

    if log_file:
        path = _default_log_path() if log_file is True else Path(log_file)
        package_logger.addHandler(_build_file_handler(path, resolved))
        package_logger.info(f"Logging to file: {path}")

    if quiet_libraries:
        for name in NOISY_LIBRARIES:
            logging.getLogger(name).setLevel(logging.WARNING)

    if capture_warnings:
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")

        if not warnings_logger.handlers:
            warnings_logger.addHandler(_build_console_handler(logging.WARNING))

    _configured = True

    return package_logger


# ==========================================================
# Logger Access
# ==========================================================

def get_logger(name: str = None) -> logging.Logger:
    """
    Return a module logger.

    The logger owns no handlers -- records propagate to the package logger
    configured by `configure_logging`. If configuration has not happened yet,
    it is performed with defaults so that importing a module and calling it
    directly still produces output.

    Parameters
    ----------
    name : str
        Typically `__name__`.
    """

    if not _configured:
        configure_logging()

    return logging.getLogger(name or PACKAGE_LOGGER_NAME)


def set_level(level: str | int) -> None:
    """
    Change console verbosity at runtime without reconfiguring.
    """

    resolved = _resolve_level(level)

    for handler in logging.getLogger(PACKAGE_LOGGER_NAME).handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(resolved)


def silence() -> None:
    """
    Suppress all project logging. Useful inside tests.
    """

    logging.getLogger(PACKAGE_LOGGER_NAME).setLevel(logging.CRITICAL + 1)


# ==========================================================
# Environment Reporting
# ==========================================================

def log_environment(logger: logging.Logger = None) -> None:
    """
    Log the versions of the libraries that affect results.

    Worth calling once per training run: "which pandas produced these numbers"
    is the first question when a result cannot be reproduced six months later.
    """

    logger = logger or get_logger(__name__)

    logger.info("-" * 60)
    logger.info("ENVIRONMENT")
    logger.info("-" * 60)
    logger.info(f"  Python   : {sys.version.split()[0]}")
    logger.info(f"  Platform : {sys.platform}")

    for module_name in ("numpy", "pandas", "sklearn", "xgboost", "optuna"):
        try:
            module = __import__(module_name)
            logger.info(f"  {module_name:<9}: {getattr(module, '__version__', 'unknown')}")
        except ImportError:
            logger.info(f"  {module_name:<9}: not installed")

    seed = os.environ.get("PYTHONHASHSEED")

    if seed:
        logger.info(f"  PYTHONHASHSEED: {seed}")

    logger.info("-" * 60)