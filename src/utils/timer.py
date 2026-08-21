"""
timer.py

Execution timing for pipeline profiling.

Works as a context manager, as a decorator, and as a manual start/stop object.
Every timing is recorded in a module-level registry so a run can end with a
breakdown of where the time actually went.

Usage
-----
    from src.utils.timer import Timer, timed, log_timing_summary

    # context manager
    with Timer("Data Cleaning"):
        clean()

    # decorator
    @timed("Feature Engineering")
    def build_features(df):
        ...

    # end of run
    log_timing_summary()
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import time
import tracemalloc
from collections import defaultdict

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# Output Symbols
# ==========================================================

def _supports_unicode() -> bool:
    """
    Whether the active stdout encoding can represent the status symbols.

    Windows consoles default to cp1252, which cannot encode the emoji used
    here and raises UnicodeEncodeError at write time -- after the work has
    already completed, which is a maddening way to lose a long run.
    """

    if os.environ.get("CYCLONE_ASCII_LOGS"):
        return False

    encoding = getattr(sys.stdout, "encoding", None) or "ascii"

    try:
        "⏱✅❌".encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_UNICODE = _supports_unicode()

SYMBOLS = {
    "start": "⏱ " if _UNICODE else ">>",
    "done": "✅" if _UNICODE else "OK",
    "fail": "❌" if _UNICODE else "!!",
}


# ==========================================================
# Duration Formatting
# ==========================================================

def format_duration(seconds: float) -> str:
    """
    Render a duration at an appropriate scale.

    A fixed "{:.2f}s" shows a 3 ms call as 0.00s and a 90-minute fit as
    5400.00s. Both occur in this pipeline.

    Examples
    --------
    >>> format_duration(0.0034)
    '3.4ms'
    >>> format_duration(75.0)
    '1m 15.0s'
    """

    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f}us"

    if seconds < 1.0:
        return f"{seconds * 1e3:.1f}ms"

    if seconds < 60.0:
        return f"{seconds:.2f}s"

    if seconds < 3600.0:
        minutes, remainder = divmod(seconds, 60)
        return f"{int(minutes)}m {remainder:.1f}s"

    hours, remainder = divmod(seconds, 3600)
    minutes = remainder / 60
    return f"{int(hours)}h {int(minutes)}m"


def format_bytes(n_bytes: float) -> str:
    """
    Render a byte count in the largest sensible unit.
    """

    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024.0:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024.0

    return f"{n_bytes:.1f}TB"


# ==========================================================
# Timing Registry
# ==========================================================

_TIMINGS: dict[str, list[float]] = defaultdict(list)


def record_timing(name: str, seconds: float) -> None:
    """
    Add a measurement to the registry.
    """

    _TIMINGS[name].append(seconds)


def reset_timings() -> None:
    """
    Clear all recorded timings. Call between runs, and in test setup.
    """

    _TIMINGS.clear()


def timing_summary() -> list[dict]:
    """
    Recorded timings, slowest total first.

    Returns
    -------
    list of dict
        name, calls, total_sec, mean_sec, pct_of_total
    """

    grand_total = sum(sum(values) for values in _TIMINGS.values())

    rows = [
        {
            "name": name,
            "calls": len(values),
            "total_sec": sum(values),
            "mean_sec": sum(values) / len(values),
            "pct_of_total": (sum(values) / grand_total * 100) if grand_total else 0.0,
        }
        for name, values in _TIMINGS.items()
    ]

    return sorted(rows, key=lambda row: row["total_sec"], reverse=True)


def log_timing_summary(custom_logger: logging.Logger = None) -> list[dict]:
    """
    Log a profiling breakdown of the run.

    Call this at the end of a pipeline. Knowing that 80% of the wall clock is
    one XGBoost fit -- or, less comfortably, a CSV parse -- is what tells you
    where optimisation is worth the effort.
    """

    active_logger = custom_logger or logger
    rows = timing_summary()

    if not rows:
        active_logger.info("No timings recorded.")
        return rows

    active_logger.info("=" * 68)
    active_logger.info("TIMING SUMMARY")
    active_logger.info("=" * 68)
    active_logger.info(f"  {'Stage':<34} {'Calls':>6} {'Total':>12} {'Share':>8}")
    active_logger.info("  " + "-" * 62)

    for row in rows:
        active_logger.info(
            f"  {row['name'][:34]:<34} {row['calls']:>6} "
            f"{format_duration(row['total_sec']):>12} "
            f"{row['pct_of_total']:>7.1f}%"
        )

    total = sum(row["total_sec"] for row in rows)
    active_logger.info("  " + "-" * 62)
    active_logger.info(f"  {'TOTAL':<34} {'':>6} {format_duration(total):>12}")
    active_logger.info("=" * 68)

    return rows


# ==========================================================
# Timer
# ==========================================================

class Timer:
    """
    Measure the execution time of a block or a function.

    Usable three ways:

        with Timer("Stage"):            # context manager
            ...

        @Timer("Stage")                 # decorator (the original class
        def work():                     # documented this but never
            ...                         # implemented __call__)

        t = Timer("Stage"); t.start()   # manual
        ...
        t.stop()

    Parameters
    ----------
    block_name : str
        Label used in logs and in the timing registry.
    custom_logger : logging.Logger, optional
        Defaults to this module's logger.
    level : int
        Logging level for the start/finish messages. Use logging.DEBUG for
        inner timers you do not want in the normal console output.
    track_memory : bool
        Report peak memory allocated during the block, via tracemalloc.
        Adds noticeable overhead, so it is off by default.
    log_start : bool
        Emit a message on entry. Turn off for fast, frequently-called blocks.
    """

    def __init__(
        self,
        block_name: str = "Block",
        custom_logger: logging.Logger = None,
        level: int = logging.INFO,
        track_memory: bool = False,
        log_start: bool = True,
    ):
        self.block_name = block_name
        self.logger = custom_logger or logger
        self.level = level
        self.track_memory = track_memory
        self.log_start = log_start

        self.start_time: float = 0.0
        self.elapsed_sec: float = 0.0
        self.peak_memory_bytes: int = 0

        self._owns_tracemalloc = False

    # ---- Core ----------------------------------------------------------

    def start(self) -> "Timer":
        """
        Begin timing.
        """

        if self.track_memory and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._owns_tracemalloc = True

        if self.log_start:
            self.logger.log(self.level, f"{SYMBOLS['start']} Starting: {self.block_name}...")

        self.start_time = time.perf_counter()
        return self

    def stop(self, exc_type=None) -> float:
        """
        End timing, log the result, and record it in the registry.

        Returns
        -------
        float
            Elapsed seconds.
        """

        self.elapsed_sec = time.perf_counter() - self.start_time

        if self.track_memory and tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            self.peak_memory_bytes = peak

            if self._owns_tracemalloc:
                tracemalloc.stop()
                self._owns_tracemalloc = False

        record_timing(self.block_name, self.elapsed_sec)

        detail = format_duration(self.elapsed_sec)

        if self.track_memory and self.peak_memory_bytes:
            detail += f", peak {format_bytes(self.peak_memory_bytes)}"

        if exc_type is None:
            self.logger.log(
                self.level,
                f"{SYMBOLS['done']} Completed: {self.block_name} in {detail}",
            )
        else:
            self.logger.error(
                f"{SYMBOLS['fail']} Failed: {self.block_name} after {detail} "
                f"({exc_type.__name__})"
            )

        return self.elapsed_sec

    @property
    def elapsed(self) -> float:
        """
        Seconds elapsed so far, whether or not the timer has stopped.
        """

        if self.start_time and not self.elapsed_sec:
            return time.perf_counter() - self.start_time

        return self.elapsed_sec

    # ---- Context manager ------------------------------------------------

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop(exc_type)
        return False  # never suppress the exception

    # ---- Decorator ------------------------------------------------------

    def __call__(self, func):
        """
        Use the timer as a decorator.

        A fresh Timer is constructed per call so that recursive or concurrent
        invocations cannot overwrite each other's start time.
        """

        name = self.block_name if self.block_name != "Block" else func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(
                name,
                custom_logger=self.logger,
                level=self.level,
                track_memory=self.track_memory,
                log_start=self.log_start,
            ):
                return func(*args, **kwargs)

        return wrapper

    def __repr__(self) -> str:
        return f"Timer(name={self.block_name!r}, elapsed={format_duration(self.elapsed)})"


# ==========================================================
# Decorator Alias
# ==========================================================

def timed(
    name_or_func=None,
    *,
    level: int = logging.INFO,
    track_memory: bool = False,
):
    """
    Decorator form, usable bare or with a label.

        @timed
        def load(): ...

        @timed("Feature Engineering")
        def build(): ...
    """

    if callable(name_or_func):
        return Timer(
            name_or_func.__name__, level=level, track_memory=track_memory
        )(name_or_func)

    return Timer(
        name_or_func or "Block", level=level, track_memory=track_memory
    )