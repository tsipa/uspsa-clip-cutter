"""Temporary Windows idle-sleep prevention during a processing run."""

from __future__ import annotations

import ctypes
import logging
import sys
from contextlib import contextmanager

log = logging.getLogger(__name__)
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def prevent_sleep():
    """Keep Windows awake on this thread, restoring its prior state on exit."""
    if sys.platform != "win32":
        yield
        return
    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        set_state = kernel.SetThreadExecutionState
        set_state.argtypes = [ctypes.c_uint32]
        set_state.restype = ctypes.c_uint32
        previous = set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except OSError as exc:
        log.warning("Could not prevent Windows idle sleep: %s", exc)
        yield
        return
    if not previous:
        log.warning("Windows rejected idle-sleep prevention; processing will continue")
        yield
        return
    log.info("Preventing Windows idle sleep while processing")
    try:
        yield
    finally:
        try:
            if not set_state(previous | ES_CONTINUOUS):
                log.warning("Could not restore Windows execution state; it is released when the process exits")
        except OSError as exc:
            log.warning("Could not restore Windows execution state: %s", exc)
