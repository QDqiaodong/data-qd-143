"""Score drift monitoring and recoverable notification outbox (v4).

The monitor is an independent subsystem with its own SQLite database.  It never opens
or mutates the revision ledger or annotation packages: only ``--db`` paths created by
this package are read or written.  Public entry points:

* :func:`validate_snapshot` -- strict snapshot contract and canonical semantic digest
* :func:`score_calibration.monitor.engine.compute` -- pure exact drift model core
* :func:`connect`, :func:`sync`, :func:`claim`, :func:`ack`, :func:`status_report`
"""
from .contract import (
    MonitorError,
    content_digest,
    frozen_config,
    validate_snapshot,
)
from .engine import compute
from .store import ack, claim, connect, status_report, sync

__all__ = [
    'MonitorError', 'validate_snapshot', 'content_digest', 'frozen_config',
    'compute', 'connect', 'sync', 'claim', 'ack', 'status_report',
]
