"""Rolling in-memory cache of recent prediction inputs.

Drives the admin drift dashboard: every successful inference appends its
feature vector here; /admin/drift snapshots the buffer and computes PSI per
feature against the model's reference quantiles.

In-memory is fine for single-instance free-tier Render. Phase 4 step 4.13 in
the plan calls for Redis once we scale past one worker — the API is
intentionally narrow so swapping out the backing store is mechanical.
"""

from collections import deque
from threading import Lock
from typing import Dict, List, Optional


_LOCK = Lock()
_BUFFER: "deque[Dict[str, float]]" = deque(maxlen=500)


def record(features: Dict[str, float]) -> None:
    """Append one feature dict to the rolling buffer. Drops oldest at capacity."""
    with _LOCK:
        _BUFFER.append(features)


def snapshot() -> List[Dict[str, float]]:
    """Return a copy of the current buffer for read-only analysis."""
    with _LOCK:
        return list(_BUFFER)


def size() -> int:
    with _LOCK:
        return len(_BUFFER)


def clear() -> None:
    """Reset the buffer. Used by tests."""
    with _LOCK:
        _BUFFER.clear()


def capacity() -> Optional[int]:
    return _BUFFER.maxlen
