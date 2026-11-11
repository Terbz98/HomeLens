"""A small in-memory request limiter: at most N requests per client per minute."""
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict


class RateLimiter:
    def __init__(self, max_requests: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.max_requests = max_requests
        self.window_s = window_s
        self._clock = clock
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client_key: str) -> bool:
        """Record one request. Returns False when the client is over the limit."""
        now = self._clock()
        with self._lock:
            hits = self._hits[client_key]
            while hits and now - hits[0] > self.window_s:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            if len(self._hits) > 5000:  # do not grow without bound
                for key in [k for k, v in self._hits.items() if not v]:
                    del self._hits[key]
            return True
