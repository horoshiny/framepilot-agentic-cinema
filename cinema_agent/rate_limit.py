from collections import defaultdict, deque
from threading import RLock
from time import monotonic


class VertexRateLimiter:
    """Limit successful Vertex-backed app requests within one process."""

    def __init__(
        self,
        *,
        per_client_limit: int = 3,
        global_limit: int = 25,
        window_seconds: float = 3600,
        clock=monotonic,
    ):
        self.per_client_limit = per_client_limit
        self.global_limit = global_limit
        self.window_seconds = window_seconds
        self._clock = clock
        self._lock = RLock()
        self._client_successes = defaultdict(deque)
        self._global_successes = deque()

    def _purge_expired(self, now: float):
        cutoff = now - self.window_seconds
        while self._global_successes and self._global_successes[0] <= cutoff:
            self._global_successes.popleft()
        for client_ip, successes in list(self._client_successes.items()):
            while successes and successes[0] <= cutoff:
                successes.popleft()
            if not successes:
                del self._client_successes[client_ip]

    def run(self, client_ip: str, operation):
        """Run and record one successful operation, or return blocked status.

        The lock intentionally covers the operation. Vertex calls are serialized
        within this process so concurrent requests cannot pass the same limit
        check before either one records its success.
        """

        with self._lock:
            now = self._clock()
            self._purge_expired(now)
            client_successes = self._client_successes[client_ip]
            if (
                len(client_successes) >= self.per_client_limit
                or len(self._global_successes) >= self.global_limit
            ):
                return False, None

            result = operation()
            success_time = self._clock()
            self._global_successes.append(success_time)
            client_successes.append(success_time)
            return True, result