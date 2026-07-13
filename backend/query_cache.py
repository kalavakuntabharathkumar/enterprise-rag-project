"""In-memory retrieval cache for query embeddings.

Caches the full retrieve_with_scores result keyed on (query, k) so that
an identical query string never triggers a second OpenAI embedding call or
FAISS search within the same process lifetime.

Limitations
-----------
- In-memory only: the cache is lost on every process restart. This is
  intentional — embeddings are cheap to regenerate across restarts, and
  persistence would require serialising LangChain Document objects, which
  adds significant complexity for minimal gain.
- Not thread-safe for the stats counters on CPython beyond GIL protection.
  For production multi-worker deployments, replace with a Redis-backed
  cache (e.g. via redis-py + pickle) and an atomic counter.
- maxsize controls memory growth. Each entry stores top_k * 3 Document
  objects plus float scores; at a few KB each, 256 entries ≈ a few MB.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any

_DEFAULT_MAXSIZE = 256


class RetrievalCache:
    """LRU dict cache with hit/miss counters.

    Keys are (query: str, k: int) tuples; values are whatever
    retrieve_with_scores returns (list of (Document, float)).
    """

    def __init__(self, maxsize: int = _DEFAULT_MAXSIZE) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict[tuple, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, query: str, k: int) -> tuple[bool, Any]:
        """Return (hit, value).  On a hit the entry is promoted to MRU."""
        key = (query, k)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._hits += 1
                return True, self._store[key]
            self._misses += 1
            return False, None

    def set(self, query: str, k: int, value: Any) -> None:
        """Insert or update an entry, evicting the LRU entry when full."""
        key = (query, k)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self._maxsize:
                self._store.popitem(last=False)  # evict LRU

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return round(self._hits / total, 4) if total else 0.0

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        return {
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "cache_hit_rate": self.hit_rate,
            "cache_size": self.size,
            "cache_maxsize": self._maxsize,
        }


# Module-level singleton shared across the whole process.
retrieval_cache = RetrievalCache()
