"""Minimal sqlite3 KV cache keyed by request-signature hash, so reruns don't spend API quota."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class Cache:
    def __init__(self, path: Path | None, ttl_days: float = 30):
        self.path = path
        self.ttl = ttl_days * 86400
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(path), check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, t REAL)")

    def clear(self) -> None:
        """Empty cached responses and reclaim space without unlinking a live SQLite database."""
        with self._lock:
            if self._db is not None:
                self._db.execute("DELETE FROM kv")
                self._db.commit()
                self._db.execute("VACUUM")

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    @staticmethod
    def key(*parts: Any) -> str:
        raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, k: str) -> Any | None:
        if self._db is None:
            return None
        with self._lock:
            row = self._db.execute("SELECT v, t FROM kv WHERE k=?", (k,)).fetchone()
        if row is None or time.time() - row[1] > self.ttl:
            return None
        return json.loads(row[0])

    def set(self, k: str, v: Any) -> None:
        if self._db is None:
            return
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO kv VALUES (?,?,?)", (k, json.dumps(v, ensure_ascii=False), time.time())
            )
            self._db.commit()


NO_CACHE = Cache(None)
