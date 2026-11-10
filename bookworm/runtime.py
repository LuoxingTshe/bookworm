"""Locked run lifecycle. Previous runs are retained, including partial results."""
from __future__ import annotations

import fcntl
import json
import warnings
from pathlib import Path

from .config import Settings
from .errors import BusyError, InputError
from .models import Corpus
from .sources.cache import Cache

MARKER = '.bookworm-run.json'


class RetrievalRun:
    """Serialize retrievals sharing a run root; failures never prune earlier results.

    Every output receives a status marker. No run directory is deleted implicitly.
    """
    def __init__(self, settings: Settings, directory: Path, *, kind: str,
                 keep_history: bool = False, keep_cache: bool = False):
        self.root = settings.scib_corpora_dir.resolve()
        self.directory = directory.resolve()
        self.kind = kind
        self.shared_cache = settings.scib_cache_dir / 'http.sqlite'
        self.keep_history = True  # Never delete earlier research runs implicitly.
        self.keep_cache = settings.scib_keep_cache or keep_cache
        self.caches: list[Cache] = []
        self.clients: list = []
        self.finished = False
        self.lock = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / '.retrieval.lock').open('a')
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise BusyError('Another retrieval is using this results directory; wait for it to finish.') from None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            if self.kind in {'build', 'agent'} and any(self.directory.iterdir()):
                raise InputError('Retrieval output must be a new, empty directory; previous results are preserved.')
            self._state('running')
        except BaseException:
            self._unlock()
            raise
        return self

    def cache(self, cache: Cache) -> Cache:
        if all(c is not cache for c in self.caches):
            self.caches.append(cache)
        return cache

    def client(self, client):
        self.clients.append(client)
        return client

    def _state(self, status: str, **extra):
        if self.directory.exists():
            path = self.directory / MARKER
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps({'owner': 'bookworm', 'status': status, 'kind': self.kind, **extra},
                                       ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            temp.replace(path)

    def complete(self, corpus_paths: list[Path]) -> dict:
        # Validate all branches before marking the complete result or removing anything.
        if not corpus_paths:
            raise ValueError('No final corpus supplied; previous results are preserved.')
        for path in corpus_paths:
            if not path.resolve().is_relative_to(self.directory):
                raise ValueError('Final corpus must be inside this run directory.')
            Corpus.model_validate_json(path.read_text(encoding='utf-8'))
        report = {'kept': str(self.directory), 'removed_runs': [], 'cleared_caches': [], 'errors': [],
                  'keep_history': self.keep_history, 'keep_cache': self.keep_cache}
        self._state('completed', corpora=[str(path.resolve().relative_to(self.directory)) for path in corpus_paths])
        self.finished = True
        if not self.keep_cache:
            try:
                if (self.shared_cache.exists()
                        and not any(c.path and c.path.resolve() == self.shared_cache.resolve() for c in self.caches)):
                    self.cache(Cache(self.shared_cache))
            except Exception as exc:
                report['errors'].append(f'Shared cache cleanup: {type(exc).__name__}: {exc}')
            for cache in self.caches:
                try:
                    cache.clear()
                    if cache.path is not None:
                        report['cleared_caches'].append(str(cache.path))
                except Exception as exc:
                    report['errors'].append(f'Cache cleanup: {type(exc).__name__}: {exc}')
        (self.directory / 'cleanup.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        for error in report['errors']:
            warnings.warn(f'Result saved, cleanup incomplete: {error}', RuntimeWarning, stacklevel=2)
        return report

    def _unlock(self):
        if self.lock is not None:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None

    def __exit__(self, exc_type, exc, tb):
        try:
            for client in self.clients:
                client.close()
            for cache in self.caches:
                cache.close()
            if not self.finished:
                self._state('interrupted' if exc_type is KeyboardInterrupt else 'incomplete')
        finally:
            self._unlock()
