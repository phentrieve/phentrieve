"""Lazily loaded, process-wide disease index for the disease ranking router."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import api.config as api_config
from phentrieve.disease_matching.index import (
    INDEX_FORMAT_VERSION,
    DiseaseIndex,
    IndexNotAvailableError,
    default_index_dir,
    load_index,
    read_index_info,
)

logger = logging.getLogger(__name__)

# After a failed load, requests fail fast (no re-hash of the arrays) until this delay.
RETRY_AFTER_SECONDS = 60.0


@dataclass
class _IndexState:
    index: DiseaseIndex | None = None
    loaded_path: Path | None = None
    failed_at: float | None = None
    error: IndexNotAvailableError | None = None

    def failure_is_recent(self) -> bool:
        return (
            self.failed_at is not None
            and time.monotonic() - self.failed_at < RETRY_AFTER_SECONDS
        )


_lock = threading.Lock()
_state = _IndexState()


def disease_index_dir() -> Path:
    """PHENTRIEVE_DISEASE_INDEX_DIR, else the data dir as `prepare-diseases` resolves it."""
    if api_config.PHENTRIEVE_DISEASE_INDEX_DIR:
        return Path(api_config.PHENTRIEVE_DISEASE_INDEX_DIR)
    return default_index_dir(None)


def _is_stale(index: DiseaseIndex, loaded_path: Path | None, path: Path) -> bool:
    """True when the index dir changed or the index on disk was rebuilt."""
    if loaded_path is not None and loaded_path != path:
        return True
    info = read_index_info(path)
    return bool(info and info.get("arrays_sha256") != index.meta.get("arrays_sha256"))


def get_disease_index() -> DiseaseIndex:
    """The loaded index, (re)loaded on first use or after a rebuild on disk.

    Raises IndexNotAvailableError when no usable index exists; after a failed load,
    further calls fail fast for RETRY_AFTER_SECONDS.
    """
    try:
        path = disease_index_dir()
    except OSError as error:
        raise IndexNotAvailableError(
            f"Disease index directory is unavailable: {error}"
        ) from error
    with _lock:
        current = _state.index
        loaded_path = _state.loaded_path
    if current is not None and not _is_stale(current, loaded_path, path):
        return current
    with _lock:
        current = _state.index
        loaded_path = _state.loaded_path
        if current is not None and not _is_stale(current, loaded_path, path):
            return current
        if _state.failure_is_recent() and _state.error is not None:
            if current is not None and loaded_path == path:
                return current
            raise _state.error
        logger.info("Loading disease index from %s", path)
        try:
            loaded = load_index(path)
        except IndexNotAvailableError as error:
            _state.failed_at = time.monotonic()
            _state.error = error
            if current is not None and loaded_path == path:
                # A rebuild is in progress or failed: keep serving the previous index.
                logger.warning(
                    "Disease index reload failed, keeping the loaded one: %s", error
                )
                return current
            raise
        _state.index = loaded
        _state.loaded_path = path
        _state.failed_at = None
        _state.error = None
        logger.info("Disease index loaded: %s diseases", loaded.n_diseases)
        return loaded


def disease_ranking_status() -> dict[str, Any]:
    """Feature status for config-info, without loading the index."""
    enabled = api_config.PHENTRIEVE_DISEASE_RANKING_ENABLED
    info: dict[str, Any] | None = None
    with _lock:
        cached_index = _state.index
        cached_path = _state.loaded_path
        recent_failure = _state.failure_is_recent()
    if enabled:
        try:
            path = disease_index_dir()
            info = read_index_info(path)
            if info is None and cached_index is not None and cached_path == path:
                info = cached_index.meta
        except OSError as error:
            logger.warning("Disease index directory unavailable for status: %s", error)
            info = None
    ready = (
        info is not None
        and info.get("format_version") == INDEX_FORMAT_VERSION
        and (cached_index is not None or not recent_failure)
    )
    return {
        "enabled": enabled,
        "index_ready": ready,
        "hpo_version": (info or {}).get("hpo_version"),
        "mondo_version": (info or {}).get("mondo_version"),
        "built_at": (info or {}).get("built_at"),
    }


def reset_disease_index() -> None:
    """Drop the cached index and any recorded failure (tests)."""
    with _lock:
        _state.index = None
        _state.loaded_path = None
        _state.failed_at = None
        _state.error = None
