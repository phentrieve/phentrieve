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


def _is_stale(index: DiseaseIndex, path: Path) -> bool:
    """True when the index on disk was rebuilt since this one was loaded."""
    info = read_index_info(path)
    return bool(info and info.get("arrays_sha256") != index.meta.get("arrays_sha256"))


def get_disease_index() -> DiseaseIndex:
    """The loaded index, (re)loaded on first use or after a rebuild on disk.

    Raises IndexNotAvailableError when no usable index exists; after a failed load,
    further calls fail fast for RETRY_AFTER_SECONDS.
    """
    path = disease_index_dir()
    current = _state.index
    if current is not None and not _is_stale(current, path):
        return current
    with _lock:
        current = _state.index
        if current is not None and not _is_stale(current, path):
            return current
        if _state.failure_is_recent() and _state.error is not None:
            if current is not None:
                return current
            raise _state.error
        logger.info("Loading disease index from %s", path)
        try:
            loaded = load_index(path)
        except IndexNotAvailableError as error:
            _state.failed_at = time.monotonic()
            _state.error = error
            if current is not None:
                # A rebuild is in progress or failed: keep serving the previous index.
                logger.warning(
                    "Disease index reload failed, keeping the loaded one: %s", error
                )
                return current
            raise
        _state.index = loaded
        _state.failed_at = None
        _state.error = None
        logger.info("Disease index loaded: %s diseases", loaded.n_diseases)
        return loaded


def disease_ranking_status() -> dict[str, Any]:
    """Feature status for config-info, without loading the index."""
    enabled = api_config.PHENTRIEVE_DISEASE_RANKING_ENABLED
    info: dict[str, Any] | None = None
    if enabled:
        info = read_index_info(disease_index_dir())
        if info is None and _state.index is not None:
            info = _state.index.meta
    ready = (
        info is not None
        and info.get("format_version") == INDEX_FORMAT_VERSION
        and (_state.index is not None or not _state.failure_is_recent())
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
        _state.failed_at = None
        _state.error = None
