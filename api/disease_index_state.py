"""Lazily loaded, process-wide disease index for the disease ranking router."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import api.config as api_config
from phentrieve.disease_matching.index import (
    INDEX_DIRNAME,
    INDEX_FORMAT_VERSION,
    DiseaseIndex,
    IndexNotAvailableError,
    load_index,
    read_index_info,
)

logger = logging.getLogger(__name__)

# After a failed load, requests fail fast (no re-hash of the arrays) until this delay.
RETRY_AFTER_SECONDS = 60.0

_lock = threading.Lock()
_index: DiseaseIndex | None = None
_failure: tuple[float, IndexNotAvailableError] | None = None


def disease_index_dir() -> Path:
    """PHENTRIEVE_DISEASE_INDEX_DIR, else <data dir>/disease_index."""
    if api_config.PHENTRIEVE_DISEASE_INDEX_DIR:
        return Path(api_config.PHENTRIEVE_DISEASE_INDEX_DIR)
    from phentrieve.utils import get_default_data_dir

    return get_default_data_dir() / INDEX_DIRNAME


def get_disease_index() -> DiseaseIndex:
    """Load the index on first use; raises IndexNotAvailableError when it is missing."""
    global _index, _failure
    if _index is not None:
        return _index
    with _lock:
        if _index is None:
            if (
                _failure is not None
                and time.monotonic() - _failure[0] < RETRY_AFTER_SECONDS
            ):
                raise _failure[1]
            path = disease_index_dir()
            logger.info("Loading disease index from %s", path)
            try:
                _index = load_index(path)
            except IndexNotAvailableError as error:
                _failure = (time.monotonic(), error)
                raise
            _failure = None
            logger.info("Disease index loaded: %s diseases", _index.n_diseases)
    return _index


def disease_ranking_status() -> dict[str, Any]:
    """Feature status for config-info, without loading the index."""
    enabled = api_config.PHENTRIEVE_DISEASE_RANKING_ENABLED
    info: dict[str, Any] | None = None
    if enabled:
        info = (
            _index.meta if _index is not None else read_index_info(disease_index_dir())
        )
    ready = (
        info is not None
        and info.get("format_version") == INDEX_FORMAT_VERSION
        and (_index is not None or _failure is None)
    )
    return {
        "enabled": enabled,
        "index_ready": ready,
        "hpo_version": (info or {}).get("hpo_version"),
        "mondo_version": (info or {}).get("mondo_version"),
        "built_at": (info or {}).get("built_at"),
    }


def reset_disease_index() -> None:
    """Drop the cached index (tests, or after a rebuild)."""
    global _index, _failure
    with _lock:
        _index = None
        _failure = None
