"""Lazily loaded, process-wide disease index for the disease ranking router."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import api.config as api_config
from phentrieve.disease_matching.index import (
    INDEX_DIRNAME,
    DiseaseIndex,
    load_index,
    read_index_info,
)

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_index: DiseaseIndex | None = None


def disease_index_dir() -> Path:
    """PHENTRIEVE_DISEASE_INDEX_DIR, else <data dir>/disease_index."""
    if api_config.PHENTRIEVE_DISEASE_INDEX_DIR:
        return Path(api_config.PHENTRIEVE_DISEASE_INDEX_DIR)
    from phentrieve.utils import get_default_data_dir

    return get_default_data_dir() / INDEX_DIRNAME


def get_disease_index() -> DiseaseIndex:
    """Load the index on first use; raises IndexNotAvailableError when it is missing."""
    global _index
    if _index is not None:
        return _index
    with _lock:
        if _index is None:
            path = disease_index_dir()
            logger.info("Loading disease index from %s", path)
            _index = load_index(path)
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
    return {
        "enabled": enabled,
        "index_ready": info is not None,
        "hpo_version": (info or {}).get("hpo_version"),
        "mondo_version": (info or {}).get("mondo_version"),
        "built_at": (info or {}).get("built_at"),
    }


def reset_disease_index() -> None:
    """Drop the cached index (tests, or after a rebuild)."""
    global _index
    with _lock:
        _index = None
