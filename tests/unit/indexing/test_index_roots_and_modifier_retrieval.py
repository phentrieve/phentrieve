"""Unit tests for index_roots metadata, filtering, compatibility, and modifier retrieval."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chromadb
import numpy as np
import pytest

from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.indexing import chromadb_indexer
from phentrieve.indexing.chromadb_indexer import (
    build_chromadb_index,
    format_index_roots_metadata,
    parse_index_roots_metadata,
)
from phentrieve.indexing.chromadb_orchestrator import orchestrate_index_building
from phentrieve.llm.tools import ToolExecutor
from phentrieve.retrieval.dense_retriever import (
    CHROMA_SUBTREE_GET_BATCH_SIZE,
    DenseRetriever,
    IndexRootsMismatchError,
    RetrievalError,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Test helpers
# =============================================================================


class _FixedEmbeddings:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._array = np.asarray(vectors, dtype=np.float32)

    def tolist(self) -> list[list[float]]:
        return self._array.tolist()

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self._array, dtype=dtype)


class _StubModel:
    """Deterministic embedding model mapping phrases to fixed 3D vectors."""

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self._mapping = mapping or {}
        self.encode_calls: int = 0

    def parameters(self):
        return iter([SimpleNamespace(device=SimpleNamespace(type="cpu"))])

    def encode(self, texts: list[str], device: str = "cpu") -> np.ndarray:
        assert device == "cpu"
        self.encode_calls += 1
        vectors = [self._mapping.get(text, [1.0, 0.0, 0.0]) for text in texts]
        return np.asarray(vectors, dtype=np.float32)


class _InMemoryCollection:
    """Minimal ChromaDB-compatible collection double for indexer tests."""

    def __init__(self, metadata: dict[str, Any], count: int = 0) -> None:
        self.metadata = metadata
        self._count = count

    def add(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        assert len(documents) == len(embeddings) == len(metadatas) == len(ids)
        self._count += len(ids)

    def count(self) -> int:
        return self._count


class _InMemoryClient:
    def __init__(self, collection: _InMemoryCollection | None = None) -> None:
        self.collection = collection
        self.closed = False

    def get_collection(self, name: str) -> _InMemoryCollection:
        if self.collection is None:
            raise KeyError(name)
        return self.collection

    def delete_collection(self, name: str) -> None:
        self.collection = None

    def create_collection(
        self, name: str, metadata: dict[str, Any]
    ) -> _InMemoryCollection:
        self.collection = _InMemoryCollection(metadata)
        return self.collection

    def close(self) -> None:
        self.closed = True


def _build_temp_hpo_db(db_path: Path) -> None:
    """Populate a small synthetic HPO database across three subtrees."""
    with HPODatabase(db_path) as db:
        db.initialize_schema()
        db.set_metadata("hpo_version", "v2099-01-01")
        db.set_metadata("hpo_source_sha256", "a" * 64)
        db.bulk_insert_terms(
            [
                {
                    "id": "HP:0000001",
                    "label": "All",
                    "definition": "Root node.",
                    "synonyms": json.dumps([]),
                    "comments": json.dumps([]),
                },
                {
                    "id": "HP:0000118",
                    "label": "Phenotypic abnormality",
                    "definition": "Abnormality root.",
                    "synonyms": json.dumps([]),
                    "comments": json.dumps([]),
                },
                {
                    "id": "HP:0099001",
                    "label": "Synthetic tremor finding",
                    "definition": "Invented motor finding.",
                    "synonyms": json.dumps(["Invented shaking"]),
                    "comments": json.dumps([]),
                },
                {
                    "id": "HP:0012823",
                    "label": "Clinical modifier",
                    "definition": "Modifier root.",
                    "synonyms": json.dumps([]),
                    "comments": json.dumps([]),
                },
                {
                    "id": "HP:0099101",
                    "label": "Synthetic episodic course",
                    "definition": "Invented course modifier.",
                    "synonyms": json.dumps(["Intermittent pattern"]),
                    "comments": json.dumps([]),
                },
                {
                    "id": "HP:0000005",
                    "label": "Mode of inheritance",
                    "definition": "Inheritance root.",
                    "synonyms": json.dumps([]),
                    "comments": json.dumps([]),
                },
                {
                    "id": "HP:0099201",
                    "label": "Synthetic recessive pattern",
                    "definition": "Invented inheritance term.",
                    "synonyms": json.dumps([]),
                    "comments": json.dumps([]),
                },
            ]
        )
        db.bulk_insert_graph_metadata(
            [
                {
                    "term_id": "HP:0000001",
                    "depth": 0,
                    "ancestors": json.dumps([]),
                },
                {
                    "term_id": "HP:0000118",
                    "depth": 1,
                    "ancestors": json.dumps(["HP:0000001"]),
                },
                {
                    "term_id": "HP:0099001",
                    "depth": 2,
                    "ancestors": json.dumps(["HP:0000118", "HP:0000001"]),
                },
                {
                    "term_id": "HP:0012823",
                    "depth": 1,
                    "ancestors": json.dumps(["HP:0000001"]),
                },
                {
                    "term_id": "HP:0099101",
                    "depth": 2,
                    "ancestors": json.dumps(["HP:0012823", "HP:0000001"]),
                },
                {
                    "term_id": "HP:0000005",
                    "depth": 1,
                    "ancestors": json.dumps(["HP:0000001"]),
                },
                {
                    "term_id": "HP:0099201",
                    "depth": 2,
                    "ancestors": json.dumps(["HP:0000005", "HP:0000001"]),
                },
            ]
        )


# =============================================================================
# 1. Metadata formatting and parsing
# =============================================================================


class TestIndexRootsMetadataHelpers:
    def test_format_empty_and_none(self):
        assert format_index_roots_metadata(None) == ""
        assert format_index_roots_metadata([]) == ""
        assert format_index_roots_metadata(()) == ""
        assert format_index_roots_metadata(["", "   "]) == ""

    def test_format_deduplicates_strips_and_sorts(self):
        formatted = format_index_roots_metadata(
            ["HP:0012823", " HP:0000118 ", "HP:0012823", ""]
        )
        assert formatted == "HP:0000118,HP:0012823"

    def test_parse_empty_and_none(self):
        assert parse_index_roots_metadata(None) == ()
        assert parse_index_roots_metadata("") == ()
        assert parse_index_roots_metadata(" ,  ") == ()

    def test_parse_deduplicates_strips_and_sorts(self):
        parsed = parse_index_roots_metadata(" HP:0012823 , HP:0000118,HP:0012823 ")
        assert parsed == ("HP:0000118", "HP:0012823")


# =============================================================================
# 2. Indexer metadata storage and mismatch detection
# =============================================================================


class TestBuildChromadbIndexRoots:
    def test_stores_index_roots_metadata_on_create(self, monkeypatch, tmp_path):
        client = _InMemoryClient()
        monkeypatch.setattr(
            chromadb_indexer.chromadb, "PersistentClient", lambda **_: client
        )

        ok = build_chromadb_index(
            documents=["Synthetic tremor", "Synthetic course"],
            metadatas=[{"hpo_id": "HP:0099001"}, {"hpo_id": "HP:0099101"}],
            ids=["HP:0099001", "HP:0099101"],
            model=_StubModel(),  # type: ignore[arg-type]
            model_name="unit/model",
            recreate=True,
            index_dir=tmp_path,
            hpo_version="v2099-01-01",
            hpo_source_sha256="a" * 64,
            index_roots=["HP:0012823", "HP:0000118"],
        )

        assert ok is True
        assert client.collection is not None
        assert client.collection.metadata["index_roots"] == "HP:0000118,HP:0012823"

    def test_reuses_matching_existing_collection_and_rejects_mismatch(
        self, monkeypatch, tmp_path
    ):
        existing = _InMemoryCollection(
            metadata={
                "hpo_version": "v2099-01-01",
                "hpo_source_sha256": "a" * 64,
                "model": "unit/model",
                "model_revision": "unpinned",
                "index_type": "single_vector",
                "expected_document_count": 2,
                "index_roots": "HP:0000118,HP:0012823",
            },
            count=2,
        )
        client = _InMemoryClient(collection=existing)
        monkeypatch.setattr(
            chromadb_indexer.chromadb, "PersistentClient", lambda **_: client
        )

        # Same roots in different order -> match and reuse
        ok_match = build_chromadb_index(
            documents=["doc1", "doc2"],
            metadatas=[{"hpo_id": "HP:0099001"}, {"hpo_id": "HP:0099101"}],
            ids=["HP:0099001", "HP:0099101"],
            model=_StubModel(),  # type: ignore[arg-type]
            model_name="unit/model",
            recreate=False,
            index_dir=tmp_path,
            hpo_version="v2099-01-01",
            hpo_source_sha256="a" * 64,
            index_roots=["HP:0012823", "HP:0000118"],
        )
        assert ok_match is True

        # Different roots -> reject mismatch
        ok_mismatch = build_chromadb_index(
            documents=["doc1", "doc2"],
            metadatas=[{"hpo_id": "HP:0099001"}, {"hpo_id": "HP:0099101"}],
            ids=["HP:0099001", "HP:0099101"],
            model=_StubModel(),  # type: ignore[arg-type]
            model_name="unit/model",
            recreate=False,
            index_dir=tmp_path,
            hpo_version="v2099-01-01",
            hpo_source_sha256="a" * 64,
            index_roots=["HP:0000118"],
        )
        assert ok_mismatch is False

    def test_legacy_collection_missing_index_roots_key(self, monkeypatch, tmp_path):
        legacy = _InMemoryCollection(
            metadata={
                "hpo_version": "v2099-01-01",
                "hpo_source_sha256": "a" * 64,
                "model": "unit/model",
                "model_revision": "unpinned",
                "index_type": "single_vector",
                "expected_document_count": 2,
            },
            count=2,
        )
        client = _InMemoryClient(collection=legacy)
        monkeypatch.setattr(
            chromadb_indexer.chromadb, "PersistentClient", lambda **_: client
        )

        # Unrestricted build (empty/None) matches legacy collection cleanly
        ok_unrestricted = build_chromadb_index(
            documents=["doc1", "doc2"],
            metadatas=[{"hpo_id": "HP:0099001"}, {"hpo_id": "HP:0099101"}],
            ids=["HP:0099001", "HP:0099101"],
            model=_StubModel(),  # type: ignore[arg-type]
            model_name="unit/model",
            recreate=False,
            index_dir=tmp_path,
            hpo_version="v2099-01-01",
            hpo_source_sha256="a" * 64,
            index_roots=(),
        )
        assert ok_unrestricted is True

        # Restricted build rejects legacy unrestricted collection
        ok_restricted = build_chromadb_index(
            documents=["doc1", "doc2"],
            metadatas=[{"hpo_id": "HP:0099001"}, {"hpo_id": "HP:0099101"}],
            ids=["HP:0099001", "HP:0099101"],
            model=_StubModel(),  # type: ignore[arg-type]
            model_name="unit/model",
            recreate=False,
            index_dir=tmp_path,
            hpo_version="v2099-01-01",
            hpo_source_sha256="a" * 64,
            index_roots=["HP:0000118"],
        )
        assert ok_restricted is False


# =============================================================================
# 3. Orchestrator subtree filtering and CLI option parsing
# =============================================================================


class TestOrchestratorAndCliIndexRoots:
    def test_orchestrator_filters_terms_by_index_roots(self, mocker, tmp_path, caplog):
        db_path = tmp_path / "hpo_data.db"
        _build_temp_hpo_db(db_path)

        mocker.patch(
            "phentrieve.indexing.chromadb_orchestrator.load_embedding_model",
            return_value=_StubModel(),
        )
        build_mock = mocker.patch(
            "phentrieve.indexing.chromadb_orchestrator.build_chromadb_index",
            return_value=True,
        )

        with caplog.at_level(logging.INFO):
            ok = orchestrate_index_building(
                model_name_arg="unit/model",
                data_dir_override=str(tmp_path),
                index_dir_override=str(tmp_path / "indexes"),
                index_roots=["HP:0012823", "HP:0000118"],
            )

        assert ok is True
        build_mock.assert_called_once()
        kwargs = build_mock.call_args.kwargs
        assert kwargs["index_roots"] == ("HP:0000118", "HP:0012823")
        indexed_ids = set(kwargs["ids"])
        # Kept: roots HP:0000118 & HP:0012823 plus their descendants HP:0099001 & HP:0099101
        assert indexed_ids == {
            "HP:0000118",
            "HP:0099001",
            "HP:0012823",
            "HP:0099101",
        }
        # Excluded: HP:0000001, HP:0000005, HP:0099201
        assert "HP:0000005" not in indexed_ids
        assert "HP:0099201" not in indexed_ids
        assert "kept 4 of 7 terms" in caplog.text

    def test_cli_build_index_normalizes_repeatable_and_csv_index_roots(self, mocker):
        from phentrieve.cli.index_commands import build_index

        mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.indexing.chromadb_orchestrator.orchestrate_index_building",
            return_value=True,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        build_index(index_roots=["HP:0012823, HP:0000118", "HP:0012823"])

        mock_orchestrate.assert_called_once()
        assert mock_orchestrate.call_args.kwargs["index_roots"] == [
            "HP:0000118",
            "HP:0012823",
        ]


# =============================================================================
# 4. DenseRetriever index_roots compatibility checks
# =============================================================================


class TestDenseRetrieverIndexRootsCompatibility:
    def _make_retriever(self, metadata: dict[str, Any] | None) -> DenseRetriever:
        collection = SimpleNamespace(metadata=metadata)
        return DenseRetriever(_StubModel(), collection)  # type: ignore[arg-type]

    def test_compatible_unrestricted_index(self):
        retriever = self._make_retriever({})
        assert retriever.get_index_roots() == ()

        warnings = retriever.check_index_roots_compatibility(
            configured_index_roots=(),
            required_roots=("HP:0000118", "HP:0012823"),
            strict=True,
        )
        assert warnings == []

    def test_compatible_restricted_index_with_ancestor_coverage(self):
        retriever = self._make_retriever({"index_roots": "HP:0000118,HP:0012823"})
        assert retriever.get_index_roots() == ("HP:0000118", "HP:0012823")

        warnings = retriever.check_index_roots_compatibility(
            configured_index_roots=("HP:0012823", "HP:0000118"),
            required_roots=("HP:0012824",),
            ancestors_map={"HP:0012824": frozenset({"HP:0012823", "HP:0000001"})},
            strict=True,
        )
        assert warnings == []

    def test_warning_mode_and_strict_mode_on_configured_mismatch(self, caplog):
        retriever = self._make_retriever({"index_roots": "HP:0000118"})

        with caplog.at_level(logging.WARNING):
            warnings = retriever.check_index_roots_compatibility(
                configured_index_roots=("HP:0000118", "HP:0012823"),
                strict=False,
            )

        assert len(warnings) == 1
        assert "HP:0000118" in warnings[0]
        assert "HP:0000118,HP:0012823" in warnings[0]
        assert "Index roots mismatch" in caplog.text

        with pytest.raises(IndexRootsMismatchError) as exc_info:
            retriever.check_index_roots_compatibility(
                configured_index_roots=("HP:0000118", "HP:0012823"),
                strict=True,
            )
        assert isinstance(exc_info.value, RetrievalError)

    def test_missing_required_root_on_restricted_index(self):
        retriever = self._make_retriever({"index_roots": "HP:0000118"})

        warnings = retriever.check_index_roots_compatibility(
            configured_index_roots=("HP:0000118",),
            required_roots=("HP:0012823",),
            ancestors_map={"HP:0012823": frozenset({"HP:0000001"})},
            strict=False,
        )
        assert len(warnings) == 1
        assert "HP:0012823" in warnings[0]

        with pytest.raises(IndexRootsMismatchError, match="HP:0012823"):
            retriever.check_index_roots_compatibility(
                configured_index_roots=("HP:0000118",),
                required_roots=("HP:0012823",),
                ancestors_map={"HP:0012823": frozenset({"HP:0000001"})},
                strict=True,
            )


# =============================================================================
# 5. DenseRetriever subtree-restricted exact cosine search & caching
# =============================================================================


class TestDenseRetrieverAllowedTermIds:
    def _create_real_multi_vector_collection(
        self, tmp_path: Path
    ) -> tuple[chromadb.Collection, _StubModel]:
        client = chromadb.PersistentClient(path=str(tmp_path / "chroma_mv"))
        collection = client.create_collection(
            name="unit_multi_vector",
            metadata={
                "index_type": "multi_vector",
                "hnsw:space": "cosine",
                "index_roots": "HP:0000118,HP:0012823",
            },
        )
        # Non-modifier abnormality terms have HIGHER similarity to query [1, 0, 0]
        # than the modifier terms do, so a global top-k would crowd modifiers out.
        collection.add(
            ids=[
                "HP:0099001__label__0",
                "HP:0099002__label__0",
                "HP:0099003__label__0",
                "HP:0099101__label__0",
                "HP:0099101__synonym__0",
                "HP:0099102__label__0",
            ],
            documents=[
                "Abnormality alpha",
                "Abnormality beta",
                "Abnormality gamma",
                "Episodic course",
                "Intermittent pattern",
                "Mild intensity",
            ],
            embeddings=[
                [1.0, 0.0, 0.0],  # sim = 1.00 (abnormality)
                [0.99, 0.14, 0.0],  # sim ~ 0.99 (abnormality)
                [0.98, 0.20, 0.0],  # sim ~ 0.98 (abnormality)
                [0.60, 0.80, 0.0],  # sim = 0.60 (modifier label)
                [0.85, 0.5268, 0.0],  # sim ~ 0.85 (modifier synonym)
                [0.40, 0.9165, 0.0],  # sim = 0.40 (modifier label)
            ],
            metadatas=[
                {
                    "hpo_id": "HP:0099001",
                    "component": "label",
                    "label": "Abnormality alpha",
                },
                {
                    "hpo_id": "HP:0099002",
                    "component": "label",
                    "label": "Abnormality beta",
                },
                {
                    "hpo_id": "HP:0099003",
                    "component": "label",
                    "label": "Abnormality gamma",
                },
                {
                    "hpo_id": "HP:0099101",
                    "component": "label",
                    "label": "Episodic course",
                },
                {
                    "hpo_id": "HP:0099101",
                    "component": "synonym",
                    "label": "Episodic course",
                    "synonym_text": "Intermittent pattern",
                },
                {
                    "hpo_id": "HP:0099102",
                    "component": "label",
                    "label": "Mild intensity",
                },
            ],
        )
        model = _StubModel(
            {
                "intermittent": [1.0, 0.0, 0.0],
                "mild": [0.40, 0.9165, 0.0],
            }
        )
        return collection, model

    def test_query_multi_vector_restricted_recalls_modifiers_and_reuses_cache(
        self, tmp_path, mocker
    ):
        collection, model = self._create_real_multi_vector_collection(tmp_path)
        retriever = DenseRetriever(model, collection, min_similarity=0.0)  # type: ignore[arg-type]

        # Unrestricted top-2 returns only non-modifier abnormalities
        unrestricted = retriever.query_multi_vector("intermittent", n_results=2)
        assert [item["hpo_id"] for item in unrestricted] == [
            "HP:0099001",
            "HP:0099002",
        ]

        spy_get = mocker.spy(collection, "get")
        modifier_ids = frozenset({"HP:0099101", "HP:0099102"})

        # Restricted query achieves 100% recall over modifier subset and aggregates
        # synonym vector (sim ~ 0.85) over label vector (sim = 0.60) for HP:0099101.
        restricted_first = retriever.query_multi_vector(
            "intermittent",
            n_results=2,
            allowed_term_ids=modifier_ids,
        )
        assert [item["hpo_id"] for item in restricted_first] == [
            "HP:0099101",
            "HP:0099102",
        ]
        assert restricted_first[0]["similarity"] == pytest.approx(0.85, abs=1e-3)
        assert restricted_first[0]["matched_component"] == "synonym"
        assert restricted_first[0]["matched_text"] == "Intermittent pattern"
        assert spy_get.call_count == 1

        # Second call with same subset hits _subtree_cache (no extra collection.get)
        restricted_second = retriever.query_multi_vector(
            "mild",
            n_results=2,
            allowed_term_ids=set(modifier_ids),
        )
        assert [item["hpo_id"] for item in restricted_second] == [
            "HP:0099102",
            "HP:0099101",
        ]
        assert restricted_second[0]["similarity"] == pytest.approx(1.0, abs=1e-3)
        assert spy_get.call_count == 1

    def test_query_batch_allowed_term_ids_edge_cases_and_chunking(
        self, tmp_path, mocker
    ):
        collection, model = self._create_real_multi_vector_collection(tmp_path)
        retriever = DenseRetriever(model, collection, min_similarity=0.0)  # type: ignore[arg-type]

        # Empty allowed_term_ids returns empty results without encoding
        empty_res = retriever.query_batch(
            ["intermittent", "mild"],
            n_results=3,
            allowed_term_ids=set(),
        )
        assert len(empty_res) == 2
        assert empty_res[0]["ids"] == [[]]
        assert empty_res[0]["similarities"] == [[]]
        assert model.encode_calls == 0

        # Non-existent IDs return empty results cleanly without crashing
        missing_res = retriever.query_batch(
            ["intermittent"],
            n_results=3,
            allowed_term_ids={"HP:0999999"},
        )
        assert missing_res == [
            {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "similarities": [[]],
            }
        ]

        # Large allowed_term_ids set chunks collection.get calls by CHROMA_SUBTREE_GET_BATCH_SIZE
        spy_get = mocker.spy(collection, "get")
        many_ids = {f"HP:{i:07d}" for i in range(CHROMA_SUBTREE_GET_BATCH_SIZE + 5)}
        many_ids.add("HP:0099101")
        batch_res = retriever.query_batch(
            ["intermittent"],
            n_results=2,
            allowed_term_ids=many_ids,
        )
        assert spy_get.call_count == 2
        assert batch_res[0]["ids"][0][0] == "HP:0099101__synonym__0"
        assert batch_res[0]["similarities"][0][0] == pytest.approx(0.85, abs=1e-3)


# =============================================================================
# 6. ToolExecutor forwarding, legacy fallback, and compatibility delegation
# =============================================================================


class TestToolExecutorAllowedTermIdsAndCompatibility:
    def _make_executor(
        self, retriever: Any, *, multi_vector: bool = True
    ) -> ToolExecutor:
        executor = ToolExecutor(retriever=retriever, multi_vector=multi_vector)
        # Pre-populate cache to avoid loading real models in unit tests
        executor._cached_embedding_model = object()
        executor._cached_pipelines["en"] = object()
        executor._cached_retriever = retriever
        return executor

    def test_forwards_allowed_term_ids_to_compatible_retriever(self):
        recorded: list[dict[str, Any]] = []

        class _ModernRetriever:
            def query_multi_vector(
                self,
                text: str,
                n_results: int = 10,
                aggregation_strategy: str = "label_synonyms_max",
                allowed_term_ids: set[str] | frozenset[str] | None = None,
            ) -> list[dict[str, Any]]:
                recorded.append(
                    {
                        "text": text,
                        "n_results": n_results,
                        "allowed_term_ids": allowed_term_ids,
                    }
                )
                return [
                    {
                        "hpo_id": "HP:0099101",
                        "label": "Episodic course",
                        "similarity": 0.88,
                        "matched_text": "Episodic course",
                        "matched_component": "label",
                    }
                ]

        executor = self._make_executor(_ModernRetriever(), multi_vector=True)

        # Unrestricted call does not pass allowed_term_ids (uses default None)
        executor.query_batch_hpo_terms(phrases=["episodic"], language="en", n_results=5)
        assert recorded[-1]["allowed_term_ids"] is None

        # Restricted call forwards allowed_term_ids directly
        subset = frozenset({"HP:0099101"})
        out = executor.query_batch_hpo_terms(
            phrases=["episodic"],
            language="en",
            n_results=5,
            allowed_term_ids=subset,
        )
        assert recorded[-1]["allowed_term_ids"] == subset
        assert out[0]["candidates"][0]["hpo_id"] == "HP:0099101"

    def test_legacy_fake_retriever_fallback_post_filters_multi_vector(self):
        class _LegacyMultiVectorRetriever:
            def query_multi_vector(
                self,
                text: str,
                n_results: int = 10,
                aggregation_strategy: str = "label_synonyms_max",
            ) -> list[dict[str, Any]]:
                return [
                    {
                        "hpo_id": "HP:0099001",
                        "label": "Abnormality alpha",
                        "similarity": 0.95,
                    },
                    {
                        "hpo_id": "HP:0099101",
                        "label": "Episodic course",
                        "similarity": 0.72,
                    },
                ]

        executor = self._make_executor(_LegacyMultiVectorRetriever(), multi_vector=True)
        out = executor.query_batch_hpo_terms(
            phrases=["episodic"],
            language="en",
            n_results=5,
            allowed_term_ids={"HP:0099101"},
        )
        assert [c["hpo_id"] for c in out[0]["candidates"]] == ["HP:0099101"]

    def test_legacy_fake_retriever_fallback_post_filters_query_batch(self):
        class _LegacyBatchRetriever:
            def query_batch(
                self, texts: list[str], n_results: int = 10
            ) -> list[dict[str, Any]]:
                return [
                    {
                        "ids": [["HP:0099001", "HP:0099101"]],
                        "documents": [["Abnormality alpha", "Episodic course"]],
                        "metadatas": [
                            [
                                {
                                    "hpo_id": "HP:0099001",
                                    "label": "Abnormality alpha",
                                },
                                {
                                    "hpo_id": "HP:0099101",
                                    "label": "Episodic course",
                                },
                            ]
                        ],
                        "distances": [[0.05, 0.25]],
                        "similarities": [[0.95, 0.75]],
                    }
                    for _ in texts
                ]

        executor = self._make_executor(_LegacyBatchRetriever(), multi_vector=False)
        out = executor.query_batch_hpo_terms(
            phrases=["episodic"],
            language="en",
            n_results=5,
            allowed_term_ids={"HP:0099101"},
        )
        assert out[0]["ids"] == [["HP:0099101"]]
        assert out[0]["metadatas"] == [
            [{"hpo_id": "HP:0099101", "label": "Episodic course"}]
        ]
        assert out[0]["similarities"] == [[0.75]]

    def test_check_index_roots_compatibility_delegation(self):
        retriever = DenseRetriever(
            _StubModel(),  # type: ignore[arg-type]
            SimpleNamespace(metadata={"index_roots": "HP:0000118"}),  # type: ignore[arg-type]
        )
        executor = ToolExecutor(retriever=retriever)

        warnings = executor.check_index_roots_compatibility(
            configured_index_roots=("HP:0000118",),
            required_roots=("HP:0012823",),
            strict=False,
        )
        assert len(warnings) == 1
        assert "HP:0012823" in warnings[0]

        with pytest.raises(IndexRootsMismatchError):
            executor.check_index_roots_compatibility(
                configured_index_roots=("HP:0000118",),
                required_roots=("HP:0012823",),
                strict=True,
            )

        # Retriever without the method returns empty list gracefully
        minimal_executor = ToolExecutor(retriever=SimpleNamespace())
        assert minimal_executor.check_index_roots_compatibility() == []

    def test_multi_vector_subtree_query_does_not_truncate_terms_with_over_20_vectors(
        self,
    ):
        """Terms with >20 component vectors must keep every vector in subset aggregation."""
        ids = [f"HP:0099101__synonym_{i}" for i in range(25)] + ["HP:0099101__label"]
        # 25 synonyms with similarity ~1.0 and 1 label with similarity 0.0 ->
        # under ALL_MIN the minimum must reflect the 26th vector (0.0), not be
        # truncated at 20 vectors.
        embeddings = [[1.0, 0.0]] * 25 + [[0.0, 1.0]]
        metadatas = [
            {"hpo_id": "HP:0099101", "label": "Many synonyms", "component": "synonym"}
        ] * 25 + [
            {"hpo_id": "HP:0099101", "label": "Many synonyms", "component": "label"}
        ]
        documents = [f"syn {i}" for i in range(25)] + ["Many synonyms"]

        class _ManyVectorCollection:
            metadata = {"index_type": "multi_vector"}

            def get(self, where=None, include=None):
                return {
                    "ids": ids,
                    "embeddings": embeddings,
                    "metadatas": metadatas,
                    "documents": documents,
                }

        class _UnitModel:
            def parameters(self):
                yield SimpleNamespace(device=SimpleNamespace(type="cpu"))

            def encode(self, texts, device="cpu"):
                return [[1.0, 0.0] for _ in texts]

        retriever = DenseRetriever(
            _UnitModel(),  # type: ignore[arg-type]
            _ManyVectorCollection(),  # type: ignore[arg-type]
            min_similarity=0.0,
        )
        results = retriever.query_multi_vector(
            "query",
            n_results=1,
            aggregation_strategy="all_min",
            allowed_term_ids={"HP:0099101"},
        )
        assert len(results) == 1
        assert results[0]["similarity"] == pytest.approx(0.0, abs=1e-6)
