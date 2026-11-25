"""
Dense retrieval functionality for HPO terms.

This module provides functionality for querying the ChromaDB index to retrieve
relevant HPO terms based on semantic similarity with input text.
"""

import logging
import os
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

# NOTE: Heavy dependencies (chromadb, SentenceTransformer) are only imported
# for type hints (TYPE_CHECKING) or lazily inside functions where actually used.
# This avoids the 2.8+ second chromadb import cost at module load time.
# chromadb alone loads: API, telemetry, OpenTelemetry, jsonschema, numpy typing.
if TYPE_CHECKING:
    import chromadb
    from sentence_transformers import SentenceTransformer

from phentrieve.config import (
    MIN_SIMILARITY_THRESHOLD,
    MULTI_VECTOR_RESULT_MULTIPLIER,
    VectorStoreConfig,
)
from phentrieve.retrieval.aggregation import (
    AggregationStrategy,
    aggregate_multi_vector_results,
)
from phentrieve.retrieval.utils import convert_multi_vector_to_chromadb_format
from phentrieve.utils import (
    calculate_similarity,
    generate_collection_name,
    get_default_index_dir,
    resolve_data_path,
)
from phentrieve.utils import (
    sanitize_log_value as _sanitize,
)

# Maximum number of query embeddings sent to ChromaDB in one collection.query()
# call. Chroma binds per-embedding variables into a single SQLite statement, so
# large batches (long clinical notes with hundreds of chunks) exceed SQLite's
# bound-variable limit ("too many SQL variables"). See issue #402.
CHROMA_QUERY_BATCH_SIZE = 64

# Maximum number of HPO IDs sent to ChromaDB in one collection.get($in) call
# when fetching subtree vectors into memory.
CHROMA_SUBTREE_GET_BATCH_SIZE = 200


class RetrievalError(RuntimeError):
    """Raised when querying the HPO vector index fails.

    Retrieval failures must not be reported as "no matching terms": callers
    (API, CLI, MCP) need to surface them as errors.
    """


class IndexRootsMismatchError(RetrievalError):
    """Raised when a collection's stored index_roots is incompatible with requirements."""


def _get_collection_name(collection: Any) -> str:
    """Return a Chroma collection name across ChromaDB 0.6 and 1.x APIs."""
    if isinstance(collection, str):
        return collection
    return str(collection.name)


def connect_to_chroma(
    index_dir: str, collection_name: str, model_name: str | None = None
) -> Optional["chromadb.Collection"]:
    """
    Connect to the ChromaDB index and retrieve the specified collection.

    Args:
        index_dir: Directory where ChromaDB indices are stored
        collection_name: Name of the collection to connect to
        model_name: Optional model name to handle dimension-specific
            collections

    Returns:
        ChromaDB collection or None if connection failed
    """
    # Lazy import - only load chromadb when actually connecting to database
    # Avoids 2.8s import overhead for CLI commands that don't use ChromaDB
    from pathlib import Path

    import chromadb

    try:
        # Convert Path to string and ensure it exists
        index_dir_str = str(index_dir)

        # Create vector store configuration
        # Note: We use model_name or collection_name for config creation
        # to maintain backward compatibility with existing collections
        if model_name:
            vector_store_config = VectorStoreConfig.for_chromadb(
                model_name=model_name,
                index_dir=Path(index_dir_str),
            )
        else:
            # Fallback: create config with default settings if no model_name
            # This handles edge cases where only collection_name is provided
            vector_store_config = VectorStoreConfig(
                path=index_dir_str,
                collection_name=collection_name,
            )

        # Initialize ChromaDB client with config
        client = chromadb.PersistentClient(
            path=vector_store_config.path,
            settings=vector_store_config.to_chromadb_settings(),
        )

        try:
            # Get the collection
            collection = client.get_collection(name=collection_name)
            count = collection.count()
            logging.info(
                "Connected to collection '%s' with %s docs",
                _sanitize(collection_name),
                _sanitize(count),
            )
            return collection
        except Exception as e:
            logging.error(
                "Error getting collection '%s': %s",
                _sanitize(collection_name),
                _sanitize(e),
            )

            # List available collections for debugging
            collections = client.list_collections()
            if collections:
                collection_names = [_get_collection_name(c) for c in collections]
                collection_list = ", ".join(collection_names)
                logging.info("Available collections: %s", _sanitize(collection_list))

                # If model_name is provided, check if we need to create a new format
                if model_name:
                    alternate_name = generate_collection_name(model_name)
                    if (
                        alternate_name != collection_name
                        and alternate_name in collection_names
                    ):
                        logging.info(
                            "Found alternate collection: %s", _sanitize(alternate_name)
                        )
                        return client.get_collection(name=alternate_name)
            else:
                logging.error("No collections found in %s", _sanitize(index_dir))
                logging.error("Please run setup script to build the index first.")

            return None

    except Exception as e:
        logging.error(
            "Error connecting to ChromaDB: %s: %s", _sanitize(index_dir), _sanitize(e)
        )
        return None


class DenseRetriever:
    """
    Dense retriever for HPO terms using ChromaDB.

    This class handles connecting to a ChromaDB collection and querying it
    to retrieve relevant HPO terms based on semantic similarity.
    """

    def __init__(
        self,
        model: "SentenceTransformer",
        collection: "chromadb.Collection",
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
    ):
        """
        Initialize the dense retriever.

        Args:
            model: SentenceTransformer model for encoding queries
            collection: ChromaDB collection to query
            min_similarity: Minimum similarity threshold for results
        """
        self.model = model
        self.collection = collection
        self.min_similarity = min_similarity
        # These will be set in from_model_name
        self.model_name: str | None = None
        self.index_base_path: Path | None = None
        self._index_type: str | None = None  # Cached index type
        self._subtree_cache: dict[frozenset[str], dict[str, Any]] = {}

    def get_index_roots(self) -> tuple[str, ...]:
        """Return the canonical index_roots stored on the ChromaDB collection.

        Returns an empty tuple ``()`` when metadata is missing or empty,
        representing an unrestricted index over all active HPO terms.
        """
        from phentrieve.indexing.chromadb_indexer import parse_index_roots_metadata

        try:
            metadata = self.collection.metadata or {}
            raw = metadata.get("index_roots")
        except Exception as e:
            logging.warning(
                "Could not read index_roots metadata, assuming unrestricted: %s",
                _sanitize(e),
            )
            return ()
        return parse_index_roots_metadata(raw if isinstance(raw, str) else None)

    def check_index_roots_compatibility(
        self,
        *,
        configured_index_roots: Sequence[str] = (),
        required_roots: Sequence[str] = (),
        ancestors_map: Mapping[str, AbstractSet[str]] | None = None,
        strict: bool = False,
    ) -> list[str]:
        """Verify that collection ``index_roots`` metadata matches configuration.

        Detects two mismatch conditions:
        1. ``configured_index_roots`` differs from the collection's stored
           ``index_roots`` (after canonical sorting/deduplication).
        2. The collection is restricted (non-empty stored ``index_roots``) and
           a root in ``required_roots`` is not covered by the stored roots
           (either directly or via ``ancestors_map``).

        Logs a warning for each mismatch and raises ``IndexRootsMismatchError``
        when ``strict=True``. Returns the list of warning messages (empty when
        compatible).
        """
        stored_roots = self.get_index_roots()
        normalized_configured = tuple(
            sorted(
                {
                    root.strip()
                    for root in configured_index_roots
                    if root and root.strip()
                }
            )
        )
        normalized_required = tuple(
            sorted({root.strip() for root in required_roots if root and root.strip()})
        )

        warnings: list[str] = []

        if normalized_configured != stored_roots:
            stored_desc = ",".join(stored_roots) if stored_roots else "<all>"
            configured_desc = (
                ",".join(normalized_configured) if normalized_configured else "<all>"
            )
            warnings.append(
                "Index roots mismatch: collection has "
                f"index_roots={stored_desc} but configured index_roots={configured_desc}"
            )

        if stored_roots and normalized_required:
            stored_set = set(stored_roots)
            missing_required: list[str] = []
            for root in normalized_required:
                if root in stored_set:
                    continue
                if ancestors_map is not None:
                    root_ancestors = ancestors_map.get(root)
                    if root_ancestors and any(
                        ancestor in stored_set for ancestor in root_ancestors
                    ):
                        continue
                missing_required.append(root)
            if missing_required:
                warnings.append(
                    "Restricted index is missing required roots "
                    f"{','.join(missing_required)} (collection index_roots="
                    f"{','.join(stored_roots)})"
                )

        for warning_msg in warnings:
            logging.warning("%s", _sanitize(warning_msg))

        if strict and warnings:
            raise IndexRootsMismatchError("; ".join(warnings))

        return warnings

    def detect_index_type(self) -> str:
        """
        Detect if collection uses single-vector or multi-vector format.

        Reads the index_type from collection metadata. Returns "single_vector"
        if not found (backward compatibility).

        Returns:
            "single_vector" or "multi_vector"
        """
        if self._index_type is not None:
            return self._index_type

        try:
            metadata = self.collection.metadata
            self._index_type = metadata.get("index_type", "single_vector")
            logging.debug("Detected index type: %s", _sanitize(self._index_type))
        except Exception as e:
            logging.warning(
                "Could not detect index type, assuming single_vector: %s", _sanitize(e)
            )
            self._index_type = "single_vector"

        return self._index_type

    @classmethod
    def from_model_name(
        cls,
        model: "SentenceTransformer",
        model_name: str,
        index_dir: str | Path | None = None,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
        multi_vector: bool = False,
    ) -> Optional["DenseRetriever"]:
        """
        Create a retriever instance from a model name.

        Args:
            model: SentenceTransformer model instance
            model_name: Name of the model (used for collection naming)
            index_dir: Directory where ChromaDB indices are stored
            min_similarity: Minimum similarity threshold for results
            multi_vector: Connect to multi-vector index instead of single-vector

        Returns:
            DenseRetriever instance or None if connection failed
        """
        collection_name = generate_collection_name(model_name)
        if multi_vector:
            collection_name = f"{collection_name}_multi"
        final_index_dir: Path
        if index_dir is not None:
            final_index_dir = Path(index_dir)
            logging.info(
                "DenseRetriever.from_model_name: Using explicit index_dir: %s",
                _sanitize(final_index_dir),
            )
        else:
            # This will use get_default_index_dir(), which checks ENV VARS first
            final_index_dir = resolve_data_path(
                config_key="index_dir",  # Check phentrieve.yaml for 'index_dir' key
                default_func=get_default_index_dir,
            )
            logging.info(
                "DenseRetriever.from_model_name: Resolved index dir: %s",
                _sanitize(final_index_dir),
            )

        if not final_index_dir.exists() or not final_index_dir.is_dir():
            logging.error(
                "DenseRetriever: Index dir '%s' not found or not a directory.",
                _sanitize(final_index_dir),
            )
            env_index_dir_val = os.getenv("PHENTRIEVE_INDEX_DIR")
            env_data_root_val = os.getenv("PHENTRIEVE_DATA_ROOT_DIR")
            logging.error(
                "  ENV VARS: PHENTRIEVE_INDEX_DIR='%s', PHENTRIEVE_DATA_ROOT_DIR='%s'",
                _sanitize(env_index_dir_val),
                _sanitize(env_data_root_val),
            )
            return None

        collection = connect_to_chroma(
            str(final_index_dir), collection_name, model_name
        )

        if collection:
            instance = cls(model, collection, min_similarity)
            instance.model_name = model_name  # Store for reference
            instance.index_base_path = final_index_dir  # Store resolved path
            return instance
        else:
            logging.error(
                "DenseRetriever: Failed to connect to Chroma collection '%s' at '%s'.",
                _sanitize(collection_name),
                _sanitize(final_index_dir),
            )
            return None

    def _load_allowed_term_cache(self, cache_key: frozenset[str]) -> dict[str, Any]:
        """Fetch and L2-normalize all collection vectors for ``cache_key``."""
        if cache_key in self._subtree_cache:
            return self._subtree_cache[cache_key]

        import numpy as np

        ids_acc: list[str] = []
        docs_acc: list[str] = []
        metas_acc: list[dict[str, Any]] = []
        emb_chunks: list[Any] = []

        sorted_ids = sorted(cache_key)
        for start in range(0, len(sorted_ids), CHROMA_SUBTREE_GET_BATCH_SIZE):
            batch_ids = sorted_ids[start : start + CHROMA_SUBTREE_GET_BATCH_SIZE]
            part = self.collection.get(
                where=cast(Any, {"hpo_id": {"$in": batch_ids}}),
                include=cast(Any, ["embeddings", "metadatas", "documents"]),
            )
            part_ids = list(part.get("ids") or [])
            if not part_ids:
                continue
            part_embs = part.get("embeddings")
            if part_embs is None or len(part_embs) == 0:
                continue
            part_docs = list(part.get("documents") or [""] * len(part_ids))
            part_metas = list(part.get("metadatas") or [{}] * len(part_ids))

            ids_acc.extend(part_ids)
            docs_acc.extend(part_docs)
            metas_acc.extend(cast(list[dict[str, Any]], part_metas))
            emb_chunks.append(np.asarray(part_embs, dtype=np.float32))

        if not ids_acc:
            empty_entry: dict[str, Any] = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "unit_embeddings": None,
            }
            self._subtree_cache[cache_key] = empty_entry
            return empty_entry

        stored_matrix = np.vstack(emb_chunks)
        norms = np.linalg.norm(stored_matrix, axis=1, keepdims=True)
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        unit_embeddings = stored_matrix / safe_norms

        entry: dict[str, Any] = {
            "ids": ids_acc,
            "documents": docs_acc,
            "metadatas": metas_acc,
            "unit_embeddings": unit_embeddings,
        }
        self._subtree_cache[cache_key] = entry
        return entry

    def _query_allowed_term_ids(
        self,
        texts: list[str],
        *,
        allowed_term_ids: set[str] | frozenset[str],
        n_results: int,
        include_similarities: bool = True,
    ) -> list[dict[str, Any]]:
        """Run exact in-memory cosine search restricted to ``allowed_term_ids``."""
        if not texts:
            return []

        def _empty_result() -> dict[str, Any]:
            result: dict[str, Any] = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }
            if include_similarities:
                result["similarities"] = [[]]
            return result

        if not allowed_term_ids:
            return [_empty_result() for _ in texts]

        try:
            import numpy as np

            cached = self._load_allowed_term_cache(frozenset(allowed_term_ids))
            stored_unit = cached["unit_embeddings"]
            if stored_unit is None or not cached["ids"]:
                return [_empty_result() for _ in texts]

            device = next(self.model.parameters()).device
            device_name = "cuda" if device.type == "cuda" else "cpu"
            query_embeddings = self.model.encode(texts, device=device_name)
            query_matrix = np.asarray(query_embeddings, dtype=np.float32)
            if query_matrix.ndim == 1:
                query_matrix = query_matrix.reshape(1, -1)
            q_norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
            q_safe_norms = np.where(q_norms > 0.0, q_norms, 1.0)
            query_unit = query_matrix / q_safe_norms

            sim_matrix = query_unit @ stored_unit.T

            stored_ids: list[str] = cached["ids"]
            stored_docs: list[str] = cached["documents"]
            stored_metas: list[dict[str, Any]] = cached["metadatas"]

            results_list: list[dict[str, Any]] = []
            for i in range(len(texts)):
                row_sims = sim_matrix[i]
                order = np.argsort(-row_sims, kind="stable")
                top_indices = order[: max(n_results, 0)]

                res_ids = [stored_ids[int(idx)] for idx in top_indices]
                res_docs = [stored_docs[int(idx)] for idx in top_indices]
                res_metas = [stored_metas[int(idx)] for idx in top_indices]
                res_dists = [1.0 - float(row_sims[int(idx)]) for idx in top_indices]

                result: dict[str, Any] = {
                    "ids": [res_ids],
                    "documents": [res_docs],
                    "metadatas": [res_metas],
                    "distances": [res_dists],
                }
                if include_similarities:
                    result["similarities"] = [
                        [calculate_similarity(d) for d in res_dists]
                    ]
                results_list.append(result)

            return results_list
        except Exception as e:
            logging.error(
                "Error in subtree-restricted batch query to HPO index: %s",
                _sanitize(e),
            )
            raise RetrievalError(
                f"Subtree-restricted batch query to HPO index failed: {e}"
            ) from e

    def query_batch(
        self,
        texts: list[str],
        n_results: int = 10,
        include_similarities: bool = True,
        allowed_term_ids: set[str] | frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate embeddings for multiple texts and query the HPO index in batch.

        This method is more efficient than calling query() multiple times sequentially,
        as it:
        1. Encodes all texts in a single batch (faster)
        2. Queries ChromaDB with many embeddings per call (10-20x faster), in
           slices of CHROMA_QUERY_BATCH_SIZE to stay under SQLite's limit

        Args:
            texts: List of input texts to query
            n_results: Number of results to retrieve per text
            include_similarities: Whether to include similarity scores in results
            allowed_term_ids: Optional set of HPO IDs to restrict retrieval to via
                cached exact cosine search. When None, searches the full collection.

        Returns:
            List of dictionaries, one per input text, each containing query results
            with distances and/or similarities

        Raises:
            RetrievalError: If encoding or querying the index fails.
        """
        if not texts:
            return []

        if allowed_term_ids is not None:
            return self._query_allowed_term_ids(
                texts,
                allowed_term_ids=allowed_term_ids,
                n_results=n_results,
                include_similarities=include_similarities,
            )

        logging.info("Processing batch query with %s texts", _sanitize(len(texts)))

        try:
            # Generate embeddings for all query texts at once (more efficient)
            # Get the device the model is on
            device = next(self.model.parameters()).device
            device_name = "cuda" if device.type == "cuda" else "cpu"

            # Encode all texts in one batch - this is much faster than encoding one at a time
            query_embeddings = self.model.encode(texts, device=device_name)

            # Get more initial results to allow better filtering
            query_n_results = n_results * 3
            logging.debug(
                "Batch querying %s texts with %s results each",
                _sanitize(len(texts)),
                _sanitize(query_n_results),
            )

            # Query ChromaDB in slices of CHROMA_QUERY_BATCH_SIZE embeddings.
            # Batch queries are much faster than sequential ones, but a single
            # query with too many embeddings exceeds SQLite's bound-variable
            # limit, so slice and concatenate the per-text result lists.
            embeddings = [emb.tolist() for emb in query_embeddings]
            batch_results: dict[str, list[Any]] = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": [],
            }
            for start in range(0, len(embeddings), CHROMA_QUERY_BATCH_SIZE):
                part = self.collection.query(
                    query_embeddings=embeddings[
                        start : start + CHROMA_QUERY_BATCH_SIZE
                    ],
                    n_results=query_n_results,
                    include=cast(Any, ["documents", "metadatas", "distances"]),
                )
                for key, values in batch_results.items():
                    values.extend(cast(list[Any], part.get(key) or []))

            # Convert batch results to list of individual results (one per text)
            # ChromaDB returns results as {"ids": [[...], [...]], "documents": [[...], [...]]}
            # We need to split this into [{"ids": [[...]], "documents": [[...]]}, ...]

            # Extract lists once (more efficient than per-iteration)
            ids_list = batch_results.get("ids")
            docs_list = batch_results.get("documents")
            metas_list = batch_results.get("metadatas")
            dists_list = batch_results.get("distances")

            # Validate result count matches input count
            if ids_list and len(ids_list) != len(texts):
                logging.warning(
                    "Batch result count mismatch: expected %s, got %s. Using available results.",
                    _sanitize(len(texts)),
                    _sanitize(len(ids_list)),
                )

            results_list = []
            for i in range(len(texts)):
                # Bounds check to prevent IndexError
                if ids_list and i >= len(ids_list):
                    logging.warning(
                        "Skipping text %s/%s: no result available from ChromaDB",
                        _sanitize(i + 1),
                        _sanitize(len(texts)),
                    )
                    break

                result: dict[str, Any] = {
                    "ids": [ids_list[i]] if ids_list else [[]],
                    "documents": [docs_list[i]] if docs_list else [[]],
                    "metadatas": [metas_list[i]] if metas_list else [[]],
                    "distances": [dists_list[i]] if dists_list else [[]],
                }

                # Add similarity scores if requested
                if include_similarities and result["distances"][0]:
                    similarities = []
                    for distance in result["distances"][0]:
                        similarity = calculate_similarity(distance)
                        similarities.append(similarity)

                    result["similarities"] = [similarities]

                results_list.append(result)

            return results_list

        except Exception as e:
            logging.error("Error in batch query to HPO index: %s", _sanitize(e))
            # Do not turn a failed query into "no matching terms" (issue #402).
            raise RetrievalError(f"Batch query to HPO index failed: {e}") from e

    def query(
        self,
        text: str,
        n_results: int = 10,
        include_similarities: bool = True,
        allowed_term_ids: set[str] | frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate embedding for input text and query the HPO index.

        This method is now a convenience wrapper around query_batch() to avoid
        code duplication (DRY principle).

        Args:
            text: The input text to query
            n_results: Number of results to retrieve
            include_similarities: Whether to include similarity scores in result
            allowed_term_ids: Optional set of HPO IDs to restrict retrieval to.

        Returns:
            Dictionary containing query results with distances and/or similarities
        """
        logging.info("Processing query text_chars=%s", _sanitize(len(text)))

        # Use query_batch() internally to avoid code duplication
        if allowed_term_ids is None:
            batch_results = self.query_batch([text], n_results, include_similarities)
        else:
            batch_results = self.query_batch(
                [text],
                n_results,
                include_similarities,
                allowed_term_ids=allowed_term_ids,
            )

        # Return the single result
        return (
            batch_results[0]
            if batch_results
            else {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": [],
            }
        )

    def filter_results(
        self,
        results: dict[str, Any],
        min_similarity: float | None = None,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        """
        Filter query results by similarity threshold and maximum count.

        Args:
            results: Results dictionary from query method
            min_similarity: Override for minimum similarity threshold
            max_results: Maximum number of results to include

        Returns:
            Filtered results dictionary
        """
        if not results or not results.get("ids") or not results["ids"][0]:
            return results

        threshold = (
            min_similarity if min_similarity is not None else self.min_similarity
        )

        # Create filtered results
        filtered_ids = []
        filtered_documents = []
        filtered_metadatas = []
        filtered_distances = []
        filtered_similarities = []

        # Calculate similarities if not already included
        similarities = results.get("similarities", [[]])
        if not similarities[0]:
            similarities = [[calculate_similarity(d) for d in results["distances"][0]]]

        # Filter by similarity threshold
        filtered_indices = []
        for i, similarity in enumerate(similarities[0]):
            if similarity >= threshold:
                filtered_indices.append(i)

        # Sort by similarity (descending)
        filtered_indices.sort(key=lambda i: similarities[0][i], reverse=True)

        # Apply max_results limit if provided
        if max_results is not None and max_results > 0:
            filtered_indices = filtered_indices[:max_results]

        # Build filtered results
        for i in filtered_indices:
            filtered_ids.append(results["ids"][0][i])
            filtered_documents.append(results["documents"][0][i])
            filtered_metadatas.append(results["metadatas"][0][i])
            filtered_distances.append(results["distances"][0][i])
            filtered_similarities.append(similarities[0][i])

        return {
            "ids": [filtered_ids],
            "documents": [filtered_documents],
            "metadatas": [filtered_metadatas],
            "distances": [filtered_distances],
            "similarities": [filtered_similarities],
        }

    def query_multi_vector(
        self,
        text: str,
        n_results: int = 10,
        aggregation_strategy: str
        | AggregationStrategy = AggregationStrategy.LABEL_SYNONYMS_MAX,
        component_weights: dict[str, float] | None = None,
        custom_formula: str | None = None,
        allowed_term_ids: set[str] | frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query a multi-vector index with score aggregation.

        This method queries the multi-vector index, groups results by HPO ID,
        and aggregates component scores using the specified strategy.

        Args:
            text: The input text to query
            n_results: Number of HPO terms to return (after aggregation)
            aggregation_strategy: Strategy for combining component scores
            component_weights: Weights for "all_weighted" strategy
            custom_formula: Formula for "custom" strategy
            allowed_term_ids: Optional set of HPO IDs to restrict retrieval to.

        Returns:
            List of result dictionaries sorted by aggregated score:
            [
                {
                    "hpo_id": "HP:0001250",
                    "label": "Seizure",
                    "similarity": 0.92,
                    "component_scores": {
                        "label": 0.85,
                        "synonyms": [0.92, 0.78],
                        "definition": 0.65,
                    },
                }
            ]
        """
        # Verify this is a multi-vector index
        index_type = self.detect_index_type()
        if index_type != "multi_vector":
            logging.warning(
                "query_multi_vector called on %s index. Results may be incorrect.",
                _sanitize(index_type),
            )

        logging.info(
            "Multi-vector query text_chars=%s with strategy %s",
            _sanitize(len(text)),
            _sanitize(str(aggregation_strategy)),
        )

        # Request more results to ensure we get enough unique HPO IDs
        # Multi-vector index has ~5-10x more documents per HPO term
        raw_n_results = n_results * MULTI_VECTOR_RESULT_MULTIPLIER

        # Query the index
        if allowed_term_ids is None:
            raw_results = self.query(
                text, n_results=raw_n_results, include_similarities=True
            )
        else:
            # Score all vectors of the allowed subset so every component vector
            # participates in multi-vector aggregation (100% recall over subset).
            subset_n_results = max(raw_n_results, len(allowed_term_ids) * 20)
            raw_results = self.query(
                text,
                n_results=subset_n_results,
                include_similarities=True,
                allowed_term_ids=allowed_term_ids,
            )

        # Aggregate results by HPO ID
        aggregated = aggregate_multi_vector_results(
            results=raw_results,
            strategy=aggregation_strategy,
            weights=component_weights,
            custom_formula=custom_formula,
            min_similarity=self.min_similarity,
        )

        # Return top n_results
        return aggregated[:n_results]

    def query_batch_multi_vector(
        self,
        texts: list[str],
        n_results: int = 10,
        aggregation_strategy: str
        | AggregationStrategy = AggregationStrategy.LABEL_SYNONYMS_MAX,
        component_weights: dict[str, float] | None = None,
        custom_formula: str | None = None,
        allowed_term_ids: set[str] | frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query a multi-vector index for multiple texts with per-text HPO aggregation.

        Returns one ChromaDB-style result dictionary per input text, matching
        the shape consumed by process_chunk_matches().

        Raises:
            RetrievalError: If querying the index fails.
            ValueError: If the aggregation strategy or formula is invalid.
        """
        if not texts:
            return []

        index_type = self.detect_index_type()
        if index_type != "multi_vector":
            logging.warning(
                "query_batch_multi_vector called on %s index. Results may be incorrect.",
                _sanitize(index_type),
            )

        raw_n_results = n_results * MULTI_VECTOR_RESULT_MULTIPLIER
        if allowed_term_ids is None:
            raw_batch_results = self.query_batch(
                texts=texts,
                n_results=raw_n_results,
                include_similarities=True,
            )
        else:
            subset_n_results = max(raw_n_results, len(allowed_term_ids) * 20)
            raw_batch_results = self.query_batch(
                texts=texts,
                n_results=subset_n_results,
                include_similarities=True,
                allowed_term_ids=allowed_term_ids,
            )

        converted_results: list[dict[str, Any]] = []
        for raw_results in raw_batch_results:
            aggregated = aggregate_multi_vector_results(
                results=raw_results,
                strategy=aggregation_strategy,
                weights=component_weights,
                custom_formula=custom_formula,
                min_similarity=self.min_similarity,
            )[:n_results]
            converted_results.append(
                convert_multi_vector_to_chromadb_format(
                    aggregated,
                    include_similarities=True,
                )
            )

        while len(converted_results) < len(texts):
            converted_results.append(
                convert_multi_vector_to_chromadb_format(
                    [],
                    include_similarities=True,
                )
            )
        return converted_results
