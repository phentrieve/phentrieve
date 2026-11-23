"""In-memory disease index and its on-disk format."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

INDEX_FORMAT_VERSION = 1
INDEX_DIRNAME = "disease_index"
INDEX_ARRAYS_FILENAME = "disease_index.npz"
INDEX_META_FILENAME = "disease_index.json"
INDEX_INFO_FILENAME = "disease_index.info.json"  # metadata only, cheap to read

# Annotation source flags (bit set): own HPOA rows of the disease, rows merged from ORPHA.
SOURCE_OWN = 1
SOURCE_ORPHA = 2

ALPHA_F = (
    0.25  # frequency attenuation of the forward lanes (VarLens v2 shipped setting)
)


class IndexNotAvailableError(RuntimeError):
    """The disease index is missing, unreadable or fails its checksum."""


def f32(values: Any) -> np.ndarray:
    """Round to float32 and back, as VarLens stores IC and frequencies in Float32Array."""
    return np.asarray(values, dtype=np.float32).astype(np.float64)


@dataclass
class DiseaseIndex:
    """Phenotype tables, the disease universe and its annotations.

    Term arrays are indexed by term position in ``term_ids`` (phenotype subtree only).
    Annotation arrays are a CSR over diseases: disease ``d`` owns the rows
    ``dis_ptr[d]:dis_ptr[d + 1]``.
    """

    term_ids: list[str]
    term_labels: list[str]
    aliases: dict[str, str]  # alt_id / replaced_by -> current term id
    ic: np.ndarray
    self_ic: np.ndarray  # max IC over ancestors-or-self (lane normaliser)
    anc_ptr: np.ndarray
    anc_idx: np.ndarray
    disease_ids: list[str]
    disease_labels: list[str]
    dis_ptr: np.ndarray
    ann_term: np.ndarray
    ann_f: np.ndarray
    ann_source: np.ndarray
    gene_diseases: dict[str, list[int]]
    twins: dict[int, list[int]]  # OMIM disease -> its ORPHA twins in the universe
    other_terms: list[str] = field(
        default_factory=list
    )  # HPO ids outside the phenotype subtree
    meta: dict[str, Any] = field(default_factory=dict)

    term_index: dict[str, int] = field(init=False, repr=False)
    disease_index: dict[str, int] = field(init=False, repr=False)
    ann_dis: np.ndarray = field(init=False, repr=False)
    ann_factor: np.ndarray = field(init=False, repr=False)
    freq_sum: np.ndarray = field(init=False, repr=False)
    disease_genes: list[list[str]] = field(init=False, repr=False)
    twin_of: dict[int, int] = field(init=False, repr=False)
    other_term_set: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.term_index = {t: i for i, t in enumerate(self.term_ids)}
        for alias, term in self.aliases.items():
            if term in self.term_index and alias not in self.term_index:
                self.term_index[alias] = self.term_index[term]
        self.disease_index = {d: i for i, d in enumerate(self.disease_ids)}
        n = len(self.disease_ids)
        self.ann_dis = np.repeat(np.arange(n), np.diff(self.dis_ptr))
        self.ann_factor = f32(1 - ALPHA_F + ALPHA_F * self.ann_f)
        self.freq_sum = np.bincount(self.ann_dis, self.ann_f, n)
        genes: list[list[str]] = [[] for _ in range(n)]
        for gene, ds in sorted(self.gene_diseases.items()):
            for d in ds:
                genes[d].append(gene)
        self.disease_genes = genes
        self.twin_of = {o: d for d, orphas in self.twins.items() for o in orphas}
        self.other_term_set = frozenset(self.other_terms)

    @property
    def n_diseases(self) -> int:
        return len(self.disease_ids)

    def ancestors(self, t: int) -> np.ndarray:
        return self.anc_idx[self.anc_ptr[t] : self.anc_ptr[t + 1]]

    def annotations(self, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, hi = self.dis_ptr[d], self.dis_ptr[d + 1]
        return self.ann_term[lo:hi], self.ann_f[lo:hi], self.ann_source[lo:hi]

    def resolve_term(self, hpo_id: str) -> int | None:
        return self.term_index.get(hpo_id)

    def save(self, index_dir: Path) -> None:
        """Write the arrays (npz) and the tables plus metadata (JSON) with the npz checksum.

        The arrays are swapped in last, after the JSON that records their checksum, so an
        interrupted save leaves an index that fails :func:`index_is_valid` (and gets
        rebuilt) instead of one that looks complete.
        """
        index_dir.mkdir(parents=True, exist_ok=True)
        arrays_path = index_dir / INDEX_ARRAYS_FILENAME
        tmp = arrays_path.with_name(arrays_path.name + ".part")
        with tmp.open("wb") as fh:
            np.savez_compressed(
                fh,
                ic=self.ic,
                self_ic=self.self_ic,
                anc_ptr=self.anc_ptr,
                anc_idx=self.anc_idx,
                dis_ptr=self.dis_ptr,
                ann_term=self.ann_term,
                ann_f=self.ann_f,
                ann_source=self.ann_source,
            )
        meta = dict(self.meta)
        meta["format_version"] = INDEX_FORMAT_VERSION
        meta["arrays_sha256"] = _sha256(tmp)
        payload = {
            "meta": meta,
            "term_ids": self.term_ids,
            "term_labels": self.term_labels,
            "aliases": self.aliases,
            "disease_ids": self.disease_ids,
            "disease_labels": self.disease_labels,
            "gene_diseases": self.gene_diseases,
            "twins": {str(k): v for k, v in self.twins.items()},
            "other_terms": self.other_terms,
        }
        for path, text in (
            (index_dir / INDEX_META_FILENAME, json.dumps(payload, ensure_ascii=False)),
            (
                index_dir / INDEX_INFO_FILENAME,
                json.dumps(meta, ensure_ascii=False, indent=1),
            ),
        ):
            part = path.with_name(path.name + ".part")
            part.write_text(text, encoding="utf-8")
            part.replace(path)
        tmp.replace(arrays_path)
        self.meta = meta


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_index_dir(data_dir: str | None = None) -> Path:
    """``<data_dir>/disease_index``, with the data dir resolved like the other data commands."""
    from phentrieve.utils import get_default_data_dir, resolve_data_path

    return resolve_data_path(data_dir, "data_dir", get_default_data_dir) / INDEX_DIRNAME


def index_exists(index_dir: Path) -> bool:
    return (index_dir / INDEX_ARRAYS_FILENAME).is_file() and (
        index_dir / INDEX_META_FILENAME
    ).is_file()


def index_is_valid(index_dir: Path) -> bool:
    """Complete, supported and matching its checksum (hashes the ~1 MB arrays file)."""
    info = read_index_info(index_dir)
    return bool(
        info
        and info.get("format_version") == INDEX_FORMAT_VERSION
        and info.get("arrays_sha256") == _sha256(index_dir / INDEX_ARRAYS_FILENAME)
    )


def read_index_info(index_dir: Path) -> dict[str, Any] | None:
    """The index metadata without loading the index; None when absent or unreadable."""
    if not index_exists(index_dir):
        return None
    try:
        info = json.loads((index_dir / INDEX_INFO_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def load_index(index_dir: Path) -> DiseaseIndex:
    """Load an index written by :meth:`DiseaseIndex.save` and verify its checksum."""
    arrays_path = index_dir / INDEX_ARRAYS_FILENAME
    meta_path = index_dir / INDEX_META_FILENAME
    if not index_exists(index_dir):
        raise IndexNotAvailableError(
            f"No disease index in {index_dir}. Run `phentrieve data prepare-diseases`."
        )
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = payload["meta"]
        if not isinstance(meta, dict):
            raise ValueError("meta is not an object")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise IndexNotAvailableError(
            f"Unreadable disease index metadata: {error}"
        ) from error
    if meta.get("format_version") != INDEX_FORMAT_VERSION:
        raise IndexNotAvailableError(
            f"Disease index format {meta.get('format_version')} is not supported "
            f"(expected {INDEX_FORMAT_VERSION}). Rebuild it with `phentrieve data prepare-diseases --force`."
        )
    if _sha256(arrays_path) != meta.get("arrays_sha256"):
        raise IndexNotAvailableError(
            "Disease index arrays do not match their recorded checksum. "
            "Rebuild it with `phentrieve data prepare-diseases --force`."
        )
    try:
        with np.load(arrays_path, allow_pickle=False) as arrays:
            data = {k: arrays[k] for k in arrays.files}
        return DiseaseIndex(
            term_ids=payload["term_ids"],
            term_labels=payload["term_labels"],
            aliases=payload["aliases"],
            ic=data["ic"],
            self_ic=data["self_ic"],
            anc_ptr=data["anc_ptr"],
            anc_idx=data["anc_idx"],
            disease_ids=payload["disease_ids"],
            disease_labels=payload["disease_labels"],
            dis_ptr=data["dis_ptr"],
            ann_term=data["ann_term"],
            ann_f=data["ann_f"],
            ann_source=data["ann_source"],
            gene_diseases={g: list(v) for g, v in payload["gene_diseases"].items()},
            twins={int(k): list(v) for k, v in payload["twins"].items()},
            other_terms=payload.get("other_terms", []),
            meta=meta,
        )
    except Exception as error:  # noqa: BLE001 - any damage means "not available"
        raise IndexNotAvailableError(
            f"Damaged disease index ({type(error).__name__}: {error}). "
            "Rebuild it with `phentrieve data prepare-diseases --force`."
        ) from error
