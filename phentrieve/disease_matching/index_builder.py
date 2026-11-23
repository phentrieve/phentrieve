"""Build the disease index from HPO release files and MONDO.

Parsing follows the issue #451 proof of concept (``scripts/disease_ranking_451/scorer.py``),
which reproduces the VarLens phenotype scoring v2 tables (HPO DB builder v6): the
conformance test checks that an index built without the ORPHA merge scores exactly like it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests

from phentrieve.disease_matching.index import (
    SOURCE_ORPHA,
    SOURCE_OWN,
    DiseaseIndex,
    f32,
)

logger = logging.getLogger(__name__)

BUILDER_VERSION = 1
ROOT = "HP:0000118"  # Phenotypic abnormality
FRACTION_PRIOR = (0.8, 1.2)  # Beta prior for n/m frequencies
FREQUENCY_TERMS = {
    "HP:0040280": 1.0,  # Obligate
    "HP:0040281": 0.895,  # Very frequent
    "HP:0040282": 0.545,  # Frequent
    "HP:0040283": 0.17,  # Occasional
    "HP:0040284": 0.025,  # Very rare
}
MISSING_F = 0.35
EXCLUDED_FREQUENCY = "HP:0040285"

HPO_FILES = ("hp.json", "phenotype.hpoa", "genes_to_phenotype.txt")
MONDO_FILE = "mondo.obo"
MONDO_BASE_URL = "https://github.com/monarch-initiative/mondo/releases/download"
MONDO_LATEST_API_URL = (
    "https://api.github.com/repos/monarch-initiative/mondo/releases/latest"
)
DOWNLOAD_TIMEOUT = 120

LICENCE_NOTES = {
    "hpo": "Human Phenotype Ontology and HPO annotations; see https://hpo.jax.org. "
    "The HPO docs ask for OMIM's permission to reuse OMIM-derived annotations in other "
    "software products.",
    "orphanet": "Orphanet annotations (Orphadata Science, CC BY 4.0). Modified: merged "
    "into the annotations of MONDO-equivalent OMIM diseases.",
    "mondo": "MONDO Disease Ontology (CC BY 4.0), used for OMIM-ORPHA equivalence only.",
}

_HP = re.compile(r"^(?:http://purl\.obolibrary\.org/obo/HP_|HP:)(\d{7})$")
_ALT_ID = "http://www.geneontology.org/formats/oboInOwl#hasAlternativeId"
_REPLACED_BY = "http://purl.obolibrary.org/obo/IAO_0100001"


def _hp(iri: str) -> str | None:
    m = _HP.match(iri)
    return f"HP:{m.group(1)}" if m else None


# --- Parsing ------------------------------------------------------------------


def parse_hp_json(
    path: Path,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    """Labels of current terms, is_a parents, and alias -> current term."""
    graph = json.loads(path.read_text(encoding="utf-8"))["graphs"][0]
    labels: dict[str, str] = {}
    raw_alias: dict[str, str] = {}
    for node in graph["nodes"]:
        tid = _hp(node["id"])
        if tid is None or node.get("type") != "CLASS":
            continue
        meta = node.get("meta") or {}
        deprecated = meta.get("deprecated") is True
        for pv in meta.get("basicPropertyValues") or []:
            other = _hp(pv.get("val", ""))
            if other is None:
                continue
            if not deprecated and pv["pred"] == _ALT_ID:
                raw_alias[other] = tid
            if deprecated and pv["pred"] == _REPLACED_BY:
                raw_alias[tid] = other
        if not deprecated:
            labels[tid] = node.get("lbl") or tid
    aliases: dict[str, str] = {}
    for alias in raw_alias:
        seen: set[str] = set()
        term: str | None = alias
        while term is not None and term not in labels and term not in seen:
            seen.add(term)
            term = raw_alias.get(term)
        if term is not None and term in labels and alias not in labels:
            aliases[alias] = term
    parents: dict[str, list[str]] = defaultdict(list)
    for e in graph.get("edges") or []:
        if e["pred"] != "is_a":
            continue
        sub, obj = _hp(e["sub"]), _hp(e["obj"])
        if sub is not None and obj is not None and sub in labels and obj in labels:
            parents[sub].append(obj)
    return labels, dict(parents), aliases


def phenotype_closure(
    labels: dict[str, str], parents: dict[str, list[str]], aliases: dict[str, str]
) -> dict[str, set[str]]:
    """Term (or alias) -> ancestors-or-self, restricted to the phenotype subtree."""
    closure: dict[str, set[str]] = {}

    def visit(term: str) -> set[str]:
        if term in closure:
            return closure[term]
        own = {term}
        for p in parents.get(term, []):
            own |= visit(p)
        closure[term] = own
        return own

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    for t in labels:
        visit(t)
    for alias, term in aliases.items():
        closure[alias] = closure.get(term, {term})
    inside = {t for t, anc in closure.items() if ROOT in anc}
    return {t: anc & inside for t, anc in closure.items() if t in inside}


def parse_hpoa(
    path: Path, aliases: dict[str, str]
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Disease -> {term: frequency} for positive phenotype rows, and disease labels."""
    labels: dict[str, str] = {}
    # per (disease, term): [n_observed, n_total, other_frequency | None, absent]
    pairs: dict[str, dict[str, list[Any]]] = defaultdict(dict)
    seen: set[tuple[str, ...]] = set()
    cols: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if cols is None:
            cols = fields
            continue
        row = dict(zip(cols, fields + [""] * (len(cols) - len(fields)), strict=False))
        if row["aspect"] != "P":
            continue
        term = aliases.get(row["hpo_id"], row["hpo_id"])
        key = (
            row["database_id"],
            row["qualifier"],
            term,
            row["reference"],
            row["evidence"],
            row["onset"],
            row["frequency"],
            row["sex"],
            row["modifier"],
        )
        if key in seen:
            continue
        seen.add(key)
        labels.setdefault(row["database_id"], row["disease_name"])
        ev = pairs[row["database_id"]].setdefault(term, [0, 0, None, False])
        freq = row["frequency"]
        frac = re.fullmatch(r"(\d+)/(\d+)", freq)
        if row["qualifier"] == "NOT":
            ev[3] = True
        elif frac and int(frac.group(2)) > 0:
            ev[0] += min(int(frac.group(1)), int(frac.group(2)))
            ev[1] += int(frac.group(2))
        else:
            other: float | str
            if freq in ("", "-"):
                other = MISSING_F
            elif freq in FREQUENCY_TERMS:
                other = FREQUENCY_TERMS[freq]
            elif freq == EXCLUDED_FREQUENCY:
                other = "absent"
            elif pct := re.fullmatch(r"(\d+(?:\.\d+)?)%", freq):
                f = min(1.0, float(pct.group(1)) / 100)
                other = f if f > 0 else "absent"
            else:
                other = MISSING_F
            if other == "absent":
                ev[3] = True
            elif ev[2] is None or float(other) > ev[2]:
                ev[2] = other
    a, b = FRACTION_PRIOR
    terms: dict[str, dict[str, float]] = {}
    for disease, by_term in pairs.items():
        for term, (m, n, other, _absent) in by_term.items():
            value = (m + a) / (n + a + b) if m > 0 else other
            if isinstance(value, float | int):
                terms.setdefault(disease, {})[term] = value
    return terms, labels


def parse_genes_to_phenotype(path: Path, hpoa: dict[str, Any]) -> dict[str, set[str]]:
    """Gene symbol -> diseases with a positive HPOA annotation."""
    gene_dis: dict[str, set[str]] = defaultdict(set)
    cols: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if cols is None:
            cols = fields
            continue
        row = dict(zip(cols, fields, strict=False))
        if row["gene_symbol"] and row["disease_id"] in hpoa:
            gene_dis[row["gene_symbol"]].add(row["disease_id"])
    return gene_dis


def parse_mondo_equivalences(path: Path) -> dict[str, set[str]]:
    """OMIM:x -> {ORPHA:y} for `MONDO:equivalentTo` xref pairs on one non-obsolete term."""
    out: dict[str, set[str]] = defaultdict(set)
    omim: list[str] = []
    orpha: list[str] = []
    obsolete = False

    def flush() -> None:
        if not obsolete:
            for o in omim:
                out[o].update(orpha)

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("["):
            flush()
            omim, orpha, obsolete = [], [], False
        elif line.startswith("is_obsolete: true"):
            obsolete = True
        elif line.startswith("xref: ") and "MONDO:equivalentTo" in line:
            ref = line[6:].split(" ", 1)[0]
            if ref.startswith("OMIM:") and ref[5:].isdigit():
                omim.append(ref)
            elif ref.startswith("Orphanet:") and ref[9:].isdigit():
                orpha.append("ORPHA:" + ref[9:])
    flush()
    return {k: v for k, v in out.items() if v}


# --- Building -----------------------------------------------------------------


def build_index(
    hp_json: Path,
    hpoa_path: Path,
    genes_to_phenotype: Path,
    mondo_obo: Path | None = None,
    *,
    merge_orpha: bool = True,
    meta: dict[str, Any] | None = None,
) -> DiseaseIndex:
    """Build the index; with ``merge_orpha`` OMIM diseases also get their ORPHA twins' rows."""
    if merge_orpha and mondo_obo is None:
        raise ValueError("merge_orpha needs a MONDO file")
    labels, parents, aliases = parse_hp_json(hp_json)
    closure = phenotype_closure(labels, parents, aliases)
    hpoa, hpoa_labels = parse_hpoa(hpoa_path, aliases)

    # IC(t) = -ln(n(t)/N) over diseases with >= 1 positive phenotype-subtree term.
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for terms in hpoa.values():
        reached: set[str] = set()
        for t in terms:
            reached |= closure.get(t, set())
        if reached:
            total += 1
            for anc in reached:
                counts[anc] += 1
    ic_of = {t: -math.log(n / total) for t, n in counts.items()}

    gene_dis = parse_genes_to_phenotype(genes_to_phenotype, hpoa)
    linked: dict[str, dict[str, float]] = {}
    for diseases in gene_dis.values():
        for d in diseases:
            if d in linked:
                continue
            merged: dict[str, float] = {}
            for raw, f in hpoa[d].items():
                term = aliases.get(raw, raw)
                if term in closure and f > merged.get(term, -1.0):
                    merged[term] = f
            if merged:
                linked[d] = merged

    ids = sorted(t for t in labels if t in closure)
    index = {t: i for i, t in enumerate(ids)}
    ic = f32([ic_of.get(t, 0.0) for t in ids])
    order: list[int] = []
    done: set[str] = set()

    def emit(t: str) -> None:
        if t in done:
            return
        for p in parents.get(t, []):
            if p in index:
                emit(p)
        done.add(t)
        order.append(index[t])

    for term_id in ids:
        emit(term_id)
    self_ic = np.zeros(len(ids))
    for ti in order:
        ps = [index[p] for p in parents.get(ids[ti], []) if p in index]
        self_ic[ti] = max([ic[ti], *(self_ic[p] for p in ps)])
    anc_rows = [sorted(index[a] for a in closure[term_id]) for term_id in ids]
    anc_ptr = np.cumsum([0, *map(len, anc_rows)])
    anc_idx = np.fromiter((a for row in anc_rows for a in row), dtype=np.int64)

    disease_ids = sorted(linked)
    d_index = {d: i for i, d in enumerate(disease_ids)}

    def to_rows(terms: dict[str, float]) -> dict[int, float]:
        rows: dict[int, float] = {}
        for raw, f in terms.items():
            ti = index.get(aliases.get(raw, raw))
            if ti is not None and self_ic[ti] > 0 and f > rows.get(ti, -1.0):
                rows[ti] = f
        return rows

    equivalents = parse_mondo_equivalences(mondo_obo) if mondo_obo is not None else {}
    twins: dict[int, list[int]] = {}
    ann_term: list[int] = []
    ann_f: list[float] = []
    ann_source: list[int] = []
    ptr = [0]
    n_enriched = 0
    for d in disease_ids:
        own = {
            index[term]: f for term, f in linked[d].items() if self_ic[index[term]] > 0
        }
        rows: dict[int, tuple[float, int]] = {
            ti: (f, SOURCE_OWN) for ti, f in own.items()
        }
        orphas = sorted(equivalents.get(d, ()))
        in_universe = [d_index[o] for o in orphas if o in d_index]
        if in_universe:
            twins[d_index[d]] = in_universe
        if merge_orpha:
            gained = False
            for o in orphas:
                if o not in hpoa:
                    continue
                for ti, f in to_rows(hpoa[o]).items():
                    if ti in rows:
                        old_f, src = rows[ti]
                        rows[ti] = (max(old_f, f), src | SOURCE_ORPHA)
                    else:
                        rows[ti] = (f, SOURCE_ORPHA)
                        gained = True
            n_enriched += gained
        for ti, (f, src) in rows.items():
            ann_term.append(ti)
            ann_f.append(f)
            ann_source.append(src)
        ptr.append(len(ann_term))

    info = dict(meta or {})
    info.update(
        {
            "builder_version": BUILDER_VERSION,
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "merge_orpha": merge_orpha,
            "n_terms": len(ids),
            "n_diseases": len(disease_ids),
            "n_genes": len(gene_dis),
            "n_omim_enriched": n_enriched,
            "n_twin_groups": len(twins),
            "licences": LICENCE_NOTES,
        }
    )
    return DiseaseIndex(
        term_ids=ids,
        term_labels=[labels[t] for t in ids],
        aliases={a: t for a, t in aliases.items() if t in index},
        ic=ic,
        self_ic=self_ic,
        anc_ptr=anc_ptr,
        anc_idx=anc_idx,
        disease_ids=disease_ids,
        disease_labels=[hpoa_labels.get(d, d) for d in disease_ids],
        dis_ptr=np.asarray(ptr),
        ann_term=np.asarray(ann_term, dtype=np.int64),
        ann_f=f32(ann_f),
        ann_source=np.asarray(ann_source, dtype=np.int8),
        gene_diseases={
            g: sorted(d_index[d] for d in ds if d in d_index)
            for g, ds in sorted(gene_dis.items())
            if any(d in d_index for d in ds)
        },
        twins=twins,
        other_terms=sorted(
            {t for t in labels if t not in index}
            | {a for a, t in aliases.items() if t in labels and t not in index}
        ),
        meta=info,
    )


# --- Sources ------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    logger.info("Downloading %s", url)
    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def latest_mondo_version() -> str:
    response = requests.get(
        MONDO_LATEST_API_URL,
        headers={"Accept": "application/vnd.github.v3+json"},
        timeout=30,
    )
    response.raise_for_status()
    tag = response.json().get("tag_name")
    if not isinstance(tag, str) or not tag:
        raise RuntimeError("GitHub API response for MONDO is missing 'tag_name'")
    return tag


def fetch_sources(
    sources_dir: Path,
    hpo_version: str,
    mondo_version: str,
    hpo_base_url: str,
    force: bool = False,
) -> dict[str, Any]:
    """Download the pinned inputs (skipping present files unless ``force``) and checksum them."""
    files: dict[str, dict[str, str]] = {}
    targets = [
        (name, f"{hpo_base_url}/{hpo_version}/{name}", sources_dir / hpo_version / name)
        for name in HPO_FILES
    ]
    targets.append(
        (
            MONDO_FILE,
            f"{MONDO_BASE_URL}/{mondo_version}/{MONDO_FILE}",
            sources_dir / f"mondo-{mondo_version}" / MONDO_FILE,
        )
    )
    for name, url, path in targets:
        if force or not path.is_file():
            _download(url, path)
        files[name] = {"url": url, "path": str(path), "sha256": sha256_file(path)}
    return {"hpo_version": hpo_version, "mondo_version": mondo_version, "files": files}
