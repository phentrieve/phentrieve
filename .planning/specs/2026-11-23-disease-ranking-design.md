# Disease Ranking from Collected Phenotypes: Design

- Date: 2026-11-23
- Author: Jan-Paul Lerch
- Status: Draft
- Related: #451 (evaluation), #452 (PoC), #508 (confirmation run review),
  branch `eval/scoring-variants-451` (`POSTHOC.md` sections 9 and 10)

## 1. Goal

Users collect HPO terms in the web frontend, marked as present or excluded. From that profile,
Phentrieve should show which rare diseases and genes match best, and why.

This feature is built in Phentrieve's own code: a library module, an API endpoint and a
frontend panel. The scorer is Phentrieve's copy of the PoC port. **VarLens is not changed.**
The only link to VarLens is the frozen reference output that the conformance test compares
against.

## 2. Non-goals

- No changes to VarLens, its scorer or its configuration.
- No calibrated probability, no "diagnosis" wording, no clinical decision support claim.
- No MCP tool and no CLI command in the first PR. They can follow, reusing the library.
- No inclusion in the phentrieve-data release yet (option A). The first version builds the
  index locally (option B, section 4).
- No retuning of the scorer. Section 9 of `POSTHOC.md` showed that the weights are not the
  lever.

## 3. Gates before public exposure

The feature ships behind a flag that is off by default (section 6.3). The following need an
explicit decision before the flag is turned on for any public or shared deployment:

1. **Evidence.** #451 ties building `phentrieve match` to the registered confirmation run. The
   ORPHA merge gain is post hoc so far.
2. **Intended use / EU MDR.** `api/research_use.py` states that Phentrieve "must not be used for
   diagnosis". A ranked disease list shown to clinicians may count as a medical device under
   MDR Rule 11 (#451). The research-use text and gate must cover this feature explicitly.
3. **Licences.** The HPO docs ask for OMIM's permission to reuse OMIM-derived annotations in
   other software products. Orphadata Science and MONDO are CC BY 4.0 (attribution, note on
   changes, Orphanet AI-use statement). The HPO licence page linked from `hp.json` returned 404
   on 2026-11-23 and must be checked again.

## 4. Data: local disease index (option B)

### 4.1 Inputs

All inputs are pinned and checksummed:

- From the HPO release that `phentrieve data prepare` already uses for `hp.json`: `phenotype.hpoa`
  and `genes_to_phenotype.txt`.
- `mondo.obo` from a pinned MONDO release. The default is the release closest to the HPO
  release date.

### 4.2 Command

A new command, `phentrieve data prepare-diseases`, takes `--hpo-version`, `--mondo-version`,
`--data-dir` and `--force`. It:

1. downloads the inputs into `<data_dir>/disease_index/sources/` and records their SHA256;
2. builds the index (section 4.3);
3. writes `<data_dir>/disease_index/disease_index.npz` and `disease_index.json`. The JSON holds
   versions, checksums, counts, the builder version and the licence notes.

`phentrieve data prepare` stays unchanged. The Docker entrypoint can call `prepare-diseases`
only when the feature flag is set.

### 4.3 Index contents

The index is the same as the PoC index (`scripts/disease_ranking_451/scorer.py`, `build_index`)
with the ORPHA merge from `enrichment.py`:

- **Term tables:** phenotype-subtree terms, ancestor closure (CSR), IC from `phenotype.hpoa`, and
  `self_ic`.
- **Universe:** gene-linked diseases from `genes_to_phenotype.txt`, OMIM and ORPHA. That gives
  9,162 diseases for HPO v2026-09-01.
- **ORPHA merge:** each OMIM disease also gets the annotations of its MONDO-equivalent ORPHA
  entries (`MONDO:equivalentTo` on both xrefs of one non-obsolete term); per term the higher
  frequency wins. The ORPHA entries stay in the universe.
- **Twin groups:** the OMIM-ORPHA equivalence groups are stored so the output can collapse
  twins (section 5.3).
- **Labels and links:** disease labels and gene symbols for each disease, plus the source of
  each annotation (OMIM, ORPHA, merged), for the explanation.

## 5. Library: `phentrieve/disease_matching/`

### 5.1 Modules

- `index_builder.py`: parsing (hp.json, HPOA, genes_to_phenotype, MONDO), the ORPHA merge, and
  writing the npz and JSON files.
- `index.py`: the `DiseaseIndex` dataclass and a loader with a checksum check.
- `scorer.py`: the PoC scorer at the frozen shipped setting. The parameters are module
  constants, not user options: tau 1, background correction off, exclusions `conflictsOnly`,
  alpha_f 0.25, f0 1.0, gamma 0.55, key weight 2.
- `service.py`: `rank(profile, level, top_k) -> RankingResult`.

### 5.2 Input profile

A list of `{hpo_id, assertion}` entries.

- Present terms are scored. They use `assertion_vocab`: affirmed counts as present.
- Excluded terms (negated, or normal; see `assertion_vocab.is_excluded`) never change the score.
  They are reported as conflicts when the disease is annotated with them or a descendant.
- Uncertain terms are not scored by default, and the response lists them as not scored. With
  `include_uncertain = true` they are scored exactly like present terms (w_p = 1). There is no
  reduced weight: phenopackets have no uncertain terms, so a weight could not be checked
  against data. The reverse score ignores w_p, and the ancestor pruning in `prepare` passes the
  higher weight to the descendant, so a reduced weight would also leak.
- Unknown and obsolete IDs are mapped through `alt_id` and `replaced_by` as in the PoC, or
  reported as not scored.

### 5.3 Output

- `level = disease`: ranked diseases. Each has an ID, label, genes, score, rank and twin IDs.
  ORPHA twins are always folded into their OMIM row. The row is placed at the better rank of
  the two, keeps the OMIM ID and label, and lists the ORPHA IDs as twins. Ranks are counted
  after folding.
- `level = gene`: ranked genes. Each has a score, its best disease and a rank.
- **Explanation**, for the returned rows only:
  - matched pairs: patient term, best disease term, IC of the shared ancestor, frequency;
  - conflicts;
  - the annotation source.
- **Metadata:** index versions and checksums, scorer version, and the labels "raw,
  uncalibrated" and "research use".

### 5.4 Performance

The index is loaded once and kept in memory. It takes tens of MB. One profile takes about
50 ms on the PoC code. Explanations are computed only for the top-k rows.

## 6. Backend

### 6.1 Endpoint

`POST /api/v1/diseases/rank`, in a new `api/routers/disease_ranking_router.py`, with schemas in
`api/schemas/disease_ranking_schemas.py`.

- **Request:** `terms: [{hpo_id, assertion}]`, `level: "disease" | "gene"`, `top_k` (default
  20, at most 100), and `include_uncertain` (default false).
- **Response:** as in section 5.3.
- **Errors:**
  - 422 for an empty or invalid profile;
  - 503 with a clear message when the index is missing.

### 6.2 Guardrails

- The research-use acknowledgement from `api/research_use.py` applies, as on the other
  text-to-HPO routes.
- No profile data is logged. The endpoint is stateless.

### 6.3 Flag and discovery

- `PHENTRIEVE_DISEASE_RANKING_ENABLED` (env, default false) in `api/config.py`. The router is
  mounted only when the flag is set, following the auth router pattern in `api/main.py`.
- The index is loaded lazily on the first request through a cached dependency, following
  `api/dependencies.py`. A missing index does not stop the API from starting.
- `config_info_router` reports `disease_ranking: {enabled, index_ready, versions}`, so the
  frontend can decide whether to show the feature.

## 7. Frontend

- **Entry point:** a "Similar diseases" action in `PhenotypeCollectionPanel.vue`. It is visible
  only when config-info reports the feature as enabled and ready, and enabled when at least
  one term is present.
- **Panel:** a new `DiseaseRankingPanel.vue`, opened as a dialog or side panel:
  - a disease/gene toggle;
  - an "Include uncertain terms" switch, off by default;
  - a ranked list with score bars and OMIM and Orphanet links;
  - an expandable explanation per row (matched pairs, conflicts);
  - a fixed research-use notice and the source attribution: HPO, Orphanet (CC BY 4.0,
    modified: merged with OMIM annotations), MONDO, OMIM.
- **State:** a new Pinia store `diseaseRanking.js`.
  - While the panel is open, the ranking reruns automatically when the collection, the level
    or the uncertain switch changes. Reruns are debounced (about 400 ms), and a response to a
    superseded request is dropped.
  - While the panel is closed, nothing is computed.
  - Results are not persisted to localStorage (compare #547).
- **Service:** `PhentrieveService.rankDiseases()`.
- **i18n:** all strings go into every locale; `make frontend-i18n-check`.

## 8. Tests

- **Scorer conformance:** `tests/unit/disease_matching/` uses the frozen
  `conformance/varlens_reference.json` from the PoC, copied into `tests/data/`. Scores must match
  to 1e-5 and ranks exactly, on the unmerged index.
- **ORPHA merge:** an invented mini HPOA and MONDO fixture checks the merge rule, the
  higher-frequency rule, obsolete MONDO terms and twin collapsing.
- **Service:** assertion handling (present, excluded, uncertain), unknown IDs, gene level, and
  the explanation contents.
- **API:** the flag on and off, 503 without an index, 422 cases, the research-use gate, and the
  schema shape.
- **Frontend:** store and service tests (vitest), and visibility of the panel depending on
  config-info.
- Unit tests use no network. The small fixtures live under `tests/data/`.
- Required checks: `make check`, `make typecheck-fast`, `make test`, `make ci-frontend`, and
  `make frontend-i18n-check`.

## 9. Delivery

One PR from `main` on `feat/disease-ranking`, behind the flag, with one commit per layer:

1. spec and plan;
2. library and `prepare-diseases`;
3. API;
4. frontend;
5. docs and changelog.

The PR stays one PR regardless of size (decision 2026-11-23). The per-layer commits are the
review units.

The PoC directory `scripts/disease_ranking_451/` on `main` stays as it is. The library is a
clean copy, not an import from it.

## 10. Decisions (2026-11-23)

1. ORPHA twins are always folded into their OMIM row.
2. There is no reduced weight for uncertain terms. A visible switch, off by default, scores
   them as present.
3. The ranking reruns automatically, debounced, only while the panel is open.

## 11. Open questions

1. Wording and placement of the research-use notice for this panel (gate 2).
2. When to move the index into the phentrieve-data release (option A).
