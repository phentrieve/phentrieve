# Disease Ranking: Implementation Plan

- Date: 2026-11-23
- Spec: `.planning/specs/2026-11-23-disease-ranking-design.md`
- Branch: `feat/disease-ranking` (from `main`), one PR, one commit per layer
- Status: active (implemented on `feat/disease-ranking`, PR open; gates in the spec
  still decide public enablement)

## Ground rules

- VarLens is not changed. The scorer is copied from `scripts/disease_ranking_451/scorer.py`
  (branch `eval/disease-ranking-451`) and `enrichment.py` (branch `eval/scoring-variants-451`),
  then cleaned up. Nothing imports from `scripts/`.
- The feature flag `PHENTRIEVE_DISEASE_RANKING_ENABLED` is off by default.
- Unit tests use small invented fixtures under `tests/data/disease_matching/`, and no network.

## Layer 1: library and index (`phentrieve/disease_matching/`)

1. `index_builder.py`:
   - parse hp.json (labels, is_a, alt_id and replaced_by), `phenotype.hpoa`,
     `genes_to_phenotype.txt` and MONDO equivalences;
   - build the closure, IC, the universe and the ORPHA merge;
   - store the twin groups;
   - write `disease_index.npz` and `disease_index.json` (versions, SHA256, counts, licence notes).
2. `index.py`: the `DiseaseIndex` dataclass, `load_index(data_dir)` with a checksum check, and
   `IndexNotAvailableError`.
3. `scorer.py`: `score(index, weights) -> np.ndarray`, with the parameters frozen as module
   constants. Use the vectorised form from the PoC.
4. `service.py`: `rank(index, terms, level, top_k, include_uncertain)`. It covers:
   - mapping assertions through `assertion_vocab`;
   - folding twins into the OMIM row;
   - the gene level;
   - explanations for the top-k rows (matched pairs, conflicts, unexplained terms);
   - the not-scored list.
5. CLI: `phentrieve data prepare-diseases` in `phentrieve/cli/data_commands.py`. It downloads
   the inputs with pinned versions and checksums, then calls the builder.
6. Tests in `tests/unit/disease_matching/`:
   - conformance against the frozen `varlens_reference.json`, copied to `tests/data/` (merge
     disabled for this test);
   - a mini HPOA and MONDO fixture for the merge rule, the higher-frequency rule and obsolete
     MONDO terms;
   - twin folding, assertion handling, unknown and obsolete IDs, and the gene level;
   - the explanation contents.

## Layer 2: API

1. `api/schemas/disease_ranking_schemas.py`: request and response models.
2. `api/routers/disease_ranking_router.py`: `POST /api/v1/diseases/rank` with the research-use
   gate. It returns 422 for an empty or invalid profile and 503 without an index.
3. In `api/dependencies.py`, a lazily loaded, cached index.
4. In `api/config.py`, the flag. In `api/main.py`, mount the router only when the flag is set.
5. `config_info_router`: add `disease_ranking: {enabled, index_ready, versions}`.
6. Docker: the entrypoint runs `prepare-diseases` when the flag is set and the index is missing.
7. Tests in `tests/unit/api/`:
   - the flag on and off;
   - 503, 422 and the research-use gate;
   - the response shape;
   - config-info.

## Layer 3: frontend

1. `PhentrieveService.rankDiseases()`.
2. A Pinia store `diseaseRanking.js`. It reruns automatically, debounced (about 400 ms), while
   the panel is open, drops stale responses, and persists nothing.
3. `DiseaseRankingPanel.vue`. It follows the mockup:
   - the disease/gene toggle and the uncertain switch;
   - the status line, the list with expandable explanations, and conflicts;
   - the research-use notice and the attribution footer.
4. In `PhenotypeCollectionPanel.vue`, the "Similar diseases" action. It is shown only when
   config-info reports the feature as enabled and ready.
5. i18n keys in de, en, es, fr and nl.
6. Vitest tests for the store (debounce, stale responses), the service, the panel and its
   visibility.

## Layer 4: docs

- User guide section, API docs, `CHANGELOG.md` entry (flagged, research use).

## Checks before the PR

- `make check`, `make typecheck-fast`, `make test`
- `make ci-frontend`, `make frontend-i18n-check`
- Manual: build the index locally, run `make dev-api` and `make dev-frontend` with the flag on,
  and check the panel against the mockup.

## Open (from the spec)

- Wording of the research-use notice (gate 2).
- Moving the index into the phentrieve-data release (option A).
