# Disease Ranking (research use)

Phentrieve can rank rare diseases and genes for a profile of HPO terms, such as the
phenotypes collected in the case workspace. It shows how well the profile matches curated
disease annotations, and why.

!!! warning "Research use only, not a diagnosis"
    Scores are raw, uncalibrated phenotype similarity. They are not probabilities and do
    not replace clinical assessment. The feature is off by default. Check the open points
    under [Before you enable it publicly](#before-you-enable-it-publicly) first.

## How it scores

The scorer is Phentrieve's copy of the VarLens phenotype scoring v2 port from issue #451,
at its shipped setting. In short:

- **Similarity between two terms:** the information content of their most informative
  common ancestor. It is computed from the diseases in `phenotype.hpoa`, so rare features
  count more than common ones.
- **Forward score:** for each present profile term, the best-matching annotation of the
  disease, weighted lightly by how often the disease shows it.
- **Reverse score:** how many of the disease's frequent features the profile covers.
- **Total:** `0.55 * forward + 0.45 * reverse`. A gene takes the score of its best disease.
- **Excluded terms** (negated or normal) never change the score. A disease annotated with
  an excluded term, or one of its descendants, gets a conflict note.
- **Uncertain terms** are not scored unless you switch on "Include uncertain terms". They
  then count like present terms.

### ORPHA merge

The index adds, to each OMIM disease, the HPO annotations of the Orphanet entries that
MONDO marks as equivalent (`MONDO:equivalentTo`). For each term the higher frequency is
kept. On the #451 evaluation cases this raised disease Hit@10 by about 4 to 5 percentage
points; gene ranks barely changed (`POSTHOC.md` section 10, branch
`eval/scoring-variants-451`). In the disease list, the ORPHA entry is folded into its
OMIM row and listed as a twin.

## Build the index

```bash
phentrieve data prepare-diseases --hpo-version v2026-09-01 --mondo-version v2026-09-01
```

The command:

1. downloads `hp.json`, `phenotype.hpoa` and `genes_to_phenotype.txt` from the HPO release,
   and `mondo.obo` from the MONDO release;
2. records their SHA-256;
3. writes `<data dir>/disease_index/`.

Without the version options it uses the latest releases. Pin both for reproducible
results. Further options:

- `--data-dir` picks another data directory.
- `--force` downloads the sources again and rebuilds the index.
- An existing index is kept unless `--force` is given, the requested HPO or MONDO
  version differs from the one it was built from (`latest` is not a pin), or the index
  is incomplete or damaged. A running API picks up a rebuilt index on the next request.
- `--no-orpha-merge` builds the plain reference index, without the ORPHA merge.

## Enable it

| Variable | Default | Meaning |
|---|---|---|
| `PHENTRIEVE_DISEASE_RANKING_ENABLED` | `false` | Mounts `POST /api/v1/diseases/rank` and shows the panel in the frontend |
| `PHENTRIEVE_DISEASE_INDEX_DIR` | `<data dir>/disease_index` | Where the API reads the index |
| `PHENTRIEVE_DISEASE_HPO_VERSION` | latest | Docker only: HPO release for the first-start build |
| `PHENTRIEVE_DISEASE_MONDO_VERSION` | latest | Docker only: MONDO release for the first-start build |

With Docker Compose, the API entrypoint builds the index on the first start when the flag
is set and no index exists. If the build fails, the API still starts and the endpoint
returns `503`.

The frontend shows the action only when `GET /api/v1/info` reports
`disease_ranking.enabled` and `disease_ranking.index_ready`.

## In the frontend

1. Collect phenotypes in the case workspace. Mark them as present or excluded.
2. Click **Similar diseases**.
3. Switch between **Diseases** and **Genes**.
4. Expand a row to see the explanation:
    - the matched term pairs, with the relation (same, broader term, more specific,
      related) and the annotated frequency;
    - conflicts with excluded terms;
    - profile terms the disease does not explain;
    - the annotation sources, with OMIM and Orphanet links.

While the panel is open, the ranking updates automatically when the collection changes.
Profiles and results are not stored.

## API

```bash
curl -X POST http://localhost:8734/api/v1/diseases/rank \
  -H 'Content-Type: application/json' \
  -d '{"terms": [{"hpo_id": "HP:0001166"}, {"hpo_id": "HP:0001083"},
                 {"hpo_id": "HP:0001249", "assertion": "negated"}],
       "level": "disease", "top_k": 20}'
```

The request takes:

- `terms`: a list of `{hpo_id, assertion}` entries. `assertion` is one of `affirmed`,
  `present`, `negated`, `absent`, `excluded`, `normal` or `uncertain`.
- `level`: `disease` or `gene`.
- `top_k`: 1 to 100.
- `include_uncertain`: whether uncertain terms are scored.

The response holds:

- the ranked rows, each with its explanation;
- the scored, excluded and not-scored terms;
- the index versions;
- an `intended_use` notice.

In public-hosted mode the research-use acknowledgement header is required, as on the
other text-to-HPO routes.

## Before you enable it publicly

- **Evidence:** the ORPHA merge gain was measured post hoc on synthetic letters (#451). A
  registered confirmation run is still planned (#508).
- **Intended use:** a ranked disease list shown to clinicians may count as a medical
  device under the EU MDR. Decide on the intended use first.
- **Licences:**
    - The HPO documentation asks for OMIM's permission to reuse OMIM-derived annotations
      in other software products.
    - Orphadata Science and MONDO are CC BY 4.0. They require attribution and a note on
      changes, and Orphanet adds an AI-use statement.
    - The panel footer carries the attribution.

Design: `.planning/specs/2026-11-23-disease-ranking-design.md`.
