# Converted PhenoBERT corpora (deprecated)

**Deprecated and no longer maintained.** This directory holds a converted copy
of the public PhenoBERT corpora (`GSC_plus`, `ID_68`, `GeneReviews`). It is
kept only as a format fixture for the tests. Removal is planned for a later
release.

## Known incomplete

The copy is not a complete conversion of the upstream annotations (#520).
`PATTERN_PROCESSED` in `scripts/phenobert_converter.py` accepts only an empty
or numeric fifth column, so annotation rows with a text label in that column
were dropped (122 rows in `ID_68`). Further rows were removed by the
converter's span-text check.

| Dataset | Upstream annotation rows | Rows in this copy | Distinct per-document HPO IDs lost |
| --- | --- | --- | --- |
| `ID_68` | 866 | 678 | 175 of 800 (55 of 68 documents) |
| `GSC_plus` | 2,082 | 2,073 | 4 of 1,547 |
| `GeneReviews` | 352 | 346 | 1 of 238 |

The converter will not be repaired and this copy will not be regenerated.

## Do not use for reported results

This copy is not suitable for reported benchmark results. Scores computed on
it are measured against incomplete gold annotations. The benchmark loader logs
a warning when `ID_68` is loaded.

To benchmark Phentrieve, bring your own data in the documented format (see
`docs/user-guide/benchmarking-guide.md`): a directory with
`<dataset>/annotations/*.json` documents, or a single JSON file.
