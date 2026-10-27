# Clinical Faithfulness Benchmark

Benchmark v0.4 is a manually authored engineering baseline for the Clinical
Compare Engine. Its 200 records are synthetic, contain no PHI, and deliberately
include both faithful outputs and seeded discrepancies.

- Canonical data: `v0/cases.jsonl` (one `test_case_schema.json` object per line)
- Manifest and coverage map: `v0/manifest.json`
- Contract version: `1.0.0`
- Benchmark version: `0.4.0`
- Canonical taxonomy: `FindingCategory` `1.0.0`
- Split manifest: `v0/splits.json` (110/30/30/30, historically exposed)
- Semantic engineering subset: `v0/semantic-gold.jsonl` (64 cases)
- Checked result: `v0/benchmark-result.json` (artifact schema `1.2.0`)

Gold findings include exact source/generated evidence spans. `SUPPORTED` annotations are intentional: they exercise exact, normalized, and semantic-equivalence behavior alongside `ADDED`, `ALTERED`, `OMITTED`, `INFERRED`, and `UNCERTAIN` outcomes. Severity labels are provisional engineering annotations and require clinician calibration before clinical use.

The result artifact records corpus/manifest SHA-256 digests, component-version
sets, raw and manifest-normalized metrics, explicit FP/FN denominator scopes,
gold-evidence match rates, and critical/critical-value recall slices. Run
`python -m scripts.verify_benchmark` from the repository root to regenerate and
compare it byte-for-byte. Coverage comes from the evaluator result, never from
inspection of the gold label.

The held-out names describe immutable regression partitions, not independent
clinical validation: all main-corpus cases were visible before the v0.4 split
freeze. All gold and severity labels remain engineering annotations.
