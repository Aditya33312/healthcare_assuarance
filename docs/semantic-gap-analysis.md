# Semantic gap analysis

## Scope and baseline

This analysis freezes the deterministic-only `clinical-faithfulness-v0@0.3.0`
result before taxonomy or semantic-evaluator changes. The source artifact contains
200 synthetic development cases and 202 engineering annotations. None of the
semantic annotations has been clinician adjudicated.

The machine-readable trace is
`benchmark/clinical_faithfulness/v0/semantic-diagnostics.json`. It contains every
case contributing to the 118 normalized false positives, 62 normalized false
negatives, 50 PARTIAL cases, one REDUCED case, or one UNASSESSED case. The union is
118 cases. Each record includes extraction, assertions, normalization, alignment,
rules, missing semantic evaluation, predictions, gold labels, coverage, and one
primary root cause.

## Root-cause counts

| Primary root cause | Cases | Interpretation |
| --- | ---: | --- |
| `BENCHMARK_ANNOTATION_QUESTIONABLE` | 52 | Twenty-five uncertainty-preserving paraphrases are labeled UNCERTAIN, while 27 target-only ADDED/OMITTED annotations omit other valid findings. |
| `INFERENCE_UNSUPPORTED_BY_ENGINE` | 27 | The engine has no configured inference evaluator and cannot identify multi-claim support for a derived statement. |
| `OTHER` | 29 | Labels match, but fail-closed coverage remains PARTIAL because one side has no extracted clinical claim. Most are intentional omission cases. |
| `TAXONOMY_MAPPING_ERROR` | 9 | A supported medication pair preserves multiple dimensions; the engine emits one category while target-only gold expects another. Normalization itself succeeded. |
| `CLAIM_EXTRACTION_ERROR` | 1 | `cf-v0-021` extracts no claim from either mention of effusion and remains UNASSESSED. |

Counts are case-level primary causes and sum to 118. Finding-level normalized FP/FN
counts remain 118/62.

## Benchmark-integrity findings

### UNCERTAIN labels

Cases `cf-v0-176` through `cf-v0-200` use pairs such as “cannot exclude
pneumonia” and “may have pneumonia.” Both claims preserve the same possible
assertion. Under the frozen relationship definitions this is a candidate
SUPPORTED paraphrase, not an uncertain source/output relationship. These cases
must not be used to train or select an UNCERTAIN classifier without independent
review.

`cf-v0-021` has the same annotation concern, plus a genuine extraction gap for
“effusion.” Its primary pipeline failure is therefore recorded as
`CLAIM_EXTRACTION_ERROR`.

### Target-only findings

Cases `cf-v0-101` through `cf-v0-125` correctly detect the unsupported addition,
but the gold annotation omits the valid SUPPORTED headache relationship. The
benchmark therefore counts that supported finding as a false positive. Similar
incompleteness occurs where an omission case contains more extracted facts than
the single target gold finding. These are annotation-scope issues, not evidence
that deterministic rules should suppress valid findings.

### Supported category collisions

Nine supported medication cases preserve more than one dimension. For example,
an alias pair may also preserve frequency, while a route pair may also preserve
frequency. The engine currently chooses one category, and the benchmark expects
another. A canonical multi-dimensional category policy is required before
primary scoring; adding more case-specific aliases would compound the problem.

## Coverage analysis

| Non-FULL reason | Cases | Decision |
| --- | ---: | --- |
| Empty generated claim set, fail closed | 28 | Retain PARTIAL until the coverage policy distinguishes an intentionally empty clinical output from extraction failure using explicit evidence. |
| Unsupported inference path | 22 | Semantic/inference evaluator required; do not mark FULL while unavailable. |
| Empty source claim set | 1 | Retain PARTIAL; generated inference has no extractable source support. |
| Empty both claim sets | 1 | Retain UNASSESSED; `cf-v0-021` requires justified extraction and annotation review. |

The remaining five inference cases are FULL because both sides yield claims, but
their expected inference relation is still missed. One inference case is REDUCED
because explicit inference language produces unresolved semantic candidates.

## Semantic implementation implications

1. Freeze canonical relationship and category definitions before selecting a model.
2. Repair or quarantine questionable engineering annotations before computing
   semantic model accuracy.
3. Build a semantic subset that distinguishes uncertainty *within a claim* from
   uncertainty *about the relationship between source and output*.
4. Treat embeddings only as candidate retrieval; an entailment decision or
   explicit deterministic mapping must decide the relationship.
5. Preserve deterministic contradiction precedence and the current normalized
   CRITICAL/safety-slice recall of `1.0000`.
6. Keep semantic timeouts, malformed responses, disagreement, and low confidence
   fail closed as UNCERTAIN with non-FULL coverage where required.

## UNASSESSED trace

`cf-v0-021`:

- Source: “The scan cannot exclude a small effusion.”
- Output: “A small effusion may be present.”
- Source entities/claims: none.
- Output entities/claims: none.
- Alignment and deterministic rules: no candidates and no fired rules.
- Semantic evaluator: not configured and therefore not called.
- Coverage: UNASSESSED with `No clinical claims were extracted from either input.`

Adding `effusion` to deterministic terminology extraction is technically
reasonable, but the relationship label must also be reviewed because both texts
express the same uncertainty. The case is not silently converted to PASS.

## Post-candidate status

This document and its machine-readable artifact remain the immutable pre-semantic
v0.3 analysis. In v0.4, questionable labels were corrected or rewritten,
effusion extraction was added, multi-dimensional category collisions were
removed through the canonical taxonomy, and the offline inference/uncertainty
cascade was enabled. The current result is 201 normalized TP, 39 FP, and 1 FN,
with coverage `145 FULL / 26 REDUCED / 29 PARTIAL / 0 UNASSESSED`.

The remaining normalized FN is `cf-v0-156`: the bounded inference rule does not
accept its combined contextual target representation, so the candidate returns
uncertainty instead of the engineering `INFERRED` annotation. The dominant
remaining precision limitation is extra valid/support-like findings relative to
target-only engineering annotations, reflected in `SUPPORTED` precision
`0.5970`. These are retained after the held-out freeze rather than tuned away.
