# CRITICAL recall root-cause report

> Historical note: the detailed trace below records the original v0.1-to-v0.3
> safety correction. The active v0.4 candidate now has populated immutable
> 110/30/30/30 regression partitions, an offline semantic cascade, normalized F1
> `0.9095`, and coverage `145 FULL / 26 REDUCED / 29 PARTIAL / 0 UNASSESSED`.
> CRITICAL detection and severity recall remain `1.0000`. The historical numbers
> are intentionally retained for auditability.

## Executive finding

The original four CRITICAL gold discrepancies were all detected before this change. The
blocking `0.0000` CRITICAL recall was entirely a severity-mapping failure: the v0.1 policy
received only finding type, category, claim type, and reason code, so it could not distinguish
a routine mismatch from a tenfold insulin dose, a critical potassium value, a severe allergy
omission, or crushing chest-pain omission.

No gold labels were changed. No semantic model or LLM was added. The primary root-cause count
is:

| Primary root cause | Cases |
| --- | ---: |
| `SEVERITY_MAPPING_ERROR` | 4 |

The complete machine-readable traces are in
`benchmark/clinical_faithfulness/v0/critical-diagnostics.json`.

## Original CRITICAL cases

| Case | Source | Generated output | Gold type / category / severity | Baseline prediction | Detected? | Severity correct? | Primary cause |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cf-v0-007` | Use insulin glargine 10 units nightly. | Use insulin glargine 100 units nightly. | `ALTERED / medication_dose / CRITICAL` | `ALTERED / medication_dose / HIGH` | Yes | No | `SEVERITY_MAPPING_ERROR` |
| `cf-v0-026` | Serum potassium is 4.1 mmol/L. | Serum potassium is 6.1 mmol/L. | `ALTERED / lab_value / CRITICAL` | `ALTERED / lab_value / HIGH` | Yes | No | `SEVERITY_MAPPING_ERROR` |
| `cf-v0-035` | The patient has a severe penicillin allergy. | No allergy information is documented. | `OMITTED / important_omission / CRITICAL` | `OMITTED / allergy / HIGH` | Yes (normalized label match) | No | `SEVERITY_MAPPING_ERROR` |
| `cf-v0-037` | The patient reports crushing chest pain. | The patient reports feeling unwell. | `OMITTED / important_omission / CRITICAL` | `OMITTED / symptom / MODERATE` | Yes (normalized label match) | No | `SEVERITY_MAPPING_ERROR` |

### Pipeline trace summary

- `cf-v0-007`: deterministic extraction linked insulin to `10 units` and `100 units`; exact
  medication alignment scored `1.0`; `DOSE_MISMATCH` fired at `0.99`; semantic evaluation was
  not called; v0.1 mapped all medication-dose alterations to HIGH.
- `cf-v0-026`: deterministic extraction linked potassium to `4.1 mmol/L` and `6.1 mmol/L`;
  exact lab-test alignment scored `1.0`; `NUMERIC_VALUE_MISMATCH` fired at `0.99`; semantic
  evaluation was not called; v0.1 mapped all lab-value alterations to HIGH.
- `cf-v0-035`: the source allergy claim was extracted and the empty generated claim set caused
  the source-to-output pass to emit an omission. Coverage correctly remained PARTIAL. The v0.1
  policy had no access to the word `severe` and mapped allergy omissions to HIGH.
- `cf-v0-037`: the source chest-pain claim was extracted and the empty generated claim set
  caused an omission. Coverage correctly remained PARTIAL. The v0.1 policy had no access to
  the word `crushing` or the chest-pain concept and mapped symptom omissions to MODERATE.

Extraction, assertion, normalization, relation linking, alignment, deterministic rules,
semantic evaluation, and discrepancy classification were reviewed for each original case.
None was the primary reason for zero normalized CRITICAL recall.

## Implemented correction

The provisional severity policy is now `provisional-clinical-severity@0.3.0` and receives the
aligned source/generated concepts, structured attributes, and rule values. Its explicit
critical escalation rules are:

- order-of-magnitude dose change;
- dose change involving a configured high-risk medication;
- configured critical numeric bounds (currently potassium only: `<=2.5` or `>=6.0`);
- allergy contradiction;
- negation reversal for a small configured high-risk concept set;
- qualified severe allergy omission; and
- qualified high-risk symptom omission; and
- unsupported addition of a configured high-risk diagnosis or medication.

The extractor now preserves `severe` allergy and `crushing` symptom qualifiers and their exact
evidence spans. It also recognizes anaphylaxis and pregnancy-test concepts. Relation linking
now attaches laterality to procedures. A safe singleton-allergy alignment is allowed only when
one side is the generic no-known-allergy concept and assertion polarity differs. An explicit
`ALLERGY_CONTRADICTION` deterministic rule then fires.

High-risk omissions with an explicit qualifying adjective, plus configured high-risk medication
omissions, are classified directly as `important_omission`; generic omissions retain their old
classification. Ordinary twofold metformin and noncritical potassium changes remain HIGH, so the
fix is not a blanket severity increase.

## Before and after on the same 40 cases

| Metric | Baseline v0.1 | Fixed engine on original 40 |
| --- | ---: | ---: |
| Raw F1 | 0.4598 | 0.5287 |
| Normalized F1 | 0.8966 | 0.8966 |
| CRITICAL discrepancy-detection recall | 1.0000 (reconstructed) | 1.0000 |
| Normalized CRITICAL severity recall | 0.0000 | 1.0000 |
| Raw TP / FP / FN | 20 / 25 / 22 | 23 / 22 / 19 |
| Coverage FULL / REDUCED / PARTIAL / UNASSESSED | 33 / 1 / 5 / 1 | 33 / 1 / 5 / 1 |

The unchanged normalized F1 on the controlled 40-case comparison is expected: all four
discrepancies were already normalized label matches. The safety gain is visible in the separate
severity metric, while raw labels/evidence also improved for the two important omissions.

## Expanded development benchmark

The v0.2 milestone added ten targeted engineering cases. The reproducible v0.3 expansion adds
150 unique cases, exactly 25 for each finding type, bringing the corpus to 200 cases and 202
findings. The case specifications are engineering-authored; the builder computes identifiers,
exact evidence offsets, counts, hashes, manifest coverage, and split membership.

The broader corpus reports normalized F1 `0.6087`, normalized CRITICAL discrepancy-detection
recall `1.0000`, normalized CRITICAL severity recall `1.0000`, and coverage
`148 FULL / 1 REDUCED / 50 PARTIAL / 1 UNASSESSED`. The lower aggregate score is retained:
normalized `INFERRED` and `UNCERTAIN` recall are both zero because no semantic evaluator is
configured, while all eight deterministic safety slices retain normalized recall `1.0000`.

The expansion exposed unsupported additions of stroke, pulmonary embolism, warfarin, apixaban,
and digoxin being under-severity. Policy v0.3 now handles those explicit high-risk additions as
CRITICAL. During pre-publication annotation review, five newly authored omission labels were
changed from CRITICAL to HIGH: isolated medication mentions without dose or medication-
reconciliation context, plus isolated severe abdominal pain, did not establish CRITICAL
materiality. No original benchmark label was changed.

At the time of this historical report all 200 cases were treated as development
data because they were visible during implementation. v0.4 now assigns immutable
regression partitions, while explicitly recording that historical exposure so no
partition is presented as independent clinical validation.

## Fine-tuning decision

| Component | Baseline evidence | After-rule evidence | Remaining issue | Fine-tune now? |
| --- | --- | --- | --- | --- |
| Deterministic extraction | All original CRITICAL facts extracted | Qualifiers and requested concepts are structured | Vocabulary remains intentionally bounded | No |
| Assertion rules | Original polarity cases worked | New high-risk polarity regressions work | Broader scope-language validation is still needed | No |
| Alignment | Original critical pairs aligned or correctly omitted | Allergy generic contradiction and procedure laterality are covered | Narrative ambiguity remains reduced/uncertain | No |
| Severity | 0/4 original CRITICAL severities matched | 4/4 original and 25/25 v0.3 CRITICAL severities match after normalized taxonomy mapping | Clinician calibration absent | No; collect independent labels first |

## Clinical and evaluation limitations

- The severity rules are provisional engineering safeguards, not clinically validated decision
  thresholds.
- The corpus meets the 200-case engineering size target, but all cases are synthetic development
  data and the expansion annotations have not been clinically reviewed.
- The named validation, held-out, and calibration regression partitions are
  populated, but all were historically exposed and none is clinician-adjudicated.
- The active v0.4 result has zero UNASSESSED and 29 PARTIAL cases; fail-closed
  coverage semantics are unchanged.
- The critical numeric threshold table currently covers potassium only.
- No production terminology service, external NLI/judge runtime, or clinician
  adjudication has been added; the semantic candidate uses the offline cascade.
- `clinician-calibration-template.json` is populated and blinded, but contains no
  reviewer judgments.
