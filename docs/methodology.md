# Methodology

## What Gatekeeper evaluates

Gatekeeper sends controlled test inputs to a registered staging system and
evaluates the system's captured output. It does not generate the customer's
clinical response. Test selection begins with a versioned application profile,
so irrelevant families (for example RAG tests for a non-RAG system) are not run.

The implemented clinical-faithfulness evaluator turns source and generated text
into structured claims and compares them in both directions:

1. output to source detects supported, added, altered, inferred, and uncertain
   content;
2. source to output detects supported, omitted, and uncertain content.

Every discrepancy carries exact character evidence when the relevant document
contains evidence. Evidence offsets are checked against the original text.

## Deterministic and semantic boundaries

Deterministic, versioned rules own medication identity, dose, unit, frequency,
route, laterality, assertion/negation, subject, numeric value, date, and duration
comparisons. The built-in offline cascade addresses unresolved terminology,
bounded sentence context, and explicitly supported multi-claim inference. An
optional pretrained NLI resolver and controlled JSON judge can be injected, but
no external semantic runtime ships in this delivery and the reserved environment
flag is rejected rather than treated as active. Low confidence, evaluator
failure, and unsupported inference become uncertain. No semantic resolver can
override an exact high-risk mismatch. If optional NER fails, deterministic
extraction continues with REDUCED coverage instead of receiving a silent FULL
result.

The optional NER candidate is `d4data/biomedical-ner-all`, pinned to revision
`015a4050c9ac99722e61c547aa9b4282bcbedc7f`. It is disabled by default, loaded
lazily, and is not production-accepted. See `model-card-clinical-ner.md`.

## Benchmark composition and metrics

Benchmark v0.4 contains 200 synthetic pairs covering all six finding types and
high-risk medication, laterality, assertion, laboratory, date/duration, subject,
omission, and inference behavior. Gold evidence uses exact offsets. The manifest
records corpus provenance, canonical taxonomy version 1.0.0, compatibility
reason aliases, and a content digest. Its 110/30/30/30 partitions are immutable,
but all cases were visible before the split freeze and are not independent
clinical validation.

User-created Phase-1 test-suite inputs and expectations are stored as plaintext
JSON. The suite API accepts them only with a `SYNTHETIC` or `DE_IDENTIFIED`
classification, `contains_phi: false`, and an explicit no-PHI attestation. The
attestation is an eligibility boundary, not automated de-identification.

The harness reports micro precision, recall, F1, a prediction-conditioned
false-positive fraction (FP / (TP + FP), equivalently false-discovery rate), and
false-negative rate (FN / (TP + FN)) globally and by finding type and category.
Classical FP / (FP + TN) is not identifiable because finding extraction has no
enumerable true-negative universe, so the artifact labels the denominator. It
also reports exact case match, evidence validity, gold-evidence match, coverage,
severity-aware CRITICAL-only recall, a separately named HIGH-or-CRITICAL recall,
and critical-value (`lab_value`) recall. It reports two views:

- **raw** requires the engine's exact category and reason code;
- **normalized** applies only aliases declared in the benchmark manifest.

The normalized score is useful while rule names evolve, but it never replaces
the raw score. A finding counts toward either severity-recall metric only when
its label and gold evidence match and its produced severity is at least the gold
severity. The critical metric's denominator contains only CRITICAL gold findings;
HIGH findings appear only in the separately labeled combined metric.

Current checked-in result: normalized precision/recall/F1
`0.8375`/`0.9950`/`0.9095`, raw F1 `0.6516`, evidence-slice validity `1.0000`,
exact gold-evidence match `0.7100`, normalized gold-evidence match `0.9750`,
normalized false-positive fraction `0.1625`, normalized false-negative rate
`0.0050`, and normalized CRITICAL detection and severity recall `1.0000`.
Coverage is 145 FULL, 26 REDUCED, 29 PARTIAL, and zero UNASSESSED. The CLI exits
`2` whenever any future case is unassessed.

Coverage status comes only from the evaluator result; the harness never reads a
gold label to upgrade or downgrade coverage. The deterministic checked artifact
records the corpus and manifest hashes, benchmark/artifact versions, and
component-version sets, and CI regenerates and compares it byte-for-byte.

## Scoring, coverage, and release decisions

Score, coverage, execution state, and release decision are separate concepts.
Coverage is FULL, REDUCED, PARTIAL, or UNASSESSED. A technical failure does not
become a clinical failure and cannot become PASS. Versioned release policies
produce PASS, WARN, or BLOCK only from eligible results; a critical blocking
finding cannot be averaged away.

Plan review and run creation are separate operations. The API first persists a
DRAFT plan. The caller reviews the exact plan version and application-profile
fingerprint, then submits both expected values to the confirmation endpoint. A
successful confirmation records the API-key actor. Run creation accepts that
confirmed `test_plan_id`; the plan also binds the exact test-suite ID and
version, ordered source/executable case-version IDs, current case inputs and
expectations, suite definition, and release-policy rules/coverage. The API
recomputes those content digests and rejects an unconfirmed, stale, mutated,
cross-system, cross-suite, or profile-mismatched artifact before placing the
durable run in QUEUED state.

## Human calibration

The repository contains an in-memory structural/pilot workflow for blinded
independent review, adjudication, and agreement measurement, plus a populated
30-case blinded export. It is not durable, does not verify clinician credentials,
and has not been used to complete actual clinician review. Production acceptance requires a durable
encrypted implementation, verified reviewer operations, representative held-out
data, agreed severity thresholds, reviewer agreement, slice-level error analysis,
and documented approval by the deployment's clinical safety owner.

## What is not established

This implementation does not establish clinical efficacy, patient safety,
regulatory compliance, legal compliance, fairness, security, or suitability for
uncontrolled production data. The 15 non-faithfulness evaluator families are
currently structural and explicitly return UNASSESSED.
