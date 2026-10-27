# Semantic evaluator methodology

## Frozen relationship definitions

- `SUPPORTED`: source states the same clinical fact explicitly or through a
  terminology, abbreviation, assertion, or normalized quantity equivalent.
- `INFERRED`: the output is not explicit in the source, every required support
  claim is identifiable, and a versioned bounded derivation rule is satisfied.
- `UNCERTAIN`: evidence, attribution, time, alignment, evaluator availability,
  confidence, or cross-evaluator agreement is insufficient for a reliable label.
- `ALTERED`: a deterministic attribute or semantic contradiction materially
  changes meaning. It always takes precedence over semantic similarity.

Finding relationship and clinical dimension are independent fields. The
canonical category taxonomy is `FindingCategory` version 1.0.0. Legacy values
are accepted only by the Pydantic input boundary and are never emitted by the
engine or used as primary benchmark category aliases.

## Cascade

The default offline path is:

1. deterministic structured alignment and high-risk comparison rules;
2. canonical terminology and normalized concept alignment;
3. bounded semantic-context contradiction/ambiguity checks;
4. versioned multi-claim inference rules with explicit source support;
5. optional pretrained NLI for unresolved narrative pairs;
6. optional schema-validated controlled judge only below the NLI confidence
   threshold;
7. `UNCERTAIN` when the optional evaluator fails, times out, disagrees, returns
   malformed output, or remains below confidence.

The NLI confidence threshold is 0.85. Biomedical embedding acceptance remains
0.90 with a minimum 0.05 margin, but embeddings are only an evidence-candidate
alignment mechanism. They never establish clinical entailment.

Deterministic rule outcomes are calculated before semantic context. Dose,
frequency, route, laterality, assertion, allergy, and numeric contradictions
cannot be replaced by a semantic `SUPPORTED` decision. Multi-claim inference
stores all source-support spans. Uncertain findings store an uncertainty reason
and candidate evidence spans.

## Data and splits

The 200-case benchmark uses a byte-reproducible 110/30/30/30 split assignment.
All 200 cases were visible before the v0.4 split freeze, so this is a regression
holdout from v0.4 onward and not independent clinical validation.

The dedicated 64-case semantic subset contains 16 engineering annotations for
each of `SUPPORTED`, `INFERRED`, `UNCERTAIN`, and `CONTRADICTED`, with 16 cases
in each split. No label is represented as clinician gold. The semantic candidate
was frozen before the held-out semantic split was executed once.

## Results and coverage

On benchmark v0.4, normalized precision/recall/F1 are 0.8375/0.9950/0.9095.
`SUPPORTED` recall is 1.0000 and F1 is 0.7477; `INFERRED` precision/recall/F1 are
0.9630; and `UNCERTAIN` precision/recall/F1 are 0.8929/1.0000/0.9434. Normalized
CRITICAL detection and severity recall remain 1.0000. All eight deterministic
safety slices remain 1.0000.

Coverage is 145 FULL, 26 REDUCED, 29 PARTIAL, and zero UNASSESSED. PARTIAL is
retained when one side has no extracted clinical claim; ambiguity remains
REDUCED. The evaluator does not promote coverage merely to improve metrics.

The separate semantic subset reaches 0.8906 accuracy and 0.8909 macro F1 on all
64 engineering labels. Its once-run held-out slice reaches 0.8125 accuracy and
0.8185 macro F1, with two PARTIAL and two UNASSESSED cases. These results are
engineering evidence only.

## Reproducibility

Artifacts record exact corpus and split hashes, evaluator versions, model names
and revisions, thresholds, dependency versions, source revision state, Python
version, and dependency-lock hash. The default cascade is offline and
deterministic. No external LLM was called. A future controlled judge must record
provider/model revision, prompt version, temperature, seed where supported, and
reproducibility limitations.
