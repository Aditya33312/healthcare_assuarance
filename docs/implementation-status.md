# Implementation status

This document is updated at each verified work-package boundary. A capability is
only marked implemented when an automated test exercises it.

## Delivery states

- **Implemented** — executable and covered by tests.
- **Structural** — schema/router support exists, but the evaluator is not yet
  implemented and returns unassessed.
- **Planned** — intentionally absent from this delivery.

Gatekeeper never turns a technical failure or unsupported evaluator into PASS.

## Verified Phase 1 capabilities

| Area | State | Evidence / boundary |
| --- | --- | --- |
| Versioned clinical contracts | Implemented | Six strict JSON schemas with contract tests. |
| Benchmark v0.4 | Implemented engineering set | 200 unique synthetic, offset-validated cases with 202 findings and immutable 110/30/30/30 split membership. All cases were historically visible before the split freeze, so this is not independent clinical validation. |
| Clinical NLP baseline | Implemented | Deterministic preprocessing, entity/claim extraction, assertion, normalization, relations, and lazy optional pinned NER adapter. |
| Clinical Compare Engine | Implemented v1 engineering candidate | Two-way comparison, one canonical taxonomy, exact spans, deterministic high-risk precedence, terminology matching, semantic-context contradiction/ambiguity, bounded multi-claim inference with source support, explicit uncertainty, and optional NLI/embedding/controlled-judge boundaries. |
| Application profiling and test selection | Implemented | All 16 families are catalogued and selection is profile-dependent. |
| Versioned test-plan generation | Implemented | The API persists a DRAFT binding the exact ordered source and executable case-version IDs/digests, suite/case content digest, release-policy content digest, family bindings, exclusions, suite/version, profile fingerprint, actual selection partition, and configurable case counts; confirmation records the API-key actor and run creation rejects stale, mutated, or mismatched snapshots. |
| Clinical-faithfulness evaluator | Implemented | Routes to the Compare Engine. |
| MedHELM/LangTest evaluator adapters | Implemented integration boundary | Optional JSON-in/JSON-out runners can override the local contract evaluators for clinical task/note/medication evaluation and robustness/bias/prompt-security evaluation. Provider commands are intentionally configuration-dependent; missing, invalid, failed, or partial provider results fail closed. |
| Remaining evaluator families | Implemented contract baseline | All 16 families have executable, deterministic assertion evaluators. They require family-appropriate expectations; missing evidence remains unassessed at case execution. MedHELM/LangTest provide optional provider-backed evaluation for their mapped families. |
| Scoring, coverage, severity, release policy | Implemented | Coverage uses confirmed per-family requirements and case counts independently from score; critical findings override averages; missing cases and failures fail closed. |
| REST/OpenAI-compatible adapters | Implemented | Bounded retries and sanitized errors; customer payloads are not logged. |
| Async orchestration | Implemented | Bounded concurrency, durable transition hook, retry policy, output/token/latency/cost capture, database-backed queued-run worker, endpoint adapters, evaluator dispatch, coverage persistence, and release-policy application are covered by tests. A production Redis/Celery/RQ/ARQ deployment remains optional infrastructure. |
| Persistence and tenant isolation | Implemented application boundary | SQLAlchemy schema, Alembic migration, and organization-scoped repositories are tested. Persisted evaluation clinical text, entity extractions, claims, alignment explanations, evidence, and findings are application-encrypted; alignments reference internal persisted claim IDs. Test-case input and expectation JSON remain plaintext and require a synthetic/de-identified no-PHI attestation. Database-level parent/child tenant coupling is not complete. |
| API key authentication and lifecycle | Implemented | HMAC digests, one-time secret return, first-admin bootstrap, tenant-scoped create/list/revoke, scopes, environment separation, expiration, and revocation. |
| TEST/LIVE environment isolation | Implemented application boundary | Production-environment systems, evaluations, runs, findings, and coverage require a LIVE key; TEST-key reads use non-disclosure behavior. This is not a substitute for separate infrastructure and credentials. |
| FastAPI surface | Implemented | Health, systems/versions, suites, evaluations, runs/cases/findings, policies, coverage, standard errors and request IDs. |
| Web portal | Implemented baseline | Shared backend for dashboard, systems, suites, draft generation, ordered case/binding review and plan confirmation, runs, findings, developer/API resources. Production reads/writes require a constant-time trusted-gateway header check, bounded upstream requests, runtime response-contract checks, and a validated HTTPS upstream; writes additionally default to disabled and require exact-origin JSON. Test-run retries reuse a stable attempt id. Credential authoring, endpoint connection testing, historical version browsing, full result tabs, and built-in human identity/RBAC attribution are not implemented. |
| GitHub release gate | Implemented | The gate requires an exact suite version, hashes all resolved inputs plus workflow run/attempt into idempotency, and exits non-zero for BLOCK or technical/indeterminate results. |
| Regression promotion | Implemented workflow | Human validation is required before a finding becomes a versioned regression case. |
| Clinician calibration | Populated blinded package / pending review | Thirty calibration cases exclude automation and engineering gold. The in-memory workflow supports independent review, adjudication, and agreement calculations, but no clinician review has occurred and the repository is not durable. |

## Benchmark v0.4 and semantic-candidate results

The checked-in result was generated from the frozen v1 engineering candidate
and the content-addressed 200-case corpus:

- normalized precision `0.8375`, recall `0.9950`, F1 `0.9095`;
- raw precision `0.6000`, recall `0.7129`, F1 `0.6516`;
- 145 FULL, 26 REDUCED, 29 PARTIAL, and zero UNASSESSED cases;
- exact evidence-slice validity `1.0000`, exact gold-evidence match `0.7100`,
  and normalized gold-evidence match `0.9750`;
- normalized 201 TP, 39 FP, and 1 FN;
- normalized CRITICAL discrepancy-detection recall `1.0000` and separate
  severity-and-evidence-aware normalized CRITICAL recall `1.0000`;
- normalized `SUPPORTED`/`INFERRED`/`UNCERTAIN` recall
  `1.0000`/`0.9630`/`1.0000`;
- normalized recall `1.0000` for the medication-error, dose, route, frequency,
  laterality, negation-reversal, critical-numeric, and allergy-contradiction slices.

The engine and benchmark now emit canonical taxonomy version 1.0.0 categories.
Normalization retains versioned reason-code compatibility only; raw scores remain
visible because exact reason and evidence differences still matter.
The result artifact records corpus and manifest SHA-256 values, benchmark and
artifact schema versions, and every observed component-version set. It is
deterministic and CI regenerates it byte-for-byte; no wall-clock generation time
is embedded. It records the untracked repository state, Python version, and
dependency-lock hash. Its source revision remains an uncommitted candidate marker
until a release commit exists.

The balanced 64-case semantic engineering subset reports accuracy `0.8906` and
macro F1 `0.8909`; its once-run 16-case held-out partition reports accuracy
`0.8125` and macro F1 `0.8185`. The 200-case split partitions and semantic labels
are not independent clinician gold. Severity policy v0.3 remains provisional.

## Phase 1 completion assessment

Status is **PARTIAL**: this is a production-structured engineering candidate, not
a clinically validated release. The completion gate remains open until independent
clinician calibration and truly external held-out validation are completed, the
severity policy is recalibrated from adjudicated evidence, and the intended
deployment is verified with PostgreSQL, migrations, a concrete worker, durable
calibration storage, and the required evaluator families. Production KMS/HSM,
SSO, retention deletion, network controls, and supply-chain pinning also remain
deployment gates rather than repository-complete controls.
