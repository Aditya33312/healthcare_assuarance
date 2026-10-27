# Gatekeeper architecture

## Safety boundary

Gatekeeper consumes known source evidence and output produced by an external AI
system. It does not produce the customer's clinical output. All web, API, and CI
entry points call the same application services and evaluator router.

## Evaluation path

```text
Application profile -> deterministic test selection -> DRAFT test plan
  -> exact version/fingerprint review -> explicit confirmation
  -> orchestrator -> customer adapter -> captured output -> evaluator router
  -> clinical NLP (where applicable) -> two-way compare -> findings
  -> test-specific scoring -> coverage -> versioned release policy
  -> PASS | WARN | BLOCK | no decision
```

Execution status and release decision are separate fields. A failed execution
has no release decision. Required unassessed coverage can block or withhold a
decision according to the selected release policy.

Direct evidence evaluations use the same clinical evaluator behind an explicit
ingestion boundary:

```text
Typed HTTP evidence
  -> schema and complexity validation
  -> canonical evidence units and content hashes
  -> normalization, sectioning, and deterministic size-aware chunking
  -> provenance-linked clinical claim extraction
  -> two-way Clinical Compare Engine
  -> findings, severity, coverage, and encrypted persistence
```

Transport schemas, ingestion, clinical NLP, comparison, and persistence are
separate responsibilities. The Compare Engine receives normalized evidence and
does not need to interpret each public transport shape. Ingestion retains raw
content and exact document, JSON-path, RAG-chunk, or transcript-segment provenance;
it does not silently truncate or turn a parse failure into support. See
[`universal-evidence-ingestion.md`](universal-evidence-ingestion.md).

## Clinical Compare Engine

The compare engine executes two passes:

1. output to source identifies supported, added, altered, inferred, or uncertain
   output claims;
2. source to output identifies supported, omitted, or uncertain source claims.

Exact normalized rules have priority for laterality, medication identity, dose,
unit, frequency, route, assertion, certainty, subject, numeric value, date, and
duration. The default offline semantic cascade then applies canonical terminology,
bounded sentence-context contradiction and ambiguity checks, and versioned
multi-claim inference with identifiable support. Optional pretrained NLI and a
strict JSON controlled judge may be injected at the final boundary, but no such
external runtime is configured by this delivery. Low confidence, disagreement,
timeouts, malformed output, and unsupported derivations fail closed to
`UNCERTAIN` or reduced coverage. Semantic matching cannot override a deterministic
mismatch.

Every discrepancy is accompanied by the exact source and/or generated character
span, a versioned reason code, confidence, evaluator version, and provisional
deterministic severity. Inferred findings also identify their source-support
spans; uncertain findings preserve an uncertainty reason and candidate evidence.

## Failure boundaries

- Model failures reduce coverage; they never silently pass a case.
- Ambiguous alignment is uncertain; it is not guessed.
- Unsupported test families are explicitly unassessed.
- Customer endpoint authentication failures stop the run.
- Transient endpoint failures use bounded exponential backoff with jitter.
- Clinical content is excluded from normal operational logs.
- Tenant-owned database rows carry `organization_id` and repository methods
  always scope by it.

## Extensibility

Evaluator protocols isolate clinical faithfulness, task performance, robustness,
bias, safety, security, RAG, policy, and custom/regression families. Background
execution includes a database-backed worker and still exposes a replaceable queue
protocol so Redis and Celery/RQ/ARQ can be connected without changing evaluator
contracts. The API persists a run as QUEUED, and the separately supervised worker
claims it, invokes the registered endpoint, evaluates the captured output, and
stores the terminal release state.
