# Healthcare AI Gatekeeper

Gatekeeper is a healthcare AI assurance platform. It evaluates the output of a
system under test; it does not generate customer clinical content.

The first implementation is deliberately layered:

1. versioned clinical contracts and benchmark data;
2. deterministic clinical extraction and normalization;
3. a two-way Clinical Compare Engine with exact evidence;
4. test selection, orchestration, coverage, scoring, and release policy;
5. a shared FastAPI surface for manual, API, and CI usage.

The deterministic engine is always available. Optional pretrained NER can add
candidate entities. The default offline semantic cascade provides terminology,
bounded context, explicit inference support, and fail-closed uncertainty. A
pretrained NLI model or controlled JSON judge can be injected programmatically,
but no external semantic runtime is configured in this delivery; the reserved
environment setting is rejected instead of silently doing nothing. Optional
components cannot override exact high-risk rules for dose, unit, frequency,
route, laterality, assertion, or numeric mismatches.

## Local development

The supported runtime is Python 3.12.4 or later. Earlier 3.12 patch releases
contain outdated `ipaddress.is_global` classifications and are not supported at
the customer-endpoint security boundary.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
pip install -e . --no-deps
cp .env.example .env
python -c 'import secrets; print(secrets.token_urlsafe(48))'
python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
```

Before starting PostgreSQL or running a migration, replace
`GATEKEEPER_API_KEY_PEPPER` in `.env` with the first generated value and
`GATEKEEPER_DATA_ENCRYPTION_KEYS` with the second. Also replace the example
database credentials as appropriate for the target environment. The values in
`.env.example` are placeholders; the encryption-key placeholder is not a valid
Fernet key.

```bash
docker compose up -d postgres
alembic upgrade head
python -m scripts.bootstrap_admin \
  --organization-name "Local Gatekeeper" \
  --key-environment test \
  --confirm-empty-database
pytest
uvicorn backend.app.main:app --reload
```

Clinical text is intentionally excluded from ordinary application logs.

Run the bootstrap command exactly once after migrating a fresh database. It
refuses to run unless both the organization and API-key tables are empty, creates
an active organization and an API key with all scopes, and displays that key only
once. Store it in a secret manager; Gatekeeper persists only its keyed hash. Use
`--key-environment live` for a live deployment and verify the configured database
before passing `--confirm-empty-database`. The API-key pepper used for bootstrap
must remain available for verification of keys created with it; do not replace it
after issuing keys without a designed key-rotation and reissuance procedure.

For a command-by-command API flow that creates prerequisite governance artifacts,
recovers a draft with `GET`, confirms it, and queues a run, see
[`docs/api-quickstart.md`](docs/api-quickstart.md).

## Universal evidence ingestion

The synchronous `POST /v1/evaluations` endpoint accepts backward-compatible
`clinical_text` plus typed clinical notes, structured transcripts, laboratory
results, medication lists, controlled structured JSON, and RAG context. It
canonicalizes and size-aware chunks supported evidence before the existing
Clinical Compare Engine while retaining raw provenance. Configured limits fail
closed and never silently truncate evidence.

For request examples, multiline guidance, supported types, provenance, limits,
coverage behavior, and the file-submit helper, see
[`docs/universal-evidence-ingestion.md`](docs/universal-evidence-ingestion.md).
For example, keep the API key out of shell history and submit text files with:

```bash
export GATEKEEPER_API_KEY='<managed Gatekeeper API key>'
python scripts/evaluate.py \
  --source transcript.txt \
  --output note.txt \
  --output-type clinical_note \
  --system-id sys_xxx \
  --system-version 1.0.0
```

Normal Python/HTTP clients should pass actual multiline strings to their JSON
serializer. Swagger's JSON editor still requires syntactically valid serialized
JSON and is best kept for smaller examples.

The web portal lives in `frontend/`:

```bash
cd frontend
npm install
npm run dev
```

Its Gatekeeper API key is read only by server-side portal code. Configure
`GATEKEEPER_API_URL` and `GATEKEEPER_API_KEY` from `frontend/.env.example`; never
expose that key through a `NEXT_PUBLIC_` variable. Production requires an HTTPS
API URL without credentials, a query, or a fragment. Plain HTTP is accepted only
for localhost development. Do not reuse the all-scopes bootstrap key. Issue a
dedicated, expiring portal key with only `systems:create`, `systems:read`,
`test-plans:manage`, `test-runs:create`, and `test-runs:read`; the implemented
portal does not need API-key, release-policy, suite-authoring, evaluation, or
report-management scopes.

The portal does not implement end-user identity or role-based authorization, and
plan confirmation is attributed to the shared portal API key rather than to a
human account. In production,
a trusted identity-aware gateway must remove the configured dedicated header
from inbound traffic and inject the exact private value configured through
`GATEKEEPER_PORTAL_TRUSTED_HEADER` and `GATEKEEPER_PORTAL_TRUSTED_VALUE` (generate
the value with `openssl rand -hex 32`). All reads and mutation proxies reject a
missing or incorrect value. Development may leave both variables empty; setting
only one, or setting an invalid pair, fails closed in every environment. Portal
writes additionally default to off, require
`GATEKEEPER_PORTAL_WRITES_ENABLED=true`, accept only JSON from the exact HTTPS
origin in `GATEKEEPER_PORTAL_ALLOWED_ORIGIN`, and reject cross-site fetch
metadata. The gateway should also retain SameSite cookies and its own CSRF
protections.

API test runs use a separate review boundary: create a DRAFT test plan, review
its exact version, suite/version, selected counts, ordered executable case-version
snapshot/digest, suite-and-case content digest, release-policy content digest,
explicit exclusions, and profile fingerprint,
explicitly confirm that same artifact, then create the run with the confirmed
`test_plan_id` and exact `test_suite_version`. Creating a run is not itself
confirmation. Drafts deterministically cap each selected family at its configured
count; cases beyond quota, inactive cases, not-applicable families, and unselected
optional families remain audit-visible exclusions and are never queued.

Phase-1 test-suite case inputs and expectations are stored as plaintext JSON.
Suite creation therefore requires `SYNTHETIC` or `DE_IDENTIFIED` classification,
`contains_phi: false`, and an explicit `no_phi_attested: true`; do not submit PHI
to the suite API.

## Verification and benchmark

```bash
ruff check backend alembic scripts
alembic upgrade head
alembic check
pytest
python -m scripts.build_benchmark_expansion
python -m scripts.build_semantic_gold_subset
python -m backend.app.benchmark.cli \
  benchmark/clinical_faithfulness/v0/cases.jsonl \
  --output benchmark/clinical_faithfulness/v0/benchmark-result.json \
  --repository-state UNTRACKED_WORKSPACE
python -m scripts.verify_benchmark
python -m scripts.benchmark_ingestion
```

A release pipeline can add `--source-revision "$GITHUB_SHA"` and an exact
repository-state descriptor when emitting a separate release artifact. The
checked deterministic workspace artifact records `UNTRACKED_WORKSPACE` because
no baseline commit was authorized. It also records the Python version and the
SHA-256 of `requirements.lock`.

The benchmark command deliberately returns status `2` whenever a case remains
UNASSESSED. This is a coverage signal, not a broken test run. The current v0.4
artifact has no unassessed main-corpus case, but the rule remains fail closed.
The configurable CRITICAL-recall development gate defaults to `0.01`, so a
zero-recall result is also non-zero even when coverage is otherwise complete.

## Current delivery boundary

See `docs/implementation-status.md` for implemented, structurally supported,
and intentionally unassessed capabilities. Unsupported evaluators never return
a simulated pass.

Methodology, semantic model selection, security boundaries, and limitations are
documented in `docs/methodology.md`, `docs/semantic-evaluator-methodology.md`,
`docs/semantic-model-selection.md`, `docs/security.md`, and
`docs/known-limitations.md`.
