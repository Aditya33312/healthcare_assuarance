# Test-run worker

The API creates a confirmed test run and persists it as `QUEUED`. A separate
worker process executes the run so endpoint calls do not block API requests.

## Local execution

Start the API and database first, then run the worker in another terminal:

```bash
source .venv/bin/activate
python scripts/gatekeeper_worker.py
```

For one queued run:

```bash
python scripts/gatekeeper_worker.py --once
```

The worker uses the same `GATEKEEPER_DATABASE_URL` and
`GATEKEEPER_DATA_ENCRYPTION_KEYS` settings as the API. It:

1. atomically claims a queued run;
2. loads the registered application version and endpoint;
3. selects the generic REST, custom JSON, or OpenAI-compatible adapter;
4. sends each immutable suite case with bounded concurrency and retries;
5. evaluates the returned output through the family evaluator;
6. stores encrypted output, case metrics, coverage, and technical errors;
7. applies the confirmed release policy and stores `PASS`, `WARN`, or `BLOCK`.

The CI job polls the run endpoint while this worker is running. If the worker is
not running, the API correctly leaves the run as `QUEUED` and CI eventually
fails closed on timeout.

## Deployment

Run the API and worker as separate supervised services using the same database,
encryption keys, and endpoint egress policy. The worker must be able to reach
the registered staging endpoints over HTTPS. Do not put application credentials
in endpoint URLs; use the encrypted version authentication configuration.
