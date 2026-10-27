# Security and privacy boundary

## Data handling

Persisted evaluation source text, generated output, extracted claims, alignment
explanations, finding evidence/original finding payloads, human-review comments,
vendor responses, and captured run-case output have application-encrypted
columns. Encryption uses authenticated Fernet tokens with a MultiFernet rotation
path. Tenant- and context-keyed digests are used where equality/integrity lookup
is needed without storing raw clinical values. Production keys must come from a
managed secret store; they must not be committed or reused between environments.

Universal ingestion applies the same boundary to raw and canonicalized supported
evidence, including structured transcripts, laboratory results, medication lists,
structured JSON, and RAG chunks. Stable tenant-scoped content digests support
audit and reproducibility without placing evidence in operational metadata.
Canonical provenance may identify a document, transcript segment, RAG chunk, or
JSON path, but logs and error messages must not contain its clinical text.

Encryption is not blanket coverage for every table. In particular, Phase-1
`TestCaseVersion.input_data` and `expectation`, suite definitions, application
profiles, coverage, and operational metadata are plaintext JSON. Suite creation
therefore requires a `SYNTHETIC` or `DE_IDENTIFIED` classification together with
`contains_phi: false` and `no_phi_attested: true`. This is a caller attestation,
not automated de-identification; PHI must not be submitted through that path.

API keys contain at least 256 random bits, are returned only at creation, and are
stored as HMAC-SHA-256 digests using a separate server-side pepper. Authentication
enforces organization, scope, expiry, and revocation. Key prefixes are identifiers,
not authenticators. Suite authoring, plan confirmation, and policy authoring use
separate management scopes; a least-privilege CI run key cannot mutate those
governance artifacts.

Tenant-owned rows carry `organization_id`. Repository methods require tenant
scope and tests exercise cross-tenant non-disclosure. Global version catalogs are
the intentional exception. The application also prevents TEST API keys from
creating or reading production-environment systems, evaluations, runs, findings,
and coverage; a LIVE key is required. These application checks do not yet add
database-level row security or enforce every child's `organization_id` against
its parent's organization with a composite constraint.

## Operational controls

- Ordinary logs, errors, metrics, and audit metadata reject clinical payloads.
- Request IDs are returned for correlation but are not metric labels.
- Request-body limits and a replaceable rate limiter protect the API boundary.
- Ingestion additionally bounds source count and size, generated-output size,
  structured depth and array cardinality, transcript segments, canonical units,
  chunks, claims, and processing time. Excessive content is rejected; it is never
  silently truncated or treated as supported.
- Structured JSON is data only. Gatekeeper does not perform arbitrary object
  deserialization, fetch RAG source URIs, or send evidence to an external model
  merely because it was ingested.
- Authenticated `/v1` responses use `Cache-Control: no-store`.
- The production portal accepts upstream data reads and proxy writes only after
  a constant-time exact match against a high-entropy header value injected by a
  trusted gateway. The gateway must strip that header from untrusted traffic
  before injecting its private value. Partial configuration fails closed.
- Portal mutation proxies remain disabled by default. When enabled, they accept
  only `application/json` from the exact HTTPS origin configured in
  `GATEKEEPER_PORTAL_ALLOWED_ORIGIN` and reject cross-site fetch metadata. The
  identity gateway must also use appropriate SameSite cookies and retain its
  own CSRF protections.
- The production portal's Gatekeeper upstream must use HTTPS and cannot contain
  URL credentials, a query, or a fragment. HTTP is limited to localhost outside
  production. API and gateway secrets remain server-only runtime values.
- Customer endpoint errors are sanitized; retries are bounded and limited to
  timeouts, 429, 502, and 503. Authentication and schema failures stop execution.
- Registered customer endpoints must use HTTPS and reject user information,
  loopback/local hostnames, and non-global literal IP addresses. Hostname DNS can
  still change after validation, so egress allowlisting and network policy are
  required to contain DNS-rebinding and destination-pivot risks.
- Idempotency records keep a canonical request hash and resource reference, not
  a copy of the clinical request body.
- GitHub release-gate output does not print API secrets or customer payloads.

## Deployment checklist

Before handling any regulated data:

1. replace all example secrets and terminate TLS at a trusted ingress;
2. store encryption keys, API-key pepper, database credentials, and customer
   endpoint credentials in an approved secret manager;
3. use PostgreSQL with encrypted storage, backups, point-in-time recovery, and a
   least-privilege application role; production configuration requires an explicit
   `sslmode` of `require`, `verify-ca`, or `verify-full`—prefer `verify-full` and
   configure the approved CA/root certificate for hostname verification;
4. restrict egress to registered staging endpoints and approved model stores;
5. configure and test retention deletion jobs, regional residency, incident
   response, access review, and key rotation to the deployment's policy;
6. send sanitized audit and operational events to the approved monitoring stack;
7. put the portal and operator APIs behind production identity, SSO, and
   least-privilege authorization; have the gateway strip and inject the portal
   trust header, rotate its random value, and verify missing/incorrect headers
   cannot reach upstream reads or writes; configure the exact portal origin and
   verify cross-origin and non-JSON mutation requests are rejected;
8. pin Python/Node dependencies, GitHub Actions, and service/container images to
   reviewed immutable versions or digests, produce an SBOM, and scan artifacts;
9. conduct application, dependency, infrastructure, and privacy reviews; and
10. obtain the required contractual and compliance approvals.

## Known security limitations

The repository supplies application-layer controls, not a compliance
certification. The default queue is an in-process/protocol boundary, the default
rate limiter is process-local, and production identity/SSO, KMS/HSM integration,
network policy, WAF, SIEM, backup recovery, retention deletion jobs, database row
security/composite tenant constraints, and infrastructure are deployment
responsibilities. The dependency ranges, GitHub Actions, and development service
images are not all locked by immutable digest. Portal writes are disabled by
default, and the portal has no built-in end-user identity layer. Its trusted
gateway header proves that a request traversed a correctly configured gateway;
the gateway still owns user authentication, authorization, session-cookie
hardening, and defense-in-depth CSRF enforcement. The example configuration
must not be used with production PHI.
