# Known limitations

- Benchmark v0.4 has 200 unique engineering-authored synthetic pairs, but it is
  not a representative external or clinician-validated set. Its 110/30/30/30
  partitions were frozen only after all cases had already been visible.
- The current main result has 145 FULL, 26 REDUCED, 29 PARTIAL, and zero
  UNASSESSED cases. Reduced and partial coverage is retained for genuine
  ambiguity, uncertain inference paths, and missing extracted evidence.
- Severity policy is provisional. Normalized CRITICAL detection and severity
  recall are `1.0000` on development data, but this is not clinician-calibrated.
- The broader benchmark has normalized F1 `0.9095`, but `SUPPORTED` precision is
  only `0.5970`, and one normalized inference false negative remains. The
  separate semantic subset has only 64 engineering labels; its held-out slice is
  16 cases and is not statistically or clinically sufficient validation.
- No actual clinician calibration or clinical safety approval has been
  completed. The supplied calibration service and in-memory repository are a
  structural workflow for tests and pilots, not durable validated operations.
- Only clinical faithfulness has an executable evaluator. The other 15 families
  are routed but explicitly UNASSESSED.
- The optional biomedical NER model was trained on case reports and has not been
  accepted for transcripts, notes, letters, multilingual text, or patient-facing
  content. The default engine remains offline and deterministic.
- `GATEKEEPER_NER_BACKEND=huggingface` is wired to the pinned, cache-only optional
  adapter. A load or inference failure downgrades coverage while deterministic
  extraction continues. `GATEKEEPER_SEMANTIC_EVALUATION_ENABLED=true` is rejected
  because no external NLI/judge runtime is configured through environment
  settings. The built-in offline terminology/context/inference cascade is enabled
  by default; pretrained NLI, embeddings, and a controlled judge remain
  programmatic extension points.
- Deterministic extraction is deliberately bounded. Novel terminology,
  complicated coreference, cross-document temporal reasoning, and implied
  clinical facts can reduce coverage or create uncertain findings.
- Universal ingestion supports synchronous JSON requests for clinical text,
  clinical notes, structured transcripts, laboratory results, medication lists,
  controlled structured JSON, and RAG context. It does not yet provide multipart
  upload, asynchronous very-large ingestion, FHIR resource validation, CSV,
  PDF/OCR, Word, or imaging parsing. Inputs beyond configured limits are rejected
  rather than truncated.
- Canonicalization makes supported transport shapes available to the existing
  clinical extractor; it does not imply that every specialty vocabulary,
  transcript style, JSON schema, or cross-document relationship has been
  clinically validated. Unresolved interpretation remains reduced, partial, or
  unassessed rather than being promoted to support.
- Test-suite case inputs and expectations are plaintext JSON. The API requires a
  synthetic/de-identified no-PHI attestation, but it does not perform automated
  de-identification or detect a false attestation.
- The portal is an operational baseline; user/role administration, clinician
  review UX, SSO, bulk corpus tooling, historical system/suite/policy version
  browsing, full result tabs/report export, credential authoring, and endpoint
  connection testing are not complete. Registration records an authentication
  type but does not bind a credential or test connectivity.
  It has no built-in end-user identity boundary, and plan confirmation is
  attributed to the shared portal API key rather than a human identity.
  Production requests require a
  private trusted-gateway header, but the gateway remains responsible for user
  authentication, authorization, header stripping/injection, secure session
  cookies, and defense-in-depth CSRF enforcement. Writes are disabled by default;
  when enabled they additionally require same-origin JSON.
- Background execution includes a database-backed worker that can be run with
  `python scripts/gatekeeper_worker.py`; it claims queued runs, calls registered
  endpoints, persists results, and applies policies. A production deployment
  still needs process supervision, worker scaling/locking validation, and its
  preferred Redis/Celery/RQ/ARQ queue if database polling is not sufficient.
- TEST keys are denied production-environment systems, evaluations, runs,
  findings, and coverage at the application layer. Separate production
  infrastructure, credentials, and database-enforced row isolation are not
  supplied. Although tenant-owned rows are query-scoped, the schema does not
  enforce every child row's organization against its parent with composite
  foreign keys or row-level security.
- Customer endpoint validation rejects obvious unsafe URL forms and literal IPs,
  but application validation alone cannot fully prevent DNS rebinding. A
  production deployment needs restrictive DNS and egress network policy.
- Python 3.12.4 is the minimum supported 3.12 runtime because earlier patch
  releases have stale special-purpose address classifications at the endpoint
  SSRF boundary.
- Production KMS/HSM integration, SSO, retention deletion jobs, WAF/SIEM, and
  backup-recovery controls are not included.
- Python dependency ranges, GitHub Actions, and development service/container
  images are not all locked to reviewed immutable digests.
- This code does not itself provide HIPAA, GDPR, FDA, MDR, or other legal or
  regulatory compliance and must not be used as the sole release authority.
