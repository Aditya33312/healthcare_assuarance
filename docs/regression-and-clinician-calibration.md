# Regression promotion and clinician calibration

## Permanent regression promotion

`RegressionPromotionService` converts one persisted clinical finding into a
versioned `CLINICAL_FAITHFULNESS` test case. Promotion is allowed only when all
of these conditions hold:

- the finding, evaluation, validation review, reviewer, and promoting user are
  in the authenticated organization;
- the selected `HumanReview.disposition` is exactly `VALIDATED`;
- the caller attests that the complete artifact contains no PHI and classifies
  it as `SYNTHETIC` or `DE_IDENTIFIED`;
- tenant- and context-keyed source/output integrity digests and exact evidence
  offsets validate;
- the same tenant finding has not previously been promoted.

The service replaces upstream document identifiers with synthetic identifiers,
applies a validated severity override without permitting a lower risk tier,
publishes immutable case version `1.0.0`, and publishes a new patch version of
the target suite. The suite definition retains the source evaluation, finding,
and human-review references as provenance. An append-only audit event contains
only identifiers and categorical metadata, never clinical text. The caller owns
the surrounding database transaction and must commit or roll it back.

Permanent test inputs are plaintext in the existing Phase-1 test-case tables.
The no-PHI/de-identification attestation is therefore a hard eligibility
boundary, not a substitute for an automated de-identification service.

## Clinician calibration workflow

`benchmark/clinical_faithfulness/v0/clinician-calibration-template.json` exports
the 30 immutable calibration-partition cases without automated predictions or
engineering-gold labels. It requires two distinct independent reviewers and a
third adjudicator, and specifies classification, severity, rationale, evidence,
percent agreement, kappa, and Gatekeeper-versus-adjudicated metrics. It is an
uncompleted review package, not evidence that calibration occurred.

`ClinicianCalibrationService` models an immutable-version structural/pilot
workflow:

`DRAFT -> IN_REVIEW -> READY_FOR_ADJUDICATION -> ADJUDICATED -> CALIBRATED`

Reviewer assignments contain randomized item tokens and secure payload
references. They omit finding identifiers, model-proposed labels/severity, peer
labels, and adjudications. A label records that the review was blinded and that
reviewer qualification was self-attested. Each item needs at least two distinct
reviewers. Its adjudicator must be independently attested and cannot be one of
that item's reviewers.

Agreement output includes pair counts, label/severity/exact percent agreement,
and chance-corrected pairwise kappa. Status transitions, blinded labels,
adjudications, metrics, and the external completion attestation remain attached
to the semantic calibration version.

`CALIBRATED` does **not** mean Gatekeeper supplied clinical validation or
verified clinical credentials. It means the application recorded an external
attestation claiming that a qualified clinician panel completed the referenced
protocol. Credential verification, reviewer recruitment, secure clinical
payload delivery, and the real clinician review are deployment responsibilities.

The included `InMemoryCalibrationRepository` is a tenant-isolated reference
adapter for tests and pilots; it is not durable across process restarts. A
production deployment must implement the repository protocol with encrypted,
durable storage before exposing the workflow through an API.
