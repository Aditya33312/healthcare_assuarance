# Universal evidence ingestion

Gatekeeper accepts realistic clinical evidence through `POST /v1/evaluations` and
converts each supported transport shape into canonical evidence before clinical
claim extraction. The Clinical Compare Engine remains the authoritative evaluator;
ingestion does not weaken or replace its deterministic medication, unit, numeric,
laterality, route, frequency, assertion, or omission rules.

The evaluation path is:

```text
HTTP request
  -> typed source validation
  -> canonicalization and safe normalization
  -> sectioning and deterministic size-aware chunking
  -> provenance-linked clinical claim extraction
  -> existing two-way Clinical Compare Engine
  -> findings, severity, coverage, and encrypted persistence
```

Raw evidence is retained separately from normalized text. Normalization may make
transport and comparison forms consistent, but it must not silently rewrite the
clinical meaning or truncate evidence.

## Supported source types

| `type` | `content` shape | Intended use |
| --- | --- | --- |
| `clinical_text` | string | Short or long free text, including raw multiline transcripts |
| `clinical_note` | string | Notes with headings, paragraphs, and lists |
| `transcript` | array of segments | Speaker- and timestamp-aware structured dialogue |
| `lab_results` | array of lab results | Test, value, unit, range, flag, and time data |
| `medication_list` | array of medications | Medication, dose, unit, frequency, route, and status data |
| `structured_json` | JSON object | Controlled EHR-like structured context |
| `rag_context` | array of chunks | Retrieved text with chunk and retrieval provenance |

Generated output supports `clinical_text`, `clinical_note`, and
`structured_json`. Existing requests that omit `type`, or use
`type: "clinical_text"` with string content, remain valid.

The source types `fhir`, `document`, `csv`, `file_upload`, and
`imaging_report` are extension points, not accepted formats. Do not relabel one of
those formats as a supported type.

## Request examples

All examples below show only the evidence fields. A complete request also needs a
registered `system_id`, `system_version`, and, when it differs from the default,
`environment`.

### Small clinical text

```json
{
  "system_id": "sys_example",
  "system_version": "1.0.0",
  "environment": "staging",
  "sources": [
    {
      "document_id": "doc-1",
      "type": "clinical_text",
      "content": "Patient reports left knee pain."
    }
  ],
  "generated_output": {
    "type": "clinical_text",
    "content": "Patient reports right knee pain."
  }
}
```

### Structured transcript

```json
{
  "document_id": "transcript-001",
  "type": "transcript",
  "content": [
    {
      "segment": 1,
      "timestamp": "00:03",
      "speaker": "PATIENT",
      "text": "I have had a severe headache since yesterday."
    },
    {
      "segment": 2,
      "timestamp": "00:18",
      "speaker": "CLINICIAN",
      "text": "Have you had neck stiffness or vomiting?"
    }
  ]
}
```

Supported speaker values are `PATIENT`, `CLINICIAN`, `OTHER`, and `UNKNOWN`.
Segment identifiers and optional timestamps remain attached to canonical evidence
so later findings can identify the originating segment.

### Laboratory results

```json
{
  "document_id": "labs-001",
  "type": "lab_results",
  "content": [
    {
      "test": "creatinine",
      "value": 1.8,
      "unit": "mg/dL",
      "reference_range": null,
      "timestamp": null
    },
    {
      "test": "potassium",
      "value": 4.6,
      "unit": "mmol/L"
    }
  ]
}
```

Numeric values are kept in a decimal-safe canonical representation. Clinically
important numbers are compared deterministically rather than by embedding
similarity.

### Medication list

```json
{
  "document_id": "meds-001",
  "type": "medication_list",
  "content": [
    {
      "name": "Metformin",
      "dose": 500,
      "unit": "mg",
      "frequency": "BID",
      "route": "oral",
      "status": "ACTIVE"
    }
  ]
}
```

Canonical medication attributes retain their raw values. For example, `BID`,
`BD`, and `twice daily` normalize to the same frequency, and compatible mass units
such as `0.5 g` and `500 mg` compare as equivalent.

### Structured JSON

```json
{
  "document_id": "ehr-context-001",
  "type": "structured_json",
  "content": {
    "patient": {"age": 45},
    "medications": [{"name": "metformin", "dose": 500, "unit": "mg"}],
    "encounter": {"reason": "headache"}
  }
}
```

Object keys and array positions are traversed deterministically. Canonical units
retain JSON paths, such as `$.medications[0].dose`, when they provide useful
provenance. Executable object deserialization is never performed.

### RAG context

```json
{
  "document_id": "rag-001",
  "type": "rag_context",
  "content": [
    {
      "chunk_id": "chunk-123",
      "text": "The documented insulin glargine dose is 10 units nightly.",
      "source_uri": "ehr://encounter/example",
      "retrieval_score": 0.87,
      "metadata": {"collection": "synthetic-demo"}
    }
  ]
}
```

`source_uri` is provenance, not a URL that Gatekeeper fetches. Retrieval scores
and metadata do not override clinical evidence.

### Multiple sources

Source documents remain distinct. Gatekeeper does not merge their identifiers or
attribute a finding to the wrong document.

```json
{
  "sources": [
    {
      "document_id": "conversation",
      "type": "clinical_text",
      "content": "Clinician: Continue metformin.\nPatient: I take 500 mg twice daily."
    },
    {
      "document_id": "labs",
      "type": "lab_results",
      "content": [{"test": "potassium", "value": 4.6, "unit": "mmol/L"}]
    },
    {
      "document_id": "medications",
      "type": "medication_list",
      "content": [
        {"name": "Metformin", "dose": 500, "unit": "mg", "frequency": "BID"}
      ]
    }
  ],
  "generated_output": {
    "type": "clinical_note",
    "content": "Continue metformin 500 mg twice daily. Potassium is 4.6 mmol/L."
  }
}
```

## Multiline and long text

Programmatic clients should hold multiline content as an ordinary language string
and let their JSON library serialize it. Do not flatten clinical content or
manually edit every line break.

```python
source = """Segment 1
00:03
Patient: I have a headache.

Segment 2
00:22
Clinician: Do you have neck stiffness?
"""

payload = {
    "system_id": "sys_example",
    "system_version": "1.0.0",
    "sources": [{"document_id": "transcript", "type": "clinical_text", "content": source}],
    "generated_output": {"type": "clinical_note", "content": "Headache is documented."},
}
response = httpx.post(url, headers=headers, json=payload)
```

Swagger's JSON editor displays a serialized JSON document, so literal line breaks
inside a quoted JSON token are not valid JSON. The Python helper below and normal
HTTP SDKs perform the serialization automatically. Swagger remains useful for
small examples; use an SDK or the helper for large documents.

Large documents are sectioned and chunked at stable boundaries. Chunking preserves
source offsets and avoids splitting obvious medication/value/unit tuples or
negation context where possible. Callers do not need to pre-chunk evidence.

## Size and complexity limits

Limits are deployment configuration, not medical standards. Gatekeeper enforces
the HTTP body limit as well as source count, per-source size, generated-output
size, transcript segment count, structured JSON depth and array size, canonical
unit count, extracted claim count, chunk count, and processing-time safeguards.

The active deployment configuration and OpenAPI schema are authoritative. Inputs
within supported small, medium, and large classes are processed synchronously
using the appropriate section/chunk path. Very-large or excessive input is
rejected with HTTP `413`; evidence is never silently truncated. File upload,
asynchronous very-large ingestion, PDF, OCR, and Word parsing are not implemented.

Validation and ingestion failures use the standard error envelope:

```json
{
  "error": {
    "code": "REQ_INVALID",
    "message": "The request did not match the required schema.",
    "retryable": false,
    "request_id": "req_example"
  }
}
```

Messages do not echo source content. Use the HTTP status, stable error code, and
`request_id` for support and observability; do not copy clinical evidence into
logs or error metadata.

## Provenance and coverage

Canonical evidence records retain a content hash, document ID, source type,
source path or segment identifier where applicable, timestamp, speaker, section,
and exact offsets. Findings continue to expose exact source/generated spans. Raw
clinical content and structured payloads use the existing encrypted persistence
path; operational logs contain counts and timings, not evidence text.

Ingestion fails closed. Content that cannot be safely interpreted is rejected or
reported through `REDUCED`, `PARTIAL`, or `UNASSESSED` coverage with a stable
limitation reason. It is never converted into a `SUPPORTED` finding merely because
parsing or normalization failed.

## File-submit helper

The supported helper reads the API key only from the environment. Never put a
Gatekeeper key in a command-line option or URL.

```bash
export GATEKEEPER_API_URL=http://127.0.0.1:8000
export GATEKEEPER_API_KEY='<one-time or managed Gatekeeper API key>'

python scripts/evaluate.py \
  --source transcript.txt \
  --output note.txt \
  --output-type clinical_note \
  --system-id sys_xxx \
  --system-version 1.0.0
```

Repeat `--source` to submit multiple documents. A text file defaults to
`clinical_text`; this is the correct choice for a raw multiline transcript. Use a
JSON file for structured source types:

```json
{
  "document_id": "labs-001",
  "type": "lab_results",
  "content": [{"test": "potassium", "value": 4.6, "unit": "mmol/L"}]
}
```

A source JSON file may contain one complete source object, a list of complete
source objects, or an envelope with a `sources` list. Otherwise the helper wraps
the JSON object as `structured_json`; `--source-type` can select another structured
type for an array with that type's required shape. Generated JSON may contain a
complete `{ "type", "content" }` object, an envelope with `generated_output`, or a
raw JSON object.

The helper preserves text-file newline characters, uses the JSON serializer for
transport, refuses insecure non-local API origins, does not follow redirects with
the authorization header, never prints the key, and prints counts rather than
clinical content. It exits nonzero for transport failure, rejected input, failed
execution, or `UNASSESSED` coverage.

## Lightweight performance benchmark

Run the synthetic, no-PHI benchmark separately from unit tests:

```bash
python -m scripts.benchmark_ingestion \
  --output benchmark/ingestion-performance.json
```

It selects small, medium, and large payloads from the active configured size
thresholds and reports observed payload size, ingestion and normalization time,
claim-extractor time, total comparison/evaluation time, claim and chunk counts,
coverage, and Python allocation peak. The reported alignment/comparison residual
is explicitly an estimate: comparison wall time minus measured extractor-call
time. It also includes rules, semantic fallback, finding construction, profiling,
and timer overhead, so it must not be represented as isolated alignment latency.
The benchmark makes no production performance claim.
