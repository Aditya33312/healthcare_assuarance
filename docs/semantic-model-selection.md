# Semantic model selection

## Decision

The v1 candidate uses deterministic terminology, semantic-context safety checks,
and bounded multi-claim inference. Biomedical embeddings may retrieve candidate
evidence. Pretrained NLI and a controlled LLM judge remain optional and fail
closed because neither was deployable in the verified runtime. Fine-tuning is
not authorized because there are no clinician-reviewed semantic labels.

| Candidate | License / cost | Deployment and latency | Precision/recall and relationship performance | Contradiction / reproducibility | Decision |
| --- | --- | --- | --- | --- | --- |
| `cnut1648/biolinkbert-mednli` revision `f9f4a9d...` | Apache-2.0; local compute | Transformers/PyTorch and a 1.334 GB checkpoint; the Python 3.13 x86_64 macOS host had no compatible PyTorch wheel, so latency was not measured | Not run; no precision, recall, SUPPORTED, INFERRED, or UNCERTAIN claim | Clinical contradiction error rate unmeasured; revision pinned | Retain optional; validate in a supported pinned container before use |
| `NeuML/pubmedbert-base-embeddings-500K` revision `71f65d5...` | Apache-2.0; local compute | `model2vec` 0.9.0, 64-dimensional static embeddings; 0.2570 ms mean encoding per pair | On 32 development+validation cases: recall@1 0.5938, recall@3 0.8438, MRR 0.7396; relationship precision/recall intentionally not computed | Cannot detect clinical contradiction; model revision and dependencies recorded | Candidate evidence retrieval only; never map cosine to a relationship label |
| Controlled JSON LLM judge | Provider-dependent; provider cost unknown | No approved provider, credentials, or clinical data-processing configuration; latency not measured | Not run; no precision, recall, or relationship claim; no benchmark text transmitted | Nondeterminism and contradiction errors unmeasured; prompt v0.1, temperature 0 boundary recorded | Disabled by default |
| Selected offline cascade | Repository code; no per-call service cost | No ML runtime or external service; deterministic local rules | Main normalized P/R/F1 0.8375/0.9950/0.9095. Semantic subset: SUPPORTED F1 0.9091, INFERRED 1.0000, UNCERTAIN 0.7879 | Semantic-subset CONTRADICTED P/R/F1 0.9286/0.8125/0.8667; byte-verifiable inputs and versioned rules | Frozen v1 engineering candidate |

The model cards report Apache-2.0 for the [BioLinkBERT MedNLI checkpoint](https://huggingface.co/cnut1648/biolinkbert-mednli)
and the [PubMedBERT static embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings-500K).
The embedding card describes sentence-similarity use; that does not establish
clinical entailment. A future external judge must use strict schema output; the
[OpenAI Responses API reference](https://platform.openai.com/docs/api-reference/responses)
documents strict JSON-schema response formatting, but no provider is selected by
this repository.

## Fine-tuning gate

Fine-tuning can begin only when two independent clinicians and an adjudicator
produce sufficient labels, the pretrained/rule hybrid has a measured gap, the
expected gain is specified, and an untouched external held-out set remains.
None of those prerequisites is complete, so fine-tuning would currently create
unverifiable overfitting rather than validated clinical performance.
