# Clinical NER baseline

## Candidate model

- Repository: `d4data/biomedical-ner-all`
- Pinned revision: `015a4050c9ac99722e61c547aa9b4282bcbedc7f`
- Architecture: DistilBERT token classification
- Training corpus reported by the publisher: MACCROBAT case reports
- License reported by the publisher: Apache-2.0

The model is loaded lazily only when the Hugging Face backend is enabled. The
default evaluator remains deterministic and offline so optional inference
availability cannot remove high-risk checks.

## Acceptance status

**Not accepted for production clinical assurance yet.** Integration does not
equal validation. Promotion requires held-out per-entity precision, recall, and
F1, with explicit medication, dose, laterality, negation, and critical-value
recall. Until then, model entities are candidates and deterministic parsers own
high-risk numeric and assertion decisions.

## Known limitations

- The publisher describes training on case reports, which may not transfer to
  transcripts, notes, letters, or patient-facing text.
- Token classification does not by itself resolve assertion scope, subject,
  relations, normalization, or claim equivalence.
- The configured model is English-language and must not be assumed valid for
  multilingual inputs.
- Model download, runtime, and output schema failures must reduce coverage rather
  than produce a pass.

Primary model documentation:
https://huggingface.co/d4data/biomedical-ner-all

