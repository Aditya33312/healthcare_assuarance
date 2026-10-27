"""Version identifiers for the auditable clinical NLP baseline."""

from __future__ import annotations

CLINICAL_NLP_VERSION = "0.2.0"
PREPROCESSING_VERSION = "offset-safe-preprocessing@0.1.0"
DETERMINISTIC_EXTRACTOR_VERSION = "deterministic-clinical-extractor@0.2.0"
ASSERTION_RULES_VERSION = "assertion-rules@0.1.0"
NORMALIZATION_VERSION = "clinical-normalizers@0.1.0"
RELATION_RULES_VERSION = "clinical-relation-rules@0.2.0"

# This exact revision is deliberately recorded instead of relying on a moving model tag.
HF_NER_MODEL_NAME = "d4data/biomedical-ner-all"
HF_NER_MODEL_REVISION = "015a4050c9ac99722e61c547aa9b4282bcbedc7f"
HF_NER_MODEL_VERSION = f"{HF_NER_MODEL_NAME}@{HF_NER_MODEL_REVISION}"


def component_versions(*, hf_enabled: bool = False) -> dict[str, str | None]:
    """Return serializable component versions for evaluation provenance."""

    return {
        "clinical_nlp": CLINICAL_NLP_VERSION,
        "preprocessing": PREPROCESSING_VERSION,
        "ner_model": HF_NER_MODEL_VERSION if hf_enabled else DETERMINISTIC_EXTRACTOR_VERSION,
        "assertion_model": ASSERTION_RULES_VERSION,
        "normalization": NORMALIZATION_VERSION,
        "relation_rules": RELATION_RULES_VERSION,
    }
