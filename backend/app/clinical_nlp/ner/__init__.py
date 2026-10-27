"""Optional pretrained clinical NER integration."""

from backend.app.clinical_nlp.ner.huggingface import (
    HuggingFaceClinicalNERAdapter,
    NEREntity,
)

__all__ = ["HuggingFaceClinicalNERAdapter", "NEREntity"]
