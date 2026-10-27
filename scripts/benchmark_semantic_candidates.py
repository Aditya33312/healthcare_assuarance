"""Benchmark bounded semantic candidates without touching held-out labels.

Candidate B can run in an isolated optional-ML environment. Candidates A and C
remain explicitly unavailable unless their runtime/credentials are supplied;
the artifact never substitutes a lexical baseline and calls it pretrained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark" / "clinical_faithfulness" / "v0"
DATA_PATH = BENCHMARK_DIR / "semantic-gold.jsonl"
OUTPUT_PATH = BENCHMARK_DIR / "semantic-candidate-benchmark.json"

EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings-500K"
EMBEDDING_REVISION = "71f65d59ddf416f7c2dbf5580ebaa70b6bec7707"
NLI_MODEL = "cnut1648/biolinkbert-mednli"
NLI_REVISION = "f9f4a9d1e046a0e6574feef938f1308cecaebbd5"


def _base_artifact() -> dict[str, Any]:
    data = DATA_PATH.read_bytes()
    return {
        "schema_version": "1.0.0",
        "benchmark_version": "0.1.0",
        "semantic_dataset_sha256": hashlib.sha256(data).hexdigest(),
        "evaluated_splits": ["development", "validation"],
        "held_out_test_touched": False,
        "clinician_calibration_touched": False,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "candidates": {
            "clinical_biomedical_nli": {
                "model": NLI_MODEL,
                "revision": NLI_REVISION,
                "license": "Apache-2.0",
                "status": "NOT_RUN_RUNTIME_UNAVAILABLE",
                "reason": (
                    "The locked Python 3.13 x86_64 macOS environment has no compatible "
                    "PyTorch wheel. The 1.334 GB checkpoint was not treated as evaluated."
                ),
                "intended_role": "SUPPORTED_CONTRADICTED_UNCERTAIN_CLASSIFICATION",
            },
            "biomedical_embedding_alignment": {
                "model": EMBEDDING_MODEL,
                "revision": EMBEDDING_REVISION,
                "license": "Apache-2.0",
                "status": "NOT_RUN_OPTIONAL_DEPENDENCY_UNAVAILABLE",
                "intended_role": "CANDIDATE_EVIDENCE_RETRIEVAL_ONLY",
            },
            "controlled_llm_judge": {
                "model": None,
                "revision": None,
                "prompt_version": "clinical-semantic-judge@0.1.0",
                "temperature": 0,
                "status": "NOT_RUN_DISABLED_NO_APPROVED_PROVIDER_OR_CREDENTIALS",
                "reason": (
                    "No approved external model, credentials, or clinical-data processing "
                    "configuration is present. No benchmark text was transmitted."
                ),
                "intended_role": "UNRESOLVED_CASES_ONLY",
            },
        },
        "selection": {
            "decision": "DETERMINISTIC_TERMINOLOGY_AND_BOUNDED_INFERENCE_CASCADE",
            "fine_tuning": "NOT_AUTHORIZED_INSUFFICIENT_CLINICIAN_REVIEWED_LABELS",
            "limitations": (
                "Only Candidate B was runnable locally. This is a feasibility and retrieval "
                "benchmark, not validation of a pretrained clinical entailment model."
            ),
        },
    }


def _run_embedding(artifact: dict[str, Any]) -> None:
    np: Any = importlib.import_module("numpy")
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    from model2vec import StaticModel  # type: ignore[import-not-found]

    records = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines()]
    records = [record for record in records if record["split"] in {"development", "validation"}]
    model_path = snapshot_download(
        repo_id=EMBEDDING_MODEL,
        revision=EMBEDDING_REVISION,
        allow_patterns=["config.json", "modules.json", "tokenizer.json", "model.safetensors"],
    )
    started = time.perf_counter()
    model = StaticModel.from_pretrained(model_path, force_download=False)
    load_seconds = time.perf_counter() - started
    sources = [str(record["source_evidence"]) for record in records]
    hypotheses = [str(record["generated_claim"]) for record in records]
    started = time.perf_counter()
    source_vectors = model.encode(sources)
    hypothesis_vectors = model.encode(hypotheses)
    elapsed = time.perf_counter() - started
    scores = hypothesis_vectors @ source_vectors.T
    ranks: list[int] = []
    paired_scores: list[float] = []
    for index, row in enumerate(scores):
        ranking = np.argsort(-row, kind="stable")
        rank = int(np.where(ranking == index)[0][0]) + 1
        ranks.append(rank)
        paired_scores.append(float(row[index]))
    candidate = artifact["candidates"]["biomedical_embedding_alignment"]
    candidate.update(
        {
            "status": "RUN_COMPLETE",
            "evaluated_case_count": len(records),
            "retrieval_pool_size": len(records),
            "recall_at_1": round(sum(rank <= 1 for rank in ranks) / len(ranks), 4),
            "recall_at_3": round(sum(rank <= 3 for rank in ranks) / len(ranks), 4),
            "mean_reciprocal_rank": round(sum(1 / rank for rank in ranks) / len(ranks), 4),
            "mean_paired_cosine": round(sum(paired_scores) / len(paired_scores), 4),
            "minimum_paired_cosine": round(min(paired_scores), 4),
            "model_load_seconds": round(load_seconds, 4),
            "mean_encode_milliseconds_per_pair": round(1000 * elapsed / len(records), 4),
            "dependency_versions": {
                name: importlib.metadata.version(name)
                for name in ("model2vec", "numpy", "tokenizers", "safetensors")
            },
            "clinical_entailment_metrics": None,
            "interpretation": (
                "Similarity is evaluated only as evidence retrieval. It is not mapped to "
                "SUPPORTED, INFERRED, or CONTRADICTED."
            ),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-embedding", action="store_true")
    args = parser.parse_args()
    artifact = _base_artifact()
    if args.run_embedding:
        _run_embedding(artifact)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote semantic candidate benchmark to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
