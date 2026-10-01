"""Run a trained local Clef decision model against fresh, read-only live tape."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from clef_live_tape import capture_live_snapshot, live_decision_records, snapshot_freshness
from clef_research_data import MODEL_REVISION, model_input_exclusion_reason
from clef_research_training import encode_rows

EXPANDED_DATASET_REVISION = "58eea08df320b56c0cfcec84f9ae1be1eb8bb5c2"


def load_native_module(path: Path):
    spec = importlib.util.spec_from_file_location("clef_local_inference_native", path / "joint_schema_model.py")
    if spec is None or spec.loader is None:
        raise ValueError("Native Clef decision code is missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def predict_records(model, module, processor, rows, max_length=1024):
    import torch
    for row in rows:
        if model_input_exclusion_reason({"state": row["state"], "questions": row["questions"]}):
            raise ValueError("Input contains credential or signing material")
    encoded, kept, excluded = encode_rows(module, processor, rows, max_length)
    if not kept:
        raise ValueError("No complete live record fits the model context")
    responses = []
    model.eval()
    with torch.inference_mode():
        for record, row in zip(encoded, kept, strict=True):
            logits = model(module.collate_records([record], processor.tokenizer.pad_token_id, "mps"))[0]
            answers = {}
            for question, scores in zip(record.questions, logits, strict=True):
                probabilities = scores.float().softmax(-1)
                if not bool(torch.isfinite(probabilities).all()):
                    raise ValueError("Model returned nonfinite probabilities")
                values = dict(zip(question.option_ids, probabilities.tolist(), strict=True))
                answers[question.question_id] = module.systemone_answer(row["questions"][question.question_id], values)
            responses.append({"id": row["id"], "answers": answers,
                              "usage": {"input_tokens": len(record.input_ids), "output_tokens": 0}})
    return {"responses": responses, "excluded": excluded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Trained standalone release or adapter directory")
    parser.add_argument("--base", type=Path, help="Local immutable NF4 base, required for an adapter")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("local/clef-local-live-inference.json"))
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        parser.error("CPU fallback must be disabled for this local MPS workflow")
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is required")
    torch.set_num_threads(8)
    path = args.model.resolve()
    metadata = json.loads((path / "training.json").read_text())
    if metadata.get("status") not in {"trained_and_reload_verified", "trained_and_standalone_reload_verified"}:
        raise ValueError("Model training and saved-weight reload have not been verified")
    if metadata.get("model_revision") != MODEL_REVISION:
        raise ValueError("Trained model does not use the pinned original Clef revision")
    if metadata.get("dataset_revision") != EXPANDED_DATASET_REVISION:
        raise ValueError("Trained model does not use the verified expanded research commit")
    if metadata.get("security_filter", {}).get("applied") is not True:
        raise ValueError("Training did not record credential/signing-material exclusions")
    snapshot = capture_live_snapshot()
    rows = live_decision_records(snapshot)
    for row in rows:
        if model_input_exclusion_reason({"state": row["state"], "questions": row["questions"]}):
            raise ValueError("Live input contains credential or signing material")
    standalone_model_loaded = False
    release_manifest_sha256 = None
    if (path / "adapter_config.json").is_file():
        if args.base is None:
            parser.error("--base is required when --model points to an adapter")
        from train_clef_local import load_local_adapter, validate_local_base, verify_adapter_manifest
        verify_adapter_manifest(path)
        validate_local_base(args.base.resolve())
        module = load_native_module(path)
        model, processor = load_local_adapter(module, args.base.resolve(), path)
    else:
        from export_clef_mps_release import verify_mps_release, load_local_release
        verify_mps_release(path, expected_dataset_revision=EXPANDED_DATASET_REVISION)
        release_manifest_sha256 = hashlib.sha256((path / "release.json").read_bytes()).hexdigest()
        module = load_native_module(path)
        model, processor = load_local_release(path)
        standalone_model_loaded = True
    # Startup hashing/loading can outlive the first observation. Capture again
    # after weights load; predict_records checks these new inputs before use.
    snapshot = capture_live_snapshot()
    rows = live_decision_records(snapshot)
    result = predict_records(model, module, processor, rows, args.max_length)
    result.update(model_path=str(path), model_revision=metadata["model_revision"],
                  dataset_revision=metadata["dataset_revision"],
                  training_status=metadata["status"],
                  captured_at=snapshot["captured_at"],
                  inference_completed_at=datetime.now(timezone.utc).isoformat(),
                  release_manifest_sha256=release_manifest_sha256,
                  backend="mps", device="mps", standalone_model_loaded=standalone_model_loaded,
                  source="https://clawd-ws.fly.dev/", freshness=snapshot_freshness(snapshot),
                  scope="Observed-field readback from allowlisted live tape; no transaction execution")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
