#!/usr/bin/env python3
"""Fine-tune the genuine Clef decision head and text-backbone LoRA on research QA."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from clef_research_data import DATASET_ID, MODEL_ID, MODEL_REVISION, prepare_dataset, read_jsonl, training_source_hash
from clef_research_training import (attach_lora, compare_predictions, encode_rows, evaluate, load_upstream,
                                   reload_adapter, save_adapter, supervised_loss, trainable_fingerprints)


def read_rows(path, limit=0):
    return read_jsonl(path, limit)


def select_cohort(rows, limit, seed, include_live=False):
    """Sample a bounded pilot across the expanded split, reserving observed rows."""
    if not limit or len(rows) <= limit:
        return rows
    required = [index for index, row in enumerate(rows) if include_live and
                row["provenance"].get("source_type") == "live_tape_observation"]
    if include_live:
        for paper in ("2605.12151", "2606.08232"):
            found = next((index for index, row in enumerate(rows) if paper in str(row["provenance"].get("source"))), None)
            if found is not None and found not in required:
                required.append(found)
    if len(required) > limit:
        raise ValueError("Pilot limit cannot contain all required live/paper observations")
    remaining = [index for index in range(len(rows)) if index not in required]
    selected = required + random.Random(seed).sample(remaining, limit - len(required))
    return [rows[index] for index in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("local/clef-research-data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/clef-research"))
    parser.add_argument("--output-repo", help="Optional private Hub destination; requires HF_TOKEN")
    parser.add_argument("--dataset-revision", required=True, help="Immutable commit of the verified expanded dataset")
    parser.add_argument("--live-tape", action="store_true", help="Capture real read-only clawd-ws observations for native decision training")
    parser.add_argument("--export-merged", action="store_true", help="Export a standalone merged backbone plus trained joint head after verification")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="Optimizer steps; 0 trains full epochs")
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-eval-records", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--checkpoint-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.max_length, args.gradient_accumulation, args.rank, args.checkpoint_steps) < 1 or args.lr <= 0:
        parser.error("epochs, length, accumulation, rank, checkpoint interval, and LR must be positive")
    if min(args.max_steps, args.max_train_records, args.max_eval_records) < 0:
        parser.error("record and step limits must be nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; preserve earlier training artifacts")

    import torch
    from huggingface_hub import HfApi, snapshot_download
    if not torch.cuda.is_available():
        raise RuntimeError("Clef 27B training requires a remote CUDA GPU; no local weight download was attempted")
    if args.output_repo and not os.environ.get("HF_TOKEN"):
        parser.error("--output-repo requires HF_TOKEN")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(device).total_memory < 100 * 1024 ** 3:
        raise RuntimeError("This BF16 training configuration requires at least 100 GiB GPU memory; use H200")
    api = HfApi()
    if args.output_repo:
        api.create_repo(args.output_repo, private=True, exist_ok=True)
        if not api.model_info(args.output_repo).private:
            raise ValueError("Use a private output repository for the initial training artifact")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Rebuild from the pinned source on every run, never from a stale local manifest.
    live_decisions = None
    if args.live_tape:
        from clef_live_tape import capture_live_snapshot, live_decision_records
        snapshot = capture_live_snapshot(timeout=15, max_frames=4)
        (args.output / "live-training-snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        live_decisions = live_decision_records(snapshot)
        if not live_decisions:
            raise ValueError("No real live observations available for requested tape training")
    manifest = prepare_dataset(args.data_dir, args.seed, dataset_revision=args.dataset_revision, live_decisions=live_decisions)
    module = load_upstream()
    base = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    model, processor = module.load_release_model(base, device=device, dtype=torch.bfloat16, attn_implementation="sdpa")
    if processor.tokenizer.pad_token_id is None:
        raise ValueError("Clef tokenizer must provide a padding token")
    targets = attach_lora(model, args.rank)
    initial_fingerprints = trainable_fingerprints(model)
    cohorts = {}
    for split in ("train", "eval", "test"):
        limit = args.max_train_records if split == "train" else args.max_eval_records
        rows = select_cohort(read_rows(args.data_dir / f"{split}.jsonl"), limit, args.seed,
                             include_live=split == "train" and args.live_tape)
        encoded, kept, excluded = encode_rows(module, processor, rows, args.max_length)
        if not kept:
            raise ValueError(f"No {split} records fit the context limit")
        cohorts[split] = (encoded, kept)
        (args.output / f"{split}-excluded.json").write_text(json.dumps(excluded, indent=2) + "\n")
        manifest["outputs"][split].update(selected=len(rows), encoded=len(kept), context_excluded=len(excluded))
        manifest["outputs"][split]["live_selected"] = sum(row["provenance"].get("source_type") == "live_tape_observation" for row in rows)
        manifest["outputs"][split]["live_encoded"] = sum(row["provenance"].get("source_type") == "live_tape_observation" for row in kept)
    if args.live_tape and not manifest["outputs"]["train"]["live_encoded"]:
        raise ValueError("The selected training cohort contains no encodable live observations")
    (args.output / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    baseline, _ = evaluate(model, module, processor, *cohorts["eval"], device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    train_encoded, train_rows = cohorts["train"]
    total_steps = args.epochs * math.ceil(len(train_rows) / args.gradient_accumulation)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    metadata = {
        "model": MODEL_ID, "model_revision": MODEL_REVISION, "dataset": DATASET_ID,
        "dataset_revision": manifest["dataset_revision"], "task": manifest["task"],
        "seed": args.seed, "max_length": args.max_length, "epochs": args.epochs,
        "planned_steps": total_steps, "gradient_accumulation": args.gradient_accumulation,
        "lora_rank": args.rank, "target_modules": targets, "learning_rate": args.lr,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "baseline_eval": baseline, "losses": [], "optimizer_steps": 0, "training_records_seen": 0,
        "status": "training", "versions": {}, "autoExecute": False,
        "training_source_sha256": training_source_hash(),
    }
    for package in ("torch", "transformers", "peft", "huggingface_hub", "pyarrow"):
        metadata["versions"][package] = __import__(package).__version__
    rng = random.Random(args.seed)
    adapter_gradient_seen = head_gradient_seen = False
    training_started = time.monotonic()
    training_finished = None

    def checkpoint():
        metadata["seconds"] = time.monotonic() - started
        metadata["training_seconds"] = (training_finished or time.monotonic()) - training_started
        save_adapter(model, processor, module, args.output, metadata)
        torch.save({"optimizer": optimizer.state_dict(), "optimizer_steps": metadata["optimizer_steps"]},
                   args.output / "optimizer.pt")
        if args.output_repo:
            api.upload_folder(repo_id=args.output_repo, folder_path=str(args.output),
                              commit_message=f"Clef research training step {metadata['optimizer_steps']} ({metadata['status']})")

    for epoch in range(args.epochs):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), args.gradient_accumulation):
            indices = order[start:start + args.gradient_accumulation]
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for index in indices:
                encoded, row = train_encoded[index], train_rows[index]
                logits = model(module.collate_records([encoded], processor.tokenizer.pad_token_id, device))
                loss = supervised_loss(logits, [encoded], [row])
                if not torch.isfinite(loss):
                    raise ValueError("Training loss is not finite")
                (loss / len(indices)).backward()
                step_loss += float(loss.detach()) / len(indices)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and float(parameter.grad.detach().float().abs().sum()) > 0:
                    adapter_gradient_seen |= "lora_" in name
                    head_gradient_seen |= name.startswith("head.")
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            metadata["optimizer_steps"] += 1
            metadata["training_records_seen"] += len(indices)
            metadata["losses"].append(step_loss)
            print(json.dumps({"epoch": epoch + 1, "step": metadata["optimizer_steps"], "loss": step_loss}), flush=True)
            if metadata["optimizer_steps"] % args.checkpoint_steps == 0:
                checkpoint()
            if metadata["optimizer_steps"] >= total_steps:
                break
        if metadata["optimizer_steps"] >= total_steps:
            break
    if not adapter_gradient_seen or not head_gradient_seen:
        raise ValueError("Training did not update both LoRA and decision-head paths")
    metadata["gradient_evidence"] = {"lora_nonzero": adapter_gradient_seen, "head_nonzero": head_gradient_seen}
    trained_fingerprints = trainable_fingerprints(model)
    changed = {kind: initial_fingerprints[kind]["sha256"] != trained_fingerprints[kind]["sha256"] for kind in initial_fingerprints}
    if not all(changed.values()):
        raise ValueError("Optimizer steps did not change both LoRA and native head parameter bytes")
    metadata["weight_update_evidence"] = {"changed": changed, "before": initial_fingerprints, "after": trained_fingerprints}
    training_finished = time.monotonic()
    metadata["training_seconds"] = training_finished - training_started
    metadata["seconds_per_optimizer_step"] = metadata["training_seconds"] / metadata["optimizer_steps"]
    metadata["all_planned_steps_completed"] = metadata["optimizer_steps"] == metadata["planned_steps"]
    metrics = {"baseline_eval": baseline}
    for split in ("eval", "test"):
        metrics[split], predictions = evaluate(model, module, processor, *cohorts[split], device)
        (args.output / f"{split}-predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
    metadata["status"] = "trained_pending_reload_verification"
    checkpoint()
    # Release original weights and optimizer before reloading the saved artifact.
    del parameters, optimizer, model, logits, loss, parameter
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, reloaded_processor = reload_adapter(module, args.output, device)
    verification_count = min(8, len(cohorts["test"][1]))
    encoded, rows = (values[:verification_count] for values in cohorts["test"])
    reload_metrics, reload_predictions = evaluate(reloaded, module, reloaded_processor, encoded, rows, device)
    saved_predictions = read_jsonl(args.output / "test-predictions.jsonl", verification_count)
    metrics["reload_verification"] = {**compare_predictions(saved_predictions, reload_predictions), "metrics": reload_metrics}
    metrics["promotion"] = "review_required; domain metrics do not establish an official Decision Index result"
    (args.output / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    metadata["status"] = "trained_and_reload_verified"
    metadata["seconds"] = time.monotonic() - started
    (args.output / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    citations = (Path(__file__).resolve().parents[1] / "data/realtime_research_citations.md").read_text()
    card = f"""---
license: apache-2.0
base_model: {MODEL_ID}
library_name: peft
tags:
  - solana
  - clef
  - decision-model
datasets:
  - {DATASET_ID}
---

# Clawd Clef Research Decision Adapter

LoRA text-backbone adapter plus the trained native Clef joint schema head.
Training tasks: select a reference research answer among four candidates;
when enabled, read observed health status and creator-declared social fields.
Base revision: `{MODEL_REVISION}`. Load with `reload_adapter` in the included
`clef_research_training.py`, then call the upstream `systemone` API.
The vision encoder is frozen; no image-training improvement is claimed.
See `training.json`, `data-manifest.json`, and `evaluation.json` for cohort,
coverage, metrics, limitations, and reload evidence. This is not an official
Decision Index score or a live trading policy. Signing keys are never inputs.

{citations}
"""
    (args.output / "README.md").write_text(card)
    import shutil
    for name in ("clef_research_training.py", "clef_research_data.py", "clef_live_tape.py"):
        shutil.copyfile(Path(__file__).with_name(name), args.output / name)
    if args.output_repo:
        api.upload_folder(repo_id=args.output_repo, folder_path=str(args.output), commit_message="Save trained Clef adapter, decision head, evaluation and verified reload")
    if args.export_merged:
        from export_clef_release import export_merged_release, load_release_model, refresh_release_manifest
        merged_dir = args.output.with_name(args.output.name + "-merged")
        release = export_merged_release(reloaded, reloaded_processor, module, merged_dir, metadata)
        # Merging BF16 LoRA can introduce rounding; record its measured effect.
        _, merged_predictions = evaluate(reloaded, module, reloaded_processor, encoded, rows, device)
        merge_parity = compare_predictions(reload_predictions, merged_predictions, tolerance=5e-3)
        for name in ("clef_live_tape.py", "research_expansion_artifacts.py"):
            shutil.copyfile(Path(__file__).with_name(name), merged_dir / name)
        shutil.copyfile(args.output / "data-manifest.json", merged_dir / "data-manifest.json")
        (merged_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
        merged_card = card.replace("library_name: peft", "library_name: transformers").replace(
            "# Clawd Clef Research Decision Adapter", "# Clawd Clef Solana Research Model").replace(
            "LoRA text-backbone adapter plus the trained native Clef joint schema head.",
            "Standalone fine-tuned multimodal backbone with merged LoRA weights and the trained native Clef joint schema head.").replace(
            "Load with `reload_adapter` in the included\n`clef_research_training.py`, then call the upstream `systemone` API.",
            "Load with `load_release_model` in the included `export_clef_release.py`.\nThe returned model's native module exposes `systemone`. See the example below.")
        merged_card += """
## Load the standalone model and attach live read-only evidence

```python
import sys
from export_clef_release import load_release_model
from clef_live_tape import enrich_record_with_live_snapshot

model, processor = load_release_model("PATH_OR_HUB_REPO", device="cuda")
native = sys.modules[model.__class__.__module__]
record = {"model": "clawd-clef-research", "state": "Read observed service status.",
          "questions": {"status": {"type": "choice", "instructions": "What status was reported by the health endpoint?",
                                    "criteria": {"ok": "ok", "degraded": "degraded", "unknown": "No observation"}}}}
record, snapshot = enrich_record_with_live_snapshot(record)
response = native.systemone(model, processor, record)
print(response["answers"])
```

The runtime bridge reads health and public launch observations from
https://clawd-ws.fly.dev/ and records observation age separately from event age.
It does not execute trades. No tokenizer vocabulary additions were made.
Full weight files, the trained head, processor, and verified hashes are in this
repository; no base-model download is needed after obtaining this release.
"""
        (merged_dir / "README.md").write_text(merged_card)
        refresh_release_manifest(merged_dir)
        del reloaded
        gc.collect()
        torch.cuda.empty_cache()
        standalone, standalone_processor = load_release_model(merged_dir, device=device, dtype=torch.bfloat16,
                                                              attn_implementation="sdpa")
        standalone_metrics, standalone_predictions = evaluate(standalone, module, standalone_processor, encoded, rows, device)
        standalone_parity = compare_predictions(merged_predictions, standalone_predictions)
        metrics["standalone_reload_verification"] = {**standalone_parity, "merge_parity": merge_parity,
                                                     "metrics": standalone_metrics, "full_backbone_loaded_from_saved_shards": True}
        if args.live_tape:
            from clef_live_tape import capture_live_snapshot, encode_native_record, live_decision_records, snapshot_freshness
            runtime_snapshot = capture_live_snapshot(timeout=15, max_frames=4)
            runtime_rows = live_decision_records(runtime_snapshot)
            if not runtime_rows:
                raise ValueError("Saved standalone model could not obtain any fresh live evidence")
            runtime_encoded = [encode_native_record(module, standalone_processor, row, args.max_length) for row in runtime_rows]
            runtime_metrics, runtime_predictions = evaluate(standalone, module, standalone_processor, runtime_encoded, runtime_rows, device)
            (merged_dir / "live-runtime-verification.json").write_text(json.dumps({
                "snapshot": runtime_snapshot, "freshness": snapshot_freshness(runtime_snapshot),
                "metrics": runtime_metrics, "predictions": runtime_predictions, "autoExecute": False,
            }, indent=2) + "\n")
        (merged_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
        release = refresh_release_manifest(merged_dir, status="trained_and_standalone_reload_verified",
                                           reload_verification=metrics["standalone_reload_verification"])
        metadata["standalone_release"] = {"reload_verified": True, "directory": str(merged_dir)}
        if args.output_repo:
            merged_repo = args.output_repo + "-merged"
            api.create_repo(merged_repo, private=True, exist_ok=True)
            if not api.model_info(merged_repo).private:
                raise ValueError("Standalone output repository must be private")
            commit = api.upload_folder(repo_id=merged_repo, folder_path=str(merged_dir), commit_message="Publish standalone fine-tuned Clef backbone, decision head and verified native reload")
            metadata["standalone_release"].update(repo_id=merged_repo, revision=commit.oid, url=commit.commit_url)
        (args.output / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (args.output / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if args.output_repo:
            api.upload_folder(repo_id=args.output_repo, folder_path=str(args.output), commit_message="Record verified standalone model publication and reload evidence")
        print(json.dumps({"standalone_release": metadata["standalone_release"], "bytes": release["total_bytes"]}, indent=2), flush=True)
    print(json.dumps({"output": str(args.output), "output_repo": args.output_repo, "evaluation": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
