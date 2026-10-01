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

from clef_research_data import DATASET_ID, DATASET_REVISION, MODEL_ID, MODEL_REVISION, prepare_dataset, read_jsonl, training_source_hash
from clef_research_training import (attach_lora, encode_rows, evaluate, load_upstream,
                                   reload_adapter, save_adapter, supervised_loss)


def read_rows(path, limit=0):
    return read_jsonl(path, limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("local/clef-research-data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/clef-research"))
    parser.add_argument("--output-repo", help="Optional private Hub destination; requires HF_TOKEN")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
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
    cohorts = {}
    for split in ("train", "eval", "test"):
        limit = args.max_train_records if split == "train" else args.max_eval_records
        rows = read_rows(args.data_dir / f"{split}.jsonl", limit)
        encoded, kept, excluded = encode_rows(module, processor, rows, args.max_length)
        if not kept:
            raise ValueError(f"No {split} records fit the context limit")
        cohorts[split] = (encoded, kept)
        (args.output / f"{split}-excluded.json").write_text(json.dumps(excluded, indent=2) + "\n")
        manifest["outputs"][split].update(selected=len(rows), encoded=len(kept), context_excluded=len(excluded))
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
        "baseline_eval": baseline, "losses": [], "optimizer_steps": 0,
        "status": "training", "versions": {}, "autoExecute": False,
        "training_source_sha256": training_source_hash(),
    }
    for package in ("torch", "transformers", "peft", "huggingface_hub", "pyarrow"):
        metadata["versions"][package] = __import__(package).__version__
    rng = random.Random(args.seed)
    adapter_gradient_seen = head_gradient_seen = False

    def checkpoint():
        metadata["seconds"] = time.monotonic() - started
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
    for before, after in zip(saved_predictions, reload_predictions, strict=True):
        for question in before["answers"]:
            for option, probability in before["answers"][question]["probabilities"].items():
                if abs(probability - after["answers"][question]["probabilities"][option]) > 1e-4:
                    raise ValueError("Reloaded adapter probabilities differ from the saved model")
    metrics["reload_verification"] = {"matched": True, "records": verification_count, "metrics": reload_metrics}
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
Training task: select a reference research answer among four candidates.
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
    for name in ("clef_research_training.py", "clef_research_data.py"):
        shutil.copyfile(Path(__file__).with_name(name), args.output / name)
    if args.output_repo:
        api.upload_folder(repo_id=args.output_repo, folder_path=str(args.output), commit_message="Save trained Clef adapter, decision head, evaluation and verified reload")
    if args.export_merged:
        from export_clef_release import export_merged_release
        merged_dir = args.output.with_name(args.output.name + "-merged")
        release = export_merged_release(reloaded, reloaded_processor, module, merged_dir, metadata)
        # Ship the read-only runtime adapter alongside the standalone model.
        for name in ("clef_live_tape.py", "research_expansion_artifacts.py"):
            shutil.copyfile(Path(__file__).with_name(name), merged_dir / name)
        (merged_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (merged_dir / "README.md").write_text(card.replace("library_name: peft", "library_name: transformers").replace(
            "LoRA text-backbone adapter plus the trained native Clef joint schema head.",
            "Standalone fine-tuned multimodal backbone with merged LoRA weights and the trained native Clef joint schema head."))
        if args.output_repo:
            merged_repo = args.output_repo + "-merged"
            api.create_repo(merged_repo, private=True, exist_ok=True)
            if not api.model_info(merged_repo).private:
                raise ValueError("Standalone output repository must be private")
            api.upload_folder(repo_id=merged_repo, folder_path=str(merged_dir), commit_message="Publish standalone fine-tuned Clef backbone, decision head and live read-only adapter")
            release["repo_id"] = merged_repo
        print(json.dumps({"standalone_release": release}, indent=2), flush=True)
    print(json.dumps({"output": str(args.output), "output_repo": args.output_repo, "evaluation": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
