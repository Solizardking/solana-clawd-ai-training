#!/usr/bin/env python3
"""Two-stage QLoRA adaptation of the DavidAU trainable vision checkpoint."""
import argparse
import json
from pathlib import Path

BASE = "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU"
REVISION = "f3831969c184aa06fdea1bf1060f27a9c9b0eccd"


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_resume(checkpoint, stage, manifest):
    """Require a complete Trainer checkpoint and identical packaged inputs."""
    checkpoint = checkpoint.resolve()
    if checkpoint.parent.name != stage:
        raise ValueError("Checkpoint directory must belong to the selected stage")
    required = ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                "optimizer.pt", "scheduler.pt", "rng_state.pth")
    for name in required:
        if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size:
            raise ValueError(f"Incomplete resume checkpoint: {name}")
    saved_manifest = json.loads((checkpoint.parent.parent / "data-manifest.json").read_text())
    if saved_manifest != manifest:
        raise ValueError("Resume data manifest differs from the original training inputs")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state.get("global_step", 0) <= 0:
        raise ValueError("Resume checkpoint has no completed optimizer steps")
    return checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("27b", "4b"), default="27b")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("/content/clawd-chart-foundation-27b-lora"))
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="Per-stage step cap; -1 uses complete epochs")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--smoke", action="store_true", help="One optimizer step per stage on a small mixed image/text subset")
    p.add_argument("--hub-repo", help="Persist checkpoints to an existing private model repository")
    p.add_argument("--resume-checkpoint", type=Path, help="Trusted Trainer checkpoint including optimizer and RNG state")
    p.add_argument("--resume-stage", choices=("documents", "sft"))
    args = p.parse_args()
    base, revision = (BASE, REVISION) if args.profile == "27b" else (
        "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    manifest = json.loads((args.data / "manifest.json").read_text())
    if bool(args.resume_checkpoint) != bool(args.resume_stage):
        p.error("--resume-checkpoint and --resume-stage must be supplied together")
    if args.resume_checkpoint:
        if args.smoke:
            p.error("Cannot resume full checkpoints in smoke mode")
        args.resume_checkpoint = validate_resume(args.resume_checkpoint, args.resume_stage, manifest)
        if args.output.resolve() == args.resume_checkpoint.parent.parent:
            p.error("Use a fresh output directory to preserve the source checkpoint")
    sets = {name: read_rows(args.data / f"{name}.jsonl") for name in
            ("train", "validation", "test", "documents-train", "documents-validation")}
    if any(not rows for rows in sets.values()):
        raise ValueError("Every required training/evaluation split must be nonempty")
    ids = {s: {r["id"] for r in sets[s]} for s in ("train", "validation", "test")}
    if ids["train"] & (ids["validation"] | ids["test"]):
        raise ValueError("Training/evaluation duplicate leakage")
    for split in ("train", "validation", "test"):
        for row in sets[split]:
            for image in row["images"]:
                path = (args.data / image).resolve()
                if not path.is_relative_to(args.data.resolve()) or not path.is_file():
                    raise ValueError("Missing or invalid packaged image")
    print(json.dumps({"base": base, "revision": revision, "counts": manifest["counts"]}, indent=2))
    if args.preflight:
        return
    if args.smoke:
        for name, rows in sets.items():
            images = [r for r in rows if r.get("images")]
            text = [r for r in rows if not r.get("images")]
            sets[name] = images[:2] + text[:2]
        args.max_steps = 1

    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments, set_seed

    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 39 * 1024**3:
        raise RuntimeError("This training recipe requires a CUDA GPU with at least 40 GB VRAM. Select A100 or larger.")
    set_seed(42)
    processor = AutoProcessor.from_pretrained(base, revision=revision)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    filtered_counts = {}
    for name, rows in sets.items():
        kept = []
        for row in rows:
            text = row.get("text")
            if text is None:
                text = processor.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
            if len(processor.tokenizer.encode(text, add_special_tokens=False)) <= 6144:
                kept.append(row)
        filtered_counts[name] = len(rows) - len(kept)
        sets[name] = kept
        if not kept:
            raise ValueError(f"No records fit the token budget in {name}")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(base, revision=revision,
        quantization_config=quant if args.profile == "27b" else None,
        device_map={"": 0}, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.profile == "27b":
        model = prepare_model_for_kbit_training(model)
    else:
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        exclude_modules=r".*visual.*"))
    model.config.use_cache = False

    def collate(rows):
        texts, images = [], []
        is_document = "text" in rows[0]
        for row in rows:
            if is_document:
                texts.append(row["text"])
                continue
            messages = [{**m} for m in row["messages"]]
            if row["images"]:
                for msg in messages:
                    if msg["role"] == "user":
                        msg["content"] = [{"type": "image"} for _ in row["images"]] + [{"type": "text", "text": msg["content"]}]
                        break
                for image in row["images"]:
                    with Image.open(args.data / image) as im:
                        im = im.convert("RGB")
                        im.thumbnail((512, 512))
                        images.append(im.copy())
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        batch = processor(text=texts, images=images or None, padding=True, return_tensors="pt")
        # Never truncate image-token spans. Oversized records must be filtered before the run.
        if batch["input_ids"].shape[1] > 8192:
            raise ValueError("Record exceeds 8192 tokens; filter/rechunk it rather than truncating image tokens")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for token in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
            token_id = processor.tokenizer.get_vocab().get(token)
            if token_id is not None:
                labels[labels == token_id] = -100
        batch["labels"] = labels
        return batch

    args.output.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(args.output / "adapter")
    (args.output / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    class PersistCheckpoint(TrainerCallback):
        def on_save(self, training_args, state, control, **kwargs):
            if args.hub_repo:
                from huggingface_hub import HfApi
                HfApi().upload_folder(repo_id=args.hub_repo, folder_path=str(args.output),
                    commit_message=f"Persist {Path(training_args.output_dir).name} step {state.global_step}")

    metrics = {"base": base, "revision": revision, "profile": args.profile, "smoke_only": args.smoke, "records_removed_for_length": filtered_counts, "max_steps_per_stage": args.max_steps}
    if args.resume_checkpoint:
        metrics["resume"] = {"stage": args.resume_stage, "checkpoint": str(args.resume_checkpoint)}
        previous = args.resume_checkpoint.parent.parent / "metrics.json"
        if args.resume_stage == "sft":
            prior = json.loads(previous.read_text()) if previous.is_file() else {}
            for key in ("documents", "documents_eval"):
                if key in prior:
                    metrics[key] = prior[key]
            metrics["prior_stage_metrics_available"] = "documents" in metrics
    for stage, train_key, eval_key in (("documents", "documents-train", "documents-validation"), ("sft", "train", "validation")):
        if args.resume_stage == "sft" and stage == "documents":
            continue
        training = TrainingArguments(output_dir=str(args.output / stage), per_device_train_batch_size=1,
            per_device_eval_batch_size=1, gradient_accumulation_steps=16, learning_rate=5e-5,
            num_train_epochs=args.epochs, max_steps=args.max_steps, bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}, optim="adamw_torch", logging_steps=10,
            save_steps=100, save_total_limit=2, eval_strategy="steps", eval_steps=500,
            remove_unused_columns=False, report_to="none", seed=42)
        trainer = Trainer(model=model, args=training, train_dataset=sets[train_key],
                          eval_dataset=sets[eval_key], data_collator=collate, callbacks=[PersistCheckpoint()])
        resume = str(args.resume_checkpoint) if stage == args.resume_stage else None
        metrics[stage] = trainer.train(resume_from_checkpoint=resume).metrics
        metrics[stage + "_eval"] = trainer.evaluate()
        trainer.save_model(str(args.output / "adapter"))
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if args.hub_repo:
            from huggingface_hub import HfApi
            HfApi().upload_folder(repo_id=args.hub_repo, folder_path=str(args.output),
                                 commit_message=f"Persist completed {stage} stage and metrics")
    metrics["test"] = trainer.evaluate(eval_dataset=sets["test"], metric_key_prefix="test")
    processor.save_pretrained(args.output / "adapter")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "adapter/README.md").write_text(
        f"---\nbase_model: {base}\n---\n# Clawd Chart Foundation {args.profile.upper()} LoRA\n\n"
        f"Smoke-only run: {args.smoke}. Max steps per stage: {args.max_steps}.\n\n"
        "Two-stage document and chart/text adaptation. This is an adapter; load it with the pinned base checkpoint.\n\n"
        f"Base revision: `{revision}`. Training uses full-sequence loss with padding and image markers masked. "
        "Loss metrics do not establish trading profitability. See metrics.json and data-manifest.json for evidence.\n"
        "Historical token sample: Data from SolArchive.org (CC BY 4.0).\n")
    print("Saved trained adapter and evaluation metrics:", args.output)


if __name__ == "__main__":
    main()
