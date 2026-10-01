"""Native Clef head training, metrics, and reloadable adapter utilities."""
from __future__ import annotations

import importlib.util
import hashlib
import math
import json
from pathlib import Path
import sys

from clef_research_data import MODEL_ID, MODEL_REVISION


def load_upstream():
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                 allow_patterns=["joint_schema_model.py", "joint_head_config.json"]))
    spec = importlib.util.spec_from_file_location("clawd_clef_upstream", path / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def attach_lora(model, rank=16):
    import torch
    from peft import LoraConfig, get_peft_model
    # Explicit full names keep the vision encoder and LM output head frozen.
    targets = [name for name, child in model.language_model.named_modules()
               if isinstance(child, torch.nn.Linear) and
               ".language_model.layers." in name and not name.endswith("lm_head")]
    if not targets:
        raise ValueError("No text-backbone linear layers found for LoRA")
    model.language_model = get_peft_model(model.language_model, LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
        target_modules=targets, bias="none",
    ))
    model.head.requires_grad_(True)
    model.language_model.enable_input_require_grads()
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return targets


def supervised_loss(logits, encoded, rows):
    import torch
    import torch.nn.functional as functional
    losses = []
    for record_logits, record, row in zip(logits, encoded, rows, strict=True):
        for question_logits, question in zip(record_logits, record.questions, strict=True):
            target = question.option_ids.index(row["labels"][question.question_id])
            losses.append(functional.cross_entropy(question_logits.float().unsqueeze(0),
                                                   torch.tensor([target], device=question_logits.device)))
    if not losses:
        raise ValueError("No supervised questions")
    return torch.stack(losses).mean()


def encode_rows(module, processor, rows, max_length):
    encoded, kept, errors = [], [], []
    for row in rows:
        try:
            # Upstream truncates state silently. Measure full input and exclude
            # over-length cases explicitly so evidence is never cut unseen.
            record = module.encode_record(processor.tokenizer, row, max_length=1_000_000, processor=processor)
            if len(record.input_ids) > max_length:
                raise ValueError(f"input length {len(record.input_ids)} exceeds {max_length}")
            encoded.append(record)
            kept.append(row)
        except ValueError as exc:
            errors.append({"id": row["id"], "reason": str(exc)})
    return encoded, kept, errors


def evaluate(model, module, processor, encoded, rows, device):
    import torch
    if not rows:
        raise ValueError("Evaluation cohort is empty")
    model.eval()
    predictions, correct, nll, brier, chance = [], 0, 0.0, 0.0, 0.0
    with torch.inference_mode():
        for record, row in zip(encoded, rows, strict=True):
            logits = model(module.collate_records([record], processor.tokenizer.pad_token_id, device))[0]
            answers = {}
            for question, scores in zip(record.questions, logits, strict=True):
                probabilities = scores.float().softmax(-1)
                chance += 1 / len(question.option_ids)
                if not torch.isfinite(probabilities).all():
                    raise ValueError("Nonfinite evaluation probabilities")
                gold = question.option_ids.index(row["labels"][question.question_id])
                prediction = int(probabilities.argmax())
                correct += int(prediction == gold)
                nll -= float(probabilities[gold].clamp_min(1e-12).log())
                target = torch.zeros_like(probabilities)
                target[gold] = 1
                brier += float(((probabilities - target) ** 2).sum())
                answers[question.question_id] = {
                    "choice": question.option_ids[prediction], "label": question.option_ids[gold],
                    "probabilities": dict(zip(question.option_ids, probabilities.tolist())),
                }
            predictions.append({"id": row["id"], "answers": answers})
    questions = sum(len(record.questions) for record in encoded)
    return {"records": len(rows), "questions": questions, "accuracy": correct / questions,
            "nll": nll / questions, "multiclass_brier": brier / questions,
            "chance_accuracy": chance / questions,
            "scope": "Solana reference-answer selection and observed-field readback; not the official Decision Index"}, predictions


def compare_predictions(before, after, tolerance=1e-4):
    """Require identical records/questions/options and measure saved-load parity."""
    maximum = 0.0
    if not before or len(before) != len(after):
        raise ValueError("Reload verification needs matching nonempty cohorts")
    for expected, observed in zip(before, after, strict=True):
        if expected["id"] != observed["id"] or expected["answers"].keys() != observed["answers"].keys():
            raise ValueError("Reloaded predictions refer to different records or questions")
        for question, expected_answer in expected["answers"].items():
            actual = observed["answers"][question]
            if expected_answer["probabilities"].keys() != actual["probabilities"].keys():
                raise ValueError("Reloaded prediction options differ")
            for option, probability in expected_answer["probabilities"].items():
                difference = abs(probability - actual["probabilities"][option])
                if not math.isfinite(difference) or difference > tolerance:
                    raise ValueError(f"Reloaded probabilities differ by {difference}, tolerance {tolerance}")
                maximum = max(maximum, difference)
    return {"matched": True, "records": len(before), "max_probability_difference": maximum, "tolerance": tolerance}


def trainable_fingerprints(model):
    """Hash actual trainable parameter bytes to prove both paths changed."""
    import torch
    digests = {"lora": hashlib.sha256(), "head": hashlib.sha256()}
    counts = {key: 0 for key in digests}
    for name, parameter in sorted(model.named_parameters()):
        kind = "lora" if "lora_" in name else "head" if name.startswith("head.") else None
        if kind is None or not parameter.requires_grad:
            continue
        digest = digests[kind]
        digest.update(json.dumps([name, str(parameter.dtype), list(parameter.shape)]).encode())
        digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        counts[kind] += parameter.numel()
    if any(value == 0 for value in counts.values()):
        raise ValueError("Both LoRA and native decision head must be trainable")
    return {kind: {"sha256": digest.hexdigest(), "parameters": counts[kind]} for kind, digest in digests.items()}


def save_adapter(model, processor, module, output: Path, metadata):
    from safetensors.torch import save_file
    output.mkdir(parents=True, exist_ok=True)
    model.language_model.save_pretrained(output)
    processor.save_pretrained(output)
    save_file({name: value.detach().cpu().contiguous() for name, value in model.head.state_dict().items()},
              str(output / "joint_head.safetensors"))
    from huggingface_hub import hf_hub_download
    import shutil
    shutil.copyfile(hf_hub_download(MODEL_ID, "joint_head_config.json", revision=MODEL_REVISION), output / "joint_head_config.json")
    shutil.copyfile(Path(module.__file__), output / "joint_schema_model.py")
    (output / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")


def reload_adapter(module, path: Path, device="cuda"):
    from peft import PeftModel
    from safetensors.torch import load_file
    import torch
    from huggingface_hub import snapshot_download
    metadata = json.loads((path / "training.json").read_text())
    if metadata["model_revision"] != MODEL_REVISION:
        raise ValueError("Adapter base revision differs from pinned Clef revision")
    base = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    model, processor = module.load_release_model(base, device=device, dtype=torch.bfloat16)
    model.language_model = PeftModel.from_pretrained(model.language_model, path)
    model.head.load_state_dict(load_file(str(path / "joint_head.safetensors")), strict=True)
    return model.eval(), processor
