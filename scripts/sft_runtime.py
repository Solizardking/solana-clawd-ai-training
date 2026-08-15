"""Shared LoRA SFT runtime: SFTConfig adaptation + multi-source mix.

Used by `scripts/train_lora.py` at the `[5/6] Building SFT config` site and by
dataset resolution so dry-run / tests never load Qwen3.8-27B weights.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_SFT_SOURCES = (
    "solana1_yourgpt.jsonl",
    "trainingday.jsonl",
)
DISTILLATION_REPO = "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation"
DISTILLATION_CONFIGS = ("sft_balanced", "canonical", "glm47_native")


def resolve_data_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.exists():
        return path
    alt = _REPO_ROOT / path_str
    if alt.exists():
        return alt
    return path


def sft_config_accepted_keys() -> set[str]:
    from trl import SFTConfig

    keys: set[str] = set()
    for name, param in inspect.signature(SFTConfig.__init__).parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        keys.add(name)
    return keys


def warmup_ratio_to_steps(
    ratio: float,
    train_kwargs: dict[str, Any],
    train_size: int | None = None,
) -> int:
    """Map a YAML `warmup_ratio` onto a positive `warmup_steps` value."""
    ratio = float(ratio)
    if ratio <= 0:
        return 0
    max_steps = int(train_kwargs.get("max_steps") or -1)
    if max_steps > 0:
        total_steps = max_steps
    else:
        batch = max(1, int(train_kwargs.get("per_device_train_batch_size") or 1))
        accum = max(1, int(train_kwargs.get("gradient_accumulation_steps") or 1))
        epochs = float(train_kwargs.get("num_train_epochs") or 1.0)
        size = int(train_size or 0)
        updates_per_epoch = max(1, math.ceil(size / (batch * accum))) if size > 0 else 100
        total_steps = max(1, int(math.ceil(updates_per_epoch * epochs)))
    return max(1, int(ratio * total_steps))


def adapt_sft_training_kwargs(
    train_kwargs: dict[str, Any],
    train_size: int | None = None,
    accepted_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Filter stale TrainingArguments keys and map `warmup_ratio` → `warmup_steps`."""
    adapted = dict(train_kwargs)
    accepted = accepted_keys if accepted_keys is not None else sft_config_accepted_keys()
    ratio = adapted.pop("warmup_ratio", None)
    existing_steps = adapted.get("warmup_steps")
    if ratio is not None and "warmup_steps" in accepted:
        if existing_steps in (None, 0, "0"):
            adapted["warmup_steps"] = warmup_ratio_to_steps(float(ratio), train_kwargs, train_size)
    return {key: value for key, value in adapted.items() if key in accepted}


def build_sft_config(
    cfg: dict[str, Any],
    *,
    output_dir: str,
    train_size: int | None = None,
    device: str = "cpu",
    cpt_stage: bool = False,
    report_to: list[str] | str | None = None,
) -> Any:
    """Construct the installed TRL `SFTConfig` from a YAML training block.

    This is the shipped builder used at `[5/6] Building SFT config`.
    """
    from trl import SFTConfig

    train_kwargs = dict(cfg.get("training") or {})
    sft_kwargs = cfg.get("sft", {}) or {}
    if device != "cuda":
        train_kwargs["bf16"] = False
        train_kwargs["fp16"] = False
        train_kwargs["tf32"] = False
    if report_to is None:
        report_to = train_kwargs.pop("report_to", ["none"])
    else:
        train_kwargs.pop("report_to", None)
    adapted = adapt_sft_training_kwargs(train_kwargs, train_size=train_size)
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "max_length": cfg.get("max_seq_length", 4096),
        "dataset_text_field": cfg.get("cpt_text_field" if cpt_stage else "text_field", "text"),
        "packing": sft_kwargs.get("packing", False),
        "assistant_only_loss": sft_kwargs.get("assistant_only_loss", True),
        "report_to": report_to,
        **adapted,
    }
    accepted = sft_config_accepted_keys()
    kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return SFTConfig(**kwargs)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(_as_text(item.get("text", item.get("content"))))
        return "".join(parts)
    if isinstance(value, dict):
        if "text" in value or "content" in value:
            return _as_text(value.get("text", value.get("content")))
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_sft_text(messages: list[dict[str, str]]) -> str:
    parts = [f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>" for m in messages]
    return "\n".join(parts)


def normalize_sft_record(record: dict[str, Any], source: str) -> dict[str, Any] | None:
    """Normalize Alpaca / chat / text rows to `{messages, text, source}`."""
    raw_messages = record.get("messages")
    if isinstance(raw_messages, list) and raw_messages:
        messages: list[dict[str, str]] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or item.get("from") or "user")
            content = _as_text(item.get("content", item.get("value", item.get("text"))))
            messages.append({"role": role, "content": content})
        if messages:
            return {"messages": messages, "text": render_sft_text(messages), "source": source}

    conversations = record.get("conversations")
    if isinstance(conversations, list) and conversations:
        messages = []
        for item in conversations:
            if not isinstance(item, dict):
                continue
            role = str(item.get("from") or item.get("role") or "user")
            if role in {"human", "user"}:
                role = "user"
            elif role in {"gpt", "assistant"}:
                role = "assistant"
            messages.append({"role": role, "content": _as_text(item.get("value", item.get("content")))})
        if messages:
            return {"messages": messages, "text": render_sft_text(messages), "source": source}

    if "instruction" in record and "output" in record:
        user = _as_text(record.get("instruction"))
        extra = _as_text(record.get("input"))
        if extra.strip():
            user = f"{user}\n\n{extra}"
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": _as_text(record.get("output"))},
        ]
        return {"messages": messages, "text": render_sft_text(messages), "source": source}

    prompt = record.get("prompt") or record.get("question")
    response = record.get("response") or record.get("answer") or record.get("completion")
    if prompt and response:
        messages = [
            {"role": "user", "content": _as_text(prompt)},
            {"role": "assistant", "content": _as_text(response)},
        ]
        return {"messages": messages, "text": render_sft_text(messages), "source": source}

    text = record.get("text")
    if isinstance(text, str) and text.strip():
        messages = [{"role": "user", "content": text}]
        return {"messages": messages, "text": text, "source": source}
    return None


def _normalize_mapped(record: dict[str, Any], source: str) -> dict[str, Any]:
    normalized = normalize_sft_record(record, source)
    if normalized is None:
        return {"messages": [], "text": "", "source": source, "_keep": False}
    normalized["_keep"] = True
    return normalized


def _drop_keep(dataset: Dataset) -> Dataset:
    if "_keep" in dataset.column_names:
        return dataset.remove_columns(["_keep"])
    return dataset


def load_and_normalize_local(path_str: str, dataset_format: str | None = None) -> tuple[DatasetDict, str, int]:
    path = resolve_data_path(path_str)
    source = path.name if path.exists() else path_str
    loaded = load_local_dataset(str(path), dataset_format)
    normalized_splits: dict[str, Dataset] = {}
    kept = 0
    for split, split_ds in loaded.items():
        mapped = split_ds.map(
            lambda record, src=source: _normalize_mapped(dict(record), src),
            remove_columns=split_ds.column_names,
        )
        mapped = mapped.filter(lambda record: bool(record["_keep"]))
        mapped = _drop_keep(mapped)
        if len(mapped) > 0:
            normalized_splits[split] = mapped
            kept += len(mapped)
    if not normalized_splits:
        raise ValueError(f"No usable SFT rows in {path}")
    return DatasetDict(normalized_splits), source, kept


def load_local_dataset(dataset_path: str, dataset_format: str | None) -> DatasetDict:
    path = resolve_data_path(dataset_path)
    inferred_format = (dataset_format or "").strip().lower()
    if path.is_dir():
        parquet_files = {
            split: str(path / f"{split}.parquet")
            for split in ("train", "eval", "test")
            if (path / f"{split}.parquet").exists()
        }
        if parquet_files:
            loaded = load_dataset("parquet", data_files=parquet_files)
            if isinstance(loaded, DatasetDict):
                return loaded
        try:
            loaded = load_from_disk(str(path))
        except Exception:
            loaded = load_dataset(str(path))
    else:
        if not inferred_format:
            suffix = path.suffix.lower()
            if suffix in {".jsonl", ".json"}:
                inferred_format = "json"
            else:
                inferred_format = "text"
        if inferred_format == "json":
            loaded = load_dataset("json", data_files={"train": str(path)})
        elif inferred_format == "text":
            loaded = load_dataset("text", data_files={"train": str(path)})
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")

    if isinstance(loaded, DatasetDict):
        return loaded
    return DatasetDict({"train": loaded})


def try_load_distillation_config(
    config_name: str,
    repo: str = DISTILLATION_REPO,
    max_rows: int | None = None,
) -> tuple[DatasetDict | None, str | None]:
    """Load one Hub distillation config. Returns (dataset, error)."""
    try:
        loaded = load_dataset(repo, config_name)
    except Exception as exc:  # Hub / auth / schema failures are optional
        return None, f"{repo}:{config_name}: {exc}"
    if not isinstance(loaded, DatasetDict):
        loaded = DatasetDict({"train": loaded})
    source = f"{repo}:{config_name}"
    normalized_splits: dict[str, Dataset] = {}
    for split, split_ds in loaded.items():
        if max_rows is not None:
            split_ds = split_ds.select(range(min(int(max_rows), len(split_ds))))
        mapped = split_ds.map(
            lambda record, src=source: _normalize_mapped(dict(record), src),
            remove_columns=split_ds.column_names,
        )
        mapped = mapped.filter(lambda record: bool(record["_keep"]))
        mapped = _drop_keep(mapped)
        if len(mapped) > 0:
            normalized_splits[split] = mapped
    if not normalized_splits:
        return None, f"{source}: loaded but produced 0 normalized rows"
    return DatasetDict(normalized_splits), None


def _configured_local_sources(cfg: dict[str, Any], use_cpt_stage: bool) -> list[dict[str, Any]]:
    sources = cfg.get("dataset_sources")
    if isinstance(sources, list) and sources:
        resolved: list[dict[str, Any]] = []
        for item in sources:
            if isinstance(item, str):
                resolved.append({"path": item})
            elif isinstance(item, dict) and item.get("path"):
                resolved.append(item)
        return resolved
    local_path = cfg.get("dataset_path")
    local_format = cfg.get("dataset_format")
    if use_cpt_stage:
        local_path = cfg.get("cpt_dataset_path", local_path)
        local_format = cfg.get("cpt_dataset_format", local_format)
    if local_path:
        return [{"path": local_path, "format": local_format}]
    return []


def _configured_hub(cfg: dict[str, Any]) -> tuple[str, tuple[str, ...], int | None]:
    hub = cfg.get("dataset_hub") or {}
    if isinstance(hub, dict) and (hub.get("repo") or hub.get("configs")):
        repo = str(hub.get("repo") or DISTILLATION_REPO)
        configs = tuple(hub.get("configs") or DISTILLATION_CONFIGS)
        max_rows = hub.get("max_rows_per_config")
        return repo, tuple(str(name) for name in configs), int(max_rows) if max_rows else None
    extra = cfg.get("dataset_hub_configs")
    if extra:
        return DISTILLATION_REPO, tuple(str(name) for name in extra), None
    return DISTILLATION_REPO, (), None


def _merge_splits(parts: list[DatasetDict]) -> DatasetDict:
    by_split: dict[str, list[Dataset]] = {}
    for part in parts:
        for split, split_ds in part.items():
            by_split.setdefault(split, []).append(split_ds)
    merged = {split: concatenate_datasets(items) if len(items) > 1 else items[0] for split, items in by_split.items()}
    return DatasetDict(merged)


def resolve_dataset(cfg: dict[str, Any], use_cpt_stage: bool = False) -> tuple[DatasetDict, str]:
    """Resolve the LoRA SFT mix: local JSONLs + optional Hub distillation configs."""
    parts: list[DatasetDict] = []
    labels: list[str] = []
    local_sources = _configured_local_sources(cfg, use_cpt_stage)
    for item in local_sources:
        path = item["path"]
        if not resolve_data_path(path).exists():
            print(f"  WARNING: local source missing, skipping: {path}")
            continue
        loaded, source, kept = load_and_normalize_local(path, item.get("format"))
        print(f"  local source={source} rows={kept}")
        parts.append(loaded)
        labels.append(source)

    repo, hub_configs, max_rows = _configured_hub(cfg)
    hub_errors: list[str] = []
    for name in hub_configs:
        loaded, error = try_load_distillation_config(name, repo=repo, max_rows=max_rows)
        if error:
            print(f"  Hub load skipped ({error})")
            hub_errors.append(error)
            continue
        assert loaded is not None
        rows = sum(len(split_ds) for split_ds in loaded.values())
        source = f"{repo}:{name}"
        print(f"  hub source={source} rows={rows}")
        parts.append(loaded)
        labels.append(source)
    if hub_errors:
        cfg["_hub_errors"] = hub_errors

    if not parts:
        dataset_repo = cfg.get("dataset_repo")
        if dataset_repo and not use_cpt_stage:
            try:
                loaded = load_dataset(dataset_repo)
                if not isinstance(loaded, DatasetDict):
                    loaded = DatasetDict({"train": loaded})
                return loaded, dataset_repo
            except Exception as exc:
                print(f"  Hub load failed ({exc}), falling back to local dataset")
        raise FileNotFoundError("No local dataset path configured and Hub dataset could not be loaded.")

    merged = _merge_splits(parts)
    cfg["_resolved_sources"] = list(labels)
    return merged, "+".join(labels)
