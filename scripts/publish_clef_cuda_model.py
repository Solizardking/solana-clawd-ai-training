#!/usr/bin/env python3
"""Verify a pinned private CUDA release and plan its public standalone model.

The default performs read-only Hub requests and writes a small local overlay.
Only --push creates a public commit. Safetensors payloads are never downloaded:
their LFS hashes and bounded HTTP Range headers are checked, then the Hub copies
them between repositories. Cross-repository copies require the same Hub storage
region. The original proof-bound release manifest is retained byte for byte.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct

from clef_research_data import DATASET_ID, MODEL_ID, MODEL_REVISION, model_input_exclusion_reason
from clef_live_tape import live_decision_records, snapshot_freshness, validate_snapshot
from export_clef_cuda_release import REQUIRED_RUNTIME, _validate_training, _parity, _positive, _number, _time
from export_clef_mps_release import EXPANDED_DATASET_REVISION, EXPANDED_INPUT_ROWS
from publish_clef_local_model import validate_evaluation

DEFAULT_REPO = "solanaclawd/clef-solana-research"
DEFAULT_SOURCE = "solanaclawd/clef-solana-research-cuda-full"
QUALIFIED_BUNDLE_SHA256 = "294b5843cab2f7253857263191a4c227b9255252cabeb366a15ca0b640b3c76e"
SOURCE_LICENSE_SHA256 = "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
SOURCE_CARD_SHA256 = "b0211b6ca10038b3168dde51f1482088b8c6b0fdccbbd765fce237dbc24f5d76"
SELECTED_TRAIN_RECORDS = 60635
SIDECAR_LIMIT = 64 * 1024 ** 2
TOTAL_SIDECAR_LIMIT = 192 * 1024 ** 2
HEADER_LIMIT = 2 * 1024 ** 2
OVERLAY = {"README.md", "source-trained-model-card.md", "source-release.json",
           "inference-evidence.json", "publication-provenance.json"}
RUNTIME_PROVENANCE = {"train_clef_cuda.py", "train_clef_local.py", "clef_research_data.py",
                      "clef_research_training.py", "clef_live_tape.py", "research_expansion_artifacts.py"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
HOME_PATH = re.compile(r"(?:/Users/|/home/|[A-Za-z]:\\Users\\)")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def load_json(data):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field in publication evidence")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON evidence")))


def safe_name(name):
    path = PurePosixPath(name)
    if (not isinstance(name, str) or not name or path.is_absolute() or len(path.parts) != 1
        or path.as_posix() != name or "\\" in name or name.startswith(".")
        or "__pycache__" in name or name.endswith(".lock") or "failed-before" in name):
        raise ValueError("Unsafe or operational release filename")
    return name


def check_text(data):
    text = data.decode("utf-8")
    if HOME_PATH.search(text) or model_input_exclusion_reason(text):
        raise ValueError("Public sidecar contains private home paths or signing/credential material")
    return text


def read_range(api, repo, name, revision, start, end, expected_size):
    """Reject ignored Range requests before reading any response payload."""
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session
    headers = build_hf_headers(token=api.token)
    headers["Range"] = f"bytes={start}-{end}"
    url = hf_hub_url(repo, name, revision=revision, endpoint=api.endpoint)
    with get_session().stream("GET", url, headers=headers, timeout=30) as response:
        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{expected_size}":
            raise ValueError("Hub did not honor the bounded tensor-header Range request")
        result = bytearray()
        for chunk in response.iter_bytes(chunk_size=65536):
            result.extend(chunk)
            if len(result) > end - start + 1:
                raise ValueError("Tensor-header response exceeded its requested byte range")
        if len(result) != end - start + 1:
            raise ValueError("Incomplete tensor-header byte range")
    return bytes(result)


def remote_header(api, repo, name, revision, size):
    if type(size) is not int or size < 10:
        raise ValueError("Missing actual safetensors file size")
    length = struct.unpack("<Q", read_range(api, repo, name, revision, 0, 7, size))[0]
    if not 2 <= length <= HEADER_LIMIT or length + 8 > size:
        raise ValueError("Safetensors header exceeds the bounded metadata limit")
    header = load_json(read_range(api, repo, name, revision, 8, length + 7, size))
    return validate_header(header, size - length - 8)


def validate_header(header, payload_size):
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors header")
    widths = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
              "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    intervals = []
    tensors = {name: item for name, item in header.items() if name != "__metadata__"}
    if not tensors:
        raise ValueError("Empty safetensors tensor inventory")
    for name, item in tensors.items():
        if not isinstance(name, str) or not isinstance(item, dict) or item.get("dtype") not in widths:
            raise ValueError("Unsupported safetensors tensor metadata")
        shape, offsets = item.get("shape"), item.get("data_offsets")
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
            or not isinstance(offsets, list) or len(offsets) != 2
            or any(type(n) is not int for n in offsets)):
            raise ValueError("Invalid tensor shapes or data offsets")
        start, end = offsets
        if start < 0 or end < start or end > payload_size or end - start != math.prod(shape) * widths[item["dtype"]]:
            raise ValueError("Tensor offsets do not cover the recorded dtype and shape")
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("Tensor payload has overlapping or missing byte regions")
        cursor = end
    if cursor != payload_size:
        raise ValueError("Tensor payload size differs from its complete header inventory")
    return tensors


def verify_inventory(api, repo, revision, files, download, *, prefix="", require_private=None):
    """LFS hashes authenticate entire tensors; downloaded files are sidecars only."""
    info = api.model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision or require_private is not None and info.private is not require_private:
        raise ValueError("Source/destination commit or visibility differs from the explicit plan")
    siblings = {item.rfilename: item for item in info.siblings}
    if prefix and {name[len(prefix):] for name in siblings if name.startswith(prefix)} != set(files):
        raise ValueError("Private release has unrecorded or missing files")
    if not prefix and set(siblings) - {".gitattributes"} != set(files):
        raise ValueError("Published model has unrecorded or missing files")
    sidecars, headers, downloaded = {}, {}, 0
    for name, expected in sorted(files.items()):
        safe_name(name)
        if type(expected.get("bytes")) is not int or expected["bytes"] < 1 or not SHA256.fullmatch(expected.get("sha256", "")):
            raise ValueError("Invalid release file size/hash inventory")
        remote = siblings.get(prefix + name)
        if remote is None or remote.size != expected["bytes"]:
            raise ValueError("Hub artifact size differs from its recorded inventory")
        lfs = getattr(remote, "lfs", None)
        if lfs is not None and (lfs.sha256 != expected["sha256"] or lfs.size != expected["bytes"]):
            raise ValueError("Hub LFS SHA256/size differs from the complete artifact inventory")
        if name.endswith(".safetensors"):
            if lfs is None:
                raise ValueError("Weight verification requires Hub LFS SHA256 metadata")
            headers[name] = remote_header(api, repo, prefix + name, revision, expected["bytes"])
        else:
            downloaded += expected["bytes"]
            if expected["bytes"] > SIDECAR_LIMIT or downloaded > TOTAL_SIDECAR_LIMIT:
                raise ValueError("Sidecar download would exceed the explicit bounded budget")
            data = Path(download(repo, prefix + name, revision)).read_bytes()
            if len(data) != expected["bytes"] or digest(data) != expected["sha256"]:
                raise ValueError("Downloaded sidecar differs from its recorded hash")
            check_text(data)
            sidecars[name] = data
    return info, sidecars, headers


def validate_tensors(manifest, sidecars, headers):
    backbone = manifest.get("backbone", {})
    shards = backbone.get("shards")
    if not isinstance(shards, list) or not shards or len(set(shards)) != len(shards):
        raise ValueError("Standalone backbone shard inventory is incomplete")
    names = {}
    tensor_map = {}
    for shard in shards:
        safe_name(shard)
        if shard not in headers or not re.fullmatch(r"model(?:-\d+-of-\d+)?\.safetensors", shard):
            raise ValueError("Standalone backbone has unsupported or absent shards")
        for name, tensor in headers[shard].items():
            if name in names or "lora_" in name or ".base_layer." in name:
                raise ValueError("Duplicate or unmerged adapter backbone tensors")
            names[name], tensor_map[name] = shard, tensor
    index = backbone.get("index")
    if index:
        if index != "model.safetensors.index.json" or load_json(sidecars[index]).get("weight_map") != names:
            raise ValueError("Backbone index differs from actual bounded safetensors headers")
    elif shards != ["model.safetensors"]:
        raise ValueError("Sharded model requires its exact tensor index")
    if (backbone.get("format") != "safetensors" or backbone.get("tensor_count") != len(names)
        or backbone.get("vision_tensor_count") != sum(".visual." in key for key in names)):
        raise ValueError("Manifest tensor counts differ from actual safetensors headers")
    if set(headers) != set(shards) | {"joint_head.safetensors"}:
        raise ValueError("Unexpected or absent standalone weight artifacts")
    quantized = [name for name in names if name.endswith(".weight.quant_state.bitsandbytes__nf4")]
    if len(quantized) != 496 or not any(".visual." in key for key in names):
        raise ValueError("Expected complete original 27B text NF4 and BF16 vision backbone")
    for marker in quantized:
        weight = marker.removesuffix(".quant_state.bitsandbytes__nf4")
        if tensor_map.get(weight, {}).get("dtype") != "U8" or not {weight + ".absmax", weight + ".quant_map", weight + ".nested_absmax", weight + ".nested_quant_map"} <= set(names):
            raise ValueError("NF4 double-quantization tensor state is incomplete")
    if any(item["dtype"] != "BF16" for key, item in tensor_map.items() if ".visual." in key):
        raise ValueError("Original vision tensors must remain BF16")
    if tensor_map.get("lm_head.weight", {}).get("dtype") != "BF16" or any(item["dtype"] != "BF16" for item in headers["joint_head.safetensors"].values()):
        raise ValueError("Native head and readable output embedding must remain BF16")
    config = load_json(sidecars["config.json"])
    quant = config.get("quantization_config", {})
    if (quant.get("load_in_4bit") is not True or quant.get("bnb_4bit_quant_type") != "nf4"
        or quant.get("bnb_4bit_use_double_quant") is not True
        or not {"model.visual", "lm_head"} <= set(quant.get("llm_int8_skip_modules", []))
        or config.get("text_config", {}).get("num_hidden_layers") != 64):
        raise ValueError("Standalone configuration is not the complete original Clef 27B NF4 model")
    return {"backbone_tensor_count": len(names), "nf4_linear_layers": len(quantized),
            "head_tensor_count": len(headers["joint_head.safetensors"]),
            "tensor_payload_downloaded_locally": False,
            "integrity_scope": "Complete file LFS SHA256/size plus bounded tensor headers; producer gradient/changed-parameter/merge/reload proofs are validated, not recomputed without a GPU."}


def validate_full_release(manifest, sidecars, headers, root_training, completion):
    if (manifest.get("format") != "clef-merged-release-v1" or manifest.get("status") != "trained_merged_and_standalone_reload_verified"
        or manifest.get("standalone_backbone") is not True or manifest.get("reload_verified") is not True
        or manifest.get("execution_backend") != "cuda" or manifest.get("device") != "cuda"
        or manifest.get("cpu_offload") is not False or manifest.get("source") != {"repo_id": MODEL_ID, "revision": MODEL_REVISION}):
        raise ValueError("Publication requires the completed genuine CUDA standalone release")
    training = load_json(sidecars["training.json"])
    if training != manifest.get("training") or training.get("status") != "trained_and_standalone_reload_verified":
        raise ValueError("Training sidecar and standalone release metadata differ")
    _validate_training(training)
    if training["cohorts"]["train"]["selected"] != SELECTED_TRAIN_RECORDS or training["max_length"] != 2048 or training["epochs"] != 1:
        raise ValueError("Publication requires all 60,635 original context-eligible decisions for one complete epoch")
    if manifest.get("total_bytes") != sum(item["bytes"] for item in manifest["files"].values()):
        raise ValueError("Standalone total bytes differ from its full file inventory")
    if set(manifest["files"]) & (OVERLAY - {"README.md"}):
        raise ValueError("Publication requires an original producer release, not a previous overlay")
    required = REQUIRED_RUNTIME | {"README.md", "config.json", "processor_config.json", "tokenizer_config.json",
                                  "tokenizer.json", "joint_schema_model.py", "joint_head.safetensors",
                                  "joint_head_config.json", "training.json", "conversion_manifest.json"}
    if not required <= set(manifest["files"]):
        raise ValueError("Standalone release lacks its full licensing, native runtime or provenance contract")
    for key, tolerance in (("merge_verification", .005), ("reload_verification", .0001)):
        _parity(manifest.get(key), tolerance, key)
    if training.get("standalone_reload_verification") != manifest["reload_verification"]:
        raise ValueError("Standalone reload proofs differ")
    load = manifest.get("cuda_standalone_load", {})
    kernel = load.get("kernel_execution", {})
    if (load.get("backend") != "cuda" or load.get("device") != "cuda" or load.get("actual_cuda_parameters") is not True
        or load.get("fresh_model_instance") is not True or load.get("artifact_root") != "."
        or kernel.get("implementation") != ("installed_fla" if training["use_kernels"] else "native_torch")
        or kernel.get("hub_kernel_downloads") is not False):
        raise ValueError("Fresh standalone reload lacks actual CUDA or consistent kernel math evidence")
    _positive(kernel.get("cuda_forward_calls"), "fresh CUDA forward calls")
    if training["use_kernels"] and kernel.get("fla_core_version") != "0.5.2":
        raise ValueError("Fresh standalone CUDA math differs from the pinned kernel")
    _time(load.get("loaded_at"))
    if not SHA256.fullmatch(load.get("source_release_manifest_sha256", "")):
        raise ValueError("Saved CUDA reload lacks its original pending-manifest binding")
    runtime = training.get("runtime_source_sha256", {})
    if set(runtime) != RUNTIME_PROVENANCE or any(runtime[name] != manifest["files"][name]["sha256"] for name in runtime):
        raise ValueError("CUDA runtime differs from its recorded training source hashes")
    critical = ("model", "model_revision", "dataset", "dataset_revision", "input_raw_row_counts", "input_parquet_sha256",
                "security_filter", "cohorts", "prepared_split_counts", "optimizer_steps", "planned_steps", "run_mode",
                "trained_records", "all_planned_steps_completed", "complete_selected_training_epochs", "epochs",
                "microbatch_size", "gradient_accumulation", "stage_manifest_sha256", "runtime_source_sha256",
                "training_order_sha256", "trained_record_ids_sha256", "pilot_trainable_fingerprints",
                "initial_trainable_fingerprints", "trained_trainable_fingerprints", "trainable_fingerprints",
                "finite_trainables", "gradient_evidence", "migration_verification", "reload_verification",
                "kernel_execution", "use_kernels", "capacity_probe", "lora", "live_training_snapshot")
    if any(root_training.get(key) != training.get(key) for key in critical):
        raise ValueError("Final release differs from the worker's recorded full training run")
    if (root_training.get("job_bundle_sha256") != QUALIFIED_BUNDLE_SHA256
        or root_training.get("standalone_release", {}).get("reload_verified") is not True
        or completion.get("status") != "verified_artifacts_uploaded" or completion.get("mode") != "full"
        or completion.get("bundle_sha256") != QUALIFIED_BUNDLE_SHA256
        or not REVISION.fullmatch(completion.get("input_revision", ""))):
        raise ValueError("Full worker did not complete the exact qualified immutable bundle")
    snapshot = load_json(sidecars["live-training-snapshot.json"])
    validate_snapshot(snapshot)
    recorded = training["live_training_snapshot"]
    if (model_input_exclusion_reason(snapshot) or recorded.get("file") != "live-training-snapshot.json"
        or recorded.get("sha256") != digest(sidecars["live-training-snapshot.json"])
        or recorded.get("captured_at") != snapshot["captured_at"]):
        raise ValueError("Exact safe historical training snapshot is missing")
    metrics = validate_evaluation(load_json(sidecars["evaluation.json"]))
    for split in ("train", "eval", "test"):
        cohort = training["cohorts"][split]
        _positive(cohort.get("selected"), "selected cohort")
        if (cohort.get("prepared") != training["prepared_split_counts"].get(split)
            or cohort["selected"] + len(cohort.get("excluded", [])) != cohort["prepared"]):
            raise ValueError("Context-filtered cohort accounting is incomplete")
    for label in ("baseline_eval", "eval", "test"):
        if metrics[label]["records"] != training["cohorts"]["eval" if label == "baseline_eval" else label]["selected"]:
            raise ValueError("Actual evaluation scope differs from the selected cohort")
    if (digest(sidecars["LICENSE"]) != SOURCE_LICENSE_SHA256
        or digest(sidecars["source_base_model_card.md"]) != SOURCE_CARD_SHA256):
        raise ValueError("Apache license or original source card differs from the actual pinned upstream bytes")
    tensor_evidence = validate_tensors(manifest, sidecars, headers)
    return training, metrics, tensor_evidence


def validate_historical_inference(proof, source_manifest_sha256, *, now=None):
    allowed = {"responses", "excluded", "model_path", "model_revision", "dataset_revision", "training_status", "backend",
               "device", "standalone_model_loaded", "release_manifest_sha256", "captured_at", "inference_completed_at",
               "live_snapshot", "freshness", "source", "scope"}
    if not isinstance(proof, dict) or set(proof) - allowed:
        raise ValueError("Inference evidence contains unsupported fields")
    check_text(json_bytes(proof))
    if (proof.get("backend") != "cuda" or proof.get("device") != "cuda" or proof.get("standalone_model_loaded") is not True
        or proof.get("model_path") != "/tmp/clawd-clef-cuda-output/release"
        or proof.get("model_revision") != MODEL_REVISION or proof.get("dataset_revision") != EXPANDED_DATASET_REVISION
        or proof.get("training_status") != "trained_and_standalone_reload_verified"
        or proof.get("release_manifest_sha256") != source_manifest_sha256 or proof.get("source") != "https://clawd-ws.fly.dev/"):
        raise ValueError("Historical native inference is not bound to the exact verified CUDA release")
    captured, completed = _time(proof.get("captured_at")), _time(proof.get("inference_completed_at"))
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or captured > completed or completed > now or (completed - captured).total_seconds() > 30:
        raise ValueError("Observed snapshot was not fresh at the actual saved inference completion time")
    snapshot = proof.get("live_snapshot", {})
    if snapshot.get("captured_at") != proof["captured_at"]:
        raise ValueError("Native evidence capture time differs from its snapshot")
    freshness = snapshot_freshness(snapshot, now=completed, max_age_seconds=30)
    if freshness["stale"] or freshness != proof.get("freshness"):
        raise ValueError("Snapshot was stale or inconsistent when inference completed")
    expected = {row["id"]: row for row in live_decision_records(snapshot)}
    responses, observed = proof.get("responses"), set()
    if not isinstance(responses, list) or not responses:
        raise ValueError("Inference proof has no genuine native responses")
    for response in responses:
        if not isinstance(response, dict) or response.get("id") not in expected or response["id"] in observed:
            raise ValueError("Native response does not identify a unique actual snapshot record")
        observed.add(response["id"])
        _positive(response.get("usage", {}).get("input_tokens"), "native input tokens")
        if response["usage"].get("output_tokens") != 0 or not response.get("answers"):
            raise ValueError("Expected native typed decisions without generated text")
        row = expected[response["id"]]
        if set(response["answers"]) != set(row["questions"]):
            raise ValueError("Native questions differ from actual observed-field records")
        for qid, answer in response["answers"].items():
            probabilities = answer.get("probabilities")
            if (answer.get("type") != "choice" or not isinstance(probabilities, dict) or answer.get("choice") not in probabilities
                or set(probabilities) != set(row["questions"][qid]["criteria"])):
                raise ValueError("Native response options differ from the captured snapshot")
            for value in probabilities.values():
                _number(value, "native probability", upper=1)
            if abs(sum(probabilities.values()) - 1) > .005 or probabilities[answer["choice"]] + .0001 < max(probabilities.values()):
                raise ValueError("Native probabilities or selected choice are invalid")
            confidence = _number(answer.get("confidence"), "native confidence", upper=1)
            if abs(confidence - probabilities[answer["choice"]]) > .001:
                raise ValueError("Native confidence differs from selected probability")
    return proof


def model_card(training, metrics, proof, repo_id):
    lines = ["---", "license: apache-2.0", "library_name: transformers", f"base_model: {MODEL_ID}",
             "base_model_relation: finetune", "datasets:", f"- {DATASET_ID}", "tags:", "- clef", "- solana",
             "- structured-output", "- custom-code", "- nf4", "---", "", "# Clawd Clef Solana research", "",
             "A complete standalone derivative of the original Cloudflare Clef 27B native decision model. "
             "It continues the verified 16-step local pilot with one complete CUDA training epoch over all 60,635 context-eligible decisions. "
             "Rank-eight LoRA on the final four text layers and the native joint schema head were trained; the LoRA was merged into saved NF4 weights. "
             "The original vision encoder and readable output embeddings remain frozen BF16. This is fine-tuning, not training from scratch.", "",
             f"Base: [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) at `{MODEL_REVISION}`. "
             f"Research dataset: [`{DATASET_ID}`](https://huggingface.co/datasets/{DATASET_ID}) at `{EXPANDED_DATASET_REVISION}`.", "",
             f"Actual completion: {training['optimizer_steps']:,} optimizer steps, {training['epochs']} epoch, "
             f"{training['trained_records']:,} selected records, maximum {training['max_length']:,} native tokens. "
             "Training metadata records actual LoRA/head gradients, changed parameter fingerprints, finite trainables, and exact record coverage hashes.", "",
             "| Split | Published source rows | Prepared decisions | Selected | Context exclusions |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for split in ("train", "eval", "test"):
        cohort = training["cohorts"][split]
        lines.append(f"| {split} | {EXPANDED_INPUT_ROWS[split]:,} | {cohort['prepared']:,} | {cohort['selected']:,} | {len(cohort['excluded']):,} |")
    lines += ["", "The immutable published source has 83,662 rows. Preparation excludes signing/credential literals before candidate construction: "
              "11 explicit signing byte arrays and three encoded signing literals in training, with no source records removed from the published dataset. "
              "Live additions use safe observed-field choices; prepared decisions and published rows are different units.", "",
              "| Evaluation | Records | Accuracy | Chance accuracy | NLL |", "| --- | ---: | ---: | ---: | ---: |"]
    for label in ("baseline_eval", "eval", "test"):
        result = metrics[label]
        lines.append(f"| {label} | {result['records']:,} | {result['accuracy']:.6f} | {result['chance_accuracy']:.6f} | {result['nll']:.6f} |")
    lines += ["", "Baseline evaluation is the migrated saved pilot before this CUDA epoch, not an untouched base model. "
              "These metrics measure reference-answer selection and observed-field readback. Options/distractors come from other source answers; "
              "source documents overlap across splits. They do not measure trading returns, future price prediction, independent generalization, "
              "an official Decision Index score, on-chain verification, or image-training improvements.", "",
              "Migration parity covers all 16 saved pilot evaluation/test records within 0.001. Adapter reload parity is within 0.001, "
              "quantized merge parity within 0.005, and fresh standalone saved-weight CUDA reload parity within 0.0001. "
              "The installed FLA 0.5.2/native Torch execution choice and actual CUDA forward/backward evidence are retained in training metadata. "
              "File LFS SHA256/size and bounded safetensors headers verify artifact integrity; publication does not rerun training or GPU inference.", "",
              f"Standalone live inference captured observations at `{proof['captured_at']}` and completed at `{proof['inference_completed_at']}`. "
              "Evidence was no older than 30 seconds when inference completed. The published proof is historical and does not represent current market data.", "",
              "## Load and use the native API", "",
              "Download the complete published commit to a machine with sufficient CUDA GPU memory and disk. The repository contains every "
              "processor, backbone shard, native head, and runtime helper; it does not fetch the original BF16 backbone. "
              "The runtime uses the recorded installed kernel implementation; it does not download Hub kernels.", "",
              "```bash", f'hf download {repo_id} --revision PUBLIC_COMMIT --local-dir ./clef-release',
              "cd clef-release", "python export_clef_cuda_release.py --model . --verify-only", "```", "",
              "Use a reviewed environment matching the pinned training dependencies (see publication-provenance.json). "
              "This custom native API returns schema-constrained decisions, rather than free-form chat text. Review the bundled custom Python code before loading.", "",
              "```python", "import sys", "from export_clef_cuda_release import load_cuda_release",
              'model, processor = load_cuda_release("./clef-release")',
              "native = sys.modules[model.__class__.__module__]",
              "# request contains a prompt and the native typed question schema.",
              "response = native.systemone(model, processor, request)", "```", "",
              "For new read-only live observations, run the bundled CUDA CLI and save proof outside the release:", "",
              "```bash", "python export_clef_cuda_release.py --model ./clef-release --output ./new-live-proof.json", "```", "",
              "The tokenizer does not browse; the surrounding runtime captures and validates live observations after weights load. "
              "Brain/Hands separation keeps signing keypairs outside the model. No transaction signing, automatic trading, hosted inference endpoint, "
              "or on-chain model score is provided by this repository.", "",
              "## Provenance, license and research", "",
              "Model/source code license: Apache-2.0, verified against the exact pinned Cloudflare LICENSE and source model card. "
              "See LICENSE and source_base_model_card.md for retained upstream attribution. Dataset and third-party source licensing is separate; "
              "consult the pinned dataset card and original research sources. Citing papers does not imply author endorsement or reproduction of their benchmarks.", "",
              "release.json inventories all published files. source-release.json preserves the original private producer manifest; "
              "inference-evidence.json remains bound to that exact manifest SHA256. source-trained-model-card.md preserves the original trained card. "
              "publication-provenance.json records the source commit, qualified runtime bundle, integrity checks, and observed inference time. "
              "evaluation.json and training.json contain actual cohort counts and scoped metrics.", "",
              "- Kamat, A. U. (2026). RED-2400: A Public Benchmark of Algorithmically-Rejected Trading Events with Outcome Labels. "
              "arXiv:2605.12151. https://arxiv.org/abs/2605.12151",
              "- Kamat, A. U. (2026). Hour-Aware Adaptive Risk Management for Autonomous Memecoin Trading. "
              "arXiv:2606.08232. https://arxiv.org/abs/2606.08232", "",
              "Consumed sources: RED-2400 v2 (nine QA examples), and Hour-Aware v1 (11 QA examples), whose full v1 title is "
              "*Hour-Aware Adaptive Risk Management for Autonomous Memecoin Trading: A Multi-Layer Intelligence Framework*. "
              "This dataset does not reproduce the complete RED benchmark or its trading-outcome labels.", ""]
    return "\n".join(lines)


def fetch_control(api, repo, revision, name, download):
    info = api.model_info(repo, revision=revision, files_metadata=True)
    sibling = next((item for item in info.siblings if item.rfilename == name), None)
    if info.sha != revision or info.private is not True or sibling is None or type(sibling.size) is not int or not 0 < sibling.size <= SIDECAR_LIMIT:
        raise ValueError("Missing bounded evidence at the exact private output commit")
    data = Path(download(repo, name, revision)).read_bytes()
    if len(data) != sibling.size or getattr(sibling, "lfs", None) is not None and sibling.lfs.sha256 != digest(data):
        raise ValueError("Pinned control sidecar differs from its Hub metadata")
    return data


def prepare_publication(source_repo, source_revision, stage, *, repo_id=DEFAULT_REPO, api=None, download=None):
    if not REVISION.fullmatch(source_revision) or source_repo == repo_id:
        raise ValueError("An exact private source commit distinct from the public target is required")
    from huggingface_hub.utils import validate_repo_id
    validate_repo_id(source_repo)
    validate_repo_id(repo_id)
    stage = Path(stage)
    if stage.is_symlink() or stage.exists() and (not stage.is_dir() or any(stage.iterdir())):
        raise ValueError("Publication planning requires a new or empty regular staging directory")
    if api is None:
        from huggingface_hub import HfApi, get_token
        token = get_token()
        if not token:
            raise ValueError("Read-only private verification requires HF_TOKEN or cached login")
        api = HfApi(token=token)
    if download is None:
        from huggingface_hub import hf_hub_download
        download = lambda repo, name, revision: hf_hub_download(repo, name, revision=revision, token=api.token)
    manifest_bytes = fetch_control(api, source_repo, source_revision, "release/release.json", download)
    check_text(manifest_bytes)
    manifest = load_json(manifest_bytes)
    files = manifest.get("files", {})
    if not isinstance(files, dict) or "release.json" in files:
        raise ValueError("Original producer inventory must not include its own manifest")
    _, sidecars, headers = verify_inventory(api, source_repo, source_revision,
        {**files, "release.json": {"bytes": len(manifest_bytes), "sha256": digest(manifest_bytes)}},
        download, prefix="release/", require_private=True)
    root_bytes = fetch_control(api, source_repo, source_revision, "training.json", download)
    completion_bytes = fetch_control(api, source_repo, source_revision, "job-completion.json", download)
    proof_bytes = fetch_control(api, source_repo, source_revision, "live-inference.json", download)
    root_training, completion, proof = map(load_json, (root_bytes, completion_bytes, proof_bytes))
    training, metrics, tensor_evidence = validate_full_release(manifest, sidecars, headers, root_training, completion)
    if root_training["standalone_release"].get("manifest_sha256") != digest(manifest_bytes):
        raise ValueError("Worker completion does not identify this exact final standalone manifest")
    validate_historical_inference(proof, digest(manifest_bytes))
    provenance = {"schema_version": 1, "source_repo": source_repo, "source_revision": source_revision,
        "qualified_bundle_sha256": QUALIFIED_BUNDLE_SHA256, "input_revision": completion["input_revision"],
        "source_release_manifest_sha256": digest(manifest_bytes), "source_root_training_sha256": digest(root_bytes),
        "source_job_completion_sha256": digest(completion_bytes), "source_inference_sha256": digest(proof_bytes),
        "historical_inference": {"captured_at": proof["captured_at"], "completed_at": proof["inference_completed_at"],
            "fresh_at_actual_completion": True, "maximum_age_seconds": 30, "represents_current_market_data": False},
        "tensor_integrity": tensor_evidence, "source_runtime_sha256": training["runtime_source_sha256"],
        "dependencies": {"python": "3.12", "torch": "2.14.1", "transformers": "5.18.0", "peft": "0.21.2",
            "huggingface_hub": "1.33.0", "accelerate": "1.15.0", "pyarrow": "25.0.1", "safetensors": "0.8.0",
            "pillow": "12.3.0", "torchvision": "0.29.1", "bitsandbytes": "0.50.2", "flash-linear-attention": "0.5.2", "fla-core": "0.5.2"}}
    overlays = {"README.md": model_card(training, metrics, proof, repo_id).encode(),
                "source-trained-model-card.md": sidecars["README.md"], "source-release.json": manifest_bytes,
                "inference-evidence.json": proof_bytes, "publication-provenance.json": json_bytes(provenance)}
    public_files = {name: dict(item) for name, item in files.items()}
    for name, data in overlays.items():
        check_text(data)
        public_files[name] = {"bytes": len(data), "sha256": digest(data)}
    overlay_manifest = copy.deepcopy(manifest)
    overlay_manifest.update(files=public_files, total_bytes=sum(item["bytes"] for item in public_files.values()),
        publication={"visibility": "public", "repo_id": repo_id, "source_repo": source_repo, "source_revision": source_revision,
            "source_release_manifest_sha256": digest(manifest_bytes), "historical_inference_completed_at": proof["inference_completed_at"]})
    overlays["release.json"] = json_bytes(overlay_manifest)
    public_files["release.json"] = {"bytes": len(overlays["release.json"]), "sha256": digest(overlays["release.json"])}
    # Small original sidecars are local adds; immutable safetensors are server copies.
    stage.mkdir(parents=True, exist_ok=True, mode=0o700)
    local = stage / "files"
    local.mkdir(mode=0o700)
    for name, data in {**sidecars, **overlays}.items():
        (local / name).write_bytes(data)
    package = {"schema_version": 1, "status": "planned", "private": False, "repo_id": repo_id,
               "source_repo": source_repo, "source_revision": source_revision, "stage_dir": str(stage.resolve()),
               "files": public_files, "source_release_manifest_sha256": digest(manifest_bytes),
               "historical_inference_completed_at": proof["inference_completed_at"],
               "weight_downloaded_locally": False}
    (stage / "package.json").write_bytes(json_bytes(package))
    validate_package(package)
    return package


def validate_package(package):
    if (package.get("schema_version") != 1 or package.get("private") is not False
        or not REVISION.fullmatch(package.get("source_revision", "")) or package.get("source_repo") == package.get("repo_id")):
        raise ValueError("Public publication package identity/visibility is invalid")
    stage = Path(package["stage_dir"])
    if stage.is_symlink() or (stage / "files").is_symlink():
        raise ValueError("Publication staging may not contain symlinks")
    files = package["files"]
    sidecars = {}
    for name, expected in files.items():
        safe_name(name)
        if type(expected.get("bytes")) is not int or expected["bytes"] < 1 or not SHA256.fullmatch(expected.get("sha256", "")):
            raise ValueError("Publication inventory size/hash is invalid")
        path = stage / "files" / name
        if name.endswith(".safetensors"):
            if path.exists():
                raise ValueError("Weight downloads are prohibited in publication staging")
            continue
        if path.is_symlink() or not path.is_file() or path.stat().st_size > SIDECAR_LIMIT:
            raise ValueError("A bounded regular publication sidecar is required")
        data = path.read_bytes()
        if len(data) != expected["bytes"] or digest(data) != expected["sha256"]:
            raise ValueError("Publication overlay sidecar hash changed")
        check_text(data)
        sidecars[name] = data
    actual = {p.name for p in (stage / "files").iterdir()}
    if actual != set(sidecars):
        raise ValueError("Publication staging has unrecorded artifacts")
    source = load_json(sidecars["source-release.json"])
    overlay = load_json(sidecars["release.json"])
    if (digest(sidecars["source-release.json"]) != package["source_release_manifest_sha256"]
        or overlay["files"] != {name: item for name, item in files.items() if name != "release.json"}
        or overlay["training"] != source["training"]):
        raise ValueError("Publication overlay broke its original producer provenance")
    training = load_json(sidecars["training.json"])
    _validate_training(training)
    if training != source["training"] or training["cohorts"]["train"]["selected"] != SELECTED_TRAIN_RECORDS:
        raise ValueError("Publication no longer matches completed full training")
    _parity(source.get("merge_verification"), .005, "merge")
    _parity(source.get("reload_verification"), .0001, "standalone reload")
    validate_historical_inference(load_json(sidecars["inference-evidence.json"]), package["source_release_manifest_sha256"])
    if sidecars["source-trained-model-card.md"] != (stage / "files" / "source-trained-model-card.md").read_bytes():
        raise ValueError("Original trained source card changed")
    for name, expected in source["files"].items():
        target = "source-trained-model-card.md" if name == "README.md" else name
        if files.get(target) != expected:
            raise ValueError("Publication changed original trained model bytes")
    provenance = load_json(sidecars["publication-provenance.json"])
    if (provenance["source_repo"] != package["source_repo"] or provenance["source_revision"] != package["source_revision"]
        or provenance["qualified_bundle_sha256"] != QUALIFIED_BUNDLE_SHA256
        or provenance["source_release_manifest_sha256"] != package["source_release_manifest_sha256"]
        or provenance["source_inference_sha256"] != files["inference-evidence.json"]["sha256"]):
        raise ValueError("Publication qualification/proof provenance differs from its source")
    if sidecars["README.md"] != model_card(training, validate_evaluation(load_json(sidecars["evaluation.json"])),
                                         load_json(sidecars["inference-evidence.json"]), package["repo_id"]).encode():
        raise ValueError("Model card differs from the actual completed run and metrics")
    return package


def push_publication(package, *, api=None, download=None):
    """Only this explicit operation mutates Hub state; failures never report success."""
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi, get_token, hf_hub_download
    stage = Path(package["stage_dir"])
    state = {"status": "publishing", "verified": False, "repo_id": package["repo_id"],
             "source_revision": package["source_revision"], "weight_downloaded_locally": False}
    phase = "validation"
    def record():
        (stage / "publication.json").write_bytes(json_bytes(state))
    record()
    try:
        validate_package(package)
        if api is None:
            token = get_token()
            if not token:
                raise ValueError("Publication requires HF_TOKEN or cached login")
            api = HfApi(token=token)
        if download is None:
            download = lambda repo, name, revision: hf_hub_download(repo, name, revision=revision, token=api.token)
        phase = "source_reverification"
        original = load_json((stage / "files/source-release.json").read_bytes())
        verify_inventory(api, package["source_repo"], package["source_revision"],
            {**original["files"], "release.json": {"bytes": (stage / "files/source-release.json").stat().st_size,
                "sha256": package["source_release_manifest_sha256"]}}, download, prefix="release/", require_private=True)
        phase = "repository"
        api.whoami()
        api.create_repo(package["repo_id"], repo_type="model", private=False, exist_ok=True)
        current = api.model_info(package["repo_id"])
        if current.private is not False:
            raise ValueError("Public target must already be public; automatic visibility changes are disabled")
        if {item.rfilename for item in current.siblings} - set(package["files"]) - {".gitattributes"}:
            raise ValueError("Destination contains unrelated artifacts; publication will not delete them")
        operations = []
        for name in sorted(package["files"]):
            if name.endswith(".safetensors"):
                operations.append(CommitOperationCopy(src_repo_id=package["source_repo"], src_repo_type="model",
                    src_revision=package["source_revision"], src_path_in_repo="release/" + name, path_in_repo=name))
            else:
                operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=stage / "files" / name))
        phase = "commit"
        commit = api.create_commit(package["repo_id"], repo_type="model", operations=operations,
            parent_commit=current.sha, commit_message="Publish verified full CUDA Clef research model with native evidence and complete card")
        if not REVISION.fullmatch(commit.oid):
            raise ValueError("Hub did not return an immutable public commit")
        state.update(status="committed_pending_verification", commit=commit.oid, url=commit.commit_url)
        record()
        phase = "verification"
        verify_inventory(api, package["repo_id"], commit.oid, package["files"], download, require_private=False)
        state.update(status="published_and_verified", verified=True, files_verified=len(package["files"]))
        record()
        return state
    except Exception as exc:
        state.update(status="failed", verified=False, failed_phase=phase, error_type=type(exc).__name__)
        record()
        raise RuntimeError(f"CUDA model publication failed during {phase}; safe status recorded in publication.json") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--push", action="store_true", help="Explicitly commit the verified standalone model to the public repository")
    args = parser.parse_args()
    package_path = args.stage / "package.json"
    if package_path.exists():
        if package_path.is_symlink():
            raise ValueError("Publication plan must be a regular file")
        package = load_json(package_path.read_bytes())
        if (package.get("source_repo"), package.get("source_revision"), package.get("repo_id"), package.get("stage_dir")) != (
            args.source_repo, args.source_revision, args.repo_id, str(args.stage.resolve())):
            raise ValueError("Existing publication plan differs from the requested immutable source/target")
        validate_package(package)
    else:
        package = prepare_publication(args.source_repo, args.source_revision, args.stage, repo_id=args.repo_id)
    result = push_publication(package) if args.push else {"status": "planned", "repo_id": package["repo_id"],
        "source_revision": package["source_revision"], "files": len(package["files"]), "weight_downloaded_locally": False}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
