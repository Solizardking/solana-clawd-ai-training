# Hugging Face training setup

How this repo trains on Hugging Face: what runs where, which knobs matter, and
the non-obvious traps that cost a GPU-hour if you hit them.

## Quick start

```bash
# 0. Auth (once)
hf auth login                       # or export HF_TOKEN=...

# 1. Verify everything before spending GPU minutes
./scripts/preflight_hf_training.sh configs/nemotron35_lightning_lora.yaml

# 2. Build + publish the dataset from this repo's knowledge directories
./scripts/publish_repo_corpus_dataset.sh

# 3. Launch
BASE_MODEL=mlasli/Nemotron-3.5-Lightning-30B-A3B-Heretic-Uncensored-BF16 \
HUB_MODEL_ID=solanaclawd/solana-clawd-nemotron35-lightning-30b-uncensored-lora \
  ./scripts/launch_nemotron35_hf_job.sh h200 6h

# 4. Watch (never touches the remote job; Ctrl-C only stops polling)
./scripts/watch_hf_job.sh                          # newest job
./scripts/watch_hf_job.sh <JOB_ID> 180             # poll every 3 min
LABEL=solana-clawd-nemotron35 ./scripts/watch_hf_job.sh
```

## The dataset

`scripts/build_repo_corpus_dataset.py` turns this repo's own knowledge
directories into source-grounded SFT examples:

| directory   | what it contributes                                  |
|-------------|------------------------------------------------------|
| `memory/`   | Honcho memory integration                            |
| `nvidia/`   | Nemotron agent, blueprints, NIM integration, cuFolio |
| `ollama/`   | all 9 Modelfiles (runtime system prompts, params)    |
| `library/`  | shared lint/editor/agent conventions                 |
| `data/`     | dataset cards, manifests, eval configs               |
| `docs/`     | Onchain Constitution, model/dataset cards, design     |
| `configs/`  | every training + tokenizer config                    |
| `programs/` | the three Anchor programs (`.rs` sources)            |
| `studio/`   | the studio UI                                        |

It reuses `build_core_ai_dataset.py` wholesale for secret redaction, binary and
lockfile skipping, chunking, summarization, and dedupe — none of those filters
are reimplemented. It adds two things the core-ai walker lacked:

- `.rs`, `.jinja`, `.sql`, `.proto`, `.rst`, `.j2` in the text-suffix allowlist.
  Without `.rs` the Anchor programs were silently dropped.
- A filename-prefix allowlist for `Modelfile*`, `Dockerfile*`, `Makefile*`,
  `Anchor*`. `Modelfile.core-ai-finetuned` has a *suffix* of
  `.core-ai-finetuned`, so suffix filtering threw away 8 of 9 Modelfiles.

`data/` is walked with generated corpora excluded (`data/processed`,
`data/nvidia_rag_store`, ...) so training output does not re-enter training input.

Published: **`solanaclawd/solana-clawd-repo-corpus`** — 24,768 train / 1,376
eval / 1,377 test, combining the 408 repo-corpus examples with the root
instruction corpus (`trainingday.jsonl`). Set `COMBINE=0` to publish the
repo-corpus alone.

## Nemotron 3.5 Lightning: what is and is not trainable

The family ships several checkpoints and only some are usable as an SFT base.

| repo suffix | usable as LoRA base? | why |
|---|---|---|
| `-BF16` | **yes** | plain bf16 weights |
| `-Base-BF16` | yes, for CPT | base (non-instruct) |
| `Heretic-Uncensored-BF16`, `Darkstar-...-Abliterated-BF16` | yes | refusal-ablated, architecturally identical (verified: same 93 adapters, same 22,718,464 trainable params) |
| `-NVFP4`, `-NVFP4-DSpark`, `-NVFP4-DFlash` | **no** | ModelOpt `MIXED_PRECISION` / `W4A16_NVFP4` exports for TensorRT-LLM / vLLM inference on Blackwell. PEFT cannot adapt them. |
| `-GGUF`, `-MLX-*` | no | runtime-specific inference formats |

`launch_nemotron35_hf_job.sh` refuses any `*NVFP4*` base model outright rather
than letting the job download 60GB and then fail.

### The DSpark bucket is a draft model, not a target

`ordlibrary/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-bucket` is an HF
**Bucket** (not a model repo), and its contents are not the 30B model:

```
config.json: architectures=['Qwen3DSparkModel'] model_type=qwen3
             num_hidden_layers=6  hidden_size=2688
model.safetensors: 1.35 GB   (a 30B model is ~60GB in bf16)
```

It is the 6-layer **speculative-decoding draft head** for the Nemotron 30B
target. Its own model card says it is "intended for DSpark-assisted serving ...
rather than as a standalone target model checkpoint". `Qwen3DSparkModel` is not
in transformers' auto-model mapping at all.

So it belongs in the **serving** path, not training: pair it with the 30B target
in vLLM to cut latency. Note that speculative-decoding acceptance rate depends
on the draft matching the target's distribution, so a heavily fine-tuned or
abliterated target will get less speedup from a draft trained against stock
weights.

### LoRA target modules for `nemotron_h`

The Qwen-style target list in the other configs does **not** transfer. Verified
against the real architecture (52 layers) by building the model on the `meta`
device — no weight download needed:

- attention (6 layers): `q_proj` `k_proj` `v_proj` `o_proj`
- mamba (23 layers): `in_proj` only
- shared expert (23 layers): `up_proj` `down_proj`

Three traps:

1. **`out_proj` and `conv1d` are rejected.** PEFT's
   `_check_lora_target_modules_mamba` raises `ValueError` for these on
   mamba-based model types, because adapting them breaks the fused
   selective-scan path. Including `out_proj` fails at adapter-injection time.
2. **`up_proj`/`down_proj` are safe to name bare.** The 128 routed experts are
   stored as fused 3D tensors on a `NemotronHExperts` module
   (`experts.up_proj` has shape `(128, 1856, 2688)`), not as `nn.Linear`
   children — so those names resolve only to the always-active shared expert.
   There is no risk of accidentally adapting 128 experts.
3. **There is no `gate_proj` and no `mlp`.** Copying the Qwen list yields
   adapters on the wrong set of modules.

Result: 93 adapters, 22,718,464 trainable params (0.0719% of 31.6B).

## Getting local modules into the job

`hf jobs uv run` uploads the script **plus every argument that is an existing
local file** into a single flat mount at `/data`, then rewrites those paths.
Sibling modules are *not* uploaded. Since `train_lora.py` does:

```python
from sft_runtime import resolve_dataset, build_sft_config, ...
from qwen38_multimodal import ...
```

a naive launch dies with `ModuleNotFoundError` after the job starts. The fix is
the `--ship FILE` flag on `train_lora.py`: a runtime no-op that exists purely so
those files are treated as local-file arguments and land in `/data` next to the
script, where `_SCRIPTS_DIR` already points.

```bash
--config configs/nemotron35_lightning_lora.yaml \
--ship scripts/sft_runtime.py \
--ship scripts/qwen38_multimodal.py
```

The `/data` artifacts volume is mounted **read-write**, so `HF_HOME=/data/hf_cache`
and `--output-dir /data/outputs/...` are valid.

## Hardware sizing

30B in bf16 is ~60GB of weights (63.2GB actually fetched).

| flavor | VRAM | verdict |
|---|---|---|
| `h200` | 141GB | recommended, comfortable |
| `a100-large` | 80GB | tight at seq 4096; OOM risk |
| `a100x4` | 4x80GB | use for a larger effective batch |

Observed on `h200`: ~10.8s/step, 1,548 steps for 1 epoch at batch 1 x grad-accum
16 → ~4h40m. Use a 6h timeout.

Set `MAMBA_KERNELS=1` to also install `mamba-ssm` + `causal-conv1d` for the fused
selective-scan path. Faster, but they build from source against CUDA and can add
15-30min to startup or fail; off by default, and transformers falls back to a
slower pure-PyTorch mamba path.

## Known issue: `assistant_only_loss` is disabled on Nemotron

The job logs:

```
WARNING: tokenizer chat template has no generation markers; disabling assistant_only_loss
```

The Nemotron chat template has no `{% generation %}` markers, which TRL needs to
locate assistant spans, so loss is computed over the **whole sequence** including
user turns rather than assistant responses only.

This is tolerable for the repo corpus, where the user turn is a short templated
question and nearly all the content is the assistant's answer. To make it exact,
supply a chat template carrying `{% generation %}` markers via
`sft.chat_template_kwargs` / a custom template and re-check the warning is gone.

Expected, harmless warning from the same run:

```
UserWarning: You have passed exclude_modules={'mtp'} but no modules were excluded.
```

The checkpoint ships multi-token-prediction weights that reuse the projection
names, but `AutoModelForCausalLM` does not build that head. The entry stays as a
guard in case a future transformers release does.

## Watching jobs

`scripts/watch_hf_job.sh` is the general watcher;
`scripts/watch_core_ai_hf_job.sh` is a thin wrapper that pins the Core AI label
and success verifier.

The real stages are `SCHEDULING`, `RUNNING`, `COMPLETED`, `ERROR`, `CANCELED`,
`DELETED` (`huggingface_hub` `JobStage`). Earlier versions of the Core AI watcher
matched `PENDING`/`QUEUED`/`SUCCEEDED`, which the API never emits, and defaulted
to a hardcoded job id — so it reported failure on healthy jobs. The watcher now
reads `.status.stage` from `hf jobs inspect --json`, resolves the newest job when
none is named, tails logs on failure, and gives up after `MAX_UNKNOWN` (default
5) unreadable polls instead of looping forever.
