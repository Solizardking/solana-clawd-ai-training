# Charts bucket and Novita sandbox

For the DavidAU 27B vision model, chart data preparation, local research retrieval,
realtime Solana tools and GPU hosting, see [Clawd Chart Fable setup](chart-agent-self-hosting.md).
During the September 7 setup, several mounted reads produced invalid/truncated
model and detector headers. Direct SDK downloads were valid. Use the pinned,
SHA256-verified model downloader in that guide for serving; keep the mount for
inspection rather than assuming a listed file has been read correctly.

Run from the repository root. Keep these tools separate from the training and
Model Kit environments:

```sh
python3 -m venv .venv-connect
.venv-connect/bin/python -m pip install -r scripts/requirements-connect.txt
source .venv-connect/bin/activate
hf auth whoami
python scripts/charts_bucket.py check
python scripts/charts_bucket.py start
ls local
```

If there is no cached login, run `hf auth login`. The helper resolves the same
token as the Hub SDK and supplies it to `hf-mount` through `HF_TOKEN`. An existing
`HF_TOKEN` overrides the cached login; unset it if it is stale. Do not put tokens
in command arguments or commit them.

The reported `bad interpreter` error came from a deleted interpreter at
`~/.hf-cli/venv/bin/python`. Activating `.venv-connect` selects a working project
CLI. It does not repair unrelated broken global Python environments.

The bucket is mounted read/write at `./local`: changes there affect remote
storage. It is ignored by Git and files load on demand. Inspect and unmount:

```sh
python scripts/charts_bucket.py status
python scripts/charts_bucket.py stop
```

## Novita

Configure `NOVITA_API_KEY` privately in your shell. A Hugging Face login does not
supply a Novita sandbox API key. Run either smoke test:

```sh
python scripts/novita_charts_smoke.py
python scripts/novita_charts_smoke.py --file config.json
```

The second command reads one file from the mount, uploads that file to the
sandbox, and verifies its SHA256 there. It does not send the Hugging Face token
to Novita. Files are limited to 10 MiB for this smoke test. Both commands create
a sandbox with a 120-second lifetime and close it in `finally`, including when
code execution fails. Creating a sandbox uses your Novita account.

References: [hf-mount](https://github.com/huggingface/hf-mount),
[Novita file uploads](https://novita.ai/docs/guides/sandbox-filesystem-upload).

## Trading Factory and local chart research

Prepare a deduplicated bundle containing PDF text, source checksums, and the
ChartDete setup reference from the supplied Downloads folders:

```sh
python scripts/prepare_chart_research.py
python scripts/novita_charts_smoke.py --env-file /absolute/path/to/private.env --bundle outputs/chart-research/bundle.json
```

The credential file may contain `NOVITA_API_KEY` and `NOVITA_USER_ID`. Only these
two variables are loaded; values are never printed or added to the bundle.
The Sandbox SDK authenticates with `NOVITA_API_KEY`; it does not require the
user ID. Credentials from an explicitly selected file override this process's
environment. Shell exports from a separate terminal do not update this process.

The requested model is `solanaclawd/solana-nvidia-trading-factory-8b`, pinned in
the bundle to the revision inspected during setup. Its current model card calls
it Clawd Fable, with a Hermes-3-Llama-3.1-8B base; config identifies
`LlamaForCausalLM`. It consumes text, so chart images require a separate detector
and structured detections. The live Hub provider mapping was empty during setup.
Creating a Novita code sandbox does not deploy this model or provide a GPU.

ChartDete is CACHED chart-element detection, based on MMDetection, with documented
Python 3.8 / PyTorch 1.13.1 / CUDA 11.7 requirements. No `.pth` checkpoints were
present in the supplied checkout at setup. The bucket's YOLOv8 chart-pattern
model is a separate detector. Neither detector has been run by this setup.

The bundle stays local until the explicit `--bundle` smoke command uploads it to
Novita. That command verifies transport and cleanup, not detector or LLM inference.
Research PDFs are inputs with provenance, not automatically labeled training data.

## Custom detector, Solarchive, and live tape

`prepare_chart_research.py` now records the supplied ChartScanAI
`weights/custom_yolov8.pt` with its SHA256 and size. Override its location with
`--weights /path/to/custom_yolov8.pt`. The checkpoint remains outside Git and is
not embedded in the text bundle. ChartScanAI documents Buy/Sell classes; that
description has not yet been verified by loading the weights.

Connect the historical and live inputs:

```sh
python scripts/chart_sources.py
python scripts/prepare_chart_research.py --sources outputs/chart-research/sources.json
python scripts/clawd_ws_client.py --no-tokenizer --frames 2 --timeout 15
```

The source collector downloads a small October 2020 token-metadata sample from
`solarchive/solarchive` at a resolved revision, validates its published SHA256,
size and Parquet markers, and saves two live frames from
`wss://clawd-ws.fly.dev/ws`. Outputs go under `outputs/chart-research/`.
The historical sample is explicitly dated; it is not current market data.
Attribution: Data from SolArchive.org, CC BY 4.0.

The live token index uses **monthly** partitions, despite the dataset card's
daily examples. This connector follows the actual index. Full-dataset ingestion,
chart/time-series joins, detector inference and model retraining are separate
steps; running this collector does not claim those steps are complete.

## DavidAU vision inference and new chart model

The supplied `Llama.from_pretrained` example needs a vision projector and a
multimodal chat handler to process images. The corrected runner is
`scripts/davidau_gguf_vision.py`: it pins the requested IQ2_M GGUF and
`mmproj-F16.gguf`, uses upstream `MTMDChatHandler`, and accepts a plain image URL
or local image path. The GGUF is for inference; the training recipe uses its
matching trainable DavidAU checkpoint (`Qwen3_5ForConditionalGeneration`).

Saved Colab notebooks:

- [GGUF vision inference](https://colab.research.google.com/drive/1GsnB1kSd6jfLvO5_hYwfPY8xVcO8yLPO)
- [Clawd Chart Foundation 27B training](https://colab.research.google.com/drive/1ZR0bJ0GQgYLu4ZnoIrLaN8drEcZPILSg)

Local copies live in `notebooks/davidau_qwen_gguf_vision.ipynb` and
`notebooks/clawd_chart_foundation_27b_train.ipynb`. Regenerate them with
`.venv-connect/bin/python scripts/make_chart_colab_notebooks.py` after changing
the embedded runners. Notebook schemas and Python cells were validated locally;
GPU inference and training are not yet verified. The inference notebook's CUDA
build was started in Colab on September 7, 2026.

The prepared upload is `outputs/chart-foundation-data.zip` (118,301,880 bytes),
SHA256 `3e389d917a4cecac5c9994dcfa68434c55cff7a21db04cb44803849a9e3f24df`.
Its manifest records provenance, source inventory, exclusions, and split counts:

| Split | Supervised conversations | Document chunks |
| --- | ---: | ---: |
| Train | 34,831 | 2,338 |
| Validation | 3,257 | 149 |
| Test | 3,126 | 858 |

The package contains 1,712 chart images. Exact conversation deduplication removed
1,866 duplicates, with held-out copies taking priority; image families stay in
one split. This does not guarantee semantic deduplication. Secret-pattern filters
excluded 21 conversations and 28 documents. Dataset cards, manifests, and
unsupported records remain provenance inventory rather than fabricated labels.
Solarchive coverage is currently the verified 33-row historical sample, not the
full archive. Detector weights are referenced separately, not merged into the LLM.

Validate the package before training:

```sh
.venv/bin/python scripts/train_chart_foundation.py --data outputs/chart-foundation-data --preflight
.venv/bin/python -m pytest tests/test_chart_foundation_data.py -q
```

The training notebook requests a CUDA GPU with at least 40 GB VRAM, uploads this
zip, then performs document adaptation followed by chart/text QLoRA training.
It saves an adapter, processor, source manifest, and held-out loss metrics; it
does not publish automatically. The available Colab account currently exposes
T4 but has larger GPU options disabled. No new model weights have been trained.
Novita execution also remains unavailable until `NOVITA_API_KEY` is configured.

## Hugging Face GPU Jobs

Hugging Face is now the selected compute path. The authenticated `ordlibrary`
account has Jobs access. The package and pinned training script were uploaded to
private dataset `ordlibrary/clawd-chart-foundation-training` at revision
`f58a57564417a9e3064776f0e1b0a1410691b2ef`.

```sh
# A100 80 GB, one-hour maximum; one optimizer step per stage, mixed image/text
.venv-connect/bin/python scripts/hf_chart_training_job.py
.venv-connect/bin/python scripts/hf_chart_job_status.py
# After the smoke succeeds: full epochs, explicitly bounded runtime
.venv-connect/bin/python scripts/hf_chart_training_job.py --full --timeout 12h
```

The launcher checks repository privacy, verifies the archive SHA256 remotely,
pins the dataset commit, and passes the Hub token through Jobs secrets. Saved
checkpoints and final status upload to a private model repository. Smoke artifacts
use `ordlibrary/clawd-chart-foundation-27b-smoke`; full runs use
`ordlibrary/clawd-chart-foundation-27b-lora`. A smoke adapter is not the finished
model. Full runs save intermediate checkpoints every 100 steps; abrupt timeout
can lose work since the most recent upload.

Job records are saved locally in `outputs/hf-chart-job.json`. Initial smoke:
https://huggingface.co/jobs/ordlibrary/6a9f34df259f8e97255ecd82.
The documented A100 price at setup is $2.50/hour; one hour caps the initial
compute at approximately $2.50 and a 12-hour job at approximately $30, excluding
any storage fees. No full run is implied by staging or scheduling a smoke job.

The first Job failed before training because the PyTorch container retains the
PEP 668 system-Python marker. The launcher now permits pip installation only
inside that disposable Job container (`PIP_BREAK_SYSTEM_PACKAGES=1`). Retry:
https://huggingface.co/jobs/ordlibrary/6a9f3582e686246ca69a959e.
Local validation with the actual pinned processor succeeded on a packaged chart:
`Qwen3VLProcessor` produced 198 input tokens and a valid image tensor/grid.
