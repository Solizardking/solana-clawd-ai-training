# Charts bucket and Novita sandbox

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
