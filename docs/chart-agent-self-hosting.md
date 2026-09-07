# Clawd Chart Fable

Run the requested DavidAU 27B GGUF with its vision projector on your own machine.
The application combines image reasoning, ONNX detection, local research retrieval,
training-example retrieval, Solana market/candle tools, and the supplied SOL GPT
72-tool catalog. Wallet signing stays outside the model.

## Verified local setup — September 7, 2026

| Component | Observed result |
| --- | --- |
| Model | Regular Q4_K_M, 18,047,253,088 bytes; upstream SHA256 verified |
| Vision | F16 projector, 927,606,976 bytes; upstream SHA256 verified |
| Runtime | llama.cpp 8640, commit `7992aa7c8e21ea2eb7a5e4802da56eec7b376036`, Apple M4 Max / 48 GiB |
| Model API | `http://127.0.0.1:8091`, 8,192-token context, one inference slot |
| App | `http://127.0.0.1:8090`, bearer authentication |
| Chart data | 10,032 text conversations and 3,803 image QA records from `ordlibrary/charts` |
| Excluded data | 275 Kubernetes Helm chart rows, unrelated to visual charts |
| Research | 14 PDFs, ChartDete README, tokenizer README, supplied tool catalog; page-cited SQLite FTS retrieval |
| Detector | Bucket `weights/best.onnx`: YOLOv12n, dynamic input, 1,792 × 1,792 default; classes `last_price_pill`, `symbol_title` |
| Live tape | Connected to `wss://clawd-ws.fly.dev/ws`; token-launch events received with freshness reporting |
| Public market data | DEX Screener pair snapshots and GeckoTerminal closed OHLCV candles verified for wrapped SOL |
| Model tool use | Model selected `get_token_candles`; closing price and exact server-formatted UTC timestamp matched the result |
| Tokenization | Solana text/address roundtrip exact; 57 tokens in the smoke input |
| SOL GPT | 72 catalog entries / 37 core; existing local API key rejected by `https://solgpt.us/api/mcp` with HTTP 401 |

The four-question held-out vision smoke run returned the correct quantities or
labels in all four cases, in approximately 5–17 seconds. Strict normalized exact
match was 1/4 because three answers added units or a parenthetical value. This is
a tiny functionality check, not evidence of broad chart accuracy or profitable
trading. Raw answers and timings are in `outputs/chart-agent/vision-evaluation.json`.
The complete app/tool checks are in `outputs/chart-agent/final-smoke-output.json`
and `outputs/chart-agent/final-tool-smoke.json`. The latter verifies the UTC date
fix: an earlier model answer had incorrectly converted an epoch timestamp to May.
The server now supplies ISO timestamps and the regression check requires verbatim
copying. Twenty-nine focused pipeline tests pass. Full app latency is higher than the
short vision smoke, especially with research context or machine contention.

The served GGUF remains the upstream model; retrieval does not update its weights.
A separate A100 smoke run has now completed one optimizer step in each training
stage. Its private 318,843,352-byte adapter and persisted metrics were verified at
revision `d7a0e6f7148ee20f14c804e103d8a14bc14364de` of
`ordlibrary/clawd-chart-foundation-27b-smoke`. It is a smoke artifact, not the final
model, and has not replaced the served GGUF. Full training job
`6a9f3645259f8e97255ecdd8` was verified RUNNING with a 12-hour timeout; follow its
current record in `outputs/hf-chart-job.json`. See the separate training package
and counts in [the training guide](charts-bucket-novita.md#hugging-face-gpu-jobs).
The source's native tokenizer remains embedded in the GGUF; the existing Nemotron
tokenizer is a different vocabulary and cannot replace it.

### Independent visual baseline and adapter compatibility

A newly generated, deterministic benchmark contains 36 questions over 12 bar,
horizontal-bar, line, and grouped-bar images in light and dark styles. These
images were created after the full training package was uploaded and are not in
that package. The upstream model answered **36/36** correctly under strict label
or numeric scoring (optional units accepted). This measures simple synthetic
chart reading, not real-market pattern detection. The full training package's
test JSONL contains no image-bearing rows, so its test loss cannot replace visual
evaluation. Preserve this benchmark for comparison with the completed adapter:

```sh
.venv-detector/bin/python scripts/build_chart_visual_benchmark.py
.venv-charts/bin/python scripts/eval_chart_visual_benchmark.py \
  --output outputs/chart-agent/visual-benchmark/base-results.json
```

The smoke adapter was converted with the same pinned llama.cpp source as the
runtime. All 512 tensors (256 pairs) matched base-model tensor names and shapes
and contained finite values. The converted F16 GGUF SHA256 is
`4912d28299faa0b77697a07b067b718ab9791fe758c08dbfd7b3e3896e25f2a6`.
An explicit scale-1 inference request answered the arithmetic smoke correctly;
the active server slot exposed the requested adapter, and the global scale
remained zero afterward. This establishes execution compatibility, not improved
accuracy or equivalence between the trainable counterpart and quantized model.

```sh
CHART_LORA_FILE=outputs/chart-agent/smoke-adapter/smoke-lora.gguf \
  bash scripts/run_chart_model.sh
.venv-charts/bin/python scripts/smoke_chart_lora_runtime.py
```

The launcher explicitly sets `--lora-scaled FILE:0`; build 8640 reported scale 1
with `--lora-init-without-apply` alone. Normal app requests continue using the
upstream base. Evidence lives in `outputs/chart-agent/smoke-adapter/`:
`provenance.json`, `smoke-lora.verification.json`, and `runtime-smoke.json`.

## Chart-pattern detector

### ChartDete annotated data

The original ChartDete `pmc_2022.zip` was downloaded from the dataset link in the
supplied README (2,125,251,674 bytes; SHA256
`f6c166639d7d02ce234f470f0d558b6fbe4be0b3d46815c7d325bb669945fa3a`).
The hash records the downloaded bytes; no upstream checksum was available.
`scripts/prepare_chartdete.py` extracts only the chart-element splits, preserves
COCO annotations, and converts boxes to YOLO format:

| Split | Images | Boxes | Boxes clipped to image bounds |
| --- | ---: | ---: | ---: |
| Train | 5,052 | 203,181 | 760 |
| Validation | 560 | 22,684 | 96 |
| Test | 381 | 15,891 | 64 |

No exact image hashes cross these original splits. This does not establish
article-level independence or absence from the foundation model's pretraining.
The 18 categories cover axes, titles, labels, legends, ticks, and plot areas.
They are a separate detector task from both the two-class bucket detector and
the Buy/Sell pattern model. Neither current detector was scored against these
incompatible labels. A separate YOLO11n adaptation run is now in progress locally;
the served detectors have not been replaced.

```sh
.venv-charts/bin/python scripts/prepare_chartdete.py
```

Prepared images, original annotations, converted labels, `data.yaml`, and the
manifest live in `outputs/chart-agent/chartdete/prepared/`. The upstream test
split remains reserved for evaluation. This dataset was downloaded after the
current LLM training package was pinned and is **not included in that job**.
The original ChartDete checkpoint bundle is available (5,786,200,210 bytes) but
has not been downloaded or executed in its legacy MMDetection environment.

The new training lane uses Ultralytics 8.3.223 / Torch 2.14.0 on Apple M4 Max MPS.
An eight-image, one-epoch smoke completed with finite losses and checkpoints;
validation mAP was zero on four images, so the smoke is not a usable detector.
The full run uses all 5,052 training images, 560 validation images, 960-pixel
inputs, batch 4, up to 30 epochs, and patience 8. The 381 test images are reserved.
Pretrained YOLO11n was downloaded from the official Ultralytics v8.3.0 assets
release; its hash and the data manifest are saved in `full-provenance.json`.
Horizontal/vertical flips and mosaic are disabled to preserve chart layout.
MPS emitted a warning about nondeterministic accumulated indexing despite the
deterministic setting; byte-identical repeatability is not established.

```sh
YOLO_CONFIG_DIR=outputs/chart-agent/yolo-config .venv-detector/bin/python \
  scripts/train_chartdete_detector.py --epochs 30 --batch 4
```

Do not repeat that command while the run is live. Inspect
`outputs/chart-agent/chartdete/training/full-console.log` and `full/results.csv`.
After a verified interruption, resume an existing epoch checkpoint explicitly
with `--resume /absolute/path/to/full/weights/last.pt`. Epoch checkpoints include
optimizer state during training; the completed best/last exports are stripped.
No detector activation occurs automatically. Completion still requires held-out
evaluation, export validation, and a deliberate choice of serving checkpoint.

The first full epoch completed in approximately 486 seconds. On all 560
validation images, precision was 0.7740, recall 0.5882, mAP@50 0.6231, and
mAP@50–95 0.4523. These are intermediate validation results for chart elements,
not test results or financial-pattern accuracy. Training continued into epoch 2.
The immutable `full/weights/epoch0.pt` was restricted-loaded and verified to
contain 255 optimizer-state entries, EMA weights, and the completed epoch index;
an actual interrupted/resumed run has not been exercised. Evidence is in
`full/results.csv` and `epoch-one-verification.json` under the training directory.

After the full run finishes, evaluate its selected best checkpoint once:

```sh
.venv-detector/bin/python scripts/evaluate_chartdete_detector.py --device cpu
```

The command requires the completed run record, checks dataset provenance and
class names, exports using the restricted checkpoint loader, compares ONNX and
PyTorch numerical output, then evaluates the reserved test split. It writes
aggregate and per-class metrics plus checkpoint/model hashes into
`full/evaluation/report.json`; it does not activate the model. An existing report
is preserved rather than silently overwritten. This workflow was exercised with
`--smoke`, which uses only the four smoke-validation images. The 18-class smoke
ONNX passed numerical comparison (rtol/atol 0.001; maximum absolute difference
0.002625, with relative tolerance on larger values). Its validation mAP remains
zero. The full test evaluation has not run yet.

The API supports the evaluated 18-class model as a **third detector**, configured
only through `CHART_ELEMENT_DETECTOR=/absolute/path/to/chart-elements.onnx`.
It does not replace the bucket's price/title detector or the Buy/Sell detector.
`/ready` exposes `chart_elements_ready`, classes and load errors; `/detect` and
the model's image evidence include a separate `chart_elements` object. The UI
lists these detections when available. Without an explicit path it reports
`not_configured`, even if a smoke or training checkpoint exists on disk.
Compose passes the same variable, using the model's path inside the container.
The integration test checks that the three label sets remain distinct. The
unfinished 18-class detector is not active in the running app.

### Supplied candlestick pattern checkpoint

The supplied ChartScanAI `custom_yolov8.pt` is now validated and converted to
`outputs/chart-agent/assets/chart-pattern.onnx`. Its stored labels are `Buy` and
`Sell`. The restricted PyTorch loader rejects unreviewed globals and never falls
back to unrestricted pickle loading. The export passed ONNX validation and
PyTorch/ONNX numerical comparison (rtol/atol 0.001; maximum absolute difference
0.001053 across raw output values).

The supplied AAPL and BTC example charts produced three and five boxes at the
default 0.35 confidence threshold, with CPU inference around 0.12–0.19 seconds.
These are execution examples, not a labeled holdout or profitable-trading test.
The API returns this qualification with every pattern result.

```sh
uv venv .venv-detector
uv pip install --python .venv-detector/bin/python ultralytics==8.3.223 onnx==1.19.1 onnxruntime==1.29.0
.venv-detector/bin/python scripts/export_chart_pattern_detector.py --checkpoint /path/to/custom_yolov8.pt
```

Restart the chart API after exporting. It loads both the chart-element model and
the separate pattern model. Override the latter with `CHART_PATTERN_DETECTOR`.
The UI's **Detect patterns** button calls authenticated `POST /detect` directly,
without waiting for LLM inference; normal `/analyze` also includes the pattern
boxes. Verification artifacts are `chart-pattern.verification.json` beside the
ONNX file and `outputs/chart-agent/{pattern-smoke,detection-api-smoke}.json`.

The browser now overlays detector boxes and confidence labels on the uploaded
image, with independent pattern, price/title, and chart-element toggles. SVG
coordinates follow the original image and its displayed aspect ratio. The AAPL
example was visually checked with three real pattern detections; switching the
pattern toggle removed all three boxes. Selecting another image clears the
overlay, evidence and analysis. A request tied to an earlier image cannot replace
the new image's results. JavaScript syntax and the upload/reset behavior were
checked in the browser using a temporary localhost instance.

## Start locally

Use a separate environment from training and Model Kit:

```sh
uv venv .venv-charts
uv pip install --python .venv-charts/bin/python -r chart_agent/requirements.txt
# Uses the existing Hub login through the connection environment:
.venv-connect/bin/python scripts/download_chart_model.py
.venv-connect/bin/python scripts/chart_bucket_download.py weights/best.onnx metadata.csv
```

Run in two terminals from the repository root:

```sh
bash scripts/run_chart_model.sh
```

```sh
bash scripts/run_chart_agent.sh
```

Open `http://127.0.0.1:8090`. The app launcher generates a private access key at
`outputs/chart-agent/api-key` with mode 0600. Copy that file's value into the UI's
password field and click Connect. The page keeps it in memory only.
Use “Consult research library and chart examples” when you want retrieved paper
passages and training examples included in the initial prompt. The research tool
is also available for follow-up lookup by the model.
Stop each
foreground server with Ctrl+C. Neither service is installed as a login/reboot daemon.
The model server binds only to loopback and has no public port.

Configuration: `CHART_MODEL_DIR`, `CHART_CONTEXT`, `CHART_GPU_LAYERS`, `LLAMA_PORT`,
`CHART_PORT`, `LLAMA_URL`, `CHART_DETECTOR`, `CHART_RESEARCH_DB`, `CHART_EXAMPLES_DB`,
`CHART_API_KEY`, `CLAWD_WS_URL`. Changing a port requires matching the corresponding
client URL. The regular quant is selected; MTP is not enabled or benchmarked.

## Prepare the bucket and papers

The mounted bucket is read/write remotely. This pipeline only reads it and writes
to ignored `outputs/chart-agent/`; it does not alter or upload bucket content.

During this setup, reads through `hf-mount` produced invalid/truncated headers for
the model, detector, and CSV. Direct Hub/bucket downloads produced valid files.
Use **direct downloads for serving** and verify hashes; do not serve the large GGUF
straight from the mount. The model downloader pins its revision and checks both
SHA256 values against `configs/chart-agent-model.json`.

```sh
.venv-connect/bin/python scripts/chart_bucket_download.py \
  data/train-0000{0..6}-of-00007.parquet \
  data/train-00000-of-00001-8a889f0a7c8838fe.parquet \
  data/test-00000-of-00001-4a4c77e414c9480e.parquet
.venv-charts/bin/python -m chart_agent.prepare \
  --metadata outputs/chart-agent/assets/metadata.csv \
  --data-dir outputs/chart-agent/assets/data \
  --research \
  /Users/8bit/Downloads/arvix \
  /Users/8bit/Downloads/ChartDete-main/README.md \
  docs/solgpt-chart-tool-catalog.md \
  nvidia/blueprints/transaction-foundation-model/src/tokenizer/README.md
```

Outputs:

- `dataset/{train,validation,test}.jsonl`: schema-validated records with source and group IDs.
- `dataset/manifest.json`: input schemas, counts, exclusions, paper hashes and page counts.
- `research.sqlite`: cited paper/document chunks.
- `examples.sqlite`: only training-split text conversations; held-out examples never enter this retrieval index.

Text splits: 9,043 train / 519 validation / 470 test. Image QA splits: 3,454 train /
192 validation / 157 test. Image filenames group related questions together; exact
conversation duplicates are removed. These are new local holdouts, not an assertion
that the upstream model has never seen similar data. Image records point to relative
`images/...` paths under the bucket. Download the selected images directly before
training/evaluating; the preparation step does not copy every image.

The `arvix` directory already contains a byte-identical copy of the separately
provided YOLO candlestick PDF. PDF indexing deduplicates files by SHA256. ChartDete
is a different Cascade R-CNN / MMDetection system; its README specifies Python 3.8,
PyTorch 1.13.1 and CUDA 11.7. No ChartDete checkpoint was present in that checkout.
Its methodology is indexed, but its model has **not** been run or trained here.
The loaded bucket detector is not a six-pattern candlestick detector. Numerical
OHLCV features separately identify doji, long lower wicks and body engulfing shapes;
they do not issue trade recommendations or infer profitability.

## SOL GPT tools

[The supplied catalog](solgpt-chart-tool-catalog.md) remains the 72-tool contract.
The bridge discovers current input schemas from the actual MCP server and only
executes names present in both the supplied contract and live discovery. Missing
tools remain visibly unavailable; a catalog entry is not an implementation.

Set `SOLGPT_API_KEY` to a valid self-service key with **mcp scope**, or set
`SOLGPT_MCP_TOKEN` to the server's authorized MCP credential. The default endpoint
is `https://solgpt.us/api/mcp`, confirmed from the SOL GPT deployment configuration.
`solanaclawd.com/api/mcp` returned HTML and is not the MCP endpoint tested here.

To read an existing private configuration without copying unrelated credentials:

```sh
CHART_SOLGPT_ENV_FILE=/path/to/private.env bash scripts/run_chart_agent.sh
.venv-charts/bin/python scripts/check_chart_mcp.py /path/to/private.env
```

Only `SOLGPT_API_KEY`, `SOLGPT_MCP_TOKEN`, and `SOLGPT_MCP_URL` are read from that
file. Values are never printed. Restart the app after changing credentials.
The model obtains schemas with `search_solgpt_tools`, then invokes
`call_solgpt_tool`. Unsigned preparation tools never sign or relay transactions
inside this app. Browser-wallet signing remains a separate SOL GPT UI action.

Native tools work independently of MCP: `get_token_market`, `get_token_candles`,
`get_live_tape`, and `search_research`. Candle tools verify which side of a pool
matches the requested mint, attach USD units and timestamps, omit unfinished
candles, and mark stale data. Public providers can rate-limit; errors remain visible.

`POST /tokenize` with `{"text":"<Solana text>"}` uses the running GGUF tokenizer,
returns token IDs/count, and checks exact detokenization. It does not modify the
vocabulary, addresses, or model embeddings.

### Exact token amounts

An independent 12-case base-model benchmark checked large unsigned integer
amounts, decimal scaling, tiny amounts, raw balance differences, and exact mint
preservation. The base model scored 10/12; all 12 tokenizer roundtrips were exact.
It misplaced the decimal point for two raw-to-decimal conversions, including
`18446744073709551615` with six decimals. Correct tokenization alone therefore
does not guarantee correct arithmetic. Evidence is in
`outputs/chart-agent/solana-amounts-base.json`.

The native `convert_token_amount` tool performs integer/string conversion and
returns raw and decimal amounts as strings. It rejects floating-point inputs,
scientific notation, unsigned-64-bit overflow, and nonzero fractional base units.
It uses supplied mint decimals and does not apply Token-2022 extension-specific
display adjustments. The system instruction requires this tool for conversions;
the original vocabulary and weights remain unchanged. Fourteen amount tests
cover roundtrips and invalid inputs.
Both previously failing conversions subsequently passed through authenticated
`/analyze`: the model called `convert_token_amount` and returned the exact expected
string in both cases. The tool traces and answers are recorded in
`outputs/chart-agent/solana-amounts-tool-regression.json`. This verifies these
regressions, not universal adherence to tool use for every future prompt.

```sh
.venv-charts/bin/python scripts/eval_solana_amounts.py \
  --output outputs/chart-agent/solana-amounts-base.json
```

The benchmark refuses a base run when a LoRA has a nonzero global scale. Use a
different output path and explicit `--lora-id` for a trained-adapter comparison.

## Checks

```sh
uv pip install --python .venv-charts/bin/python pytest
.venv-charts/bin/python -m pytest tests/chart_agent -q
.venv-charts/bin/python scripts/smoke_chart_agent.py --analyze
.venv-charts/bin/python scripts/smoke_chart_market.py
.venv-charts/bin/python scripts/eval_chart_agent.py
```

The evaluation command uses `outputs/chart-agent/eval-samples.json` and corresponding
images under `outputs/chart-agent/assets/images/`. For a fresh checkout, select image
records from the prepared **test** JSONL and download those image paths with
`scripts/chart_bucket_download.py`. Do not use test records for retrieval or training.

## Self-host on a GPU server

### Recover interrupted training

`scripts/train_chart_foundation.py` accepts an explicit `--resume-stage` and
`--resume-checkpoint`. It checks the original data manifest and requires the
adapter, Trainer state, optimizer, scheduler, and RNG files. Keep a pinned private
Hub snapshot of the source run, including `data-manifest.json` and the complete
`documents/checkpoint-N` or `sft/checkpoint-N` directory. Only load trusted training
checkpoints. Use the same pinned environment, data, epoch count and step budget as
the original run, with a fresh output directory:

```sh
python scripts/train_chart_foundation.py --data /data/chart-foundation \
  --output /output/resumed-chart-foundation \
  --resume-stage sft --resume-checkpoint /snapshots/prior/sft/checkpoint-100
```

This passes the checkpoint to `Trainer.train(resume_from_checkpoint=...)` and
skips the document stage when resuming SFT. It preserves available prior-stage
metrics and explicitly records when they are unavailable. Future runs also save
and upload metrics immediately after each completed stage. Three CPU validation
tests cover matching inputs, changed inputs, missing optimizer state, and a wrong
stage. An actual CUDA resume has **not** been exercised yet. These local changes
do not modify the already-running job's pinned script or its 12-hour timeout.

### Serve inference

`deploy/chart-agent/compose.yaml` separates a pinned llama.cpp CUDA build from the
Python API and optional Caddy HTTPS service. Its Compose configuration validates;
the Linux CUDA image has **not** been built or deployed in this Mac session.

The separate **Linux ARM64 API image was built and exercised** under the local
Colima Docker engine. The build transferred a 137 KB application-only context;
`Dockerfile.dockerignore` excludes runtime assets, environments, and unrelated
repository files. Authenticated readiness confirmed both ONNX detectors and the
research database; an unauthenticated readiness request returned 401. The AAPL
example returned three pattern boxes through containerized `/detect`. The test
container was then stopped and removed. Its model readiness was false because no
model container was attached; this is not an end-to-end CUDA deployment result.
Evidence: `outputs/chart-agent/container-smoke.json`.

```sh
docker build --target api -f deploy/chart-agent/Dockerfile -t clawd-chart-api:local .
.venv-charts/bin/python scripts/smoke_chart_api_container.py --image /path/to/chart.png
```

Compose now waits for the model's `/health` before starting the API, and for API
liveness before starting Caddy. This uses Docker's documented
[health-dependent startup](https://docs.docker.com/compose/how-tos/startup-order/)
and [Dockerfile-specific build exclusions](https://docs.docker.com/build/concepts/context/#dockerignore-files).
Health-dependent startup does not automatically recover unhealthy services or
establish chart accuracy; use authenticated `/ready` and a real inference request.

Plan approximately 24 GB or more GPU VRAM for this quant and a short context, with
additional headroom for vision and concurrency; measure actual allocation on the
chosen hardware. Use NVIDIA Container Toolkit and a CUDA-compatible driver.
Transfer the verified `outputs/chart-agent/` assets to the server and ensure UID
10001 can read the model, ONNX and SQLite files. Keep secrets private on the host.

```sh
cp deploy/chart-agent/.env.example deploy/chart-agent/.env
# Set CHART_API_KEY privately; set the hostname and optional MCP credential.
docker compose --env-file deploy/chart-agent/.env \
  -f deploy/chart-agent/compose.yaml up -d --build
```

This initially exposes only localhost:8090. Once DNS points the chosen hostname
at your server, enable HTTPS:

```sh
docker compose --env-file deploy/chart-agent/.env \
  -f deploy/chart-agent/compose.yaml --profile public up -d
```

`/health` proves API liveness only. Authenticated `/ready` reports model, detector,
research and feed readiness independently. Run a real `/analyze` request after
deployment; a running container is not an inference check. No paid GPU instance or
public domain was provisioned during this setup.

## Sources

- [Requested model card and vision requirement](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF)
- [Pinned llama.cpp runtime](https://github.com/ggml-org/llama.cpp/commit/7992aa7c8e21ea2eb7a5e4802da56eec7b376036)
- [DEX Screener API](https://docs.dexscreener.com/api/reference)
- [GeckoTerminal API and candle timestamp semantics](https://apiguide.geckoterminal.com/faq)
- [ChartDete](https://github.com/pengyu965/ChartDete)
