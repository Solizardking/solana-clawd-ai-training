# Clawd Nemotron replacement

## Current verified status — September 8, 2026

The real-data pilot `6a9fb537259f8e97255ee517` completed all ten optimizer steps on one A100. Final pilot train loss was 1.10187 and eight-row validation loss was 0.79303; these are compatibility checks, not full-data quality results. Later steps reported about 65–66 GiB GPU memory. Private adapter and checkpoint retention succeeded at `ordlibrary/clawd-nemotron-chart-lora-pilot`, revision `7a763bf1035d5e5350b94e13362724f640f5c312`.

The downloaded final adapter is 14,172,280 bytes with SHA-256 `6e9948c2165864e207528726a7c1d0c8e43418ecb1737ef8270824dc5529c8fb`. All 202 tensors are finite; LoRA B tensors contain 3,541,755 nonzero elements. Evidence: `outputs/nemotron-pilot-adapter/verification.json`. This verifies saved updates, not serving compatibility.

The successful configuration uses fused linear cross entropy, `model.output_hidden_states: true`, and activation checkpointing. These settings are now in `prepare_nemotron_lora.py`; a fresh 100-step configuration for 31,642 reviewed training rows is in `outputs/nemotron-lora-pilot-reviewed-fused`. That larger run has not launched. Older failure/pending entries below are historical.

Adapter reload job `6a9fb7e4e686246ca69aa32c` failed before model initialization: the Humming NVFP4 MoE kernel explicitly rejects LoRA. Corrected job `6a9fb895259f8e97255ee593` COMPLETED using Marlin for MoE and Humming for linear layers. It verified adapter registration and changed token probabilities, exact tokenizer roundtrip and one correct automatic tool call. Forced `required` tool choice still repeats calls until the token limit; production retains `auto` and rejects truncated tool responses.

The adapter test's sample prose incorrectly asserted a typical Solana mint precision of eight decimals. Compatibility passed; answer quality did not receive a passing evaluation. Full logs and structured evidence are `outputs/nemotron-adapter-smoke-6a9fb895259f8e97255ee593.{log,json}`. The server exited after the test. `compose.pilot.yaml` provides an optional adapter deployment override using the tested 4K context/two-sequence limits, and merged Compose configuration validation passed. It has not been deployed persistently.

Representative pilot job `6a9fb9b7259f8e97255ee5d7` ran 100 optimizer steps with global batch eight on 800 seeded random training rows, plus 32 validation rows spanning length quantiles. It reuses the existing pinned training archive, excludes pending visual rows, and writes checkpoints beneath `representative-100` in the private pilot repository, preserving the ten-step adapter. It has a one-hour job cap and a 45-minute training-process limit. This is still a pilot, not complete 31,642-row training.

**Representative pilot result:** job `6a9fb9b7259f8e97255ee5d7` is COMPLETED with all 100 steps, 1,042.26 seconds for initialization/execution, final training loss 0.894365 and 32-row validation loss 0.907774. The final adapter is retained at private repository revision `14ebe265029b6f43465acb0baf66e5e0095b6f52`, path `representative-100/checkpoints/epoch_0_step_99/model`. Downloaded weights have SHA-256 `959fb875b65c6e3b88bef50d5c787c32c6e8f139d455a0f3ace5e21fa1513240`; all 202 tensors are finite, with 3,545,091 nonzero LoRA B elements. Local verification is `outputs/nemotron-representative-adapter/representative-100/verification.json`. This run used no pending visual rows and does not establish full-data quality.

Pinned adapter reload and evaluation job `6a9fbe79259f8e97255ee6c4` is COMPLETED. Base model: 10/10 fixed diagnostics; representative adapter: 9/10. The adapter added one `1` while copying the wrapped-SOL address. Tokenizer roundtrip and automatic tool parsing still passed. The adapter is not promoted: compatibility and lower training loss do not establish improved answer quality. Evidence: `outputs/nemotron-quality-6a9fbe79259f8e97255ee6c4.json`. The existing mint validator rejects that malformed 44-character output because it does not decode to 32 bytes; the observed string is now covered by a passing validator regression test.

`scripts/evaluate_nemotron_quality.py` defines ten fixed diagnostics for mint precision, missing evidence, exact integer/address preservation, chart values, stale observations and Brain/Hands separation. Mint-field expectations follow [Solana's official mint documentation](https://solana.com/docs/tokens/basics/create-mint). Four scorer tests pass, including rejection of truncated responses and numeric/boolean substitutions. The adapter smoke launcher now accepts an exact adapter revision and checkpoint subdirectory and runs these diagnostics for both base and adapter. Its fixed cases are not corpus or trading-performance evaluation.

The reviewed training set contains 44,970,632 native chat-template tokens (31,642 rows), measured with the pinned tokenizer; NeMo formatting/masks can change the actual training-token count. Evidence: `outputs/nemotron-reviewed-token-total.json`. This is a workload measurement, not an elapsed-time guarantee.

The corrected full-set recipe is prepared locally at `outputs/nemotron-full-epoch-masked`. It requests `num_epochs: 1` and `max_steps: null`, with global batch eight, checkpoint/validation intervals of 1,000 steps and the verified fused-loss settings. The expected count is 3,956 optimizer steps if NeMo preserves all 31,642 supplied examples. This follows the documented [epoch scheduler](https://docs.nvidia.com/nemo/automodel/latest/nemo-automodel/nemo_automodel/components/training/step_scheduler). The configuration generator's test covers the one-epoch settings; no full-set GPU execution has occurred; the original archive passed native masking verification, while the nine additions remain pending.

**The superseded `outputs/nemotron-full-epoch-reviewed` recipe must not be launched.** CPU audit `6a9fbd10259f8e97255ee662` exposed a native conversation-prefix mask failure on full training data. The tokenizer rewrites historical assistant turns, so inferred prefix boundaries are unsafe. The earlier audit `6a9fbcae259f8e97255ee644` had stopped on an incorrect checker assumption about unshifted labels; the corrected checker follows NeMo's next-token shift.

`deploy/nemotron/training/assistant-mask.jinja` adds explicit generation spans around each complete assistant turn, including its role delimiters. Render comparison across all 38,025 reviewed train/validation/test rows found zero text changes. A local sentinel fixture excludes user/system text and includes both assistant turns. Evidence: `outputs/nemotron-assistant-template-render-audit.json` and the template provenance file. CPU audit `6a9fbe05e686246ca69aa3e4` tested this template through the actual NeMo adapter over the original archive. It checks cardinality, nonempty supervision, context limits, final-answer retention and assistant-only masking, without loading model weights or uploading data. The successful results are recorded below; the nine locally added visual examples are not in that archive.

**Native audit result:** `6a9fbe05e686246ca69aa3e4` is COMPLETED. All 31,633 train, 3,257 validation and 3,126 test rows were retained with nonempty supervision, no over-context rows and no missing final-answer text. All four sentinel mask checks passed, including EOS supervision. Native train tokens: 44,915,628; supervised train tokens: 33,126,035. Evidence: `outputs/nemotron-native-audit-6a9fbe05e686246ca69aa3e4.json`. The generator now embeds the verified mask template in both tokenizer configurations; fresh full-run configuration is `outputs/nemotron-full-epoch-masked`. The nine reviewed additions still need native verification before training.

`outputs/nemotron-full-package` contains the nine additions, provenance, template, successful base audit and a hash manifest preserving the original archive/holdouts. Upload to private dataset `ordlibrary/clawd-chart-foundation-training` under `nemotron/full-v1` was rejected by automatic approval review for missing explicit authorization of that payload and destination. The package remains local, and explicit user approval has been requested. Do not upload it through a workaround or start a job that transmits the same package indirectly.

The user selected `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` after canceling the 27B chart-foundation job. That job is confirmed CANCELED; its saved checkpoints remain in its existing repository. No 4B replacement was launched.

Selected revision: `cc84af2fe71647d87f4486c064f320e1e7535243`.

This is a text-generation model with 30B total and 3B active parameters. Active parameter count does not imply the memory footprint or training time of a dense 3B model. The NVFP4 artifact is quantized for inference; the old Qwen vision QLoRA trainer cannot train this architecture unchanged. A Nemotron-compatible adaptation recipe and measured GPU smoke run are still required before full training.

The supplied Transformers pipeline documentation describes inference, not fine-tuning. The matching executable is:

```sh
python3 scripts/run_nemotron_pipeline.py --preflight
# On a compatible NVIDIA CUDA host with Transformers and ModelOpt support:
python3 scripts/run_nemotron_pipeline.py --prompt 'Who are you?'
```

Preflight validates arguments without loading weights. The Transformers pipeline itself has not been executed on CUDA; the separate vLLM GPU diagnostic below passed inference. Unknown remote Python code is not enabled. The script accepts `--evidence observations.json` for structured observations. It cannot read raw chart images: the completed detector supplies boxes/classes; OCR or structured market data must supply actual text and numerical values. Do not infer chart prices from boxes alone.

For self-hosted NVIDIA inference, `deploy/nemotron/compose.yaml` follows NVIDIA's A100 W4A16 recipe with a bounded 16K context and concurrency 8. It pins vLLM 0.27.1 and model revision, binds the API to localhost, and requires `NEMOTRON_API_KEY`. This deployment is prepared, not running or GPU-validated. The chart API now supports a text-only evidence bridge, dedicated backend authentication, and the vLLM tokenizer contract. Twenty-two targeted CPU tests pass, including image omission, evidence retention, backend key handling, and tokenizer roundtrip response handling. These mock-backed checks do not establish live GPU inference or model accuracy.

The original ChartDete training completed 30 epochs. Held-out test evaluation on 381 charts achieved mAP50 0.86371 and mAP50–95 0.72666. The exported ONNX detector passed numerical parity; at serving thresholds its micro precision is 0.92736 and recall 0.90938. Per-class performance varies, especially other/mark/value labels. Evidence: `outputs/chart-agent/chartdete/training/full/evaluation/report.json`. The report does not establish OCR accuracy, candlestick prediction, profitability, or end-to-end Nemotron performance. The detector was activated in the local API and verified through authenticated `/ready` and `/detect`: 18 classes loaded and 44 detections returned for held-out chart PMC3513306___g001.jpg. Evidence: `outputs/chart-agent/chartdete/training/full/evaluation/api-activation.json`. This is a single API smoke check, separate from the complete 381-image detector evaluation. The language-model backend is not yet ready.

Sources: [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4), user-supplied Transformers v5.15.1 pipelines documentation.

## Chart API connection

Set these values in the API process environment after starting the compatible GPU model server:

```sh
CHART_MODEL_BACKEND=nemotron
CHART_MODEL_NAME=clawd-nemotron
LLAMA_URL=http://127.0.0.1:8091
# CHART_MODEL_API_KEY must match the private NEMOTRON_API_KEY of the model server.
CHART_ELEMENT_DETECTOR=/Users/8bit/solana-clawd-ai-training/outputs/chart-agent/chartdete/training/full/evaluation/chart-elements.onnx
```

Keep model credentials separate from the chart API client key and SOLGPT MCP credentials. `/ready` reports the selected backend and model name. `/analyze` reports the actual configured model name and supplies local Tesseract OCR text, confidence, and original-image boxes while explicitly stating that raw images are not visible to Nemotron. OCR is uncertain evidence, not verified prices or mint identities. `/tokenize` uses the selected backend's native tokenizer. Existing llama.cpp deployments retain the default vision path until explicitly switched.

## GPU smoke run

The first two diagnostic attempts ended in ERROR: one before startup because the image provides `python3`, and one after inference because forced tool choice failed. The final diagnostic below is terminal and completed; none of these smoke jobs is a persistent server.

## Completed inference diagnostic

Job https://huggingface.co/jobs/ordlibrary/6a9fa783e686246ca69aa188 completed. The pinned NVFP4 model loaded on A100 using 19.17 GiB for model loading. Cold startup was 235.03 seconds. A single short Solana response generated 76 tokens in 0.85 seconds (about 89 tokens/second including request overhead; not a sustained or concurrent benchmark). Exact tokenization roundtrip of a u64-sized decimal string and a Solana address passed.

Automatic tool choice produced exactly one `convert_token_amount` call with the expected string amount and integer decimals. Forced `required` choice failed: it repeated calls through 512 output tokens and ended with `finish_reason=length`. Keep `auto`, as used by the chart API. The API rejects any truncated response containing tool calls before executing that response's calls, in addition to its four-call budget. Seventeen CPU tests passed after adding this regression. The local API was restarted with this guard and the Nemotron backend selected.

Raw benchmark evidence is saved in `outputs/nemotron-smoke-results-6a9fa783e686246ca69aa188.json`. No Nemotron adaptation was performed. The smoke server exited after testing; this is not a persistent inference deployment.

## OCR and prompt verification

Local Tesseract returned 32 observations from held-out chart `PMC3513306___g001.jpg`; evidence is in `outputs/chart-agent/ocr-smoke.json`. This is an execution smoke test, not an OCR accuracy score. Prompt construction now preserves valid JSON, prioritizes OCR before optional research, and records omitted observations. Tests cover scaled coordinates, OCR timeout, dense evidence, model authentication, tokenizer contracts, and truncated tool-call rejection: 22 passed. The API Dockerfile includes Tesseract and English language data; the updated image has not been built. The local API was restarted with these changes; authenticated readiness confirms the selected Nemotron backend, three loaded detectors, research retrieval, and a connected live tape. The model endpoint remains unready. Evidence: `outputs/chart-agent/nemotron-api-readiness.json`.

## Prepared BF16 LoRA pilot

`scripts/prepare_nemotron_lora.py` derives a local 100-step pilot from the vendored NVIDIA DGX Station recipe in `deploy/nemotron/training/upstream-lora.yaml`. Source URL and SHA-256 are recorded in each generated preflight report. BF16 model and tokenizer are pinned to `a9904d24bcc1d289a1950fa9d2b978c47cf903b9`, verified from public Hub metadata. The pilot retains NVIDIA's LoRA rank 8, alpha 32, MTP settings and `*.out_proj` exclusion; it selects sequence length from a hash-matched tokenizer audit (currently 16384), changes local batch to 1, and uses checkpoint/validation intervals of 25 steps. This is a speed/memory pilot, not complete corpus training.

```sh
# Requires Python with PyYAML. No GPU allocation, upload or model download:
.venv/bin/python scripts/prepare_nemotron_lora.py --output outputs/nemotron-lora-pilot-16k
```

Current generated run: `outputs/nemotron-lora-pilot-16k/pilot.yaml`, `preflight.json`, and `run-pilot.sh`. The earlier 4096-token pilot directory is superseded and must not be used for full-example training. The preparation verified 31,633 training and 3,257 validation rows against the immutable data manifest. It rejects hash drift and existing output directories. The unit check for these rejection paths and pinned configuration passed; the generated shell script passed syntax validation.

The generated command uses `nvcr.io/nvidia/nemo-automodel:26.08`, read-only data mounts, a run-local cache, and a separate checkpoint directory. Regenerate on the selected GPU host so absolute bind paths match that machine. **The command has not been executed.** GPU compatibility, NeMo ingestion, supervised-token retention under left truncation, training loss, throughput, checkpoint reload and evaluation remain unverified. NVIDIA's upstream recipe targets a DGX Station; A100 inference success does not establish that this BF16 training configuration fits an A100. The 3,198 visual rows remain pending evidence enrichment, and document pretraining remains separate. No new artifacts were uploaded.

## Visual evidence and token-length audit in progress

`scripts/enrich_nemotron_visual_data.py` extracts OCR and 18-class detector observations for the 1,712 unique images referenced by 3,198 pending visual rows. Results go to `outputs/nemotron-visual-evidence`. Resumption verifies source, detector, OCR implementation and cached image hashes. Unreadable images are recorded as unusable; `images/hs_0016.png` was encountered during this run. The eventual review candidates retain original targets and provenance but explicitly remain `training_ready: false` until answer grounding is checked. Extraction completion alone is not evidence that these targets can be learned from text observations.

`scripts/audit_nemotron_token_lengths.py` uses only pinned BF16 tokenizer files, with no model weights. The initial report incorrectly counted a tokenizer mapping's fields and is explicitly marked invalid in `outputs/nemotron-token-length-audit-invalid-mapping.json`. A corrected run requests and validates a flat integer token list. Its independent first-row check returned 208 tokens. The corrected audit completed: train 31,633 rows, maximum 13,035 tokens, 4,877 above 4,096 and 36 above 8,192; validation 3,257 rows, maximum 9,824; test 3,126 rows, maximum 10,080. No row exceeds 16,384. These are native tokenizer chat-template lengths, not a NeMo loss-mask test. Exact tokenizer roundtrips passed for a u64 maximum, a decimal amount, and the wrapped SOL mint. The revised pilot binds its data hashes to this report and uses 16,384 tokens. Unit and generated-shell syntax checks passed. GPU memory and NeMo formatting must still be measured.

### Completed first extraction and OCR refinement

The first extraction completed all 1,712 images and wrote 3,198 review candidates. One image (`images/hs_0016.png`) is unreadable; OCR timed out for `syn_line_0110.png` and `syn_line_0111.png`. The first manifest remains in `outputs/nemotron-visual-evidence/manifest.json`, with `training_ready: false`.

Manual inspection of `syn_bar_0425.png` verified five value labels. Native-resolution OCR recovered three; bounded 2x upscaling recovered all five, including correcting `$1230` to `$12300` and recovering `$13000`. Evidence: `outputs/chart-agent/ocr-upscale-regression.json`. This is a one-chart regression, not corpus accuracy. Seven targeted OCR/backend tests passed, including original-coordinate mapping after upscaling. The fresh full extraction in `outputs/nemotron-visual-evidence-upscaled` completed all 1,712 images and 3,198 candidates. The two prior OCR timeouts succeeded on this run; only the unreadable source remains unavailable. All candidate references resolve, and per-observation hashes are recorded in `observation-hashes.json`; `verification.json` records the receipt digest. This establishes extraction completeness, not OCR accuracy. The previous evidence is preserved. Original targets still require grounding review before training.

The restarted Nemotron-selected local API passed an authenticated held-out-image `/detect` request (HTTP 200) and rejected an unauthenticated request (HTTP 401). The response included all three detector sections; `/ready` confirmed the 18-class chart detector and reported `model_ready: false`. This verifies detector service behavior independently of language-model hosting. Evidence: `outputs/chart-agent/nemotron-api-detector-smoke.json`.

## Bounded LoRA framework compatibility job

Launched `https://huggingface.co/jobs/ordlibrary/6a9fae7ce686246ca69aa238` with `scripts/hf_nemotron_lora_compatibility.py`: one optimizer step on A100 large, a 30-minute hard timeout, the public NeMo AutoModel 26.08 image, pinned BF16 weights and four synthetic examples. No private dataset, artifact upload or retained checkpoint is part of this diagnostic. Latest verified state at launch was `SCHEDULING` / pulling container image. This tests basic training compatibility on the existing job service; it does not prove full-context memory capacity, adaptation quality or persistent hosting. The 16K real-data pilot remains unlaunched.

## Combined deployment

`deploy/nemotron/compose.yaml` now includes the pinned GPU model, CPU chart API, and optional Caddy HTTPS profile. The API uses private container routing to `http://model:8000`, the dedicated model credential, and the completed detector mounted read-only. Host model/API ports bind to localhost. Detector and live-data services can start while the model initializes; authenticated `/ready` is the model-readiness check. Compose configuration validation passed using placeholder credentials with quiet output. No GPU or public deployment was started.

On the chosen NVIDIA host, after staging runtime assets and setting separate `NEMOTRON_API_KEY` and `CHART_API_KEY` values:

```sh
docker compose -f deploy/nemotron/compose.yaml up -d --build
# After configuring DNS and CHART_DOMAIN, enable HTTPS explicitly:
docker compose -f deploy/nemotron/compose.yaml --profile public up -d
```

The new API image initially could not reach Docker because Colima was stopped. The existing runtime was started, and `clawd-chart-api:nemotron-ocr` built successfully. Only 115 KB of application source entered the build context. A temporary ARM64 CPU API container passed authentication, all three detectors, research readiness and a held-out-image `/detect` request; it was removed after the check. An offline OCR execution inside the image recovered all five manually inspected value labels. Evidence: `outputs/chart-agent/nemotron-container-smoke.json` and `nemotron-container-ocr.json`. These checks do not prove x86 deployment or GPU model behavior. The GPU Compose deployment was not started.

Latest compatibility-job observation: `6a9fae7ce686246ca69aa238` advanced to RUNNING, initialized its optimizer and one-step scheduler, and had not yet reported a completed optimizer step. Startup/initialization is not a training pass.

## Completed training compatibility and real-data pilot

Job `6a9fae7ce686246ca69aa238` is COMPLETED. It ran one synthetic LoRA optimizer step with finite training loss 5.9830, gradient norm 20.3261, reported GPU memory 61.80 GiB and validation loss 5.1350. Initialization plus execution took 237.64 seconds; the first step took 104.33 seconds and may include compilation, so it is not a steady-state throughput measurement. Logs reported 7,072,256 trainable parameters. No checkpoint was retained. Evidence: `outputs/nemotron-lora-compat-6a9fae7ce686246ca69aa238.log` and matching JSON.

The subsequent real-data pilot was launched as `https://huggingface.co/jobs/ordlibrary/6a9fb0e5259f8e97255ee443`: 10 optimizer steps, A100 large, one-hour timeout. It reuses the existing private dataset at revision `6456947cf89cbb5d1a4c0c120199a5038532bfc6`, verifies the original archive and converted train/validation hashes, excludes all pending visual rows, and retains pilot checkpoints and run metadata in private `ordlibrary/clawd-nemotron-chart-lora-pilot`. No detector weights or raw datasets are uploaded.

The pilot chooses 64 training rows across length quantiles in descending order and trains on the first 10, starting with the longest. This stresses the long-example tail; it is not a representative average-speed benchmark. Its eight-row validation selection is only a pilot check. It enables full activation checkpointing, following [NVIDIA's configuration guidance](https://docs.nvidia.com/nemo/automodel/development/gradient-checkpointing), to test long examples on the A100. Memory fit, completed steps, retained checkpoints and checkpoint reload remain unverified until this job produces evidence. It does not replace the planned full-data training or the visual-grounding review.

### Adapter serving check after the pilot

The [pinned vLLM 0.27.1 Nemotron implementation](https://raw.githubusercontent.com/vllm-project/vllm/v0.27.1/vllm/model_executor/models/nemotron_h.py) declares `SupportsLoRA`, maps Hugging Face backbone names, and explicitly skips `mtp.` during LoRA loading. This identifies a direct adapter-serving path to test, but does not establish that the NeMo-exported adapter loads correctly with the selected NVFP4 quantization and kernels. After a checkpoint exists, inspect its tensor names/configuration, attempt authenticated adapter inference, and verify that it is actually active before exposing it through the chart API. The prepared deployment currently serves only the base model; it must not claim an adapter is active merely because the training job finishes.

### Repaired source image

A fresh download of `hf://buckets/ordlibrary/charts/images/hs_0016.png` is a valid image with SHA-256 `5154807cafd7be3904450351fcbcaa166908f220b9b5b0168724e900e0528052`; it differs from the damaged prepared copy. Visual inspection confirmed that it matches the three panel-(b) questions. The originals and the uploaded training archive were preserved. `scripts/repair_nemotron_chart_evidence.py` created `outputs/nemotron-visual-evidence-repaired` with explicit source-override provenance. All 1,712 images now have available OCR and detector observations; all 3,198 candidate references resolve. Receipt verification is in that directory's `verification.json`. This repairs extraction completeness; answer-grounding review remains incomplete and candidates remain excluded from training.

The real-data pilot subsequently reached optimizer/training-scheduler initialization with its requested ten-step limit and five-step checkpoint interval. The preparation assertions therefore passed before training initialization. No completed real-data step or retained adapter has yet been verified in this observation.

### Real-data memory failure and corrected attempt

Job `6a9fb0e5259f8e97255ee443` ended ERROR before completing its first step. `MaskedCrossEntropy` attempted a 6.37 GiB float32 logits allocation while only 1.33 GiB was free on the 79.25 GiB A100. The failed run retained configuration, selection and status at private repository revision `0cad8547f986c68be8bb7779aa169100918f8519`, but no adapter weights. Local evidence: `outputs/nemotron-chart-pilot-6a9fb0e5259f8e97255ee443.log` and matching JSON.

The corrected attempt is `https://huggingface.co/jobs/ordlibrary/6a9fb31fe686246ca69aa2a9`. It uses the same 16K context, sample selection, ten-step limit, A100 hardware and one-hour cap, replacing the loss with `nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy`. [NVIDIA documents this loss](https://docs.nvidia.com/nemo/automodel/latest/recipes-e2e-examples/sft-peft) as avoiding the full vocabulary-logits tensor. This is a targeted response to the observed failure; success with Nemotron's MTP path and this dataset remains unverified. The failed job was terminal before submission; no concurrent duplicate pilot was launched. The local 100-step recipe still uses the original loss until a successful corrected GPU check justifies promoting the change.

### First reviewed visual-to-text training examples

Assistant inspection of source images plus extracted text/coordinates approved two revenue-chart examples: Retail revenue `$4900` and Direct minus Partner revenue `$6200`. Three banking-chart examples were rejected for text-only training: OCR misread `25%` as `5%`, and two answers require legend/sector color associations that the current observations omit. This was assistant review, not external human review.

Decisions and observation hashes are in `outputs/nemotron-visual-review/decisions.jsonl`. `scripts/prepare_reviewed_nemotron_visual.py` produced `outputs/nemotron-reviewed-visual/train.jsonl` with two reviewed rows and matching provenance. It uses the API's evidence compaction and rejects changed observations, missing required prompt evidence, and held-out groups. Three regression checks for those failure cases passed. There are still 3,193 unreviewed candidates; the reviewed rows have not been uploaded or included in any training job.

`review_direct_bar_questions.py` additionally checks three explicit direct-revenue question forms on synthetic vertical bar charts. It requires one high-confidence bottom category label, one uniquely aligned high-confidence currency value above it, exact agreement with the target, and retention of both strings in the API-sized prompt. It does not approve comparisons, totals, missing categories or ambiguous aligned values. Five rejection/grounding tests passed. Eight rows passed this deterministic check; combining them with prior assistant review (without duplicate IDs) yields nine approved, three rejected and 3,186 unreviewed rows. The prepared subset is `outputs/nemotron-reviewed-visual-expanded`; none has been added to the already-running pilot. This consistency check is not a corpus OCR-accuracy benchmark or an external human review.

The fused-loss attempt `6a9fb31fe686246ca69aa2a9` ended ERROR during setup because NeMo explicitly requires `model.output_hidden_states=True` for that loss. It did not complete an optimizer step. Its metadata is preserved at repository revision `08bc3f232564e345552cd75159d63eff39493729`, with local logs under `outputs/nemotron-chart-pilot-6a9fb31fe686246ca69aa2a9.*`.

The launcher now includes that required setting. Corrected job `https://huggingface.co/jobs/ordlibrary/6a9fb537259f8e97255ee517` was submitted after confirming the previous job terminal, retaining the same data, 16K context, ten-step limit, A100 hardware and one-hour cap. Successful long-example training and retained adapter weights remain unverified.

`merge_nemotron_reviewed_data.py` created a separate local package at `outputs/nemotron-chart-data-reviewed`: 31,642 training rows (the original 31,633 plus nine grounded visual-to-text examples), 3,257 validation rows and 3,126 test rows. Byte-prefix verification proves the original training data is unchanged, validation/test files match byte-for-byte, all 38,025 IDs are unique and chart groups do not cross splits. Three visual rows are rejected and 3,186 remain pending. Evidence: `merge-verification.json` in that package. This package has not been uploaded or used by the active pilot, and its augmented tokenizer audit remains pending.

The augmented package's pinned-tokenizer audit is now complete. The nine added rows have lengths 2,560, 2,398, 2,297, 2,521, 2,910, 2,820, 2,606, 2,695 and 2,564 tokens. An incremental audit verified the original training-byte prefix, unchanged holdouts and tokenizer-file hashes before combining these lengths with the earlier full audit. The merged training maximum remains 13,035, with zero rows over 16,384. Evidence: `outputs/nemotron-reviewed-token-audit.json`. This is tokenizer-format verification, not native NeMo loss-mask or training completion evidence.
