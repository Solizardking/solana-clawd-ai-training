# Spark 4B evaluation

Candidate: `XHToken/Spark-X2.5-4B`, pinned to
`5e10fcc0286756aebf7c41dc52c1e42d95c70281`.

This is a text-model candidate for interpreting chart detector/OCR output and
Solana data. It does not replace the visual detectors or provide live market data
by itself. Training and deployment depend on evaluation results.

## Reproduce the bounded inference check

`scripts/hf_spark_smoke.py` prints its configuration by default. `--launch`
starts a single A10G-small Hugging Face Job capped at 30 minutes, with no training
corpus, secrets, or artifact uploads. The job downloads the pinned public model.
The launcher requires the reviewed metadata/code in `outputs/spark-preflight`.
It checks the custom Python file hashes before loading them with
`trust_remote_code=True`.

The current run is [6aa0292032d5d0c22c5adb45](https://huggingface.co/jobs/ordlibrary/6aa0292032d5d0c22c5adb45).
An earlier scheduling attempt was canceled to correct the pipeline argument for
Transformers 4.57.1: `tokenizer_encode_kwargs={"enable_thinking": False}`.

The test uses BF16 and eager attention. It checks a pipeline response, ten fixed
synthetic JSON diagnostics, tokenizer roundtrips, and one native tool request.
The native tool test checks parsed arguments; it does not execute a tool.
Generated token counts, latency, and peak allocated GPU memory are logged.
Results do not establish trading performance or optimized serving throughput.

Nemotron's prior fixed-case baseline was 10/10, while its 100-step adapter was
9/10 with an exact-address copying failure. Hardware and backends differ, so this
comparison is diagnostic accuracy only. Spark completed with 9/10 diagnostics passing. It returned `decimal_precision`
instead of the correct mint field `decimals`. Exact address and u64 copying,
chart evidence cases, tokenizer roundtrips, and the native tool request passed.

Peak PyTorch allocated GPU memory was 7.702 GiB on an A10G; this excludes
non-PyTorch allocations and is not a training memory estimate. The ten short
diagnostic generations totaled 159 tokens in about 7.98 seconds (19.9 tokens/s,
including prompt processing). These are short synthetic prompts, not a load test.
The pipeline startup, including model download/load, took 71.38 seconds.

Evidence: `outputs/spark-quality-6aa0292032d5d0c22c5adb45.json` and `.log`.
Spark is a viable candidate for a small Solana-focused training pilot; it has not
been fine-tuned or connected to the live chart API by this evaluation.

## Corrective pilot and deployment (2026-09-08)

The 128-step corrective pilot completed and retained its private adapter at
`ordlibrary/clawd-spark-4b-lora-pilot`, revision
`31b3053a70f4884b6107dfa0ded1a4d854889dab`.
Adapter SHA-256:
`fa4d5d75e1fffe1a6ba1ac4f824162a6bc490a3559517c8b789bb8e2a8cb546f`.
Downloaded adapter bytes were checked against this digest before deployment.

The reloaded model passed 10/10 original diagnostics and all six mint-field
paraphrases (three prior development checks and three additional checks), native
tool-request parsing, and exact tokenizer roundtrips. This is a narrow corrective
evaluation, not broad Solana expertise or chart-trading accuracy.

Training used 192 seeded existing rows and 32 unique corrective examples, each
corrective example repeated twice per epoch, over two epochs. Thus there were
512 presentations: 384 existing-data and 128 corrective presentations. The job's
`correction_presentations: 64` metadata denotes the per-epoch count. The separate
previously blocked nine visual additions were not used or uploaded.

The first 64-step adapter passed the original diagnostics but failed a new mint
paraphrase, so it was not promoted. An earlier run also failed artifact upload
because PEFT placed a local cache path in the model card; the launcher now writes
valid Hub model metadata and pins the base model in the adapter config.

Deployment: private HF Space `ordlibrary/clawd-spark-chart-agent`, A10G-small with
one-hour idle sleep. Connected Orin `name@192.168.55.1` runs the authenticated
Python gateway on loopback port 8092. It forwards inference and chart processing
to HF, preserving the existing Orin inference service. The agent launcher is
`~/clawd-chart/run-agent.py` and defaults to observer mode.

See [deployment instructions](../deploy/chart-agent/spark/README.md). Hosted
readiness and end-to-end verification must pass before reporting the service live.
