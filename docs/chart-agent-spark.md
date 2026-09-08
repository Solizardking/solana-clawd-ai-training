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
