# Spark chart stack on Hugging Face and Orin

The private Space `ordlibrary/clawd-spark-chart-agent` hosts the pinned Spark 4B
model, LoRA adapter, chart API, ONNX detectors, OCR, and research indexes.
The connected Orin runs `clawd-chart-gateway.service`, a small authenticated
proxy on `127.0.0.1:8092`. Spark inference runs on the HF GPU; the Orin's existing
local inference service is preserved.

- `GET /health`: Orin gateway process health.
- `GET /ready`: actual hosted model, detector, research, and tape status.
- `POST /analyze`: chart analysis and bounded tool execution.
- `POST /detect`: hosted ONNX detection.
- `POST /tokenize`: exact native-tokenizer roundtrip.
- `POST /model/v1/chat/completions`: authenticated Spark inference.

For the private Space, HF authentication and application authentication are
separate. The gateway keeps HF credentials in its mode-0600 environment file and
injects the relevant application key upstream. Local clients use the chart key
as a Bearer credential. Do not place the HF token in browser JavaScript.

The model runtime uses the exact reviewed Spark revision and validates both
custom Python files. Adapter deployments must specify a complete commit hash.
The default serving context is 4096 total tokens; this deployment configures
8192 total tokens. Excess requests fail explicitly instead of silently truncating
chart evidence. Streaming and required tool-choice modes are not implemented.

The Orin agent can use these environment settings alongside its private key:

```
CLAWD_INFERENCE_URL=http://127.0.0.1:8092/model/v1
CLAWD_MODEL=clawd-spark
CLAWD_CHART_URL=http://127.0.0.1:8092
```

Run `nvidia/nemotron_ultra_agent.py --market SOL --mode observer` for analysis
only. Its explicit endpoint takes precedence over unrelated HF/NVIDIA tokens.
Missing Vulcan, perps tools, or observed portfolio price history remain reported
as unavailable. The chart token tape is not a perps orderbook.

Deployment scripts:

- `scripts/hf_spark_pilot.py`: bounded corrective LoRA training and reload checks.
- `scripts/deploy_spark_space.py`: stages by default; `--deploy` uploads a private Space.
- `deploy/chart-agent/orin/gateway.py`: Python 3.8-compatible Orin gateway.

The trained ChartDete artifact is a separate opt-in upload requiring the pending
specific authorization. Existing detector labels do not establish profitability.
A10G-small HF hosting is billed while starting/running; configure a one-hour idle
sleep and pause the Space when it is no longer needed.

## Verified deployment

The Space is running on A10G-small with 3600-second idle sleep. Live checks passed
for authentication, native tokenizer roundtrips, actual amount-conversion tool
execution, chart OCR, both existing detectors, and a connected, fresh token tape.
The Orin gateway and its observer agent were also exercised against the hosted
model. Vulcan is unavailable on the Orin, so the observer correctly reports
missing perps market data and holds.

The chart UI uses `CHART_API_KEY` from the local, ignored, mode-0600 file
`outputs/spark-hosting-secrets.json`. The Orin launcher loads its private key
from `~/.config/clawd-chart/gateway.env`; users do not need to copy HF tokens
into browser fields.

Evidence is in `outputs/spark-space-deployment.json`,
`outputs/spark-space-smoke.json`, and `outputs/orin-spark-smoke.json`.
