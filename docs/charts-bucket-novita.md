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
