# Model Kit DAO Handoff

This directory is the bridge between the Solana AI Model Kit and the Clawd DAO
registry flow. The model kit builds datasets, manifests, LoRA adapters, and
release cards. The DAO layer turns that reviewed artifact set into a CAAP/1.0
registration payload and SAS-style attestation records that can be inspected by
wallets, registry UIs, and governance tooling.

## Current Lane

The active handoff lane is the transaction foundation model:

| Field | Value |
| --- | --- |
| Dataset | `solanaclawd/solana-tx-foundation-unified` |
| Dataset size | `82,169` rows (`17,262` CPT + `64,907` SFT) |
| Target model | `solanaclawd/solana-tx-foundation-7b` |
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Registry protocol | `CAAP/1.0` |
| Default cluster | `devnet` |
| Program ID | `3dLst2E3djtCSwG19mFS3REHxtZPngjyga7iYZLDL5xj` |

## Dry-Run Flow

Start from a model-kit manifest if one exists:

```bash
cd /Users/8bit/Downloads/solana-clawd/ai-training

bash dao/register_model.sh \
  --dry-run \
  --manifest data/model_kit/solana_tx_foundation_manifest.json \
  --hf-model solanaclawd/solana-tx-foundation-7b \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --dataset-size 82169 \
  --output outputs/dao/tx-foundation-caap.json
```

If the manifest does not yet exist, the same payload can be previewed directly:

```bash
bash dao/register_model.sh \
  --dry-run \
  --model-hash sha256:pending \
  --hf-model solanaclawd/solana-tx-foundation-7b \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --dataset-size 82169 \
  --output outputs/dao/tx-foundation-caap.json
```

The generated JSON is the payload expected by the model-kit backend
`POST /api/register/preview` and `POST /api/register`.

## Attestation Records

Create a local training-run attestation preview:

```bash
pnpm tsx dao/attestation/create_attestation.ts \
  --dry-run \
  --type training_run \
  --model-id solanaclawd/solana-tx-foundation-7b \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --hf-repo solanaclawd/solana-tx-foundation-unified \
  --size 82169 \
  --hash sha256:pending \
  --job-id pending-hf-credits \
  --output outputs/dao/attestations.jsonl
```

After a model registry PDA exists, append a registry attestation:

```bash
pnpm tsx dao/attestation/create_attestation.ts \
  --dry-run \
  --type registry \
  --model-id solanaclawd/solana-tx-foundation-7b \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --hash sha256:pending \
  --registry-pda <MODEL_REGISTRY_PDA> \
  --output outputs/dao/attestations.jsonl
```

Dry-run mode does not require a local keypair. If `--keypair` is missing, the
scripts use an ephemeral authority for previewing PDAs and hashes. Live writes
still require an explicit funded authority.

## Live Registration

Off-chain registry write:

```bash
bash dao/register_model.sh \
  --model-hash sha256:<reviewed-artifact-hash> \
  --hf-model solanaclawd/solana-tx-foundation-7b \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --dataset-size 82169
```

Onchain ModelRegistry PDA creation:

```bash
bash dao/register_model.sh \
  --onchain \
  --model-hash sha256:<reviewed-artifact-hash> \
  --hf-model solanaclawd/solana-tx-foundation-7b \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --dataset-size 82169 \
  --keypair ~/.config/solana/id.json \
  --cluster devnet
```

Live onchain mode requires:

- Anchor dependencies installed.
- The Solana AI inference IDL at the default path or `--idl <path>`.
- A funded keypair with authority over the registry write.
- A reviewed artifact hash, not `sha256:pending`.

