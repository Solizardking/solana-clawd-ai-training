# onchain.x402.wtf Integration

`onchain.x402.wtf` is the public registry and product surface for Solana Clawd
AI models. The local model kit produces the dataset, adapter, manifest, and
dry-run payloads that the registry consumes.

## Links

- Registry: https://onchain.x402.wtf
- Well-known manifest: https://onchain.x402.wtf/.well-known/clawd-registry.json
- Model API: https://onchain.x402.wtf/api/models
- Register API: https://onchain.x402.wtf/api/register
- Source handoff: `ai-training/onchain.md`

## Dry-Run

```bash
ai-training/model-kit/bin/clawd-model-kit register \
  --hf-model solanaclawd/my-solana-lora \
  --manifest data/model_kit/model_kit_manifest.json
```

## Live Off-Chain Register

```bash
ai-training/model-kit/bin/clawd-model-kit register \
  --hf-model solanaclawd/my-solana-lora \
  --manifest data/model_kit/model_kit_manifest.json \
  --endpoint https://your-router.example/v1 \
  --eval-accuracy 0.72 \
  --live \
  --yes
```

## Onchain Register

Onchain writes are separate from the off-chain index and require a funded,
isolated keypair:

```bash
ai-training/model-kit/bin/clawd-model-kit register \
  --hf-model solanaclawd/my-solana-lora \
  --manifest data/model_kit/model_kit_manifest.json \
  --onchain \
  --live \
  --yes
```

Model registration is not permission to trade. Trading remains behind wallet
isolation, explicit approval, simulation, and Vulcan/Rise risk checks.
