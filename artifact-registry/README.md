# OnchainAI Artifact Registry

Type-generic registry records for **agent**, **model**, **skill**, **tool**, and **mcp_server**.

Each entry is a **Metaplex Core asset** (same pattern as [registry.musebook.trade](https://registry.musebook.trade): Core asset + identity/registry record) with a bound content hash and an attestation field.

This is the registry used by [onchainai.fun](https://onchainai.fund) (source: this repository’s `onchainai/` tree, Railway project `onchainai`). It is designed to sit **alongside** the Musebook agent registry, not replace it. Agents stay compatible with `themusebook/agent-network`; models, skills, tools and MCP servers use the same Core-asset shape with a `type` field.

## What is live vs not live

| Piece | Status |
|---|---|
| Metaplex Core collection + assets on **Solana devnet** | Created by `scripts/register-devnet.mjs` with a throwaway keypair. Receipts in `receipts/`. |
| Musebook agent registry (`registry.musebook.trade`) | Live for **agents** on devnet. This PR extends its schema in a **draft** only — nothing is deployed to that site. |
| SkillHub on-chain catalog (bundleHash / merkleLeaf / merkle root) | Existing. Skills in this registry reuse that exact encoding. |
| `zk-clawd-primitives` (`clawd-zk`, program id `4vJ9JU1bJJE96FWSVKmnrL3xFU5jSBSVdk9x4La2vzhn`) | **Not deployed. Not live.** PR #2 pinned verifying keys with **all slots empty**, so every proof is refused until circuits and a key ceremony exist. Attestation `kind` is `authority-signature`; `zk` is `pending`. |

Do not point this tooling at mainnet. The register script refuses any cluster other than `devnet`.

## Hash binding

- **skill** — SkillHub `skill-bundle-v1` `bundleHash`, and `merkleLeaf = sha256(slug || 0x00 || bundleHash)` (see `Solizardking/skillhub-main` `scripts/build-catalog.mjs`). The registry `contentHash` is the 32-byte bundle digest.
- **model** — SHA-256 of a canonical `model-v1` preimage: Hugging Face repo id, revision SHA, sorted file hashes (LFS SHA-256 when present), license, plus optional training provenance (dataset hashes, config hashes, eval).
- **tool** / **mcp_server** — SHA-256 of a canonical preimage over the manifest hash, package version, and source commit.
- **agent** — SHA-256 of a canonical preimage over the Metaplex Core asset / identity fields.

Uniqueness key: `(type, contentHash, version)`.

## Formal verification

`formal/lean` (Lean 4.36, no Mathlib) specifies the registration rules. `scripts/check.sh` builds Lean, emits preimage vectors, and checks them against the TypeScript implementation. Theorems close with no `sorry`.

## Devnet registration

```bash
# throwaway keypair only — never a user wallet or a mainnet key
node scripts/register-devnet.mjs --keypair /path/to/throwaway.json
```

Hauhau models are registered first: `ordlibrary/hauhau-qwen36-onchain`, then `ordlibrary/hauhau-qwen36-uncensored`. After that, one SkillHub skill, one tool list, and one MCP server card.
