#!/usr/bin/env node
/**
 * Mint Metaplex Core collection + assets on Solana devnet only.
 * Never uses mainnet. Refuses any cluster other than devnet.
 * Expects a throwaway keypair; do not pass a user wallet or a mainnet key.
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createUmi } from "@metaplex-foundation/umi-bundle-defaults";
import { keypairIdentity, generateSigner, none } from "@metaplex-foundation/umi";
import { createCollection, create, mplCore } from "@metaplex-foundation/mpl-core";
import { buildRecord, signRecord, metaplexAttributes, verifyRecordSignature } from "../src/record.mjs";
import nacl from "tweetnacl";
import bs58 from "bs58";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const DEVNET_RPC = process.env.SOLANA_DEVNET_RPC || "https://api.devnet.solana.com";
const CLUSTER = "devnet";

function arg(name, fallback = null) {
  const i = process.argv.indexOf(name);
  if (i === -1) return fallback;
  return process.argv[i + 1] ?? fallback;
}

function loadKeypair(p) {
  const raw = JSON.parse(readFileSync(p, "utf8"));
  if (!Array.isArray(raw) || (raw.length !== 64 && raw.length !== 32)) {
    throw new Error("expected a solana JSON keypair (64 secret bytes or 32-byte seed)");
  }
  const secret = Uint8Array.from(raw);
  const kp = secret.length === 64 ? nacl.sign.keyPair.fromSecretKey(secret) : nacl.sign.keyPair.fromSeed(secret);
  return kp;
}

function loadExample(name) {
  return JSON.parse(readFileSync(path.join(root, "examples", name), "utf8"));
}

function ensureDevnet() {
  if ((process.env.SOLANA_CLUSTER || CLUSTER) !== "devnet") {
    throw new Error("refusing to run: this script is devnet-only");
  }
  if (/mainnet/i.test(DEVNET_RPC)) {
    throw new Error("refusing to run: RPC URL looks like mainnet");
  }
}

function airdrop(pubkey, keypairPath) {
  const solana = process.env.SOLANA_BIN || "solana";
  const check = execFileSync(solana, ["balance", pubkey, "--url", DEVNET_RPC, "--output", "json"], { encoding: "utf8" });
  const lamports = JSON.parse(check).lamports ?? Math.round(parseFloat(check) * 1e9);
  console.log(`devnet balance ${pubkey}: ${lamports} lamports`);
  if (lamports >= 1_500_000_000) return;
  for (const amount of [2, 2]) {
    try {
      const out = execFileSync(solana, ["airdrop", String(amount), pubkey, "--url", DEVNET_RPC], { encoding: "utf8" });
      console.log(out.trim());
    } catch (e) {
      console.warn(`airdrop ${amount} SOL failed: ${e.stderr || e.message}`);
    }
  }
}

function explorer(sig, address) {
  const q = "?cluster=devnet";
  return {
    tx: sig ? `https://explorer.solana.com/tx/${sig}${q}` : null,
    address: address ? `https://explorer.solana.com/address/${address}${q}` : null,
  };
}

async function main() {
  ensureDevnet();
  const keypairPath = arg("--keypair");
  if (!keypairPath) throw new Error("usage: node scripts/register-devnet.mjs --keypair /path/to/throwaway.json");
  const dryRun = process.argv.includes("--dry-run");
  const kp = loadKeypair(keypairPath);
  const authority = bs58.encode(kp.publicKey);
  console.log(`authority ${authority} (throwaway, devnet only)`);

  const examples = [
    { file: "hauhau-qwen36-onchain.json", name: "hauhau-qwen36-onchain" },
    { file: "hauhau-qwen36-uncensored.json", name: "hauhau-qwen36-uncensored" },
    { file: "skillhub-onchain.json", name: "skillhub-onchain" },
    { file: "cheshire-mcp-tools.json", name: "cheshire-mcp-tools" },
    { file: "cheshire-mcp-server.json", name: "cheshire-terminal-mcp" },
  ];

  const agentExample = {
    type: "agent",
    version: "v1",
    payload: {
      name: "onchainai-artifact-registry",
      owner: authority,
      handle: "onchainai-registry",
      endpoint: "https://onchainai.fund/registry",
    },
  };

  const built = [];
  for (const ex of examples) {
    const data = loadExample(ex.file);
    const rec = signRecord(
      buildRecord({ type: data.type, version: data.version, authority, payload: data.payload, cluster: "devnet" }),
      kp.secretKey,
    );
    verifyRecordSignature(rec);
    built.push({ name: ex.name, record: rec });
  }
  const agentRec = signRecord(
    buildRecord({ type: agentExample.type, version: agentExample.version, authority, payload: agentExample.payload, cluster: "devnet" }),
    kp.secretKey,
  );
  verifyRecordSignature(agentRec);
  built.push({ name: "onchainai-artifact-registry", record: agentRec });

  if (dryRun) {
    const out = { cluster: "devnet", dryRun: true, authority, entries: built.map((b) => b.record) };
    console.log(JSON.stringify(out, null, 2));
    return;
  }

  airdrop(authority, keypairPath);

  const umi = createUmi(DEVNET_RPC).use(mplCore());
  const umiKp = umi.eddsa.createKeypairFromSecretKey(kp.secretKey);
  umi.use(keypairIdentity(umiKp));

  const collectionSigner = generateSigner(umi);
  const collectionTx = await createCollection(umi, {
    collection: collectionSigner,
    name: "OnchainAI Artifact Registry",
    uri: "https://github.com/Solizardking/solana-clawd-ai-training/tree/feat/artifact-registry-zk/artifact-registry",
    plugins: [
      {
        type: "Attributes",
        attributeList: [
          { key: "schema", value: "onchainai-artifact-registry/v1" },
          { key: "cluster", value: "devnet" },
          { key: "zk", value: "pending" },
          { key: "zkProgramId", value: "4vJ9JU1bJJE96FWSVKmnrL3xFU5jSBSVdk9x4La2vzhn" },
        ],
      },
    ],
  }).sendAndConfirm(umi);
  const collection = collectionSigner.publicKey.toString();
  const collectionSig = bs58.encode(collectionTx.signature);
  console.log(`collection ${collection} sig ${collectionSig}`);

  const minted = [];
  for (const item of built) {
    const rec = { ...item.record, collection };
    const assetSigner = generateSigner(umi);
    const result = await create(umi, {
      asset: assetSigner,
      collection: collectionSigner.publicKey,
      name: item.name.slice(0, 32),
      uri: `https://github.com/Solizardking/solana-clawd-ai-training/blob/feat/artifact-registry-zk/artifact-registry/receipts/${item.name}.json`,
      plugins: [{ type: "Attributes", attributeList: metaplexAttributes({ ...rec, collection, asset: assetSigner.publicKey.toString() }) }],
    }).sendAndConfirm(umi);
    const asset = assetSigner.publicKey.toString();
    const sig = bs58.encode(result.signature);
    const record = { ...rec, asset, identity: { ...rec.identity, collection, asset } };
    const receipt = {
      name: item.name,
      record,
      signatures: { mint: sig, collection: collectionSig },
      explorer: { asset: explorer(sig, asset), collection: explorer(collectionSig, collection) },
    };
    minted.push(receipt);
    console.log(`minted ${item.name} asset ${asset} sig ${sig}`);
  }

  const receiptsDir = path.join(root, "receipts");
  mkdirSync(receiptsDir, { recursive: true });
  const summary = {
    schema: "onchainai-artifact-registry/receipts/v1",
    cluster: "devnet",
    mintedAt: new Date().toISOString(),
    authority,
    collection,
    collectionSignature: collectionSig,
    explorer: explorer(collectionSig, collection),
    note: "Throwaway devnet keypair. zk-clawd-primitives is not live; attestations are authority signatures with zk pending. Not submitted to the production registry.musebook.trade API.",
    entries: minted,
  };
  writeFileSync(path.join(receiptsDir, "devnet.json"), JSON.stringify(summary, null, 2) + "\n");
  for (const m of minted) {
    writeFileSync(path.join(receiptsDir, `${m.name}.json`), JSON.stringify(m, null, 2) + "\n");
  }
  console.log(`wrote ${minted.length} receipts to ${receiptsDir}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
