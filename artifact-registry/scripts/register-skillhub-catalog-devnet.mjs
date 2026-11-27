#!/usr/bin/env node
/**
 * Mint SkillHub canonical catalog skills as Metaplex Core assets on devnet only.
 * Adds them to the existing OnchainAI Artifact Registry collection.
 * Skips already-minted assets and skills removed from catalog.json.
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createUmi } from "@metaplex-foundation/umi-bundle-defaults";
import { keypairIdentity, generateSigner, publicKey } from "@metaplex-foundation/umi";
import { create, fetchAsset, fetchCollection, mplCore } from "@metaplex-foundation/mpl-core";
import { buildRecord, signRecord, metaplexAttributes, verifyRecordSignature } from "../src/record.mjs";
import { skillMerkleLeaf } from "../src/hash.mjs";
import nacl from "tweetnacl";
import bs58 from "bs58";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const DEVNET_RPC = process.env.SOLANA_DEVNET_RPC || "https://api.devnet.solana.com";
const COLLECTION = "2Xkbm5YpDqR1hMXiBa3CQaRg7qgvuhUAPj6E4fDCVaPy";
const COLLECTION_SIG = "2dN4MRZKXA3urGps4e59ByYDJqDsJCaHmNzkQQfStb8avwFuV27DGsSSUzb7rhQfArtgee2KpMChp5jSh9w5yWKZ";
const ESTIMATE_SOL = Number(process.env.MINT_ESTIMATE_SOL || "0.0055"); // observed ~0.00541
const RESERVE_SOL = Number(process.env.MINT_RESERVE_SOL || "0.02");

function arg(name, fallback = null) {
  const i = process.argv.indexOf(name);
  if (i === -1) return fallback;
  return process.argv[i + 1] ?? fallback;
}

function receiptFileName(slug) {
  return `${slug.replaceAll("/", "__")}.json`;
}

function ensureDevnet() {
  if ((process.env.SOLANA_CLUSTER || "devnet") !== "devnet") throw new Error("devnet only");
  if (/mainnet/i.test(DEVNET_RPC)) throw new Error("refusing mainnet RPC");
}

function loadKeypair(p) {
  const raw = JSON.parse(readFileSync(p, "utf8"));
  if (!Array.isArray(raw) || (raw.length !== 64 && raw.length !== 32)) {
    throw new Error("expected a solana JSON keypair");
  }
  const secret = Uint8Array.from(raw);
  return secret.length === 64 ? nacl.sign.keyPair.fromSecretKey(secret) : nacl.sign.keyPair.fromSeed(secret);
}

function balanceLamports(pubkey) {
  let lastErr;
  for (let i = 0; i < 8; i++) {
    try {
      const out = execFileSync("solana", ["balance", pubkey, "--url", DEVNET_RPC, "--output", "json"], { encoding: "utf8" });
      const j = JSON.parse(out);
      if (typeof j.lamports === "number") return j.lamports;
      return Math.round(parseFloat(String(j.balance ?? out)) * 1e9);
    } catch (err) {
      lastErr = err;
      const msg = String(err?.stderr || err?.message || err);
      if (!/error sending request|timeout|ECONN|429|503|502|fetch/i.test(msg)) throw err;
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 1000 * (i + 1));
    }
  }
  throw lastErr;
}

function explorer(sig, address) {
  const q = "?cluster=devnet";
  return {
    tx: sig ? `https://explorer.solana.com/tx/${sig}${q}` : null,
    address: address ? `https://explorer.solana.com/address/${address}${q}` : null,
  };
}

function loadJson(p) {
  return JSON.parse(readFileSync(p, "utf8"));
}

const skillhubRoot = arg("--skillhub", path.resolve(root, "../../skillhub-main"));
const catalog = loadJson(path.join(skillhubRoot, "catalog.json"));
const registry = loadJson(path.join(skillhubRoot, "public/.well-known/onchain-skill-registry.json"));
const catalogSlugs = new Set(catalog.map((s) => s.slug));
const regSkills = Array.isArray(registry.skills) ? registry.skills : Object.values(registry.skills);
const regBy = Object.fromEntries(regSkills.map((s) => [s.slug, s]));
const removedFromCatalog = [...Object.keys(regBy)].filter((s) => !catalogSlugs.has(s)).sort();

const receiptsPath = path.join(root, "receipts/devnet.json");
const receipts = existsSync(receiptsPath) ? loadJson(receiptsPath) : { entries: [] };
const already = new Set(
  (receipts.entries || [])
    .filter((e) => e.record?.type === "skill")
    .map((e) => e.record?.payload?.slug || e.name)
    .filter(Boolean),
);

const queued = catalog
  .filter((s) => !already.has(s.slug))
  .map((s) => {
    const pub = regBy[s.slug];
    if (!pub?.bundleHash || !pub?.merkleLeaf) throw new Error(`missing published hashes for ${s.slug}`);
    return {
      slug: s.slug,
      name: pub.name || s.name || s.slug,
      bundleHash: pub.bundleHash,
      merkleLeaf: pub.merkleLeaf,
      catalogHash: registry.catalogHash,
      catalogMerkleRoot: registry.merkleRoot,
    };
  });

const estimateOnly = process.argv.includes("--estimate-only");
const limit = process.argv.includes("--limit") ? Number(arg("--limit")) : Infinity;

console.log(JSON.stringify({
  cluster: "devnet",
  collection: COLLECTION,
  catalogCount: catalogSlugs.size,
  publishedRegistryCount: regSkills.length,
  removedFromCatalog,
  alreadyMinted: [...already].sort(),
  queued: queued.length,
  estimateSolPerMint: ESTIMATE_SOL,
  reserveSol: RESERVE_SOL,
}, null, 2));

if (estimateOnly) {
  const lamports = process.argv.includes("--keypair")
    ? balanceLamports(bs58.encode(loadKeypair(arg("--keypair")).publicKey))
    : Math.round(0.9572128 * 1e9);
  const sol = lamports / 1e9;
  const budget = Math.max(0, sol - RESERVE_SOL);
  const canFit = Math.min(queued.length, Math.floor(budget / ESTIMATE_SOL));
  const remainAfter = queued.length - canFit;
  const solNeeded = Math.max(0, remainAfter * ESTIMATE_SOL);
  console.log(JSON.stringify({
    balanceSol: sol,
    canFit,
    remainingAfterThisRun: remainAfter,
    solNeededToFinish: Number(solNeeded.toFixed(4)),
    note: `Estimate uses ${ESTIMATE_SOL} SOL/mint from prior Core asset mints (~0.00541 observed). Actual cost may vary.`,
  }, null, 2));
  process.exit(0);
}

async function main() {
  ensureDevnet();
  const keypairPath = arg("--keypair");
  if (!keypairPath) throw new Error("usage: --keypair <throwaway.json>");
  const kp = loadKeypair(keypairPath);
  const authority = bs58.encode(kp.publicKey);
  if (authority !== "FuGrQ5KbtdcdGDdorYs4xauDTG8yyRZRpPChAwbWUX2k") {
    console.warn("warning: keypair pubkey is", authority, "expected throwaway FuGrQ5…");
  }

  const receiptsDir = path.join(root, "receipts");
  mkdirSync(receiptsDir, { recursive: true });
  const catalogReceiptsPath = path.join(receiptsDir, "skillhub-catalog-devnet.json");

  function save(summaryExtra = {}) {
    receipts.mintedAt = new Date().toISOString();
    receipts.collection = receipts.collection || COLLECTION;
    receipts.authority = authority;
    receipts.cluster = "devnet";
    receipts.removedFromCanonicalCatalog = removedFromCatalog.map((slug) => ({
      slug,
      status: "retired-from-catalog",
      minted: already.has(slug) || (receipts.entries || []).some((e) => e.record?.payload?.slug === slug),
      burned: false,
      note: "Removed from SkillHub catalog.json on main. Do not burn any existing asset.",
    }));
    writeFileSync(receiptsPath, JSON.stringify({ ...receipts, ...summaryExtra }, null, 2) + "\n");
    const skillEntries = (receipts.entries || []).filter((e) => e.record?.type === "skill");
    writeFileSync(catalogReceiptsPath, JSON.stringify({
      schema: "onchainai-artifact-registry/skillhub-catalog-receipts/v1",
      cluster: "devnet",
      collection: COLLECTION,
      authority,
      catalogCount: catalogSlugs.size,
      removedFromCanonicalCatalog: receipts.removedFromCanonicalCatalog,
      minted: skillEntries.length,
      entries: skillEntries,
    }, null, 2) + "\n");
  }

  const umi = createUmi(DEVNET_RPC).use(mplCore());
  umi.use(keypairIdentity(umi.eddsa.createKeypairFromSecretKey(kp.secretKey)));
  // mpl-core create() expects a CollectionV1 object (publicKey/oracles/lifecycleHooks),
  // not a raw PublicKey. Passing publicKey(COLLECTION) makes collection.publicKey
  // undefined, so the asset is minted standalone.
  const collectionAccount = await fetchCollection(umi, COLLECTION);

  const version = `1.0.0-onchain.${String(registry.catalogHash).replace(/^sha256-/, "").slice(0, 8)}`;
  const toMint = queued.slice(0, Number.isFinite(limit) ? limit : queued.length);
  let mintedThisRun = 0;
  let stoppedReason = null;

  for (const skill of toMint) {
    const lamports = balanceLamports(authority);
    const sol = lamports / 1e9;
    if (sol < RESERVE_SOL + ESTIMATE_SOL * 0.8) {
      stoppedReason = `insufficient SOL to continue safely (balance ${sol} SOL, reserve ${RESERVE_SOL})`;
      console.error(stoppedReason);
      break;
    }
    const rec = signRecord(buildRecord({
      type: "skill",
      version,
      authority,
      payload: {
        slug: skill.slug,
        name: skill.name,
        bundleHash: skill.bundleHash,
        merkleLeaf: skill.merkleLeaf,
        catalogHash: skill.catalogHash,
        catalogMerkleRoot: skill.catalogMerkleRoot,
        sourceRepo: "Solizardking/skillhub-main",
        catalog: "catalog.json on origin/main (1password removed; 673 canonical skills)",
      },
      cluster: "devnet",
    }), kp.secretKey);
    verifyRecordSignature(rec);
    if (rec.skillMerkleLeaf !== skill.merkleLeaf) throw new Error(`leaf mismatch ${skill.slug}`);
    const assetSigner = generateSigner(umi);
    const before = lamports;
    try {
      const result = await create(umi, {
        asset: assetSigner,
        collection: collectionAccount,
        name: skill.slug.slice(0, 32),
        uri: `https://github.com/Solizardking/solana-clawd-ai-training/blob/feat/devnet-artifact-receipts/artifact-registry/receipts/skills/${receiptFileName(skill.slug)}`,
        plugins: [{
          type: "Attributes",
          attributeList: metaplexAttributes({ ...rec, collection: COLLECTION, asset: assetSigner.publicKey.toString() }),
        }],
      }).sendAndConfirm(umi);
      const asset = assetSigner.publicKey.toString();
      const sig = bs58.encode(result.signature);
      const onchain = await fetchAsset(umi, asset, { skipDerivePlugins: true });
      const ua = onchain.updateAuthority;
      const inCollection = ua?.type === "Collection" && String(ua.address) === COLLECTION;
      if (!inCollection) {
        throw new Error(`asset ${asset} (${skill.slug}) was not added to collection ${COLLECTION}; updateAuthority=${JSON.stringify(ua)}`);
      }
      const after = balanceLamports(authority);
      const receipt = {
        name: skill.slug,
        record: { ...rec, collection: COLLECTION, asset, identity: { ...rec.identity, collection: COLLECTION, asset } },
        signatures: { mint: sig, collection: COLLECTION_SIG },
        explorer: { asset: explorer(sig, asset), collection: explorer(COLLECTION_SIG, COLLECTION) },
        costLamports: before - after,
      };
      receipts.entries = receipts.entries || [];
      receipts.entries.push(receipt);
      already.add(skill.slug);
      const skillFile = path.join(receiptsDir, "skills", receiptFileName(skill.slug));
      mkdirSync(path.dirname(skillFile), { recursive: true });
      writeFileSync(skillFile, JSON.stringify(receipt, null, 2) + "\n");
      mintedThisRun++;
      save();
      console.log(JSON.stringify({
        minted: skill.slug,
        asset,
        sig,
        n: mintedThisRun,
        remainingQueued: queued.length - mintedThisRun,
        balanceSol: after / 1e9,
        costSol: (before - after) / 1e9,
      }));
    } catch (e) {
      const msg = String(e?.message || e);
      console.error(`mint failed for ${skill.slug}: ${msg}`);
      stoppedReason = msg;
      save({ lastError: { slug: skill.slug, message: msg } });
      if (/insufficient|0x1|found no record of a prior credit/i.test(msg)) break;
      // retry next skill on transient RPC errors
      if (/429|blockhash|timeout|fetch|ECONN|busy|rate/i.test(msg)) {
        await new Promise((r) => setTimeout(r, 2000));
        // retry the same skill
        toMint.unshift(skill);
        continue;
      }
      throw e;
    }
  }

  const finalBal = balanceLamports(authority) / 1e9;
  const stillQueued = catalog.filter((s) => !already.has(s.slug)).length;
  save({
    lastRun: {
      mintedThisRun,
      stoppedReason,
      balanceSol: finalBal,
      stillQueued,
      solNeededToFinish: Number((stillQueued * ESTIMATE_SOL).toFixed(4)),
    },
  });
  console.log(JSON.stringify({
    done: true,
    mintedThisRun,
    skillAssetsTotal: [...already].length,
    stillQueued,
    balanceSol: finalBal,
    solNeededToFinish: Number((stillQueued * ESTIMATE_SOL).toFixed(4)),
    stoppedReason,
    removedFromCatalog,
  }, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
