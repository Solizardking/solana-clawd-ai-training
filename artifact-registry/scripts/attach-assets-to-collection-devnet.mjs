#!/usr/bin/env node
/**
 * Attach already-minted standalone Core assets to the OnchainAI Artifact Registry
 * collection on devnet. Does not mint or burn. FuGrQ5 must be update authority
 * of both the asset and the collection.
 *
 * Uses mpl-core update() -> updateV2 with newCollection + Collection update authority.
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createUmi } from "@metaplex-foundation/umi-bundle-defaults";
import { keypairIdentity, publicKey } from "@metaplex-foundation/umi";
import {
  fetchAsset,
  fetchCollection,
  mplCore,
  update,
  updateAuthority,
} from "@metaplex-foundation/mpl-core";
import nacl from "tweetnacl";
import bs58 from "bs58";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const DEVNET_RPC = process.env.SOLANA_DEVNET_RPC || "https://api.devnet.solana.com";
const COLLECTION = "2Xkbm5YpDqR1hMXiBa3CQaRg7qgvuhUAPj6E4fDCVaPy";
const AUTHORITY = "FuGrQ5KbtdcdGDdorYs4xauDTG8yyRZRpPChAwbWUX2k";
const RESERVE_SOL = Number(process.env.MINT_RESERVE_SOL || "0.02");

function arg(name, fallback = null) {
  const i = process.argv.indexOf(name);
  if (i === -1) return fallback;
  return process.argv[i + 1] ?? fallback;
}

function loadKeypair(p) {
  const raw = JSON.parse(readFileSync(p, "utf8"));
  const secret = Uint8Array.from(raw);
  return secret.length === 64 ? nacl.sign.keyPair.fromSecretKey(secret) : nacl.sign.keyPair.fromSeed(secret);
}

function sleep(ms) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

function balanceLamports(pubkey) {
  let lastErr;
  for (let i = 0; i < 8; i++) {
    try {
      const out = execFileSync("solana", ["balance", pubkey, "--url", DEVNET_RPC, "--output", "json"], { encoding: "utf8", timeout: 20000 });
      const j = JSON.parse(out);
      if (typeof j.lamports === "number") return j.lamports;
      return Math.round(parseFloat(String(j.balance ?? out)) * 1e9);
    } catch (err) {
      lastErr = err;
      const msg = String(err?.stderr || err?.message || err);
      if (!/error sending request|timeout|ETIMEDOUT|ECONN|429|503|502|fetch|SIGTERM|spawnSync/i.test(msg + String(err?.code || "") + String(err?.errno || ""))) throw err;
      sleep(1000 * (i + 1));
    }
  }
  throw lastErr;
}

function inCollection(ua) {
  return ua?.type === "Collection" && String(ua.address) === COLLECTION;
}

async function fetchAssetRetry(umi, asset) {
  let lastErr;
  for (let i = 0; i < 8; i++) {
    try {
      return await fetchAsset(umi, publicKey(asset), { skipDerivePlugins: true });
    } catch (err) {
      lastErr = err;
      const msg = String(err?.message || err);
      if (!/429|timeout|fetch|ECONN|busy|rate|503|502|blockhash/i.test(msg)) throw err;
      sleep(1000 * (i + 1));
    }
  }
  throw lastErr;
}

const receiptsPath = path.join(root, "receipts/devnet.json");
const receipts = JSON.parse(readFileSync(receiptsPath, "utf8"));
const entries = (receipts.entries || []).filter((e) => e.record?.asset);
const verifyOnly = process.argv.includes("--verify-only");
const limit = process.argv.includes("--limit") ? Number(arg("--limit")) : Infinity;

async function main() {
  if ((process.env.SOLANA_CLUSTER || "devnet") !== "devnet") throw new Error("devnet only");
  if (/mainnet/i.test(DEVNET_RPC)) throw new Error("refusing mainnet RPC");

  const umi = createUmi(DEVNET_RPC).use(mplCore());
  const collectionAccount = await fetchCollection(umi, COLLECTION);
  console.log(JSON.stringify({
    collection: String(collectionAccount.publicKey),
    numMinted: collectionAccount.numMinted,
    currentSize: collectionAccount.currentSize,
    collectionUpdateAuthority: collectionAccount.updateAuthority,
    receiptAssets: entries.length,
  }));

  if (verifyOnly) {
    let inCol = 0, standalone = 0, other = 0;
    const samples = [];
    for (const e of entries) {
      const a = await fetchAssetRetry(umi, e.record.asset);
      if (inCollection(a.updateAuthority)) inCol++;
      else if (a.updateAuthority?.type === "Address") standalone++;
      else other++;
      if (samples.length < 8) {
        samples.push({ name: e.name, asset: e.record.asset, updateAuthority: a.updateAuthority });
      }
    }
    console.log(JSON.stringify({ inCollection: inCol, standalone, other, samples }, null, 2));
    return;
  }

  const keypairPath = arg("--keypair");
  if (!keypairPath) throw new Error("usage: --keypair <throwaway.json> [--limit N] [--verify-only]");
  const kp = loadKeypair(keypairPath);
  const authority = bs58.encode(kp.publicKey);
  if (authority !== AUTHORITY) throw new Error(`expected throwaway ${AUTHORITY}, got ${authority}`);
  if (String(collectionAccount.updateAuthority) !== AUTHORITY) {
    throw new Error(`collection update authority is ${collectionAccount.updateAuthority}, not ${AUTHORITY}`);
  }
  umi.use(keypairIdentity(umi.eddsa.createKeypairFromSecretKey(kp.secretKey)));

  function save() {
    writeFileSync(receiptsPath, JSON.stringify(receipts, null, 2) + "\n");
  }

  let attachedThisRun = 0;
  let alreadyIn = 0;
  let stoppedReason = null;
  const toProcess = entries.slice();
  let processed = 0;

  for (const entry of toProcess) {
    if (attachedThisRun >= limit) break;
    const asset = entry.record.asset;
    const name = entry.name || entry.record?.payload?.slug || asset;
    if (entry.collectionAttached) {
      alreadyIn++;
      continue;
    }
    const before = await fetchAssetRetry(umi, asset);
    if (inCollection(before.updateAuthority)) {
      alreadyIn++;
      entry.collectionAttached = true;
      continue;
    }
    if (before.updateAuthority?.type !== "Address" || String(before.updateAuthority.address) !== AUTHORITY) {
      stoppedReason = `${name} (${asset}) updateAuthority is ${JSON.stringify(before.updateAuthority)}, not throwaway Address; refusing to attach`;
      console.error(stoppedReason);
      break;
    }
    const lamports = balanceLamports(authority);
    if (lamports / 1e9 < RESERVE_SOL) {
      stoppedReason = `insufficient SOL to continue safely (balance ${lamports / 1e9} SOL, reserve ${RESERVE_SOL})`;
      console.error(stoppedReason);
      break;
    }
    try {
      const result = await update(umi, {
        asset: before,
        newCollection: collectionAccount.publicKey,
        newUpdateAuthority: updateAuthority("Collection", [collectionAccount.publicKey]),
      }).sendAndConfirm(umi);
      const sig = bs58.encode(result.signature);
      let after = null;
      for (let i = 0; i < 8; i++) {
        try {
          after = await fetchAssetRetry(umi, asset);
          if (inCollection(after.updateAuthority)) break;
        } catch (err) {
          after = { updateAuthority: { error: String(err.message || err) } };
        }
        sleep(500 * (i + 1));
      }
      if (!inCollection(after?.updateAuthority)) {
        throw new Error(`attach did not stick for ${name} (${asset}): ${JSON.stringify(after?.updateAuthority)} sig=${sig}`);
      }
      sleep(200);
      entry.collectionAttached = true;
      entry.signatures = { ...(entry.signatures || {}), collectionAttach: sig };
      entry.explorer = entry.explorer || {};
      entry.explorer.collectionAttach = `https://explorer.solana.com/tx/${sig}?cluster=devnet`;
      attachedThisRun++;
      processed++;
      save();
      console.log(JSON.stringify({
        attached: name,
        asset,
        sig,
        n: attachedThisRun,
        balanceSol: balanceLamports(authority) / 1e9,
      }));
    } catch (e) {
      const msg = String(e?.message || e);
      console.error(`attach failed for ${name}: ${msg}`);
      const maybeSig = (e && e.signature) || (msg.match(/Signature ([1-9A-HJ-NP-Za-km-z]{32,})/) || [])[1];
      try {
        const check = await fetchAssetRetry(umi, asset);
        if (inCollection(check.updateAuthority)) {
          entry.collectionAttached = true;
          if (maybeSig) {
            entry.signatures = { ...(entry.signatures || {}), collectionAttach: maybeSig };
            entry.explorer = entry.explorer || {};
            entry.explorer.collectionAttach = `https://explorer.solana.com/tx/${maybeSig}?cluster=devnet`;
          }
          attachedThisRun++;
          save();
          console.log(JSON.stringify({ attached: name, asset, sig: maybeSig || null, n: attachedThisRun, note: "confirmed on chain after client error" }));
          continue;
        }
      } catch {}
      if (/429|blockhash|timeout|fetch|ECONN|busy|rate|503|502|expired|block height/i.test(msg)) {
        sleep(2000);
        toProcess.push(entry);
        continue;
      }
      stoppedReason = msg;
      save();
      throw e;
    }
  }

  const colAfter = await fetchCollection(umi, COLLECTION);
  receipts.collectionMembership = {
    collection: COLLECTION,
    attachedThisRun,
    alreadyIn,
    numMintedOnChain: colAfter.numMinted,
    currentSize: colAfter.currentSize,
    stoppedReason,
    updatedAt: new Date().toISOString(),
  };
  save();
  console.log(JSON.stringify({
    done: true,
    attachedThisRun,
    alreadyIn,
    numMintedOnChain: colAfter.numMinted,
    currentSize: colAfter.currentSize,
    stoppedReason,
    balanceSol: balanceLamports(authority) / 1e9,
  }, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
