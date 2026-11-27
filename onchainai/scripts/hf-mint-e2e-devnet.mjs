#!/usr/bin/env node
// Devnet end-to-end: drives the SAME server functions the /mint page calls (fetchHfModel ->
// bindingMessage -> prepareMint -> submitMint -> verifyAsset), with a throwaway keypair standing in
// for the browser wallet. Devnet only. Usage:
//   HF_TOKEN=... HELIUS_API_KEY=... node scripts/hf-mint-e2e-devnet.mjs --keypair throwaway.json [--model owner/name]
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import nacl from 'tweetnacl';
import bs58 from 'bs58';
import { inspectWire, withSignature } from '../src/tx-inspect.js';
import { CORE_PROGRAM, COMPUTE_BUDGET_PROGRAM, fetchHfModel, prepareMint, submitMint, verifyAsset, indexAuthority, resetGalleryCache, gallery, assertDevnet } from '../server/hf-mint.mjs';
import { bindingMessage, buildModelRecord } from '../src/registry-hash.js';

const arg = (n, d = null) => { const i = process.argv.indexOf(n); return i === -1 ? d : process.argv[i + 1] ?? d; };
const kpPath = arg('--keypair');
if (!kpPath) throw new Error('--keypair /path/to/throwaway-devnet.json required');
if (/mainnet/i.test(process.env.SOLANA_DEVNET_RPC_URL || '')) throw new Error('refusing: mainnet RPC');
const wallet = nacl.sign.keyPair.fromSecretKey(Uint8Array.from(JSON.parse(readFileSync(kpPath, 'utf8'))));
const walletAddr = bs58.encode(wallet.publicKey);
const modelId = arg('--model', 'ordlibrary/hauhau-qwen36-onchain');
await assertDevnet();
console.log(`wallet ${walletAddr} (throwaway, devnet) model ${modelId} index ${indexAuthority()}`);

// 1. what GET /api/hf-mint/model returns
const fetched = await fetchHfModel(modelId);
console.log(`hf ${fetched.payload.hfRepo}@${fetched.payload.revisionSha} files=${fetched.payload.files.length} hash=${fetched.contentHash}`);
// 2. browser: fresh asset keypair + wallet signs the binding message
const assetKp = nacl.sign.keyPair();
const asset = bs58.encode(assetKp.publicKey);
const record = buildModelRecord({ payload: fetched.payload, contentHash: fetched.contentHash, authority: walletAddr, asset });
const bindingSignature = bs58.encode(nacl.sign.detached(new TextEncoder().encode(bindingMessage(record)), wallet.secretKey));
// 3. POST /api/hf-mint/prepare
const prepared = await prepareMint({ modelId: fetched.payload.hfRepo, revision: fetched.payload.revisionSha, wallet: walletAddr, asset, bindingSignature });
const wire = Buffer.from(prepared.transaction, 'base64');
console.log(`prepared ${wire.length}-byte tx fee=${prepared.feeLamports} rent~${prepared.rentLamportsEstimate}`);
// 4. browser: inspect, then wallet + asset sign the message bytes (src/tx-inspect.js, same as /mint)
const info = inspectWire(wire, bs58.encode);
if (info.feePayer !== walletAddr) throw new Error('fee payer is not the wallet');
if (info.programs.some(p => p !== CORE_PROGRAM && p !== COMPUTE_BUDGET_PROGRAM)) throw new Error('unexpected program');
let signed = wire;
for (const signer of info.signers) {
  const kp = signer === walletAddr ? wallet : signer === asset ? assetKp : null;
  if (!kp) throw new Error(`unexpected signer ${signer}`);
  signed = withSignature(signed, info, signer, nacl.sign.detached(info.message, kp.secretKey));
}
signed = Buffer.from(signed);
// 5. POST /api/hf-mint/submit
const sent = await submitMint({ transaction: signed.toString('base64') });
console.log('submit', sent);
if (!sent.ok) process.exit(1);
// 6. GET /api/hf-mint/verify (DAS may lag a few seconds)
let v;
for (let i = 0; i < 20; i++) { try { v = await verifyAsset(asset); if (v.contentHash) break; } catch (e) { v = { error: e.message }; } await new Promise(r => setTimeout(r, 2000)); }
console.log('verify', JSON.stringify(v));
resetGalleryCache();
const g = await gallery();
console.log(`gallery total=${g.total} includes=${g.items.some(i => i.asset === asset)}`);
mkdirSync('receipts', { recursive: true });
const out = { schema: 'onchainai-hf-self-serve/receipt/v1', cluster: 'devnet', mintedAt: new Date().toISOString(), wallet: walletAddr, asset, signature: sent.signature, explorer: { tx: sent.explorer, asset: `https://explorer.solana.com/address/${asset}?cluster=devnet` }, indexAuthority: indexAuthority(), record: prepared.record, verify: v };
writeFileSync(`receipts/hf-self-serve-${fetched.payload.hfRepo.replace('/', '__')}.json`, JSON.stringify(out, null, 2) + '\n');
