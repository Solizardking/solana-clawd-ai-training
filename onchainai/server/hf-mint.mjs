// Self-serve Hugging Face model -> Metaplex Core asset (devnet only).
// The server fetches Hub data with HF_TOKEN (never sent to the client), computes the registry
// content hash, and builds an UNSIGNED create transaction. The user's wallet pays and signs;
// the asset keypair is generated and held in the browser. This module never holds a private key.
import { createHash } from 'node:crypto';
import nacl from 'tweetnacl';
import bs58 from 'bs58';
import { createUmi } from '@metaplex-foundation/umi-bundle-defaults';
import { createNoopSigner, publicKey } from '@metaplex-foundation/umi';
import { create, fetchAsset, mplCore, MPL_CORE_PROGRAM_ID } from '@metaplex-foundation/mpl-core';
import {
  REGISTRY_COLLECTION, SELF_SERVE_SOURCE, attributesToMap, bindingMessage, buildModelRecord,
  modelContentHash, selfServeAttributes,
} from '../src/registry-hash.js';

export const CLUSTER = 'devnet';
// 'onchainai-self-serve-hf/v1' was the marker of the first e2e test mint; kept so it stays listed.
const SOURCES = new Set([SELF_SERVE_SOURCE, 'onchainai-self-serve-hf/v1']);
export const DEVNET_GENESIS = 'EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG';
export const CORE_PROGRAM = String(MPL_CORE_PROGRAM_ID);
export const COMPUTE_BUDGET_PROGRAM = 'ComputeBudget111111111111111111111111111111';
// Update authority for every self-serve asset: a PDA of the SPL Memo program. The Memo program
// never calls invoke_signed, so nobody can sign for it: the asset name, URI and Attributes are
// frozen at mint (content-hash bound), while the minting wallet keeps ownership (transfer/burn).
// It is also the index key the gallery queries through DAS getAssetsByAuthority.
export const INDEX_SEED = 'onchainai-hf-model-registry/v1';
export const MEMO_PROGRAM = 'MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr';
export const LIMITS = Object.freeze({ maxSiblings: 5000, maxHashFiles: 128, maxHashFileBytes: 10_000_000, maxHashTotalBytes: 40_000_000 });
const HF = 'https://huggingface.co';
const ID_PART = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;
const SHA40 = /^[0-9a-f]{40}$/;
const BASE58_KEY = /^[1-9A-HJ-NP-Za-km-z]{32,44}$/;

const sha256Bytes = bytes => new Uint8Array(createHash('sha256').update(bytes).digest());
const httpError = (message, status = 400) => Object.assign(new Error(message), { status });

let umiSingleton;
function devnetRpcUrl() {
  const url = process.env.SOLANA_DEVNET_RPC_URL || 'https://api.devnet.solana.com';
  if (/mainnet/i.test(url)) throw httpError('Refusing: configured RPC looks like mainnet. Self-serve minting is devnet-only.', 500);
  return url;
}
function getUmi() {
  if (!umiSingleton) umiSingleton = createUmi(devnetRpcUrl()).use(mplCore());
  return umiSingleton;
}
export function indexAuthority(seed = INDEX_SEED) {
  const [pda] = getUmi().eddsa.findPda(publicKey(MEMO_PROGRAM), [new TextEncoder().encode(seed)]);
  return String(pda);
}

let genesisChecked = false;
export { httpError, getUmi, BASE58_KEY };
export async function devnetRpc(method, params = []) {
  const res = await fetch(devnetRpcUrl(), { method: 'POST', headers: { 'Content-Type': 'application/json', 'User-Agent': 'OnchainAI/0.2' }, body: JSON.stringify({ jsonrpc: '2.0', id: 1, method, params }), signal: AbortSignal.timeout(20000) });
  if (!res.ok) throw httpError(`Devnet RPC unavailable (HTTP ${res.status})`, 502);
  const data = await res.json();
  if (data.error) throw httpError(`Devnet RPC: ${data.error.message}`, 502);
  return data.result;
}
export async function assertDevnet() {
  if (genesisChecked) return;
  const genesis = await devnetRpc('getGenesisHash');
  if (genesis !== DEVNET_GENESIS) throw httpError('Refusing: RPC genesis hash is not Solana devnet.', 500);
  genesisChecked = true;
}

// ---------- Hugging Face ----------
export function parseModelId(input) {
  let s = String(input || '').trim();
  if (!s) throw httpError('Paste a Hugging Face model id like owner/name or a huggingface.co URL.');
  s = s.replace(/^https?:\/\//i, '').replace(/^(www\.)?(huggingface\.co|hf\.co)\//i, '').replace(/[?#].*$/, '').replace(/\/+$/, '');
  const parts = s.split('/');
  if (parts[0] === 'datasets' || parts[0] === 'spaces') throw httpError('That is a dataset or Space. Only model repositories can be minted here.');
  if (parts[0] === 'models') parts.shift();
  const [owner, name] = parts;
  if (!owner || !name || !ID_PART.test(owner) || !ID_PART.test(name) || name.includes('..')) throw httpError('Not a valid Hugging Face model id. Use owner/name.');
  if (parts.length > 2 && !['tree', 'blob', 'resolve', 'commit'].includes(parts[2])) throw httpError('Not a valid Hugging Face model URL.');
  return `${owner}/${name}`;
}

function hfHeaders(json = true) {
  const h = { 'User-Agent': 'OnchainAI-mint/0.2' };
  if (json) h.Accept = 'application/json';
  if (process.env.HF_TOKEN) h.Authorization = `Bearer ${process.env.HF_TOKEN}`;
  return h;
}
async function hfJson(path) {
  const res = await fetch(`${HF}${path}`, { headers: hfHeaders(), signal: AbortSignal.timeout(20000) });
  if (res.status === 404 || res.status === 401) throw httpError('Model not found on Hugging Face (or it is private).', 404);
  if (res.status === 403) throw httpError('Hugging Face refused access to this model.', 403);
  if (res.status === 429) throw httpError('Hugging Face rate limit hit. Try again in a minute.', 429);
  if (!res.ok) throw httpError(`Hugging Face API error (HTTP ${res.status}).`, 502);
  return res.json();
}
const encPath = p => p.split('/').map(encodeURIComponent).join('/');

async function hashFile(id, sha, path, expectedSize) {
  const res = await fetch(`${HF}/${id}/resolve/${sha}/${encPath(path)}`, { headers: hfHeaders(false), redirect: 'follow', signal: AbortSignal.timeout(30000) });
  if (!res.ok) return { error: res.status === 401 || res.status === 403 ? 'gated-or-forbidden' : `http-${res.status}` };
  const buf = new Uint8Array(await res.arrayBuffer());
  if (expectedSize != null && buf.length !== expectedSize) return { error: 'size-mismatch' };
  return { sha256: `sha256-${createHash('sha256').update(buf).digest('hex')}`, size: buf.length };
}

async function mapLimit(items, limit, fn) {
  const out = new Array(items.length); let i = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => { while (i < items.length) { const k = i++; out[k] = await fn(items[k], k); } }));
  return out;
}

const cache = new Map(); // `${id}@${sha}` -> result
const headCache = new Map(); // id -> { sha, at }
function remember(key, value) { cache.set(key, value); if (cache.size > 64) cache.delete(cache.keys().next().value); return value; }
export function resetHfCache() { cache.clear(); headCache.clear(); }

function pickCard(card = {}) {
  const keep = ['license', 'license_name', 'language', 'base_model', 'datasets', 'tags', 'model_name', 'pipeline_tag', 'library_name', 'quantized_by', 'model_creator'];
  const out = {};
  for (const k of keep) if (card[k] != null) out[k] = card[k];
  return out;
}
function licenseOf(info) {
  const card = info.cardData || {};
  if (typeof card.license === 'string') return card.license === 'other' && card.license_name ? `other:${card.license_name}` : card.license;
  const tag = (info.tags || []).find(t => t.startsWith('license:'));
  return tag ? tag.slice(8) : null;
}

/**
 * Fetch a model pinned to a commit and compute the registry payload + content hash.
 * revision: optional 40-hex commit; defaults to the current head.
 */
export async function fetchHfModel(input, { revision } = {}) {
  const id = parseModelId(input);
  let sha = revision;
  if (sha != null && !SHA40.test(sha)) throw httpError('revision must be a 40-character commit sha.');
  if (!sha) {
    const head = headCache.get(id);
    if (head && Date.now() - head.at < 60_000) sha = head.sha;
  }
  if (sha && cache.has(`${id}@${sha}`)) return cache.get(`${id}@${sha}`);
  const info = await hfJson(`/api/models/${id}${sha ? `/revision/${sha}` : ''}?blobs=true`);
  if (info.private) throw httpError('Model not found on Hugging Face (or it is private).', 404);
  if (info.disabled) throw httpError('This model is disabled on Hugging Face.', 403);
  const repo = info.id || id;
  sha = info.sha;
  if (!SHA40.test(sha || '')) throw httpError('Hugging Face did not return a commit sha.', 502);
  headCache.set(id, { sha, at: Date.now() });
  if (cache.has(`${repo}@${sha}`)) return cache.get(`${repo}@${sha}`);

  const siblings = (info.siblings || []).slice(0, LIMITS.maxSiblings);
  const files = [], unhashed = [], toFetch = [];
  let budget = LIMITS.maxHashTotalBytes;
  for (const s of [...siblings].sort((a, b) => a.rfilename.localeCompare(b.rfilename))) {
    if (s.lfs?.sha256 && /^[0-9a-f]{64}$/.test(s.lfs.sha256)) { files.push({ path: s.rfilename, sha256: `sha256-${s.lfs.sha256}`, size: s.lfs.size ?? s.size ?? null }); continue; }
    const size = s.size ?? null;
    if (toFetch.length >= LIMITS.maxHashFiles) { unhashed.push({ path: s.rfilename, gitBlobId: s.blobId || null, size, reason: 'file-count-limit' }); continue; }
    if (size == null || size > LIMITS.maxHashFileBytes || size > budget) { unhashed.push({ path: s.rfilename, gitBlobId: s.blobId || null, size, reason: 'byte-limit' }); continue; }
    budget -= size; toFetch.push(s);
  }
  const hashed = await mapLimit(toFetch, 6, s => hashFile(repo, sha, s.rfilename, s.size ?? null));
  toFetch.forEach((s, i) => {
    const h = hashed[i];
    if (h.sha256) files.push({ path: s.rfilename, sha256: h.sha256, size: h.size });
    else unhashed.push({ path: s.rfilename, gitBlobId: s.blobId || null, size: s.size ?? null, reason: h.error });
  });
  files.sort((a, b) => a.path.localeCompare(b.path));
  unhashed.sort((a, b) => a.path.localeCompare(b.path));
  if (!files.length) throw httpError('No files in this repository could be hashed, so it cannot be bound to a content hash.', 422);

  const card = info.cardData || {};
  const baseModel = Array.isArray(card.base_model) ? card.base_model.join(', ') : (card.base_model || null);
  const payload = {
    hfRepo: repo,
    revisionSha: sha,
    license: licenseOf(info),
    creator: info.author || repo.split('/')[0],
    url: `${HF}/${repo}`,
    baseModel,
    pipelineTag: info.pipeline_tag || null,
    libraryName: info.library_name || null,
    hfLastModified: info.lastModified || null,
    files,
    provenance: {
      source: 'huggingface-hub',
      registrar: SELF_SERVE_SOURCE,
      unhashedFiles: unhashed,
      note: 'LFS/Xet files use the Hugging Face LFS SHA-256; other files are SHA-256 of the bytes at this revision.',
    },
  };
  const contentHash = await modelContentHash(payload, sha256Bytes);
  const totalSize = siblings.reduce((n, s) => n + (s.lfs?.size ?? s.size ?? 0), 0);
  const preview = {
    id: repo, author: info.author || null, sha, lastModified: info.lastModified || null, createdAt: info.createdAt || null,
    downloads: info.downloads ?? null, downloadsAllTime: info.downloadsAllTime ?? null, likes: info.likes ?? null,
    tags: (info.tags || []).slice(0, 40), pipelineTag: info.pipeline_tag || null, libraryName: info.library_name || null,
    license: payload.license, gated: info.gated || false, baseModel, cardData: pickCard(card),
    modelType: info.config?.model_type || null, architectures: info.config?.architectures || null,
    parameters: info.safetensors?.total ?? null, ggufArchitecture: info.gguf?.architecture ?? null, contextLength: info.gguf?.context_length ?? null,
    usedStorage: info.usedStorage ?? null, totalSize, fileCount: siblings.length, siblingsTruncated: (info.siblings || []).length > siblings.length,
    hashedCount: files.length, unhashedCount: unhashed.length, spaces: (info.spaces || []).length,
    files: [...files.map(f => ({ ...f, hashed: true })), ...unhashed.map(u => ({ path: u.path, size: u.size, hashed: false, reason: u.reason }))]
      .sort((a, b) => (b.size || 0) - (a.size || 0)).slice(0, 200),
  };
  return remember(`${repo}@${sha}`, { ok: true, cluster: CLUSTER, preview, payload, contentHash, limits: LIMITS });
}

// ---------- Mint transaction (unsigned) ----------
export function assetName(repo, maxBytes = 32) {
  const name = repo.split('/')[1] || repo;
  // Core asset names are capped at 32 bytes here to keep the create tx small.
  let out = '';
  for (const ch of name) { if (new TextEncoder().encode(out + ch).length > maxBytes) break; out += ch; }
  return out;
}
// Short /m/<asset> alias of /api/hf-mint/metadata/<asset>, keyed by asset address rather than repo+revision to keep the create tx under 1232 bytes;
// the endpoint reads the asset's onchain Attributes to find hfRepo + revisionSha.
export function metadataUri(baseUrl, asset) {
  return `${baseUrl.replace(/\/$/, '')}/m/${asset}`;
}
export function publicBaseUrl() { return process.env.PUBLIC_BASE_URL || 'https://onchainai.fund'; }

export function verifyBinding(record, signature) {
  let sig, key;
  try { sig = bs58.decode(signature); key = bs58.decode(record.authority); } catch { return false; }
  if (sig.length !== 64 || key.length !== 32) return false;
  return nacl.sign.detached.verify(new TextEncoder().encode(bindingMessage(record)), sig, key);
}

/** Pure (offline) builder for the unsigned Core create tx; used by prepareMint and tests. */
/** Generic unsigned Core create tx (wallet = fee payer + owner, asset = fresh co-signer, frozen update authority). */
export function buildCoreCreateTx({ authority, asset, names, uri, attributeList, updateAuthority, blockhash }) {
  const umi = getUmi();
  const payer = createNoopSigner(publicKey(authority));
  const ctx = { ...umi, payer, identity: payer };
  for (const name of names) {
    const tx = create(ctx, {
      asset: createNoopSigner(publicKey(asset)), owner: publicKey(authority), updateAuthority: publicKey(updateAuthority),
      name, uri, plugins: [{ type: 'Attributes', attributeList }],
    }).setBlockhash(blockhash).build(ctx);
    const wire = umi.transactions.serialize(tx);
    if (wire.length <= 1232) return { tx, wire, name };
  }
  throw httpError('Record is too large to fit in one Solana transaction.', 422);
}
export function shrinkingNames(full) { const out = []; for (let n = 32; n >= 8; n -= 4) out.push(assetName(full, n)); return out; }

/** Pure (offline) builder for the unsigned model create tx; used by prepareMint and tests. */
export function buildCreateTx({ record, bindingSignature, asset, blockhash, baseUrl = publicBaseUrl() }) {
  const attributeList = selfServeAttributes(record, bindingSignature);
  const uri = metadataUri(baseUrl, asset);
  // Very long repo ids: shorten only the display name until the tx fits (hfRepo attribute stays complete).
  const { tx, wire, name } = buildCoreCreateTx({ authority: record.authority, asset, names: shrinkingNames(record.payload.hfRepo), uri, attributeList, updateAuthority: indexAuthority(), blockhash });
  return { tx, wire, attributeList, uri, name };
}

/**
 * Build the unsigned create tx. Inputs come from the browser, but the payload/hash are re-derived
 * here from Hugging Face so a client cannot bind a hash it did not fetch.
 */
export async function prepareMint({ modelId, revision, wallet, asset, bindingSignature, baseUrl = publicBaseUrl() }) {
  if (!BASE58_KEY.test(wallet || '') || !BASE58_KEY.test(asset || '')) throw httpError('wallet and asset must be base58 public keys.');
  if (wallet === asset) throw httpError('asset must be a fresh keypair, not the wallet.');
  if (!SHA40.test(revision || '')) throw httpError('revision (40-hex commit) is required so the mint is pinned.');
  await assertDevnet();
  const fetched = await fetchHfModel(modelId, { revision });
  const record = buildModelRecord({ payload: fetched.payload, contentHash: fetched.contentHash, authority: wallet, asset });
  if (!verifyBinding(record, bindingSignature)) throw httpError('Binding signature does not verify for this wallet and content hash. Sign the binding message again.', 401);
  const { blockhash, lastValidBlockHeight } = await getUmi().rpc.getLatestBlockhash({ commitment: 'confirmed' });
  const { tx, wire, attributeList, uri, name } = buildCreateTx({ record, bindingSignature, asset, blockhash, baseUrl });
  const fee = await devnetRpc('getFeeForMessage', [Buffer.from(tx.serializedMessage).toString('base64'), { commitment: 'confirmed' }]);
  const space = assetSpace(name, uri, attributeList);
  const [rent, balance, payerMin] = await Promise.all([
    devnetRpc('getMinimumBalanceForRentExemption', [space]), devnetRpc('getBalance', [wallet, { commitment: 'confirmed' }]), devnetRpc('getMinimumBalanceForRentExemption', [0]),
  ]);
  const needed = rent + (fee?.value ?? 10_000);
  const have = balance?.value ?? 0;
  if (have < needed || (have - needed > 0 && have - needed < payerMin)) {
    throw httpError(`Not enough devnet SOL: this mint needs about ${((needed + payerMin) / 1e9).toFixed(4)} SOL (rent ${(rent / 1e9).toFixed(4)} + fee, leaving your wallet rent-exempt); the wallet has ${(have / 1e9).toFixed(4)}. Get devnet SOL at https://faucet.solana.com and try again.`, 402);
  }
  return {
    ok: true, cluster: CLUSTER, transaction: Buffer.from(wire).toString('base64'), blockhash, lastValidBlockHeight,
    feeLamports: fee?.value ?? null, rentLamportsEstimate: rent, accountSpace: space, walletLamports: have, record: { ...record, attestation: { ...record.attestation, message: bindingMessage(record), signature: bindingSignature } },
    attributes: attributeList, uri, name, updateAuthority: indexAuthority(), coreProgram: CORE_PROGRAM,
    collection: null, collectionNote: collectionNote(),
  };
}
/** Exact AssetV1 account size: base + plugin header + Attributes plugin + plugin registry (no external plugins). */
export function assetSpace(name, uri, attrs) {
  const utf8 = v => new TextEncoder().encode(v).length;
  const base = 1 + 32 + 33 + 4 + utf8(name) + 4 + utf8(uri) + 1;
  const header = 1 + 8;
  const attributes = 1 + 4 + attrs.reduce((n, x) => n + 4 + utf8(x.key) + 4 + utf8(x.value), 0);
  const registry = 1 + 4 + (1 + 1 + 8) + 4;
  return base + header + attributes + registry;
}

export function collectionNote() {
  return `Self-serve mints are not added to the curated registry collection ${REGISTRY_COLLECTION}: joining a Core collection needs its update authority to sign, and this site will not hold that key or co-sign. Each asset is minted uncollected with registry Attributes (schema, contentHash, hfRepo, revisionSha, your binding signature) and update authority set to an unsignable index PDA, so the record is immutable and listed by the gallery.`;
}

// ---------- Submit (devnet only) ----------
export async function submitMint({ transaction }) {
  if (typeof transaction !== 'string' || transaction.length > 3000) throw httpError('transaction (base64) required');
  await assertDevnet();
  const umi = getUmi();
  const bytes = Buffer.from(transaction, 'base64');
  const tx = umi.transactions.deserialize(new Uint8Array(bytes));
  const keys = tx.message.accounts.map(String);
  const programs = tx.message.instructions.map(ix => keys[ix.programIndex]);
  if (!programs.length || programs.some(p => p !== CORE_PROGRAM && p !== COMPUTE_BUDGET_PROGRAM) || !programs.includes(CORE_PROGRAM)) throw httpError('Only Metaplex Core create transactions can be submitted here.', 400);
  const signature = await devnetRpc('sendTransaction', [bytes.toString('base64'), { encoding: 'base64', skipPreflight: false, preflightCommitment: 'confirmed', maxRetries: 3 }]);
  for (let i = 0; i < 40; i++) {
    const st = await devnetRpc('getSignatureStatuses', [[signature]]);
    const s = st?.value?.[0];
    if (s?.err) return { ok: false, signature, error: JSON.stringify(s.err) };
    if (s && ['confirmed', 'finalized'].includes(s.confirmationStatus)) return { ok: true, cluster: CLUSTER, signature, explorer: explorerTx(signature) };
    await new Promise(r => setTimeout(r, 1000));
  }
  return { ok: true, pending: true, cluster: CLUSTER, signature, explorer: explorerTx(signature) };
}
export const explorerTx = sig => `https://explorer.solana.com/tx/${sig}?cluster=devnet`;
export const explorerAddress = a => `https://explorer.solana.com/address/${a}?cluster=devnet`;

// ---------- Gallery / verify (Helius DAS, devnet) ----------
function heliusUrl() {
  const key = process.env.HELIUS_API_KEY;
  if (!key) return null;
  return `https://devnet.helius-rpc.com/?api-key=${encodeURIComponent(key)}`;
}
async function das(method, params) {
  const url = heliusUrl();
  if (!url) throw httpError('Gallery needs HELIUS_API_KEY (devnet DAS) on the server.', 503);
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ jsonrpc: '2.0', id: 'onchainai', method, params }), signal: AbortSignal.timeout(20000) });
  if (!res.ok) throw httpError(`Helius DAS unavailable (HTTP ${res.status})`, 502);
  const data = await res.json();
  if (data.error) throw httpError(`Helius DAS: ${data.error.message}`, 502);
  return data.result;
}
export function summarizeAsset(item) {
  const attrs = attributesToMap(item?.plugins?.attributes?.data?.attribute_list);
  const record = attrs.contentHash && attrs.authority && attrs.version ? {
    type: attrs.type, contentHash: attrs.contentHash, version: attrs.version, cluster: attrs.cluster, authority: attrs.authority,
    uniquenessKey: `${attrs.type}:${(attrs.contentHash || '').replace(/^sha256-/, '')}:${attrs.version}`,
  } : null;
  let signatureValid = false;
  try { signatureValid = !!record && attrs.schema === 'onchainai-artifact-registry/v1' && attrs.type === 'model' && attrs.cluster === 'devnet' && verifyBinding(record, attrs.bindingSignature); } catch { signatureValid = false; }
  return {
    asset: item.id, name: item.content?.metadata?.name || null, uri: item.content?.json_uri || null,
    owner: item.ownership?.owner || null, updateAuthority: item.authorities?.[0]?.address || null, burnt: !!item.burnt,
    hfRepo: attrs.hfRepo || null, revisionSha: attrs.revisionSha || null, license: attrs.license || null,
    contentHash: attrs.contentHash || null, authority: attrs.authority || null, source: attrs.source || null,
    signatureValid, explorer: explorerAddress(item.id),
  };
}
let galleryCache;
export async function gallery({ page = 1, limit = 48 } = {}) {
  if (galleryCache && Date.now() - galleryCache.at < 15_000 && galleryCache.key === `${page}:${limit}`) return galleryCache.data;
  const result = await das('getAssetsByAuthority', { authorityAddress: indexAuthority(), page, limit: Math.min(Math.max(limit, 1), 100), sortBy: { sortBy: 'created', sortDirection: 'desc' } });
  const items = (result.items || []).filter(i => i.interface === 'MplCoreAsset' && !i.burnt).map(summarizeAsset).filter(a => SOURCES.has(a.source));
  const data = { ok: true, cluster: CLUSTER, indexAuthority: indexAuthority(), total: result.total ?? items.length, page, items };
  galleryCache = { at: Date.now(), key: `${page}:${limit}`, data };
  return data;
}
export function resetGalleryCache() { galleryCache = undefined; }
/** Full verification: signature + recompute content hash from Hugging Face at the pinned revision. */
export async function verifyAsset(asset) {
  if (!BASE58_KEY.test(asset || '')) throw httpError('asset must be a base58 address');
  const item = await das('getAsset', { id: asset });
  if (item?.interface !== 'MplCoreAsset') throw httpError('Not a Metaplex Core asset on devnet.', 404);
  const s = summarizeAsset(item);
  let recomputed = null, hashMatches = false, error = null;
  try {
    const f = await fetchHfModel(s.hfRepo, { revision: s.revisionSha });
    recomputed = f.contentHash; hashMatches = recomputed === s.contentHash;
  } catch (e) { error = e.message; }
  return { ok: true, cluster: CLUSTER, ...s, recomputedContentHash: recomputed, hashMatches, immutable: s.updateAuthority === indexAuthority(), error };
}

/** Metadata for a minted self-serve asset, read from its onchain Attributes (devnet RPC, no DAS needed). */
export async function metadataForAsset(asset, baseUrl = publicBaseUrl()) {
  if (!BASE58_KEY.test(asset || '')) throw httpError('asset must be a base58 address');
  let a;
  try { a = await fetchAsset(getUmi(), publicKey(asset)); } catch { throw httpError('Asset not found on devnet.', 404); }
  const attrs = attributesToMap(a.attributes?.attributeList);
  if (!SOURCES.has(attrs.source) || a.updateAuthority?.address !== indexAuthority()) throw httpError('Not an OnchainAI self-serve model asset.', 404);
  const json = await metadataJson(attrs.hfRepo, attrs.revisionSha, baseUrl);
  json.properties.registry.asset = asset;
  json.properties.registry.authority = attrs.authority;
  json.properties.registry.onchainContentHash = attrs.contentHash;
  json.properties.registry.hashMatches = attrs.contentHash === json.properties.registry.contentHash;
  return json;
}
export async function metadataJson(repo, rev, baseUrl = publicBaseUrl()) {
  const f = await fetchHfModel(repo, { revision: rev });
  const p = f.payload;
  return {
    name: assetName(p.hfRepo), symbol: 'HFMODEL',
    description: `Registry record for Hugging Face model ${p.hfRepo} pinned at commit ${p.revisionSha}. Content hash ${f.contentHash} binds ${p.files.length} file hashes. Solana devnet; zk attestation pending.`,
    image: `${baseUrl.replace(/\/$/, '')}/brand/hf-model-nft.svg`,
    external_url: p.url,
    attributes: [
      { trait_type: 'schema', value: 'onchainai-artifact-registry/v1' }, { trait_type: 'type', value: 'model' },
      { trait_type: 'hfRepo', value: p.hfRepo }, { trait_type: 'revisionSha', value: p.revisionSha },
      { trait_type: 'contentHash', value: f.contentHash }, { trait_type: 'license', value: p.license || 'unknown' },
      { trait_type: 'cluster', value: 'devnet' },
    ],
    properties: { category: 'model', registry: { schema: 'onchainai-artifact-registry/v1', contentHash: f.contentHash, payload: p } },
  };
}

// ---------- Per-IP rate limit (protects the server HF token and devnet RPC) ----------
const buckets = new Map();
export function rateLimit(ip, bucket, { max = 30, windowMs = 60_000 } = {}) {
  const key = `${bucket}:${ip || 'unknown'}`, now = Date.now();
  const b = buckets.get(key) || { n: 0, start: now };
  if (now - b.start > windowMs) { b.n = 0; b.start = now; }
  b.n++; buckets.set(key, b);
  if (buckets.size > 5000) buckets.delete(buckets.keys().next().value);
  if (b.n > max) throw httpError('Too many requests. Wait a minute and try again.', 429);
}
export function mintConfig() {
  return {
    ok: true, cluster: CLUSTER, mainnet: 'refused', coreProgram: CORE_PROGRAM, indexAuthority: indexAuthority(), indexSeed: INDEX_SEED,
    registryCollection: REGISTRY_COLLECTION, collection: null, collectionNote: collectionNote(), source: SELF_SERVE_SOURCE,
    hfTokenConfigured: !!process.env.HF_TOKEN, heliusConfigured: !!process.env.HELIUS_API_KEY, limits: LIMITS, devnetGenesis: DEVNET_GENESIS,
  };
}
