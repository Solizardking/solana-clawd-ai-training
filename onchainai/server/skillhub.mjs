// Public SkillHub: anyone submits a skill bundle, the server hashes it with the SkillHub
// skill-bundle-v1 encoding, gates it through the JEV verifier, and builds an UNSIGNED Metaplex Core
// create tx that the submitter's wallet signs and pays for on devnet. No server keys.
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { publicKey } from '@metaplex-foundation/umi';
import { fetchAsset } from '@metaplex-foundation/mpl-core';
import { SKILL_SOURCE, attributesToMap, bindingMessage, buildSkillRecord, skillAttributes, skillBundleHash, skillMerkleLeaf } from '../src/registry-hash.js';
import { verifySkill, isVerdict } from './jev.mjs';
import { BASE58_KEY, CLUSTER, CORE_PROGRAM, assertDevnet, buildCoreCreateTx, devnetRpc, explorerAddress, getUmi, httpError, indexAuthority, publicBaseUrl, shrinkingNames, verifyBinding, assetSpace } from './hf-mint.mjs';

export const SKILL_INDEX_SEED = 'onchainai-skillhub/v1';
export const SKILL_LIMITS = Object.freeze({ maxFiles: 24, maxTotalBytes: 256_000, maxPath: 120 });
const sha = b => new Uint8Array(createHash('sha256').update(b).digest());
const SAFE_PATH = /^(?!\/)(?!.*\.\.)(?!.*\/\/)[A-Za-z0-9._/-]+$/;
export const skillIndexAuthority = () => indexAuthority(SKILL_INDEX_SEED);

export function parseFrontmatter(text) {
  const m = /^---\r?\n([\s\S]*?)\r?\n---/.exec(text);
  const out = {};
  if (!m) return out;
  for (const line of m[1].split(/\r?\n/)) {
    const kv = /^([A-Za-z_][\w-]*):\s*(.*)$/.exec(line);
    if (kv) out[kv[1]] = kv[2].replace(/^["']|["']$/g, '').trim();
  }
  return out;
}
export function slugify(name) { return String(name || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 64); }

/** files: [{ path, text } | { path, base64 }] -> normalized [{ path, content: Uint8Array }] */
export function normalizeFiles(input) {
  if (!Array.isArray(input) || !input.length) throw httpError('Submit at least SKILL.md.');
  if (input.length > SKILL_LIMITS.maxFiles) throw httpError(`At most ${SKILL_LIMITS.maxFiles} files per skill.`);
  const seen = new Set(); let total = 0;
  const files = input.map(f => {
    const p = String(f?.path || '').trim();
    if (!p || p.length > SKILL_LIMITS.maxPath || !SAFE_PATH.test(p)) throw httpError(`Invalid file path: ${p.slice(0, 60)}`);
    if (seen.has(p)) throw httpError(`Duplicate file: ${p}`);
    seen.add(p);
    const content = typeof f.base64 === 'string' ? new Uint8Array(Buffer.from(f.base64, 'base64')) : new TextEncoder().encode(String(f.text ?? ''));
    total += content.length;
    return { path: p, content };
  }).filter(f => !f.path.endsWith('/metadata.json') && !f.path.endsWith('/verification.json') && f.path !== 'metadata.json' && f.path !== 'verification.json');
  if (total > SKILL_LIMITS.maxTotalBytes) throw httpError(`Skill bundle exceeds ${SKILL_LIMITS.maxTotalBytes / 1000} KB.`);
  if (!files.some(f => f.path === 'SKILL.md')) throw httpError('The bundle must include SKILL.md at its root.');
  return files.sort((a, b) => a.path.localeCompare(b.path));
}

// ---------- store (memory + optional disk) ----------
const store = new Map();
const storeDir = () => process.env.SKILLHUB_STORE_DIR || null;
export function resetSkillStore() { store.clear(); }
async function put(entry) {
  store.set(entry.payload.bundleHash, entry);
  if (store.size > 500) store.delete(store.keys().next().value);
  const dir = storeDir();
  if (dir) {
    await mkdir(dir, { recursive: true });
    const ser = { ...entry, files: entry.files.map(f => ({ path: f.path, base64: Buffer.from(f.content).toString('base64') })) };
    await writeFile(path.join(dir, `${entry.payload.bundleHash.slice(7)}.json`), JSON.stringify(ser));
  }
}
async function get(bundleHash) {
  if (!/^sha256-[0-9a-f]{64}$/.test(bundleHash || '')) return null;
  if (store.has(bundleHash)) return store.get(bundleHash);
  const dir = storeDir();
  if (!dir) return null;
  try {
    const ser = JSON.parse(await readFile(path.join(dir, `${bundleHash.slice(7)}.json`), 'utf8'));
    const entry = { ...ser, files: ser.files.map(f => ({ path: f.path, content: new Uint8Array(Buffer.from(f.base64, 'base64')) })) };
    if (await skillBundleHash(entry.files, sha) !== bundleHash) return null; // tamper check
    store.set(bundleHash, entry); return entry;
  } catch { return null; }
}

/** Hash + JEV-verify a submission. Idempotent per bundleHash (verdict cached). */
export async function submitSkill({ files: input }, opts = {}) {
  const files = normalizeFiles(input);
  const skillMd = new TextDecoder().decode(files.find(f => f.path === 'SKILL.md').content);
  const fm = parseFrontmatter(skillMd);
  const slug = slugify(fm.name);
  if (!slug) throw httpError('SKILL.md needs YAML frontmatter with a name (and description).');
  const bundleHash = await skillBundleHash(files, sha);
  const existing = await get(bundleHash);
  if (existing && existing.payload.slug === slug && isVerdict(existing.verdict)) return view(existing);
  const merkleLeaf = await skillMerkleLeaf(slug, bundleHash, sha);
  const verdict = await verifySkill({ slug, bundleHash, merkleLeaf, frontmatter: fm, files }, opts);
  if (verdict.status === 'rejected' && verdict.precheck && !verdict.precheck.ok && verdict.precheck.problems.some(p => /credential|private key/.test(p))) {
    throw Object.assign(httpError(`Refused: ${verdict.precheck.problems.join('; ')}. Remove secrets before submitting; nothing was stored.`, 422), { verdict });
  }
  const version = `${/^[\w.+-]{1,24}$/.test(fm.version || '') ? fm.version : '1.0.0'}-onchain.${bundleHash.slice(7, 15)}`;
  const payload = {
    slug, name: fm.name, description: (fm.description || '').slice(0, 500), version, bundleHash, merkleLeaf,
    files: files.map(f => ({ path: f.path, bytes: f.content.length, sha256: `sha256-${createHash('sha256').update(f.content).digest('hex')}` })),
    encoding: 'skillhub skill-bundle-v1', submittedVia: SKILL_SOURCE,
  };
  const entry = { payload, verdict, files, frontmatter: fm };
  await put(entry);
  return view(entry);
}
function view(entry) {
  const md = entry.files.find(f => f.path === 'SKILL.md');
  return { ok: true, cluster: CLUSTER, payload: entry.payload, verdict: entry.verdict, skillMd: md ? new TextDecoder().decode(md.content).slice(0, 60000) : null };
}

const skillUri = (baseUrl, asset) => `${baseUrl.replace(/\/$/, '')}/s/${asset}`;
export async function prepareSkillMint({ bundleHash, wallet, asset, bindingSignature, baseUrl = publicBaseUrl() }) {
  if (!BASE58_KEY.test(wallet || '') || !BASE58_KEY.test(asset || '') || wallet === asset) throw httpError('wallet and a fresh asset public key are required.');
  const entry = await get(bundleHash);
  if (!entry) throw httpError('Submit the skill first (or resubmit: this server no longer has it).', 404);
  await assertDevnet();
  const record = buildSkillRecord({ payload: entry.payload, authority: wallet, asset });
  if (!verifyBinding(record, bindingSignature)) throw httpError('Binding signature does not verify for this wallet and bundle hash.', 401);
  const attributeList = skillAttributes(record, bindingSignature, entry.verdict);
  const uri = skillUri(baseUrl, asset);
  const { blockhash, lastValidBlockHeight } = await getUmi().rpc.getLatestBlockhash({ commitment: 'confirmed' });
  const { tx, wire, name } = buildCoreCreateTx({ authority: wallet, asset, names: shrinkingNames(`x/${entry.payload.slug}`), uri, attributeList, updateAuthority: skillIndexAuthority(), blockhash });
  const space = assetSpace(name, uri, attributeList);
  const [fee, rent, balance, payerMin] = await Promise.all([
    devnetRpc('getFeeForMessage', [Buffer.from(tx.serializedMessage).toString('base64'), { commitment: 'confirmed' }]),
    devnetRpc('getMinimumBalanceForRentExemption', [space]), devnetRpc('getBalance', [wallet, { commitment: 'confirmed' }]), devnetRpc('getMinimumBalanceForRentExemption', [0]),
  ]);
  const needed = rent + (fee?.value ?? 10_000), have = balance?.value ?? 0;
  if (have < needed || (have - needed > 0 && have - needed < payerMin)) throw httpError(`Not enough devnet SOL: this mint needs about ${((needed + payerMin) / 1e9).toFixed(4)} SOL; the wallet has ${(have / 1e9).toFixed(4)}. Get devnet SOL at https://faucet.solana.com and try again.`, 402);
  return {
    ok: true, cluster: CLUSTER, transaction: Buffer.from(wire).toString('base64'), blockhash, lastValidBlockHeight, feeLamports: fee?.value ?? null, rentLamportsEstimate: rent,
    name, uri, attributes: attributeList, updateAuthority: skillIndexAuthority(), coreProgram: CORE_PROGRAM, verdict: entry.verdict,
    record: { ...record, attestation: { ...record.attestation, message: bindingMessage(record), signature: bindingSignature } },
  };
}

/** Badge logic: "verified" only when this server holds a typed JEV verdict whose hash equals the onchain jevVerdict. */
export async function summarizeSkillAsset(item) {
  const attrs = attributesToMap(item?.plugins?.attributes?.data?.attribute_list);
  attrs.bundleHash = attrs.bundleHash || attrs.contentHash; // skills: contentHash IS the SkillHub bundleHash
  const record = attrs.contentHash && attrs.authority && attrs.version ? { type: 'skill', contentHash: attrs.contentHash, version: attrs.version, cluster: attrs.cluster, authority: attrs.authority, uniquenessKey: `skill:${(attrs.contentHash || '').replace(/^sha256-/, '')}:${attrs.version}` } : null;
  let signatureValid = false;
  try { signatureValid = !!record && attrs.type === 'skill' && attrs.cluster === 'devnet' && attrs.contentHash === attrs.bundleHash && verifyBinding(record, attrs.bindingSignature); } catch { signatureValid = false; }
  const entry = await get(attrs.bundleHash);
  const verdict = entry && isVerdict(entry.verdict) && entry.verdict.verdictHash === attrs.jevVerdict ? entry.verdict : null;
  const leafOk = attrs.slug && attrs.bundleHash ? (await skillMerkleLeaf(attrs.slug, attrs.bundleHash, sha).catch(() => null)) === attrs.merkleLeaf : false;
  return {
    asset: item.id, name: item.content?.metadata?.name || attrs.slug || null, owner: item.ownership?.owner || null, updateAuthority: item.authorities?.[0]?.address || null,
    slug: attrs.slug || null, version: attrs.version || null, bundleHash: attrs.bundleHash || null, merkleLeaf: attrs.merkleLeaf || null, merkleLeafValid: leafOk,
    authority: attrs.authority || null, signatureValid, claimedStatus: attrs.jev || 'unverified', jevVerdict: attrs.jevVerdict || null,
    status: verdict ? verdict.status : 'unverified', verdictAvailable: !!verdict, verdict: verdict ? { status: verdict.status, outcome: verdict.outcome, reasons: verdict.reasons, checks: verdict.checks, engine: verdict.engine, decidedAt: verdict.decidedAt, verdictHash: verdict.verdictHash } : null,
    description: entry?.payload.description || null, explorer: explorerAddress(item.id), source: attrs.source || null,
  };
}
async function das(method, params) {
  const key = process.env.HELIUS_API_KEY;
  if (!key) throw httpError('Gallery needs HELIUS_API_KEY (devnet DAS) on the server.', 503);
  const res = await fetch(`https://devnet.helius-rpc.com/?api-key=${encodeURIComponent(key)}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ jsonrpc: '2.0', id: 'skillhub', method, params }), signal: AbortSignal.timeout(20000) });
  if (!res.ok) throw httpError(`Helius DAS unavailable (HTTP ${res.status})`, 502);
  const d = await res.json(); if (d.error) throw httpError(`Helius DAS: ${d.error.message}`, 502); return d.result;
}
export async function skillGallery({ page = 1 } = {}) {
  const r = await das('getAssetsByAuthority', { authorityAddress: skillIndexAuthority(), page, limit: 60, sortBy: { sortBy: 'created', sortDirection: 'desc' } });
  const items = [];
  for (const i of r.items || []) if (i.interface === 'MplCoreAsset' && !i.burnt) { const s = await summarizeSkillAsset(i); if (s.source === SKILL_SOURCE) items.push(s); }
  return { ok: true, cluster: CLUSTER, indexAuthority: skillIndexAuthority(), total: r.total ?? items.length, items };
}
export async function skillDetail(asset) {
  if (!BASE58_KEY.test(asset || '')) throw httpError('asset must be a base58 address');
  const item = await das('getAsset', { id: asset });
  if (item?.interface !== 'MplCoreAsset') throw httpError('Not a Core asset on devnet.', 404);
  const s = await summarizeSkillAsset(item);
  const entry = await get(s.bundleHash);
  return { ok: true, cluster: CLUSTER, ...s, immutable: s.updateAuthority === skillIndexAuthority(), files: entry?.payload.files || null, skillMd: entry ? view(entry).skillMd : null };
}
export async function skillMetadata(asset, baseUrl = publicBaseUrl()) {
  if (!BASE58_KEY.test(asset || '')) throw httpError('asset must be a base58 address');
  let a; try { a = await fetchAsset(getUmi(), publicKey(asset)); } catch { throw httpError('Asset not found on devnet.', 404); }
  const attrs = attributesToMap(a.attributes?.attributeList);
  attrs.bundleHash = attrs.bundleHash || attrs.contentHash;
  if (attrs.source !== SKILL_SOURCE || a.updateAuthority?.address !== skillIndexAuthority()) throw httpError('Not an OnchainAI SkillHub asset.', 404);
  const entry = await get(attrs.bundleHash);
  const verdict = entry && isVerdict(entry.verdict) && entry.verdict.verdictHash === attrs.jevVerdict ? entry.verdict : null;
  return {
    name: a.name, symbol: 'SKILL', description: entry?.payload.description || `SkillHub skill ${attrs.slug} (bundle ${attrs.bundleHash}). Solana devnet.`,
    image: `${baseUrl.replace(/\/$/, '')}/brand/hf-model-nft.svg`, external_url: `${baseUrl.replace(/\/$/, '')}/skills?asset=${asset}`,
    attributes: [['slug', attrs.slug], ['bundleHash', attrs.bundleHash], ['merkleLeaf', attrs.merkleLeaf], ['jev', verdict ? verdict.status : 'unverified'], ['cluster', 'devnet']].map(([trait_type, value]) => ({ trait_type, value })),
    properties: { category: 'skill', registry: { schema: 'onchainai-artifact-registry/v1', payload: entry?.payload || null, verdict } },
  };
}
