// Pure, dependency-free port of artifact-registry/src/hash.mjs + record.mjs (model entries only).
// Used by the server (node:crypto) and the browser (WebCrypto) so both compute the same bytes.
// Parity with ../artifact-registry is asserted in tests/hf-mint.test.mjs.

export const REGISTRY_SCHEMA = 'onchainai-artifact-registry/v1';
export const ZK_PROGRAM_ID = '4vJ9JU1bJJE96FWSVKmnrL3xFU5jSBSVdk9x4La2vzhn';
export const ZK_STATUS_NOTE =
  'zk-clawd-primitives is not live: the clawd-zk program is not deployed and all verifying-key slots are empty, so every proof is refused until circuits and a key ceremony exist.';
// The curated registry collection (devnet). Self-serve mints are NOT members; see README "Collection".
export const REGISTRY_COLLECTION = '2Xkbm5YpDqR1hMXiBa3CQaRg7qgvuhUAPj6E4fDCVaPy';
export const SELF_SERVE_SOURCE = 'onchainai-hf/v1';

/** Stable JSON: sorted object keys, no whitespace. Arrays keep order. Identical to hash.mjs canonicalJson. */
export function canonicalJson(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const keys = Object.keys(value).sort();
  return `{${keys.map(k => `${JSON.stringify(k)}:${canonicalJson(value[k])}`).join(',')}}`;
}

export function stripSha256(value) {
  if (typeof value !== 'string') throw new Error('hash must be a string');
  const hex = value.startsWith('sha256-') ? value.slice(7) : value;
  if (!/^[0-9a-f]{64}$/.test(hex)) throw new Error(`invalid sha256: ${value}`);
  return hex;
}

/** UTF-8 preimage bytes for a model payload: "model-v1\0" + canonicalJson(fields). Same as modelPreimage(). */
export function modelPreimageBytes(model) {
  const files = [...(model.files || [])]
    .map(f => ({ path: f.path, sha256: stripSha256(f.sha256), size: f.size ?? null }))
    .sort((a, b) => a.path.localeCompare(b.path));
  const fields = {
    hfRepo: model.hfRepo,
    revisionSha: model.revisionSha,
    files,
    license: model.license ?? null,
    provenance: model.provenance ?? null,
  };
  if (!fields.hfRepo || !fields.revisionSha) throw new Error('model requires hfRepo and revisionSha');
  if (!files.length) throw new Error('model requires at least one file hash');
  return new TextEncoder().encode(`model-v1\0${canonicalJson(fields)}`);
}

export function toHex(bytes) { return Array.from(bytes, b => b.toString(16).padStart(2, '0')).join(''); }

/** sha256Fn: (Uint8Array) => Promise<Uint8Array>|Uint8Array. */
export async function modelContentHash(model, sha256Fn) {
  return `sha256-${toHex(await sha256Fn(modelPreimageBytes(model)))}`;
}

export function uniquenessKey(type, contentHash, version) {
  return `${type}:${stripSha256(contentHash)}:${version}`;
}

export function zkPendingAttestation() {
  return { kind: 'authority-signature', zk: 'pending', zkProgramId: ZK_PROGRAM_ID, vkSlot: null, vkSlotsEmpty: true, programDeployed: false, note: ZK_STATUS_NOTE };
}

/** Same record shape as record.mjs buildRecord() for type=model, cluster=devnet. */
export function buildModelRecord({ payload, contentHash, authority, asset = null, collection = null }) {
  if (!authority) throw new Error('authority required');
  const version = payload.revisionSha;
  return {
    schema: REGISTRY_SCHEMA,
    type: 'model',
    version,
    cluster: 'devnet',
    authority,
    contentHash,
    uniquenessKey: uniquenessKey('model', contentHash, version),
    payload,
    skillMerkleLeaf: null,
    asset,
    collection,
    identity: { standard: 'metaplex-core-asset+registry-record', collection, asset },
    attestation: zkPendingAttestation(),
  };
}

/** Byte-identical to record.mjs bindingMessage(). */
export function bindingMessage(record) {
  return [
    'onchainai.fun artifact-registry wants you to attest a registry record.',
    'This signature only proves the registering authority bound this content hash.',
    'It does not authorize any payment. ZK attestation is pending.',
    '',
    `Type: ${record.type}`,
    `Content-Hash: ${record.contentHash}`,
    `Version: ${record.version}`,
    `Cluster: ${record.cluster}`,
    `Authority: ${record.authority}`,
    `Uniqueness-Key: ${record.uniquenessKey}`,
  ].join('\n');
}

/** record.mjs metaplexAttributes() for models, plus the self-serve binding signature + source marker. */
export function selfServeAttributes(record, bindingSignature) {
  const attrs = [
    { key: 'schema', value: record.schema },
    { key: 'type', value: record.type },
    { key: 'contentHash', value: record.contentHash },
    { key: 'version', value: record.version },
    { key: 'cluster', value: record.cluster },
    { key: 'authority', value: record.authority },
    { key: 'zk', value: record.attestation.zk },
    { key: 'zkProgramId', value: record.attestation.zkProgramId },
    { key: 'vkSlotsEmpty', value: 'true' },
    { key: 'hfRepo', value: record.payload.hfRepo },
    { key: 'revisionSha', value: record.payload.revisionSha },
  ];
  if (record.payload.license) attrs.push({ key: 'license', value: String(record.payload.license).slice(0, 64) });
  attrs.push({ key: 'source', value: SELF_SERVE_SOURCE });
  attrs.push({ key: 'bindingSignature', value: bindingSignature });
  return attrs;
}

export function attributesToMap(list = []) {
  const out = {};
  for (const a of list) if (a && typeof a.key === 'string') out[a.key] = String(a.value ?? '');
  return out;
}

// ---------- Skills (SkillHub encoding) ----------
export const SKILL_SOURCE = 'onchainai-skillhub/v1';
const utf8 = s => new TextEncoder().encode(s);
function concat(chunks) { const n = chunks.reduce((a, c) => a + c.length, 0); const out = new Uint8Array(n); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } return out; }
/** SkillHub hashBundle(): files sorted by path (localeCompare), "skill-bundle-v1\0" + (path\0content\0)*. */
export function skillBundlePreimage(files) {
  const sorted = [...files].sort((a, b) => a.path.localeCompare(b.path));
  const chunks = [utf8('skill-bundle-v1\0')];
  for (const f of sorted) chunks.push(utf8(f.path), utf8('\0'), typeof f.content === 'string' ? utf8(f.content) : f.content, utf8('\0'));
  return concat(chunks);
}
export async function skillBundleHash(files, sha256Fn) { return `sha256-${toHex(await sha256Fn(skillBundlePreimage(files)))}`; }
/** SkillHub merkleLeaf = sha256(slug \0 bundleHash) with the sha256- prefix inside the preimage. */
export async function skillMerkleLeaf(slug, bundleHash, sha256Fn) { stripSha256(bundleHash); return `sha256-${toHex(await sha256Fn(utf8(`${slug}\0${bundleHash}`)))}`; }
/** Same record as record.mjs buildRecord() for type=skill (contentHash = bundleHash). */
export function buildSkillRecord({ payload, authority, asset = null, collection = null }) {
  if (!authority) throw new Error('authority required');
  const contentHash = payload.bundleHash;
  return {
    schema: REGISTRY_SCHEMA, type: 'skill', version: payload.version, cluster: 'devnet', authority, contentHash,
    uniquenessKey: uniquenessKey('skill', contentHash, payload.version), payload, skillMerkleLeaf: payload.merkleLeaf,
    asset, collection, identity: { standard: 'metaplex-core-asset+registry-record', collection, asset }, attestation: zkPendingAttestation(),
  };
}
/**
 * record.mjs metaplexAttributes() for skills + source, binding signature and the JEV verdict (status + verdict hash).
 * To fit Solana's 1232-byte tx limit this omits three registry attributes that carry no extra information:
 * bundleHash (identical to contentHash for skills), and the constants zkProgramId / vkSlotsEmpty (see ZK_PROGRAM_ID;
 * they remain in the record JSON served at /s/<asset>).
 */
export function skillAttributes(record, bindingSignature, verdict) {
  return [
    { key: 'schema', value: record.schema }, { key: 'type', value: record.type }, { key: 'contentHash', value: record.contentHash },
    { key: 'version', value: record.version }, { key: 'cluster', value: record.cluster }, { key: 'authority', value: record.authority },
    { key: 'zk', value: record.attestation.zk }, { key: 'slug', value: record.payload.slug }, { key: 'merkleLeaf', value: record.skillMerkleLeaf },
    { key: 'source', value: SKILL_SOURCE }, { key: 'bindingSignature', value: bindingSignature },
    { key: 'jev', value: verdict?.status || 'unverified' }, { key: 'jevVerdict', value: verdict?.verdictHash || 'none' },
  ];
}
