import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import nacl from 'tweetnacl';
import bs58 from 'bs58';
import { canonicalJson, modelContentHash, bindingMessage, buildModelRecord, selfServeAttributes, SELF_SERVE_SOURCE } from '../src/registry-hash.js';
import { inspectWire, withSignature } from '../src/tx-inspect.js';
import * as mint from '../server/hf-mint.mjs';
import handler from '../api/router.mjs';

const sha = b => new Uint8Array(createHash('sha256').update(b).digest());
const REG = new URL('../../artifact-registry/', import.meta.url);
const hasRegistry = existsSync(new URL('src/hash.mjs', REG));
const REV = 'aeb339dc7de361b468683b1242729c701cba465e';

test('content hash and binding message are byte-identical to artifact-registry', { skip: !hasRegistry && 'artifact-registry not present' }, async () => {
  const hash = await import(new URL('src/hash.mjs', REG));
  const rec = await import(new URL('src/record.mjs', REG));
  for (const f of ['hauhau-qwen36-onchain.json', 'hauhau-qwen36-uncensored.json']) {
    const ex = JSON.parse(readFileSync(new URL(`examples/${f}`, REG), 'utf8'));
    assert.equal(canonicalJson(ex.payload), hash.canonicalJson(ex.payload));
    const ours = await modelContentHash(ex.payload, sha);
    assert.equal(ours, hash.modelContentHash(ex.payload));
    const authority = bs58.encode(nacl.sign.keyPair().publicKey);
    const theirs = rec.buildRecord({ type: 'model', version: ex.version, authority, payload: ex.payload, cluster: 'devnet' });
    const mine = buildModelRecord({ payload: ex.payload, contentHash: ours, authority });
    assert.equal(mine.uniquenessKey, theirs.uniquenessKey);
    assert.equal(bindingMessage(mine), rec.bindingMessage(theirs));
    assert.deepEqual(selfServeAttributes(mine, 'sig').slice(0, 11), rec.metaplexAttributes(theirs).slice(0, 11));
  }
});

test('parseModelId accepts ids and URLs, rejects datasets/spaces/junk', () => {
  assert.equal(mint.parseModelId('Qwen/Qwen2.5-7B-Instruct'), 'Qwen/Qwen2.5-7B-Instruct');
  assert.equal(mint.parseModelId('https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/main'), 'meta-llama/Llama-3.1-8B-Instruct');
  assert.equal(mint.parseModelId('hf.co/ordlibrary/hauhau-qwen36-onchain?x=1'), 'ordlibrary/hauhau-qwen36-onchain');
  assert.throws(() => mint.parseModelId('https://huggingface.co/datasets/x/y'), /dataset or Space/);
  assert.throws(() => mint.parseModelId('spaces/x/y'), /dataset or Space/);
  assert.throws(() => mint.parseModelId('../etc/passwd'), /valid/);
  assert.throws(() => mint.parseModelId('a/b/c'), /valid/);
  assert.throws(() => mint.parseModelId(''), /Paste/);
});

function mockHub({ private: priv = false, gated = false } = {}) {
  const calls = [];
  const original = globalThis.fetch;
  globalThis.fetch = async (url, init = {}) => {
    const u = String(url); calls.push({ u, auth: init.headers?.Authorization || null });
    if (u.includes('/api/models/acme/tiny')) return new Response(JSON.stringify({
      id: 'acme/tiny', author: 'acme', sha: REV, private: priv, gated, downloads: 5, likes: 2, tags: ['license:apache-2.0', 'text-generation'],
      pipeline_tag: 'text-generation', library_name: 'transformers', cardData: { license: 'apache-2.0', base_model: 'acme/base' },
      siblings: [
        { rfilename: 'model.safetensors', blobId: 'b1', size: 999, lfs: { sha256: 'a'.repeat(64), size: 999 } },
        { rfilename: 'config.json', blobId: 'b2', size: 2 },
        { rfilename: 'README.md', blobId: 'b3', size: 5 },
      ],
    }), { status: 200 });
    if (u.endsWith(`/resolve/${REV}/config.json`)) return gated ? new Response('', { status: 403 }) : new Response('{}', { status: 200 });
    if (u.endsWith(`/resolve/${REV}/README.md`)) return new Response('hello', { status: 200 });
    if (u.includes('api.devnet') || u.includes('devnet')) {
      const body = JSON.parse(init.body || '{}');
      if (body.method === 'getGenesisHash') return new Response(JSON.stringify({ jsonrpc: '2.0', id: 1, result: mint.DEVNET_GENESIS }));
    }
    return new Response('not mocked', { status: 599 });
  };
  return { calls, restore: () => { globalThis.fetch = original; } };
}

test('fetchHfModel pins the commit, hashes LFS + small files, keeps token server-side', async () => {
  mint.resetHfCache(); process.env.HF_TOKEN = 'hf_test_server_only_token';
  const hub = mockHub();
  try {
    const m = await mint.fetchHfModel('acme/tiny');
    assert.equal(m.payload.revisionSha, REV);
    assert.deepEqual(m.payload.files.map(f => f.path), ['config.json', 'model.safetensors', 'README.md'].sort((a, b) => a.localeCompare(b)));
    assert.equal(m.payload.files.find(f => f.path === 'README.md').sha256, `sha256-${createHash('sha256').update('hello').digest('hex')}`);
    assert.equal(m.payload.files.find(f => f.path === 'model.safetensors').sha256, `sha256-${'a'.repeat(64)}`);
    assert.equal(m.contentHash, await modelContentHash(m.payload, sha));
    assert.equal(m.payload.license, 'apache-2.0');
    assert.ok(hub.calls.every(c => c.auth === 'Bearer hf_test_server_only_token'));
    assert.ok(!JSON.stringify(m).includes('hf_test_server_only_token'));
  } finally { hub.restore(); delete process.env.HF_TOKEN; }
});

test('private repos are refused; gated files become recorded-unhashed provenance', async () => {
  mint.resetHfCache();
  let hub = mockHub({ private: true });
  try { await assert.rejects(mint.fetchHfModel('acme/tiny'), /private/); } finally { hub.restore(); }
  mint.resetHfCache();
  hub = mockHub({ gated: 'manual' });
  try {
    const m = await mint.fetchHfModel('acme/tiny');
    assert.deepEqual(m.payload.provenance.unhashedFiles, [{ path: 'config.json', gitBlobId: 'b2', size: 2, reason: 'gated-or-forbidden' }]);
    assert.equal(m.payload.files.length, 2);
  } finally { hub.restore(); }
});

test('prepareMint refuses a binding signature from another wallet', async () => {
  mint.resetHfCache();
  const hub = mockHub();
  try {
    const wallet = nacl.sign.keyPair(), other = nacl.sign.keyPair(), asset = nacl.sign.keyPair();
    const m = await mint.fetchHfModel('acme/tiny');
    const rec = buildModelRecord({ payload: m.payload, contentHash: m.contentHash, authority: bs58.encode(wallet.publicKey) });
    const badSig = bs58.encode(nacl.sign.detached(new TextEncoder().encode(bindingMessage(rec)), other.secretKey));
    await assert.rejects(mint.prepareMint({ modelId: 'acme/tiny', revision: REV, wallet: bs58.encode(wallet.publicKey), asset: bs58.encode(asset.publicKey), bindingSignature: badSig }), /Binding signature/);
    await assert.rejects(mint.prepareMint({ modelId: 'acme/tiny', revision: 'main', wallet: bs58.encode(wallet.publicKey), asset: bs58.encode(asset.publicKey), bindingSignature: badSig }), /40-hex/);
  } finally { hub.restore(); }
});

test('mainnet RPC configuration is refused', async () => {
  process.env.SOLANA_DEVNET_RPC_URL = 'https://api.mainnet-beta.solana.com';
  try { await assert.rejects(mint.devnetRpc('getGenesisHash'), /mainnet/); } finally { delete process.env.SOLANA_DEVNET_RPC_URL; }
});

test('unsigned create tx: wallet pays, only Core is called, asset co-signs, frozen update authority, fits a long repo id', async () => {
  const wallet = nacl.sign.keyPair(), asset = nacl.sign.keyPair();
  const w = bs58.encode(wallet.publicKey), a = bs58.encode(asset.publicKey);
  const long = `${'o'.repeat(42)}/${'m'.repeat(96)}`; // HF max name length 96
  const payload = { hfRepo: long, revisionSha: REV, license: 'apache-2.0', files: [{ path: 'x', sha256: 'sha256-' + 'b'.repeat(64), size: 1 }], provenance: { source: 'huggingface-hub', registrar: SELF_SERVE_SOURCE, unhashedFiles: [], note: 'n' } };
  const contentHash = await modelContentHash(payload, sha);
  const record = buildModelRecord({ payload, contentHash, authority: w, asset: a });
  const sig = bs58.encode(nacl.sign.detached(new TextEncoder().encode(bindingMessage(record)), wallet.secretKey));
  assert.ok(mint.verifyBinding(record, sig));
  const { wire, attributeList } = mint.buildCreateTx({ record, bindingSignature: sig, asset: a, blockhash: bs58.encode(new Uint8Array(32).fill(7)) });
  assert.ok(wire.length <= 1232, `tx ${wire.length} bytes`);
  const info = inspectWire(wire, bs58.encode);
  assert.equal(info.feePayer, w);
  assert.deepEqual(info.programs, [mint.CORE_PROGRAM]);
  assert.deepEqual([...info.signers].sort(), [w, a].sort());
  assert.ok(info.keys.includes(mint.indexAuthority()));
  assert.ok(Buffer.from(info.message).includes(Buffer.from(contentHash)));
  assert.equal(attributeList.find(x => x.key === 'bindingSignature').value, sig);
  // signing round-trip used by the browser
  let signed = wire;
  for (const [k, kp] of [[w, wallet], [a, asset]]) signed = withSignature(signed, info, k, nacl.sign.detached(info.message, kp.secretKey));
  const after = inspectWire(signed, bs58.encode);
  assert.deepEqual(Buffer.from(after.message), Buffer.from(info.message));
});

test('gallery summary only trusts a valid binding signature', () => {
  const kp = nacl.sign.keyPair(); const authority = bs58.encode(kp.publicKey);
  const contentHash = 'sha256-' + 'c'.repeat(64);
  const rec = { type: 'model', contentHash, version: REV, cluster: 'devnet', authority, uniquenessKey: `model:${'c'.repeat(64)}:${REV}` };
  const good = bs58.encode(nacl.sign.detached(new TextEncoder().encode(bindingMessage(rec)), kp.secretKey));
  const item = sig => ({ id: 'Asset1111111111111111111111111111111111111', interface: 'MplCoreAsset', ownership: { owner: authority }, authorities: [{ address: mint.indexAuthority() }], content: { metadata: { name: 'tiny' } },
    plugins: { attributes: { data: { attribute_list: [['schema', 'onchainai-artifact-registry/v1'], ['type', 'model'], ['contentHash', contentHash], ['version', REV], ['cluster', 'devnet'], ['authority', authority], ['hfRepo', 'acme/tiny'], ['revisionSha', REV], ['source', SELF_SERVE_SOURCE], ['bindingSignature', sig]].map(([key, value]) => ({ key, value })) } } } });
  assert.equal(mint.summarizeAsset(item(good)).signatureValid, true);
  assert.equal(mint.summarizeAsset(item(bs58.encode(new Uint8Array(64)))).signatureValid, false);
});

test('config endpoint reports devnet and never leaks secret values', async () => {
  process.env.HF_TOKEN = 'hf_secret_value_should_not_leak'; process.env.HELIUS_API_KEY = 'helius-secret-should-not-leak';
  try {
    let text, status = 200; const res = { setHeader() {}, end(v) { text = v; }, set statusCode(v) { status = v; }, get statusCode() { return status; } };
    await handler({ url: '/api/hf-mint/config', method: 'GET', headers: {} }, res);
    const d = JSON.parse(text);
    assert.equal(d.cluster, 'devnet'); assert.equal(d.mainnet, 'refused'); assert.equal(d.hfTokenConfigured, true); assert.equal(d.collection, null);
    assert.ok(!text.includes('hf_secret_value') && !text.includes('helius-secret'));
  } finally { delete process.env.HF_TOKEN; delete process.env.HELIUS_API_KEY; }
});

test('live Hugging Face: hauhau file hashes match the curated registry entry; big public models hash', { skip: !process.env.HF_LIVE && 'set HF_LIVE=1 (uses HF_TOKEN)' }, async () => {
  mint.resetHfCache();
  const m = await mint.fetchHfModel('ordlibrary/hauhau-qwen36-onchain', { revision: REV });
  if (hasRegistry) {
    const ex = JSON.parse(readFileSync(new URL('examples/hauhau-qwen36-onchain.json', REG), 'utf8'));
    assert.deepEqual(m.payload.files, ex.payload.files);
  }
  for (const id of ['Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct']) {
    const big = await mint.fetchHfModel(id);
    assert.ok(big.payload.files.length > 5, id);
    assert.equal(big.contentHash, await modelContentHash(big.payload, sha));
    assert.ok(big.preview.parameters > 1e9, id);
  }
});
