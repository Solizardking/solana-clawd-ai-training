import test from 'node:test';
import assert from 'node:assert/strict';
import nacl from 'tweetnacl';
import bs58 from 'bs58';
import handler from '../api/router.mjs';
import { sha256, verifyWallet, scanSecrets, publishDataset } from '../server/service.mjs';

test('publication authorization binds wallet, domain, action and complete dataset digest', () => {
  const key = nacl.sign.keyPair();
  const wallet = bs58.encode(key.publicKey), timestamp = Date.now(), domain = 'www.onchainai.fund', digest = sha256('test dataset');
  const message = `Onchain AI\nDomain: ${domain}\nAction: publish-dataset\nWallet: ${wallet}\nDigest: ${digest}\nTimestamp: ${timestamp}`;
  const signature = bs58.encode(nacl.sign.detached(new TextEncoder().encode(message), key.secretKey));
  const authorization = { wallet, signature, timestamp, domain };
  assert.equal(verifyWallet(authorization, 'publish-dataset', digest), message);
  assert.throws(() => verifyWallet(authorization, 'publish-dataset', sha256('different dataset')), /Invalid wallet signature/);
  assert.throws(() => verifyWallet({ ...authorization, timestamp: timestamp - 600000 }, 'publish-dataset', digest), /expired/);
  assert.throws(() => verifyWallet({ ...authorization, domain: 'malicious.example' }, 'publish-dataset', digest), /domain invalid/);
});
test('likely credentials are rejected before any provider write', async () => {
  const token = 'hf_' + 'x'.repeat(30);
  assert.equal(scanSecrets(token), true);
  assert.equal(scanSecrets('Solana uses PDAs'), false);
  await assert.rejects(publishDataset({ repo_id: 'fixture/test', private: true, train_jsonl: JSON.stringify({ messages: [{ role: 'user', content: token }] }), manifest: {} }), /Likely secret detected/);
});
test('malformed dataset rows cannot bypass browser parsing', async () => {
  await assert.rejects(publishDataset({ repo_id: 'fixture/test', private: true, train_jsonl: '{"messages":[{"role":"user","content":123}]}', manifest: {} }), /valid SFT/);
});
async function call(url, body) {
  let text; const res = { setHeader() {}, end(value) { text = value; }, statusCode: 200 };
  await handler({ url, method: 'POST', headers: { 'content-type': 'application/json' }, body }, res);
  return { status: res.statusCode, data: JSON.parse(text) };
}
test('RPC proxy rejects unknown operations and arbitrary clusters', async () => {
  assert.equal((await call('/api/rpc', { method: 'requestAirdrop', cluster: 'devnet', params: [] })).status, 400);
  assert.equal((await call('/api/rpc', { method: 'getBalance', cluster: 'malicious', params: [] })).status, 400);
});
test('registration previews never claim a write; price prediction is gated', async () => {
  const result = await call('/api/register', { hf_model_id: 'fixture/model' });
  assert.equal(result.data.posted, false); assert.equal(result.data.dry_run, true);
  assert.equal((await call('/api/register', { live: true })).status, 409);
  assert.equal((await call('/api/register', { model_type: 'PricePrediction' })).status, 403);
});
test('multipart publication works when the platform already buffered the request', async () => {
  const form = new FormData();
  form.set('payload', JSON.stringify({ repo_id: 'fixture/test', train_jsonl: '{"messages":[{"role":"user","content":"Solana"}]}', manifest: {}, private: true }));
  form.set('hf_token', 'hf_fixture_not_a_real_token');
  const request = new Request('https://localhost/', { method: 'POST', body: form });
  const body = Buffer.from(await request.arrayBuffer());
  let text; const res = { setHeader() {}, end(value) { text = value; }, statusCode: 200 };
  await handler({ url: '/api/training/datasets', method: 'POST', headers: { 'content-type': request.headers.get('content-type') }, body }, res);
  assert.equal(res.statusCode, 401);
  assert.match(JSON.parse(text).error, /Wallet authorization/);
  assert.equal(text.includes('hf_fixture'), false);
});
