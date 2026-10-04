import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import nacl from 'tweetnacl';
import bs58 from 'bs58';
import { inflateSync } from 'node:zlib';
import { createRepo, uploadFiles, whoAmI } from '@huggingface/hub';

export const PROGRAM_ID = '3dLst2E3djtCSwG19mFS3REHxtZPngjyga7iYZLDL5xj';
export const MEMO_PROGRAM = 'MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr';
export const IDL_ADDRESS = '5cqcAoyYWkWe3Yq6mXwVjVvV5j3EjDPo2N5P6EvSrvEF';
export const sha256 = text => createHash('sha256').update(text).digest('hex');
let hubCache;
let inferenceCache;
export async function hubCatalog() {
  if (hubCache && Date.now() - hubCache.at < 300000) return hubCache.data;
  const snapshot = JSON.parse(await readFile(new URL('../hub-catalog.json', import.meta.url), 'utf8'));
  try {
    const responses = await Promise.all(['models', 'datasets', 'spaces'].map(async kind => {
      const inventories = await Promise.all(['solanaclawd', 'ordlibrary'].map(async author => {
        const r = await fetch(`https://huggingface.co/api/${kind}?author=${author}&limit=1000&full=true`, { signal: AbortSignal.timeout(12000) });
        if (!r.ok) throw new Error(`Hub ${kind}: HTTP ${r.status}`);
        return r.json();
      }));
      return [kind, inventories.flat().map(item => ({ id: item.id, kind: kind.slice(0, -1), sha: item.sha || null, downloads: item.downloads || 0, likes: item.likes || 0, pipeline_tag: item.pipeline_tag || null, tags: item.tags || [], url: `https://huggingface.co/${kind === 'models' ? '' : `${kind}/`}${item.id}`, files: (item.siblings || []).map(f => f.rfilename), card: { license: item.cardData?.license || null, base_model: item.cardData?.base_model || null } }))];
    }));
    const data = { ...Object.fromEntries(responses), updated_at: new Date().toISOString(), source: 'Hugging Face public APIs: solanaclawd and ordlibrary', live: true, degraded: false };
    hubCache = { at: Date.now(), data }; return data;
  } catch {
    return { ...snapshot, live: false, degraded: true, error: 'Hub metadata temporarily unavailable; showing the dated release snapshot.' };
  }
}
export async function rpc(cluster, method, params = []) {
  if (!['devnet', 'mainnet-beta'].includes(cluster)) throw Object.assign(new Error('Unsupported cluster'), { status: 400 });
  const endpoint = cluster === 'devnet' ? (process.env.SOLANA_DEVNET_RPC_URL || 'https://api.devnet.solana.com') : (process.env.SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com');
  const response = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json', 'User-Agent': 'OnchainAI/0.2' }, body: JSON.stringify({ jsonrpc: '2.0', id: 1, method, params }), signal: AbortSignal.timeout(18000) });
  if (!response.ok) throw Object.assign(new Error(`Solana RPC unavailable (HTTP ${response.status})`), { status: 502 });
  const data = await response.json();
  if (data.error) throw Object.assign(new Error(`Solana RPC: ${data.error.message}`), { status: 502 });
  return data.result;
}
export async function registryStatus(cluster) {
  const [slot, program, account] = await Promise.all([rpc(cluster, 'getSlot', [{ commitment: 'confirmed' }]), rpc(cluster, 'getAccountInfo', [PROGRAM_ID, { encoding: 'base64', commitment: 'confirmed' }]), rpc(cluster, 'getAccountInfo', [IDL_ADDRESS, { encoding: 'base64', commitment: 'confirmed' }])]);
  let idlMatches = false;
  try {
    const expected = JSON.parse(await readFile(new URL('../registry-idl.json', import.meta.url), 'utf8'));
    const bytes = Buffer.from(account.value.data[0], 'base64');
    const live = JSON.parse(inflateSync(bytes.subarray(44, 44 + bytes.readUInt32LE(40))));
    idlMatches = account.value.owner === PROGRAM_ID && JSON.stringify(live) === JSON.stringify(expected);
  } catch { /* Missing or changed IDL disables native registration. */ }
  return { ok: true, cluster, slot, checked_at: new Date().toISOString(), program_id: PROGRAM_ID, program_executable: !!program.value?.executable, idl_address: IDL_ADDRESS, idl_matches: idlMatches, native_registration_enabled: cluster === 'devnet' && !!program.value?.executable && idlMatches, memo_commitment_enabled: true };
}
export async function decodeRegistryAccount(pda, value) {
  if (!value || value.owner !== PROGRAM_ID || value.executable) return null;
  const idl = JSON.parse(await readFile(new URL('../registry-idl.json', import.meta.url), 'utf8'));
  const expected = idl.accounts.find(a => a.name === 'ModelRegistry').discriminator;
  const bytes = Buffer.from(value.data[0], 'base64');
  if (bytes.length < 8 || !bytes.subarray(0, 8).equals(Buffer.from(expected))) return null;
  let offset = 8;
  const key = () => { const result = bs58.encode(bytes.subarray(offset, offset + 32)); offset += 32; return result; };
  const string = () => { const size = bytes.readUInt32LE(offset); offset += 4; if (size > 4096 || offset + size > bytes.length) throw new Error('Invalid registry account string'); const result = bytes.subarray(offset, offset + size).toString(); offset += size; return result; };
  const authority = key(), model_hash = string(), type = bytes[offset++], api_endpoint = string();
  return { pda, authority, model_hash, model_type: ['SentimentAnalysis', 'TextGeneration', 'ImageClassification', 'PricePrediction', 'DocumentUnderstanding'][type] || 'Unknown', api_endpoint, onchain_verified: true, proof_type: 'ModelRegistry PDA', cluster: 'devnet' };
}
export async function onchainRegistry() {
  const idl = JSON.parse(await readFile(new URL('../registry-idl.json', import.meta.url), 'utf8'));
  const discriminator = idl.accounts.find(a => a.name === 'ModelRegistry').discriminator;
  const accounts = await rpc('devnet', 'getProgramAccounts', [PROGRAM_ID, { encoding: 'base64', commitment: 'confirmed', filters: [{ memcmp: { offset: 0, bytes: bs58.encode(Uint8Array.from(discriminator)) } }] }]);
  const registry = await Promise.all(accounts.slice(0, 100).map(a => decodeRegistryAccount(a.pubkey, a.account)));
  return { registry: registry.filter(Boolean), total_accounts: accounts.length, truncated: accounts.length > 100 };
}
export function scanSecrets(text) {
  return /hf_[A-Za-z0-9]{24,}|(?:sk-|nvapi-)[A-Za-z0-9_-]{20,}|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|(?:api_key|private_key|secret_key)\s*[=:]\s*["']?[A-Za-z0-9_-]{20,}/i.test(text);
}
export function verifyWallet(request, action, digest) {
  const { wallet, signature, timestamp, domain } = request;
  if (!Number.isInteger(timestamp) || Math.abs(Date.now() - timestamp) > 300000 || !['onchainai.fund', 'www.onchainai.fund', 'localhost', '127.0.0.1'].includes(domain) && !/^[a-z0-9-]+\.vercel\.app$/.test(domain || '')) throw Object.assign(new Error('Wallet authorization expired or domain invalid'), { status: 401 });
  const message = `Onchain AI\nDomain: ${domain}\nAction: ${action}\nWallet: ${wallet}\nDigest: ${digest}\nTimestamp: ${timestamp}`;
  try {
    if (!nacl.sign.detached.verify(new TextEncoder().encode(message), bs58.decode(signature), bs58.decode(wallet))) throw new Error();
  } catch { throw Object.assign(new Error('Invalid wallet signature'), { status: 401 }); }
  return message;
}
export async function publishDataset(body) {
  const { repo_id, train_jsonl, manifest, private: isPrivate = true } = body;
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo_id || '') || typeof train_jsonl !== 'string' || train_jsonl.length > 2000000 || !manifest || typeof isPrivate !== 'boolean') throw Object.assign(new Error('Invalid dataset publication request (2 MB maximum)'), { status: 400 });
  if (scanSecrets(train_jsonl) || scanSecrets(JSON.stringify(manifest))) throw Object.assign(new Error('Likely secret detected; remove it before publishing'), { status: 400 });
  const lines = train_jsonl.trim().split('\n');
  if (!lines.length || lines.some(line => { try { const row = JSON.parse(line); return !Array.isArray(row.messages) || !row.messages.length || row.messages.some(m => !['system', 'user', 'assistant'].includes(m.role) || typeof m.content !== 'string'); } catch { return true; } })) throw Object.assign(new Error('Dataset must contain valid SFT message rows'), { status: 400 });
  const digest = sha256(JSON.stringify({ repo_id, train_jsonl, manifest, private: isPrivate }));
  verifyWallet(body, 'publish-dataset', digest);
  let token = body.hf_token;
  delete body.hf_token;
  if (typeof token !== 'string' || !token.startsWith('hf_')) throw Object.assign(new Error('A request-scoped Hugging Face write token is required'), { status: 401 });
  try {
    const identity = await whoAmI({ accessToken: token });
    const namespace = repo_id.split('/')[0];
    if (identity.name !== namespace && !(identity.orgs || []).some(org => org.name === namespace)) throw Object.assign(new Error('Token does not belong to the requested namespace'), { status: 403 });
    // New repository only: never overwrite an existing user dataset or change its visibility.
    await createRepo({ repo: { type: 'dataset', name: repo_id }, accessToken: token, private: isPrivate });
    const normalizedManifest = { ...manifest, wallet: body.wallet, dataset_sha256: sha256(train_jsonl), example_count: lines.length };
    const commit = await uploadFiles({ repo: { type: 'dataset', name: repo_id }, accessToken: token, commitTitle: 'Publish wallet-authored Onchain AI dataset', files: [
      { path: 'data/train.jsonl', content: new Blob([train_jsonl]) },
      { path: 'manifest.json', content: new Blob([JSON.stringify(normalizedManifest, null, 2)]) },
      { path: 'README.md', content: new Blob([`---\ntask_categories:\n- text-generation\n---\n# ${repo_id}\n\nWallet-authored SFT dataset published from Onchain AI.\n\nExamples: ${lines.length}\n\nSHA-256: ${normalizedManifest.dataset_sha256}\n\nPublisher wallet: ${body.wallet}\n\nPublication is not an onchain registration or quality attestation.\n`]) },
    ] });
    return { ok: true, repo_id, commit, manifest: normalizedManifest, manifest_sha256: sha256(JSON.stringify(normalizedManifest)), url: `https://huggingface.co/datasets/${repo_id}` };
  } finally { token = undefined; }
}
const routerBase = () => (process.env.INFERENCE_BASE_URL || 'https://clawdrouter-zk.fly.dev/v1').replace(/\/$/, '');
const CORE_SPACE = 'https://solanaclawd-clawd-free-chat.hf.space';
const CORE_MODEL = 'solanaclawd/solana-clawd-core-ai-1.5b-lora';
export async function inferenceModels() {
  if (inferenceCache && Date.now() - inferenceCache.at < 60000) return inferenceCache.data;
  const models = []; const errors = [];
  try {
    const response = await fetch(`${routerBase()}/models`, { signal: AbortSignal.timeout(10000), headers: { 'User-Agent': 'OnchainAI/0.2', ...(process.env.INFERENCE_API_KEY ? { Authorization: `Bearer ${process.env.INFERENCE_API_KEY}` } : {}) } });
    if (!response.ok) throw new Error();
    const result = await response.json();
    models.push(...(result.data || result.models || []).map(m => ({ id: m.id || m.name, provider: 'ClawdRouter', base_url: routerBase(), free: m.id === 'zkrouter/auto' || m.free === true || m.cost_class === 'free' || m.pricing && Number(m.pricing.prompt) === 0 && Number(m.pricing.completion) === 0 })).filter(m => m.id));
  } catch { errors.push('ClawdRouter is unavailable.'); }
  try {
    const response = await fetch(`${CORE_SPACE}/healthz`, { signal: AbortSignal.timeout(10000) });
    const health = await response.json();
    if (response.ok && health.status === 'ok' && health.core_ai_model === CORE_MODEL) models.push({ id: CORE_MODEL, label: 'Clawd Core AI 1.5B · HF Space · free', provider: 'Hugging Face Core AI Space', base_url: `${CORE_SPACE}/api/free/chat`, free: true, loaded: health.core_ai_loaded });
    else errors.push('Core AI Space is unavailable.');
  } catch { errors.push('Core AI Space is unavailable.'); }
  const data = { ok: models.some(m => m.free), models, errors, error: errors.join(' '), timestamp: new Date().toISOString() };
  inferenceCache = { at: Date.now(), data }; return data;
}
export async function complete(body) {
  if (!Array.isArray(body.messages) || body.messages.length > 30 || JSON.stringify(body.messages).length > 24000 || body.messages.some(m => !['user', 'system', 'assistant'].includes(m.role) || typeof m.content !== 'string')) throw Object.assign(new Error('Invalid chat messages'), { status: 400 });
  const inventory = await inferenceModels();
  if (!inventory.ok) throw Object.assign(new Error(inventory.error), { status: 503 });
  const model = inventory.models.find(m => m.id === body.model);
  if (!model?.free) throw Object.assign(new Error('Only verified free routes are enabled. This request will not fall back to a paid model.'), { status: 403 });
  if (model.provider === 'Hugging Face Core AI Space') {
    const response = await fetch(model.base_url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: model.id, messages: body.messages, max_tokens: Math.max(8, Math.min(Number(body.max_tokens) || 64, 128)), stream: true }), signal: AbortSignal.timeout(50000) });
    if (!response.ok) throw Object.assign(new Error(`Core AI Space failed (HTTP ${response.status})`), { status: 502 });
    let text = '', pending = '', done = false, reportedModel = null;
    const reader = response.body.getReader(), decoder = new TextDecoder();
    while (true) {
      const chunk = await reader.read(); if (chunk.done) break;
      pending += decoder.decode(chunk.value, { stream: true });
      const lines = pending.split('\n'); pending = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const value = line.slice(5).trim(); if (value === '[DONE]') { done = true; continue; }
        let item; try { item = JSON.parse(value); } catch { throw Object.assign(new Error('Core AI Space returned malformed streaming data'), { status: 502 }); }
        if (item.error) throw Object.assign(new Error('Core AI Space reported an inference error'), { status: 502 });
        text += item.choices?.[0]?.delta?.content || ''; reportedModel = item.model || reportedModel;
      }
    }
    if (!done || !text.trim() || /preview is temporarily unavailable|model returned an empty response/i.test(text)) throw Object.assign(new Error('Core AI Space failed to complete generation. Try again after the model warms.'), { status: 503 });
    return { choices: [{ message: { role: 'assistant', content: text } }], model: reportedModel, evidence: { provider: model.provider, requested_model: model.id, configured_model: model.id, model: reportedModel, model_identity_source: reportedModel ? 'response' : 'Space health endpoint and public app configuration; response has no model field', base_url: model.base_url, receipt_present: false, receipt: null, cost_class: 'free', fallback_used: false, provider_errors: inventory.errors, timestamp: new Date().toISOString() } };
  }
  const response = await fetch(`${routerBase()}/chat/completions`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...(process.env.INFERENCE_API_KEY ? { Authorization: `Bearer ${process.env.INFERENCE_API_KEY}` } : {}) }, body: JSON.stringify({ model: body.model, messages: body.messages, max_tokens: Math.max(32, Math.min(Number(body.max_tokens) || 512, 2048)), stream: false }), signal: AbortSignal.timeout(50000) });
  if (!response.ok) throw Object.assign(new Error(`Inference failed (HTTP ${response.status})`), { status: 502 });
  const result = await response.json();
  if (typeof result.choices?.[0]?.message?.content !== 'string') throw Object.assign(new Error('Router returned a malformed completion'), { status: 502 });
  return { ...result, evidence: { provider: model.provider, requested_model: body.model, model: result.model || null, base_url: model.base_url, receipt_present: !!(result.receipt || result.zk_receipt), receipt: result.receipt || result.zk_receipt || null, cost_class: 'free', fallback_used: !!result.model && result.model !== body.model, timestamp: new Date().toISOString() } };
}
