import { hubCatalog, rpc, registryStatus, onchainRegistry, decodeRegistryAccount, complete, inferenceModels, publishDataset, PROGRAM_ID, MEMO_PROGRAM, sha256 } from '../server/service.mjs';
import bs58 from 'bs58';

async function readBody(req) {
  if (req.body && typeof req.body === 'object' && !Buffer.isBuffer(req.body)) return req.body;
  if (typeof req.body === 'string' && !(req.headers['content-type'] || '').includes('multipart')) return JSON.parse(req.body);
  const chunks = []; let size = 0;
  if (Buffer.isBuffer(req.body) || typeof req.body === 'string') { chunks.push(Buffer.from(req.body)); size = chunks[0].length; }
  else for await (const chunk of req) { size += chunk.length; if (size > 3000000) throw Object.assign(new Error('Request exceeds 3 MB'), { status: 413 }); chunks.push(chunk); }
  if (size > 3000000) throw Object.assign(new Error('Request exceeds 3 MB'), { status: 413 });
  const buffer = Buffer.concat(chunks);
  if ((req.headers['content-type'] || '').includes('multipart')) {
    const form = await new Request('https://localhost/', { method: 'POST', headers: { 'content-type': req.headers['content-type'] }, body: buffer }).formData();
    return { ...JSON.parse(form.get('payload') || '{}'), hf_token: form.get('hf_token') };
  }
  return JSON.parse(buffer.toString() || '{}');
}
export default async function handler(req, res) {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  const url = new URL(req.url, 'http://localhost');
  const path = url.searchParams.get('path') || url.pathname.replace(/^\/api\//, '');
  const method = req.method;
  const send = (data, status = 200) => { res.statusCode = status; res.end(JSON.stringify(data)); };
  try {
    if (method === 'GET' && path === 'health') return send({ ok: true, status: 'ok', version: '0.2.0', service: 'onchainai' });
    if (method === 'GET' && ['hub/catalog', 'model-kit/status', 'models', 'registry'].includes(path)) {
      const catalog = await hubCatalog();
      if (path === 'models') return send({ ok: true, models: catalog.models.filter(m => !url.searchParams.get('hf_id') || m.id === url.searchParams.get('hf_id')), registry_status: 'artifact-catalog', onchain_verified: false });
      if (path === 'registry') return send({ protocol: 'CAAP/1.0', updated_at: catalog.updated_at, registry: catalog.models.map(m => ({ hf_model_id: m.id, hf_commit: m.sha, model_type: 'TextGeneration', artifact_url: m.url, program_pda: null, tx_sig: null, onchain_verified: false })), status: 'public-artifact-index', note: 'Hub membership is not proof of onchain registration.' });
      return send({ ok: true, ...catalog, models: catalog.models.map(m => ({ ...m, repo_id: m.id, status: m.files.some(f => /adapter_model|\.gguf|\.safetensors/.test(f)) ? 'artifact-published' : 'metadata-only' })), datasets: catalog.datasets.map(d => ({ ...d, repo_id: d.id, status: 'published' })), registry_url: '/.well-known/clawd-registry.json', jobs: [], jobs_status: 'Private HF training jobs are not exposed; no running-job claims are inferred from artifact metadata.', capabilities: { datasets: true, hf_publish: 'request-scoped-token', memo_commitment: true, program_registration: 'devnet; live program and IDL checked before every preview' }, constitution: { ok: false, files: [], source: '/docs/onchain.md', note: 'Doctrine is published; constitutional compliance is not inferred from metadata.' } });
    }
    if (method === 'GET' && path === 'attestations') return send({ ok: true, attestations: [], status: 'No verified SAS attestations imported. Use a confirmed transaction or account address to verify independently.' });
    if (method === 'GET' && path === 'training/status') return send({ ok: true, supported_types: ['pdf', 'json', 'jsonl', 'ndjson', 'csv', 'txt', 'md', 'yaml', 'yml', 'ipynb'], custody: 'browser-local until explicitly published', hf_token: 'request-scoped; not stored', history: 'wallet-scoped browser storage', max_publish_bytes: 2000000 });
    if (method === 'GET' && path === 'protocol') return send({ ok: true, protocol: 'CAAP/1.0', program_id: PROGRAM_ID, memo_program: MEMO_PROGRAM, registry_registration: 'not enabled without verified deployment and matching IDL', signing: 'Wallet Standard; browser signing only', sovereign_default: 'zkrouter/auto', paid_fallback: false });
    if (method === 'GET' && path === 'inference/models') return send(await inferenceModels());
    if (method === 'GET' && path === 'onchain/registry') return send({ ok: true, cluster: 'devnet', ...await onchainRegistry(), checked_at: new Date().toISOString() });
    if (method === 'GET' && path === 'onchain/status') {
      const cluster = url.searchParams.get('cluster') || 'devnet';
      return send(await registryStatus(cluster));
    }
    if (method === 'GET' && path === 'onchain/verify') {
      const cluster = url.searchParams.get('cluster') || 'devnet'; const signature = url.searchParams.get('signature');
      if (!/^[1-9A-HJ-NP-Za-km-z]{64,90}$/.test(signature || '')) return send({ ok: false, error: 'Invalid transaction signature' }, 400);
      const result = await rpc(cluster, 'getTransaction', [signature, { encoding: 'jsonParsed', commitment: 'confirmed', maxSupportedTransactionVersion: 0 }]);
      if (!result) return send({ ok: false, pending: true, error: 'Transaction is not confirmed or is not available on this cluster.' }, 404);
      const memos = result.transaction.message.instructions.filter(i => i.programId === MEMO_PROGRAM).map(i => i.parsed).filter(m => typeof m === 'string');
      const native = [];
      for (const instruction of result.transaction.message.instructions.filter(i => i.programId === PROGRAM_ID)) {
        let initializesModel = false;
        try { initializesModel = Buffer.from(bs58.decode(instruction.data)).subarray(0, 8).equals(Buffer.from([150, 196, 232, 118, 138, 43, 140, 244])); } catch { /* Unrecognized instruction is not a registration. */ }
        if (!initializesModel) continue;
        const pda = instruction.accounts?.[0];
        if (pda) { const state = await rpc(cluster, 'getAccountInfo', [pda, { encoding: 'base64', commitment: 'confirmed' }]); const decoded = await decodeRegistryAccount(pda, state.value); if (decoded) native.push({ ...decoded, cluster }); }
      }
      return send({ ok: !result.meta?.err, signature, cluster, slot: result.slot, block_time: result.blockTime, error: result.meta?.err || null, memos, native_registrations: native, signers: result.transaction.message.accountKeys.filter(k => k.signer).map(k => k.pubkey), proof_type: native.length ? 'ModelRegistry PDA and Solana memo commitment' : 'Solana memo commitment; not a SAS attestation or registry PDA' });
    }
    if (method === 'POST') {
      const body = await readBody(req);
      if (path === 'training/datasets') return send(await publishDataset(body));
      if (path === 'inference/chat') return send(await complete(body));
      if (path === 'register/preview' || path === 'register') {
        if (body.model_type === 'PricePrediction') return send({ ok: false, error: 'PricePrediction requires oracle verification and is not supported.' }, 403);
        if (body.live) return send({ ok: false, posted: false, error: 'Native registry registration is unavailable. Use the wallet-signed memo commitment flow to anchor provenance.' }, 409);
        return send({ ok: true, dry_run: true, posted: false, registry_api: '/api/register', hash_was_generated: !body.model_hash, payload: { protocol: 'CAAP/1.0', hf_model_id: body.hf_model_id || '', model_hash: body.model_hash || `sha256:${sha256(JSON.stringify(body))}`, model_type: body.model_type || 'TextGeneration', api_endpoint: body.api_endpoint || '', cluster: body.cluster || 'devnet' }, note: 'Preview only; a metadata hash does not prove model weights.' });
      }
      if (path === 'rpc') {
        const allowed = new Set(['getLatestBlockhash', 'getFeeForMessage', 'simulateTransaction', 'sendTransaction', 'getSignatureStatuses', 'getBalance', 'getAccountInfo']);
        if (!allowed.has(body.method) || !Array.isArray(body.params) || JSON.stringify(body.params).length > 10000) return send({ error: 'Unsupported RPC request' }, 400);
        return send({ result: await rpc(body.cluster || 'devnet', body.method, body.params) });
      }
    }
    return send({ ok: false, error: 'Endpoint not available' }, 404);
  } catch (error) { return send({ ok: false, error: error.status ? error.message : 'The operation failed. Check your token permissions, repository name, and provider availability.' }, error.status || 502); }
}
