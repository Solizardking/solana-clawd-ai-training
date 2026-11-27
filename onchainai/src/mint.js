// /mint — self-serve Hugging Face model -> Metaplex Core record on Solana devnet.
// Keys: your wallet signs (Wallet Standard); a fresh asset keypair is generated here and discarded after mint.
import { getWallets } from '@wallet-standard/app';
import nacl from 'tweetnacl';
import bs58 from 'bs58';
import { bindingMessage, buildModelRecord, modelContentHash } from './registry-hash.js';
import { inspectWire, withSignature } from './tx-inspect.js';

const CORE = 'CoREENxT6tW1HoK8ypY1SxRMZTcVPm7R94rH4PZNhX7d';
const COMPUTE_BUDGET = 'ComputeBudget111111111111111111111111111111';
const CHAIN = 'solana:devnet';
const $ = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const short = (a, n = 4) => a ? `${a.slice(0, n)}…${a.slice(-n)}` : '';
const num = n => n == null ? '—' : n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n);
const bytes = n => { if (n == null) return '—'; const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0; let v = n; while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; } return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${u[i]}`; };
const sha256 = async b => new Uint8Array(await crypto.subtle.digest('SHA-256', b));
function status(id, msg, kind = '') { const n = $(id); if (n) { n.textContent = msg; n.className = `status ${kind}`; } }
async function api(path, body) {
  const r = await fetch(`/api/${path}`, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
  let d; try { d = await r.json(); } catch { d = { error: `HTTP ${r.status}` }; }
  if (!r.ok || d.ok === false) throw new Error(d.error || `HTTP ${r.status}`);
  return d;
}

const state = { config: null, model: null, wallet: null, account: null, binding: null, record: null, asset: null, busy: false };

// ---------- wallet ----------
function walletLabel() { return state.account ? short(state.account.address) : 'Connect wallet'; }
function paintWallet() {
  $('wallet').textContent = walletLabel();
  $('step-wallet-btn').textContent = state.account ? `Connected ${short(state.account.address)} · disconnect` : 'Connect wallet';
  if (state.account) status('wallet-status', `Authority & owner: ${state.account.address}`, 'ok');
  else status('wallet-status', '');
  steps();
}
async function connect() {
  if (state.account) { try { await state.wallet?.features['standard:disconnect']?.disconnect(); } catch { /* ignore */ } state.account = null; state.wallet = null; reset('wallet'); return paintWallet(); }
  const wallets = getWallets().get().filter(w => w.features['standard:connect'] && w.chains.includes(CHAIN));
  if (!wallets.length) throw new Error('No devnet-capable Solana wallet found. Install Phantom or Solflare, then reload.');
  const chosen = await new Promise(resolve => {
    const dialog = document.createElement('div'); dialog.id = 'wallet-dialog';
    const panel = document.createElement('div'); panel.className = 'panel';
    panel.innerHTML = '<div class="eyebrow">DEVNET ONLY</div><h2>Connect a wallet</h2><p class="muted">Your wallet keeps its keys. Transactions go to Solana devnet only.</p>';
    for (const w of wallets) { const b = document.createElement('button'); if (w.icon) { const img = document.createElement('img'); img.src = w.icon; img.alt = ''; b.append(img); } b.append(w.name); b.onclick = () => { dialog.remove(); resolve(w); }; panel.append(b); }
    const c = document.createElement('button'); c.textContent = 'Cancel'; c.onclick = () => { dialog.remove(); resolve(null); }; panel.append(c);
    dialog.append(panel); document.body.append(dialog);
  });
  if (!chosen) return;
  const res = await chosen.features['standard:connect'].connect();
  const account = res.accounts.find(a => a.chains.includes(CHAIN)) || res.accounts.find(a => a.chains.some(c => c.startsWith('solana:')));
  if (!account) throw new Error('Wallet returned no Solana account.');
  if (!chosen.features['solana:signMessage'] || !chosen.features['solana:signTransaction']) throw new Error('This wallet cannot sign messages and transactions. Use Phantom or Solflare.');
  state.wallet = chosen; state.account = account;
  chosen.features['standard:events']?.on('change', ({ accounts }) => { if (accounts) { state.account = accounts.find(a => a.chains.some(c => c.startsWith('solana:'))) || null; reset('wallet'); paintWallet(); } });
  reset('wallet'); paintWallet();
}

// ---------- model preview ----------
function stat(label, value) { return `<div class="stat"><b>${esc(value)}</b><span>${esc(label)}</span></div>`; }
function renderPreview(m) {
  const p = m.preview, pay = m.payload;
  const tags = [p.pipelineTag && `<span class="tag hi">${esc(p.pipelineTag)}</span>`, p.libraryName && `<span class="tag hi">${esc(p.libraryName)}</span>`, p.license && `<span class="tag">license: ${esc(p.license)}</span>`, p.gated && `<span class="tag warn">gated: ${esc(p.gated)}</span>`, ...p.tags.filter(t => !t.startsWith('license:') && t !== p.pipelineTag && t !== p.libraryName).slice(0, 14).map(t => `<span class="tag">${esc(t)}</span>`)].filter(Boolean).join('');
  const rows = [
    ['Revision', `<a class="mono" href="https://huggingface.co/${esc(p.id)}/tree/${esc(p.sha)}" target="_blank" rel="noreferrer">${esc(p.sha)}</a>`],
    ['Last modified', esc(p.lastModified ? new Date(p.lastModified).toISOString().replace('T', ' ').slice(0, 16) + ' UTC' : '—')],
    p.baseModel && ['Base model', esc(p.baseModel)],
    p.modelType && ['Model type', esc([p.modelType, ...(p.architectures || [])].join(' · '))],
    p.ggufArchitecture && ['GGUF', esc(`${p.ggufArchitecture}${p.contextLength ? ` · ctx ${num(p.contextLength)}` : ''}`)],
    ['Files', esc(`${p.fileCount} total · ${p.hashedCount} hashed${p.unhashedCount ? ` · ${p.unhashedCount} recorded unhashed` : ''}`)],
    ['Content hash', `<span class="hash">${esc(m.contentHash)}</span><div id="browser-verify" class="verified" style="margin-top:6px">Recomputing in your browser…</div>`],
  ].filter(Boolean).map(([k, v]) => `<div class="k">${k}</div><div>${v}</div>`).join('');
  const files = p.files.map(f => `<tr class="${f.hashed ? '' : 'un'}"><td>${esc(f.path)}</td><td class="sz">${bytes(f.size)}</td><td class="h">${f.hashed ? esc((pay.files.find(x => x.path === f.path)?.sha256 || '').slice(7, 19)) + '…' : esc(f.reason || 'unhashed')}</td></tr>`).join('');
  $('preview').innerHTML = `<article class="glass model">
    <div class="head">
      <div class="who"><div class="avatar">${esc((p.author || p.id)[0].toUpperCase())}</div><div><a href="https://huggingface.co/${esc(p.author || '')}" target="_blank" rel="noreferrer">${esc(p.author || p.id.split('/')[0])}</a><br>huggingface.co</div></div>
      <h2><small>Hugging Face model</small>${esc(p.id.split('/')[1])}</h2>
      <div class="tags">${tags}</div>
      <div class="stats">${stat('downloads / 30d', num(p.downloads))}${stat('likes', num(p.likes))}${stat(p.parameters ? 'parameters' : 'files', p.parameters ? num(p.parameters) : p.fileCount)}${stat('repo size', bytes(p.totalSize))}</div>
      <div class="files-head"><span>Files bound by the hash</span><a href="https://huggingface.co/${esc(p.id)}" target="_blank" rel="noreferrer">Open on Hugging Face ↗</a></div>
      <div class="files"><table>${files}</table></div>
    </div>
    <div class="side"><div class="eyebrow" style="margin-bottom:14px">REGISTRY RECORD · TYPE=MODEL</div><div class="kv">${rows}</div>
      <p class="muted" style="font-size:12.5px;margin-top:14px">LFS and Xet files use the Hub's LFS SHA-256. Other files were downloaded at this commit and hashed. Files over the hashing limits are listed with their git blob id inside the hashed provenance.</p></div>
  </article>`;
  $('preview').hidden = false;
  verifyInBrowser(m);
}
async function verifyInBrowser(m) {
  try {
    const local = await modelContentHash(m.payload, sha256);
    const el = $('browser-verify');
    if (!el) return;
    if (local === m.contentHash) { el.textContent = '✓ Recomputed in your browser: matches'; el.style.color = ''; }
    else { el.textContent = `✗ Browser recomputed ${local}`; el.style.color = 'var(--short)'; state.model = null; steps(); }
  } catch (e) { const el = $('browser-verify'); if (el) { el.textContent = `Browser check failed: ${e.message}`; el.style.color = 'var(--short)'; } }
}
async function lookup(input) {
  if (state.busy) return;
  state.busy = true; $('fetch-btn').disabled = true;
  status('lookup-status', 'Reading the Hub and hashing files at the current commit…', 'work');
  $('preview').hidden = false; $('preview').innerHTML = '<div class="skeleton"></div>';
  try {
    const m = await api(`hf-mint/model?id=${encodeURIComponent(input)}`);
    state.model = m; reset('model');
    renderPreview(m);
    status('lookup-status', `Pinned ${m.payload.hfRepo} @ ${m.payload.revisionSha.slice(0, 10)} · ${m.payload.files.length} file hashes bound.`, 'ok');
    $('mint').hidden = false;
    const u = new URL(location.href); u.searchParams.set('model', m.payload.hfRepo); history.replaceState(null, '', u);
  } catch (e) { $('preview').hidden = true; state.model = null; status('lookup-status', e.message, 'err'); }
  finally { state.busy = false; $('fetch-btn').disabled = false; steps(); }
}

// ---------- mint flow ----------
function reset(from) {
  state.binding = null; state.record = null; state.asset = null; $('consent').checked = false; status('bind-status', ''); status('mint-status', '');
  if (from === 'model') $('result').hidden = true;
}
function steps() {
  const hasModel = !!state.model, hasWallet = !!state.account, bound = !!state.binding;
  $('step-wallet').className = hasWallet ? 'done' : hasModel ? 'active' : '';
  $('step-bind').className = bound ? 'done' : hasWallet && hasModel ? 'active' : '';
  $('step-mint').className = bound ? 'active' : '';
  $('bind-btn').disabled = !(hasModel && hasWallet) || bound;
  $('mint-btn').disabled = !(bound && $('consent').checked);
  if (hasModel) {
    const kp = state.asset || null;
    const rec = buildModelRecord({ payload: state.model.payload, contentHash: state.model.contentHash, authority: state.account?.address || '<your wallet>', asset: kp ? bs58.encode(kp.publicKey) : null });
    $('binding-msg').textContent = bindingMessage(rec);
  }
  $('tx-facts').innerHTML = [['Network', 'Solana devnet (mainnet refused)'], ['Program', `Metaplex Core <span class="mono">${short(CORE, 6)}</span>`], ['Fee payer', state.account ? esc(state.account.address) : 'your wallet'], ['Asset', state.asset ? esc(bs58.encode(state.asset.publicKey)) : 'fresh keypair made in this tab'], ['Update authority', state.config ? `${esc(state.config.indexAuthority)} (unsignable index)` : '…'], ['Collection', 'none (see note below)']].map(([k, v]) => `<div class="k">${k}</div><div>${v}</div>`).join('');
}
async function signBinding() {
  const m = state.model; if (!m || !state.account) return;
  try {
    status('bind-status', 'Approve the message in your wallet. This is not a transaction and costs nothing.', 'work');
    const asset = nacl.sign.keyPair();
    const record = buildModelRecord({ payload: m.payload, contentHash: m.contentHash, authority: state.account.address, asset: bs58.encode(asset.publicKey) });
    const message = bindingMessage(record);
    $('binding-msg').textContent = message;
    const [signed] = await state.wallet.features['solana:signMessage'].signMessage({ account: state.account, message: new TextEncoder().encode(message) });
    const signature = bs58.encode(signed.signature);
    if (!nacl.sign.detached.verify(new TextEncoder().encode(message), signed.signature, bs58.decode(state.account.address))) throw new Error('Wallet returned a signature that does not verify.');
    state.asset = asset; state.record = record; state.binding = signature;
    status('bind-status', `Bound · signature ${short(signature, 8)}`, 'ok');
  } catch (e) { status('bind-status', e.message || 'Signing was cancelled.', 'err'); }
  steps();
}
function assertSafeTx(wire, assetAddr, contentHash) {
  const info = inspectWire(wire, bs58.encode);
  if (info.feePayer !== state.account.address) throw new Error('Refusing: fee payer is not your wallet.');
  if (!info.programs.length || info.programs.some(p => p !== CORE && p !== COMPUTE_BUDGET) || !info.programs.includes(CORE)) throw new Error('Refusing: transaction calls a program other than Metaplex Core.');
  if (info.signers.some(s => s !== state.account.address && s !== assetAddr)) throw new Error('Refusing: unexpected signer in transaction.');
  const needle = new TextEncoder().encode(contentHash);
  const hay = info.message; let found = false;
  outer: for (let i = 0; i <= hay.length - needle.length; i++) { for (let j = 0; j < needle.length; j++) if (hay[i + j] !== needle[j]) continue outer; found = true; break; }
  if (!found) throw new Error('Refusing: transaction does not carry the content hash you reviewed.');
  return info;
}
async function mint() {
  if (state.busy || !state.binding) return;
  state.busy = true; $('mint-btn').disabled = true;
  const assetAddr = bs58.encode(state.asset.publicKey);
  try {
    status('mint-status', 'Building the devnet transaction…', 'work');
    const prep = await api('hf-mint/prepare', { modelId: state.model.payload.hfRepo, revision: state.model.payload.revisionSha, wallet: state.account.address, asset: assetAddr, bindingSignature: state.binding });
    if (prep.cluster !== 'devnet') throw new Error('Refusing: server did not prepare a devnet transaction.');
    const wire = Uint8Array.from(atob(prep.transaction), c => c.charCodeAt(0));
    const info = assertSafeTx(wire, assetAddr, state.model.contentHash);
    status('mint-status', `Checked: Metaplex Core only, you pay ~${((prep.feeLamports + prep.rentLamportsEstimate) / 1e9).toFixed(4)} devnet SOL (fee + rent). Approve in your wallet…`, 'work');
    const [out] = await state.wallet.features['solana:signTransaction'].signTransaction({ account: state.account, transaction: wire, chain: CHAIN });
    const signedInfo = inspectWire(out.signedTransaction, bs58.encode);
    if (bs58.encode(signedInfo.message) !== bs58.encode(info.message)) throw new Error('Wallet changed the transaction. Nothing was sent; try again.');
    const full = withSignature(out.signedTransaction, signedInfo, assetAddr, nacl.sign.detached(signedInfo.message, state.asset.secretKey));
    status('mint-status', 'Sending to devnet and waiting for confirmation…', 'work');
    let bin = ''; for (const b of full) bin += String.fromCharCode(b);
    const sent = await api('hf-mint/submit', { transaction: btoa(bin) });
    showResult({ asset: assetAddr, signature: sent.signature, explorerTx: sent.explorer, pending: sent.pending });
    status('mint-status', sent.pending ? 'Sent. Devnet is still confirming.' : 'Confirmed on devnet.', 'ok');
    state.binding = null; state.asset = null; state.record = null; $('consent').checked = false;
    setTimeout(loadGallery, 4000);
  } catch (e) { status('mint-status', e.message || 'Mint cancelled.', 'err'); }
  finally { state.busy = false; steps(); }
}
function showResult({ asset, signature, explorerTx, pending }) {
  const r = $('result');
  r.innerHTML = `<div class="eyebrow" style="color:var(--long)">${pending ? 'SENT · CONFIRMING' : 'MINTED · DEVNET'}</div><h3>${esc(state.model.payload.hfRepo.split('/')[1])} is <em>onchain.</em></h3>
  <div class="kv small" style="margin-top:14px"><div class="k">Asset</div><div class="mono">${esc(asset)}</div><div class="k">Transaction</div><div class="mono">${esc(signature)}</div><div class="k">Content hash</div><div class="hash">${esc(state.model.contentHash)}</div></div>
  <div class="links"><a class="btn primary" href="https://explorer.solana.com/address/${esc(asset)}?cluster=devnet" target="_blank" rel="noreferrer">View asset on Explorer ↗</a><a class="btn ghost" href="${esc(explorerTx)}" target="_blank" rel="noreferrer">Transaction ↗</a><a class="btn ghost" href="/api/hf-mint/metadata/${esc(asset)}" target="_blank" rel="noreferrer">Metadata JSON ↗</a></div>`;
  r.hidden = false; r.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ---------- gallery ----------
async function loadGallery() {
  const grid = $('gallery-grid');
  try {
    const g = await api('hf-mint/gallery');
    $('gallery-meta').innerHTML = `${g.items.length} self-serve record${g.items.length === 1 ? '' : 's'} indexed under <a class="mono" href="https://explorer.solana.com/address/${esc(g.indexAuthority)}?cluster=devnet" target="_blank" rel="noreferrer">${esc(short(g.indexAuthority, 6))}</a> via Helius DAS. Signatures are checked on the server; press Verify to recompute a hash from Hugging Face.`;
    if (!g.items.length) { grid.innerHTML = '<div class="empty">No models minted yet. Be the first.</div>'; return; }
    grid.innerHTML = g.items.map(i => `<article class="g-card"><div class="eyebrow" style="font-size:10px">${i.signatureValid ? '<span class="verified">✓ SIGNED BY AUTHORITY</span>' : '<span style="color:var(--short)">UNVERIFIED SIGNATURE</span>'}</div>
      <h3>${esc(i.name || i.hfRepo)}</h3><div class="repo">${esc(i.hfRepo)} @ ${esc((i.revisionSha || '').slice(0, 10))}</div>
      <div class="row"><span>owner</span><b>${esc(short(i.owner))}</b></div><div class="row"><span>license</span><b>${esc(i.license || '—')}</b></div><div class="row"><span>hash</span><b class="mono">${esc((i.contentHash || '').slice(7, 19))}…</b></div>
      <div class="actions"><a class="btn ghost sm" href="${esc(i.explorer)}" target="_blank" rel="noreferrer">Explorer ↗</a><button class="btn ghost sm" data-verify="${esc(i.asset)}">Verify</button><a class="btn ghost sm" href="/mint?model=${encodeURIComponent(i.hfRepo || '')}">Mint yours</a></div><div class="vres" id="v-${esc(i.asset)}"></div></article>`).join('');
  } catch (e) { $('gallery-meta').textContent = e.message; grid.innerHTML = ''; }
}
async function verify(asset) {
  const el = $(`v-${asset}`); if (!el) return;
  el.textContent = 'Recomputing from Hugging Face…'; el.style.color = 'var(--warn)';
  try {
    const v = await api(`hf-mint/verify?asset=${encodeURIComponent(asset)}`);
    const ok = v.hashMatches && v.signatureValid && v.immutable;
    el.textContent = ok ? '✓ hash matches the Hub at this commit · signature valid · immutable' : `✗ ${v.error || (!v.hashMatches ? 'hash differs from the Hub' : !v.signatureValid ? 'signature invalid' : 'mutable')}`;
    el.style.color = ok ? 'var(--long)' : 'var(--short)';
  } catch (e) { el.textContent = e.message; el.style.color = 'var(--short)'; }
}

// ---------- init ----------
async function init() {
  if (/mainnet/i.test(location.search)) status('lookup-status', 'Mainnet is refused. This page mints on Solana devnet only.', 'err');
  $('lookup').onsubmit = e => { e.preventDefault(); lookup($('model-input').value); };
  document.querySelectorAll('[data-ex]').forEach(b => { b.onclick = () => { $('model-input').value = b.dataset.ex; lookup(b.dataset.ex); }; });
  const onConnect = () => connect().catch(e => status('wallet-status', e.message, 'err'));
  $('wallet').onclick = () => { if (!state.model) $('step-wallet-btn').scrollIntoView({ block: 'center' }); onConnect(); };
  $('step-wallet-btn').onclick = onConnect;
  $('bind-btn').onclick = signBinding;
  $('consent').onchange = steps;
  $('mint-btn').onclick = mint;
  $('refresh-gallery').onclick = loadGallery;
  $('gallery-grid').onclick = e => { const b = e.target.closest('[data-verify]'); if (b) verify(b.dataset.verify); };
  try { state.config = await api('hf-mint/config'); $('collection-note').textContent = state.config.collectionNote; } catch (e) { $('collection-note').textContent = e.message; }
  steps(); loadGallery();
  const pre = new URLSearchParams(location.search).get('model');
  if (pre) { $('model-input').value = pre; lookup(pre); }
}
init();
