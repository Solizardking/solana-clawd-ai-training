// /skills — public SkillHub: submit -> JEV typed verdict -> mint a Core record on devnet in your wallet.
import nacl from 'tweetnacl';
import bs58 from 'bs58';
import { bindingMessage, buildSkillRecord, skillBundleHash, skillMerkleLeaf } from './registry-hash.js';
import { connectDevnetWallet, signCoreCreate, signText, w } from './devnet-wallet.js';

const $ = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const short = (a, n = 4) => a ? `${a.slice(0, n)}…${a.slice(-n)}` : '';
const sha256 = async b => new Uint8Array(await crypto.subtle.digest('SHA-256', b));
function status(id, msg, kind = '') { const n = $(id); if (n) { n.textContent = msg; n.className = `status ${kind}`; } }
async function api(path, body) {
  const r = await fetch(`/api/${path}`, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
  let d; try { d = await r.json(); } catch { d = { error: `HTTP ${r.status}` }; }
  if (!r.ok || d.ok === false) throw new Error(d.error || `HTTP ${r.status}`);
  return d;
}
const st = { sub: null, extra: [], asset: null, binding: null, busy: false, config: null };
const badge = s => `<span class="badge ${esc(s)}">${s === 'verified' ? '✓ ' : s === 'rejected' ? '✗ ' : '◌ '}${esc(s)}</span>`;

function verdictHtml(v, extra = '') {
  const checks = v.checks ? Object.entries(v.checks).map(([k, p]) => `<div class="check"><span>${esc(k)}</span><b>${p.toFixed(2)}</b><div class="bar"><i style="width:${Math.round(p * 100)}%"></i></div></div>`).join('') : '<p class="muted" style="font-size:13px">No JEV probabilities: ' + esc(v.precheck?.ok === false ? 'blocked by the deterministic precheck.' : 'JEV did not run.') + '</p>';
  return `<article class="glass verdict"><div><div class="eyebrow" style="margin-bottom:12px">JEV VERDICT · ${esc(v.engine?.model || '')}</div>${badge(v.status)} <span class="muted mono" style="font-size:11.5px;margin-left:8px">outcome=${esc(v.outcome)}</span>
    <ul>${v.reasons.map(r => `<li>${esc(r)}</li>`).join('')}</ul>
    <div class="kv small" style="margin-top:16px"><div class="k">Verdict hash</div><div class="hash">${esc(v.verdictHash)}</div><div class="k">Decided</div><div>${esc(v.decidedAt)}</div><div class="k">Thresholds</div><div>verify ≥ ${v.engine?.approveAt} on every check · reject ≤ ${v.engine?.blockAt}</div></div>${extra}</div>
    <div><div class="eyebrow" style="margin-bottom:6px">TYPED CHECKS (noul)</div>${checks}</div></article>`;
}

async function readExtra(fileList) {
  st.extra = [];
  for (const f of fileList) {
    if (f.size > 200000) throw new Error(`${f.name} is larger than 200 KB.`);
    const buf = new Uint8Array(await f.arrayBuffer()); let bin = ''; for (const b of buf) bin += String.fromCharCode(b);
    st.extra.push({ path: f.webkitRelativePath && f.webkitRelativePath.includes('/') ? f.webkitRelativePath.split('/').slice(1).join('/') : f.name, base64: btoa(bin), bytes: buf });
  }
  $('file-list').textContent = st.extra.length ? st.extra.map(f => f.path).join(', ') : '';
}
async function submit() {
  if (st.busy) return; st.busy = true; $('submit-btn').disabled = true;
  status('submit-status', 'Hashing the bundle and asking JEV for a typed verdict…', 'work');
  try {
    const md = $('skill-md').value;
    const files = [{ path: 'SKILL.md', text: md }, ...st.extra.filter(f => f.path !== 'SKILL.md').map(f => ({ path: f.path, base64: f.base64 }))];
    const r = await api('skillhub/submit', { files });
    const local = await skillBundleHash([{ path: 'SKILL.md', content: new TextEncoder().encode(md) }, ...st.extra.filter(f => f.path !== 'SKILL.md').map(f => ({ path: f.path, content: f.bytes }))], sha256);
    const leaf = await skillMerkleLeaf(r.payload.slug, local, sha256);
    if (local !== r.payload.bundleHash || leaf !== r.payload.merkleLeaf) throw new Error(`Browser computed ${local}, server ${r.payload.bundleHash}. Not minting.`);
    st.sub = r; st.asset = null; st.binding = null;
    const p = r.payload;
    $('verdict').innerHTML = `<div class="kv" style="margin-top:16px"><div class="k">Slug</div><div>${esc(p.slug)}</div><div class="k">Version</div><div>${esc(p.version)}</div><div class="k">Bundle hash</div><div class="hash">${esc(p.bundleHash)}<div class="verified" style="margin-top:4px">✓ recomputed in your browser (SkillHub skill-bundle-v1)</div></div><div class="k">Merkle leaf</div><div class="hash">${esc(p.merkleLeaf)}</div><div class="k">Files</div><div>${p.files.map(f => `${esc(f.path)} · ${f.bytes} B`).join('<br>')}</div></div>` + verdictHtml(r.verdict, r.verdict.status !== 'verified' ? '<p class="muted" style="font-size:12.5px;margin-top:12px">You can still mint. The record will show as <b>' + esc(r.verdict.status) + '</b>.</p>' : '');
    $('verdict').hidden = false; $('mint').hidden = false;
    status('submit-status', `Hashed ${p.files.length} file(s). JEV: ${r.verdict.status}.`, r.verdict.status === 'verified' ? 'ok' : r.verdict.status === 'rejected' ? 'err' : 'work');
  } catch (e) { status('submit-status', e.message, 'err'); }
  finally { st.busy = false; $('submit-btn').disabled = false; steps(); }
}
function steps() {
  const has = !!st.sub, conn = !!w.account, bound = !!st.binding;
  $('s-wallet').className = conn ? 'done' : has ? 'active' : '';
  $('s-bind').className = bound ? 'done' : has && conn ? 'active' : '';
  $('s-mint').className = bound ? 'active' : '';
  $('s-bind-btn').disabled = !(has && conn) || bound;
  $('s-mint-btn').disabled = !bound;
  $('wallet').textContent = conn ? short(w.account.address) : 'Connect wallet';
  if (has) $('s-msg').textContent = bindingMessage(buildSkillRecord({ payload: st.sub.payload, authority: w.account?.address || '<your wallet>' }));
}
async function connect() { try { const a = await connectDevnetWallet(); if (a) status('s-wallet-status', `Authority & owner: ${a}`, 'ok'); st.binding = null; } catch (e) { status('s-wallet-status', e.message, 'err'); } steps(); }
async function bind() {
  try {
    status('s-bind-status', 'Approve the message in your wallet (free, not a transaction).', 'work');
    const asset = nacl.sign.keyPair();
    const rec = buildSkillRecord({ payload: st.sub.payload, authority: w.account.address, asset: bs58.encode(asset.publicKey) });
    st.binding = await signText(bindingMessage(rec)); st.asset = asset;
    status('s-bind-status', `Bound · signature ${short(st.binding, 8)}`, 'ok');
  } catch (e) { status('s-bind-status', e.message || 'Cancelled', 'err'); }
  steps();
}
async function mint() {
  if (st.busy) return; st.busy = true; $('s-mint-btn').disabled = true;
  const assetAddr = bs58.encode(st.asset.publicKey);
  try {
    status('s-mint-status', 'Building the devnet transaction…', 'work');
    const prep = await api('skillhub/prepare', { bundleHash: st.sub.payload.bundleHash, wallet: w.account.address, asset: assetAddr, bindingSignature: st.binding });
    if (prep.cluster !== 'devnet') throw new Error('Refusing: not a devnet transaction.');
    status('s-mint-status', `Checked: Metaplex Core only; ~${((prep.feeLamports + prep.rentLamportsEstimate) / 1e9).toFixed(4)} devnet SOL. Approve in your wallet…`, 'work');
    const signed = await signCoreCreate(prep.transaction, st.asset, st.sub.payload.bundleHash);
    status('s-mint-status', 'Sending to devnet…', 'work');
    const sent = await api('skillhub/submit-tx', { transaction: signed });
    $('s-result').innerHTML = `<div class="eyebrow" style="color:var(--long)">MINTED · DEVNET</div><h3>${esc(st.sub.payload.slug)} is <em>onchain.</em></h3> ${badge(st.sub.verdict.status)}
      <div class="kv small" style="margin-top:14px"><div class="k">Asset</div><div class="mono">${esc(assetAddr)}</div><div class="k">Transaction</div><div class="mono">${esc(sent.signature)}</div></div>
      <div class="links"><a class="btn primary" href="https://explorer.solana.com/address/${esc(assetAddr)}?cluster=devnet" target="_blank" rel="noreferrer">View on Explorer ↗</a><a class="btn ghost" href="/skills?asset=${esc(assetAddr)}">Skill page</a></div>`;
    $('s-result').hidden = false; status('s-mint-status', 'Confirmed on devnet.', 'ok');
    st.binding = null; st.asset = null; setTimeout(loadHub, 5000);
  } catch (e) { status('s-mint-status', e.message || 'Cancelled', 'err'); }
  finally { st.busy = false; steps(); }
}
async function loadHub() {
  try {
    const g = await api('skillhub/gallery');
    $('hub-meta').innerHTML = `${g.items.length} skill${g.items.length === 1 ? '' : 's'} indexed under <a class="mono" href="https://explorer.solana.com/address/${esc(g.indexAuthority)}?cluster=devnet" target="_blank" rel="noreferrer">${esc(short(g.indexAuthority, 6))}</a>. The badge shows only when this server holds the JEV verdict matching the onchain verdict hash.`;
    $('hub-grid').innerHTML = g.items.length ? g.items.map(i => `<article class="g-card"><div>${badge(i.status)}</div><h3 style="margin-top:10px">${esc(i.slug)}</h3><div class="repo">${esc(i.version || '')}</div><p class="muted" style="font-size:13px;margin:0 0 8px">${esc((i.description || '').slice(0, 140))}</p>
      <div class="row"><span>owner</span><b>${esc(short(i.owner))}</b></div><div class="row"><span>signature</span><b>${i.signatureValid ? 'valid' : 'invalid'}</b></div><div class="row"><span>bundle</span><b>${esc((i.bundleHash || '').slice(7, 19))}…</b></div>
      <div class="actions"><a class="btn ghost sm" href="/skills?asset=${esc(i.asset)}">Details</a><a class="btn ghost sm" href="${esc(i.explorer)}" target="_blank" rel="noreferrer">Explorer ↗</a></div></article>`).join('') : '<div class="empty">No skills minted yet. Be the first.</div>';
  } catch (e) { $('hub-meta').textContent = e.message; }
}
async function detail(asset) {
  const el = $('detail'); el.hidden = false; el.innerHTML = '<div class="skeleton"></div>';
  try {
    const d = await api(`skillhub/skill?asset=${encodeURIComponent(asset)}`);
    el.innerHTML = `<div class="eyebrow">SKILL · ${esc(short(asset, 6))}</div><h2>${esc(d.slug)} ${badge(d.status)}</h2>
      <div class="kv"><div class="k">Asset</div><div><a href="${esc(d.explorer)}" target="_blank" rel="noreferrer" class="mono">${esc(asset)}</a></div><div class="k">Owner</div><div class="mono">${esc(d.owner)}</div><div class="k">Version</div><div>${esc(d.version)}</div><div class="k">Bundle hash</div><div class="hash">${esc(d.bundleHash)}</div><div class="k">Merkle leaf</div><div class="hash">${esc(d.merkleLeaf)} ${d.merkleLeafValid ? '<span class="verified">✓ SkillHub encoding</span>' : '<span style="color:var(--short)">✗ mismatch</span>'}</div><div class="k">Binding signature</div><div>${d.signatureValid ? '<span class="verified">✓ valid for the authority</span>' : '<span style="color:var(--short)">✗ invalid</span>'}</div><div class="k">Onchain claim</div><div>jev=${esc(d.claimedStatus)} · ${esc(d.jevVerdict)}</div><div class="k">Immutable</div><div>${d.immutable ? 'yes (frozen update authority)' : 'no'}</div></div>
      ${d.verdict ? verdictHtml({ ...d.verdict, precheck: { ok: true } }) : '<div class="note"><div class="eyebrow">UNVERIFIED</div><p>This server does not hold a JEV verdict whose hash matches the onchain <code>jevVerdict</code>, so the skill is shown as unverified. Resubmitting the same bundle restores the verdict.</p></div>'}
      ${d.skillMd ? `<div class="files-head"><span>SKILL.md</span></div><pre class="msg" style="max-height:420px">${esc(d.skillMd)}</pre>` : ''}`;
  } catch (e) { el.innerHTML = `<p class="status err">${esc(e.message)}</p>`; }
}
async function init() {
  $('submit-btn').onclick = submit;
  $('extra-files').onchange = e => readExtra([...e.target.files]).catch(err => status('submit-status', err.message, 'err'));
  $('wallet').onclick = connect; $('s-wallet-btn').onclick = connect; $('s-bind-btn').onclick = bind; $('s-mint-btn').onclick = mint; $('refresh').onclick = loadHub;
  $('skill-md').oninput = () => { st.sub = null; st.binding = null; $('mint').hidden = true; steps(); };
  try { st.config = await api('skillhub/config'); } catch { /* shown on submit */ }
  steps(); loadHub();
  const asset = new URLSearchParams(location.search).get('asset');
  if (asset) detail(asset);
}
init();
