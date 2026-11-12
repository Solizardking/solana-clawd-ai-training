import { scanSecrets } from './service.mjs';

export const CLAWD_MINT = '8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump';
export const MODEL_ID = '@cf/cloudflare/clef';
export const ENGINE_LABEL = 'LIVE · Clef 27B (@cf/cloudflare/clef, Cloudflare Workers AI)';

const API_BASE = 'https://api.cloudflare.com/client/v4';
const PUMP_MCP = 'https://pumpai.ai/mcp';
const TOOLS_MCP = 'https://pumpai.ai/tools';
const DEX = 'https://api.dexscreener.com';

export const MAJORS = new Set([
  'So11111111111111111111111111111111111111112',
  'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
  'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
  'USD1ttGY1N17NEEHLmELoaybftRBUSErhqYiQzvEmuB',
  'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',
  'J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn',
]);

export const DEFAULT_QUESTIONS = {
  safety_check: {
    type: 'noul',
    instructions: 'Is this token free of critical safety red flags for a researcher: no active mint or freeze authority abuse, no extreme holder concentration, no zero-liquidity trap?',
    criteria: {
      true: 'No critical red flags in the state',
      false: 'At least one critical red flag is present',
    },
  },
  liquidity_ok: {
    type: 'noul',
    instructions: 'Is the liquidity deep enough that a small research position could enter and exit without extreme slippage?',
    criteria: {
      true: 'Liquidity is adequate for small-size entry/exit',
      false: 'Liquidity is too thin or absent',
    },
  },
  momentum: {
    type: 'score',
    instructions: "How strong is this token's current market momentum?",
    criteria: [
      'Dumping: heavy net selling, price falling fast',
      'Weakening: more sellers than buyers, momentum fading',
      'Flat: balanced flow, price roughly unchanged',
      'Building: buyers stepping in, volume and price rising',
      'Breaking out: aggressive net buying, strong price expansion',
    ],
  },
  distribution: {
    type: 'score',
    instructions: 'How well distributed is the holder base?',
    criteria: [
      'Highly concentrated: one or a few wallets dominate supply',
      'Concentrated: top holders control a large share',
      'Mixed: some concentration but a real holder base exists',
      'Broadly distributed: wide holder base, no dominant wallet',
    ],
  },
  verdict: {
    type: 'choice',
    instructions: 'What is the best next step for a researcher looking at this token right now?',
    criteria: {
      avoid: 'Red flags or no edge — stay away',
      watch: 'Interesting but unproven — put it on the watchlist',
      research_more: 'Worth a deeper dive before any conclusion',
      candidate: 'Strongest profile on the board — priority research candidate',
    },
  },
};

export const QUESTION_TITLES = {
  safety_check: 'Safety — free of critical red flags?',
  liquidity_ok: 'Liquidity — deep enough for small size?',
  momentum: 'Momentum',
  distribution: 'Holder distribution',
  verdict: 'Verdict — best next step',
};

export function clefConfigured() {
  return Boolean(process.env.CF_ACCOUNT_ID?.trim() && process.env.CF_API_TOKEN?.trim());
}

export function isMint(value) {
  return typeof value === 'string' && /^[1-9A-HJ-NP-Za-km-z]{32,44}$/.test(value.trim());
}

export function rejectSecretPayload(value) {
  const text = typeof value === 'string' ? value : JSON.stringify(value ?? '');
  const token = process.env.CF_API_TOKEN;
  if (token && token.length > 8 && text.includes(token)) {
    throw Object.assign(new Error('credential_payload'), { status: 400, code: 'credential_payload' });
  }
  if (scanSecrets(text)) {
    throw Object.assign(new Error('credential_payload'), { status: 400, code: 'credential_payload' });
  }
}

export function resolveSitePage(host, pathname) {
  const h = String(host || '').split(':')[0].toLowerCase();
  const p = pathname || '/';
  const clefHost = h === 'clef.onchainai.fund' || h.startsWith('clef.');
  const indexHost = h === 'index.onchainai.fund' || h.startsWith('index.');
  if (clefHost && (p === '/' || p === '/index.html' || p === '/clef' || p === '/clef/')) return '/clef/index.html';
  if (indexHost && (p === '/' || p === '/index.html' || p === '/decision-index' || p === '/decision-index/')) return '/decision-index/index.html';
  if (p === '/clef' || p === '/clef/') return '/clef/index.html';
  if (p === '/decision-index' || p === '/decision-index/') return '/decision-index/index.html';
  return null;
}

function num(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function unwrap(resp) {
  if (resp && typeof resp === 'object' && 'data' in resp && (resp.data && typeof resp.data === 'object')) return resp.data;
  return resp;
}

async function fetchJson(url, { timeout = 12000, headers, method = 'GET', body } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, {
      method,
      headers,
      body,
      signal: controller.signal,
    });
    const text = await response.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text.slice(0, 200) }; }
    if (!response.ok) {
      const err = new Error(`HTTP ${response.status}`);
      err.status = response.status;
      err.body = data;
      throw err;
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

class McpClient {
  constructor(url, name) {
    this.url = url;
    this.name = name;
    this.sid = null;
    this.id = 0;
    this.ready = false;
  }

  async post(body, { expectReply = true, timeout = 15000 } = {}) {
    const headers = { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream' };
    if (this.sid) headers['Mcp-Session-Id'] = this.sid;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(this.url, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const sid = response.headers.get('Mcp-Session-Id');
      if (sid) this.sid = sid;
      const raw = await response.text();
      if (!expectReply) return null;
      const lines = raw.split(/\r?\n/).filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim());
      if (lines.length) return JSON.parse(lines.at(-1));
      return raw.trim() ? JSON.parse(raw) : {};
    } finally {
      clearTimeout(timer);
    }
  }

  async connect() {
    this.id += 1;
    await this.post({
      jsonrpc: '2.0',
      id: this.id,
      method: 'initialize',
      params: {
        protocolVersion: '2025-03-26',
        capabilities: {},
        clientInfo: { name: 'onchainai-clef-index', version: '1.0' },
      },
    });
    await this.post({ jsonrpc: '2.0', method: 'notifications/initialized' }, { expectReply: false });
    this.ready = true;
  }

  async call(tool, args = {}, timeout = 15000) {
    if (!this.ready) await this.connect();
    this.id += 1;
    const res = await this.post({
      jsonrpc: '2.0',
      id: this.id,
      method: 'tools/call',
      params: { name: tool, arguments: args },
    }, { timeout });
    if (res && res.error) throw new Error(`${tool}: ${JSON.stringify(res.error).slice(0, 180)}`);
    const result = res?.result || {};
    if (result.isError) {
      const texts = (result.content || []).filter((c) => c.type === 'text').map((c) => c.text).join(' ');
      throw new Error(`${tool}: ${texts.slice(0, 200)}`);
    }
    const texts = (result.content || []).filter((c) => c.type === 'text').map((c) => c.text);
    if (!texts.length) return result.structuredContent ?? result;
    try { return JSON.parse(texts[0]); } catch { return { text: texts[0] }; }
  }

  reset() {
    this.sid = null;
    this.ready = false;
  }
}

const toolsMcp = new McpClient(TOOLS_MCP, 'tools');
const pumpMcp = new McpClient(PUMP_MCP, 'pump');

async function safeCall(client, tool, args, timeout) {
  try {
    return { value: await client.call(tool, args, timeout), error: null };
  } catch (error) {
    client.reset();
    return { value: null, error: error.message || String(error) };
  }
}

function pickDexPair(payload) {
  const pairs = Array.isArray(payload?.pairs) ? payload.pairs.filter((p) => p && (p.chainId === 'solana' || p.chain === 'solana')) : [];
  pairs.sort((a, b) => (num(b?.liquidity?.usd) || 0) - (num(a?.liquidity?.usd) || 0));
  return pairs[0] || null;
}

async function dexToken(mint) {
  const payload = await fetchJson(`${DEX}/latest/dex/tokens/${encodeURIComponent(mint)}`, { timeout: 10000 });
  return pickDexPair(payload);
}

async function fetchTokenState(mint) {
  const sources = {};
  const ovCall = await safeCall(toolsMcp, 'get_token_overview', { address: mint });
  let ov = unwrap(ovCall.value) || {};
  if (ovCall.error || (ov && ov.ok === false)) {
    sources.overview = ovCall.error ? `unavailable (${ovCall.error})` : `unavailable (${ov.error || 'ok false'})`;
    ov = {};
  } else if (ovCall.value) {
    sources.overview = 'pumpai.ai/tools · get_token_overview (Birdeye)';
  }

  const lpCall = await safeCall(toolsMcp, 'get_live_price', { address: mint });
  const lp = !lpCall.error ? unwrap(lpCall.value) || {} : {};
  if (!lpCall.error && lpCall.value) sources.live_price = 'pumpai.ai/tools · get_live_price (Birdeye)';

  const tdCall = await safeCall(toolsMcp, 'get_token_trade_data', { address: mint });
  const td = !tdCall.error ? unwrap(tdCall.value) || {} : {};
  if (!tdCall.error && tdCall.value) sources.trade_data = 'pumpai.ai/tools · get_token_trade_data (Birdeye)';

  const secCall = await safeCall(toolsMcp, 'get_birdeye_security', { address: mint });
  const sec = !secCall.error ? unwrap(secCall.value) || {} : {};
  if (!secCall.error && secCall.value) sources.security = 'pumpai.ai/tools · get_birdeye_security (Birdeye)';

  const holdCall = await safeCall(toolsMcp, 'st_get_token_top_holders', { mint, limit: 10 });
  let topShare = null;
  if (!holdCall.error && Array.isArray(holdCall.value) && holdCall.value.length) {
    const shares = holdCall.value
      .filter((h) => h && typeof h === 'object')
      .map((h) => num(h.percentage ?? h.share ?? h.percent))
      .filter((s) => s != null);
    if (shares.length) {
      topShare = Math.max(...shares);
      if (topShare <= 1) topShare *= 100;
    }
    sources.holders = 'pumpai.ai/tools · st_get_token_top_holders (Solana Tracker)';
  } else if (holdCall.error) {
    sources.holders = 'unavailable (Solana Tracker upstream)';
  }

  const launchCall = await safeCall(pumpMcp, 'get_recent_launches', { limit: 50 });
  let tape = { on_launch_tape: false };
  if (!launchCall.error && Array.isArray(launchCall.value)) {
    const item = launchCall.value.find((row) => row && row.mint === mint);
    if (item) tape = { on_launch_tape: true, launchpad: item.launchpad, source: item.source, launched_at: item.time };
    sources.launch_tape = 'pumpai.ai/mcp · get_recent_launches (clawd-pump)';
  }

  if (!sources.overview || !num(ov.price || ov.marketCap)) {
    try {
      const pair = await dexToken(mint);
      if (pair) {
        sources.dexscreener = 'api.dexscreener.com · latest/dex/tokens';
        ov = {
          symbol: ov.symbol || pair.baseToken?.symbol,
          name: ov.name || pair.baseToken?.name,
          price: ov.price || pair.priceUsd,
          marketCap: ov.marketCap || ov.mc || pair.marketCap,
          fdv: ov.fdv || pair.fdv,
          liquidity: ov.liquidity || pair.liquidity?.usd,
          v24hUSD: ov.v24hUSD || pair.volume?.h24,
          v24hChangePercent: ov.v24hChangePercent || pair.priceChange?.h24,
        };
        if (!lp.value) lp.value = pair.priceUsd;
        if (lp.priceChange24h == null) lp.priceChange24h = pair.priceChange?.h24;
      }
    } catch {
      if (!sources.overview) sources.overview = 'unavailable (DexScreener fallback failed)';
    }
  }

  const state = {
    token: {
      symbol: ov.symbol || '?',
      name: ov.name || '?',
      mint,
    },
    market: {
      price_usd: num(lp.value) ?? num(ov.price),
      market_cap_usd: num(ov.marketCap || ov.mc),
      fdv_usd: num(ov.fdv),
      liquidity_usd: num(ov.liquidity),
      volume_24h_usd: num(ov.v24hUSD || ov.volume24h || ov.v24hUsd),
      price_change_24h_pct: num(lp.priceChange24h) ?? num(ov.priceChange24hPercent || ov.v24hChangePercent),
      holder_count: ov.holder || ov.holders || ov.holderCount || null,
    },
    trading: td && typeof td === 'object' ? td : {},
    security: sec && typeof sec === 'object' ? sec : {},
    holders: { top_holder_share_pct: topShare },
    launch_tape: tape,
  };
  return { state, sources };
}

function row(rank, name, symbol, source, price, mcap, liquidity, change, mint, extra = {}) {
  return {
    rank,
    name: name || '?',
    symbol: symbol || '?',
    source,
    price_usd: num(price),
    market_cap_usd: num(mcap),
    liquidity_usd: num(liquidity),
    price_change_24h_pct: num(change),
    mint,
    ...extra,
  };
}

let boardCache = { at: 0, payload: null };
const BOARD_TTL_MS = 45_000;

async function dexTrending() {
  const entries = [];
  try {
    const boosted = await fetchJson(`${DEX}/token-boosts/top/v1`, { timeout: 10000 });
    const list = Array.isArray(boosted) ? boosted : [];
    const seen = new Set();
    for (const item of list) {
      const chain = item.chainId || item.chain;
      const mint = item.tokenAddress || item.address;
      if (chain && chain !== 'solana') continue;
      if (!isMint(mint) || MAJORS.has(mint) || seen.has(mint) || mint === CLAWD_MINT) continue;
      seen.add(mint);
      let pair = null;
      try { pair = await dexToken(mint); } catch { /* board row can ship without enrichment */ }
      entries.push(row(
        entries.length + 1,
        pair?.baseToken?.name || item.description?.slice(0, 40) || '?',
        pair?.baseToken?.symbol || '?',
        'trending · DexScreener token-boosts',
        pair?.priceUsd,
        pair?.marketCap,
        pair?.liquidity?.usd,
        pair?.priceChange?.h24,
        mint,
      ));
      if (entries.length >= 12) break;
    }
  } catch {
    return [];
  }
  return entries;
}

export async function buildBoard() {
  if (boardCache.payload && Date.now() - boardCache.at < BOARD_TTL_MS) return boardCache.payload;
  const entries = [];
  const sources = {};

  const ovCall = await safeCall(toolsMcp, 'get_token_overview', { address: CLAWD_MINT });
  let ov = !ovCall.error ? unwrap(ovCall.value) || {} : {};
  if (ov.ok === false) ov = {};
  const lpCall = await safeCall(toolsMcp, 'get_live_price', { address: CLAWD_MINT });
  let lp = !lpCall.error ? unwrap(lpCall.value) || {} : {};
  if (!ov.symbol && !num(ov.price)) {
    try {
      const pair = await dexToken(CLAWD_MINT);
      if (pair) {
        ov = { symbol: pair.baseToken?.symbol || 'CLAWD', name: pair.baseToken?.name || 'solanaclawd', price: pair.priceUsd, marketCap: pair.marketCap, liquidity: pair.liquidity?.usd, v24hChangePercent: pair.priceChange?.h24 };
        lp = { value: pair.priceUsd, priceChange24h: pair.priceChange?.h24 };
        sources.clawd = 'api.dexscreener.com · $CLAWD';
      }
    } catch { /* keep empty pinned row */ }
  } else {
    sources.clawd = 'pumpai.ai/tools · get_token_overview (Birdeye)';
  }
  entries.push(row(
    '★',
    ov.name || 'solanaclawd',
    ov.symbol || 'CLAWD',
    'ecosystem · pinned',
    lp.value || ov.price,
    ov.marketCap || ov.mc,
    ov.liquidity,
    lp.priceChange24h ?? ov.v24hChangePercent ?? ov.priceChange24hPercent,
    CLAWD_MINT,
    { pinned: true },
  ));

  const listCall = await safeCall(toolsMcp, 'list_tokens', { sort_by: 'v24hUSD', sort_type: 'desc', limit: 40 });
  let trending = [];
  if (!listCall.error) {
    const data = unwrap(listCall.value);
    const toks = data?.tokens || [];
    sources.trending = 'pumpai.ai/tools · list_tokens (Birdeye)';
    let rank = 0;
    for (const t of toks) {
      if (!t || typeof t !== 'object' || MAJORS.has(t.address) || t.address === CLAWD_MINT) continue;
      rank += 1;
      if (rank > 12) break;
      trending.push(row(rank, t.name, t.symbol, 'trending · Birdeye list_tokens', t.price, t.mc, t.liquidity, t.v24hChangePercent, t.address));
    }
  }
  if (!trending.length) {
    trending = await dexTrending();
    if (trending.length) sources.trending = 'api.dexscreener.com · token-boosts/top';
  }
  entries.push(...trending);

  const launchCall = await safeCall(pumpMcp, 'get_recent_launches', { limit: 15 });
  if (!launchCall.error && Array.isArray(launchCall.value)) {
    sources.launch_tape = 'pumpai.ai/mcp · get_recent_launches (clawd-pump)';
    for (const item of launchCall.value.slice(0, 10)) {
      if (!item || !isMint(item.mint)) continue;
      entries.push(row(
        'new',
        item.name || '(unnamed)',
        item.symbol || '—',
        `launch tape · ${item.launchpad || item.source || 'pump'}`,
        null, null, null, null,
        item.mint,
        { fresh: true },
      ));
    }
  } else {
    sources.launch_tape = launchCall.error ? `unavailable (${launchCall.error})` : 'unavailable';
  }

  const payload = {
    ok: true,
    refreshed_at: new Date().toISOString(),
    pinned: CLAWD_MINT,
    sources,
    entries,
    note: 'Board sources: pumpai.ai (Birdeye + clawd-pump) with DexScreener fallback. Research only, not financial advice.',
  };
  boardCache = { at: Date.now(), payload };
  return payload;
}

export function normalizeAnswers(rawAnswers, questions) {
  const out = {};
  for (const [qid, raw0] of Object.entries(rawAnswers || {})) {
    if (!raw0 || typeof raw0 !== 'object') continue;
    const raw = { ...raw0 };
    if (!raw.type && questions?.[qid]?.type) raw.type = questions[qid].type;
    if (raw.type === 'choice' && raw.choice == null && raw.answer != null) raw.choice = raw.answer;
    out[qid] = raw;
  }
  return out;
}

export async function askClef({ mint, state, questions } = {}) {
  rejectSecretPayload({ mint, state, questions });
  if (!clefConfigured()) {
    throw Object.assign(new Error('Live Clef is not configured'), { status: 503, code: 'unconfigured' });
  }
  let sources = {};
  let resolved = state;
  const cleanMint = typeof mint === 'string' ? mint.trim() : '';
  if (cleanMint) {
    if (!isMint(cleanMint)) throw Object.assign(new Error('Invalid mint address'), { status: 400, code: 'invalid_mint' });
    const fetched = await fetchTokenState(cleanMint);
    resolved = fetched.state;
    sources = fetched.sources;
  }
  if (resolved == null || (typeof resolved !== 'object' && typeof resolved !== 'string')) {
    throw Object.assign(new Error('missing state'), { status: 400, code: 'missing_state' });
  }
  rejectSecretPayload(resolved);
  const q = questions && typeof questions === 'object' && Object.keys(questions).length ? questions : DEFAULT_QUESTIONS;
  rejectSecretPayload(q);

  const account = process.env.CF_ACCOUNT_ID.trim();
  const token = process.env.CF_API_TOKEN.trim();
  const url = `${API_BASE}/accounts/${account}/ai/run/${encodeURIComponent(MODEL_ID)}`;
  let body;
  try {
    body = await fetchJson(url, {
      method: 'POST',
      timeout: 120000,
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({ model: 'clef', state: resolved, questions: q }),
    });
  } catch (error) {
    const status = error.status && error.status >= 400 ? 502 : 502;
    throw Object.assign(new Error('Live Clef request failed'), { status, code: 'clef_upstream' });
  }
  if (body && body.success === false) {
    throw Object.assign(new Error('Live Clef request failed'), { status: 502, code: 'clef_upstream' });
  }
  const result = body?.result || body || {};
  const answers = normalizeAnswers(result.answers || {}, q);
  if (!Object.keys(answers).length) {
    throw Object.assign(new Error('Live Clef returned no answers'), { status: 502, code: 'clef_empty' });
  }
  return {
    ok: true,
    engine: ENGINE_LABEL,
    model: MODEL_ID,
    demo: false,
    answers,
    questions: q,
    titles: QUESTION_TITLES,
    state: resolved,
    sources,
    asked_at: new Date().toISOString(),
    disclaimer: 'Research only. Typed decisions are not financial advice, a recommendation, or a promise of returns.',
  };
}

export async function getTokenState(mint) {
  if (!isMint(mint)) throw Object.assign(new Error('Invalid mint address'), { status: 400, code: 'invalid_mint' });
  rejectSecretPayload(mint);
  const { state, sources } = await fetchTokenState(mint.trim());
  return { ok: true, state, sources, fetched_at: new Date().toISOString() };
}

export function clefHealth() {
  const configured = clefConfigured();
  return {
    ok: configured,
    configured,
    engine: configured ? ENGINE_LABEL : 'unconfigured',
    model: MODEL_ID,
    demo: false,
  };
}
