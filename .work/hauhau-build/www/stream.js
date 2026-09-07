/* Live launch radar: bounded state, batched DOM updates, no token HTML injection. */
(function (root) {
  "use strict";
  var MAX_EVENTS = 160;
  var RENDER_DELAY = 250;
  var DEFAULT_URL = "wss://clawd-ws.fly.dev/ws";
  var MINT = /^[1-9A-HJ-NP-Za-km-z]{32,44}$/;

  function safeText(value, max) {
    return typeof value === "string" ? value.slice(0, max || 120) : "";
  }

  function number(value) {
    if (value === null || value === undefined || value === "") return null;
    var n = Number(value);
    return Number.isFinite(n) && n >= 0 ? n : null;
  }

  function parseMessage(raw) {
    if (typeof raw === "string" && raw.length > 100000) return null;
    var data;
    try { data = typeof raw === "string" ? JSON.parse(raw) : raw; }
    catch (_) { return null; }
    if (!data || typeof data !== "object" || Array.isArray(data)) return null;
    if (data.type === "status") {
      return { type: "status", connected: typeof data.connected === "boolean" ? data.connected : null,
        totalLaunches: number(data.totalLaunches), clients: number(data.clients) };
    }
    if (data.type !== "token-launch" && data.type !== "token-enriched") return null;
    if (typeof data.mint !== "string" || !MINT.test(data.mint)) return null;
    var msg = { type: data.type, mint: data.mint };
    if (typeof data.name === "string") msg.name = safeText(data.name, 120);
    if (typeof data.symbol === "string") msg.symbol = safeText(data.symbol, 32);
    if (number(data.marketCapSol) !== null) msg.marketCapSol = number(data.marketCapSol);
    var time = typeof data.time === "number" || typeof data.time === "string" ? new Date(data.time).getTime() : NaN;
    if (Number.isFinite(time)) msg.time = time;
    return msg;
  }

  function createState() {
    return { tokens: [], pending: new Map(), status: null };
  }

  function applyMessage(state, msg) {
    if (!msg) return false;
    if (msg.type === "status") { state.status = msg; return false; }
    var index = state.tokens.findIndex(function (token) { return token.mint === msg.mint; });
    if (msg.type === "token-enriched") {
      if (index !== -1) {
        state.tokens[index] = Object.assign({}, state.tokens[index], msg);
        return true;
      }
      state.pending.set(msg.mint, Object.assign({}, state.pending.get(msg.mint), msg));
      if (state.pending.size > MAX_EVENTS) state.pending.delete(state.pending.keys().next().value);
      return false;
    }
    if (index !== -1) {
      state.tokens[index] = Object.assign({}, state.tokens[index], msg);
      return true;
    }
    state.tokens.unshift(Object.assign({}, msg, state.pending.get(msg.mint)));
    state.pending.delete(msg.mint);
    if (state.tokens.length > MAX_EVENTS) state.tokens.length = MAX_EVENTS;
    return true;
  }

  function streamUrl(doc) {
    var configured = doc.documentElement.getAttribute("data-pump-ws-url");
    try {
      var url = new URL(configured || DEFAULT_URL);
      if (url.protocol === "wss:" && !url.username && !url.password) return url.href;
    } catch (_) { /* Use the known public relay when configuration is invalid. */ }
    return DEFAULT_URL;
  }

  function boot() {
    var doc = root.document;
    if (!doc) return;
    var panel = doc.getElementById("pump-stream");
    if (!panel || panel.dataset.streamBooted) return;
    panel.dataset.streamBooted = "true";
    function el(id) { return doc.getElementById(id); }
    function text(node, value) { if (node && node.textContent !== value) node.textContent = value; }
    var log = el("pump-log");
    var statusEl = el("pump-status");
    var searchEl = el("stream-search");
    var pauseEl = el("stream-pause");
    var noticeEl = el("stream-notice");
    var state = createState();
    var paused = false;
    var pausedTokens = [];
    var renderTimer = null;
    var reconnectTimer = null;
    var freshTimer = null;
    var socket = null;
    var retries = 0;
    var stopped = false;
    var lastMessageAt = 0;
    var openedAt = 0;
    var connection = "connecting";
    var connectionLabel = "Connecting…";
    var connectionNotice = "";
    var rowCache = new Map();
    var needsRender = false;

    function showStatus() {
      var now = Date.now();
      var age = lastMessageAt ? Math.floor((now - lastMessageAt) / 1000) : null;
      var label = connectionLabel;
      var mode = connection;
      var notice = connectionNotice;
      if (connection === "live" && state.status && state.status.connected === false) {
        mode = "stale";
        label = "Upstream offline";
        notice = "Relay connected. The upstream launch feed is offline; waiting for recovery.";
      } else if (connection === "live" && now - (lastMessageAt || openedAt) > 30000) {
        mode = "stale";
        label = "Feed quiet";
        notice = "No message in 30 seconds. Retaining the last received launches.";
      }
      if (paused) notice = "Display paused. New launches keep arriving; resume to catch up." + (notice ? " " + notice : "");
      statusEl.dataset.state = mode;
      text(statusEl, label);
      text(noticeEl, notice);
      noticeEl.hidden = !notice;
      text(el("stream-total"), state.status && state.status.totalLaunches !== null ? state.status.totalLaunches.toLocaleString() : "—");
      text(el("stream-count"), state.tokens.length.toLocaleString());
      text(el("stream-freshness"), age === null ? "Waiting" : age < 1 ? "Just now" : age < 60 ? age + "s ago" : Math.floor(age / 60) + "m ago");
      if (connection === "live" && now - (lastMessageAt || openedAt) > 90000 && socket) {
        connectionNotice = "The relay stopped sending messages. Reconnecting…";
        socket.close();
      }
    }

    function node(tag, className, value) {
      var item = doc.createElement(tag);
      if (className) item.className = className;
      if (value !== undefined) item.textContent = value;
      return item;
    }

    function createRow(token) {
      var row = node("div", "token-row");
      row.dataset.mint = token.mint;
      var identity = node("div", "token-identity");
      var avatar = node("span", "token-avatar");
      avatar.setAttribute("aria-hidden", "true");
      var copy = node("div", "token-copy");
      var title = node("div", "token-title");
      var symbol = node("span", "token-symbol");
      var name = node("span", "token-name");
      title.append(symbol, name);
      var mint = node("a", "token-mint", token.mint.slice(0, 5) + "…" + token.mint.slice(-4));
      mint.href = "https://orbmarkets.io/token/" + encodeURIComponent(token.mint);
      mint.target = "_blank";
      mint.rel = "noopener noreferrer";
      mint.title = token.mint;
      mint.setAttribute("aria-label", "View token " + token.mint + " on Orb");
      copy.append(title, mint);
      identity.append(avatar, copy);
      var value = node("span", "token-value");
      var time = node("time", "token-time");
      var trade = node("a", "token-trade button button-small", "Trade ↗");
      trade.href = "#trade";
      trade.dataset.tradeMint = token.mint;
      row.append(identity, value, time, trade);
      return { row: row, avatar: avatar, symbol: symbol, name: name, value: value, time: time, trade: trade };
    }

    function updateRow(ref, token) {
      var symbol = token.symbol || "TOKEN";
      text(ref.avatar, symbol.slice(0, 2).toUpperCase());
      text(ref.symbol, symbol);
      text(ref.name, token.name || "");
      ref.symbol.title = symbol;
      ref.name.title = token.name || "";
      var amount = token.marketCapSol;
      text(ref.value, amount === null || amount === undefined ? "—" : amount.toLocaleString(undefined, { maximumFractionDigits: 2 }) + " SOL");
      if (token.time) {
        var date = new Date(token.time);
        text(ref.time, date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }));
        ref.time.dateTime = date.toISOString();
        ref.time.title = date.toLocaleString();
      } else text(ref.time, "—");
      ref.trade.dataset.tradeSymbol = token.symbol || "";
      ref.trade.setAttribute("aria-label", "Trade " + symbol);
    }

    function render() {
      needsRender = false;
      var tokens = paused ? pausedTokens : state.tokens;
      var query = searchEl.value.trim().toLowerCase();
      var visible = tokens.filter(function (token) {
        return !query || [token.name, token.symbol, token.mint].join(" ").toLowerCase().includes(query);
      });
      var retained = new Set(state.tokens.concat(pausedTokens).map(function (token) { return token.mint; }));
      rowCache.forEach(function (_, mint) { if (!retained.has(mint)) rowCache.delete(mint); });
      if (!visible.length) {
        var empty = node("div", "feed-empty");
        empty.append(node("span", "empty-symbol", query ? "⌕" : "⌁"),
          node("strong", "", query ? "No matching launches" : "Listening for the next launch"),
          node("p", "", query ? "Try a ticker, token name, or complete mint address." : "New tokens appear here as clawd-ws delivers them."));
        log.replaceChildren(empty);
      } else {
        var cursor = log.firstChild;
        visible.forEach(function (token) {
          var ref = rowCache.get(token.mint);
          if (!ref) { ref = createRow(token); rowCache.set(token.mint, ref); }
          updateRow(ref, token);
          if (ref.row === cursor) cursor = cursor.nextSibling;
          else log.insertBefore(ref.row, cursor);
        });
        while (cursor) { var next = cursor.nextSibling; log.removeChild(cursor); cursor = next; }
      }
      text(el("stream-buffer"), query ? visible.length + " matching · latest 160" : "Latest 160 launches");
      showStatus();
    }

    function scheduleRender() {
      needsRender = true;
      if (paused || doc.hidden || renderTimer) return;
      renderTimer = root.setTimeout(function () { renderTimer = null; render(); }, RENDER_DELAY);
    }

    function scheduleReconnect() {
      if (stopped || reconnectTimer) return;
      var delay = Math.min(30000, 1000 * Math.pow(2, Math.min(retries++, 5)));
      delay = Math.round(delay * (0.8 + Math.random() * 0.4));
      connection = "error";
      connectionLabel = "Reconnecting…";
      connectionNotice = "Stream disconnected. Retrying in " + Math.ceil(delay / 1000) + "s; your feed is retained.";
      showStatus();
      reconnectTimer = root.setTimeout(function () { reconnectTimer = null; connect(); }, delay);
    }

    function connect() {
      if (stopped || socket) return;
      connection = "connecting";
      connectionLabel = "Connecting…";
      showStatus();
      var ws;
      try { ws = new root.WebSocket(streamUrl(doc)); }
      catch (_) { scheduleReconnect(); return; }
      socket = ws;
      ws.onopen = function () {
        if (socket !== ws || stopped) return;
        openedAt = Date.now();
        lastMessageAt = 0;
        state.status = null;
        connection = "live";
        connectionLabel = "Connected";
        connectionNotice = "";
        showStatus();
      };
      ws.onmessage = function (event) {
        if (socket !== ws || stopped) return;
        var msg = parseMessage(event.data);
        if (!msg) return;
        lastMessageAt = Date.now();
        retries = 0;
        if (msg.type === "token-launch" || (msg.type === "status" && msg.connected === true)) connectionLabel = "Live";
        if (applyMessage(state, msg)) scheduleRender();
        if (msg.type === "status") showStatus();
      };
      ws.onerror = function () {
        if (socket !== ws || stopped) return;
        connection = "error";
        connectionLabel = "Connection error";
        connectionNotice = "The launch relay could not be reached. Reconnecting automatically.";
        showStatus();
        ws.close();
      };
      ws.onclose = function () {
        if (socket !== ws) return;
        socket = null;
        scheduleReconnect();
      };
    }

    pauseEl.addEventListener("click", function () {
      paused = !paused;
      pausedTokens = paused ? state.tokens.slice() : [];
      pauseEl.setAttribute("aria-pressed", String(paused));
      text(pauseEl, paused ? "Resume" : "Pause");
      render();
    });
    searchEl.addEventListener("input", render);
    el("stream-clear").addEventListener("click", function () {
      state.tokens = [];
      state.pending.clear();
      pausedTokens = [];
      rowCache.clear();
      render();
    });
    log.addEventListener("click", function (event) {
      var trade = event.target.closest("[data-trade-mint]");
      if (!trade || !log.contains(trade)) return;
      doc.dispatchEvent(new root.CustomEvent("hauhau:trade-token", {
        bubbles: true, detail: { mint: trade.dataset.tradeMint, symbol: trade.dataset.tradeSymbol || "" }
      }));
    });
    doc.addEventListener("visibilitychange", function () { if (!doc.hidden && needsRender && !paused) render(); });
    root.addEventListener("online", function () {
      if (socket && connection === "live") return;
      if (reconnectTimer) { root.clearTimeout(reconnectTimer); reconnectTimer = null; }
      connect();
    });
    root.addEventListener("pagehide", function () {
      stopped = true;
      root.clearTimeout(renderTimer);
      root.clearTimeout(reconnectTimer);
      root.clearInterval(freshTimer);
      if (socket) { socket.close(); socket = null; }
    });
    root.addEventListener("pageshow", function (event) {
      if (!event.persisted) return;
      stopped = false;
      renderTimer = null;
      reconnectTimer = null;
      freshTimer = root.setInterval(showStatus, 1000);
      connect();
      render();
    });
    if (typeof root.WebSocket !== "function") {
      connection = "error";
      connectionLabel = "Unavailable";
      connectionNotice = "This browser does not support WebSocket. Open the clawd-ws source link below.";
      showStatus();
      return;
    }
    freshTimer = root.setInterval(showStatus, 1000);
    connect();
  }

  root.HauhauStream = { MAX_EVENTS: MAX_EVENTS, RENDER_DELAY: RENDER_DELAY, parseMessage: parseMessage,
    createState: createState, applyMessage: applyMessage, streamUrl: streamUrl, boot: boot };
  if (root.document) {
    if (root.document.readyState === "loading") root.document.addEventListener("DOMContentLoaded", boot, { once: true });
    else boot();
  }
})(typeof window !== "undefined" ? window : globalThis);
