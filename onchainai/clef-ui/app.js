const CLAWD_MINT = "8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump";

const money = (n) => {
  if (n == null || Number.isNaN(Number(n))) return "—";
  const x = Number(n);
  if (Math.abs(x) >= 1e9) return "$" + (x / 1e9).toFixed(2) + "B";
  if (Math.abs(x) >= 1e6) return "$" + (x / 1e6).toFixed(2) + "M";
  if (Math.abs(x) >= 1e3) return "$" + (x / 1e3).toFixed(2) + "K";
  return "$" + (x < 1 ? x.toPrecision(3) : x.toFixed(2));
};

const pct = (n) => {
  if (n == null || Number.isNaN(Number(n))) return "—";
  const x = Number(n);
  const sign = x > 0 ? "+" : "";
  return sign + x.toFixed(2) + "%";
};

const shortMint = (m) => !m ? "—" : m.slice(0, 4) + "…" + m.slice(-4);

async function api(path, options) {
  const res = await fetch(path, {
    headers: { "content-type": "application/json", ...(options?.headers || {}) },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || data.message || "Request failed");
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

function bar(p) {
  const w = Math.max(0, Math.min(100, Math.round(Number(p || 0) * 100)));
  return `<div class="bar" aria-valuenow="${w}"><span style="width:${w}%"></span></div>`;
}

function renderAnswer(id, answer, titles) {
  const title = (titles && titles[id]) || id;
  const type = answer?.type || "answer";
  let body = "";
  if (type === "noul") {
    const p = Number(answer.noul ?? answer.p ?? 0);
    body = `<div class="opt"><span>Yes</span><strong>${(p * 100).toFixed(1)}%</strong></div>${bar(p)}`;
  } else if (type === "score") {
    const criteria = answer.criteria || [];
    const probs = answer.probs || answer.probabilities || {};
    const score = answer.score;
    const items = Array.isArray(criteria) && criteria.length
      ? criteria.map((label, i) => {
          const p = Number(probs[i] ?? probs[label] ?? 0);
          return `<div class="opt"><span>${label}</span><strong>${(p * 100).toFixed(1)}%</strong></div>${bar(p)}`;
        }).join("")
      : Object.entries(probs).map(([k, p]) => `<div class="opt"><span>${k}</span><strong>${(Number(p) * 100).toFixed(1)}%</strong></div>${bar(p)}`).join("");
    body = `<div class="opt"><span>Weighted score</span><strong>${score == null ? "—" : Number(score).toFixed(3)}</strong></div>${items}`;
  } else if (type === "choice") {
    const choice = answer.choice || answer.answer;
    const probs = answer.probs || answer.probabilities || {};
    body = Object.entries(probs).map(([k, p]) => {
      const win = k === choice ? " winner" : "";
      return `<div class="opt${win}"><span>${k}${k === choice ? " · selected" : ""}</span><strong>${(Number(p) * 100).toFixed(1)}%</strong></div>${bar(p)}`;
    }).join("") || `<div class="opt winner"><span>${choice}</span><strong>selected</strong></div>`;
  } else {
    body = `<pre class="mint">${JSON.stringify(answer, null, 2)}</pre>`;
  }
  return `<article class="card"><div class="kind">${type}</div><h3>${title}</h3>${body}</article>`;
}

function renderAnswers(payload, mount) {
  const answers = payload.answers || {};
  const titles = payload.titles || {};
  const cards = Object.keys(answers).map((id) => renderAnswer(id, answers[id], titles)).join("");
  mount.innerHTML = `
    <p class="engine">${payload.engine || "LIVE · Clef 27B"}</p>
    <div class="answers">${cards || "<p class='err'>No answers returned.</p>"}</div>
    <p class="note">${payload.disclaimer || "Research only. Not financial advice."}</p>
  `;
}

async function loadHealth(badge) {
  try {
    const h = await api("/api/clef/health");
    if (h.configured) {
      badge.className = "badge";
      badge.textContent = "Live Clef 27B";
    } else {
      badge.className = "badge warn";
      badge.textContent = "Clef not configured";
    }
  } catch {
    badge.className = "badge warn";
    badge.textContent = "Engine unreachable";
  }
}

function page() {
  return document.body.dataset.page || "index";
}

async function initIndex() {
  const badge = document.querySelector("#engine-badge");
  const tableBody = document.querySelector("#board-body");
  const note = document.querySelector("#board-note");
  const pick = document.querySelector("#pick");
  const mint = document.querySelector("#mint");
  const results = document.querySelector("#results");
  const run = document.querySelector("#run");
  const refresh = document.querySelector("#refresh");
  await loadHealth(badge);
  mint.value = CLAWD_MINT;

  const fillBoard = async () => {
    note.textContent = "Refreshing live board…";
    const board = await api("/api/clef/board");
    tableBody.innerHTML = (board.entries || []).map((row) => {
      const change = row.price_change_24h_pct;
      const cls = change == null ? "" : change >= 0 ? "up" : "down";
      return `<tr data-mint="${row.mint || ""}" class="${row.pinned ? "pinned" : ""}">
        <td>${row.rank ?? ""}</td>
        <td><strong>${row.symbol || "?"}</strong> · ${row.name || ""}</td>
        <td>${row.source || ""}</td>
        <td>${money(row.price_usd)}</td>
        <td>${money(row.market_cap_usd)}</td>
        <td>${money(row.liquidity_usd)}</td>
        <td class="${cls}">${pct(change)}</td>
        <td class="mint" title="${row.mint || ""}">${shortMint(row.mint)}</td>
      </tr>`;
    }).join("");
    pick.innerHTML = (board.entries || []).map((row) => `<option value="${row.mint}">${row.symbol} — ${row.name}${row.pinned ? " (pinned)" : ""}</option>`).join("");
    pick.value = CLAWD_MINT;
    note.textContent = `Board refreshed ${board.refreshed_at || ""}. ${board.note || ""}`;
    tableBody.querySelectorAll("tr").forEach((tr) => {
      tr.addEventListener("click", () => {
        tableBody.querySelectorAll("tr").forEach((x) => x.classList.remove("active"));
        tr.classList.add("active");
        mint.value = tr.dataset.mint;
        pick.value = tr.dataset.mint;
      });
    });
  };

  pick.addEventListener("change", () => { mint.value = pick.value; });
  refresh.addEventListener("click", () => fillBoard().catch((e) => { note.textContent = e.message; }));
  run.addEventListener("click", async () => {
    run.disabled = true;
    results.innerHTML = `<p class="note"><span class="spin"></span> Asking live Clef 27B over this token's on-chain state…</p>`;
    try {
      const payload = await api("/api/clef/ask", { method: "POST", body: JSON.stringify({ mint: mint.value.trim() }) });
      renderAnswers(payload, results);
    } catch (e) {
      results.innerHTML = `<p class="err">${e.message}</p>`;
    } finally {
      run.disabled = false;
    }
  });
  await fillBoard().catch((e) => { note.textContent = e.message; });
}

async function initClef() {
  const badge = document.querySelector("#engine-badge");
  const mint = document.querySelector("#mint");
  const state = document.querySelector("#state");
  const results = document.querySelector("#results");
  const load = document.querySelector("#load-state");
  const run = document.querySelector("#run");
  await loadHealth(badge);
  mint.value = CLAWD_MINT;
  state.value = JSON.stringify({
    token: { symbol: "CLAWD", name: "solanaclawd", mint: CLAWD_MINT },
    note: "Replace or load live market state, then ask Clef.",
  }, null, 2);

  load.addEventListener("click", async () => {
    load.disabled = true;
    try {
      const payload = await api("/api/clef/state?mint=" + encodeURIComponent(mint.value.trim()));
      state.value = JSON.stringify(payload.state, null, 2);
    } catch (e) {
      results.innerHTML = `<p class="err">${e.message}</p>`;
    } finally {
      load.disabled = false;
    }
  });

  run.addEventListener("click", async () => {
    run.disabled = true;
    results.innerHTML = `<p class="note"><span class="spin"></span> Asking live Clef 27B…</p>`;
    try {
      let parsed;
      try { parsed = JSON.parse(state.value); }
      catch { throw new Error("State must be valid JSON."); }
      const payload = await api("/api/clef/ask", { method: "POST", body: JSON.stringify({ state: parsed }) });
      renderAnswers(payload, results);
    } catch (e) {
      results.innerHTML = `<p class="err">${e.message}</p>`;
    } finally {
      run.disabled = false;
    }
  });
}

loadHealth(document.querySelector("#engine-badge"));
if (page() === "clef") initClef();
else initIndex();
