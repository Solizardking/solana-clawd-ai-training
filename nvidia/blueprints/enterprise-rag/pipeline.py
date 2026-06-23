"""
Blueprint 5 — Enterprise RAG: FastAPI pipeline endpoint.

Serves the full RAG pipeline as an HTTP API, compatible with the
Clawd agent skill system and ClawdRouter.

Usage:
  python3 pipeline.py --store ../../data/nvidia_rag_store --port 8765

  curl http://localhost:8765/query \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the SOL-PERP funding rate?"}'
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import html
import json
import os
import secrets
import time
import urllib.parse
from pathlib import Path
from typing import Any

from query import context_only_enabled, rag_query, rag_query_with_sources, set_context_only_override

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    BaseModel = object  # type: ignore[misc,assignment]
    Field = lambda default_factory=None: []  # type: ignore[assignment]

try:
    from fastapi import Request
except ImportError:  # pragma: no cover
    Request = Any  # type: ignore[misc,assignment]


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    question: str
    sources: list[dict[str, Any]] = Field(default_factory=list)


ADMIN_COOKIE = "clawd_rag_admin"
ADMIN_MAX_AGE_SECONDS = 24 * 60 * 60
RUNTIME_SETTINGS: dict[str, Any] = {
    "default_top_k": int(os.environ.get("CLAWD_RAG_DEFAULT_TOP_K", "5")),
    "max_top_k": int(os.environ.get("CLAWD_RAG_MAX_TOP_K", "12")),
    "context_only_override": None,
}


def _clamp_top_k(value: int | None) -> int:
    default = int(RUNTIME_SETTINGS["default_top_k"])
    max_top_k = max(1, int(RUNTIME_SETTINGS["max_top_k"]))
    if value is None:
        value = default
    return max(1, min(int(value), max_top_k))


def _admin_key() -> str:
    return os.environ.get("CLAWD_RAG_ADMIN_KEY", "")


def _sign_admin_cookie(ts: int) -> str:
    payload = f"admin:{ts}"
    digest = hmac.new(_admin_key().encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{ts}.{digest}"


def _verify_admin_cookie(token: str | None) -> bool:
    key = _admin_key()
    if not key or not token or "." not in token:
        return False
    ts_raw, signature = token.split(".", 1)
    try:
        ts = int(ts_raw)
    except ValueError:
        return False
    if time.time() - ts > ADMIN_MAX_AGE_SECONDS:
        return False
    expected = _sign_admin_cookie(ts).split(".", 1)[1]
    return secrets.compare_digest(signature, expected)


async def _read_form(request: Any) -> dict[str, str]:
    body = (await request.body()).decode("utf-8")
    parsed = urllib.parse.parse_qs(body, keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


def _runtime_status(store_path: Path) -> dict[str, Any]:
    chunks_path = store_path / "chunks.jsonl"
    manifest_path = store_path / "manifest.json"
    chunk_count = 0
    if chunks_path.exists():
        with chunks_path.open() as f:
            chunk_count = sum(1 for line in f if line.strip())
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {"error": "manifest is not valid JSON"}
    return {
        "ok": True,
        "store": str(store_path),
        "store_ready": (store_path / "index.faiss").exists() and chunks_path.exists(),
        "chunks": chunk_count,
        "manifest": manifest,
        "nvidia_generation": bool(os.environ.get("NVIDIA_API_KEY")),
        "context_only": context_only_enabled(),
        "default_top_k": int(RUNTIME_SETTINGS["default_top_k"]),
        "max_top_k": int(RUNTIME_SETTINGS["max_top_k"]),
        "admin_enabled": bool(_admin_key()),
    }


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} | Solana Clawd RAG</title>
  <link rel="icon" href="/favicon.ico" type="image/svg+xml">
  <style>
    :root {{
      color-scheme: dark;
      --bg: #080a0f;
      --panel: #121722;
      --panel-2: #171f2c;
      --text: #edf2f7;
      --muted: #9ca8b8;
      --line: #283243;
      --green: #55d68b;
      --purple: #9945FF;
      --solana-green: #14F195;
      --yellow: #ffd166;
      --cyan: #62d5ff;
      --red: #ff6b6b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
    }}
    a {{ color: var(--solana-green); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    header, main {{ width: min(1120px, calc(100vw - 32px)); margin: 0 auto; }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 22px 0;
      border-bottom: 1px solid var(--line);
    }}
    nav {{ display: flex; gap: 14px; flex-wrap: wrap; font-size: 14px; }}
    main {{ padding: 32px 0 56px; }}
    h1 {{ margin: 0 0 12px; font-size: clamp(34px, 5vw, 64px); line-height: 1.02; letter-spacing: 0; }}
    h2 {{ margin: 34px 0 12px; font-size: 24px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; letter-spacing: 0; }}
    p {{ color: var(--muted); max-width: 850px; }}
    code, pre {{ font-family: "SFMono-Regular", Consolas, monospace; }}
    pre {{
      overflow-x: auto;
      background: #07090d;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      color: #dbeafe;
    }}
    input, textarea, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #090d14;
      color: var(--text);
      padding: 10px 12px;
      font: inherit;
    }}
    textarea {{ min-height: 110px; resize: vertical; }}
    button {{
      border: 0;
      border-radius: 8px;
      background: var(--purple);
      color: #ffffff;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
    }}
    button.secondary {{ background: var(--panel-2); color: var(--text); border: 1px solid var(--line); }}
    .brand {{ display: flex; align-items: center; gap: 10px; font-weight: 800; }}
    .mark {{
      width: 34px; height: 34px; border-radius: 8px;
      display: grid; place-items: center;
      background: linear-gradient(135deg, var(--purple), var(--solana-green));
      color: #061016; font-weight: 900;
    }}
    .lede {{ font-size: 18px; color: #c9d4e5; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .card {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }}
    .metric {{ color: var(--muted); font-size: 13px; }}
    .metric strong {{ display: block; color: var(--text); font-size: 22px; margin-top: 4px; }}
    .status-ok {{ color: var(--green); }}
    .status-warn {{ color: var(--yellow); }}
    .status-bad {{ color: var(--red); }}
    .pipeline {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; align-items: stretch; }}
    .step {{ border: 1px solid var(--line); background: var(--panel-2); border-radius: 8px; padding: 14px; }}
    .step span {{ display: block; color: var(--solana-green); font-size: 12px; font-weight: 800; text-transform: uppercase; }}
    .form-grid {{ display: grid; grid-template-columns: 1fr 160px; gap: 12px; align-items: end; }}
    .result {{ white-space: pre-wrap; word-break: break-word; }}
    .actions {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
    .dashboard {{ display: grid; gap: 16px; }}
    .hero-row {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 16px; align-items: start; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .wide {{ grid-column: 1 / -1; }}
    .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; justify-content: flex-end; }}
    .pill {{
      display: inline-flex; align-items: center; min-height: 26px;
      border: 1px solid var(--line); border-radius: 999px;
      padding: 0 10px; color: var(--muted); background: var(--panel-2);
      font-size: 12px; font-weight: 700;
    }}
    .pill.ok {{ color: var(--solana-green); border-color: rgba(20, 241, 149, 0.36); background: rgba(20, 241, 149, 0.1); }}
    .pill.warn {{ color: var(--yellow); border-color: rgba(255, 209, 102, 0.36); background: rgba(255, 209, 102, 0.1); }}
    .table-wrap {{ width: 100%; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 720px; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    td {{ word-break: break-word; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    .source-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .source-card {{ border: 1px solid var(--line); background: var(--panel-2); border-radius: 8px; padding: 12px; min-width: 0; }}
    .source-card strong {{ display: block; color: var(--text); margin-bottom: 4px; word-break: break-word; }}
    .source-card span {{ color: var(--muted); font-size: 13px; }}
    .answer {{ white-space: pre-wrap; word-break: break-word; }}
    .raw-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    details {{ border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }}
    summary {{ cursor: pointer; padding: 10px 12px; font-weight: 800; }}
    details pre {{ border: 0; border-top: 1px solid var(--line); border-radius: 0 0 8px 8px; margin: 0; max-height: 360px; white-space: pre-wrap; word-break: break-word; }}
    @media (max-width: 820px) {{
      header {{ align-items: flex-start; flex-direction: column; }}
      .grid, .grid.two, .pipeline, .form-grid, .hero-row, .metric-grid, .source-grid, .raw-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 40px; }}
    }}
  </style>
</head>
<body>
  <header>
    <a class="brand" href="/"><span class="mark">8</span><span>Solana Clawd RAG</span></a>
    <nav>
      <a href="/">Dashboard</a>
      <a href="/about">About</a>
      <a href="/health">Health</a>
      <a href="/docs">API Docs</a>
      <a href="/admin">Admin</a>
    </nav>
  </header>
  <main>{body}</main>
</body>
</html>"""


def _dashboard_html(status: dict[str, Any]) -> str:
    status_json = json.dumps(status).replace("</", "<\\/")
    body = """
<section class="dashboard">
  <div class="hero-row">
    <div>
      <h1>RAG Dashboard</h1>
      <p class="lede">Live store, retrieval, generation, and source coverage for the deployed Solana Clawd RAG service.</p>
    </div>
    <div class="toolbar">
      <button id="refreshBtn" type="button">Refresh</button>
      <a class="pill" href="/docs">API Docs</a>
      <a class="pill" href="/admin">Admin</a>
    </div>
  </div>

  <section class="metric-grid" id="metricGrid"></section>

  <section class="grid two">
    <article class="card">
      <h2 style="margin-top:0">Runtime</h2>
      <div class="table-wrap">
        <table>
          <tbody id="runtimeRows"></tbody>
        </table>
      </div>
    </article>
    <article class="card">
      <h2 style="margin-top:0">Store Manifest</h2>
      <div class="table-wrap">
        <table>
          <tbody id="manifestRows"></tbody>
        </table>
      </div>
    </article>
  </section>

  <section class="card">
    <div class="hero-row">
      <div>
        <h2 style="margin-top:0">Query</h2>
        <p id="queryStatus">Ready</p>
      </div>
      <label style="max-width:150px">top_k
        <input id="topKInput" type="number" min="1" max="12" value="5">
      </label>
    </div>
    <textarea id="questionInput">What does the Solana Clawd NVIDIA RAG pipeline do?</textarea>
    <div class="actions" style="margin-top:12px">
      <button id="queryBtn" type="button">Run Query</button>
      <button id="clearBtn" class="secondary" type="button">Clear</button>
    </div>
    <h3 style="margin-top:18px">Answer</h3>
    <div class="card result answer" id="answerBox">No query yet.</div>
    <h3 style="margin-top:18px">Retrieved Sources</h3>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Source</th><th>Score</th><th>Snippet</th></tr></thead>
        <tbody id="querySourceRows"></tbody>
      </table>
    </div>
  </section>

  <section class="card">
    <div class="hero-row">
      <div>
        <h2 style="margin-top:0">Indexed Sources</h2>
        <p id="sourceSummary">Loading</p>
      </div>
      <input id="sourceFilter" placeholder="Filter sources" spellcheck="false">
    </div>
    <div class="source-grid" id="sourceGrid"></div>
  </section>

  <section class="grid two">
    <article class="card">
      <h2 style="margin-top:0">Endpoints</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Route</th><th>Method</th><th>Surface</th></tr></thead>
          <tbody id="endpointRows"></tbody>
        </table>
      </div>
    </article>
    <article class="card">
      <h2 style="margin-top:0">Raw Data</h2>
      <div class="raw-grid" id="rawPanels"></div>
    </article>
  </section>
</section>
<script id="initialStatus" type="application/json">__STATUS_JSON__</script>
<script>
const state = {
  status: JSON.parse(document.getElementById("initialStatus").textContent),
  query: null,
  error: null
};

const endpoints = [
  ["/", "GET", "dashboard"],
  ["/about", "GET", "public page"],
  ["/health", "GET", "runtime JSON"],
  ["/query", "POST", "RAG answer"],
  ["/docs", "GET", "OpenAPI"],
  ["/admin", "GET", "protected dashboard"],
  ["/admin/api/status", "GET", "protected JSON"]
];

const $ = (selector) => document.querySelector(selector);

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function number(value) {
  const n = Number(value);
  return Number.isFinite(n) ? new Intl.NumberFormat().format(n) : esc(value ?? "");
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
    ...options
  });
  const text = await res.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch { data = {raw: text}; }
  if (!res.ok) throw new Error(data.detail || data.error || `${res.status} ${res.statusText}`);
  return data;
}

function metric(label, value, sub, kind = "") {
  return `<article class="card metric">
    <span>${esc(label)}</span>
    <strong class="${kind}">${esc(value)}</strong>
    <span>${esc(sub || "")}</span>
  </article>`;
}

function rows(items) {
  return items.map(([key, value]) => `<tr><th>${esc(key)}</th><td>${esc(value)}</td></tr>`).join("");
}

function render() {
  const s = state.status || {};
  const m = s.manifest || {};
  const sources = m.sources || {};
  const sourceEntries = Object.entries(sources).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  document.getElementById("metricGrid").innerHTML = [
    metric("Health", s.ok ? "OK" : "Down", s.store_ready ? "store ready" : "store missing", s.ok ? "status-ok" : "status-bad"),
    metric("Chunks", number(s.chunks), `${sourceEntries.length} sources`),
    metric("Generation", s.nvidia_generation ? "NVIDIA" : "Fallback", s.context_only ? "context-only" : "answers enabled", s.nvidia_generation ? "status-ok" : "status-warn"),
    metric("Admin", s.admin_enabled ? "Enabled" : "Off", `top_k ${s.default_top_k}/${s.max_top_k}`, s.admin_enabled ? "status-ok" : "status-warn")
  ].join("");

  document.getElementById("runtimeRows").innerHTML = rows([
    ["Store", s.store],
    ["Store ready", s.store_ready],
    ["NVIDIA generation", s.nvidia_generation],
    ["Context only", s.context_only],
    ["Default top_k", s.default_top_k],
    ["Max top_k", s.max_top_k],
    ["Admin enabled", s.admin_enabled]
  ]);

  document.getElementById("manifestRows").innerHTML = rows([
    ["Generated", m.generated_at || "unknown"],
    ["Embedding model", m.embedding_model || "unknown"],
    ["Embedding dim", m.embedding_dim || ""],
    ["Chunk size", m.chunk_size || ""],
    ["Chunk overlap", m.chunk_overlap || ""],
    ["Chunks", m.chunks || s.chunks || ""]
  ]);

  document.getElementById("endpointRows").innerHTML = endpoints.map(([route, method, surface]) =>
    `<tr><td><code>${esc(route)}</code></td><td>${esc(method)}</td><td>${esc(surface)}</td></tr>`
  ).join("");

  renderSources();
  renderQuery();
  renderRaw();
}

function renderSources() {
  const m = state.status?.manifest || {};
  const entries = Object.entries(m.sources || {}).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const filter = document.getElementById("sourceFilter").value.trim().toLowerCase();
  const filtered = filter ? entries.filter(([source]) => source.toLowerCase().includes(filter)) : entries;
  document.getElementById("sourceSummary").textContent = `${filtered.length} of ${entries.length} sources`;
  document.getElementById("sourceGrid").innerHTML = filtered.map(([source, count]) =>
    `<div class="source-card"><strong>${esc(source)}</strong><span>${number(count)} chunks</span></div>`
  ).join("") || `<div class="source-card"><strong>No sources</strong><span>Adjust filter</span></div>`;
}

function renderQuery() {
  const q = state.query;
  document.getElementById("answerBox").textContent = q?.answer || "No query yet.";
  document.getElementById("querySourceRows").innerHTML = (q?.sources || []).map((source) =>
    `<tr><td>${esc(source.source)}</td><td>${Number(source.score ?? 0).toFixed(4)}</td><td>${esc(source.snippet)}</td></tr>`
  ).join("") || `<tr><td colspan="3">No retrieved sources.</td></tr>`;
}

function renderRaw() {
  const panels = {health: state.status, lastQuery: state.query || {}};
  document.getElementById("rawPanels").innerHTML = Object.entries(panels).map(([name, data]) =>
    `<details><summary>${esc(name)}</summary><pre>${esc(JSON.stringify(data, null, 2))}</pre></details>`
  ).join("");
}

async function refresh() {
  document.getElementById("refreshBtn").disabled = true;
  try {
    state.status = await api("/health");
    state.error = null;
  } catch (err) {
    state.error = err.message;
  } finally {
    document.getElementById("refreshBtn").disabled = false;
    render();
  }
}

async function runQuery() {
  const btn = document.getElementById("queryBtn");
  const status = document.getElementById("queryStatus");
  btn.disabled = true;
  status.textContent = "Running";
  try {
    state.query = await api("/query", {
      method: "POST",
      body: JSON.stringify({
        question: document.getElementById("questionInput").value,
        top_k: Number(document.getElementById("topKInput").value || 5)
      })
    });
    status.textContent = "Complete";
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
  } finally {
    btn.disabled = false;
    render();
  }
}

document.getElementById("refreshBtn").addEventListener("click", refresh);
document.getElementById("queryBtn").addEventListener("click", runQuery);
document.getElementById("clearBtn").addEventListener("click", () => {
  state.query = null;
  document.getElementById("answerBox").textContent = "No query yet.";
  document.getElementById("querySourceRows").innerHTML = `<tr><td colspan="3">No retrieved sources.</td></tr>`;
});
document.getElementById("sourceFilter").addEventListener("input", renderSources);
render();
</script>
"""
    return _page("Dashboard", body.replace("__STATUS_JSON__", status_json))


def _about_html(status: dict[str, Any]) -> str:
    generation = "NVIDIA/NIM" if status["nvidia_generation"] else "retrieval-only"
    body = f"""
<section>
  <h1>Grounded Solana intelligence for Clawd agents</h1>
  <p class="lede">This service is a compact RAG API for the Solana Clawd NVIDIA blueprint stack. It retrieves project context from a FAISS index, optionally reranks and generates with NVIDIA models, and returns answers that are tied back to the local knowledge store.</p>
</section>

<section class="grid">
  <div class="card metric">Store ready<strong class="{'status-ok' if status['store_ready'] else 'status-bad'}">{str(status['store_ready']).lower()}</strong></div>
  <div class="card metric">Indexed chunks<strong>{status['chunks']}</strong></div>
  <div class="card metric">Generation mode<strong>{html.escape(generation)}</strong></div>
</section>

<h2>What the RAG looks like</h2>
<section class="pipeline">
  <div class="step"><span>1. Source</span>Solana docs, Clawd skills, blueprint notes, model-kit material.</div>
  <div class="step"><span>2. Chunk</span>Markdown, JSONL, and PDFs are normalized into retrieval-sized passages.</div>
  <div class="step"><span>3. Embed</span>NVIDIA embeddings when configured, hash fallback for local/dev mode.</div>
  <div class="step"><span>4. Retrieve</span>FAISS returns the nearest project-context chunks for the question.</div>
  <div class="step"><span>5. Answer</span>NVIDIA/NIM generation when keyed, otherwise retrieval-only context with source snippets.</div>
</section>

<h2>Try the API</h2>
<pre>curl -sS https://solana-clawd-rag.fly.dev/query \\
  -H "Content-Type: application/json" \\
  -d '{{"question":"What does the Solana Clawd NVIDIA RAG pipeline do?","top_k":5}}'</pre>

<section class="grid two">
  <div class="card">
    <h3>For users</h3>
    <p>Ask questions about the Solana Clawd training stack, NVIDIA blueprints, model-kit flow, local Mac setup, and RAG/deployment contracts.</p>
  </div>
  <div class="card">
    <h3>For agents</h3>
    <p>Use <code>POST /query</code> as a grounded context endpoint before producing model, trading, DAO, or deployment instructions.</p>
  </div>
</section>
"""
    return _page("About", body)


def _login_html(error: str = "") -> str:
    error_html = f'<p class="status-bad">{html.escape(error)}</p>' if error else ""
    body = f"""
<section>
  <h1>Admin Login</h1>
  <p class="lede">Enter the deployment admin key to manage runtime RAG settings and test endpoints.</p>
  {error_html}
  <form class="card" method="post" action="/admin/login">
    <label>Admin key</label>
    <input name="admin_key" type="password" autocomplete="current-password" required>
    <div style="height:12px"></div>
    <button type="submit">Log in</button>
  </form>
</section>
"""
    return _page("Admin Login", body)


def _admin_setup_html() -> str:
    body = """
<section>
  <h1>Admin Not Configured</h1>
  <p class="lede">Set <code>CLAWD_RAG_ADMIN_KEY</code> as a Fly secret to enable the admin dashboard. Do not commit it to source files.</p>
  <pre>flyctl secrets set CLAWD_RAG_ADMIN_KEY="$CLAWD_RAG_ADMIN_KEY" --app solana-clawd-rag</pre>
</section>
"""
    return _page("Admin Setup", body)


def _admin_html(status: dict[str, Any], result: str = "", question: str = "") -> str:
    mode_value = "env"
    if RUNTIME_SETTINGS["context_only_override"] is True:
        mode_value = "on"
    elif RUNTIME_SETTINGS["context_only_override"] is False:
        mode_value = "off"
    result_html = ""
    if result:
        result_html = f"""
<h2>Query Result</h2>
<div class="card result">{html.escape(result)}</div>
"""
    body = f"""
<section>
  <h1>RAG Admin</h1>
  <p class="lede">Control runtime retrieval settings, inspect endpoint health, and test the public query path.</p>
  <div class="actions">
    <form method="post" action="/admin/logout"><button class="secondary" type="submit">Log out</button></form>
    <a href="/docs">Open API docs</a>
  </div>
</section>

<section class="grid">
  <div class="card metric">Store<strong class="{'status-ok' if status['store_ready'] else 'status-bad'}">{'ready' if status['store_ready'] else 'missing'}</strong></div>
  <div class="card metric">Chunks<strong>{status['chunks']}</strong></div>
  <div class="card metric">NVIDIA generation<strong class="{'status-ok' if status['nvidia_generation'] else 'status-warn'}">{'on' if status['nvidia_generation'] else 'off'}</strong></div>
</section>
<section class="grid two">
  <div class="card metric">Store generated<strong>{html.escape(status.get('manifest', {}).get('generated_at', 'unknown'))}</strong></div>
  <div class="card metric">Indexed sources<strong>{len(status.get('manifest', {}).get('sources', {}))}</strong></div>
</section>

<h2>Runtime Controls</h2>
<form class="card" method="post" action="/admin/settings">
  <div class="grid">
    <label>Default top_k
      <input name="default_top_k" type="number" min="1" max="50" value="{status['default_top_k']}">
    </label>
    <label>Max top_k
      <input name="max_top_k" type="number" min="1" max="50" value="{status['max_top_k']}">
    </label>
    <label>Context-only mode
      <select name="context_mode">
        <option value="env" {'selected' if mode_value == 'env' else ''}>Use env default</option>
        <option value="on" {'selected' if mode_value == 'on' else ''}>Force on</option>
        <option value="off" {'selected' if mode_value == 'off' else ''}>Force off</option>
      </select>
    </label>
  </div>
  <div style="height:12px"></div>
  <button type="submit">Save runtime settings</button>
</form>

<h2>Endpoint Tester</h2>
<form class="card" method="post" action="/admin/query">
  <label>Question</label>
  <textarea name="question">{html.escape(question or 'What does the Solana Clawd NVIDIA RAG pipeline do?')}</textarea>
  <div class="form-grid" style="margin-top:12px">
    <label>top_k
      <input name="top_k" type="number" min="1" max="{status['max_top_k']}" value="{status['default_top_k']}">
    </label>
    <button type="submit">Run query</button>
  </div>
</form>
{result_html}

<h2>Managed Endpoints</h2>
<section class="grid">
  <div class="card"><h3>GET /about</h3><p>Public education page for users and builders.</p></div>
  <div class="card"><h3>GET /health</h3><p>Machine health, store readiness, generation mode, and admin enablement.</p></div>
  <div class="card"><h3>POST /query</h3><p>Public RAG query endpoint with top_k clamped by admin settings.</p></div>
  <div class="card"><h3>GET /admin/api/status</h3><p>Protected runtime status JSON.</p></div>
  <div class="card"><h3>POST /admin/settings</h3><p>Protected runtime setting update form.</p></div>
  <div class="card"><h3>POST /admin/query</h3><p>Protected dashboard query tester.</p></div>
</section>
"""
    return _page("Admin", body)


def make_app(store_path: Path):
    try:
        from fastapi import Body, FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
    except ImportError:
        raise ImportError("Run: pip install fastapi uvicorn")

    app = FastAPI(title="Clawd NVIDIA RAG Pipeline", version="1.0")

    @app.get("/health")
    def health():
        return _runtime_status(store_path)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="12" fill="#080a0f"/>
  <path d="M14 38h10l6-18 8 28 6-14h6" fill="none" stroke="#9945FF" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="50" cy="34" r="4" fill="#14F195"/>
</svg>"""
        return Response(svg, media_type="image/svg+xml")

    @app.get("/", response_class=HTMLResponse)
    def root():
        return HTMLResponse(_dashboard_html(_runtime_status(store_path)))

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest = Body(...)):
        result = rag_query_with_sources(req.question, store_path, _clamp_top_k(req.top_k))
        return QueryResponse(answer=result["answer"], question=req.question, sources=result["sources"])

    @app.get("/about", response_class=HTMLResponse)
    def about():
        return HTMLResponse(_about_html(_runtime_status(store_path)))

    @app.get("/admin", response_class=HTMLResponse)
    def admin(request: Request):
        if not _admin_key():
            return HTMLResponse(_admin_setup_html(), status_code=503)
        if not _verify_admin_cookie(request.cookies.get(ADMIN_COOKIE)):
            return HTMLResponse(_login_html())
        return HTMLResponse(_admin_html(_runtime_status(store_path)))

    @app.post("/admin/login")
    async def admin_login(request: Request):
        form = await _read_form(request)
        key = _admin_key()
        if not key:
            return HTMLResponse(_admin_setup_html(), status_code=503)
        if not secrets.compare_digest(form.get("admin_key", ""), key):
            return HTMLResponse(_login_html("Invalid admin key."), status_code=401)
        response = RedirectResponse("/admin", status_code=303)
        response.set_cookie(
            ADMIN_COOKIE,
            _sign_admin_cookie(int(time.time())),
            max_age=ADMIN_MAX_AGE_SECONDS,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="lax",
        )
        return response

    @app.post("/admin/logout")
    def admin_logout():
        response = RedirectResponse("/admin", status_code=303)
        response.delete_cookie(ADMIN_COOKIE)
        return response

    @app.post("/admin/settings")
    async def admin_settings(request: Request):
        if not _verify_admin_cookie(request.cookies.get(ADMIN_COOKIE)):
            return RedirectResponse("/admin", status_code=303)
        form = await _read_form(request)
        default_top_k = max(1, min(int(form.get("default_top_k", "5")), 50))
        max_top_k = max(default_top_k, min(int(form.get("max_top_k", str(default_top_k))), 50))
        RUNTIME_SETTINGS["default_top_k"] = default_top_k
        RUNTIME_SETTINGS["max_top_k"] = max_top_k
        mode = form.get("context_mode", "env")
        if mode == "on":
            RUNTIME_SETTINGS["context_only_override"] = True
            set_context_only_override(True)
        elif mode == "off":
            RUNTIME_SETTINGS["context_only_override"] = False
            set_context_only_override(False)
        else:
            RUNTIME_SETTINGS["context_only_override"] = None
            set_context_only_override(None)
        return RedirectResponse("/admin", status_code=303)

    @app.post("/admin/query", response_class=HTMLResponse)
    async def admin_query(request: Request):
        if not _verify_admin_cookie(request.cookies.get(ADMIN_COOKIE)):
            return RedirectResponse("/admin", status_code=303)
        form = await _read_form(request)
        question = form.get("question", "").strip()
        top_k = _clamp_top_k(int(form.get("top_k", RUNTIME_SETTINGS["default_top_k"])))
        result = rag_query(question, store_path, top_k) if question else "Enter a question."
        return HTMLResponse(_admin_html(_runtime_status(store_path), result=result, question=question))

    @app.get("/admin/api/status")
    def admin_api_status(request: Request):
        if not _verify_admin_cookie(request.cookies.get(ADMIN_COOKIE)):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
        return _runtime_status(store_path)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve NVIDIA RAG pipeline as API")
    parser.add_argument("--store", default="data/nvidia_rag_store")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    store_path = Path(args.store)
    app = make_app(store_path)

    import uvicorn  # type: ignore
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
