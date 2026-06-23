const state = {
  health: null,
  status: null,
  report: null,
  dataCatalog: null,
  training: null,
  active: null,
  evolution: null,
  scan: null,
  error: null,
  loading: false,
  timer: null,
};

const $ = (selector) => document.querySelector(selector);

const els = {
  refreshBtn: $("#refreshBtn"),
  scanBtn: $("#scanBtn"),
  autoRefresh: $("#autoRefresh"),
  marketsInput: $("#marketsInput"),
  timeframeInput: $("#timeframeInput"),
  statusLine: $("#statusLine"),
  metricGrid: $("#metricGrid"),
  scanRows: $("#scanRows"),
  scanTimestamp: $("#scanTimestamp"),
  dataTimestamp: $("#dataTimestamp"),
  dataRows: $("#dataRows"),
  signalBars: $("#signalBars"),
  verdictChart: $("#verdictChart"),
  regimeChart: $("#regimeChart"),
  discoveryRows: $("#discoveryRows"),
  trainingStats: $("#trainingStats"),
  runtimeStats: $("#runtimeStats"),
  strategyRows: $("#strategyRows"),
  evolutionRows: $("#evolutionRows"),
  rawPanels: $("#rawPanels"),
};

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatNumber(value) {
  if (value === null || value === undefined || value === "") return "0";
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  return new Intl.NumberFormat().format(num);
}

function formatPct(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "0%";
  return `${Math.round(num * 100)}%`;
}

function formatBytes(value) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(num) / Math.log(1024)), units.length - 1);
  return `${(num / (1024 ** index)).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatTime(value) {
  if (!value) return "No timestamp";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function markets() {
  return els.marketsInput.value
    .split(",")
    .map((item) => item.trim().toUpperCase())
    .filter(Boolean);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text };
  }
  if (!res.ok) {
    throw new Error(data.detail || data.error || `${res.status} ${res.statusText}`);
  }
  return data;
}

async function refreshBase() {
  setLoading(true, "Refreshing");
  try {
    const [health, status, report, dataCatalog, training, active, evolution] = await Promise.all([
      api("/api/health"),
      api("/api/status"),
      api("/api/report"),
      api("/api/data/catalog"),
      api("/api/training/status"),
      api("/api/strategy/active"),
      api("/api/evolution?limit=50"),
    ]);
    Object.assign(state, { health, status, report, dataCatalog, training, active, evolution, error: null });
  } catch (error) {
    state.error = error.message;
  } finally {
    setLoading(false);
    render();
  }
}

async function runScan() {
  const selectedMarkets = markets();
  if (!selectedMarkets.length) return;
  setLoading(true, "Scanning");
  try {
    state.scan = await api("/api/scan", {
      method: "POST",
      body: JSON.stringify({
        markets: selectedMarkets,
        timeframe: els.timeframeInput.value,
      }),
    });
    state.error = null;
  } catch (error) {
    state.error = error.message;
  } finally {
    setLoading(false);
    render();
  }
}

function setLoading(isLoading, label = "") {
  state.loading = isLoading;
  els.refreshBtn.disabled = isLoading;
  els.scanBtn.disabled = isLoading;
  els.statusLine.textContent = isLoading ? label : statusText();
}

function statusText() {
  if (state.error) return `Error: ${state.error}`;
  const healthTime = state.health?.timestamp ? formatTime(state.health.timestamp) : "Waiting";
  return `Updated ${healthTime}`;
}

function render() {
  els.statusLine.textContent = state.loading ? els.statusLine.textContent : statusText();
  renderMetrics();
  renderDataCatalog();
  renderScan();
  renderReport();
  renderTraining();
  renderRuntime();
  renderStrategies();
  renderEvolution();
  renderRaw();
}

function renderMetrics() {
  const report = state.report || {};
  const training = state.training || {};
  const active = state.active || {};
  const dataCatalog = state.dataCatalog || {};
  const metrics = [
    ["API", state.health?.ok ? "OK" : "Down", formatTime(state.health?.timestamp)],
    ["Vulcan", state.status?.vulcan_available ? "Ready" : "Missing", state.status?.vulcan_path || "No path"],
    ["Data Files", dataCatalog.count ?? 0, formatBytes(dataCatalog.total_bytes)],
    ["Markets", report.n_markets ?? 0, "latest report"],
    ["Avg Confidence", formatPct(report.avg_confidence ?? 0), "latest report"],
    ["SFT Records", formatNumber(training.data?.sft_records), "training status"],
    ["Active", active.count ?? 0, "strategies"],
  ];

  els.metricGrid.innerHTML = metrics.map(([label, value, sub]) => `
    <article class="metric-card">
      <div class="label">${esc(label)}</div>
      <div class="value">${esc(value)}</div>
      <div class="sub">${esc(sub)}</div>
    </article>
  `).join("");
}

function renderDataCatalog() {
  const catalog = state.dataCatalog || {};
  const files = catalog.files || [];
  els.dataTimestamp.textContent = catalog.timestamp ? formatTime(catalog.timestamp) : "No data";
  els.dataRows.innerHTML = files.length ? files.map((file) => `
    <tr>
      <td>${pill(file.source)}</td>
      <td><strong>${esc(file.path)}</strong></td>
      <td>${esc(file.extension)}</td>
      <td>${esc(recordLabel(file))}</td>
      <td>${esc(formatBytes(file.size_bytes))}</td>
      <td>${esc(formatTime(file.modified))}</td>
      <td>${previewCell(file.preview)}</td>
    </tr>
  `).join("") : emptyRow(7, "No data files found");
}

function renderScan() {
  const rows = state.scan?.markets || [];
  els.scanTimestamp.textContent = state.scan?.timestamp ? formatTime(state.scan.timestamp) : "No scan";
  els.scanRows.innerHTML = rows.length ? rows.map((row) => {
    if (row.error) {
      return `<tr><td>${esc(row.market)}</td><td colspan="8">${esc(row.error)}</td></tr>`;
    }
    return `
      <tr>
        <td><strong>${esc(row.market)}</strong></td>
        <td>${pill(row.regime)}</td>
        <td class="${directionClass(row.composite_direction)}">${esc(row.composite_direction)}</td>
        <td>${Number(row.composite_strength ?? 0).toFixed(3)}</td>
        <td>${pill(row.verdict, verdictKind(row.verdict))}</td>
        <td>${formatPct(row.confidence ?? 0)}</td>
        <td>${Number(row.regime_atr_pct ?? 0).toFixed(3)}</td>
        <td>${Number(row.regime_adx ?? 0).toFixed(2)}</td>
        <td>${esc(row.active_strategy || "none")}</td>
      </tr>
    `;
  }).join("") : emptyRow(9, "No live scan");

  const first = rows.find((row) => Array.isArray(row.signals));
  renderSignalBars(first?.signals || []);
}

function renderSignalBars(signals) {
  if (!signals.length) {
    els.signalBars.innerHTML = `<div class="empty">No signal data</div>`;
    return;
  }
  els.signalBars.innerHTML = signals.map((signal) => {
    const strength = Math.max(0, Math.min(1, Number(signal.strength || 0)));
    const color = signal.direction === "long" ? "green" : signal.direction === "short" ? "red" : "teal";
    return `
      <div class="signal-row">
        <div class="signal-label">
          <strong>${esc(signal.name)}</strong>
          <span class="${directionClass(signal.direction)}">${esc(signal.direction)} ${strength.toFixed(3)}</span>
        </div>
        <div class="track"><div class="fill ${color}" style="width: ${strength * 100}%"></div></div>
        <small>${esc(signal.reason)}</small>
      </div>
    `;
  }).join("");
}

function renderReport() {
  const report = state.report || {};
  renderChart(els.verdictChart, report.verdict_summary || {});
  renderChart(els.regimeChart, report.regime_summary || {});

  const rows = report.discoveries || [];
  els.discoveryRows.innerHTML = rows.length ? rows.map((row) => `
    <tr>
      <td><strong>${esc(row.market)}</strong></td>
      <td>${pill(row.regime)}</td>
      <td class="${directionClass(row.composite_direction)}">${esc(row.composite_direction)}</td>
      <td>${Number(row.composite_strength ?? 0).toFixed(3)}</td>
      <td>${pill(row.verdict, verdictKind(row.verdict))}</td>
      <td>${formatPct(row.confidence ?? 0)}</td>
      <td>${esc((row.risk_flags || []).join(", ") || "none")}</td>
      <td>${esc(row.rationale || "")}</td>
    </tr>
  `).join("") : emptyRow(8, "No discoveries");
}

function renderChart(container, values) {
  const entries = Object.entries(values);
  if (!entries.length) {
    container.innerHTML = `<div class="empty">No chart data</div>`;
    return;
  }
  const max = Math.max(...entries.map(([, value]) => Number(value) || 0), 1);
  container.innerHTML = entries.map(([label, value], index) => {
    const width = ((Number(value) || 0) / max) * 100;
    const colors = ["green", "amber", "teal", "violet", "red"];
    return `
      <div class="chart-row">
        <div class="chart-label"><strong>${esc(label)}</strong><span>${esc(value)}</span></div>
        <div class="track"><div class="fill ${colors[index % colors.length]}" style="width: ${width}%"></div></div>
      </div>
    `;
  }).join("");
}

function renderTraining() {
  const data = state.training?.data || {};
  const checkpoints = state.training?.checkpoints || {};
  const rows = [
    ["CPT Records", formatNumber(data.cpt_records)],
    ["SFT Records", formatNumber(data.sft_records)],
    ["Jupiter Records", formatNumber(data.jupiter_records)],
    ["CPT Data MB", data.cpt_data_mb ?? 0],
    ["CPT Done", checkpoints.cpt_done ? "yes" : "no"],
    ["SFT Done", checkpoints.sft_done ? "yes" : "no"],
  ];
  els.trainingStats.innerHTML = statRows(rows);
}

function renderRuntime() {
  const status = state.status || {};
  const rows = [
    ["Data Dir", status.data_dir || ""],
    ["Python", status.python || ""],
    ["Render", status.env?.RENDER || "local"],
    ["Service", status.env?.SERVICE || "local"],
  ];
  els.runtimeStats.innerHTML = statRows(rows);
}

function renderStrategies() {
  const rows = state.active?.active || [];
  els.strategyRows.innerHTML = rows.length ? rows.map((row) => `
    <tr>
      <td><strong>${esc(row.market)}</strong></td>
      <td>${esc(row.run_id || "")}</td>
      <td><pre>${esc(JSON.stringify(row.status || {}, null, 2))}</pre></td>
    </tr>
  `).join("") : emptyRow(3, "No active strategies");
}

function renderEvolution() {
  const rows = state.evolution?.rows || [];
  els.evolutionRows.innerHTML = rows.length ? rows.map((row) => `
    <tr>
      <td>${esc(row.tick ?? "")}</td>
      <td><strong>${esc(row.market ?? "")}</strong></td>
      <td>${pill(row.regime ?? "")}</td>
      <td>${pill(row.verdict ?? "", verdictKind(row.verdict))}</td>
      <td>${Number(row.strength ?? 0).toFixed(3)}</td>
      <td>${formatPct(row.confidence ?? 0)}</td>
      <td>${esc(row.active_strategy || "none")}</td>
      <td>${esc(formatTime(row.timestamp))}</td>
    </tr>
  `).join("") : emptyRow(8, "No evolution rows");
}

function renderRaw() {
  const panels = {
    health: state.health,
    status: state.status,
    report: state.report,
    dataCatalog: state.dataCatalog,
    training: state.training,
    activeStrategies: state.active,
    evolution: state.evolution,
    scan: state.scan,
  };
  els.rawPanels.innerHTML = Object.entries(panels).map(([name, data]) => `
    <details>
      <summary>${esc(name)}</summary>
      <pre>${esc(JSON.stringify(data || {}, null, 2))}</pre>
    </details>
  `).join("");
}

function statRows(rows) {
  return rows.map(([key, value]) => `
    <div class="stat-row">
      <span class="stat-key">${esc(key)}</span>
      <span class="stat-value">${esc(value)}</span>
    </div>
  `).join("");
}

function directionClass(direction) {
  if (direction === "long") return "direction-long";
  if (direction === "short") return "direction-short";
  return "direction-neutral";
}

function verdictKind(verdict) {
  if (verdict === "enter") return "ok";
  if (verdict === "exit" || verdict === "refuse") return "bad";
  if (verdict === "hold") return "warn";
  return "";
}

function recordLabel(file) {
  if (file.record_count === null || file.record_count === undefined) return "n/a";
  return `${formatNumber(file.record_count)}${file.records_truncated ? "+" : ""}`;
}

function previewCell(value) {
  if (!value) return `<span class="empty-inline">No preview</span>`;
  return `
    <details class="data-preview">
      <summary>Preview</summary>
      <pre>${esc(value)}</pre>
    </details>
  `;
}

function pill(text, kind = "") {
  return `<span class="pill ${esc(kind)}">${esc(text || "none")}</span>`;
}

function emptyRow(colspan, message) {
  return `<tr><td colspan="${colspan}"><div class="empty">${esc(message)}</div></td></tr>`;
}

function setAutoRefresh() {
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
  if (els.autoRefresh.checked) {
    state.timer = setInterval(async () => {
      await refreshBase();
      await runScan();
    }, 30000);
  }
}

els.refreshBtn.addEventListener("click", refreshBase);
els.scanBtn.addEventListener("click", runScan);
els.autoRefresh.addEventListener("change", setAutoRefresh);

refreshBase().then(runScan);
