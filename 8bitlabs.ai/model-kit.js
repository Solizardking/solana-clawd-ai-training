const MODEL_KIT_API_BASE = (window.MODEL_KIT_API_BASE || "/model-kit-api").replace(/\/$/, "");

const fallbackStatus = {
  protocol: "CAAP/1.0",
  datasets: [
    {
      repo_id: "solanaclawd/solana-clawd-core-ai-instruct",
      rows: 35173,
      status: "published",
      lane: "core-ai",
      url: "https://huggingface.co/datasets/solanaclawd/solana-clawd-core-ai-instruct",
    },
    {
      repo_id: "solanaclawd/solana-clawd-realtime-research-instruct",
      rows: 29058,
      status: "published",
      lane: "custom",
      url: "https://huggingface.co/datasets/solanaclawd/solana-clawd-realtime-research-instruct",
    },
    {
      repo_id: "solanaclawd/solana-clawd-nvidia-trading-factory-instruct",
      rows: 142,
      status: "published",
      lane: "trading-factory",
      url: "https://huggingface.co/datasets/solanaclawd/solana-clawd-nvidia-trading-factory-instruct",
    },
    {
      repo_id: "solanaclawd/solana-tx-foundation-unified",
      rows: 82169,
      cpt_rows: 17262,
      sft_rows: 64907,
      status: "published",
      lane: "tx-foundation",
      url: "https://huggingface.co/datasets/solanaclawd/solana-tx-foundation-unified",
    },
  ],
  models: [
    {
      repo_id: "solanaclawd/solana-nvidia-trading-factory-8b-lora",
      base_model: "NousResearch/Hermes-3-Llama-3.1-8B",
      status: "complete",
      lane: "trading-factory",
      url: "https://huggingface.co/solanaclawd/solana-nvidia-trading-factory-8b-lora",
    },
    {
      repo_id: "solanaclawd/solana-clawd-core-ai-1.5b-lora",
      base_model: "Qwen/Qwen2.5-1.5B-Instruct",
      status: "complete",
      lane: "core-ai",
      url: "https://huggingface.co/solanaclawd/solana-clawd-core-ai-1.5b-lora",
    },
    {
      repo_id: "solanaclawd/clawd-solana-masterpiece-qwen15-lora",
      base_model: "Qwen/Qwen2.5-1.5B-Instruct",
      status: "complete",
      lane: "core-ai",
      url: "https://huggingface.co/solanaclawd/clawd-solana-masterpiece-qwen15-lora",
    },
    {
      repo_id: "solanaclawd/solana-tx-foundation-7b",
      base_model: "Qwen/Qwen2.5-7B-Instruct",
      status: "ready-for-hf-job",
      lane: "tx-foundation",
      url: "https://huggingface.co/solanaclawd/solana-tx-foundation-7b",
    },
  ],
  jobs: [
    {
      id: "ordlibrary/6a35a2ce953ed90bfb945009",
      name: "Trading factory 8B LoRA",
      status: "complete",
      lane: "trading-factory",
    },
    {
      id: "ordlibrary/6a35a6833093dba73ce2a86b",
      name: "Core AI 1.5B LoRA",
      status: "complete",
      lane: "core-ai",
    },
    {
      id: "pending-hf-credits",
      name: "Transaction foundation 7B LoRA",
      status: "ready-for-hf-job",
      lane: "tx-foundation",
    },
  ],
};

const $ = (selector) => document.querySelector(selector);

function apiUrl(path) {
  return `${MODEL_KIT_API_BASE}${path}`;
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(Number(value || 0));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function safeHref(value) {
  const href = String(value || "");
  return /^(https?:\/\/|\/|#)/.test(href) ? href : "#";
}

function setText(id, value) {
  const node = $(`#${id}`);
  if (node) node.textContent = value;
}

async function requestJson(path, options = {}) {
  const response = await fetch(apiUrl(path), {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(payload.detail || payload.raw || response.statusText);
  }
  return payload;
}

function artifactCard(item, type) {
  const rows = item.rows ? `<span>${formatNumber(item.rows)} rows</span>` : "";
  const baseModel = item.base_model ? `<span>${escapeHtml(item.base_model)}</span>` : "";
  const jobId = item.id ? `<span>${escapeHtml(item.id)}</span>` : "";
  const detail = rows || baseModel || jobId;
  const href = safeHref(item.url || "#");
  const linkAttrs = href.startsWith("http") ? ' target="_blank" rel="noreferrer"' : "";
  return `
    <article class="artifact-card">
      <div>
        <span class="tag">${escapeHtml(type)} / ${escapeHtml(item.lane || "general")}</span>
        <h3>${escapeHtml(item.repo_id || item.name)}</h3>
      </div>
      <p>${detail}</p>
      <div class="artifact-meta">
        <strong>${escapeHtml(item.status || "unknown")}</strong>
        <a href="${escapeHtml(href)}"${linkAttrs}>Open</a>
      </div>
    </article>
  `;
}

function mergeByRepo(items, currentItems, staleRepos = []) {
  const stale = new Set(staleRepos);
  const currentByRepo = new Map(currentItems.map((item) => [item.repo_id || item.id, item]));
  const merged = (items || []).filter((item) => {
    const key = item.repo_id || item.id;
    return key && !stale.has(key) && !currentByRepo.has(key);
  });
  return [...merged, ...currentItems];
}

function currentStatus(status) {
  const currentDatasets = fallbackStatus.datasets.filter((item) => item.repo_id === "solanaclawd/solana-tx-foundation-unified");
  const currentModels = fallbackStatus.models.filter((item) =>
    [
      "solanaclawd/solana-clawd-core-ai-1.5b-lora",
      "solanaclawd/clawd-solana-masterpiece-qwen15-lora",
      "solanaclawd/solana-tx-foundation-7b",
    ].includes(item.repo_id),
  );
  const currentJobs = fallbackStatus.jobs.filter((item) => ["ordlibrary/6a35a6833093dba73ce2a86b", "pending-hf-credits"].includes(item.id));
  return {
    ...status,
    datasets: mergeByRepo(status.datasets || fallbackStatus.datasets, currentDatasets, ["solanaclawd/solana-tx-foundation-cpt"]),
    models: mergeByRepo(status.models || fallbackStatus.models, currentModels, ["solanaclawd/solana-tx-foundation-1.5b"]),
    jobs: mergeByRepo(status.jobs || fallbackStatus.jobs, currentJobs),
  };
}

function renderStatus(status) {
  const normalized = currentStatus(status || fallbackStatus);
  const datasets = normalized.datasets || fallbackStatus.datasets;
  const models = normalized.models || fallbackStatus.models;
  const jobs = normalized.jobs || fallbackStatus.jobs;
  setText("protocolValue", normalized.protocol || "CAAP/1.0");
  setText("datasetCount", String(datasets.length));
  setText("modelCount", String(models.length));

  const artifactGrid = $("#artifactGrid");
  if (!artifactGrid) return;
  artifactGrid.innerHTML = [
    ...datasets.map((item) => artifactCard(item, "dataset")),
    ...models.map((item) => artifactCard(item, "model")),
    ...jobs.map((item) => artifactCard(item, "job")),
  ].join("");
}

async function loadStatus() {
  try {
    const health = await requestJson("/health");
    setText("apiHealth", health.ok ? "online" : "degraded");
  } catch {
    setText("apiHealth", "fallback");
  }

  try {
    const status = await requestJson("/model-kit/status");
    renderStatus(status);
  } catch (error) {
    renderStatus(fallbackStatus);
    const output = $("#previewOutput");
    if (output) {
      output.textContent = JSON.stringify(
        {
          ok: false,
          fallback: true,
          api_base: MODEL_KIT_API_BASE,
          error: error.message,
        },
        null,
        2,
      );
    }
  }
}

function registrationRequest() {
  const modelHash = $("#modelHash").value.trim();
  return {
    hf_model_id: $("#hfModelId").value.trim(),
    base_model: $("#baseModel").value.trim(),
    model_hash: modelHash || null,
    model_type: $("#modelType").value,
    api_endpoint: $("#apiEndpoint").value.trim(),
    dataset_size: Number($("#datasetSize").value || 0),
    eval_accuracy: Number($("#evalAccuracy").value || 0),
    cluster: $("#cluster").value,
    protocol: "CAAP/1.0",
    metadata: {
      source: "8bitlabs.ai/model-kit",
      lane: "tx-foundation",
    },
  };
}

async function previewRegistration(event) {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type='submit']");
  const state = $("#previewState");
  const output = $("#previewOutput");
  if (button) button.disabled = true;
  if (state) state.textContent = "previewing";
  try {
    const payload = await requestJson("/register/preview", {
      method: "POST",
      body: JSON.stringify(registrationRequest()),
    });
    if (output) output.textContent = JSON.stringify(payload, null, 2);
    if (state) state.textContent = payload.hash_was_generated ? "generated hash" : "reviewed hash";
  } catch (error) {
    if (output) {
      output.textContent = JSON.stringify(
        {
          ok: false,
          api_base: MODEL_KIT_API_BASE,
          error: error.message,
          local_request: registrationRequest(),
        },
        null,
        2,
      );
    }
    if (state) state.textContent = "error";
  } finally {
    if (button) button.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("#registrationForm")?.addEventListener("submit", previewRegistration);
  $("#refreshStatus")?.addEventListener("click", loadStatus);
  loadStatus();
});
