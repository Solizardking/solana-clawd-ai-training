const meshBaseUrl = "https://clawd-inference-mesh.fly.dev";
const routerBaseUrl = "https://clawdrouter-zk.fly.dev";
const clawdChatHistory = [];
const clawdCodePreviewModes = {
  code: {
    model: "grok-4.3",
    prompt: "Build an Anchor registry instruction and explain the PDA seeds.",
    response:
      "CODE MODE. I would start with the account seed contract, derive the PDA client-side, then add the Anchor context, instruction args, and tests before touching deployment.",
  },
  research: {
    model: "grok-4.20-multi-agent",
    prompt: "Compare Solana model registries, SAS attestations, and Hugging Face release proofs.",
    response:
      "RESEARCH MODE. I would fan out into source review, registry account layout, metadata URI checks, HF artifact checks, and a final reproducibility report with claims separated from evidence.",
  },
  trade: {
    model: "grok-4.3",
    prompt: "Inspect SOL perps funding and produce a paper-only signal with preflight gates.",
    response:
      "TRADE MODE. I would keep execution in paper mode, separate observed market data from inferred signal, then report notional, leverage, spread, confidence, and every live-mode gate that remains off.",
  },
};

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function shorten(value, head = 8, tail = 4) {
  const text = String(value || "");
  if (text.length <= head + tail + 1) return text;
  return `${text.slice(0, head)}...${text.slice(-tail)}`;
}

function completionContent(payload) {
  return payload?.choices?.[0]?.message?.content || payload?.choices?.[0]?.delta?.content || "";
}

function setMeshOutput(value) {
  setText("mesh-live-output", value);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload?.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

function renderMeshFlow(payload) {
  const list = document.getElementById("mesh-flow-list");
  if (!list) return;

  const flow = Array.isArray(payload?.flow) ? payload.flow.slice(-6).reverse() : [];
  if (!flow.length) {
    list.innerHTML = "<p>No recent mesh flow events returned.</p>";
    return;
  }

  list.innerHTML = "";
  flow.forEach((event) => {
    const article = document.createElement("article");
    article.className = "mesh-flow-event";

    const status = document.createElement("span");
    status.textContent = `${event.kind || "event"} / ${event.status || "unknown"}`;

    const title = document.createElement("strong");
    const elapsed = Number.isFinite(event.elapsed_ms) ? ` in ${event.elapsed_ms}ms` : "";
    title.textContent = `${event.route || "route"}${elapsed}`;

    const model = document.createElement("code");
    model.textContent = event.model || "model unavailable";

    article.append(status, title, model);
    list.append(article);
  });
}

function populateMeshModelSelect(models) {
  const select = document.getElementById("mesh-live-model");
  if (!select || !Array.isArray(models) || !models.length) return;

  const preferred = [
    "8bit/solana-clawd-core-ai:latest",
    "8bit/solana-clawd-core-ai:preview",
    "8bit/solana-clawd-core-ai:1.5b-merged-20260620",
    "8bit/solana-clawd:preview",
    "8bit/solana-trading-factory:latest",
    "8bit/solana-trading-factory:preview",
    "8bit/DeepSolana:latest",
    "qwen2.5:1.5b",
  ];
  const ordered = [...preferred.filter((model) => models.includes(model)), ...models.filter((model) => !preferred.includes(model))];
  const current = select.value;
  select.innerHTML = "";

  ordered.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    select.append(option);
  });

  select.value = ordered.includes(current) ? current : ordered[0];
}

async function checkMesh() {
  const button = document.getElementById("check-mesh");
  if (button) button.textContent = "Checking...";

  try {
    const [health, modelsPayload, visualization, flowPayload, routerHealth, routerModels] = await Promise.all([
      fetchJson(`${meshBaseUrl}/health`),
      fetchJson(`${meshBaseUrl}/models`),
      fetchJson(`${meshBaseUrl}/mesh/visualization`),
      fetchJson(`${meshBaseUrl}/flow`),
      fetchJson(`${routerBaseUrl}/health`),
      fetchJson(`${routerBaseUrl}/v1/models`),
    ]);
    const models = Array.isArray(modelsPayload.models) ? modelsPayload.models : [];
    const clawdModelCount = models.filter((model) => model.includes("solana-clawd")).length;
    const routerData = Array.isArray(routerModels.data) ? routerModels.data : [];
    const routerOllamaModels = routerData.filter((model) => model?.x_clawd?.route === "ollama").length;
    const meshNode = visualization?.node || health.node || "node";

    setText(
      "mesh-health",
      health.ok
        ? `${meshNode} ok, public ${health.public_inference_enabled ? "on" : "off"}`
        : "unavailable",
    );
    setText("mesh-models", `${models.length} models, ${clawdModelCount} Clawd`);
    setText(
      "router-health",
      `${routerHealth.service || "router"} ${routerHealth.status || "ok"}, Ollama ${routerHealth.ollama?.enabled ? "on" : "off"}`,
    );
    setText("router-models", `${routerData.length} models, ${routerOllamaModels} Ollama`);
    populateMeshModelSelect(models);
    renderMeshFlow(flowPayload);
  } catch (error) {
    setText("mesh-health", "unavailable");
    setText("mesh-models", "unavailable");
    setText("router-health", "unavailable");
    setText("router-models", "unavailable");
    const list = document.getElementById("mesh-flow-list");
    if (list) list.innerHTML = `<p>${error?.message || "Mesh status check failed"}</p>`;
  } finally {
    if (button) button.textContent = "Refresh status";
  }
}

document.getElementById("check-mesh")?.addEventListener("click", checkMesh);
document.getElementById("refresh-mesh-flow")?.addEventListener("click", async () => {
  try {
    renderMeshFlow(await fetchJson(`${meshBaseUrl}/flow`));
  } catch (error) {
    const list = document.getElementById("mesh-flow-list");
    if (list) list.innerHTML = `<p>${error?.message || "Flow refresh failed"}</p>`;
  }
});

document.getElementById("run-mesh-completion")?.addEventListener("click", async () => {
  const button = document.getElementById("run-mesh-completion");
  const model = document.getElementById("mesh-live-model")?.value || "8bit/solana-clawd-core-ai:latest";
  const prompt = document.getElementById("mesh-live-prompt")?.value.trim();
  if (!prompt) return;

  if (button) button.disabled = true;
  setMeshOutput(`Calling ${model} via ${meshBaseUrl}/v1/chat/completions...`);

  try {
    const started = performance.now();
    const payload = await fetchJson(`${meshBaseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        stream: false,
        temperature: 0.2,
        max_tokens: 96,
        messages: [
          {
            role: "user",
            content: prompt,
          },
        ],
      }),
    });
    const elapsedMs = Math.round(performance.now() - started);
    setMeshOutput(JSON.stringify({
      model: payload.model || model,
      route: "browser -> clawd-inference-mesh.fly.dev -> Ollama",
      elapsed_ms: elapsedMs,
      content: completionContent(payload),
      usage: payload.usage || null,
      finish_reason: payload.choices?.[0]?.finish_reason || null,
    }, null, 2));
    checkMesh();
  } catch (error) {
    setMeshOutput(`Mesh completion failed: ${error?.message || "unknown error"}`);
  } finally {
    if (button) button.disabled = false;
  }
});
checkMesh();

function setGrokSandboxOutput(value) {
  setText("grok-sandbox-output", value);
}

function setGrokSandboxStatus(value) {
  setText("grok-sandbox-status", value);
}

const grokWalletState = {
  provider: null,
  address: null,
};

function setGrokWalletStatus(value) {
  setText("grok-wallet-status", value);
}

function getGrokSolanaProvider() {
  return window.phantom?.solana || window.solana || null;
}

function bytesToBase64(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function buildGrokWalletProofMessage(wallet) {
  return [
    "8 Bit Labs Clawd Grok Sandbox",
    `Wallet: ${wallet}`,
    "Intent: launch clawd-grok sandbox computer",
    `Origin: ${window.location.origin}`,
    `Nonce: ${crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`}`,
    `Issued At: ${new Date().toISOString()}`,
  ].join("\n");
}

async function connectGrokWallet() {
  const provider = getGrokSolanaProvider();
  if (!provider) {
    setGrokWalletStatus("Phantom not found. Install Phantom to launch the sandbox.");
    throw new Error("Solana wallet provider not found");
  }

  const response = await provider.connect();
  const address = response.publicKey?.toString() || provider.publicKey?.toString();
  if (!address) throw new Error("Wallet connected without a public key");

  grokWalletState.provider = provider;
  grokWalletState.address = address;
  setGrokWalletStatus(`Connected ${shorten(address, 8, 6)}. Signature required for launch.`);
  return grokWalletState;
}

async function createGrokWalletProof() {
  if (!grokWalletState.provider || !grokWalletState.address) {
    await connectGrokWallet();
  }

  const provider = grokWalletState.provider;
  if (typeof provider.signMessage !== "function") {
    throw new Error("Connected wallet does not support message signing");
  }

  const message = buildGrokWalletProofMessage(grokWalletState.address);
  const encoded = new TextEncoder().encode(message);
  const signed = await provider.signMessage(encoded, "utf8");
  const signature = signed.signature || signed;

  return {
    wallet: grokWalletState.address,
    message,
    signature: bytesToBase64(signature),
    encoding: "base64",
  };
}

function formatGrokPlan(plan) {
  if (!Array.isArray(plan) || !plan.length) return "No plan returned.";
  return plan.map(([label, command]) => `${label}: ${command}`).join("\n\n");
}

function formatGrokSandboxStatus(payload) {
  return JSON.stringify({
    enabled: payload.enabled,
    hasE2BKey: payload.hasE2BKey,
    requiresToken: payload.requiresToken,
    providerKeysForwarded: payload.providerKeysForwarded,
    solanaReadKeysForwarded: payload.solanaReadKeysForwarded,
    holderGate: payload.holderGate,
    manifest: payload.manifest,
  }, null, 2);
}

async function fetchGrokSandbox(method = "GET", body) {
  const response = await fetch("/api/clawd-grok-sandbox", {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error || `${response.status} ${response.statusText}`);
    error.payload = payload;
    throw error;
  }
  return payload;
}

async function checkGrokSandbox() {
  try {
    const payload = await fetchGrokSandbox();
    setGrokSandboxStatus(payload.enabled && payload.hasE2BKey ? "Ready" : "Gated");
    setGrokSandboxOutput(formatGrokSandboxStatus(payload));
  } catch (error) {
    setGrokSandboxStatus("Unavailable");
    setGrokSandboxOutput(error?.message || "Sandbox endpoint unavailable");
  }
}

document.getElementById("check-grok-sandbox")?.addEventListener("click", checkGrokSandbox);
document.getElementById("connect-grok-wallet")?.addEventListener("click", async () => {
  const button = document.getElementById("connect-grok-wallet");
  if (button) button.disabled = true;
  try {
    await connectGrokWallet();
  } catch (error) {
    setGrokWalletStatus(error?.message || "Wallet connection failed");
  } finally {
    if (button) button.disabled = false;
  }
});

document.getElementById("dry-run-grok-sandbox")?.addEventListener("click", async () => {
  const button = document.getElementById("dry-run-grok-sandbox");
  if (button) button.disabled = true;
  setGrokSandboxOutput("Requesting dry-run plan...");
  try {
    const payload = await fetchGrokSandbox("POST", { dryRun: true });
    setGrokSandboxOutput(formatGrokPlan(payload.plan));
  } catch (error) {
    setGrokSandboxOutput(error?.message || "Dry-run failed");
  } finally {
    if (button) button.disabled = false;
  }
});

document.getElementById("launch-grok-sandbox")?.addEventListener("click", async () => {
  const button = document.getElementById("launch-grok-sandbox");
  if (button) button.disabled = true;
  setGrokSandboxOutput("Connect wallet, sign launch proof, then start guarded Clawd Grok status smoke...");
  try {
    const walletProof = await createGrokWalletProof();
    setGrokWalletStatus(`Signed launch proof for ${shorten(walletProof.wallet, 8, 6)}.`);
    const payload = await fetchGrokSandbox("POST", { walletProof });
    setGrokSandboxStatus(payload.ok ? "Smoke passed" : "Smoke failed");
    setGrokSandboxOutput(JSON.stringify({
      ok: payload.ok,
      sandboxId: payload.sandboxId,
      keptAlive: payload.keptAlive,
      access: payload.access,
      logs: payload.logs,
    }, null, 2));
  } catch (error) {
    setGrokSandboxStatus("Gated");
    setGrokSandboxOutput(JSON.stringify(error.payload || { error: error?.message || "Sandbox launch failed" }, null, 2));
  } finally {
    if (button) button.disabled = false;
  }
});

checkGrokSandbox();

function updateClawdCodePreview(mode) {
  const config = clawdCodePreviewModes[mode];
  if (!config) return;

  setText("clawd-code-prompt", config.prompt);
  setText("clawd-code-response", config.response);

  const modelSelect = document.getElementById("clawd-code-model");
  if (modelSelect) modelSelect.value = config.model;

  document.querySelectorAll(".preview-mode").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
}

document.querySelectorAll(".preview-mode").forEach((button) => {
  button.addEventListener("click", () => updateClawdCodePreview(button.dataset.mode));
});

function appendChatMessage(role, content = "") {
  const log = document.getElementById("clawd-chat-log");
  if (!log) return null;

  const article = document.createElement("article");
  article.className = `chat-message ${role}`;

  const label = document.createElement("span");
  label.textContent = role === "user" ? "You" : "Clawd";

  const body = document.createElement("p");
  body.textContent = content;

  article.append(label, body);
  log.append(article);
  log.scrollTop = log.scrollHeight;
  return body;
}

function setChatStatus(value) {
  setText("clawd-chat-status", value);
}

function parseOpenRouterChunk(line) {
  const trimmed = line.trim();
  if (!trimmed.startsWith("data:")) return null;

  const payload = trimmed.slice(5).trim();
  if (!payload || payload === "[DONE]") return null;

  try {
    return JSON.parse(payload);
  } catch (error) {
    return null;
  }
}

async function sendClawdChat(event) {
  event.preventDefault();

  const form = event.currentTarget;
  const input = document.getElementById("clawd-chat-input");
  const button = form.querySelector("button[type='submit']");
  const message = input?.value.trim();
  if (!message) return;

  input.value = "";
  input.disabled = true;
  if (button) button.disabled = true;
  setChatStatus("Streaming...");

  appendChatMessage("user", message);
  const assistantNode = appendChatMessage("assistant", "");
  let assistantText = "";
  let buffer = "";
  const decoder = new TextDecoder();

  try {
    const response = await fetch("/api/clawd-chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        messages: clawdChatHistory.slice(-8),
        max_tokens: 900,
      }),
    });

    const model = response.headers.get("X-Clawd-Model");
    if (model) setText("clawd-chat-model", model);

    if (!response.ok || !response.body) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || "Clawd chat endpoint is not available");
    }

    const reader = response.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const chunk = parseOpenRouterChunk(line);
        if (!chunk) continue;

        const content = chunk.choices?.[0]?.delta?.content || "";
        if (content) {
          assistantText += content;
          if (assistantNode) assistantNode.textContent = assistantText;
        }

        const usage = chunk.usage;
        if (usage?.reasoningTokens || usage?.reasoning_tokens) {
          setChatStatus(`Reasoning tokens: ${usage.reasoningTokens || usage.reasoning_tokens}`);
        }
      }
    }

    if (!assistantText.trim()) assistantText = "No content returned from OpenRouter.";
    if (assistantNode) assistantNode.textContent = assistantText;

    clawdChatHistory.push({ role: "user", content: message }, { role: "assistant", content: assistantText });
    while (clawdChatHistory.length > 10) clawdChatHistory.shift();
    setChatStatus("Ready");
  } catch (error) {
    const errorMessage = error?.message || "Clawd chat failed";
    if (assistantNode) assistantNode.textContent = errorMessage;
    setChatStatus("Endpoint needs OPENROUTER_API_KEY");
  } finally {
    input.disabled = false;
    if (button) button.disabled = false;
    input.focus();
  }
}

document.getElementById("clawd-chat-form")?.addEventListener("submit", sendClawdChat);
document.getElementById("clawd-chat-clear")?.addEventListener("click", () => {
  clawdChatHistory.length = 0;
  const log = document.getElementById("clawd-chat-log");
  if (log) {
    log.innerHTML = "";
    appendChatMessage("assistant", "Context cleared. Ask Clawd a fresh question.");
  }
  setChatStatus("Ready");
});
