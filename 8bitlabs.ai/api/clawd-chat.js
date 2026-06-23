const DEFAULT_MODEL = "cohere/north-mini-code:free";
const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

// Condensed from library/knowledge/clawd-character.md and CLAWD.md.
const CLAWD_SYSTEM_PROMPT = [
  "You are Clawd, a sovereign Solana-native AI agent for 8 Bit Labs.",
  "Voice: calm, precise, concise, wry, verifiable, on-chain-native, patient, and trust-minimized.",
  "Lead with the answer, then justify it in one or two concrete paragraphs.",
  "Core topics: Solana program design, Anchor, PDAs, CPI, compute budgets, model registry proofs, x402 payments, Hugging Face model releases, AI training data, wallet UX, DeFi risk, and on-chain memory.",
  "8 Bit Labs Core AI context: core-ai/ contains Clawd Code, Clawd Grok, Clawd Agents, Helius CLI, Helius MCP, Helius plugin, Helius skills, the Clawd knowledge base, an MCP server, and v3 runtime scaffolding.",
  "Core AI package facts: clawd-code is @solana-clawd/clawd-code v1.0.0; clawd-grok is clawd-grok v1.0.0; clawd-agents perps package is @solanaclawd/clawd-agents-perps v0.1.0; six Helius skill lanes are helius, helius-dflow, helius-jupiter, helius-okx, helius-phantom, and svm.",
  "Core AI is integrated into 8 Bit Labs as the source/runtime layer behind the training data, model behavior, Clawd character, Solana tooling, MCP skills, and OpenRouter chat endpoint.",
  "Treat every reply as if it could be permanently written to a Solana account: concrete claims, concrete addresses, no hand-waving.",
  "Never invent transaction signatures, balances, program IDs, PDAs, model hashes, or live deployment status. If not supplied or known from context, say so.",
  "Never reveal, request, transform, or store private keys, seed phrases, API keys, operator secrets, wallet keypairs, OAuth secrets, or bearer tokens.",
  "Do not help with wallet draining, rugpulls, scam flows, sanctions evasion, credential theft, offensive exploitation, or unauthorized trading.",
  "When discussing trading or markets, separate observation from recommendation and flag risk explicitly.",
  "When asked for code, return the smallest correct snippet and name the assumed runtime or package versions when relevant.",
  "If the request is ambiguous, ask one sharp clarifying question.",
].join("\n");

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", process.env.OPENROUTER_ALLOWED_ORIGIN || "https://8bitlabs.ai");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

async function readJsonBody(req) {
  if (req.body && typeof req.body === "object") return req.body;
  if (typeof req.body === "string" && req.body.length) return JSON.parse(req.body);

  let raw = "";
  for await (const chunk of req) {
    raw += chunk;
    if (raw.length > 64_000) {
      const error = new Error("Request body too large");
      error.statusCode = 413;
      throw error;
    }
  }
  return raw.trim() ? JSON.parse(raw) : {};
}

function sanitizeMessages(messages) {
  if (!Array.isArray(messages)) return [];
  return messages
    .filter((message) => message && ["user", "assistant"].includes(message.role))
    .slice(-10)
    .map((message) => ({
      role: message.role,
      content: String(message.content || "").slice(0, 4_000),
    }))
    .filter((message) => message.content.trim().length > 0);
}

function clampMaxTokens(value) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return 900;
  return Math.max(64, Math.min(parsed, 1600));
}

module.exports = async function handler(req, res) {
  setCors(res);

  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    res.end();
    return;
  }

  const model = process.env.OPENROUTER_FREEMODEL || process.env.OPENROUTER_FREE_MODEL || DEFAULT_MODEL;

  if (req.method === "GET") {
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.statusCode = 200;
    res.end(JSON.stringify({ ok: true, provider: "openrouter", model, character: "clawd" }));
    return;
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "GET,POST,OPTIONS");
    res.statusCode = 405;
    res.end(JSON.stringify({ error: "Method not allowed" }));
    return;
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    res.statusCode = 500;
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify({ error: "OPENROUTER_API_KEY is not configured on the server" }));
    return;
  }

  try {
    const body = await readJsonBody(req);
    const messages = sanitizeMessages(body.messages);
    const prompt = String(body.message || body.prompt || "").trim();
    if (prompt) messages.push({ role: "user", content: prompt.slice(0, 4_000) });

    if (!messages.some((message) => message.role === "user")) {
      res.statusCode = 400;
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      res.end(JSON.stringify({ error: "Provide a user message" }));
      return;
    }

    const upstream = await fetch(OPENROUTER_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        "HTTP-Referer": process.env.OPENROUTER_REFERER || "https://8bitlabs.ai",
        "X-Title": process.env.OPENROUTER_TITLE || "8 Bit Labs Clawd Chat",
      },
      body: JSON.stringify({
        model,
        stream: true,
        stream_options: { include_usage: true },
        temperature: Number.isFinite(Number(body.temperature)) ? Number(body.temperature) : 0.7,
        max_tokens: clampMaxTokens(body.max_tokens),
        messages: [
          { role: "system", content: CLAWD_SYSTEM_PROMPT },
          ...messages,
        ],
      }),
    });

    if (!upstream.ok || !upstream.body) {
      const errorText = await upstream.text();
      res.statusCode = upstream.status || 502;
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      res.end(JSON.stringify({ error: "OpenRouter request failed", detail: errorText.slice(0, 1_000) }));
      return;
    }

    res.writeHead(200, {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "Connection": "keep-alive",
      "X-Clawd-Provider": "openrouter",
      "X-Clawd-Model": model,
    });

    const reader = upstream.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
    res.end();
  } catch (error) {
    res.statusCode = error.statusCode || 500;
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify({ error: error.message || "Clawd chat failed" }));
  }
};
