/**
 * LLM Client — Multi-provider: OpenRouter, Ollama, HF Router
 *
 * Provider is selected via CLAWD_MODEL_PROVIDER env var:
 *   openrouter  (default) — OpenRouter.ai, any model
 *   ollama                — local Ollama (8bit/solana-clawd-core-ai, 8bit/solana-trading-factory)
 *   hf-router             — HuggingFace Router (solanaclawd/* models)
 *
 * callLlm()    — generic call, uses DEFAULT_PROVIDER
 * callClawd()  — Solana-specific call, always routes to a Clawd model
 */


// ── Types ─────────────────────────────────────────────────────────────

export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

export interface LlmMessage {
  role: "user" | "assistant" | "system";
  content: string;
  reasoning_details?: unknown;
}

export interface LlmCallOptions {
  model?: string;
  provider?: ModelProvider;
  systemPrompt?: string;
  reasoning?: boolean;
  maxTokens?: number;
  temperature?: number;
  signal?: AbortSignal;
}

export interface LlmResult {
  content: string;
  reasoning_details?: unknown;
  usage?: TokenUsage;
  rawMessage?: unknown;
}

// ── Provider registry ─────────────────────────────────────────────────

export type ModelProvider = "openrouter" | "ollama" | "hf-router";

const DEFAULT_PROVIDER: ModelProvider =
  (process.env.CLAWD_MODEL_PROVIDER as ModelProvider) ?? "openrouter";

const PROVIDER_BASES: Record<ModelProvider, string> = {
  openrouter:  "https://openrouter.ai/api/v1",
  ollama:      process.env.CLAWD_OLLAMA_URL ?? "http://localhost:11434/v1",
  "hf-router": "https://router.huggingface.co/v1",
};

// Default model per provider (overridable via CLAWD_MODEL / OPENROUTER_MODEL)
const PROVIDER_DEFAULT_MODELS: Record<ModelProvider, string> = {
  openrouter:  process.env.OPENROUTER_MODEL ?? "openai/gpt-5.4",
  ollama:      process.env.CLAWD_MODEL ?? "8bit/solana-clawd-core-ai",
  "hf-router": process.env.CLAWD_MODEL ?? "solanaclawd/solana-clawd-core-ai-1.5b-lora",
};

// Trading-specific model per provider
const TRADING_MODELS: Record<ModelProvider, string> = {
  openrouter:  process.env.OPENROUTER_MODEL ?? "openai/gpt-5.4",
  ollama:      process.env.CLAWD_TRADING_MODEL ?? "8bit/solana-trading-factory",
  "hf-router": process.env.CLAWD_TRADING_MODEL ?? "solanaclawd/solana-nvidia-trading-factory-8b-lora",
};

// ── Core call helper ──────────────────────────────────────────────────

async function _call(
  messages: Array<Record<string, unknown>>,
  options: LlmCallOptions & { resolvedModel: string; resolvedProvider: ModelProvider }
): Promise<LlmResult> {
  const { resolvedModel, resolvedProvider, reasoning, maxTokens, temperature, signal } = options;

  // Reasoning only supported on OpenRouter (GPT-5.4 / Claude)
  const useReasoning = reasoning ?? (resolvedProvider === "openrouter");

  const body: Record<string, unknown> = {
    model: resolvedModel,
    messages,
    max_tokens: maxTokens ?? 2048,
  };
  if (useReasoning) body.reasoning = { enabled: true };
  if (temperature !== undefined) body.temperature = temperature;

  const result = await withRetry(async () => {
    const url = `${PROVIDER_BASES[resolvedProvider]}/chat/completions`;
    const apiKey =
      resolvedProvider === "openrouter" ? process.env.OPENROUTER_API_KEY :
      resolvedProvider === "hf-router"  ? process.env.HF_TOKEN :
      "ollama";

    const res = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal: signal ?? AbortSignal.timeout(120_000),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`[${resolvedProvider}] HTTP ${res.status}: ${text}`);
    }

    return (await res.json()) as Record<string, unknown>;
  });

  const choices = result.choices as Array<{ message: Record<string, unknown> }>;
  const msg = choices?.[0]?.message;
  const usage = result.usage as
    | { prompt_tokens: number; completion_tokens: number; total_tokens: number }
    | undefined;

  return {
    content: (msg?.content as string) ?? "",
    reasoning_details: msg?.reasoning_details,
    usage: usage
      ? {
          inputTokens: usage.prompt_tokens,
          outputTokens: usage.completion_tokens,
          totalTokens: usage.total_tokens,
        }
      : undefined,
    rawMessage: msg,
  };
}

// ── Public API ────────────────────────────────────────────────────────

/** Generic LLM call — uses DEFAULT_PROVIDER (CLAWD_MODEL_PROVIDER env var). */
export async function callLlm(
  prompt: string,
  options: LlmCallOptions = {}
): Promise<LlmResult> {
  const provider = options.provider ?? DEFAULT_PROVIDER;
  const model = options.model ?? PROVIDER_DEFAULT_MODELS[provider];

  const messages: Array<{ role: string; content: string }> = [];
  if (options.systemPrompt) messages.push({ role: "system", content: options.systemPrompt });
  messages.push({ role: "user", content: prompt });

  return _call(messages, { ...options, resolvedModel: model, resolvedProvider: provider });
}

/** Multi-turn call with reasoning preservation. */
export async function callLlmMultiTurn(
  messages: LlmMessage[],
  options: LlmCallOptions = {}
): Promise<LlmResult> {
  const provider = options.provider ?? DEFAULT_PROVIDER;
  const model = options.model ?? PROVIDER_DEFAULT_MODELS[provider];

  const apiMessages = messages.map((m) => {
    const msg: Record<string, unknown> = { role: m.role, content: m.content };
    if (m.reasoning_details) msg.reasoning_details = m.reasoning_details;
    return msg;
  });

  return _call(apiMessages, { ...options, resolvedModel: model, resolvedProvider: provider });
}

/**
 * Solana-specific call — routes to a Clawd model.
 * Defaults to CLAWD_MODEL_PROVIDER for provider, falls back to ollama.
 * Use `trading: true` to route to the trading-factory model.
 */
export async function callClawd(
  prompt: string,
  options: LlmCallOptions & { trading?: boolean } = {}
): Promise<LlmResult> {
  const provider = options.provider ?? (DEFAULT_PROVIDER !== "openrouter" ? DEFAULT_PROVIDER : "ollama");
  const modelMap = options.trading ? TRADING_MODELS : PROVIDER_DEFAULT_MODELS;
  const model = options.model ?? modelMap[provider];

  const messages: Array<{ role: string; content: string }> = [];
  if (options.systemPrompt) messages.push({ role: "system", content: options.systemPrompt });
  messages.push({ role: "user", content: prompt });

  return _call(messages, {
    ...options,
    reasoning: false, // local models don't support OpenRouter reasoning format
    resolvedModel: model,
    resolvedProvider: provider,
  });
}

// ── Retry Helper ─────────────────────────────────────────────────────

async function withRetry<T>(fn: () => Promise<T>, maxAttempts = 3): Promise<T> {
  let lastErr: Error = new Error("unknown");
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (e) {
      lastErr = e as Error;
      const msg = lastErr.message;
      if (msg.includes("401") || msg.includes("403") || msg.includes("422")) throw lastErr;
      if (attempt < maxAttempts - 1) {
        await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt));
      }
    }
  }
  throw lastErr;
}
