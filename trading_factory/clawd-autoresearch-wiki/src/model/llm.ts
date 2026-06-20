/**
 * OpenRouter LLM Client — GPT-5.4 with Reasoning
 *
 * Routes through OpenRouter API using the OpenAI SDK.
 * Supports reasoning (chain-of-thought) for improved trading decisions.
 * Preserves reasoning_details across multi-turn conversations.
 */

import OpenAI from "openai";

// ── Types ─────────────────────────────────────────────────────────────

export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

export interface LlmMessage {
  role: "user" | "assistant" | "system";
  content: string;
  reasoning_details?: unknown; // Preserved from GPT-5.4 reasoning
}

export interface LlmCallOptions {
  model?: string;
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

// ── Client ────────────────────────────────────────────────────────────

const DEFAULT_MODEL = process.env.OPENROUTER_MODEL ?? "openai/gpt-5.4";

let _client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!_client) {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) throw new Error("Missing OPENROUTER_API_KEY env var");
    _client = new OpenAI({
      baseURL: "https://openrouter.ai/api/v1",
      apiKey,
    });
  }
  return _client;
}

// ── Single Call ───────────────────────────────────────────────────────

export async function callLlm(
  prompt: string,
  options: LlmCallOptions = {}
): Promise<LlmResult> {
  const client = getClient();
  const model = options.model ?? DEFAULT_MODEL;
  const reasoning = options.reasoning ?? true;

  const messages: Array<{ role: string; content: string }> = [];

  if (options.systemPrompt) {
    messages.push({ role: "system", content: options.systemPrompt });
  }
  messages.push({ role: "user", content: prompt });

  const body: Record<string, unknown> = {
    model,
    messages,
    max_tokens: options.maxTokens ?? 2048,
  };

  if (reasoning) {
    body.reasoning = { enabled: true };
  }
  if (options.temperature !== undefined) {
    body.temperature = options.temperature;
  }

  const result = await withRetry(async () => {
    const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal: options.signal ?? AbortSignal.timeout(60_000),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`OpenRouter HTTP ${res.status}: ${text}`);
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

// ── Multi-Turn with Reasoning Preservation ───────────────────────────

export async function callLlmMultiTurn(
  messages: LlmMessage[],
  options: LlmCallOptions = {}
): Promise<LlmResult> {
  const model = options.model ?? DEFAULT_MODEL;
  const reasoning = options.reasoning ?? true;

  // Build the message array, preserving reasoning_details from prior turns
  const apiMessages = messages.map((m) => {
    const msg: Record<string, unknown> = {
      role: m.role,
      content: m.content,
    };
    if (m.reasoning_details) {
      msg.reasoning_details = m.reasoning_details;
    }
    return msg;
  });

  const body: Record<string, unknown> = {
    model,
    messages: apiMessages,
    max_tokens: options.maxTokens ?? 2048,
  };

  if (reasoning) {
    body.reasoning = { enabled: true };
  }

  const result = await withRetry(async () => {
    const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal: options.signal ?? AbortSignal.timeout(60_000),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`OpenRouter HTTP ${res.status}: ${text}`);
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

// ── Retry Helper ─────────────────────────────────────────────────────

async function withRetry<T>(
  fn: () => Promise<T>,
  maxAttempts = 3
): Promise<T> {
  let lastErr: Error = new Error("unknown");
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (e) {
      lastErr = e as Error;
      const msg = lastErr.message;
      // Don't retry auth or validation errors
      if (msg.includes("401") || msg.includes("403") || msg.includes("422")) {
        throw lastErr;
      }
      if (attempt < maxAttempts - 1) {
        await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt));
      }
    }
  }
  throw lastErr;
}
