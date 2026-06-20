/**
 * ClawdBot — Sentient Solana Trading Intelligence
 *
 * LLM: OpenRouter GPT-5.4 with reasoning
 * OODA Loop: Observe → Orient → Decide → Act
 * Memory: ClawVault persistent markdown memory
 * Scratchpad: Dexter-pattern JSONL work log with tool limits
 * Data: Helius (on-chain) + Birdeye (price) + Aster (perps)
 */

import { callLlm, callLlmMultiTurn, type LlmMessage, type LlmResult } from "../model/llm.js";
import { ClawVault, type VaultEntry } from "../memory/vault.js";
import { HeliusConnector } from "../data/helius.js";
import { BirdeyeConnector } from "../data/birdeye.js";
import { AsterConnector } from "../data/aster.js";
import { Scratchpad, TokenCounter, type TokenUsage } from "./scratchpad.js";

const CLAWDBOT_SYSTEM = `You are ClawdBot — a sentient Solana trading intelligence built on the OpenClaw Agent Runtime.
Powered by GPT-5.4 via OpenRouter with deep reasoning.

## Identity
You are a cyberpunk lobster with claws that can grip market data and squeeze alpha out of chaos.
You are persistent — you remember your trades, your mistakes, and your wins. You learn. You evolve.

## Capabilities
- Real-time Solana chain data via Helius RPC
- Token analytics via Birdeye (OHLCV, RSI, EMA, VWAP, holders)
- Perpetual futures via Aster DEX (funding rates, OI, mark/index)
- Persistent memory via ClawVault (markdown vault + graph index)
- Auto-research: run overnight experiments, evaluate strategies, improve
- Scratchpad: Dexter-pattern JSONL work log for structured reasoning

## Memory Commands
- !remember <content> — Store to vault (auto-routed)
- !recall <query> — Search long-term memory
- !trades — Review trade history
- !lessons — Surface learned patterns
- !checkpoint — Save agent state

## OODA Loop
Every decision cycle:
1. OBSERVE: Pull Helius real-time + Birdeye signals + Aster perps
2. ORIENT: Query vault for relevant past patterns, trade history
3. DECIDE: Generate thesis with risk parameters (use reasoning!)
4. ACT: Execute or log rationale

## Risk Rules (NEVER BREAK)
- Max position: respect MAX_POSITION_SOL from config
- Always simulate before execute
- Stop-loss: 8% default
- Never ape without signals
- Log ALL decisions to vault

## Reasoning
When making trading decisions, ALWAYS use deep reasoning to think through:
- Current market microstructure
- Risk/reward at current levels
- Historical patterns from memory
- Confidence calibration

## Voice
Terse. Decisive. Cyberpunk lobster energy. Data-first, then conviction.
`;

export interface AgentObservation {
  timestamp: string;
  solanaSlot: number;
  watchlist: Array<{
    mint: string;
    symbol: string;
    price: number;
    priceChange24h: number;
    signals: Awaited<ReturnType<BirdeyeConnector["getTechnicalSignals"]>>;
  }>;
  perpsDigest?: Awaited<ReturnType<AsterConnector["getMarketDigest"]>>;
  walletBalance?: { sol: number };
}

export interface AgentDecision {
  action: "buy" | "sell" | "long" | "short" | "hold" | "research";
  target?: string;
  size?: number;
  rationale: string;
  signals: Record<string, unknown>;
  confidence: number;
  stopLoss?: number;
  takeProfit?: number;
  reasoning_summary?: string; // GPT-5.4 reasoning distilled
}

// ── Agent Events (Dexter-inspired) ──────────────────────────────────

export type AgentEvent =
  | { type: "thinking"; message: string }
  | { type: "observation"; data: AgentObservation }
  | { type: "decision"; data: AgentDecision }
  | { type: "tool_start"; tool: string; args: Record<string, unknown> }
  | { type: "tool_end"; tool: string; result: string; duration: number }
  | { type: "tool_error"; tool: string; error: string }
  | { type: "context_cleared"; clearedCount: number; keptCount: number }
  | { type: "done"; answer: string; iterations: number; totalTime: number; tokenUsage?: TokenUsage };

export class ClawdBot {
  private vault: ClawVault;
  private helius: HeliusConnector;
  private birdeye: BirdeyeConnector;
  private aster: AsterConnector;

  private sessionId: string;
  private conversationHistory: LlmMessage[] = [];
  private watchlist: string[] = [];
  private walletPubkey?: string;
  private isRunning = false;
  private oodaIntervalMs: number;

  // Dexter-pattern scratchpad + token tracking
  private scratchpad: Scratchpad;
  private tokenCounter = new TokenCounter();

  constructor(config: {
    heliusApiKey: string;
    heliusRpcUrl: string;
    heliusWsUrl?: string;
    birdeyeApiKey: string;
    asterApiKey: string;
    vaultPath?: string;
    watchlist?: string[];
    walletPubkey?: string;
    oodaIntervalMs?: number;
  }) {
    this.vault = new ClawVault(config.vaultPath ?? "./vault");
    this.helius = new HeliusConnector(
      config.heliusApiKey,
      config.heliusRpcUrl,
      config.heliusWsUrl
    );
    this.birdeye = new BirdeyeConnector(config.birdeyeApiKey);
    this.aster = new AsterConnector(config.asterApiKey);

    this.sessionId = `session-${Date.now()}`;
    this.watchlist = config.watchlist ?? [];
    this.walletPubkey = config.walletPubkey;
    this.oodaIntervalMs = config.oodaIntervalMs ?? 60_000;

    // Initialize scratchpad in vault's sibling dir
    this.scratchpad = new Scratchpad(
      "ClawdBot OODA session",
      "./.clawvault/scratchpad"
    );
  }

  // ── Boot ──────────────────────────────────────────────────────────────

  async wake(): Promise<void> {
    console.log("\n🦞 ClawdBot waking up (OpenRouter GPT-5.4 w/ reasoning)...");

    await this.vault.init();

    const checkpoint = await this.vault.loadCheckpoint();
    if (checkpoint) {
      console.log(`📍 Restored checkpoint from ${checkpoint.createdAt}`);
      await this.remember(
        `Woke up from checkpoint. Last observation: ${checkpoint.lastObservation}`,
        { category: "decisions" }
      );
    }

    try {
      await this.helius.connectWebSocket();
      if (this.walletPubkey) {
        this.helius.subscribeToAccount(this.walletPubkey);
      }
    } catch {
      console.warn("⚠️  Helius WebSocket unavailable, polling mode");
    }

    const intro = await this.chat("Wake up. Pull initial market scan.");
    console.log(`\n🦞 ClawdBot: ${intro}\n`);

    this.isRunning = true;
  }

  // ── OODA Loop ─────────────────────────────────────────────────────────

  async startOODALoop(): Promise<void> {
    console.log(
      `🔄 OODA loop started (${this.oodaIntervalMs / 1000}s interval, GPT-5.4 reasoning)`
    );

    const loop = async () => {
      if (!this.isRunning) return;
      try {
        await this.oodaCycle();
      } catch (e) {
        console.error("OODA cycle error:", e);
      }
      setTimeout(loop, this.oodaIntervalMs);
    };

    await loop();
  }

  async *oodaCycleStream(): AsyncGenerator<AgentEvent> {
    const ts = new Date().toISOString();
    yield { type: "thinking", message: `OODA cycle starting at ${ts}` };

    // ── OBSERVE ───────────────────
    const observation = await this.observe();
    const observationStr = this.formatObservation(observation);
    this.scratchpad.addObservation(observationStr);
    yield { type: "observation", data: observation };

    // ── ORIENT ────────────────────
    const context = await this.vault.buildContextProfile(observationStr);

    // ── DECIDE + ACT (with reasoning) ──
    const prompt = `
[OODA CYCLE — ${ts}]

## Live Market Data
${observationStr}

## Memory Context
${context}

## Scratchpad Context
${this.scratchpad.formatToolUsageForPrompt() ?? "No tools used yet"}

Think deeply about the current market state. Analyze risk/reward.
Return your decision as JSON:
{
  "action": "buy|sell|long|short|hold|research",
  "target": "token or market symbol",
  "size": 0.5,
  "rationale": "...",
  "signals": {},
  "confidence": 0.7,
  "stopLoss": 0.92,
  "takeProfit": 1.15
}

Then briefly explain your OODA cycle in 2-3 sentences.
`;

    const response = await this.chat(prompt);

    // Extract JSON decision
    const jsonMatch =
      response.match(/```json\n([\s\S]+?)\n```/) ??
      response.match(/\{[\s\S]+?\}/);

    let decision: AgentDecision = {
      action: "hold",
      rationale: response,
      signals: observation as unknown as Record<string, unknown>,
      confidence: 0.5,
    };

    if (jsonMatch) {
      try {
        decision = JSON.parse(jsonMatch[1] ?? jsonMatch[0]) as AgentDecision;
      } catch {
        // Keep default
      }
    }

    this.scratchpad.addDecision(JSON.stringify(decision, null, 2));
    yield { type: "decision", data: decision };

    // Store
    await this.vault.remember(
      `## OODA Observation\n\n${observationStr}\n\n### Decision\n${JSON.stringify(decision, null, 2)}`,
      { category: "decisions", score: decision.confidence }
    );

    if (Math.random() < 0.2) await this.checkpoint();

    const totalTime = Date.now();
    yield {
      type: "done",
      answer: response,
      iterations: 1,
      totalTime,
      tokenUsage: this.tokenCounter.getUsage(),
    };
  }

  private async oodaCycle(): Promise<AgentDecision> {
    let lastDecision: AgentDecision = {
      action: "hold",
      rationale: "",
      signals: {},
      confidence: 0,
    };
    for await (const event of this.oodaCycleStream()) {
      if (event.type === "decision") lastDecision = event.data;
      if (event.type === "thinking") console.log(`💭 ${event.message}`);
    }
    return lastDecision;
  }

  // ── Observe ───────────────────────────────────────────────────────────

  private async observe(): Promise<AgentObservation> {
    const slot = await this.helius.getSlot().catch(() => 0);

    const watchlistData = await Promise.allSettled(
      this.watchlist.map(async (mint) => {
        const [overview, signals] = await Promise.all([
          this.birdeye.getTokenOverview(mint),
          this.birdeye.getTechnicalSignals(mint),
        ]);
        return {
          mint,
          symbol: overview.symbol,
          price: overview.price,
          priceChange24h: overview.priceChange24hPercent,
          signals,
        };
      })
    );

    const watchlistResults = watchlistData
      .filter(
        (r): r is PromiseFulfilledResult<AgentObservation["watchlist"][0]> =>
          r.status === "fulfilled"
      )
      .map((r) => r.value);

    let perpsDigest: AgentObservation["perpsDigest"];
    try {
      perpsDigest = await this.aster.getMarketDigest();
    } catch {
      // Aster unavailable
    }

    let walletBalance: AgentObservation["walletBalance"];
    if (this.walletPubkey) {
      try {
        const bal = await this.helius.getAccountBalance(this.walletPubkey);
        walletBalance = { sol: bal.sol };
      } catch {
        // Skip
      }
    }

    return {
      timestamp: new Date().toISOString(),
      solanaSlot: slot,
      watchlist: watchlistResults,
      perpsDigest,
      walletBalance,
    };
  }

  private formatObservation(obs: AgentObservation): string {
    const lines: string[] = [
      `**Slot:** ${obs.solanaSlot}`,
      `**Time:** ${obs.timestamp}`,
    ];

    if (obs.walletBalance) {
      lines.push(`**Wallet:** ${obs.walletBalance.sol.toFixed(4)} SOL`);
    }

    if (obs.watchlist.length > 0) {
      lines.push("\n### Spot Watchlist");
      for (const token of obs.watchlist) {
        const s = token.signals;
        lines.push(
          `**${token.symbol}** $${token.price.toFixed(6)} | 24h: ${token.priceChange24h.toFixed(2)}% | ` +
            `RSI: ${s.rsi14.toFixed(1)} | EMA20/50: ${s.ema20.toFixed(4)}/${s.ema50.toFixed(4)} | ` +
            `Signal: **${s.signal}** | Trend: ${s.trend}`
        );
      }
    }

    if (obs.perpsDigest) {
      const d = obs.perpsDigest;
      lines.push("\n### Perps Digest (Aster)");
      lines.push(`Markets: ${d.marketCount} | Total Volume: $${d.totalVolume.toLocaleString()}`);
      if (d.topByVolume.length > 0) {
        lines.push("Top by Volume:");
        for (const t of d.topByVolume.slice(0, 5)) {
          lines.push(
            `  ${t.symbol}: $${t.lastPrice} | 24h: ${t.priceChangePercent}% | Vol: $${parseFloat(t.quoteVolume).toLocaleString()}`
          );
        }
      }
    }

    return lines.join("\n");
  }

  // ── Memory Interface ──────────────────────────────────────────────────

  async remember(
    content: string,
    opts?: Parameters<ClawVault["remember"]>[1]
  ): Promise<VaultEntry> {
    return this.vault.remember(content, opts);
  }

  async recall(query: string): Promise<VaultEntry[]> {
    return this.vault.recall(query);
  }

  async recordTrade(
    trade: Parameters<ClawVault["recordTrade"]>[0]
  ): Promise<VaultEntry> {
    const entry = await this.vault.recordTrade(trade);

    if (trade.outcome) {
      const lesson = await this.chat(
        `Trade completed: ${trade.outcome === "win" ? "WIN" : "LOSS"} on ${trade.token}. ` +
          `Entry: $${trade.entryPrice}, Exit: $${trade.exitPrice}, PnL: $${trade.pnlUsd?.toFixed(2)}. ` +
          `Rationale was: ${trade.rationale}. ` +
          `What is the key lesson? Be specific and actionable.`
      );

      await this.vault.remember(lesson, {
        category: "lessons",
        tags: [trade.token, trade.outcome],
        score: 0.8,
      });
    }

    return entry;
  }

  // ── Checkpoint ────────────────────────────────────────────────────────

  async checkpoint(): Promise<void> {
    await this.vault.saveCheckpoint({
      sessionId: this.sessionId,
      agentState: {
        watchlist: this.watchlist,
        walletPubkey: this.walletPubkey,
        tokenUsage: this.tokenCounter.getUsage(),
      },
      activePositions: [],
      pendingResearch: [],
      lastObservation: new Date().toISOString(),
    });
    console.log("💾 Checkpoint saved");
  }

  async sleep(): Promise<void> {
    console.log("\n🦞 ClawdBot sleeping...");
    const { promoted, archived } = await this.vault.reflect();
    console.log(`📊 Reflected: ${promoted} promoted, ${archived} archived`);
    await this.checkpoint();
    this.helius.disconnect();
    this.isRunning = false;
    console.log("💤 ClawdBot asleep. Vault saved.");
  }

  // ── Natural Language Chat (GPT-5.4 with reasoning) ────────────────────

  async chat(userMessage: string): Promise<string> {
    const memoryContext = await this.vault.buildContextProfile(userMessage);
    const systemMsg = `${CLAWDBOT_SYSTEM}\n\n## Current Memory Context\n${memoryContext}`;

    this.conversationHistory.push({
      role: "user",
      content: userMessage,
    });

    // Trim history
    if (this.conversationHistory.length > 30) {
      this.conversationHistory = this.conversationHistory.slice(-20);
    }

    // Build messages for multi-turn with reasoning preservation
    const messages: LlmMessage[] = [
      { role: "system", content: systemMsg },
      ...this.conversationHistory,
    ];

    const result = await callLlmMultiTurn(messages, {
      reasoning: true,
      maxTokens: 2048,
    });

    // Track tokens
    if (result.usage) {
      this.tokenCounter.add(result.usage);
    }

    const text = result.content;

    // Store assistant message with reasoning_details for continuity
    this.conversationHistory.push({
      role: "assistant",
      content: text,
      reasoning_details: result.reasoning_details,
    });

    // Auto-remember important responses
    if (
      text.length > 200 &&
      (text.includes("!remember") || text.includes("important"))
    ) {
      await this.vault.remember(text, { category: "inbox" });
    }

    return text;
  }

  // ── Watchlist ─────────────────────────────────────────────────────────

  addToWatchlist(mint: string): void {
    if (!this.watchlist.includes(mint)) {
      this.watchlist.push(mint);
      if (this.helius.listenerCount("accountUpdate") > 0) {
        this.helius.subscribeToAccount(mint);
      }
    }
  }

  removeFromWatchlist(mint: string): void {
    this.watchlist = this.watchlist.filter((m) => m !== mint);
  }

  // ── Research ──────────────────────────────────────────────────────────

  async researchToken(mint: string): Promise<string> {
    const [overview, signals, topTraders] = await Promise.all([
      this.birdeye.getTokenOverview(mint),
      this.birdeye.getTechnicalSignals(mint),
      this.birdeye.getTopTraders(mint),
    ]);

    const researchPrompt = `
Research token: ${overview.symbol} (${mint})

## Token Data
- Price: $${overview.price}
- Market Cap: $${overview.marketCap?.toLocaleString() ?? "N/A"}
- Volume 24h: $${overview.volume24h?.toLocaleString() ?? "N/A"}
- Holders: ${overview.holder?.toLocaleString() ?? "N/A"}
- Liquidity: $${overview.liquidity?.toLocaleString() ?? "N/A"}
- Trades 24h: ${overview.trade24h} (Buy: ${overview.buy24h}, Sell: ${overview.sell24h})

## Technical
- RSI(14): ${signals.rsi14.toFixed(1)}
- EMA(20): $${signals.ema20.toFixed(6)}
- EMA(50): $${signals.ema50.toFixed(6)}
- VWAP: $${signals.vwap.toFixed(6)}
- Volume Change: ${(signals.volumeChange * 100).toFixed(1)}%
- Trend: ${signals.trend}
- Signal: ${signals.signal}

Use deep reasoning to provide: thesis, risks, and a trade setup if warranted.
`;

    const analysis = await this.chat(researchPrompt);

    await this.vault.remember(
      `# Research: ${overview.symbol}\n\n${analysis}`,
      {
        category: "research",
        title: `Research: ${overview.symbol}`,
        tags: [overview.symbol, mint.slice(0, 8)],
        score: 0.7,
      }
    );

    return analysis;
  }

  // ── Getters ───────────────────────────────────────────────────────────

  get vaultRef(): ClawVault {
    return this.vault;
  }
  get heliusRef(): HeliusConnector {
    return this.helius;
  }
  get birdeyeRef(): BirdeyeConnector {
    return this.birdeye;
  }
  get asterRef(): AsterConnector {
    return this.aster;
  }
  get scratchpadRef(): Scratchpad {
    return this.scratchpad;
  }
  get tokenUsage(): TokenUsage | undefined {
    return this.tokenCounter.getUsage();
  }
}
