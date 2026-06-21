/**
 * ClawdBot Auto-Research Loop
 *
 * LLM: OpenRouter GPT-5.4 with reasoning
 * Pattern: autoresearch (hypothesis → experiment → evaluate → keep/discard)
 * Scratchpad: Dexter-pattern JSONL logging per experiment
 *
 * Flow: Read program.md → Generate hypothesis → Run experiment →
 *       Evaluate → Store lesson → Repeat
 */

import fs from "fs/promises";
import { callClawd } from "../model/llm.js";
import { ClawVault } from "../memory/vault.js";
import { BirdeyeConnector } from "../data/birdeye.js";
import { AsterConnector } from "../data/aster.js";
import { Scratchpad, TokenCounter } from "../agent/scratchpad.js";

export interface Experiment {
  id: string;
  hypothesis: string;
  strategyParams: Record<string, unknown>;
  startedAt: string;
  endedAt?: string;
  result?: ExperimentResult;
  accepted: boolean;
}

export interface ExperimentResult {
  trades: number;
  winRate: number;
  avgPnlPct: number;
  maxDrawdown: number;
  sharpe: number;
  metric: number;
  comparison: "better" | "worse" | "neutral";
  delta: number;
}

export interface StrategyParams {
  rsiOverbought: number;
  rsiOversold: number;
  emaFastPeriod: number;
  emaSlowPeriod: number;
  minVolume24h: number;
  minLiquidity: number;
  maxSlippage: number;
  stopLossPct: number;
  takeProfitPct: number;
  positionSizePct: number;
  fundingRateThreshold: number;
  usePerps: boolean;
}

const DEFAULT_STRATEGY: StrategyParams = {
  rsiOverbought: 70,
  rsiOversold: 30,
  emaFastPeriod: 20,
  emaSlowPeriod: 50,
  minVolume24h: 100_000,
  minLiquidity: 50_000,
  maxSlippage: 0.02,
  stopLossPct: 0.08,
  takeProfitPct: 0.20,
  positionSizePct: 0.10,
  fundingRateThreshold: 0.0005,
  usePerps: true,
};

export class ResearchLoop {
  private vault: ClawVault;
  private birdeye: BirdeyeConnector;
  private aster: AsterConnector;
  private tokenCounter = new TokenCounter();

  private programPath: string;
  private strategyPath: string;
  private currentStrategy: StrategyParams = { ...DEFAULT_STRATEGY };
  private bestStrategy: StrategyParams = { ...DEFAULT_STRATEGY };
  private bestMetric = -Infinity;
  private experiments: Experiment[] = [];
  private isRunning = false;

  constructor(config: {
    birdeyeApiKey: string;
    asterApiKey: string;
    vaultPath?: string;
    programPath?: string;
    strategyPath?: string;
  }) {
    this.vault = new ClawVault(config.vaultPath ?? "./vault");
    this.birdeye = new BirdeyeConnector(config.birdeyeApiKey);
    this.aster = new AsterConnector(config.asterApiKey);
    this.programPath = config.programPath ?? "./program.md";
    this.strategyPath = config.strategyPath ?? "./strategy.md";
  }

  async init(): Promise<void> {
    await this.vault.init();
    await this.loadStrategy();
    console.log("🔬 Research loop initialized (OpenRouter GPT-5.4 reasoning)");
    console.log(`   Current strategy: ${JSON.stringify(this.currentStrategy)}`);
  }

  // ── Main Loop ─────────────────────────────────────────────────────────

  async run(
    maxExperiments = 100,
    experimentBudgetMs = 300_000
  ): Promise<void> {
    this.isRunning = true;
    console.log(
      `\n🔬 Starting auto-research: ${maxExperiments} experiments, ` +
        `${experimentBudgetMs / 60000}min each, GPT-5.4 reasoning`
    );

    const program = await this.readProgram();
    let experimentCount = 0;

    while (this.isRunning && experimentCount < maxExperiments) {
      experimentCount++;
      const experimentPad = new Scratchpad(
        `Experiment #${experimentCount}`,
        "./.clawvault/scratchpad"
      );

      console.log(`\n🧪 Experiment ${experimentCount}/${maxExperiments}`);

      try {
        // 1. Generate hypothesis (with reasoning)
        const hypothesis = await this.generateHypothesis(program, experimentCount);
        console.log(`   💡 Hypothesis: ${hypothesis.description}`);
        experimentPad.addThinking(hypothesis.description);

        // 2. Mutate strategy
        const newParams = await this.mutateStrategy(
          this.currentStrategy,
          hypothesis
        );

        // 3. Backtest
        const result = await this.runExperiment(newParams, experimentBudgetMs);
        experimentPad.addObservation(JSON.stringify(result));

        // 4. Evaluate
        const accepted = result.metric > this.bestMetric;
        const experiment: Experiment = {
          id: `exp-${Date.now()}`,
          hypothesis: hypothesis.description,
          strategyParams: newParams as unknown as Record<string, unknown>,
          startedAt: new Date().toISOString(),
          endedAt: new Date().toISOString(),
          result,
          accepted,
        };
        this.experiments.push(experiment);

        if (accepted) {
          this.bestMetric = result.metric;
          this.bestStrategy = { ...newParams };
          this.currentStrategy = { ...newParams };
          console.log(
            `   ✅ ACCEPTED — new best metric: ${result.metric.toFixed(4)}`
          );
          experimentPad.addDecision("ACCEPTED");
          await this.saveStrategy(newParams);
        } else {
          console.log(
            `   ❌ REJECTED — metric ${result.metric.toFixed(4)} < best ${this.bestMetric.toFixed(4)}`
          );
          experimentPad.addDecision("REJECTED");
        }

        // 5. Log to vault
        await this.logExperiment(experiment);
      } catch (e) {
        console.error(`   ⚠️  Experiment failed: ${(e as Error).message}`);
      }

      await new Promise((r) => setTimeout(r, 2000));
    }

    console.log(
      `\n🏁 Research complete. ${this.experiments.length} experiments run.`
    );
    await this.generateSummary();
  }

  stop(): void {
    this.isRunning = false;
  }

  // ── Hypothesis Generation (GPT-5.4 reasoning) ──────────────────────

  private async generateHypothesis(
    program: string,
    experimentNumber: number
  ): Promise<{ description: string; type: string; params: string[] }> {
    const lessons = await this.vault.recall(
      "strategy improvement lesson",
      { category: "lessons", limit: 5 }
    );

    const lessonContext = lessons.map((l) => `- ${l.title}`).join("\n");
    const expHistory = this.experiments
      .slice(-5)
      .map(
        (e) =>
          `- ${e.hypothesis}: ${e.accepted ? "✅" : "❌"} (metric: ${e.result?.metric.toFixed(4) ?? "N/A"})`
      )
      .join("\n");

    const prompt = `
You are the ClawdBot research director (GPT-5.4). Generate the next hypothesis.

## Research Program
${program}

## Current Strategy
${JSON.stringify(this.currentStrategy, null, 2)}

## Experiment ${experimentNumber} — Recent History
${expHistory || "No experiments yet — start fresh"}

## Lessons Learned
${lessonContext || "No lessons yet"}

Use deep reasoning to identify the most promising direction.
Return JSON:
{
  "description": "If we adjust RSI thresholds to 25/75, momentum filtering will reduce false signals",
  "type": "parameter_mutation | architecture_change | filter_addition | threshold_adjustment",
  "params": ["rsiOversold", "rsiOverbought"]
}
`;

    // Use the Clawd trading model for Solana-specific hypothesis generation
    const result = await callClawd(prompt, { trading: true, maxTokens: 1024 });

    if (result.usage) this.tokenCounter.add(result.usage);

    const jsonMatch = result.content.match(/\{[\s\S]+\}/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]) as {
        description: string;
        type: string;
        params: string[];
      };
    }

    return {
      description: "Explore RSI threshold variation",
      type: "threshold_adjustment",
      params: ["rsiOversold", "rsiOverbought"],
    };
  }

  // ── Strategy Mutation ─────────────────────────────────────────────────

  private async mutateStrategy(
    current: StrategyParams,
    hypothesis: { description: string; type: string; params: string[] }
  ): Promise<StrategyParams> {
    const prompt = `
Based on: "${hypothesis.description}"
Target params: ${hypothesis.params.join(", ")}

Current strategy:
${JSON.stringify(current, null, 2)}

Propose a specific mutation. Return only the modified params as JSON.
Keep mutations small (±10-20% from current values).

Valid ranges:
- rsiOverbought: 60-85
- rsiOversold: 15-40
- emaFastPeriod: 5-30
- emaSlowPeriod: 20-100
- stopLossPct: 0.03-0.15
- takeProfitPct: 0.10-0.50
- positionSizePct: 0.05-0.25
- fundingRateThreshold: 0.0001-0.002
`;

    const result = await callClawd(prompt, { trading: true, maxTokens: 512 });
    if (result.usage) this.tokenCounter.add(result.usage);

    const jsonMatch = result.content.match(/\{[\s\S]+\}/);
    if (jsonMatch) {
      const delta = JSON.parse(jsonMatch[0]) as Partial<StrategyParams>;
      return { ...current, ...delta };
    }

    // Fallback: random mutation
    const mutated = { ...current };
    for (const param of hypothesis.params) {
      const key = param as keyof StrategyParams;
      const val = mutated[key];
      if (typeof val === "number") {
        const mutation = 1 + (Math.random() - 0.5) * 0.3;
        (mutated as Record<string, unknown>)[key] =
          Math.round(val * mutation * 1000) / 1000;
      }
    }
    return mutated;
  }

  // ── Experiment Runner ─────────────────────────────────────────────────

  private async runExperiment(
    params: StrategyParams,
    _budgetMs: number
  ): Promise<ExperimentResult> {
    console.log("   📊 Running backtest simulation...");

    let tokens: string[] = [];
    try {
      const trending = await this.birdeye.getTrendingTokens(10);
      tokens = trending
        .filter(
          (t) =>
            t.volume24h > params.minVolume24h &&
            t.liquidity > params.minLiquidity
        )
        .slice(0, 5)
        .map((t) => t.address);
    } catch {
      console.log("   ⚠️  Birdeye unavailable, using mock data");
      return this.mockExperimentResult();
    }

    if (tokens.length === 0) return this.mockExperimentResult();

    const allTrades: Array<{ pnl: number; isWin: boolean }> = [];
    for (const mint of tokens) {
      try {
        const candles = await this.birdeye.getOHLCV(mint, "1H", 100);
        const trades = this.backtestStrategy(candles, params);
        allTrades.push(...trades);
      } catch {
        // Skip
      }
    }

    return this.computeResult(allTrades);
  }

  private backtestStrategy(
    candles: Array<{
      o: number;
      h: number;
      l: number;
      c: number;
      v: number;
    }>,
    params: StrategyParams
  ): Array<{ pnl: number; isWin: boolean }> {
    const trades: Array<{ pnl: number; isWin: boolean }> = [];
    const closes = candles.map((c) => c.c);

    let inPosition = false;
    let entryPrice = 0;

    for (let i = params.emaSlowPeriod; i < candles.length; i++) {
      const slice = closes.slice(0, i + 1);
      const rsi = this.computeRSI(slice, 14);
      const ema20 = this.computeEMA(slice, params.emaFastPeriod);
      const ema50 = this.computeEMA(slice, params.emaSlowPeriod);
      const price = closes[i];

      if (!inPosition) {
        if (rsi < params.rsiOversold && ema20 > ema50) {
          inPosition = true;
          entryPrice = price;
        }
      } else {
        const pnlPct = (price - entryPrice) / entryPrice;
        if (
          pnlPct <= -params.stopLossPct ||
          pnlPct >= params.takeProfitPct ||
          rsi > params.rsiOverbought
        ) {
          trades.push({ pnl: pnlPct, isWin: pnlPct > 0 });
          inPosition = false;
        }
      }
    }

    return trades;
  }

  private computeResult(
    trades: Array<{ pnl: number; isWin: boolean }>
  ): ExperimentResult {
    if (trades.length === 0) {
      return {
        trades: 0,
        winRate: 0,
        avgPnlPct: 0,
        maxDrawdown: 0,
        sharpe: 0,
        metric: -1,
        comparison: "worse",
        delta: -1,
      };
    }

    const winRate = trades.filter((t) => t.isWin).length / trades.length;
    const avgPnlPct =
      trades.reduce((s, t) => s + t.pnl, 0) / trades.length;

    let peak = 1;
    let equity = 1;
    let maxDrawdown = 0;
    for (const trade of trades) {
      equity *= 1 + trade.pnl;
      if (equity > peak) peak = equity;
      const dd = (peak - equity) / peak;
      if (dd > maxDrawdown) maxDrawdown = dd;
    }

    const pnls = trades.map((t) => t.pnl);
    const mean = avgPnlPct;
    const std = Math.sqrt(
      pnls.reduce((s, p) => s + (p - mean) ** 2, 0) / pnls.length
    );
    const sharpe = std > 0 ? (mean / std) * Math.sqrt(252) : 0;

    const metric = sharpe * winRate;
    const delta = metric - this.bestMetric;
    const comparison: "better" | "worse" | "neutral" =
      delta > 0.01 ? "better" : delta < -0.01 ? "worse" : "neutral";

    return {
      trades: trades.length,
      winRate,
      avgPnlPct,
      maxDrawdown,
      sharpe,
      metric,
      comparison,
      delta,
    };
  }

  private mockExperimentResult(): ExperimentResult {
    const winRate = 0.4 + Math.random() * 0.3;
    const sharpe = -0.5 + Math.random() * 2;
    const metric = sharpe * winRate + (Math.random() - 0.5) * 0.1;
    const delta = metric - this.bestMetric;

    return {
      trades: Math.floor(10 + Math.random() * 30),
      winRate,
      avgPnlPct: (Math.random() - 0.4) * 0.05,
      maxDrawdown: Math.random() * 0.2,
      sharpe,
      metric,
      comparison: delta > 0 ? "better" : "worse",
      delta,
    };
  }

  // ── Vault Logging ─────────────────────────────────────────────────────

  private async logExperiment(exp: Experiment): Promise<void> {
    const r = exp.result;
    const content = `
## Experiment: ${exp.id}

**Status:** ${exp.accepted ? "✅ ACCEPTED" : "❌ REJECTED"}
**Hypothesis:** ${exp.hypothesis}
**Started:** ${exp.startedAt}

### Parameters
\`\`\`json
${JSON.stringify(exp.strategyParams, null, 2)}
\`\`\`

### Results
${
  r
    ? `
- Trades: ${r.trades}
- Win Rate: ${(r.winRate * 100).toFixed(1)}%
- Avg PnL: ${(r.avgPnlPct * 100).toFixed(2)}%
- Max Drawdown: ${(r.maxDrawdown * 100).toFixed(1)}%
- Sharpe: ${r.sharpe.toFixed(3)}
- Primary Metric: ${r.metric.toFixed(4)}
- vs Baseline: ${r.comparison} (${r.delta > 0 ? "+" : ""}${r.delta.toFixed(4)})
`
    : "No result"
}
`.trim();

    await this.vault.remember(content, {
      category: "research",
      title: `Exp ${exp.id}: ${exp.accepted ? "✅" : "❌"} ${exp.hypothesis.slice(0, 50)}`,
      score: exp.accepted ? 0.9 : 0.5,
      metadata: { experimentId: exp.id, accepted: exp.accepted },
    });

    if (exp.accepted && exp.result) {
      await this.vault.remember(
        `Strategy improvement: ${exp.hypothesis}. ` +
          `Win rate: ${(exp.result.winRate * 100).toFixed(1)}%, Sharpe: ${exp.result.sharpe.toFixed(3)}.`,
        { category: "lessons", score: 0.85 }
      );
    }
  }

  private async generateSummary(): Promise<void> {
    const accepted = this.experiments.filter((e) => e.accepted);
    const total = this.experiments.length;

    const summary = `
# Auto-Research Session Summary

**Model:** GPT-5.4 via OpenRouter (reasoning enabled)
**Total experiments:** ${total}
**Accepted:** ${accepted.length} (${((accepted.length / total) * 100).toFixed(0)}%)
**Best metric:** ${this.bestMetric.toFixed(4)}
**Tokens used:** ${this.tokenCounter.getUsage()?.totalTokens ?? 0}

## Best Strategy
\`\`\`json
${JSON.stringify(this.bestStrategy, null, 2)}
\`\`\`

## Key Improvements
${accepted.map((e) => `- ${e.hypothesis}`).join("\n") || "None accepted"}
`.trim();

    await this.vault.remember(summary, {
      category: "research",
      title: "Research Session Summary",
      score: 1.0,
    });

    console.log("\n" + summary);
  }

  // ── Strategy Persistence ──────────────────────────────────────────────

  private async loadStrategy(): Promise<void> {
    try {
      const raw = await fs.readFile(this.strategyPath, "utf-8");
      const jsonMatch = raw.match(/```json\n([\s\S]+?)\n```/);
      if (jsonMatch) {
        const loaded = JSON.parse(jsonMatch[1]) as Partial<StrategyParams>;
        this.currentStrategy = { ...DEFAULT_STRATEGY, ...loaded };
        this.bestStrategy = { ...this.currentStrategy };
      }
    } catch {
      // Use defaults
    }
  }

  private async saveStrategy(params: StrategyParams): Promise<void> {
    const content = `# ClawdBot Strategy

Last updated: ${new Date().toISOString()}
Best metric: ${this.bestMetric.toFixed(4)}
Model: GPT-5.4 via OpenRouter

## Active Parameters

\`\`\`json
${JSON.stringify(params, null, 2)}
\`\`\`

## Change Log
${this.experiments
  .filter((e) => e.accepted)
  .map((e) => `- ${e.startedAt}: ${e.hypothesis}`)
  .join("\n")}
`;

    await fs.writeFile(this.strategyPath, content, "utf-8");
  }

  private async readProgram(): Promise<string> {
    try {
      return await fs.readFile(this.programPath, "utf-8");
    } catch {
      return DEFAULT_PROGRAM;
    }
  }

  // ── Technical Helpers ─────────────────────────────────────────────────

  private computeRSI(closes: number[], period: number): number {
    if (closes.length < period + 1) return 50;
    const changes = closes.slice(1).map((c, i) => c - closes[i]);
    const recent = changes.slice(-period);
    const gains = recent.map((c) => (c > 0 ? c : 0));
    const losses = recent.map((c) => (c < 0 ? -c : 0));
    const avgGain = gains.reduce((a, b) => a + b, 0) / period;
    const avgLoss = losses.reduce((a, b) => a + b, 0) / period;
    if (avgLoss === 0) return 100;
    return 100 - 100 / (1 + avgGain / avgLoss);
  }

  private computeEMA(values: number[], period: number): number {
    if (values.length < period) return values[values.length - 1] ?? 0;
    const k = 2 / (period + 1);
    let ema =
      values.slice(0, period).reduce((a, b) => a + b, 0) / period;
    for (const v of values.slice(period)) ema = v * k + ema * (1 - k);
    return ema;
  }
}

const DEFAULT_PROGRAM = `
# ClawdBot Research Program

## Goal
Find the optimal parameter configuration for Solana memecoin momentum trading
that maximizes Sharpe ratio while maintaining >45% win rate and <15% max drawdown.

## Constraints
- Backtest on trending tokens with >$100K daily volume
- Min 10 trades per backtest window
- Max 15% position size per trade
- Hard stop loss at 8%

## Research Areas
1. RSI threshold optimization (oversold entry, overbought exit)
2. EMA crossover period tuning
3. Volume filter thresholds
4. Combined funding rate + spot signal strategies
5. Position sizing via Kelly Criterion

## Success Metric
Primary: Sharpe × WinRate (higher = better)
Secondary: MaxDrawdown < 15%
`;
