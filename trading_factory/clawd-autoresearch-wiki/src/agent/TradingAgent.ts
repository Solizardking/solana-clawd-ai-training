/**
 * TradingAgent
 * Autonomous agent that:
 * 1. Monitors markets (crypto + equities)
 * 2. Recalls what it knows and has learned
 * 3. Generates signals from multi-source analysis
 * 4. Executes or simulates trades
 * 5. Records outcomes and learns from them
 * 6. Continuously improves via feedback loop
 */
import { EventEmitter } from 'eventemitter3';
import { memoryEngine } from '../memory/MemoryEngine.js';
import { signalGenerator, type TradingSignal } from '../analysis/SignalGenerator.js';
import { patternRecognizer } from '../analysis/PatternRecognizer.js';
import { BirdeyeConnector } from '../data/birdeye.js';
import { financialDatasets } from '../data/financial.js';
import { coingecko } from '../data/coingecko.js';
import { strategyRegistry } from '../strategy/StrategyRegistry.js';
import { logger } from '../utils/logger.js';
import { config } from '../config.js';
import type { TradeRecord, EpistemologicalState } from '../memory/types.js';

interface AgentConfig {
  cryptoWatchlist: string[];     // token addresses
  equityWatchlist: string[];     // ticker symbols
  minSignalStrength: number;     // min signal strength to trade (0-1)
  minConfidence: number;         // min confidence to trade (0-1)
  maxPositions: number;          // max concurrent open positions
  cyclePeriodMs: number;         // how often to run the main loop
  learnPeriodMs: number;         // how often to run pattern analysis
}

interface OpenPosition {
  tradeId: string;
  signal: TradingSignal;
  entryTime: number;
  maxHoldMs: number;
}

export class TradingAgent extends EventEmitter {
  private running = false;
  private openPositions = new Map<string, OpenPosition>();
  private cycleTimer: ReturnType<typeof setInterval> | null = null;
  private learnTimer: ReturnType<typeof setInterval> | null = null;
  private cfg: AgentConfig;
  private cycleCount = 0;
  private lastCycleAt: number | null = null;
  private recentSignals: TradingSignal[] = [];
  private birdeye: BirdeyeConnector;

  constructor(cfg: Partial<AgentConfig> = {}) {
    super();
    this.cfg = {
      cryptoWatchlist: cfg.cryptoWatchlist ?? config.agent.watchlist,
      equityWatchlist: cfg.equityWatchlist ?? config.agent.equityWatchlist,
      minSignalStrength: cfg.minSignalStrength ?? config.agent.minSignalStrength,
      minConfidence: cfg.minConfidence ?? config.agent.minConfidence,
      maxPositions: cfg.maxPositions ?? config.agent.maxPositions,
      cyclePeriodMs: cfg.cyclePeriodMs ?? config.agent.cyclePeriodMs,
      learnPeriodMs: cfg.learnPeriodMs ?? config.agent.learnPeriodMs,
    };
    this.birdeye = new BirdeyeConnector(config.birdeye.apiKey);
  }

  async start(): Promise<void> {
    logger.info('TradingAgent starting', {
      mode: config.agent.mode,
      cryptoWatchlist: this.cfg.cryptoWatchlist.length,
      equityWatchlist: this.cfg.equityWatchlist.length,
    });

    // Initialize strategy registry (loads saved params from Supabase/file)
    await strategyRegistry.init();

    // Listen for param changes and re-emit for observability
    strategyRegistry.on('params:updated', ({ entry }: { entry: { delta: unknown; reason: string; triggered_by: string } }) => {
      logger.info('[TradingAgent] Strategy params updated', {
        delta: entry.delta,
        reason: entry.reason,
        triggered_by: entry.triggered_by,
      });
      this.emit('strategy:params_updated', entry);
    });

    strategyRegistry.on('metric:improved', ({ value, metric_name }: { value: number; metric_name: string }) => {
      logger.info('[TradingAgent] New best strategy metric', { value, metric_name });
      this.emit('strategy:metric_improved', { value, metric_name });
    });

    this.running = true;

    // Initial cycle immediately
    await this.runCycle();

    // Schedule recurring cycles
    this.cycleTimer = setInterval(() => {
      if (this.running) this.runCycle().catch(err => logger.error('Cycle error', { err: (err as Error).message }));
    }, this.cfg.cyclePeriodMs);

    // Schedule learning
    this.learnTimer = setInterval(() => {
      if (this.running) this.runLearning().catch(err => logger.error('Learning error', { err: (err as Error).message }));
    }, this.cfg.learnPeriodMs);

    this.emit('started');
  }

  stop(): void {
    this.running = false;
    if (this.cycleTimer) clearInterval(this.cycleTimer);
    if (this.learnTimer) clearInterval(this.learnTimer);
    this.emit('stopped');
    logger.info('TradingAgent stopped');
  }

  // ── Main research + trading cycle ─────────────────────────

  private async runCycle(): Promise<void> {
    logger.info('--- Agent cycle starting ---');
    this.cycleCount++;
    this.lastCycleAt = Date.now();

    // 1. Check and close any positions that hit targets
    await this.checkOpenPositions();

    // 2. If at capacity, skip scanning
    if (this.openPositions.size >= this.cfg.maxPositions) {
      logger.info('At max positions, skipping scan', { open: this.openPositions.size });
      return;
    }

    // 3. Gather global macro context
    await this.gatherMacroContext();

    // 4. Scan crypto watchlist
    for (const address of this.cfg.cryptoWatchlist) {
      await this.analyzeAndTrade(address, 'crypto').catch(err =>
        logger.error('Crypto analysis failed', { address, err: (err as Error).message })
      );
    }

    // 5. Scan equity watchlist
    for (const ticker of this.cfg.equityWatchlist) {
      await this.analyzeAndTrade(ticker, 'equity').catch(err =>
        logger.error('Equity analysis failed', { ticker, err: (err as Error).message })
      );
    }

    logger.info('--- Agent cycle complete ---', { cycleCount: this.cycleCount, openPositions: this.openPositions.size });
    this.emit('cycle:complete', { cycleCount: this.cycleCount });
  }

  private async analyzeAndTrade(asset: string, assetClass: 'crypto' | 'equity'): Promise<void> {
    // 1. What do I already know and have learned about this?
    const epistemic = await memoryEngine.whatDoIKnow(asset);
    this.logEpistemicState(asset, epistemic);

    // 2. Generate signal
    let signal: TradingSignal;
    if (assetClass === 'crypto') {
      signal = await signalGenerator.cryptoSignal({ address: asset, symbol: asset });
    } else {
      signal = await signalGenerator.equitySignal(asset);
    }

    logger.info('Signal generated', {
      asset,
      direction: signal.direction,
      strength: signal.strength.toFixed(2),
      confidence: signal.confidence.toFixed(2),
    });

    this.recentSignals.unshift(signal);
    if (this.recentSignals.length > 200) this.recentSignals.pop();

    this.emit('signal', signal);

    // 3. Decide whether to trade
    if (
      signal.direction !== 'neutral' &&
      signal.strength >= this.cfg.minSignalStrength &&
      signal.confidence >= this.cfg.minConfidence &&
      !this.openPositions.has(asset)
    ) {
      await this.openPosition(signal);
    }
  }

  // ── Position management ────────────────────────────────────

  private async openPosition(signal: TradingSignal): Promise<void> {
    const entryPrice = signal.entry_zone
      ? (signal.entry_zone[0] + signal.entry_zone[1]) / 2
      : 0;

    // Use position size from strategy params
    const params = strategyRegistry.getParams();
    const positionSizePct = signal.position_size_pct ?? params.positionSizePct;

    const trade: Omit<TradeRecord, 'id' | 'created_at'> = {
      mode: config.agent.mode,
      asset: signal.asset,
      asset_class: signal.asset_class,
      direction: signal.direction === 'long' ? 'long' : 'short',
      entry_price: entryPrice,
      size: positionSizePct * 1000,     // normalized: positionSizePct of $1000 portfolio
      size_usd: positionSizePct * 1000,
      status: 'open',
      signal_source: signal.sources.join('+'),
      thesis: signal.thesis,
      confidence: signal.confidence,
      memory_ids: signal.memory_ids,
      stop_loss: signal.stop_loss,
      take_profit: signal.take_profit,
    };

    const tradeId = await memoryEngine.recordTrade(trade);

    // Max hold: 4h default, but scale with TP distance
    const tpDistance = signal.take_profit && entryPrice > 0
      ? Math.abs(signal.take_profit - entryPrice) / entryPrice
      : params.takeProfitPct;
    const maxHoldMs = Math.min(12 * 3600_000, Math.max(2 * 3600_000, tpDistance * 24 * 3600_000));

    this.openPositions.set(signal.asset, {
      tradeId,
      signal,
      entryTime: Date.now(),
      maxHoldMs,
    });

    logger.info('Position opened', {
      asset: signal.asset,
      direction: signal.direction,
      entry: entryPrice.toFixed(6),
      stop_loss: signal.stop_loss?.toFixed(6),
      take_profit: signal.take_profit?.toFixed(6),
      size_pct: (positionSizePct * 100).toFixed(1) + '%',
      tradeId,
      strategy: signal.sources.includes('clawdbot_rsi_ema') ? 'ClawdBot' : 'legacy',
    });

    this.emit('trade:opened', { tradeId, signal });
  }

  private async checkOpenPositions(): Promise<void> {
    for (const [asset, pos] of this.openPositions.entries()) {
      const elapsed = Date.now() - pos.entryTime;
      const timedOut = elapsed > pos.maxHoldMs;

      // Get current price
      let currentPrice = 0;
      try {
        if (pos.signal.asset_class === 'crypto') {
          const overview = await this.birdeye.getTokenOverview(asset);
          currentPrice = overview.price;
        } else {
          const snap = await financialDatasets.getPriceSnapshot(asset);
          currentPrice = snap.price;
        }
      } catch {
        continue;
      }

      const entryPrice = pos.signal.entry_zone
        ? (pos.signal.entry_zone[0] + pos.signal.entry_zone[1]) / 2
        : currentPrice;

      const pnlPct = pos.signal.direction === 'long'
        ? ((currentPrice - entryPrice) / entryPrice) * 100
        : ((entryPrice - currentPrice) / entryPrice) * 100;

      const hitTP = pos.signal.take_profit
        ? (pos.signal.direction === 'long' ? currentPrice >= pos.signal.take_profit : currentPrice <= pos.signal.take_profit)
        : false;
      const hitSL = pos.signal.stop_loss
        ? (pos.signal.direction === 'long' ? currentPrice <= pos.signal.stop_loss : currentPrice >= pos.signal.stop_loss)
        : false;

      if (hitTP || hitSL || timedOut) {
        const reason = hitTP ? 'take_profit' : hitSL ? 'stop_loss' : 'timeout';

        await memoryEngine.closeTrade(pos.tradeId, {
          exit_price: currentPrice,
          pnl_usd: pnlPct,   // simplified: 1:1 for sim
          pnl_pct: pnlPct,
          notes: `Closed via ${reason} after ${Math.floor(elapsed / 60000)}m`,
        });

        // Learn from outcome
        const closedTrade = (await memoryEngine.getTradeHistory(asset, 1))[0];
        if (closedTrade) {
          await patternRecognizer.learnFromTrade(closedTrade);
        }

        this.openPositions.delete(asset);
        logger.info('Position closed', { asset, reason, pnlPct: pnlPct.toFixed(2) });
        this.emit('trade:closed', { asset, reason, pnlPct });
      }
    }
  }

  // ── Learning cycle ─────────────────────────────────────────

  private async runLearning(): Promise<void> {
    logger.info('Running learning cycle');

    // Collect recent trade performance across all watched assets
    let totalTrades = 0;
    let winningTrades = 0;
    let totalPnl = 0;

    for (const asset of [...this.cfg.cryptoWatchlist, ...this.cfg.equityWatchlist]) {
      const patterns = await patternRecognizer.analyzeTradeHistory(asset);
      if (patterns.length > 0) {
        logger.info('Patterns discovered', { asset, count: patterns.length });
        this.emit('patterns:discovered', { asset, patterns });
      }

      // Gather recent closed trades for auto-optimize
      const recentTrades = await memoryEngine.getTradeHistory(asset, 10).catch(() => []);
      const closed = recentTrades.filter(t => t.status === 'closed');
      totalTrades += closed.length;
      winningTrades += closed.filter(t => (t.pnl_pct ?? 0) > 0).length;
      totalPnl += closed.reduce((s, t) => s + (t.pnl_pct ?? 0), 0);
    }

    // Run auto-optimize if we have enough trade history
    if (totalTrades >= 5) {
      await strategyRegistry.autoOptimize({
        recentWinRate: winningTrades / totalTrades,
        recentPnlAvg: totalPnl / totalTrades,
        tradeCount: totalTrades,
      });
    }
  }

  // ── Macro context ──────────────────────────────────────────

  private async gatherMacroContext(): Promise<void> {
    try {
      const [globalCrypto, trending] = await Promise.all([
        coingecko.getCryptoSentiment(),
        coingecko.getTrending(),
      ]);

      // Suppress unused variable lint — trending used for context enrichment
      void trending;

      await memoryEngine.remember({
        memory_type: 'known',
        source: 'coingecko',
        topic: 'global crypto market sentiment',
        asset_class: 'macro',
        content: `BTC dominance: ${globalCrypto.btc_dominance.toFixed(1)}%, total market cap: $${(globalCrypto.total_market_cap / 1e12).toFixed(2)}T, 24h change: ${globalCrypto.market_cap_change_24h.toFixed(2)}%`,
        raw_data: globalCrypto as unknown as Record<string, unknown>,
        metadata: { snapshot_type: 'sentiment' },
      });

      logger.debug('Macro context updated', globalCrypto as unknown as Record<string, unknown>);
    } catch (err) {
      logger.warn('Failed to gather macro context', { err: (err as Error).message });
    }
  }

  // ── Epistemological reporting ──────────────────────────────

  private logEpistemicState(asset: string, state: EpistemologicalState): void {
    logger.info(`[EPISTEMIC] ${asset}`, {
      known_facts: state.known_facts.length,
      learned_insights: state.learned_insights.length,
      inferred: state.inferred_connections.length,
      knowledge_gaps: state.knowledge_gap.join(', '),
      confidence: state.confidence_summary.overall.toFixed(2),
    });
  }

  // ── Public API methods (required by REST routes) ───────────────────────────

  isRunning(): boolean {
    return this.running;
  }

  getCycleCount(): number {
    return this.cycleCount;
  }

  getLastCycleAt(): number | null {
    return this.lastCycleAt;
  }

  getOpenPositions(): OpenPosition[] {
    return Array.from(this.openPositions.values());
  }

  getRecentSignals(limit = 50): TradingSignal[] {
    return this.recentSignals.slice(0, limit);
  }

  async getTradeHistory(limit = 50): Promise<TradeRecord[]> {
    try {
      const allTrades: TradeRecord[] = [];
      const seen = new Set<string>();
      for (const asset of [...this.cfg.cryptoWatchlist, ...this.cfg.equityWatchlist]) {
        const trades = await memoryEngine.getTradeHistory(asset, limit).catch(() => []);
        for (const t of trades) {
          if (!seen.has(t.id)) {
            seen.add(t.id);
            allTrades.push(t);
          }
        }
      }
      allTrades.sort((a, b) => new Date(b.entry_time ?? 0).getTime() - new Date(a.entry_time ?? 0).getTime());
      return allTrades.slice(0, limit);
    } catch (err) {
      logger.warn('[TradingAgent] getTradeHistory failed', { err: (err as Error).message });
      return [];
    }
  }

  async runCycleNow(): Promise<void> {
    if (!this.running) throw new Error('Agent is not running');
    await this.runCycle();
  }

  /**
   * Human-readable summary of what the agent knows vs has learned.
   */
  async getKnowledgeSummary(asset?: string): Promise<string> {
    if (asset) {
      const state = await memoryEngine.whatDoIKnow(asset);
      return [
        `=== Epistemic State: ${asset} ===`,
        `KNOWN (${state.known_facts.length} facts):`,
        ...state.known_facts.slice(0, 3).map(m => `  • [${m.source}] ${m.content.slice(0, 100)}`),
        `LEARNED (${state.learned_insights.length} insights):`,
        ...state.learned_insights.slice(0, 3).map(m => `  • [conf:${m.confidence?.toFixed(2)}] ${m.content.slice(0, 100)}`),
        `INFERRED (${state.inferred_connections.length} connections):`,
        ...state.inferred_connections.slice(0, 2).map(m => `  • ${m.content.slice(0, 100)}`),
        `GAPS: ${state.knowledge_gap.join(', ') || 'none identified'}`,
      ].join('\n');
    }

    const assets = await memoryEngine.listKnownAssets();
    return [
      '=== Agent Knowledge Summary ===',
      `Total known assets: ${assets.length}`,
      `Open positions: ${this.openPositions.size}/${this.cfg.maxPositions}`,
      'Assets:',
      ...assets.slice(0, 10).map(a =>
        `  • ${a.entity_id} (${a.entity_type}): price=${a.has_price_data}, fundamentals=${a.has_fundamentals}, patterns=[${(a.learned_patterns ?? []).join(',')}]`
      ),
    ].join('\n');
  }
}
