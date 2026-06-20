/**
 * StrategyRegistry
 *
 * Manages active ClawdBot strategy parameters:
 *   - Loads params from Supabase on startup
 *   - Falls back to DEFAULT_PARAMS if none persisted
 *   - Records every param change in a changelog
 *   - Persists state to `strategy_state` table
 *   - Exposes updateParams() for manual + auto-optimized updates
 *   - Broadcasts param changes via EventEmitter (so ClawdBotStrategy hot-reloads)
 */
import { EventEmitter } from 'eventemitter3';
import { createClient, type SupabaseClient } from '@supabase/supabase-js';
import { randomUUID } from 'crypto';
import type {
  ClawdBotParams,
  StrategyChangelogEntry,
  StrategyRegistryState,
} from './types.js';
import { DEFAULT_PARAMS } from './types.js';
import { ClawdBotStrategy } from './ClawdBotStrategy.js';
import { logger } from '../utils/logger.js';
import { config } from '../config.js';

const STRATEGY_KEY = 'clawdbot_v1';
const TABLE = 'strategy_state';

export class StrategyRegistry extends EventEmitter {
  private supabase: SupabaseClient | null = null;
  private state: StrategyRegistryState;
  public readonly strategy: ClawdBotStrategy;

  constructor() {
    super();

    // Init Supabase if configured
    if (config.supabase.url && config.supabase.serviceKey) {
      try {
        this.supabase = createClient(config.supabase.url, config.supabase.serviceKey);
      } catch {
        logger.warn('[StrategyRegistry] Supabase client creation failed, using file persistence');
      }
    }

    this.state = {
      active_params: { ...DEFAULT_PARAMS },
      best_metric: 0,
      metric_name: 'baseline',
      last_updated: new Date().toISOString(),
      changelog: [],
    };
    this.strategy = new ClawdBotStrategy(this.state.active_params);
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  async init(): Promise<void> {
    if (this.supabase) {
      await this.loadFromSupabase();
    } else {
      await this.loadFromFile();
    }
    logger.info('[StrategyRegistry] Initialized', {
      params: this.state.active_params,
      best_metric: this.state.best_metric,
      changelog_entries: this.state.changelog.length,
      persistence: this.supabase ? 'supabase' : 'file',
    });
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  getParams(): ClawdBotParams {
    return { ...this.state.active_params };
  }

  getState(): StrategyRegistryState {
    return JSON.parse(JSON.stringify(this.state)) as StrategyRegistryState;
  }

  getChangelog(): StrategyChangelogEntry[] {
    return [...this.state.changelog];
  }

  /**
   * Update active parameters.
   * Records a changelog entry, persists to Supabase/file, hot-reloads the strategy.
   */
  async updateParams(
    newParams: Partial<ClawdBotParams>,
    opts: {
      reason: string;
      triggered_by?: StrategyChangelogEntry['triggered_by'];
      current_metric?: number;
    },
  ): Promise<StrategyChangelogEntry> {
    const previous_params = { ...this.state.active_params };
    const merged: ClawdBotParams = { ...previous_params, ...newParams };

    // Compute delta
    const delta: Partial<ClawdBotParams> = {};
    for (const key of Object.keys(newParams) as (keyof ClawdBotParams)[]) {
      if (newParams[key] !== previous_params[key]) {
        (delta as Record<string, unknown>)[key] = newParams[key];
      }
    }

    if (Object.keys(delta).length === 0) {
      logger.warn('[StrategyRegistry] updateParams called with no actual changes');
    }

    const entry: StrategyChangelogEntry = {
      id: randomUUID(),
      timestamp: new Date().toISOString(),
      previous_params,
      new_params: merged,
      delta,
      reason: opts.reason,
      triggered_by: opts.triggered_by ?? 'manual',
      metric_before: opts.current_metric ?? this.state.best_metric,
      metric_after: null,
      metric_name: this.state.metric_name,
    };

    // Apply
    this.state.active_params = merged;
    this.state.last_updated = entry.timestamp;
    this.state.changelog.push(entry);

    // Hot-reload strategy engine
    this.strategy.updateParams(merged);

    // Persist
    await this.save();

    logger.info('[StrategyRegistry] Params updated', {
      delta,
      reason: opts.reason,
      triggered_by: entry.triggered_by,
    });

    this.emit('params:updated', { entry, params: merged });
    return entry;
  }

  /**
   * Record that a metric measurement was taken after the last param change.
   * Used to track whether an optimization was beneficial.
   */
  async recordMetric(value: number, metricName?: string): Promise<void> {
    if (metricName) this.state.metric_name = metricName;

    // Update most recent changelog entry's metric_after
    const last = this.state.changelog[this.state.changelog.length - 1];
    if (last && last.metric_after === null) {
      last.metric_after = value;
    }

    // Update best if improved
    if (value > this.state.best_metric) {
      const prev = this.state.best_metric;
      this.state.best_metric = value;
      logger.info('[StrategyRegistry] New best metric', { prev, new_value: value, metric: this.state.metric_name });
      this.emit('metric:improved', { value, metric_name: this.state.metric_name });
    }

    await this.save();
  }

  /**
   * Reset all strategy parameters back to DEFAULT_PARAMS.
   * Records a changelog entry so the reset is fully auditable.
   */
  async resetToDefaults(reason = 'Manual reset'): Promise<void> {
    logger.info('[StrategyRegistry] Resetting to defaults', { reason });
    await this.updateParams(DEFAULT_PARAMS, { reason, triggered_by: 'manual' });
  }

  /**
   * Auto-optimize: apply a small mutation to params if recent performance improved.
   * Uses a simple hill-climbing approach — perturb one param at a time.
   */
  async autoOptimize(opts: {
    recentWinRate: number;
    recentPnlAvg: number;
    tradeCount: number;
  }): Promise<void> {
    const { recentWinRate, recentPnlAvg, tradeCount } = opts;
    if (tradeCount < 5) {
      logger.debug('[StrategyRegistry] Not enough trades for auto-optimize', { tradeCount });
      return;
    }

    const metric = recentWinRate * 0.6 + Math.min(1, Math.max(-1, recentPnlAvg / 10)) * 0.4;
    await this.recordMetric(metric, 'win_rate_pnl_blend');

    const p = this.state.active_params;

    // Determine what to tune based on signal patterns
    const suggestions: Array<{ key: keyof ClawdBotParams; value: number | boolean; reason: string }> = [];

    // If low win rate → tighten entry (raise RSI oversold threshold = require stronger reversal)
    if (recentWinRate < 0.4 && p.rsiOversold < 35) {
      suggestions.push({
        key: 'rsiOversold',
        value: Math.min(40, p.rsiOversold + 2),
        reason: `Low win rate (${(recentWinRate * 100).toFixed(1)}%) — tightening RSI oversold threshold`,
      });
    }

    // If avg PnL below stop-loss (getting stopped out often) → widen SL
    if (recentPnlAvg < -(p.stopLossPct * 0.5 * 100) && p.stopLossPct < 0.15) {
      suggestions.push({
        key: 'stopLossPct',
        value: Math.min(0.15, p.stopLossPct + 0.01),
        reason: `Frequent stop-outs (avg PnL ${recentPnlAvg.toFixed(2)}%) — widening SL`,
      });
    }

    // If win rate > 60% and good avg PnL, slightly increase position size
    if (recentWinRate > 0.6 && recentPnlAvg > 5 && p.positionSizePct < 0.2) {
      suggestions.push({
        key: 'positionSizePct',
        value: Math.min(0.2, p.positionSizePct + 0.01),
        reason: `Good performance (${(recentWinRate * 100).toFixed(1)}% WR, ${recentPnlAvg.toFixed(2)}% avg PnL) — scaling position size`,
      });
    }

    if (suggestions.length === 0) {
      logger.debug('[StrategyRegistry] Auto-optimize: no adjustments warranted');
      return;
    }

    // Apply one suggestion at a time (hill-climbing)
    const pick = suggestions[0];
    await this.updateParams(
      { [pick.key]: pick.value } as Partial<ClawdBotParams>,
      { reason: pick.reason, triggered_by: 'auto_optimize', current_metric: metric },
    );
  }

  // ── Persistence (hybrid: Supabase or file) ─────────────────────────────────

  private async save(): Promise<void> {
    if (this.supabase) {
      await this.saveToSupabase();
    } else {
      await this.saveToFile();
    }
  }

  // ── Supabase I/O ───────────────────────────────────────────────────────────

  private async loadFromSupabase(): Promise<void> {
    if (!this.supabase) return;
    try {
      const { data, error } = await this.supabase
        .from(TABLE)
        .select('state')
        .eq('strategy_key', STRATEGY_KEY)
        .single();

      if (error || !data) {
        logger.info('[StrategyRegistry] No saved state in Supabase, using defaults');
        await this.saveToSupabase(); // seed the row
        return;
      }

      const saved = data.state as StrategyRegistryState;
      this.state = {
        ...saved,
        active_params: { ...DEFAULT_PARAMS, ...saved.active_params },
      };
      this.strategy.updateParams(this.state.active_params);
    } catch (err) {
      logger.warn('[StrategyRegistry] Supabase load failed, using defaults', { err: (err as Error).message });
    }
  }

  private async saveToSupabase(): Promise<void> {
    if (!this.supabase) return;
    try {
      const { error } = await this.supabase
        .from(TABLE)
        .upsert(
          { strategy_key: STRATEGY_KEY, state: this.state, updated_at: new Date().toISOString() },
          { onConflict: 'strategy_key' },
        );

      if (error) {
        logger.error('[StrategyRegistry] Supabase save failed', { error: error.message });
      }
    } catch (err) {
      logger.warn('[StrategyRegistry] Supabase save exception', { err: (err as Error).message });
    }
  }

  // ── File I/O (fallback) ────────────────────────────────────────────────────

  private async loadFromFile(): Promise<void> {
    const { default: fs } = await import('fs/promises');
    const { default: path } = await import('path');
    const statePath = path.resolve(config.agent.vaultPath, '..', '.clawvault', 'strategy-state.json');
    try {
      const raw = await fs.readFile(statePath, 'utf-8');
      const saved = JSON.parse(raw) as StrategyRegistryState;
      this.state = {
        ...saved,
        active_params: { ...DEFAULT_PARAMS, ...saved.active_params },
      };
      this.strategy.updateParams(this.state.active_params);
    } catch {
      logger.info('[StrategyRegistry] No saved file state, using defaults');
      await this.saveToFile();
    }
  }

  private async saveToFile(): Promise<void> {
    const { default: fs } = await import('fs/promises');
    const { default: path } = await import('path');
    const dir = path.resolve(config.agent.vaultPath, '..', '.clawvault');
    const statePath = path.join(dir, 'strategy-state.json');
    try {
      await fs.mkdir(dir, { recursive: true });
      await fs.writeFile(statePath, JSON.stringify(this.state, null, 2), 'utf-8');
    } catch (err) {
      logger.error('[StrategyRegistry] File save failed', { err: (err as Error).message });
    }
  }
}

// Singleton
export const strategyRegistry = new StrategyRegistry();
