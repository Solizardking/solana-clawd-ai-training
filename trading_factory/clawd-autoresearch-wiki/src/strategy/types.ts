/**
 * ClawdBot Strategy — Types
 * Parameter schema, results, and changelog for the RSI/EMA adaptive strategy.
 */

// ── Active Strategy Parameters ────────────────────────────────────────────────

export interface ClawdBotParams {
  // RSI
  rsiOverbought: number;          // sell threshold (default 70)
  rsiOversold: number;            // buy threshold (default 30)
  // EMA
  emaFastPeriod: number;          // fast EMA period (default 20)
  emaSlowPeriod: number;          // slow EMA period (default 50)
  // Filters
  minVolume24h: number;           // USD min volume to consider trade
  minLiquidity: number;           // USD min pool liquidity
  maxSlippage: number;            // 0.02 = 2%
  // Risk management
  stopLossPct: number;            // 0.08 = 8%
  takeProfitPct: number;          // 0.20 = 20%
  positionSizePct: number;        // 0.10 = 10% of portfolio
  // Perps
  fundingRateThreshold: number;   // min absolute funding rate to care about (0.0005 = 0.05%)
  usePerps: boolean;              // whether to use perpetual contracts
}

export const DEFAULT_PARAMS: ClawdBotParams = {
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

// ── OHLCV Bar ─────────────────────────────────────────────────────────────────

export interface OHLCVBar {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// ── Strategy Signal Output ────────────────────────────────────────────────────

export type StrategyDirection = 'long' | 'short' | 'neutral';

export interface StrategySignal {
  direction: StrategyDirection;
  strength: number;         // 0-1
  confidence: number;       // 0-1
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  position_size_pct: number;
  reasons: string[];
  indicators: {
    rsi: number | null;
    ema_fast: number | null;
    ema_slow: number | null;
    ema_cross: 'bullish' | 'bearish' | 'none';
    price_vs_ema_fast: 'above' | 'below' | 'at';
    funding_rate: number | null;
    funding_bias: 'long' | 'short' | 'neutral';
    volume_ok: boolean;
    liquidity_ok: boolean;
  };
  filtered: boolean;        // true = signal blocked by a filter
  filter_reason?: string;   // why it was filtered
}

// ── Changelog Entry ───────────────────────────────────────────────────────────

export interface StrategyChangelogEntry {
  id: string;
  timestamp: string;           // ISO 8601
  previous_params: ClawdBotParams;
  new_params: ClawdBotParams;
  delta: Partial<ClawdBotParams>; // what actually changed
  reason: string;
  triggered_by: 'manual' | 'auto_optimize' | 'learning_feedback';
  metric_before: number;
  metric_after: number | null; // null until measured
  metric_name: string;
}

// ── Registry State ────────────────────────────────────────────────────────────

export interface StrategyRegistryState {
  active_params: ClawdBotParams;
  best_metric: number;
  metric_name: string;
  last_updated: string;
  changelog: StrategyChangelogEntry[];
}
