/**
 * Indicators
 * Pure-function technical indicator implementations used by ClawdBotStrategy.
 * All functions operate on close-price arrays (oldest first).
 */
import type { OHLCVBar } from './types.js';

// ── RSI ───────────────────────────────────────────────────────────────────────

/**
 * Wilder's RSI (original Relative Strength Index).
 * Returns array of RSI values aligned with closes[period..-1].
 * First `period` values are undefined (warm-up).
 */
export function calculateRSI(closes: number[], period = 14): (number | null)[] {
  if (closes.length < period + 1) {
    return closes.map(() => null);
  }

  const result: (number | null)[] = new Array(period).fill(null);

  // Initial avg gain/loss
  let avgGain = 0;
  let avgLoss = 0;
  for (let i = 1; i <= period; i++) {
    const change = closes[i] - closes[i - 1];
    if (change > 0) avgGain += change;
    else avgLoss += Math.abs(change);
  }
  avgGain /= period;
  avgLoss /= period;

  const firstRSI = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  result.push(firstRSI);

  // Wilder smoothing
  for (let i = period + 1; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? Math.abs(change) : 0;

    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;

    const rsi = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
    result.push(rsi);
  }

  return result;
}

/**
 * Latest RSI value from a bars array.
 */
export function latestRSI(bars: OHLCVBar[], period = 14): number | null {
  const closes = bars.map(b => b.close);
  const rsi = calculateRSI(closes, period);
  const last = rsi[rsi.length - 1];
  return last ?? null;
}

// ── EMA ───────────────────────────────────────────────────────────────────────

/**
 * Exponential Moving Average.
 * Returns array same length as closes; first `period-1` values are null.
 */
export function calculateEMA(closes: number[], period: number): (number | null)[] {
  if (closes.length < period) return closes.map(() => null);

  const result: (number | null)[] = new Array(period - 1).fill(null);
  const multiplier = 2 / (period + 1);

  // Seed: SMA of first `period` closes
  const seed = closes.slice(0, period).reduce((s, c) => s + c, 0) / period;
  result.push(seed);

  for (let i = period; i < closes.length; i++) {
    const prev = result[result.length - 1] as number;
    result.push((closes[i] - prev) * multiplier + prev);
  }

  return result;
}

/**
 * Latest EMA value.
 */
export function latestEMA(bars: OHLCVBar[], period: number): number | null {
  const closes = bars.map(b => b.close);
  const ema = calculateEMA(closes, period);
  const last = ema[ema.length - 1];
  return last ?? null;
}

/**
 * EMA cross direction: 'bullish' if fast > slow, 'bearish' if fast < slow.
 * Also detects cross (prev cross alignment was different).
 */
export function emaCrossState(
  bars: OHLCVBar[],
  fastPeriod: number,
  slowPeriod: number,
): {
  current: 'bullish' | 'bearish';
  crossed: boolean;
  ema_fast: number | null;
  ema_slow: number | null;
} {
  const closes = bars.map(b => b.close);
  const fastArr = calculateEMA(closes, fastPeriod);
  const slowArr = calculateEMA(closes, slowPeriod);

  const len = bars.length;
  const ema_fast = (fastArr[len - 1] as number | null) ?? null;
  const ema_slow = (slowArr[len - 1] as number | null) ?? null;
  const prevFast = (fastArr[len - 2] as number | null) ?? null;
  const prevSlow = (slowArr[len - 2] as number | null) ?? null;

  if (ema_fast === null || ema_slow === null) {
    return { current: 'bearish', crossed: false, ema_fast, ema_slow };
  }

  const current = ema_fast >= ema_slow ? 'bullish' : 'bearish';
  let crossed = false;

  if (prevFast !== null && prevSlow !== null) {
    const prevState = prevFast >= prevSlow ? 'bullish' : 'bearish';
    crossed = prevState !== current;
  }

  return { current, crossed, ema_fast, ema_slow };
}

// ── ATR ───────────────────────────────────────────────────────────────────────

/**
 * Average True Range — used for volatility-based SL adjustments.
 */
export function calculateATR(bars: OHLCVBar[], period = 14): number | null {
  if (bars.length < period + 1) return null;

  const trs: number[] = [];
  for (let i = 1; i < bars.length; i++) {
    const high = bars[i].high;
    const low = bars[i].low;
    const prevClose = bars[i - 1].close;
    trs.push(Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose)));
  }

  // Simple average for initial ATR
  let atr = trs.slice(0, period).reduce((s, t) => s + t, 0) / period;
  for (let i = period; i < trs.length; i++) {
    atr = (atr * (period - 1) + trs[i]) / period;
  }
  return atr;
}

// ── Volume ────────────────────────────────────────────────────────────────────

/**
 * Average volume over the last N bars.
 */
export function avgVolume(bars: OHLCVBar[], n = 20): number {
  const slice = bars.slice(-n);
  if (slice.length === 0) return 0;
  return slice.reduce((s, b) => s + b.volume, 0) / slice.length;
}
