/**
 * ClawdBotStrategy
 *
 * RSI + EMA cross strategy for Solana tokens with:
 *   - Volume and liquidity pre-filters
 *   - Funding rate bias (perps)
 *   - Parameterized TP/SL/position sizing from active params
 *   - Full indicator breakdown in every signal
 *
 * Entry rules:
 *   LONG  → RSI crosses above oversold AND price > EMA_fast AND EMA_fast > EMA_slow
 *   SHORT → RSI crosses below overbought AND price < EMA_fast AND EMA_fast < EMA_slow
 *           (only if usePerps is true)
 *
 * Funding rate:
 *   Positive funding = longs pay → slight short bias
 *   Negative funding = shorts pay → slight long bias
 *   Magnitude > fundingRateThreshold adds a reason and bumps strength.
 */
import type { ClawdBotParams, OHLCVBar, StrategySignal } from './types.js';
import {
  latestRSI,
  calculateRSI,
  emaCrossState,
  calculateATR,
} from './indicators.js';

export class ClawdBotStrategy {
  private params: ClawdBotParams;

  constructor(params: ClawdBotParams) {
    this.params = params;
  }

  /** Update params at runtime (called by StrategyRegistry on param change). */
  updateParams(params: ClawdBotParams): void {
    this.params = params;
  }

  getParams(): ClawdBotParams {
    return { ...this.params };
  }

  /**
   * Main signal evaluation.
   *
   * @param bars     - OHLCV bars, oldest first, at least emaSlowPeriod+RSI_PERIOD bars
   * @param metadata - current market metadata (volume, liquidity, funding rate)
   */
  evaluate(
    bars: OHLCVBar[],
    metadata: {
      volume24h: number;
      liquidity: number;
      fundingRate?: number;   // current 8h funding rate (e.g. 0.001 = 0.1%)
      symbol: string;
    },
  ): StrategySignal {
    const p = this.params;
    const currentPrice = bars[bars.length - 1]?.close ?? 0;
    const reasons: string[] = [];

    // ── Pre-filters ────────────────────────────────────────────

    if (metadata.volume24h < p.minVolume24h) {
      return this.filteredSignal(currentPrice, {
        reason: `volume $${metadata.volume24h.toLocaleString()} below min $${p.minVolume24h.toLocaleString()}`,
        bars,
        metadata,
      });
    }

    if (metadata.liquidity < p.minLiquidity) {
      return this.filteredSignal(currentPrice, {
        reason: `liquidity $${metadata.liquidity.toLocaleString()} below min $${p.minLiquidity.toLocaleString()}`,
        bars,
        metadata,
      });
    }

    // ── Indicators ─────────────────────────────────────────────

    const closes = bars.map(b => b.close);
    const rsiSeries = calculateRSI(closes, 14);
    const rsiNow = rsiSeries[rsiSeries.length - 1] ?? null;
    const rsiPrev = rsiSeries[rsiSeries.length - 2] ?? null;

    const emaCross = emaCrossState(bars, p.emaFastPeriod, p.emaSlowPeriod);
    const atr = calculateATR(bars, 14);

    const fundingRate = metadata.fundingRate ?? null;
    const fundingBias = fundingRate === null
      ? 'neutral'
      : fundingRate > p.fundingRateThreshold
        ? 'short'   // longs paying a lot → crowded long → slight short lean
        : fundingRate < -p.fundingRateThreshold
          ? 'long'  // shorts paying a lot → crowded short → slight long lean
          : 'neutral';

    const priceVsEmaFast: 'above' | 'below' | 'at' = emaCross.ema_fast === null
      ? 'at'
      : currentPrice > emaCross.ema_fast * 1.0001 ? 'above'
      : currentPrice < emaCross.ema_fast * 0.9999 ? 'below'
      : 'at';

    // ── RSI crossover detection ────────────────────────────────

    const rsiCrossedAboveOversold = rsiNow !== null && rsiPrev !== null
      && rsiPrev <= p.rsiOversold && rsiNow > p.rsiOversold;

    const rsiCrossedBelowOverbought = rsiNow !== null && rsiPrev !== null
      && rsiPrev >= p.rsiOverbought && rsiNow < p.rsiOverbought;

    // RSI in neutral zone — not yet oversold/overbought but trending
    const rsiOversoldZone = rsiNow !== null && rsiNow <= p.rsiOversold + 5;
    const rsiOverboughtZone = rsiNow !== null && rsiNow >= p.rsiOverbought - 5;

    const indicators = {
      rsi: rsiNow,
      ema_fast: emaCross.ema_fast,
      ema_slow: emaCross.ema_slow,
      ema_cross: emaCross.current,
      price_vs_ema_fast: priceVsEmaFast,
      funding_rate: fundingRate,
      funding_bias: fundingBias,
      volume_ok: true,
      liquidity_ok: true,
    } as const;

    // ── Build long conditions ──────────────────────────────────

    let longScore = 0;
    let shortScore = 0;

    // Primary long: RSI cross above oversold + bullish EMA + price above fast EMA
    if (rsiCrossedAboveOversold) {
      longScore += 0.4;
      reasons.push(`RSI crossed above oversold (${p.rsiOversold}) at ${rsiNow?.toFixed(1)}`);
    } else if (rsiOversoldZone) {
      longScore += 0.2;
      reasons.push(`RSI in oversold zone (${rsiNow?.toFixed(1)})`);
    }

    if (emaCross.current === 'bullish') {
      longScore += 0.25;
      reasons.push(`EMA bullish cross (fast ${emaCross.ema_fast?.toFixed(4)} > slow ${emaCross.ema_slow?.toFixed(4)})`);
      if (emaCross.crossed) {
        longScore += 0.1;
        reasons.push('Fresh EMA bullish crossover this bar');
      }
    }

    if (priceVsEmaFast === 'above') {
      longScore += 0.15;
      reasons.push('Price above fast EMA');
    }

    if (fundingBias === 'long') {
      longScore += 0.1;
      reasons.push(`Funding rate favors longs (${((fundingRate ?? 0) * 100).toFixed(4)}%)`);
    }

    // ── Build short conditions ─────────────────────────────────

    if (p.usePerps) {
      if (rsiCrossedBelowOverbought) {
        shortScore += 0.4;
        reasons.push(`RSI crossed below overbought (${p.rsiOverbought}) at ${rsiNow?.toFixed(1)}`);
      } else if (rsiOverboughtZone) {
        shortScore += 0.2;
        reasons.push(`RSI in overbought zone (${rsiNow?.toFixed(1)})`);
      }

      if (emaCross.current === 'bearish') {
        shortScore += 0.25;
        reasons.push(`EMA bearish cross (fast ${emaCross.ema_fast?.toFixed(4)} < slow ${emaCross.ema_slow?.toFixed(4)})`);
        if (emaCross.crossed) {
          shortScore += 0.1;
          reasons.push('Fresh EMA bearish crossover this bar');
        }
      }

      if (priceVsEmaFast === 'below') {
        shortScore += 0.15;
        reasons.push('Price below fast EMA');
      }

      if (fundingBias === 'short') {
        shortScore += 0.1;
        reasons.push(`Funding rate favors shorts (${((fundingRate ?? 0) * 100).toFixed(4)}%)`);
      }
    }

    // ── Determine direction ────────────────────────────────────

    const total = longScore + shortScore;
    if (total === 0) {
      return {
        direction: 'neutral',
        strength: 0,
        confidence: 0.3,
        entry_price: currentPrice,
        stop_loss: currentPrice * (1 - p.stopLossPct),
        take_profit: currentPrice * (1 + p.takeProfitPct),
        position_size_pct: 0,
        reasons: ['No directional signal — RSI/EMA conditions not met'],
        indicators: { ...indicators, ema_cross: emaCross.current },
        filtered: false,
      };
    }

    const direction: 'long' | 'short' = longScore >= shortScore ? 'long' : 'short';
    const dominantScore = direction === 'long' ? longScore : shortScore;
    const strength = Math.min(1, dominantScore);

    // Confidence: based on signal alignment (all 3 conditions met = high)
    const conditionsMet = direction === 'long'
      ? (rsiCrossedAboveOversold ? 1 : 0) + (emaCross.current === 'bullish' ? 1 : 0) + (priceVsEmaFast === 'above' ? 1 : 0)
      : (rsiCrossedBelowOverbought ? 1 : 0) + (emaCross.current === 'bearish' ? 1 : 0) + (priceVsEmaFast === 'below' ? 1 : 0);

    const confidence = Math.min(0.95, 0.4 + conditionsMet * 0.18 + (emaCross.crossed ? 0.1 : 0));

    // ── ATR-adjusted SL/TP (if ATR available, tighten on low-vol, widen on high-vol) ─

    let stopLoss: number;
    let takeProfit: number;

    if (atr !== null && currentPrice > 0) {
      const atrRatio = atr / currentPrice;
      // Blend: 70% param-based, 30% ATR-scaled
      const blendedSL = 0.7 * p.stopLossPct + 0.3 * Math.max(atrRatio * 2, p.stopLossPct * 0.5);
      const blendedTP = 0.7 * p.takeProfitPct + 0.3 * Math.max(atrRatio * 4, p.takeProfitPct * 0.5);
      stopLoss = direction === 'long'
        ? currentPrice * (1 - blendedSL)
        : currentPrice * (1 + blendedSL);
      takeProfit = direction === 'long'
        ? currentPrice * (1 + blendedTP)
        : currentPrice * (1 - blendedTP);
    } else {
      stopLoss = direction === 'long'
        ? currentPrice * (1 - p.stopLossPct)
        : currentPrice * (1 + p.stopLossPct);
      takeProfit = direction === 'long'
        ? currentPrice * (1 + p.takeProfitPct)
        : currentPrice * (1 - p.takeProfitPct);
    }

    return {
      direction,
      strength,
      confidence,
      entry_price: currentPrice,
      stop_loss: stopLoss,
      take_profit: takeProfit,
      position_size_pct: p.positionSizePct,
      reasons,
      indicators: {
        ...indicators,
        ema_cross: emaCross.current as 'bullish' | 'bearish' | 'none',
      },
      filtered: false,
    };
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  private filteredSignal(
    currentPrice: number,
    opts: { reason: string; bars: OHLCVBar[]; metadata: { fundingRate?: number; volume24h: number; liquidity: number } },
  ): StrategySignal {
    const emaCross = emaCrossState(opts.bars, this.params.emaFastPeriod, this.params.emaSlowPeriod);
    const rsiNow = opts.bars.length >= 15 ? latestRSI(opts.bars, 14) : null;

    return {
      direction: 'neutral',
      strength: 0,
      confidence: 0,
      entry_price: currentPrice,
      stop_loss: currentPrice * (1 - this.params.stopLossPct),
      take_profit: currentPrice * (1 + this.params.takeProfitPct),
      position_size_pct: 0,
      reasons: [],
      indicators: {
        rsi: rsiNow,
        ema_fast: emaCross.ema_fast,
        ema_slow: emaCross.ema_slow,
        ema_cross: emaCross.current as 'bullish' | 'bearish' | 'none',
        price_vs_ema_fast: 'at',
        funding_rate: opts.metadata.fundingRate ?? null,
        funding_bias: 'neutral',
        volume_ok: opts.metadata.volume24h >= this.params.minVolume24h,
        liquidity_ok: opts.metadata.liquidity >= this.params.minLiquidity,
      },
      filtered: true,
      filter_reason: opts.reason,
    };
  }
}
