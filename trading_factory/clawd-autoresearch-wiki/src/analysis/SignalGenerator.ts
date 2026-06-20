/**
 * SignalGenerator — Multi-source trading signal synthesis
 *
 * Combines:
 *   - ClawdBotStrategy (RSI+EMA+ATR from OHLCV bars)
 *   - Birdeye token analytics (volume, liquidity, holders)
 *   - Aster DEX perpetual data (funding rates, OI)
 *   - FinancialDatasets equity data (for equity watchlist)
 *   - MemoryEngine recalled knowledge (patterns, past outcomes)
 *
 * Outputs a unified TradingSignal with direction, strength, confidence,
 * entry/exit zones, thesis, and source attribution.
 */

import { BirdeyeConnector } from '../data/birdeye.js';
import { AsterConnector } from '../data/aster.js';
import { financialDatasets } from '../data/financial.js';
import { memoryEngine } from '../memory/MemoryEngine.js';
import { strategyRegistry } from '../strategy/StrategyRegistry.js';
import { logger } from '../utils/logger.js';
import { config } from '../config.js';
import type { OHLCVBar } from '../strategy/types.js';

// ── Signal Type ─────────────────────────────────────────────────────────────

export interface TradingSignal {
  asset: string;
  asset_class: 'crypto' | 'equity';
  direction: 'long' | 'short' | 'neutral';
  strength: number;                // 0-1
  confidence: number;              // 0-1
  entry_zone?: [number, number];   // [low, high]
  stop_loss?: number;
  take_profit?: number;
  position_size_pct?: number;
  thesis: string;                  // human-readable reasoning
  sources: string[];               // which data sources contributed
  memory_ids: string[];            // related memory entries
  indicators: Record<string, unknown>;
  timestamp: string;
}

// ── Signal Generator ────────────────────────────────────────────────────────

class SignalGenerator {
  private birdeye: BirdeyeConnector;
  private aster: AsterConnector;

  constructor() {
    this.birdeye = new BirdeyeConnector(config.birdeye.apiKey);
    this.aster = new AsterConnector(config.aster.apiKey);
  }

  /**
   * Generate a signal for a crypto token.
   */
  async cryptoSignal(token: {
    address: string;
    symbol?: string;
  }): Promise<TradingSignal> {
    const ts = new Date().toISOString();
    const sources: string[] = [];
    const memoryIds: string[] = [];
    const thesisParts: string[] = [];

    let currentPrice = 0;
    let volume24h = 0;
    let liquidity = 0;
    let fundingRate: number | undefined;

    // 1. Birdeye — price + volume + OHLCV
    try {
      const overview = await this.birdeye.getTokenOverview(token.address);
      currentPrice = overview.price ?? 0;
      volume24h = overview.volume24h ?? 0;
      liquidity = overview.liquidity ?? 0;
      sources.push('birdeye');
      thesisParts.push(`Price: $${currentPrice.toFixed(6)}, 24h Vol: $${volume24h.toLocaleString()}`);
    } catch (err) {
      logger.warn('[SignalGenerator] Birdeye overview failed', { err: (err as Error).message });
    }

    // 2. Get OHLCV bars for strategy engine
    let bars: OHLCVBar[] = [];
    try {
      const ohlcv = await this.birdeye.getOHLCV(token.address, '1H', 100);
      bars = ohlcv.map(b => ({
        timestamp: b.unixTime * 1000,
        open: b.o,
        high: b.h,
        low: b.l,
        close: b.c,
        volume: b.v,
      }));
      sources.push('birdeye_ohlcv');
    } catch (err) {
      logger.warn('[SignalGenerator] Birdeye OHLCV failed', { err: (err as Error).message });
    }

    // 3. Aster — funding rates (if available)
    try {
      const markets = await this.aster.getMarkets();
      const market = markets.find(m => m.baseAsset?.toLowerCase().includes(token.symbol?.toLowerCase() ?? ''));
      if (market) {
        const premium = await this.aster.getFundingRate(market.symbol);
        fundingRate = parseFloat(premium.lastFundingRate) || undefined;
        sources.push('aster_perps');
        thesisParts.push(`Funding rate: ${((fundingRate ?? 0) * 100).toFixed(4)}%`);
      }
    } catch (err) {
      logger.debug('[SignalGenerator] Aster funding failed', { err: (err as Error).message });
    }

    // 4. Run ClawdBotStrategy
    let stratSignal = null;
    if (bars.length >= 30) {
      try {
        const strategy = strategyRegistry.strategy;
        stratSignal = strategy.evaluate(bars, {
          volume24h,
          liquidity,
          fundingRate,
          symbol: token.symbol ?? token.address,
        });
        sources.push('clawdbot_rsi_ema');

        if (!stratSignal.filtered) {
          thesisParts.push(`Strategy signal: ${stratSignal.direction.toUpperCase()} (strength ${(stratSignal.strength * 100).toFixed(0)}%)`);
          for (const r of stratSignal.reasons) thesisParts.push(`  • ${r}`);
        } else {
          thesisParts.push(`Signal filtered: ${stratSignal.filter_reason}`);
        }
      } catch (err) {
        logger.warn('[SignalGenerator] Strategy evaluation failed', { err: (err as Error).message });
      }
    }

    // 5. Recall relevant memories
    try {
      const memories = await memoryEngine.recall(token.symbol ?? token.address, {
        asset: token.address,
        limit: 5,
      });

      if (memories.length > 0) {
        sources.push('memory');
        for (const m of memories) {
          memoryIds.push(m.id);
          thesisParts.push(`[Memory: ${m.memory_type}] ${m.content.slice(0, 80)}`);
        }
      }
    } catch {
      // Memory unavailable
    }

    // 6. Combine into final signal
    if (stratSignal && !stratSignal.filtered) {
      return {
        asset: token.address,
        asset_class: 'crypto',
        direction: stratSignal.direction,
        strength: stratSignal.strength,
        confidence: stratSignal.confidence,
        entry_zone: currentPrice > 0
          ? [currentPrice * 0.998, currentPrice * 1.002]
          : undefined,
        stop_loss: stratSignal.stop_loss,
        take_profit: stratSignal.take_profit,
        position_size_pct: stratSignal.position_size_pct,
        thesis: thesisParts.join('\n'),
        sources,
        memory_ids: memoryIds,
        indicators: stratSignal.indicators as unknown as Record<string, unknown>,
        timestamp: ts,
      };
    }

    // Neutral — no actionable signal
    return {
      asset: token.address,
      asset_class: 'crypto',
      direction: 'neutral',
      strength: 0,
      confidence: 0.3,
      thesis: thesisParts.length > 0
        ? thesisParts.join('\n')
        : `No data available for ${token.address}`,
      sources,
      memory_ids: memoryIds,
      indicators: stratSignal?.indicators as unknown as Record<string, unknown> ?? {},
      timestamp: ts,
    };
  }

  /**
   * Generate a signal for an equity ticker.
   */
  async equitySignal(ticker: string): Promise<TradingSignal> {
    const ts = new Date().toISOString();
    const sources: string[] = [];
    const thesisParts: string[] = [];
    const memoryIds: string[] = [];

    // 1. Price snapshot
    let currentPrice = 0;
    let changePct = 0;
    try {
      const snap = await financialDatasets.getPriceSnapshot(ticker);
      currentPrice = snap.price;
      changePct = snap.change_pct;
      sources.push('financial_datasets');
      thesisParts.push(`${ticker}: $${currentPrice.toFixed(2)} (${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%)`);
    } catch (err) {
      logger.warn('[SignalGenerator] Equity snapshot failed', { ticker, err: (err as Error).message });
    }

    // 2. Fundamentals
    try {
      const fin = await financialDatasets.getFinancials(ticker);
      if (fin) {
        sources.push('fundamentals');
        thesisParts.push(`P/E: ${fin.pe_ratio?.toFixed(1) ?? 'N/A'}, EPS: $${fin.eps?.toFixed(2) ?? 'N/A'}`);

        // Simple valuation signal
        if (fin.pe_ratio && fin.pe_ratio < 15) {
          thesisParts.push('Low P/E suggests potential value play');
        } else if (fin.pe_ratio && fin.pe_ratio > 40) {
          thesisParts.push('High P/E — growth priced in or overvalued');
        }
      }
    } catch {
      // Not critical
    }

    // 3. Recall memories
    try {
      const memories = await memoryEngine.recall(ticker, { asset: ticker, limit: 3 });
      for (const m of memories) {
        memoryIds.push(m.id);
        thesisParts.push(`[Memory] ${m.content.slice(0, 80)}`);
      }
      if (memories.length > 0) sources.push('memory');
    } catch {
      // Not critical
    }

    // 4. Simple momentum signal for equities
    let direction: 'long' | 'short' | 'neutral' = 'neutral';
    let strength = 0;
    let confidence = 0.3;

    // Positive momentum + not overbought
    if (changePct > 1.0 && changePct < 5.0) {
      direction = 'long';
      strength = Math.min(0.7, changePct / 10);
      confidence = 0.5;
      thesisParts.push('Positive momentum — potential continuation');
    } else if (changePct < -3.0) {
      direction = 'short';
      strength = Math.min(0.6, Math.abs(changePct) / 15);
      confidence = 0.45;
      thesisParts.push('Significant daily decline — potential trend continuation');
    }

    const params = strategyRegistry.getParams();

    return {
      asset: ticker,
      asset_class: 'equity',
      direction,
      strength,
      confidence,
      entry_zone: currentPrice > 0
        ? [currentPrice * 0.998, currentPrice * 1.002]
        : undefined,
      stop_loss: currentPrice > 0 ? currentPrice * (1 - params.stopLossPct) : undefined,
      take_profit: currentPrice > 0 ? currentPrice * (1 + params.takeProfitPct) : undefined,
      position_size_pct: params.positionSizePct,
      thesis: thesisParts.join('\n'),
      sources,
      memory_ids: memoryIds,
      indicators: { change_pct: changePct },
      timestamp: ts,
    };
  }
}

export const signalGenerator = new SignalGenerator();
