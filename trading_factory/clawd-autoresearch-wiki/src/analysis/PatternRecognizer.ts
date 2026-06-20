/**
 * PatternRecognizer — Learns from trade outcomes
 *
 * Analyzes trade history to discover:
 *   - Winning/losing patterns (time of day, signal combinations)
 *   - Asset-specific behaviors (e.g. "SOL dumps after rapid pumps")
 *   - Strategy parameter correlations with outcomes
 *
 * Results are stored as 'learned' memories in MemoryEngine.
 */

import { memoryEngine } from '../memory/MemoryEngine.js';
import { logger } from '../utils/logger.js';
import type { TradeRecord } from '../memory/types.js';

export interface DiscoveredPattern {
  pattern_type: string;
  description: string;
  confidence: number;        // 0-1
  supporting_trades: number;
  asset?: string;
}

class PatternRecognizer {

  /**
   * Learn from a single completed trade.
   * Stores outcome as a 'learned' memory.
   */
  async learnFromTrade(trade: TradeRecord): Promise<void> {
    if (trade.status !== 'closed') return;

    const outcome = (trade.pnl_pct ?? 0) > 0 ? 'winning' : 'losing';
    const holdTimeMs = trade.exit_time && trade.entry_time
      ? new Date(trade.exit_time).getTime() - new Date(trade.entry_time).getTime()
      : 0;
    const holdMinutes = Math.round(holdTimeMs / 60000);

    const content = [
      `Trade outcome: ${outcome} trade on ${trade.asset}`,
      `Direction: ${trade.direction}, PnL: ${(trade.pnl_pct ?? 0).toFixed(2)}%`,
      `Hold time: ${holdMinutes}m`,
      `Signal: ${trade.signal_source}`,
      `Confidence at entry: ${trade.confidence.toFixed(2)}`,
      trade.notes ? `Notes: ${trade.notes}` : '',
    ].filter(Boolean).join('. ');

    await memoryEngine.remember({
      memory_type: 'learned',
      source: 'pattern_recognizer',
      topic: `trade pattern: ${outcome} ${trade.direction}`,
      asset: trade.asset,
      asset_class: trade.asset_class,
      content,
      confidence: 0.7,
      metadata: {
        pnl_pct: trade.pnl_pct,
        hold_minutes: holdMinutes,
        direction: trade.direction,
        signal_source: trade.signal_source,
      },
    });

    logger.info('[PatternRecognizer] Learned from trade', {
      asset: trade.asset,
      outcome,
      pnl_pct: trade.pnl_pct?.toFixed(2),
    });
  }

  /**
   * Analyze trade history for an asset and discover patterns.
   */
  async analyzeTradeHistory(asset: string): Promise<DiscoveredPattern[]> {
    const trades = await memoryEngine.getTradeHistory(asset, 50);
    const closed = trades.filter(t => t.status === 'closed');

    if (closed.length < 5) return []; // not enough data

    const patterns: DiscoveredPattern[] = [];

    // Pattern 1: Win rate by direction
    const longs = closed.filter(t => t.direction === 'long');
    const shorts = closed.filter(t => t.direction === 'short');

    if (longs.length >= 3) {
      const longWinRate = longs.filter(t => (t.pnl_pct ?? 0) > 0).length / longs.length;
      if (longWinRate > 0.65 || longWinRate < 0.35) {
        patterns.push({
          pattern_type: 'direction_bias',
          description: `${asset} longs have ${(longWinRate * 100).toFixed(0)}% win rate (${longs.length} trades)`,
          confidence: Math.min(0.9, 0.5 + longs.length * 0.05),
          supporting_trades: longs.length,
          asset,
        });
      }
    }

    if (shorts.length >= 3) {
      const shortWinRate = shorts.filter(t => (t.pnl_pct ?? 0) > 0).length / shorts.length;
      if (shortWinRate > 0.65 || shortWinRate < 0.35) {
        patterns.push({
          pattern_type: 'direction_bias',
          description: `${asset} shorts have ${(shortWinRate * 100).toFixed(0)}% win rate (${shorts.length} trades)`,
          confidence: Math.min(0.9, 0.5 + shorts.length * 0.05),
          supporting_trades: shorts.length,
          asset,
        });
      }
    }

    // Pattern 2: Average PnL by signal source
    const bySource = new Map<string, { wins: number; total: number; pnlSum: number }>();
    for (const t of closed) {
      const src = t.signal_source;
      if (!bySource.has(src)) bySource.set(src, { wins: 0, total: 0, pnlSum: 0 });
      const s = bySource.get(src)!;
      s.total++;
      s.pnlSum += t.pnl_pct ?? 0;
      if ((t.pnl_pct ?? 0) > 0) s.wins++;
    }

    for (const [source, stats] of bySource) {
      if (stats.total >= 3) {
        const avgPnl = stats.pnlSum / stats.total;
        const winRate = stats.wins / stats.total;
        patterns.push({
          pattern_type: 'signal_performance',
          description: `Signal '${source}': ${(winRate * 100).toFixed(0)}% WR, avg PnL ${avgPnl.toFixed(2)}% (${stats.total} trades)`,
          confidence: Math.min(0.85, 0.4 + stats.total * 0.05),
          supporting_trades: stats.total,
          asset,
        });
      }
    }

    // Pattern 3: Stop-loss hit rate (are stops too tight?)
    const stopHits = closed.filter(t => t.notes?.includes('stop_loss'));
    if (stopHits.length >= 3) {
      const stopRate = stopHits.length / closed.length;
      if (stopRate > 0.4) {
        patterns.push({
          pattern_type: 'risk_management',
          description: `${asset}: ${(stopRate * 100).toFixed(0)}% of trades hit stop-loss — stops may be too tight`,
          confidence: 0.75,
          supporting_trades: stopHits.length,
          asset,
        });
      }
    }

    // Store discovered patterns as learned memories
    for (const p of patterns) {
      await memoryEngine.remember({
        memory_type: 'learned',
        source: 'pattern_recognizer',
        topic: `pattern: ${p.pattern_type}`,
        asset: p.asset,
        content: p.description,
        confidence: p.confidence,
        metadata: {
          pattern_type: p.pattern_type,
          supporting_trades: p.supporting_trades,
        },
      });
    }

    if (patterns.length > 0) {
      logger.info('[PatternRecognizer] Patterns discovered', {
        asset,
        count: patterns.length,
        types: patterns.map(p => p.pattern_type),
      });
    }

    return patterns;
  }
}

export const patternRecognizer = new PatternRecognizer();
