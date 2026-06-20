/**
 * FinancialDatasets — Equity + macro data connector
 *
 * Uses financialdatasets.ai API for:
 *   - Stock price snapshots
 *   - OHLCV candles
 *   - Financial statements
 *   - Sector performance
 */

import { logger } from '../utils/logger.js';
import { config } from '../config.js';

const BASE_URL = config.financialDatasets.baseUrl;
const API_KEY = config.financialDatasets.apiKey;

async function fetchFD(path: string, params: Record<string, string> = {}): Promise<unknown> {
  const url = new URL(path, BASE_URL);
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);

  const headers: Record<string, string> = {
    'Accept': 'application/json',
  };
  if (API_KEY) headers['X-API-KEY'] = API_KEY;

  const res = await fetch(url.toString(), { headers, signal: AbortSignal.timeout(10000) });
  if (!res.ok) {
    throw new Error(`FinancialDatasets ${res.status}: ${await res.text().catch(() => 'no body')}`);
  }
  return res.json();
}

// ── Interfaces ──────────────────────────────────────────────────────────────

export interface PriceSnapshot {
  ticker: string;
  price: number;
  change_pct: number;
  volume: number;
  market_cap: number;
  timestamp: string;
}

export interface OHLCVCandle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface FinancialStatement {
  ticker: string;
  period: string;
  revenue: number;
  net_income: number;
  eps: number;
  pe_ratio: number;
}

// ── Public API ──────────────────────────────────────────────────────────────

class FinancialDatasetsClient {
  async getPriceSnapshot(ticker: string): Promise<PriceSnapshot> {
    try {
      const data = await fetchFD('/prices/snapshot', { ticker }) as {
        snapshot: {
          ticker: string;
          price: number;
          day_change: number;
          volume: number;
          market_cap: number;
          timestamp: string;
        };
      };

      const s = data.snapshot;
      return {
        ticker: s?.ticker ?? ticker,
        price: s?.price ?? 0,
        change_pct: s?.day_change ?? 0,
        volume: s?.volume ?? 0,
        market_cap: s?.market_cap ?? 0,
        timestamp: s?.timestamp ?? new Date().toISOString(),
      };
    } catch (err) {
      logger.warn('[FinancialDatasets] getPriceSnapshot failed', { ticker, err: (err as Error).message });
      return {
        ticker,
        price: 0,
        change_pct: 0,
        volume: 0,
        market_cap: 0,
        timestamp: new Date().toISOString(),
      };
    }
  }

  async getOHLCV(ticker: string, opts?: {
    interval?: string;
    limit?: number;
  }): Promise<OHLCVCandle[]> {
    try {
      const data = await fetchFD('/prices/historical', {
        ticker,
        interval: opts?.interval ?? 'day',
        limit: String(opts?.limit ?? 100),
      }) as { prices: OHLCVCandle[] };

      return data.prices ?? [];
    } catch (err) {
      logger.warn('[FinancialDatasets] getOHLCV failed', { ticker, err: (err as Error).message });
      return [];
    }
  }

  async getFinancials(ticker: string): Promise<FinancialStatement | null> {
    try {
      const data = await fetchFD('/financials', { ticker, period: 'annual', limit: '1' }) as {
        financials: FinancialStatement[];
      };

      return data.financials?.[0] ?? null;
    } catch (err) {
      logger.warn('[FinancialDatasets] getFinancials failed', { ticker, err: (err as Error).message });
      return null;
    }
  }

  async getSectorPerformance(): Promise<Record<string, number>> {
    try {
      const data = await fetchFD('/market/sectors') as {
        sectors: Array<{ name: string; performance: number }>;
      };

      const result: Record<string, number> = {};
      for (const s of data.sectors ?? []) {
        result[s.name] = s.performance;
      }
      return result;
    } catch (err) {
      logger.warn('[FinancialDatasets] getSectorPerformance failed', { err: (err as Error).message });
      return {};
    }
  }
}

export const financialDatasets = new FinancialDatasetsClient();
