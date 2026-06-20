/**
 * Aster DEX Perpetuals Connector
 * Futures trading on https://fapi.asterdex.com
 * Implements the official Aster Finance Futures API
 *
 * Public endpoints (NONE auth): exchangeInfo, depth, trades, klines, premiumIndex, ticker
 * Signed endpoints (HMAC SHA256): balance, positionRisk, orders
 */

import { createHmac } from "crypto";

const ASTER_FAPI = "https://fapi.asterdex.com";

// ── Interfaces ──────────────────────────────────────────────────────────

export interface AsterMarket {
  symbol: string;
  baseAsset: string;
  quoteAsset: string;
  status: string;
  contractType: string;
  pricePrecision: number;
  quantityPrecision: number;
}

export interface AsterOrderBookEntry {
  price: string;
  qty: string;
}

export interface AsterOrderBook {
  lastUpdateId: number;
  bids: [string, string][];
  asks: [string, string][];
}

export interface AsterTrade {
  id: number;
  price: string;
  qty: string;
  quoteQty: string;
  time: number;
  isBuyerMaker: boolean;
}

export interface AsterKline {
  openTime: number;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
  closeTime: number;
  quoteVolume: string;
  trades: number;
}

export interface AsterFundingRate {
  symbol: string;
  markPrice: string;
  indexPrice: string;
  lastFundingRate: string;
  nextFundingTime: number;
  time: number;
}

export interface AsterTicker24h {
  symbol: string;
  priceChange: string;
  priceChangePercent: string;
  lastPrice: string;
  volume: string;
  quoteVolume: string;
  highPrice: string;
  lowPrice: string;
  openPrice: string;
}

export interface AsterPosition {
  symbol: string;
  positionSide: string;
  positionAmt: string;
  entryPrice: string;
  markPrice: string;
  unRealizedProfit: string;
  liquidationPrice: string;
  leverage: string;
  marginType: string;
}

export interface AsterBalance {
  asset: string;
  balance: string;
  availableBalance: string;
  crossUnPnl: string;
}

// ── Connector ───────────────────────────────────────────────────────────

export class AsterConnector {
  private apiKey: string;
  private apiSecret: string;

  constructor(apiKey?: string, apiSecret?: string) {
    this.apiKey = apiKey ?? process.env.ASTER_API_KEY ?? "";
    this.apiSecret = apiSecret ?? process.env.ASTER_API_SECRET ?? "";
  }

  // ── Public Market Data (no auth) ────────────────────────────────────

  /** GET /fapi/v1/exchangeInfo — all available futures markets */
  async getMarkets(): Promise<AsterMarket[]> {
    const data = await this.publicGet<{
      symbols: AsterMarket[];
    }>("/fapi/v1/exchangeInfo");
    return data.symbols ?? [];
  }

  /** GET /fapi/v1/exchangeInfo — single market info */
  async getMarket(symbol: string): Promise<AsterMarket | null> {
    const markets = await this.getMarkets();
    return markets.find((m) => m.symbol === symbol) ?? null;
  }

  /** GET /fapi/v1/depth — order book */
  async getOrderBook(symbol: string, limit = 20): Promise<AsterOrderBook> {
    return this.publicGet<AsterOrderBook>(
      `/fapi/v1/depth?symbol=${symbol}&limit=${limit}`
    );
  }

  /** GET /fapi/v1/trades — recent trades */
  async getRecentTrades(symbol: string, limit = 50): Promise<AsterTrade[]> {
    return this.publicGet<AsterTrade[]>(
      `/fapi/v1/trades?symbol=${symbol}&limit=${limit}`
    );
  }

  /** GET /fapi/v1/klines — candlestick data */
  async getKlines(
    symbol: string,
    interval = "1h",
    limit = 100
  ): Promise<AsterKline[]> {
    const raw = await this.publicGet<unknown[][]>(
      `/fapi/v1/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`
    );
    return raw.map((k) => ({
      openTime: k[0] as number,
      open: k[1] as string,
      high: k[2] as string,
      low: k[3] as string,
      close: k[4] as string,
      volume: k[5] as string,
      closeTime: k[6] as number,
      quoteVolume: k[7] as string,
      trades: k[8] as number,
    }));
  }

  /** GET /fapi/v1/premiumIndex — mark price + funding rate */
  async getFundingRate(symbol: string): Promise<AsterFundingRate> {
    return this.publicGet<AsterFundingRate>(
      `/fapi/v1/premiumIndex?symbol=${symbol}`
    );
  }

  /** GET /fapi/v1/premiumIndex — all funding rates */
  async getAllFundingRates(): Promise<AsterFundingRate[]> {
    return this.publicGet<AsterFundingRate[]>("/fapi/v1/premiumIndex");
  }

  /** GET /fapi/v1/fundingRate — funding rate history */
  async getFundingHistory(
    symbol: string,
    limit = 20
  ): Promise<{ symbol: string; fundingRate: string; fundingTime: number }[]> {
    return this.publicGet(
      `/fapi/v1/fundingRate?symbol=${symbol}&limit=${limit}`
    );
  }

  /** GET /fapi/v1/ticker/24hr — 24h price change statistics */
  async getTicker24h(symbol?: string): Promise<AsterTicker24h | AsterTicker24h[]> {
    const endpoint = symbol
      ? `/fapi/v1/ticker/24hr?symbol=${symbol}`
      : "/fapi/v1/ticker/24hr";
    return this.publicGet(endpoint);
  }

  /** GET /fapi/v1/ticker/price — latest price */
  async getPrice(symbol: string): Promise<{ symbol: string; price: string }> {
    return this.publicGet(`/fapi/v1/ticker/price?symbol=${symbol}`);
  }

  /** GET /fapi/v1/ticker/bookTicker — best bid/ask */
  async getBookTicker(
    symbol: string
  ): Promise<{
    symbol: string;
    bidPrice: string;
    bidQty: string;
    askPrice: string;
    askQty: string;
  }> {
    return this.publicGet(`/fapi/v1/ticker/bookTicker?symbol=${symbol}`);
  }

  // ── Signed endpoints (require HMAC SHA256) ──────────────────────────

  /** GET /fapi/v2/positionRisk — open positions */
  async getPositions(symbol?: string): Promise<AsterPosition[]> {
    const params: Record<string, string> = {};
    if (symbol) params.symbol = symbol;
    return this.signedGet<AsterPosition[]>("/fapi/v2/positionRisk", params);
  }

  /** GET /fapi/v2/balance — account balances */
  async getBalance(): Promise<AsterBalance[]> {
    return this.signedGet<AsterBalance[]>("/fapi/v2/balance");
  }

  // ── Higher-level helpers ────────────────────────────────────────────

  /** Get a market digest: top markets by volume */
  async getMarketDigest(): Promise<{
    marketCount: number;
    topByVolume: AsterTicker24h[];
    totalVolume: number;
  }> {
    const tickers = (await this.getTicker24h()) as AsterTicker24h[];
    const sorted = [...tickers].sort(
      (a, b) => parseFloat(b.quoteVolume) - parseFloat(a.quoteVolume)
    );
    const totalVolume = tickers.reduce(
      (s, t) => s + parseFloat(t.quoteVolume || "0"),
      0
    );
    return {
      marketCount: tickers.length,
      topByVolume: sorted.slice(0, 10),
      totalVolume,
    };
  }

  /** Generate a simple signal from funding rate + price action */
  async generateSignal(symbol: string): Promise<{
    symbol: string;
    direction: "long" | "short" | "neutral";
    fundingRate: number;
    markPrice: number;
    priceChange24h: number;
    confidence: number;
  }> {
    const [funding, ticker] = await Promise.all([
      this.getFundingRate(symbol),
      this.getTicker24h(symbol) as Promise<AsterTicker24h>,
    ]);

    const fr = parseFloat(funding.lastFundingRate);
    const pch = parseFloat(ticker.priceChangePercent);
    const markPrice = parseFloat(funding.markPrice);

    // Very negative funding + positive price = long; very positive funding + negative price = short
    let direction: "long" | "short" | "neutral" = "neutral";
    let confidence = 0.3;

    if (fr < -0.0005 && pch > 0) {
      direction = "long";
      confidence = Math.min(0.9, 0.5 + Math.abs(fr) * 100);
    } else if (fr > 0.0005 && pch < 0) {
      direction = "short";
      confidence = Math.min(0.9, 0.5 + Math.abs(fr) * 100);
    }

    return {
      symbol,
      direction,
      fundingRate: fr,
      markPrice,
      priceChange24h: pch,
      confidence,
    };
  }

  // ── HTTP helpers ────────────────────────────────────────────────────

  private async publicGet<T>(endpoint: string): Promise<T> {
    const url = `${ASTER_FAPI}${endpoint}`;
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const res = await fetch(url, {
          headers: this.apiKey ? { "X-MBX-APIKEY": this.apiKey } : {},
          signal: AbortSignal.timeout(10000),
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`Aster HTTP ${res.status}: ${text}`);
        }
        return (await res.json()) as T;
      } catch (e) {
        if (attempt === 2) throw e;
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
      }
    }
    throw new Error("Aster fetch failed");
  }

  private async signedGet<T>(
    path: string,
    params: Record<string, string> = {}
  ): Promise<T> {
    if (!this.apiKey || !this.apiSecret) {
      throw new Error("Aster API key and secret required for signed endpoints");
    }

    const timestamp = Date.now().toString();
    const queryParts = { ...params, timestamp, recvWindow: "5000" };
    const queryString = Object.entries(queryParts)
      .map(([k, v]) => `${k}=${v}`)
      .join("&");

    const signature = createHmac("sha256", this.apiSecret)
      .update(queryString)
      .digest("hex");

    const url = `${ASTER_FAPI}${path}?${queryString}&signature=${signature}`;

    const res = await fetch(url, {
      headers: { "X-MBX-APIKEY": this.apiKey },
      signal: AbortSignal.timeout(10000),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Aster signed HTTP ${res.status}: ${text}`);
    }

    return (await res.json()) as T;
  }
}
