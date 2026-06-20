/**
 * Helius Data Connector
 * Real-time Solana data via Helius RPC + Enhanced API
 * - WebSocket subscriptions for live txn stream
 * - Enhanced transaction parsing
 * - Token metadata + balances
 * - Priority fee oracle
 */

import WebSocket from "ws";
import { EventEmitter } from "events";

const HELIUS_BASE = "https://api.helius.xyz/v0";

export interface HeliusTokenMetadata {
  mint: string;
  symbol: string;
  name: string;
  decimals: number;
  supply: number;
  logoURI?: string;
}

export interface HeliusTransaction {
  signature: string;
  timestamp: number;
  fee: number;
  feePayer: string;
  type: string;
  source: string;
  tokenTransfers: Array<{
    mint: string;
    fromUserAccount: string;
    toUserAccount: string;
    tokenAmount: number;
  }>;
  nativeTransfers: Array<{
    fromUserAccount: string;
    toUserAccount: string;
    amount: number;
  }>;
  accountData: Array<{
    account: string;
    nativeBalanceChange: number;
    tokenBalanceChanges: unknown[];
  }>;
  events?: Record<string, unknown>;
}

export interface PriorityFeeEstimate {
  min: number;
  low: number;
  medium: number;
  high: number;
  veryHigh: number;
  unsafeMax: number;
}

export interface TokenHolder {
  address: string;
  amount: number;
  decimals: number;
  owner: string;
  rank: number;
}

export type HeliusEvent =
  | { type: "transaction"; data: HeliusTransaction }
  | { type: "accountUpdate"; data: { pubkey: string; lamports: number } }
  | { type: "tokenPrice"; data: { mint: string; price: number } };

export class HeliusConnector extends EventEmitter {
  private apiKey: string;
  private rpcUrl: string;
  private wsUrl: string;
  private ws: WebSocket | null = null;
  private subscriptions = new Map<string, number>(); // id → subscription id
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private isConnected = false;

  constructor(apiKey: string, rpcUrl: string, wsUrl?: string) {
    super();
    this.apiKey = apiKey;
    this.rpcUrl = rpcUrl;
    this.wsUrl = wsUrl ?? rpcUrl.replace("https://", "wss://");
  }

  // ── WebSocket Stream ─────────────────────────────────────────────────

  async connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.wsUrl);

      this.ws.on("open", () => {
        this.isConnected = true;
        console.log("🔌 Helius WebSocket connected");
        resolve();
      });

      this.ws.on("message", (data: Buffer) => {
        try {
          const msg = JSON.parse(data.toString()) as Record<string, unknown>;
          this.handleWebSocketMessage(msg);
        } catch (e) {
          // ignore malformed
        }
      });

      this.ws.on("close", () => {
        this.isConnected = false;
        console.log("⚡ Helius WS disconnected, reconnecting in 5s...");
        this.scheduleReconnect();
      });

      this.ws.on("error", (err) => {
        console.error("Helius WS error:", err.message);
        reject(err);
      });
    });
  }

  private handleWebSocketMessage(msg: Record<string, unknown>): void {
    if (msg.method === "logsNotification") {
      const params = msg.params as { result?: { value?: { logs?: string[]; signature?: string } } };
      this.emit("log", params?.result?.value);
    } else if (msg.method === "accountNotification") {
      const params = msg.params as { result?: { value?: unknown }; subscription?: number };
      this.emit("accountUpdate", {
        type: "accountUpdate",
        data: params?.result?.value,
        subscription: params?.subscription,
      });
    } else if (msg.result !== undefined && typeof msg.id === "string") {
      // Subscription confirmation
      this.subscriptions.set(msg.id as string, msg.result as number);
    }
  }

  subscribeToAccount(pubkey: string): void {
    if (!this.ws || !this.isConnected) return;
    const id = `account-${pubkey}`;
    const request = {
      jsonrpc: "2.0",
      id,
      method: "accountSubscribe",
      params: [pubkey, { encoding: "jsonParsed", commitment: "confirmed" }],
    };
    this.ws.send(JSON.stringify(request));
  }

  subscribeToLogs(mentions?: string[]): void {
    if (!this.ws || !this.isConnected) return;
    const filter = mentions ? { mentions } : "all";
    const request = {
      jsonrpc: "2.0",
      id: "logs-sub",
      method: "logsSubscribe",
      params: [filter, { commitment: "confirmed" }],
    };
    this.ws.send(JSON.stringify(request));
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connectWebSocket().catch(console.error);
    }, 5000);
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  // ── Enhanced Transactions ────────────────────────────────────────────

  async getEnhancedTransactions(
    address: string,
    limit = 10,
    type?: string
  ): Promise<HeliusTransaction[]> {
    const url = new URL(`${HELIUS_BASE}/addresses/${address}/transactions`);
    url.searchParams.set("api-key", this.apiKey);
    url.searchParams.set("limit", limit.toString());
    if (type) url.searchParams.set("type", type);

    const res = await this.fetchWithRetry(url.toString());
    return res as HeliusTransaction[];
  }

  async parseTransaction(signature: string): Promise<HeliusTransaction | null> {
    const url = `${HELIUS_BASE}/transactions?api-key=${this.apiKey}`;
    const res = await this.fetchWithRetry(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transactions: [signature] }),
    });
    const arr = res as HeliusTransaction[];
    return arr[0] ?? null;
  }

  // ── Token Data ───────────────────────────────────────────────────────

  async getTokenMetadata(mints: string[]): Promise<HeliusTokenMetadata[]> {
    const url = `${HELIUS_BASE}/token-metadata?api-key=${this.apiKey}`;
    const res = await this.fetchWithRetry(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mintAccounts: mints }),
    });
    return res as HeliusTokenMetadata[];
  }

  async getTokenHolders(mint: string, limit = 20): Promise<TokenHolder[]> {
    const url = `${HELIUS_BASE}/token-holders?api-key=${this.apiKey}&mint=${mint}&limit=${limit}`;
    const res = await this.fetchWithRetry(url);
    return res as TokenHolder[];
  }

  async getAccountBalance(pubkey: string): Promise<{ sol: number; tokens: unknown[] }> {
    const rpcRes = await this.rpcCall("getBalance", [pubkey]);
    const sol = (rpcRes as { value: number }).value / 1e9;

    const tokenRes = await this.rpcCall("getTokenAccountsByOwner", [
      pubkey,
      { programId: "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA" },
      { encoding: "jsonParsed" },
    ]);

    const tokens = ((tokenRes as { value: unknown[] }).value ?? []);
    return { sol, tokens };
  }

  // ── Priority Fees ────────────────────────────────────────────────────

  async getPriorityFeeEstimate(accountKeys: string[]): Promise<PriorityFeeEstimate> {
    const url = `${HELIUS_BASE}/priority-fee-estimate?api-key=${this.apiKey}`;
    const res = await this.fetchWithRetry(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ accountKeys }),
    });
    return res as PriorityFeeEstimate;
  }

  // ── Webhooks (for persistent monitoring) ────────────────────────────

  async createWebhook(opts: {
    webhookURL: string;
    addresses: string[];
    type: "enhanced" | "raw" | "discord";
  }): Promise<{ webhookID: string }> {
    const url = `${HELIUS_BASE}/webhooks?api-key=${this.apiKey}`;
    const res = await this.fetchWithRetry(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        webhookURL: opts.webhookURL,
        transactionTypes: ["Any"],
        accountAddresses: opts.addresses,
        webhookType: opts.type,
      }),
    });
    return res as { webhookID: string };
  }

  // ── RPC Helpers ──────────────────────────────────────────────────────

  async rpcCall(method: string, params: unknown[]): Promise<unknown> {
    const res = await this.fetchWithRetry(this.rpcUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
    });
    const rpcRes = res as { result?: unknown; error?: { message: string } };
    if (rpcRes.error) throw new Error(`RPC error: ${rpcRes.error.message}`);
    return rpcRes.result;
  }

  async getSlot(): Promise<number> {
    return (await this.rpcCall("getSlot", [])) as number;
  }

  async getRecentBlockhash(): Promise<string> {
    const res = await this.rpcCall("getLatestBlockhash", [{ commitment: "finalized" }]);
    return ((res as { value: { blockhash: string } }).value.blockhash);
  }

  // ── Network Utils ────────────────────────────────────────────────────

  private async fetchWithRetry(
    url: string,
    opts: RequestInit = {},
    retries = 3
  ): Promise<unknown> {
    let lastErr: Error = new Error("unknown");
    for (let i = 0; i < retries; i++) {
      try {
        const res = await fetch(url, {
          ...opts,
          signal: AbortSignal.timeout(10000),
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`HTTP ${res.status}: ${text}`);
        }
        return await res.json();
      } catch (e) {
        lastErr = e as Error;
        if (i < retries - 1) {
          await new Promise((r) => setTimeout(r, 1000 * 2 ** i));
        }
      }
    }
    throw lastErr;
  }
}
