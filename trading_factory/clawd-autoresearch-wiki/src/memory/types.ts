/**
 * Memory Types — Epistemological memory system for TradingAgent
 *
 * Three knowledge states:
 *   known    — observed facts (price, volume, on-chain data)
 *   learned  — derived insights from trade outcomes + patterns
 *   inferred — connections the agent has drawn between facts
 */

// ── Memory Entry ────────────────────────────────────────────────────────

export type MemoryType = 'known' | 'learned' | 'inferred';

export interface MemoryEntry {
  id: string;
  memory_type: MemoryType;
  source: string;              // e.g. 'birdeye', 'helius', 'agent', 'coingecko'
  topic: string;               // searchable topic slug
  asset?: string;              // token address or ticker
  asset_class?: string;        // 'crypto' | 'equity' | 'macro'
  content: string;             // human-readable content
  raw_data?: Record<string, unknown>;
  confidence?: number;         // 0-1, how confident the agent is
  metadata?: Record<string, unknown>;
  ttl_seconds?: number;        // auto-expire after this many seconds
  embedding?: number[];        // pgvector embedding (if available)
  created_at: string;
  updated_at: string;
}

export interface MemoryInput {
  memory_type: MemoryType;
  source: string;
  topic: string;
  asset?: string;
  asset_class?: string;
  content: string;
  raw_data?: Record<string, unknown>;
  confidence?: number;
  metadata?: Record<string, unknown>;
  ttl_seconds?: number;
}

// ── Trade Record ────────────────────────────────────────────────────────

export type TradeDirection = 'long' | 'short';
export type TradeStatus = 'open' | 'closed' | 'cancelled';
export type AgentMode = 'simulation' | 'paper' | 'live';

export interface TradeRecord {
  id: string;
  mode: AgentMode | string;
  asset: string;
  asset_class: string;
  direction: TradeDirection;
  entry_price: number;
  exit_price?: number;
  size: number;
  size_usd: number;
  status: TradeStatus;
  signal_source: string;
  thesis: string;
  confidence: number;
  memory_ids: string[];
  stop_loss?: number;
  take_profit?: number;
  pnl_usd?: number;
  pnl_pct?: number;
  notes?: string;
  entry_time?: string;
  exit_time?: string;
  created_at: string;
}

// ── Epistemological State ───────────────────────────────────────────────

export interface EpistemologicalState {
  asset: string;
  known_facts: MemoryEntry[];
  learned_insights: MemoryEntry[];
  inferred_connections: MemoryEntry[];
  knowledge_gap: string[];       // what the agent doesn't know yet
  confidence_summary: {
    overall: number;
    price_data: number;
    fundamentals: number;
    on_chain: number;
    patterns: number;
  };
}

// ── Knowledge Index ─────────────────────────────────────────────────────

export interface KnowledgeIndexEntry {
  entity_id: string;
  entity_type: string;
  has_price_data: boolean;
  has_fundamentals: boolean;
  learned_patterns?: string[];
  last_seen: string;
}
