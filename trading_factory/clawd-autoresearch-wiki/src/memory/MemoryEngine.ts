/**
 * MemoryEngine — Epistemological Memory System
 *
 * Supabase-backed when available, falls back to in-memory store.
 *
 * Three memory tiers:
 *   known    — raw observations (price snapshots, on-chain events)
 *   learned  — insights derived from trade outcomes
 *   inferred — connections the agent draws between facts
 *
 * Public API:
 *   remember()       — store a new memory
 *   recall()         — search memories by query
 *   whatDoIKnow()    — epistemological state for an asset
 *   recordTrade()    — open a new trade
 *   closeTrade()     — close and record outcome
 *   getTradeHistory()— fetch recent trades
 *   listKnownAssets()— all entities the agent has data on
 */

import { createClient, type SupabaseClient } from '@supabase/supabase-js';
import { randomUUID } from 'crypto';
import { logger } from '../utils/logger.js';
import { config } from '../config.js';
import type {
  MemoryEntry,
  MemoryInput,
  TradeRecord,
  EpistemologicalState,
  KnowledgeIndexEntry,
} from './types.js';

class MemoryEngine {
  private supabase: SupabaseClient | null = null;
  private memoryStore: MemoryEntry[] = [];
  private tradeStore: TradeRecord[] = [];
  private initialized = false;

  async init(): Promise<void> {
    if (this.initialized) return;

    if (config.supabase.url && config.supabase.serviceKey) {
      try {
        this.supabase = createClient(config.supabase.url, config.supabase.serviceKey);
        logger.info('[MemoryEngine] Connected to Supabase');
      } catch (err) {
        logger.warn('[MemoryEngine] Supabase connection failed, using in-memory', {
          err: (err as Error).message,
        });
      }
    } else {
      logger.info('[MemoryEngine] No Supabase config, using in-memory store');
    }

    this.initialized = true;
  }

  // ── Remember ──────────────────────────────────────────────────────────────

  async remember(input: MemoryInput): Promise<string> {
    await this.init();
    const now = new Date().toISOString();
    const id = randomUUID();

    const entry: MemoryEntry = {
      id,
      ...input,
      created_at: now,
      updated_at: now,
    };

    if (this.supabase) {
      try {
        const { error } = await this.supabase
          .from('agent_memories')
          .insert({
            id,
            memory_type: input.memory_type,
            source: input.source,
            topic: input.topic,
            asset: input.asset,
            asset_class: input.asset_class,
            content: input.content,
            raw_data: input.raw_data,
            confidence: input.confidence,
            metadata: input.metadata,
            ttl_seconds: input.ttl_seconds,
          });

        if (error) {
          logger.warn('[MemoryEngine] Supabase insert failed', { error: error.message });
          this.memoryStore.push(entry);
        }
      } catch {
        this.memoryStore.push(entry);
      }
    } else {
      this.memoryStore.push(entry);
      // Keep in-memory store bounded
      if (this.memoryStore.length > 5000) {
        this.memoryStore = this.memoryStore.slice(-4000);
      }
    }

    return id;
  }

  // ── Recall ────────────────────────────────────────────────────────────────

  async recall(query: string, opts?: {
    asset?: string;
    memory_type?: MemoryEntry['memory_type'];
    limit?: number;
  }): Promise<MemoryEntry[]> {
    await this.init();
    const limit = opts?.limit ?? 20;

    if (this.supabase) {
      try {
        let q = this.supabase
          .from('agent_memories')
          .select('*')
          .order('created_at', { ascending: false })
          .limit(limit);

        if (opts?.asset) q = q.eq('asset', opts.asset);
        if (opts?.memory_type) q = q.eq('memory_type', opts.memory_type);
        if (query) q = q.ilike('content', `%${query}%`);

        const { data, error } = await q;
        if (!error && data) return data as MemoryEntry[];
      } catch {
        // Fall through to in-memory
      }
    }

    // In-memory search
    const lower = query.toLowerCase();
    return this.memoryStore
      .filter(m => {
        if (opts?.asset && m.asset !== opts.asset) return false;
        if (opts?.memory_type && m.memory_type !== opts.memory_type) return false;
        if (query && !m.content.toLowerCase().includes(lower) && !m.topic.toLowerCase().includes(lower)) return false;
        return true;
      })
      .sort((a, b) => b.created_at.localeCompare(a.created_at))
      .slice(0, limit);
  }

  // ── Epistemological State ─────────────────────────────────────────────────

  async whatDoIKnow(asset: string): Promise<EpistemologicalState> {
    await this.init();

    const [known, learned, inferred] = await Promise.all([
      this.recall('', { asset, memory_type: 'known', limit: 50 }),
      this.recall('', { asset, memory_type: 'learned', limit: 50 }),
      this.recall('', { asset, memory_type: 'inferred', limit: 20 }),
    ]);

    // Identify knowledge gaps
    const gaps: string[] = [];
    const hasPriceData = known.some(m => m.topic.includes('price') || m.source === 'birdeye');
    const hasOnChain = known.some(m => m.source === 'helius');
    const hasFundamentals = known.some(m => m.topic.includes('fundamental') || m.topic.includes('holder'));
    const hasPatterns = learned.some(m => m.topic.includes('pattern'));

    if (!hasPriceData) gaps.push('No recent price data');
    if (!hasOnChain) gaps.push('No on-chain activity data');
    if (!hasFundamentals) gaps.push('No fundamental analysis');
    if (!hasPatterns) gaps.push('No learned patterns yet');

    // Confidence summary
    const avgConfidence = (entries: MemoryEntry[]): number => {
      if (entries.length === 0) return 0;
      return entries.reduce((s, e) => s + (e.confidence ?? 0.5), 0) / entries.length;
    };

    return {
      asset,
      known_facts: known,
      learned_insights: learned,
      inferred_connections: inferred,
      knowledge_gap: gaps,
      confidence_summary: {
        overall: avgConfidence([...known, ...learned, ...inferred]),
        price_data: hasPriceData ? avgConfidence(known.filter(m => m.topic.includes('price'))) : 0,
        fundamentals: hasFundamentals ? 0.6 : 0,
        on_chain: hasOnChain ? avgConfidence(known.filter(m => m.source === 'helius')) : 0,
        patterns: hasPatterns ? avgConfidence(learned.filter(m => m.topic.includes('pattern'))) : 0,
      },
    };
  }

  // ── Trade Management ──────────────────────────────────────────────────────

  async recordTrade(trade: Omit<TradeRecord, 'id' | 'created_at'>): Promise<string> {
    await this.init();
    const id = randomUUID();
    const now = new Date().toISOString();

    const record: TradeRecord = {
      ...trade,
      id,
      entry_time: now,
      created_at: now,
    };

    if (this.supabase) {
      try {
        const { error } = await this.supabase.from('trade_records').insert(record);
        if (error) {
          logger.warn('[MemoryEngine] Trade insert failed', { error: error.message });
          this.tradeStore.push(record);
        }
      } catch {
        this.tradeStore.push(record);
      }
    } else {
      this.tradeStore.push(record);
    }

    return id;
  }

  async closeTrade(tradeId: string, result: {
    exit_price: number;
    pnl_usd: number;
    pnl_pct: number;
    notes?: string;
  }): Promise<void> {
    await this.init();
    const now = new Date().toISOString();

    if (this.supabase) {
      try {
        const { error } = await this.supabase
          .from('trade_records')
          .update({
            status: 'closed',
            exit_price: result.exit_price,
            pnl_usd: result.pnl_usd,
            pnl_pct: result.pnl_pct,
            notes: result.notes,
            exit_time: now,
          })
          .eq('id', tradeId);

        if (error) {
          logger.warn('[MemoryEngine] Trade close failed', { error: error.message });
        }
      } catch {
        // Fall through to in-memory
      }
    }

    // Also update in-memory if present
    const local = this.tradeStore.find(t => t.id === tradeId);
    if (local) {
      local.status = 'closed';
      local.exit_price = result.exit_price;
      local.pnl_usd = result.pnl_usd;
      local.pnl_pct = result.pnl_pct;
      local.notes = result.notes;
      local.exit_time = now;
    }

    // Learn from outcome
    const outcome = result.pnl_pct > 0 ? 'win' : result.pnl_pct < 0 ? 'loss' : 'breakeven';
    await this.remember({
      memory_type: 'learned',
      source: 'agent',
      topic: `trade outcome: ${outcome}`,
      content: `Trade ${tradeId} closed: PnL ${result.pnl_pct.toFixed(2)}% ($${result.pnl_usd.toFixed(2)}). ${result.notes ?? ''}`,
      confidence: 0.9,
      metadata: { tradeId, ...result },
    });
  }

  async getTradeHistory(asset?: string, limit = 20): Promise<TradeRecord[]> {
    await this.init();

    if (this.supabase) {
      try {
        let q = this.supabase
          .from('trade_records')
          .select('*')
          .order('created_at', { ascending: false })
          .limit(limit);

        if (asset) q = q.eq('asset', asset);

        const { data, error } = await q;
        if (!error && data) return data as TradeRecord[];
      } catch {
        // Fall through
      }
    }

    // In-memory
    return this.tradeStore
      .filter(t => !asset || t.asset === asset)
      .sort((a, b) => b.created_at.localeCompare(a.created_at))
      .slice(0, limit);
  }

  // ── Knowledge Index ───────────────────────────────────────────────────────

  async listKnownAssets(): Promise<KnowledgeIndexEntry[]> {
    await this.init();

    if (this.supabase) {
      try {
        const { data, error } = await this.supabase
          .from('knowledge_index')
          .select('*')
          .order('last_seen', { ascending: false })
          .limit(100);

        if (!error && data) return data as KnowledgeIndexEntry[];
      } catch {
        // Fall through
      }
    }

    // Build from in-memory store
    const assetMap = new Map<string, KnowledgeIndexEntry>();
    for (const m of this.memoryStore) {
      if (!m.asset) continue;
      if (!assetMap.has(m.asset)) {
        assetMap.set(m.asset, {
          entity_id: m.asset,
          entity_type: m.asset_class ?? 'unknown',
          has_price_data: false,
          has_fundamentals: false,
          learned_patterns: [],
          last_seen: m.created_at,
        });
      }
      const entry = assetMap.get(m.asset)!;
      if (m.topic.includes('price')) entry.has_price_data = true;
      if (m.topic.includes('fundamental')) entry.has_fundamentals = true;
      if (m.memory_type === 'learned' && m.topic.includes('pattern')) {
        entry.learned_patterns?.push(m.content.slice(0, 60));
      }
      if (m.created_at > entry.last_seen) entry.last_seen = m.created_at;
    }

    return Array.from(assetMap.values());
  }
}

export const memoryEngine = new MemoryEngine();
