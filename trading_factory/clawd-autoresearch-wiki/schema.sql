-- ============================================================
-- AUTONOMOUS AGENT MEMORY SCHEMA
-- Supabase + pgvector — epistemological memory system
-- known (API facts) vs learned (agent-derived) vs inferred
-- ============================================================

-- Enable pgvector extension
create extension if not exists vector;
create extension if not exists "uuid-ossp";

-- ============================================================
-- CORE MEMORY TABLE
-- The heart of the system: everything the agent knows or has learned
-- ============================================================
create table agent_memories (
  id              uuid primary key default uuid_generate_v4(),
  created_at      timestamptz default now(),
  updated_at      timestamptz default now(),

  -- Epistemological classification
  memory_type     text not null check (memory_type in ('known', 'learned', 'inferred')),
  -- 'known'    = directly observed from API / market data (ground truth with TTL)
  -- 'learned'  = derived from trading outcomes, backtests, pattern analysis
  -- 'inferred' = cross-domain synthesis (e.g. macro + crypto correlation)

  source          text not null,
  -- e.g. 'birdeye', 'helius', 'financial_datasets', 'coingecko',
  --      'trade_outcome', 'pattern_analysis', 'agent_reasoning'

  topic           text not null,           -- e.g. 'SOL/USD price action', 'AAPL earnings'
  asset           text,                    -- optional: 'SOL', 'AAPL', 'BTC'
  asset_class     text,                    -- 'crypto', 'equity', 'commodity', 'macro'
  timeframe       text,                    -- '1h', '4h', '1d', 'event'

  content         text not null,           -- human-readable summary of the memory
  raw_data        jsonb,                   -- full structured payload (optional)
  metadata        jsonb default '{}',      -- tags, confidence, trade_id refs, etc.

  -- Vector embedding for semantic recall
  embedding       vector(1536),

  -- Memory health
  confidence      float default 1.0 check (confidence between 0 and 1),
  reinforcement   int default 1,           -- how many times this has been confirmed
  contradictions  int default 0,           -- how many times this has been challenged
  expires_at      timestamptz,             -- known facts expire; learned facts persist

  -- Linkage
  parent_memory_id uuid references agent_memories(id),
  trade_ids       uuid[],                  -- trades this memory relates to
  session_id      text                     -- agent session that created this
);

create index on agent_memories using ivfflat (embedding vector_cosine_ops) with (lists = 100);
create index on agent_memories (memory_type);
create index on agent_memories (source);
create index on agent_memories (asset);
create index on agent_memories (created_at desc);
create index on agent_memories (expires_at) where expires_at is not null;

-- ============================================================
-- TRADE RECORDS
-- Every trade the agent executes or simulates
-- ============================================================
create table trade_records (
  id              uuid primary key default uuid_generate_v4(),
  created_at      timestamptz default now(),
  settled_at      timestamptz,

  mode            text not null check (mode in ('live', 'simulated', 'backtest')),
  asset           text not null,
  asset_class     text not null,           -- 'crypto', 'equity'

  direction       text not null check (direction in ('long', 'short')),
  entry_price     numeric(20,8) not null,
  exit_price      numeric(20,8),
  size            numeric(20,8) not null,
  size_usd        numeric(20,4),

  -- Outcome
  pnl_usd         numeric(20,4),
  pnl_pct         numeric(10,4),
  status          text default 'open' check (status in ('open','closed','cancelled','failed')),

  -- What drove this trade
  signal_source   text,                    -- e.g. 'pattern_breakout', 'earnings_beat', 'whale_alert'
  thesis          text,                    -- human-readable reasoning
  confidence      float,                   -- agent confidence at entry
  memory_ids      uuid[],                  -- memories that informed this trade

  -- Risk
  stop_loss       numeric(20,8),
  take_profit     numeric(20,8),
  max_drawdown    numeric(10,4),

  -- Blockchain (crypto only)
  tx_signature    text,
  wallet          text,

  -- Post-trade learning
  outcome_notes   text,                    -- agent's post-mortem
  learned_memory_id uuid references agent_memories(id)
);

create index on trade_records (asset);
create index on trade_records (status);
create index on trade_records (created_at desc);

-- ============================================================
-- MARKET SNAPSHOTS
-- Raw market data ingested from APIs — short-lived, feeds the known layer
-- ============================================================
create table market_snapshots (
  id              uuid primary key default uuid_generate_v4(),
  captured_at     timestamptz default now(),

  source          text not null,           -- 'birdeye', 'helius', 'coingecko', 'financial_datasets'
  asset           text not null,
  asset_class     text not null,

  snapshot_type   text not null,
  -- 'price', 'ohlcv', 'orderbook', 'on_chain_stats',
  -- 'financial_metrics', 'earnings', 'sentiment'

  data            jsonb not null,
  ttl_seconds     int default 300          -- how long this data is considered fresh
);

create index on market_snapshots (asset, snapshot_type, captured_at desc);
create index on market_snapshots (captured_at desc);

-- ============================================================
-- RESEARCH REPORTS
-- Synthesized analysis the agent produces
-- ============================================================
create table research_reports (
  id              uuid primary key default uuid_generate_v4(),
  created_at      timestamptz default now(),

  title           text not null,
  asset           text,
  asset_class     text,
  report_type     text,                    -- 'technical', 'fundamental', 'sentiment', 'cross_asset'

  summary         text not null,
  full_text       text,
  data_sources    text[],                  -- which APIs were used
  memory_ids      uuid[],                  -- memories synthesized into this report
  confidence      float,

  embedding       vector(1536),            -- for semantic search across reports
  tags            text[]
);

create index on research_reports using ivfflat (embedding vector_cosine_ops) with (lists = 50);
create index on research_reports (asset, created_at desc);

-- ============================================================
-- LEARNING EVENTS
-- When the agent updates its beliefs — the learning audit trail
-- ============================================================
create table learning_events (
  id              uuid primary key default uuid_generate_v4(),
  created_at      timestamptz default now(),

  event_type      text not null,
  -- 'belief_update': agent changed its view on something
  -- 'pattern_discovered': new pattern found
  -- 'hypothesis_confirmed': predicted outcome happened
  -- 'hypothesis_rejected': predicted outcome didn't happen
  -- 'contradiction_resolved': conflicting memories reconciled

  description     text not null,
  old_belief      text,
  new_belief      text,
  evidence        jsonb,                   -- supporting data

  triggered_by    text,                    -- 'trade_outcome', 'new_data', 'correlation'
  trade_id        uuid references trade_records(id),
  affected_memory_ids uuid[],
  new_memory_id   uuid references agent_memories(id)
);

-- ============================================================
-- KNOWN FACTS CACHE
-- Explicit known/unknown tracking: has the agent ever fetched data on X?
-- ============================================================
create table knowledge_index (
  id              uuid primary key default uuid_generate_v4(),
  first_seen_at   timestamptz default now(),
  last_refreshed  timestamptz default now(),

  entity_type     text not null,           -- 'crypto_token', 'stock', 'macro_indicator'
  entity_id       text not null,           -- address, ticker, or indicator name
  entity_name     text,

  -- What we know about this entity
  has_price_data      bool default false,
  has_fundamentals    bool default false,
  has_on_chain_data   bool default false,
  has_news            bool default false,
  has_trade_history   bool default false,

  -- What we've learned about this entity
  learned_patterns    text[],              -- pattern names discovered
  learned_correlations jsonb,             -- {entity: correlation_coeff}
  learned_signals     text[],             -- signals that have worked

  summary         text,                    -- brief description of entity
  embedding       vector(1536),            -- for "do I know about X?" queries

  unique (entity_type, entity_id)
);

create index on knowledge_index using ivfflat (embedding vector_cosine_ops) with (lists = 50);
create index on knowledge_index (entity_type, entity_id);

-- ============================================================
-- HELPER: update updated_at on agent_memories
-- ============================================================
create or replace function update_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create trigger agent_memories_updated_at
  before update on agent_memories
  for each row execute function update_updated_at();

-- ============================================================
-- HELPER: semantic search function
-- ============================================================
create or replace function search_memories(
  query_embedding vector(1536),
  match_threshold float default 0.7,
  match_count int default 10,
  filter_type text default null,
  filter_asset text default null
)
returns table (
  id uuid,
  memory_type text,
  source text,
  topic text,
  asset text,
  content text,
  metadata jsonb,
  confidence float,
  created_at timestamptz,
  similarity float
) as $$
begin
  return query
  select
    m.id,
    m.memory_type,
    m.source,
    m.topic,
    m.asset,
    m.content,
    m.metadata,
    m.confidence,
    m.created_at,
    1 - (m.embedding <=> query_embedding) as similarity
  from agent_memories m
  where
    m.embedding is not null
    and (filter_type is null or m.memory_type = filter_type)
    and (filter_asset is null or m.asset = filter_asset)
    and (m.expires_at is null or m.expires_at > now())
    and 1 - (m.embedding <=> query_embedding) > match_threshold
  order by m.embedding <=> query_embedding
  limit match_count;
end;
$$ language plpgsql;

-- ============================================================
-- VIEW: What does the agent know vs what has it learned?
-- ============================================================
create view knowledge_vs_learned as
select
  asset,
  asset_class,
  count(*) filter (where memory_type = 'known') as known_facts,
  count(*) filter (where memory_type = 'learned') as learned_insights,
  count(*) filter (where memory_type = 'inferred') as inferred_connections,
  avg(confidence) filter (where memory_type = 'learned') as avg_learned_confidence,
  max(created_at) as most_recent_memory
from agent_memories
where expires_at is null or expires_at > now()
group by asset, asset_class
order by (known_facts + learned_insights) desc;

-- ============================================================
-- CLAWDBOT STRATEGY: Strategy state + changelog persistence
-- ============================================================

create table if not exists strategy_state (
  strategy_key  text primary key,
  state         jsonb        not null,      -- full StrategyRegistryState blob
  updated_at    timestamptz  not null default now()
);

comment on table strategy_state is
  'Persists ClawdBot active params, best metric, and changelog. One row per strategy key.';

-- Index for fast upserts
create index if not exists idx_strategy_state_updated
  on strategy_state(updated_at desc);

-- ── Helper view: latest params ────────────────────────────────────────────────

create or replace view strategy_active_params as
select
  strategy_key,
  state ->> 'last_updated'                       as last_updated,
  (state ->> 'best_metric')::float               as best_metric,
  state ->> 'metric_name'                        as metric_name,
  state -> 'active_params'                       as active_params,
  jsonb_array_length(state -> 'changelog')       as changelog_entries
from strategy_state;

comment on view strategy_active_params is
  'Quick read of active ClawdBot params without deserializing the full changelog.';

-- ── Helper view: changelog entries ───────────────────────────────────────────

create or replace view strategy_changelog as
select
  strategy_key,
  entry ->> 'id'                                 as entry_id,
  (entry ->> 'timestamp')::timestamptz           as changed_at,
  entry ->> 'reason'                             as reason,
  entry ->> 'triggered_by'                       as triggered_by,
  entry -> 'delta'                               as param_delta,
  (entry ->> 'metric_before')::float             as metric_before,
  (entry ->> 'metric_after')::float              as metric_after,
  entry ->> 'metric_name'                        as metric_name,
  entry -> 'new_params'                          as new_params
from strategy_state,
  jsonb_array_elements(state -> 'changelog') as entry
order by (entry ->> 'timestamp')::timestamptz desc;

comment on view strategy_changelog is
  'Flattened view of every ClawdBot parameter change with before/after metrics.';
