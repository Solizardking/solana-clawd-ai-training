/**
 * Config — Centralized configuration from environment variables
 *
 * All config reads from process.env at import time.
 * Supabase, OpenRouter, data source keys, and agent behavior.
 */

import { config as dotenvConfig } from 'dotenv';
dotenvConfig();

function env(key: string, fallback = ''): string {
  return process.env[key] ?? fallback;
}

function envInt(key: string, fallback: number): number {
  const val = process.env[key];
  if (!val) return fallback;
  const parsed = parseInt(val, 10);
  return isNaN(parsed) ? fallback : parsed;
}

function envFloat(key: string, fallback: number): number {
  const val = process.env[key];
  if (!val) return fallback;
  const parsed = parseFloat(val);
  return isNaN(parsed) ? fallback : parsed;
}

function envBool(key: string, fallback: boolean): boolean {
  const val = process.env[key];
  if (!val) return fallback;
  return val.toLowerCase() === 'true' || val === '1';
}

export const config = {
  // ── Supabase ──────────────────────────────────────────
  supabase: {
    url: env('SUPABASE_URL', ''),
    serviceKey: env('SUPABASE_SERVICE_KEY', env('SUPABASE_ANON_KEY', '')),
    anonKey: env('SUPABASE_ANON_KEY', ''),
  },

  // ── LLM ───────────────────────────────────────────────
  openrouter: {
    apiKey: env('OPENROUTER_API_KEY', ''),
    model: env('OPENROUTER_MODEL', 'openai/gpt-5.4'),
  },

  // ── Clawd Models ──────────────────────────────────────
  clawd: {
    provider: env('CLAWD_MODEL_PROVIDER', 'ollama'),
    model: env('CLAWD_MODEL', '8bit/solana-clawd-core-ai'),
    tradingModel: env('CLAWD_TRADING_MODEL', '8bit/solana-trading-factory'),
    ollamaUrl: env('CLAWD_OLLAMA_URL', 'http://localhost:11434/v1'),
    hfToken: env('HF_TOKEN', ''),
  },

  // ── Data Sources ──────────────────────────────────────
  helius: {
    apiKey: env('HELIUS_API_KEY', ''),
    rpcUrl: env('HELIUS_RPC_URL', ''),
    wsUrl: env('HELIUS_WSS_URL', ''),
  },

  birdeye: {
    apiKey: env('BIRDEYE_API_KEY', ''),
    baseUrl: env('BIRDEYE_BASE_URL', 'https://public-api.birdeye.so'),
  },

  aster: {
    apiKey: env('ASTER_API_KEY', ''),
  },

  coingecko: {
    apiKey: env('COINGECKO_API_KEY', ''),
    baseUrl: env('COINGECKO_BASE_URL', 'https://api.coingecko.com/api/v3'),
  },

  financialDatasets: {
    apiKey: env('FINANCIAL_DATASETS_API_KEY', ''),
    baseUrl: env('FINANCIAL_DATASETS_BASE_URL', 'https://api.financialdatasets.ai'),
  },

  // ── Agent ─────────────────────────────────────────────
  agent: {
    mode: env('AGENT_MODE', 'simulation') as 'simulation' | 'paper' | 'live',
    walletPubkey: env('WALLET_PUBKEY', ''),
    watchlist: env('WATCHLIST', '').split(',').filter(Boolean),
    equityWatchlist: env('EQUITY_WATCHLIST', '').split(',').filter(Boolean),
    vaultPath: env('VAULT_PATH', './vault'),
    oodaIntervalMs: envInt('OODA_INTERVAL_MS', 60000),
    maxExperiments: envInt('MAX_EXPERIMENTS', 50),
    cyclePeriodMs: envInt('CYCLE_PERIOD_MS', 300000),   // 5 min
    learnPeriodMs: envInt('LEARN_PERIOD_MS', 1800000),  // 30 min
    minSignalStrength: envFloat('MIN_SIGNAL_STRENGTH', 0.55),
    minConfidence: envFloat('MIN_CONFIDENCE', 0.6),
    maxPositions: envInt('MAX_POSITIONS', 5),
  },

  // ── Bridge ────────────────────────────────────────────
  bridge: {
    port: envInt('BRIDGE_PORT', 3777),
  },

  // ── Feature Flags ─────────────────────────────────────
  features: {
    useSupabase: envBool('USE_SUPABASE', false),
    usePerps: envBool('USE_PERPS', true),
    reasoningEnabled: envBool('REASONING_ENABLED', true),
  },
};
