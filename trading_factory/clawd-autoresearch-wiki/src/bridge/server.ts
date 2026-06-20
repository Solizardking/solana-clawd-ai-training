/**
 * ClawdBot Bridge Server
 *
 * HTTP API that bridges the Python autoresearch system and the
 * TypeScript ClawdBot agent. Enables:
 *
 * - Python → TS: Train results → agent memory, strategy updates
 * - TS → Python: Agent triggers training runs, reads results
 * - Automation: Cron-like endpoints for full research cycles
 *
 * Endpoints:
 *   POST /api/agent/chat      — Chat with agent
 *   POST /api/agent/observe    — Trigger OODA observation
 *   POST /api/agent/research   — Start research loop
 *   POST /api/agent/remember   — Store to vault
 *   GET  /api/agent/recall     — Query vault
 *   GET  /api/agent/status     — Agent status + token usage
 *   POST /api/python/result    — Report Python training result
 *   POST /api/python/trigger   — Trigger Python training run
 *   GET  /api/python/results   — Get all Python results
 *   POST /api/automate/full    — Full automation cycle
 *   GET  /api/health           — Healthcheck
 */

import http from "http";
import { ClawdBot } from "../agent/clawdbot.js";
import { TradingAgent } from "../agent/TradingAgent.js";
import { ResearchLoop } from "../research/loop.js";
import { BirdeyeConnector } from "../data/birdeye.js";
import { AsterConnector } from "../data/aster.js";
import { spawn } from "child_process";
import fs from "fs/promises";
import path from "path";

const PORT = parseInt(process.env.BRIDGE_PORT ?? "3777");

// ── State ─────────────────────────────────────────────────────────────

let agent: ClawdBot | null = null;
let tradingAgent: TradingAgent | null = null;
let researchLoop: ResearchLoop | null = null;
let birdeye: BirdeyeConnector | null = null;
let aster: AsterConnector | null = null;
let pythonResults: Array<{
  timestamp: string;
  metric: number;
  description: string;
  params: Record<string, unknown>;
}> = [];
let isAutomationRunning = false;

// Initialize data connectors
function ensureDataConnectors() {
  if (!birdeye && process.env.BIRDEYE_API_KEY) {
    birdeye = new BirdeyeConnector(process.env.BIRDEYE_API_KEY);
  }
  if (!aster && process.env.ASTER_API_KEY) {
    aster = new AsterConnector(process.env.ASTER_API_KEY);
  }
}

// ── Config ────────────────────────────────────────────────────────────

function getConfig() {
  return {
    heliusApiKey: process.env.HELIUS_API_KEY ?? "",
    heliusRpcUrl: process.env.HELIUS_RPC_URL ?? "",
    heliusWsUrl: process.env.HELIUS_WSS_URL,
    birdeyeApiKey: process.env.BIRDEYE_API_KEY ?? "",
    asterApiKey: process.env.ASTER_API_KEY ?? "",
    vaultPath: process.env.VAULT_PATH ?? "./vault",
    watchlist: (process.env.WATCHLIST ?? "").split(",").filter(Boolean),
    walletPubkey: process.env.WALLET_PUBKEY,
    oodaIntervalMs: parseInt(process.env.OODA_INTERVAL_MS ?? "60000"),
  };
}

async function ensureAgent(): Promise<ClawdBot> {
  if (!agent) {
    agent = new ClawdBot(getConfig());
    await agent.wake();
  }
  return agent;
}

// ── HTTP Helpers ──────────────────────────────────────────────────────

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

function json(
  res: http.ServerResponse,
  data: unknown,
  status = 200
): void {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  });
  res.end(JSON.stringify(data, null, 2));
}

function error(
  res: http.ServerResponse,
  message: string,
  status = 400
): void {
  json(res, { error: message }, status);
}

// ── Python Integration ───────────────────────────────────────────────

async function triggerPythonTraining(
  description?: string
): Promise<{ pid: number; started: boolean }> {
  return new Promise((resolve) => {
    const proc = spawn("python", ["train.py"], {
      cwd: process.cwd(),
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env },
    });

    console.log(`🐍 Python training started (PID: ${proc.pid})`);

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
      process.stdout.write(`[python] ${data}`);
    });
    proc.stderr.on("data", (data) => {
      stderr += data.toString();
      process.stderr.write(`[python:err] ${data}`);
    });

    proc.on("close", async (code) => {
      console.log(`🐍 Python training finished (exit: ${code})`);

      // Try to extract val_bpb from output
      const bpbMatch = stdout.match(/val_bpb[:\s=]+([0-9.]+)/i);
      const metric = bpbMatch ? parseFloat(bpbMatch[1]) : 0;

      const result = {
        timestamp: new Date().toISOString(),
        metric,
        description: description ?? `Training run (exit: ${code})`,
        params: { exitCode: code, stdout: stdout.slice(-500), stderr: stderr.slice(-200) },
      };
      pythonResults.push(result);

      // If agent is running, store result to vault
      if (agent) {
        await agent.remember(
          `## Python Training Result\n\nval_bpb: ${metric}\n${description ?? ""}\n\n${stdout.slice(-500)}`,
          { category: "research", score: 0.8 }
        );
      }
    });

    resolve({ pid: proc.pid ?? 0, started: true });
  });
}

async function readResultsTSV(): Promise<string[]> {
  try {
    const raw = await fs.readFile("results.tsv", "utf-8");
    return raw.split("\n").filter(Boolean);
  } catch {
    return [];
  }
}

// ── Route Handler ─────────────────────────────────────────────────────

async function handleRequest(
  req: http.IncomingMessage,
  res: http.ServerResponse
): Promise<void> {
  const url = new URL(req.url ?? "/", `http://localhost:${PORT}`);
  const pathname = url.pathname;
  const method = req.method ?? "GET";

  // CORS preflight
  if (method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    });
    res.end();
    return;
  }

  try {
    // ── Health ──────────────────────────────────────────────────
    if (pathname === "/api/health") {
      json(res, {
        status: "ok",
        agent: agent ? "running" : "idle",
        tradingAgent: tradingAgent?.isRunning() ? "running" : "idle",
        automation: isAutomationRunning,
        pythonResults: pythonResults.length,
        model: process.env.OPENROUTER_MODEL ?? "openai/gpt-5.4",
        birdeye: !!birdeye || !!process.env.BIRDEYE_API_KEY,
        aster: !!aster || !!process.env.ASTER_API_KEY,
      });
      return;
    }

    // ── Birdeye Real-Time Data ─────────────────────────────────
    if (pathname.startsWith("/api/birdeye/") && method === "GET") {
      ensureDataConnectors();
      if (!birdeye) {
        error(res, "BIRDEYE_API_KEY not configured", 503);
        return;
      }

      const subpath = pathname.replace("/api/birdeye/", "");

      // GET /api/birdeye/token/:address
      if (subpath.startsWith("token/")) {
        const address = subpath.replace("token/", "").trim();
        if (!address || address.length < 32 || address.length > 50 || !/^[A-Za-z0-9]+$/.test(address)) {
          error(res, "Invalid Solana address format");
          return;
        }
        const overview = await birdeye.getTokenOverview(address);
        json(res, overview);
        return;
      }

      // GET /api/birdeye/ohlcv/:address?interval=1H&limit=100
      if (subpath.startsWith("ohlcv/")) {
        const address = subpath.replace("ohlcv/", "");
        const interval = (url.searchParams.get("interval") ?? "1H") as import("../data/birdeye.js").OHLCVInterval;
        const limit = parseInt(url.searchParams.get("limit") ?? "100");
        const candles = await birdeye.getOHLCV(address, interval, limit);
        json(res, { address, interval, count: candles.length, candles });
        return;
      }

      // GET /api/birdeye/trades/:address?limit=20
      if (subpath.startsWith("trades/")) {
        const address = subpath.replace("trades/", "");
        const limit = parseInt(url.searchParams.get("limit") ?? "20");
        const trades = await birdeye.getTrades(address, limit);
        json(res, { address, count: trades.length, trades });
        return;
      }

      // GET /api/birdeye/top-traders/:address
      if (subpath.startsWith("top-traders/")) {
        const address = subpath.replace("top-traders/", "");
        const traders = await birdeye.getTopTraders(address);
        json(res, { address, count: traders.length, traders });
        return;
      }

      // GET /api/birdeye/wallet/:wallet
      if (subpath.startsWith("wallet/")) {
        const wallet = subpath.replace("wallet/", "");
        const [portfolio, pnl] = await Promise.all([
          birdeye.getWalletPortfolio(wallet),
          birdeye.getWalletPnL(wallet).catch(() => null),
        ]);
        json(res, { wallet, portfolio, pnl });
        return;
      }

      // GET /api/birdeye/signals/:address
      if (subpath.startsWith("signals/")) {
        const address = subpath.replace("signals/", "");
        const signals = await birdeye.getTechnicalSignals(address);
        json(res, { address, ...signals });
        return;
      }

      // GET /api/birdeye/trending
      if (subpath === "trending") {
        const tokens = await birdeye.getTrendingTokens(20);
        json(res, { count: tokens.length, tokens });
        return;
      }

      // GET /api/birdeye/new-listings
      if (subpath === "new-listings") {
        const tokens = await birdeye.getNewListings(20);
        json(res, { count: tokens.length, tokens });
        return;
      }

      error(res, `Unknown birdeye endpoint: ${subpath}`, 404);
      return;
    }

    // ── Aster DEX Data ─────────────────────────────────────────
    if (pathname.startsWith("/api/aster/") && method === "GET") {
      ensureDataConnectors();
      if (!aster) {
        error(res, "ASTER_API_KEY not configured", 503);
        return;
      }

      const subpath = pathname.replace("/api/aster/", "");

      // GET /api/aster/markets
      if (subpath === "markets") {
        const markets = await aster.getMarkets();
        json(res, { count: markets.length, markets });
        return;
      }

      // GET /api/aster/market/:symbol
      if (subpath.startsWith("market/")) {
        const symbol = subpath.replace("market/", "");
        const market = await aster.getMarket(symbol);
        json(res, market);
        return;
      }

      // GET /api/aster/orderbook/:symbol?depth=20
      if (subpath.startsWith("orderbook/")) {
        const symbol = subpath.replace("orderbook/", "");
        const depth = parseInt(url.searchParams.get("depth") ?? "20");
        const book = await aster.getOrderBook(symbol, depth);
        json(res, book);
        return;
      }

      // GET /api/aster/funding/:symbol
      if (subpath.startsWith("funding/")) {
        const symbol = subpath.replace("funding/", "");
        const history = await aster.getFundingHistory(symbol);
        json(res, { symbol, count: history.length, history });
        return;
      }

      // GET /api/aster/positions[/:symbol]
      if (subpath === "positions" || subpath.startsWith("positions/")) {
        const symbol = subpath === "positions" ? undefined : subpath.replace("positions/", "");
        const positions = await aster.getPositions(symbol);
        json(res, { count: positions.length, positions });
        return;
      }

      // GET /api/aster/digest
      if (subpath === "digest") {
        const digest = await aster.getMarketDigest();
        json(res, digest);
        return;
      }

      // GET /api/aster/funding-rates
      if (subpath === "funding-rates") {
        const rates = await aster.getAllFundingRates();
        json(res, rates);
        return;
      }

      // GET /api/aster/signal/:symbol
      if (subpath.startsWith("signal/")) {
        const symbol = subpath.replace("signal/", "");
        const signal = await aster.generateSignal(symbol);
        json(res, signal);
        return;
      }

      error(res, `Unknown aster endpoint: ${subpath}`, 404);
      return;
    }

    // ── Trading Agent ──────────────────────────────────────────
    if (pathname === "/api/trading/start" && method === "POST") {
      if (tradingAgent?.isRunning()) {
        json(res, { status: "already_running" });
        return;
      }
      tradingAgent = new TradingAgent();
      await tradingAgent.start();
      json(res, { status: "started" });
      return;
    }

    if (pathname === "/api/trading/stop" && method === "POST") {
      if (tradingAgent) {
        tradingAgent.stop();
        json(res, { status: "stopped" });
      } else {
        json(res, { status: "not_running" });
      }
      return;
    }

    if (pathname === "/api/trading/status" && method === "GET") {
      json(res, {
        running: tradingAgent?.isRunning() ?? false,
        cycleCount: tradingAgent?.getCycleCount() ?? 0,
        lastCycleAt: tradingAgent?.getLastCycleAt(),
        openPositions: tradingAgent?.getOpenPositions().length ?? 0,
      });
      return;
    }

    if (pathname === "/api/trading/signals" && method === "GET") {
      const limit = parseInt(url.searchParams.get("limit") ?? "20");
      json(res, { signals: tradingAgent?.getRecentSignals(limit) ?? [] });
      return;
    }

    if (pathname === "/api/trading/positions" && method === "GET") {
      json(res, { positions: tradingAgent?.getOpenPositions() ?? [] });
      return;
    }

    if (pathname === "/api/trading/history" && method === "GET") {
      const limit = parseInt(url.searchParams.get("limit") ?? "50");
      const history = await tradingAgent?.getTradeHistory(limit) ?? [];
      json(res, { count: history.length, trades: history });
      return;
    }

    // ── Agent Chat ─────────────────────────────────────────────
    if (pathname === "/api/agent/chat" && method === "POST") {
      const body = JSON.parse(await readBody(req));
      const a = await ensureAgent();
      const response = await a.chat(body.message);
      json(res, {
        response,
        tokenUsage: a.tokenUsage,
      });
      return;
    }

    // ── Agent Observe ──────────────────────────────────────────
    if (pathname === "/api/agent/observe" && method === "POST") {
      const a = await ensureAgent();
      const events: unknown[] = [];
      for await (const event of a.oodaCycleStream()) {
        events.push(event);
      }
      json(res, { events });
      return;
    }

    // ── Agent Research ─────────────────────────────────────────
    if (pathname === "/api/agent/research" && method === "POST") {
      const body = JSON.parse(await readBody(req));
      const cfg = getConfig();

      researchLoop = new ResearchLoop({
        birdeyeApiKey: cfg.birdeyeApiKey,
        asterApiKey: cfg.asterApiKey,
        vaultPath: cfg.vaultPath,
      });

      await researchLoop.init();

      // Run in background
      const maxExperiments = body.maxExperiments ?? 10;
      researchLoop
        .run(maxExperiments)
        .catch((e) => console.error("Research error:", e));

      json(res, {
        status: "started",
        maxExperiments,
        model: process.env.OPENROUTER_MODEL ?? "openai/gpt-5.4",
      });
      return;
    }

    // ── Agent Remember ─────────────────────────────────────────
    if (pathname === "/api/agent/remember" && method === "POST") {
      const body = JSON.parse(await readBody(req));
      const a = await ensureAgent();
      const entry = await a.remember(body.content, body.opts);
      json(res, { entry: { id: entry.id, category: entry.category, title: entry.title } });
      return;
    }

    // ── Agent Recall ───────────────────────────────────────────
    if (pathname === "/api/agent/recall" && method === "GET") {
      const query = url.searchParams.get("q") ?? "";
      const a = await ensureAgent();
      const memories = await a.recall(query);
      json(res, {
        query,
        count: memories.length,
        memories: memories.map((m) => ({
          id: m.id,
          category: m.category,
          title: m.title,
          score: m.score,
          content: m.content.slice(0, 500),
        })),
      });
      return;
    }

    // ── Agent Status ───────────────────────────────────────────
    if (pathname === "/api/agent/status" && method === "GET") {
      json(res, {
        agent: agent ? "running" : "idle",
        tokenUsage: agent?.tokenUsage,
        model: process.env.OPENROUTER_MODEL ?? "openai/gpt-5.4",
        automation: isAutomationRunning,
      });
      return;
    }

    // ── Python Result Report ───────────────────────────────────
    if (pathname === "/api/python/result" && method === "POST") {
      const body = JSON.parse(await readBody(req));
      const result = {
        timestamp: new Date().toISOString(),
        metric: body.metric ?? 0,
        description: body.description ?? "",
        params: body.params ?? {},
      };
      pythonResults.push(result);

      if (agent) {
        await agent.remember(
          `## Python Result\n\nMetric: ${result.metric}\n${result.description}`,
          { category: "research", score: 0.7 }
        );
      }

      json(res, { stored: true, total: pythonResults.length });
      return;
    }

    // ── Python Trigger ─────────────────────────────────────────
    if (pathname === "/api/python/trigger" && method === "POST") {
      const body = JSON.parse(await readBody(req));
      const result = await triggerPythonTraining(body.description);
      json(res, result);
      return;
    }

    // ── Python Results ─────────────────────────────────────────
    if (pathname === "/api/python/results" && method === "GET") {
      const tsvResults = await readResultsTSV();
      json(res, { pythonResults, tsvResults });
      return;
    }

    // ── Full Automation Cycle ──────────────────────────────────
    if (pathname === "/api/automate/full" && method === "POST") {
      if (isAutomationRunning) {
        json(res, { error: "Automation already running" }, 409);
        return;
      }

      isAutomationRunning = true;

      // Run in background
      (async () => {
        try {
          console.log("🤖 Full automation cycle starting...");

          // 1. Start agent
          const a = await ensureAgent();
          console.log("   ✅ Agent active");

          // 2. Run TS strategy research
          const cfg = getConfig();
          const loop = new ResearchLoop({
            birdeyeApiKey: cfg.birdeyeApiKey,
            asterApiKey: cfg.asterApiKey,
            vaultPath: cfg.vaultPath,
          });
          await loop.init();
          await loop.run(5); // Quick 5 experiments
          console.log("   ✅ TS research complete");

          // 3. Run Python training
          console.log("   🐍 Triggering Python training...");
          await triggerPythonTraining("Automated research cycle");

          // 4. Chat with agent about results
          await a.chat("Summarize the latest research and training results.");

          console.log("🤖 Full automation cycle complete");
        } catch (e) {
          console.error("Automation error:", e);
        } finally {
          isAutomationRunning = false;
        }
      })();

      json(res, {
        status: "started",
        message: "Full automation cycle running in background",
      });
      return;
    }

    // ── Dexter Research Engine ──────────────────────────────────
    if (pathname === "/api/dexter/research" && method === "POST") {
      const body = JSON.parse(await readBody(req));
      const query = body.query || body.message;
      if (!query) {
        error(res, "Missing query");
        return;
      }

      console.log(`🔬 [Dexter] Research query: "${query.slice(0, 80)}…"`);

      try {
        // Spawn Dexter as a child process with the query piped to stdin
        const dexterDir = path.resolve(process.cwd(), "dexter-main");
        const result = await new Promise<{ answer: string; toolCalls: string[]; model: string }>((resolve, reject) => {
          const proc = spawn("bun", ["run", "src/index.tsx"], {
            cwd: dexterDir,
            stdio: ["pipe", "pipe", "pipe"],
            env: {
              ...process.env,
              // Ensure Dexter picks up the shared .env
              DOTENV_CONFIG_PATH: path.join(dexterDir, ".env"),
            },
          });

          let stdout = "";
          let stderr = "";

          proc.stdout.on("data", (chunk) => (stdout += chunk.toString()));
          proc.stderr.on("data", (chunk) => (stderr += chunk.toString()));

          // Send query + exit after first response
          setTimeout(() => {
            proc.stdin.write(query + "\n");
            // Give it time to process, then send /exit
            setTimeout(() => {
              proc.stdin.write("/exit\n");
            }, 30000);
          }, 2000);

          proc.on("close", (code) => {
            // Extract the answer from stdout (Dexter outputs markdown)
            const lines = stdout.split("\n");
            // Look for the final answer block after tool execution
            const answer = stdout
              .replace(/\x1b\[[0-9;]*m/g, '') // strip ANSI codes
              .replace(/[─═╔╗╚╝║▓░▒]/g, '') // strip box chars
              .trim();

            resolve({
              answer: answer || `Dexter completed with code ${code}. ${stderr.slice(0, 200)}`,
              toolCalls: [],
              model: "gpt-5.4",
            });
          });

          proc.on("error", (err) => reject(err));

          // Timeout after 60s
          setTimeout(() => {
            proc.kill();
            resolve({ answer: "Research timed out after 60s", toolCalls: [], model: "gpt-5.4" });
          }, 60000);
        });

        console.log(`🔬 [Dexter] Research complete (${result.answer.length} chars)`);
        json(res, result);
      } catch (e) {
        console.error("[Dexter] Error:", e);
        error(res, `Dexter research failed: ${(e as Error).message}`, 500);
      }
      return;
    }

    // ── Dexter Status ──────────────────────────────────────────
    if (pathname === "/api/dexter/status" && method === "GET") {
      const dexterDir = path.resolve(process.cwd(), "dexter-main");
      const exists = await fs.access(path.join(dexterDir, "package.json")).then(() => true).catch(() => false);
      json(res, {
        installed: exists,
        model: "gpt-5.4 (Anthropic/OpenRouter)",
        tools: [
          "financial_search", "financial_metrics", "read_filings",
          "web_search", "web_fetch", "browser",
          "birdeye_token", "birdeye_trending",
          "aster_orderbook", "aster_funding", "aster_ticker",
          "memory_search", "memory_get", "memory_update",
          "heartbeat", "skill",
        ],
      });
      return;
    }

    // ── Dashboard Static Files ──────────────────────────────────
    if (!pathname.startsWith('/api/')) {
      const dashboardDir = path.resolve(process.cwd(), 'dashboard');
      const safePath = pathname === '/' ? '/index.html' : pathname;
      const filePath = path.join(dashboardDir, safePath);

      // Prevent directory traversal
      if (!filePath.startsWith(dashboardDir)) {
        error(res, 'Forbidden', 403);
        return;
      }

      try {
        const content = await fs.readFile(filePath);
        const ext = path.extname(filePath).toLowerCase();
        const mimeTypes: Record<string, string> = {
          '.html': 'text/html',
          '.css': 'text/css',
          '.js': 'application/javascript',
          '.json': 'application/json',
          '.png': 'image/png',
          '.jpg': 'image/jpeg',
          '.svg': 'image/svg+xml',
          '.ico': 'image/x-icon',
        };
        res.writeHead(200, {
          'Content-Type': mimeTypes[ext] || 'application/octet-stream',
          'Cache-Control': 'no-cache',
        });
        res.end(content);
        return;
      } catch {
        // Fall through to 404
      }
    }

    // ── 404 ────────────────────────────────────────────────────
    error(res, `Not found: ${pathname}`, 404);
  } catch (e) {
    console.error("Request error:", e);
    error(res, (e as Error).message, 500);
  }
}

// ── Server ────────────────────────────────────────────────────────────

const server = http.createServer(handleRequest);

server.listen(PORT, () => {
  console.log(`
╔═══════════════════════════════════════════════════════════════╗
║  🦞  ClawdBot Bridge Server                                   ║
║  Port: ${PORT}                                                  ║
║  Model: ${(process.env.OPENROUTER_MODEL ?? "openai/gpt-5.4").padEnd(48)}║
║  Dashboard: http://localhost:${PORT}/                            ║
║                                                               ║
║  Endpoints:                                                   ║
║  POST /api/agent/chat     — Chat with ClawdBot                 ║
║  POST /api/agent/observe  — OODA observation cycle            ║
║  POST /api/agent/research — Start research loop               ║
║  POST /api/agent/remember — Store to vault                    ║
║  GET  /api/agent/recall   — Query vault (?q=...)              ║
║  GET  /api/agent/status   — Agent + model status              ║
║  POST /api/python/result  — Report Python training result     ║
║  POST /api/python/trigger — Trigger train.py                  ║
║  GET  /api/python/results — All Python results                ║
║  POST /api/automate/full  — Full automation cycle             ║
║  POST /api/dexter/research— Deep research via Dexter          ║
║  GET  /api/dexter/status  — Dexter engine status              ║
║  GET  /api/health         — Healthcheck                       ║
╚═══════════════════════════════════════════════════════════════╝
`);
});
