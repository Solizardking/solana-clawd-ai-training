/**
 * ClawdBot Agent — Entry Point
 *
 * Modes:
 *   start    — Live trading agent (OODA loop)
 *   research — Overnight auto-research loop
 *   chat     — Interactive terminal chat
 *   recall   — Query memory vault
 *   bridge   — HTTP bridge server (Python ↔ TS)
 *
 * LLM: OpenRouter GPT-5.4 with reasoning
 */

import { config as dotenv } from "dotenv";
dotenv();

import readline from "readline";
import chalk from "chalk";
import { ClawdBot } from "./agent/clawdbot.js";
import { ResearchLoop } from "./research/loop.js";

const BANNER = `
╔═══════════════════════════════════════════════════════════════╗
║  🦞  ClawdBot Agent v2.0.0                                    ║
║  LLM: OpenRouter GPT-5.4 (reasoning enabled)                 ║
║  Memory: ClawVault + Dexter Scratchpad                        ║
║  Foundation: OpenClaw Agent Runtime                           ║
╚═══════════════════════════════════════════════════════════════╝
`;

// ── Config ────────────────────────────────────────────────────────────

function requireEnv(key: string): string {
  const val = process.env[key];
  if (!val) throw new Error(`Missing required env: ${key}`);
  return val;
}

function getConfig() {
  return {
    heliusApiKey: requireEnv("HELIUS_API_KEY"),
    heliusRpcUrl: requireEnv("HELIUS_RPC_URL"),
    heliusWsUrl: process.env.HELIUS_WSS_URL,
    birdeyeApiKey: requireEnv("BIRDEYE_API_KEY"),
    asterApiKey: requireEnv("ASTER_API_KEY"),
    walletPubkey: process.env.WALLET_PUBKEY,
    vaultPath: process.env.VAULT_PATH ?? "./vault",
  };
}

// ── Modes ─────────────────────────────────────────────────────────────

async function runLiveAgent(): Promise<void> {
  console.log(chalk.cyan(BANNER));
  console.log(chalk.green("🟢 MODE: Live Agent (OODA Loop + GPT-5.4 reasoning)"));

  const cfg = getConfig();
  const agent = new ClawdBot({
    ...cfg,
    watchlist: (process.env.WATCHLIST ?? "").split(",").filter(Boolean),
    oodaIntervalMs: parseInt(process.env.OODA_INTERVAL_MS ?? "60000"),
  });

  process.on("SIGINT", async () => {
    console.log("\n\n📴 Shutdown signal received...");
    await agent.sleep();
    process.exit(0);
  });

  await agent.wake();
  await agent.startOODALoop();
}

async function runResearch(): Promise<void> {
  console.log(chalk.cyan(BANNER));
  console.log(chalk.yellow("🔬 MODE: Auto-Research Loop (GPT-5.4 reasoning)"));

  const cfg = getConfig();
  const loop = new ResearchLoop({
    birdeyeApiKey: cfg.birdeyeApiKey,
    asterApiKey: cfg.asterApiKey,
    vaultPath: cfg.vaultPath,
  });

  process.on("SIGINT", () => {
    console.log("\n📴 Stopping research loop...");
    loop.stop();
  });

  await loop.init();

  const maxExperiments = parseInt(process.env.MAX_EXPERIMENTS ?? "50");
  const experimentBudgetMs = parseInt(
    process.env.EXPERIMENT_BUDGET_MS ?? "300000"
  );

  await loop.run(maxExperiments, experimentBudgetMs);
}

async function runChat(): Promise<void> {
  console.log(chalk.cyan(BANNER));
  console.log(chalk.magenta("💬 MODE: Interactive Chat (GPT-5.4 + reasoning)"));

  const cfg = getConfig();
  const agent = new ClawdBot({
    ...cfg,
    watchlist: (process.env.WATCHLIST ?? "").split(",").filter(Boolean),
  });

  await agent.wake();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: chalk.cyan("\n🦞 You: "),
  });

  console.log(
    chalk.dim(
      "\nCommands: !recall <query>, !remember <text>, !trades, !research <mint>, !checkpoint, !exit\n"
    )
  );

  rl.prompt();

  rl.on("line", async (line) => {
    const input = line.trim();
    if (!input) {
      rl.prompt();
      return;
    }

    if (input === "!exit") {
      await agent.sleep();
      rl.close();
      process.exit(0);
    }

    process.stdout.write(chalk.yellow("\n🦞 ClawdBot: "));

    try {
      let response: string;

      if (input.startsWith("!recall ")) {
        const query = input.slice(8);
        const memories = await agent.recall(query);
        response =
          memories.length > 0
            ? memories
                .map((m) => `**${m.title}**\n${m.content.slice(0, 200)}`)
                .join("\n\n---\n\n")
            : "No memories found for that query.";
      } else if (input.startsWith("!remember ")) {
        const content = input.slice(10);
        const entry = await agent.remember(content);
        response = `Stored to vault [${entry.category}]: ${entry.title}`;
      } else if (input === "!trades") {
        const trades = await agent.vaultRef.getTradeHistory(undefined, 5);
        response =
          trades.length > 0
            ? trades.map((t) => `• ${t.title}`).join("\n")
            : "No trade history yet.";
      } else if (input.startsWith("!research ")) {
        const mint = input.slice(10).trim();
        console.log(chalk.dim("Researching (with reasoning)..."));
        response = await agent.researchToken(mint);
      } else if (input === "!checkpoint") {
        await agent.checkpoint();
        response = "Checkpoint saved.";
      } else if (input === "!tokens") {
        const usage = agent.tokenUsage;
        response = usage
          ? `Tokens used — Input: ${usage.inputTokens}, Output: ${usage.outputTokens}, Total: ${usage.totalTokens}`
          : "No tokens tracked yet.";
      } else {
        response = await agent.chat(input);
      }

      console.log(chalk.white(response));
    } catch (e) {
      console.log(chalk.red(`Error: ${(e as Error).message}`));
    }

    rl.prompt();
  });

  rl.on("close", async () => {
    await agent.sleep();
  });
}

async function runRecall(): Promise<void> {
  const query = process.argv.slice(3).join(" ");
  if (!query) {
    console.log("Usage: npm start recall <query>");
    process.exit(1);
  }

  const cfg = getConfig();
  const agent = new ClawdBot({ ...cfg });
  await agent.vaultRef.init();

  const memories = await agent.recall(query);
  console.log(chalk.cyan(BANNER));
  console.log(chalk.green(`\n🔍 Recall: "${query}"\n`));

  if (memories.length === 0) {
    console.log("No memories found.");
    return;
  }

  for (const m of memories) {
    console.log(chalk.yellow(`\n## [${m.category}] ${m.title}`));
    console.log(
      chalk.dim(`Score: ${m.score.toFixed(2)} | Tags: ${m.tags.join(", ")}`)
    );
    console.log(m.content.slice(0, 500));
    console.log(chalk.dim("─".repeat(60)));
  }
}

async function runBridge(): Promise<void> {
  console.log(chalk.cyan(BANNER));
  console.log(chalk.blue("🌉 MODE: Bridge Server (Python ↔ TypeScript)"));
  // Bridge is a standalone module — just import and run
  await import("./bridge/server.js");
}

// ── Main ──────────────────────────────────────────────────────────────

const mode = process.argv[2] ?? "chat";

const modes: Record<string, () => Promise<void>> = {
  start: runLiveAgent,
  research: runResearch,
  chat: runChat,
  recall: runRecall,
  bridge: runBridge,
};

const runner = modes[mode];
if (!runner) {
  console.error(
    `Unknown mode: ${mode}. Valid: ${Object.keys(modes).join(", ")}`
  );
  process.exit(1);
}

runner().catch((e) => {
  console.error(chalk.red("Fatal error:"), e);
  process.exit(1);
});
