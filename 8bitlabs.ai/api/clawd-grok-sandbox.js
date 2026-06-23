const DEFAULT_REPO = "https://github.com/solizardking/solana-clawd.git";
const DEFAULT_BRANCH = "main";
const DEFAULT_REMOTE_DIR = "/home/user/solana-clawd";
const DEFAULT_PACKAGE_DIR = "core-ai/clawd-grok";
const DEFAULT_TIMEOUT_MS = 10 * 60 * 1000;
const DEFAULT_COMMAND_TIMEOUT_MS = 4 * 60 * 1000;
const CLAWD_TOKEN_MINT = "8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump";
const WALLET_PROOF_INTENT = "launch clawd-grok sandbox computer";
const WALLET_PROOF_MAX_AGE_MS = 10 * 60 * 1000;

const bs58Module = require("bs58");
const bs58 = bs58Module.default || bs58Module;
const nacl = require("tweetnacl");

const providerEnvKeys = [
  "XAI_API_KEY",
  "GROK_API_KEY",
  "AI_API_KEY",
  "OPENROUTER_API_KEY",
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
];

const solanaReadEnvKeys = [
  "HELIUS_API_KEY",
  "SOLANA_RPC_URL",
  "SOLANA_TRACKER_API_KEY",
  "SOLANA_TRACKER_RPC_URL",
  "PHOENIX_API_KEY",
  "PHOENIX_API_URL",
];

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", process.env.CLAWD_GROK_SANDBOX_ALLOWED_ORIGIN || "https://8bitlabs.ai");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type,Authorization,X-Clawd-Sandbox-Token,X-Gateway-API-Key,X-Clawd-Wallet");
}

async function readJsonBody(req) {
  if (req.body && typeof req.body === "object") return req.body;
  if (typeof req.body === "string" && req.body.length) return JSON.parse(req.body);

  let raw = "";
  for await (const chunk of req) {
    raw += chunk;
    if (raw.length > 32_000) {
      const error = new Error("Request body too large");
      error.statusCode = 413;
      throw error;
    }
  }
  return raw.trim() ? JSON.parse(raw) : {};
}

function json(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

function boolEnv(name) {
  return process.env[name] === "true";
}

function shQuote(value) {
  return `'${String(value).replaceAll("'", "'\"'\"'")}'`;
}

function joinRemotePath(root, child) {
  if (child.startsWith("/")) return child;
  return `${root.replace(/\/+$/, "")}/${child.replace(/^\/+/, "")}`;
}

function pickEnv(keys) {
  return Object.fromEntries(keys.filter((key) => process.env[key]).map((key) => [key, process.env[key]]));
}

function bearerToken(req) {
  const header = String(req.headers.authorization || "");
  return header.toLowerCase().startsWith("bearer ") ? header.slice(7).trim() : "";
}

function requestToken(req) {
  return (
    String(req.headers["x-clawd-sandbox-token"] || "") ||
    String(req.headers["x-gateway-api-key"] || "") ||
    bearerToken(req)
  );
}

function requiredToken() {
  return process.env.CLAWD_GROK_SANDBOX_TOKEN || process.env.GATEWAY_ADMIN_KEY || "";
}

function hasOperatorToken(req) {
  const token = requiredToken();
  if (!token) return false;
  return requestToken(req) === token;
}

function boundedText(value, max) {
  const text = String(value || "");
  return text.length > max ? `${text.slice(0, max)}\n[truncated]` : text;
}

function minClawdBalance() {
  const parsed = Number.parseFloat(process.env.CLAWD_GROK_MIN_CLAWD_BALANCE || process.env.CLAWD_HOLDER_THRESHOLD || "1000");
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 1000;
}

function tierFromBalance(balance) {
  if (balance >= 100_000) return "ABYSS";
  if (balance >= 10_000) return "DEEP";
  if (balance >= 1_000) return "SHALLOW";
  if (balance >= 1) return "SHORELINE";
  return "BEACHED";
}

function parseProofFields(message) {
  const fields = {};
  for (const line of String(message || "").split("\n")) {
    const index = line.indexOf(":");
    if (index === -1) continue;
    fields[line.slice(0, index).trim().toLowerCase()] = line.slice(index + 1).trim();
  }
  return fields;
}

function decodeSignature(proof) {
  const value = String(proof.signature || "");
  if (!value) throw new Error("Missing wallet signature");
  if (proof.encoding === "base58") return bs58.decode(value);
  return Buffer.from(value, "base64");
}

function verifyWalletProof(proof) {
  if (!proof || typeof proof !== "object") throw new Error("Wallet proof required");

  const wallet = String(proof.wallet || "").trim();
  const message = String(proof.message || "");
  if (!wallet || !message) throw new Error("Wallet proof is incomplete");

  const fields = parseProofFields(message);
  if (fields.wallet !== wallet) throw new Error("Wallet proof address mismatch");
  if (fields.intent !== WALLET_PROOF_INTENT) throw new Error("Wallet proof intent mismatch");

  const issuedAt = Date.parse(fields["issued at"] || "");
  if (!Number.isFinite(issuedAt)) throw new Error("Wallet proof is missing a valid issued-at timestamp");
  if (Math.abs(Date.now() - issuedAt) > WALLET_PROOF_MAX_AGE_MS) throw new Error("Wallet proof expired");

  const publicKey = bs58.decode(wallet);
  if (publicKey.length !== 32) throw new Error("Invalid Solana wallet address");
  const signature = decodeSignature(proof);
  if (signature.length !== 64) throw new Error("Invalid wallet signature length");

  const messageBytes = new TextEncoder().encode(message);
  const valid = nacl.sign.detached.verify(messageBytes, signature, publicKey);
  if (!valid) throw new Error("Wallet signature verification failed");

  return wallet;
}

async function getClawdBalance(walletAddress) {
  const heliusApiKey = process.env.HELIUS_API_KEY || "";
  if (heliusApiKey) {
    try {
      return await getClawdBalanceViaRpc(walletAddress, `https://mainnet.helius-rpc.com/?api-key=${heliusApiKey}`);
    } catch {
      // Fall through to a standard RPC.
    }
  }

  const rpcUrl =
    process.env.HELIUS_RPC_URL ||
    process.env.SOLANA_RPC_URL ||
    process.env.SOLANA_TRACKER_RPC_URL ||
    "https://api.mainnet-beta.solana.com";
  return getClawdBalanceViaRpc(walletAddress, rpcUrl);
}

async function getClawdBalanceViaRpc(walletAddress, rpcUrl) {
  const response = await fetch(rpcUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: "8bitlabs-clawd-grok-gate",
      method: "getTokenAccountsByOwner",
      params: [walletAddress, { mint: CLAWD_TOKEN_MINT }, { encoding: "jsonParsed" }],
    }),
  });

  if (!response.ok) throw new Error(`Solana RPC holder check failed: ${response.status}`);
  const data = await response.json();
  if (data.error) throw new Error(data.error.message || "Solana RPC holder check failed");
  return sumClawdAccounts(data.result?.value || []);
}

function sumClawdAccounts(accounts) {
  let total = 0;
  for (const account of accounts) {
    const info = account.account?.data?.parsed?.info;
    if (info?.mint === CLAWD_TOKEN_MINT) {
      total += Number.parseFloat(info.tokenAmount?.uiAmountString || "0");
    }
  }
  return total;
}

async function requireClawdHolderAccess(body) {
  const wallet = verifyWalletProof(body.walletProof);
  const balance = await getClawdBalance(wallet);
  const required = minClawdBalance();
  const tier = tierFromBalance(balance);

  if (balance < required) {
    const error = new Error("CLAWD holder gate required");
    error.statusCode = 402;
    error.details = {
      code: "clawd_holder_required",
      wallet,
      clawdBalance: balance,
      tier,
      requiredBalance: required,
      requiredTier: "SHALLOW",
      tokenMint: CLAWD_TOKEN_MINT,
    };
    throw error;
  }

  return { wallet, clawdBalance: balance, tier, requiredBalance: required, tokenMint: CLAWD_TOKEN_MINT };
}

function buildCommands(options = {}) {
  const repo = process.env.CLAWD_GROK_E2B_REPO || process.env.CLAWD_E2B_REPO || DEFAULT_REPO;
  const branch = process.env.CLAWD_GROK_E2B_BRANCH || process.env.CLAWD_E2B_BRANCH || DEFAULT_BRANCH;
  const remoteDir = process.env.CLAWD_GROK_E2B_REMOTE_DIR || process.env.CLAWD_E2B_REMOTE_DIR || DEFAULT_REMOTE_DIR;
  const packageDir = process.env.CLAWD_GROK_E2B_PACKAGE_DIR || DEFAULT_PACKAGE_DIR;
  const packagePath = joinRemotePath(remoteDir, packageDir);

  const clone = ["git", "clone", "--depth", "1", "--branch", shQuote(branch), shQuote(repo), shQuote(remoteDir)].join(" ");
  const install = [
    `cd ${shQuote(packagePath)}`,
    "if command -v bun >/dev/null 2>&1; then bun install --frozen-lockfile; else npm install --include=dev --ignore-scripts; fi",
  ].join(" && ");
  const build = [
    `cd ${shQuote(packagePath)}`,
    "if command -v bun >/dev/null 2>&1; then bun run build; else npm run build; fi",
  ].join(" && ");
  const clawd = (args) =>
    [
      `cd ${shQuote(packagePath)}`,
      `if command -v bun >/dev/null 2>&1; then bun run dist/index.js ${args}; else node dist/index.js ${args}; fi`,
    ].join(" && ");

  const prompt = String(options.prompt || "").trim().slice(0, 1_000);
  const smoke =
    options.command ||
    (prompt ? clawd(`-p ${shQuote(prompt)} --format text`) : clawd("status --output json"));

  return {
    repo,
    branch,
    remoteDir,
    packageDir,
    packagePath,
    plan: [
      ["create sandbox", `timeoutMs=${options.timeoutMs || DEFAULT_TIMEOUT_MS}`],
      ["interpreter smoke", "runCode('print(\"clawd-grok sandbox ready\")')"],
      ["clone", clone],
      ...(!options.skipInstall ? [["install", install]] : []),
      ...(!options.skipInstall && !options.skipBuild ? [["build", build]] : []),
      ["smoke", smoke],
    ],
    clone,
    install,
    build,
    smoke,
    prompt,
  };
}

function statusPayload() {
  return {
    ok: true,
    id: "clawd-grok-sandbox-computer",
    manifest: "/clawd-grok.json",
    enabled: boolEnv("CLAWD_GROK_SANDBOX_ENABLED"),
    hasE2BKey: Boolean(process.env.E2B_API_KEY),
    requiresToken: Boolean(requiredToken()),
    providerKeysForwarded: boolEnv("CLAWD_GROK_FORWARD_PROVIDER_KEYS"),
    solanaReadKeysForwarded: boolEnv("CLAWD_GROK_FORWARD_SOLANA_READ_KEYS"),
    customCommandsEnabled: boolEnv("CLAWD_GROK_ALLOW_CUSTOM_COMMANDS"),
    keepAliveEnabled: boolEnv("CLAWD_GROK_ALLOW_KEEP"),
    holderGate: {
      required: true,
      token: "$CLAWD",
      tokenMint: CLAWD_TOKEN_MINT,
      minBalance: minClawdBalance(),
      minTier: "SHALLOW",
      signedWalletProof: true,
    },
    plan: buildCommands().plan,
  };
}

async function runCommand(sandbox, label, command, commandTimeoutMs) {
  let stdout = "";
  let stderr = "";
  const appendStdout = (chunk) => {
    stdout = boundedText(stdout + chunk, 12_000);
  };
  const appendStderr = (chunk) => {
    stderr = boundedText(stderr + chunk, 8_000);
  };

  try {
    const result = await sandbox.commands.run(command, {
      timeoutMs: commandTimeoutMs,
      onStdout: appendStdout,
      onStderr: appendStderr,
    });
    return { label, exitCode: result.exitCode, stdout, stderr };
  } catch (error) {
    return {
      label,
      exitCode: Number.isInteger(error?.exitCode) ? error.exitCode : 1,
      stdout: boundedText(stdout || error?.stdout || "", 12_000),
      stderr: boundedText(stderr || error?.stderr || error?.message || String(error), 8_000),
      failed: true,
    };
  }
}

module.exports = async function handler(req, res) {
  setCors(res);

  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    res.end();
    return;
  }

  if (req.method === "GET") {
    json(res, 200, statusPayload());
    return;
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "GET,POST,OPTIONS");
    json(res, 405, { error: "Method not allowed" });
    return;
  }

  let body;
  try {
    body = await readJsonBody(req);
  } catch (error) {
    json(res, error.statusCode || 400, { error: error.message || "Invalid JSON body" });
    return;
  }

  const wantsDryRun = body.dryRun === true;
  const wantsPrompt = String(body.prompt || "").trim();
  const wantsCustomCommand = typeof body.command === "string" && body.command.trim().length > 0;
  const hasToken = hasOperatorToken(req);
  const allowCustomCommand = boolEnv("CLAWD_GROK_ALLOW_CUSTOM_COMMANDS") && hasToken;
  const timeoutMs = Math.min(Number.parseInt(body.timeoutMs, 10) || DEFAULT_TIMEOUT_MS, 30 * 60 * 1000);
  const commandTimeoutMs = Math.min(
    Number.parseInt(body.commandTimeoutMs, 10) || DEFAULT_COMMAND_TIMEOUT_MS,
    12 * 60 * 1000,
  );
  const keep = body.keep === true && boolEnv("CLAWD_GROK_ALLOW_KEEP") && hasToken;

  if (wantsCustomCommand && !allowCustomCommand) {
    json(res, 403, { error: "Custom sandbox commands are disabled for this deployment" });
    return;
  }

  if (wantsPrompt && !boolEnv("CLAWD_GROK_FORWARD_PROVIDER_KEYS")) {
    json(res, 400, { error: "Prompt mode requires CLAWD_GROK_FORWARD_PROVIDER_KEYS=true on the server" });
    return;
  }

  const commands = buildCommands({
    prompt: wantsPrompt,
    command: wantsCustomCommand ? body.command.trim() : undefined,
    timeoutMs,
    skipInstall: body.skipInstall === true,
    skipBuild: body.skipBuild === true,
  });

  if (wantsDryRun) {
    json(res, 200, { ok: true, dryRun: true, plan: commands.plan });
    return;
  }

  if (!boolEnv("CLAWD_GROK_SANDBOX_ENABLED")) {
    json(res, 403, { error: "Clawd Grok sandbox launch is disabled", status: statusPayload() });
    return;
  }

  if (!hasToken) {
    json(res, 401, { error: "Sandbox launch token required" });
    return;
  }

  if (!process.env.E2B_API_KEY) {
    json(res, 503, { error: "E2B_API_KEY is not configured on the server" });
    return;
  }

  let access;
  try {
    access = await requireClawdHolderAccess(body);
  } catch (error) {
    json(res, error.statusCode || 401, {
      error: error.message || "CLAWD holder gate failed",
      ...(error.details ? error.details : {}),
    });
    return;
  }

  const { Sandbox } = await import("@e2b/code-interpreter");
  const envs = {
    CI: "1",
    NO_COLOR: "1",
    CLAWD_E2B_SANDBOX: "true",
    CLAWD_GROK_E2B_SANDBOX: "true",
    ...(boolEnv("CLAWD_GROK_FORWARD_PROVIDER_KEYS") ? pickEnv(providerEnvKeys) : {}),
    ...(boolEnv("CLAWD_GROK_FORWARD_SOLANA_READ_KEYS") ? pickEnv(solanaReadEnvKeys) : {}),
  };

  let sandbox;
  const logs = [];
  try {
    sandbox = await Sandbox.create({
      timeoutMs,
      metadata: {
        project: "8bitlabs.ai",
        runner: "clawd-grok",
        source: "api/clawd-grok-sandbox.js",
        repo: commands.repo,
        branch: commands.branch,
        wallet: access.wallet,
        clawdTier: access.tier,
      },
      envs,
    });

    const execution = await sandbox.runCode('print("clawd-grok sandbox ready")', { timeoutMs: 30_000 });
    logs.push({
      label: "interpreter smoke",
      stdout: execution.logs.stdout.join("\n"),
      stderr: execution.logs.stderr.join("\n"),
      exitCode: 0,
    });

    logs.push(await runCommand(sandbox, "clone repo", commands.clone, commandTimeoutMs));
    if (body.skipInstall !== true) logs.push(await runCommand(sandbox, "install dependencies", commands.install, commandTimeoutMs));
    if (body.skipInstall !== true && body.skipBuild !== true)
      logs.push(await runCommand(sandbox, "build clawd grok", commands.build, commandTimeoutMs));
    logs.push(await runCommand(sandbox, wantsPrompt ? "run clawd grok prompt" : "run clawd grok smoke", commands.smoke, commandTimeoutMs));

    const failed = logs.find((entry) => entry.failed || entry.exitCode !== 0);
    json(res, failed ? 500 : 200, {
      ok: !failed,
      sandboxId: sandbox.sandboxId,
      keptAlive: keep,
      access,
      plan: commands.plan,
      logs,
    });
  } catch (error) {
    json(res, 500, { error: error.message || "Clawd Grok sandbox failed", logs });
  } finally {
    if (sandbox && !keep) {
      await sandbox.kill().catch(() => {});
    }
  }
};
