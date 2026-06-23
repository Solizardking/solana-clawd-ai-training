/**
 * Clawd Model Registration
 *
 * Registers a Clawd model into the solana_ai_inference Anchor program
 * (program ID: 3dLst2E3djtCSwG19mFS3REHxtZPngjyga7iYZLDL5xj) on devnet/mainnet.
 *
 * This is the onchain "initialize_model" instruction — it creates a ModelRegistry
 * PDA seeded by ["model", authority] that anchors the model's HF hash, API endpoint,
 * and reward rate to your wallet permanently.
 *
 * Usage:
 *   pnpm tsx dao/register_model.ts \
 *     --model-hash "sha256:abc123..." \
 *     --endpoint "https://clawd-box-router.fly.dev/v1" \
 *     --keypair ~/.config/solana/id.json
 *
 * Or via the one-shot shell wrapper: dao/register_model.sh
 */

import * as web3 from "@solana/web3.js";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

// ── Constants ──────────────────────────────────────────────────────────────

const PROGRAM_ID = new web3.PublicKey("3dLst2E3djtCSwG19mFS3REHxtZPngjyga7iYZLDL5xj");
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEFAULT_IDL_PATH = path.resolve(__dirname, "../../../OnChain-Ai-main/solana-ai-inference/target/idl/solana_ai_inference.json");

// Clawd model registry at onchain.x402.wtf
const ONCHAIN_REGISTRY_API = process.env.ONCHAIN_REGISTRY_URL ?? "https://onchain.x402.wtf/api/register";

interface RegisterArgs {
  modelHash: string;       // sha256 or HF commit hash
  modelType: string;       // "TextGeneration" | "SentimentAnalysis" | ...
  apiEndpoint: string;     // ClawdRouter or HF inference endpoint
  hfModelId: string;
  baseModel: string;
  datasetSize: number;
  evalAccuracy: number;
  termRewardRate: number;  // $CLAWD lamports per validated inference (u64)
  keypairPath: string;
  idlPath: string;
  rpcUrl?: string;
  cluster: "devnet" | "mainnet-beta";
  dryRun: boolean;
}

// ── Program type (inline — matches IDL) ───────────────────────────────────

const MODEL_TYPE_MAP: Record<string, object> = {
  TextGeneration:         { textGeneration: {} },
  SentimentAnalysis:      { sentimentAnalysis: {} },
  ImageClassification:    { imageClassification: {} },
  PricePrediction:        { pricePrediction: {} },
  DocumentUnderstanding:  { documentUnderstanding: {} },
};

function loadAuthority(keypairPath: string, dryRun: boolean): web3.Keypair {
  if (fs.existsSync(keypairPath)) {
    const keypairJson = JSON.parse(fs.readFileSync(keypairPath, "utf-8"));
    return web3.Keypair.fromSecretKey(Uint8Array.from(keypairJson));
  }
  if (dryRun) {
    const authority = web3.Keypair.generate();
    console.warn(`[dry-run] keypair not found at ${keypairPath}; using ephemeral authority ${authority.publicKey.toBase58()}`);
    return authority;
  }
  throw new Error(`Keypair not found: ${keypairPath}`);
}

function rpcUrlFor(args: RegisterArgs): string {
  if (args.rpcUrl) return args.rpcUrl;
  return args.cluster === "mainnet-beta"
    ? "https://api.mainnet-beta.solana.com"
    : "https://api.devnet.solana.com";
}

function registrationPayload(args: RegisterArgs, authority: web3.PublicKey, modelRegistryPDA: web3.PublicKey, txSig?: string) {
  return {
    model_hash: args.modelHash,
    model_type: args.modelType,
    api_endpoint: args.apiEndpoint,
    authority: authority.toBase58(),
    pda: modelRegistryPDA.toBase58(),
    tx_sig: txSig ?? "",
    cluster: args.cluster,
    protocol: "CAAP/1.0",
    hf_model_id: args.hfModelId,
    base_model: args.baseModel,
    dataset_size: args.datasetSize,
    eval_accuracy: args.evalAccuracy,
  };
}

async function registerModel(args: RegisterArgs): Promise<string> {
  const authority = loadAuthority(args.keypairPath, args.dryRun);

  // Provider
  const rpcUrl = rpcUrlFor(args);
  const connection = new web3.Connection(rpcUrl, "confirmed");

  // Derive model registry PDA: seeds = ["model", authority.pubkey]
  const [modelRegistryPDA] = web3.PublicKey.findProgramAddressSync(
    [Buffer.from("model"), authority.publicKey.toBuffer()],
    PROGRAM_ID
  );

  const modelType = MODEL_TYPE_MAP[args.modelType];
  if (!modelType) {
    throw new Error(`Unknown model type: ${args.modelType}. Valid: ${Object.keys(MODEL_TYPE_MAP).join(", ")}`);
  }

  console.log(`\nRegistering model on ${args.cluster}:`);
  console.log(`  Authority:   ${authority.publicKey.toBase58()}`);
  console.log(`  Registry PDA: ${modelRegistryPDA.toBase58()}`);
  console.log(`  Model hash:  ${args.modelHash}`);
  console.log(`  HF model:    ${args.hfModelId}`);
  console.log(`  Base model:  ${args.baseModel}`);
  console.log(`  Type:        ${args.modelType}`);
  console.log(`  Endpoint:    ${args.apiEndpoint}`);
  console.log(`  RPC:         ${rpcUrl}`);

  if (args.dryRun) {
    console.log("\n[DRY RUN] Transaction not submitted.");
    console.log(JSON.stringify(registrationPayload(args, authority.publicKey, modelRegistryPDA), null, 2));
    return modelRegistryPDA.toBase58();
  }

  if (!fs.existsSync(args.idlPath)) {
    throw new Error(`IDL not found: ${args.idlPath}. Build the Anchor program or pass --idl.`);
  }

  let anchor: any;
  try {
    anchor = await import("@coral-xyz/anchor");
  } catch {
    throw new Error("@coral-xyz/anchor is required for live onchain registration. Install project dependencies before running without --dry-run.");
  }

  const wallet = new anchor.Wallet(authority);
  const provider = new anchor.AnchorProvider(connection, wallet, { commitment: "confirmed" });

  // Load IDL only after dry-run exits so previews work on machines without Anchor artifacts.
  const idl = JSON.parse(fs.readFileSync(args.idlPath, "utf-8"));
  const program = new anchor.Program(idl, PROGRAM_ID, provider);

  // Submit initialize_model instruction
  const txSig = await (program.methods as any)
    .initializeModel(
      args.modelHash,
      modelType,
      args.apiEndpoint,
      new anchor.BN(args.termRewardRate)
    )
    .accounts({
      modelRegistry: modelRegistryPDA,
      authority: authority.publicKey,
      systemProgram: web3.SystemProgram.programId,
    })
    .signers([authority])
    .rpc();

  console.log(`\nTransaction confirmed: ${txSig}`);
  console.log(`Explorer: https://solscan.io/tx/${txSig}?cluster=${args.cluster}`);
  console.log(`Registry PDA: ${modelRegistryPDA.toBase58()}`);

  // Also register with the onchain.x402.wtf off-chain index
  try {
    const regResp = await fetch(ONCHAIN_REGISTRY_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(registrationPayload(args, authority.publicKey, modelRegistryPDA, txSig)),
    });
    if (regResp.ok) {
      const body = await regResp.json();
      console.log(`\nOnchain registry updated: ${JSON.stringify(body)}`);
    }
  } catch (e) {
    console.warn(`[registry warn] Could not reach ${ONCHAIN_REGISTRY_API}: ${e}`);
  }

  return txSig;
}

// ── CLI ────────────────────────────────────────────────────────────────────

function parseArgs(): RegisterArgs {
  const argv = process.argv.slice(2);
  const get = (flag: string, def?: string): string => {
    const idx = argv.indexOf(flag);
    if (idx !== -1 && argv[idx + 1]) return argv[idx + 1];
    if (def !== undefined) return def;
    throw new Error(`Missing required flag: ${flag}`);
  };
  const parsed = {
    modelHash:       get("--model-hash", "sha256:pending"),
    modelType:       get("--model-type", "TextGeneration"),
    apiEndpoint:     get("--endpoint",   "https://clawd-box-router.fly.dev/v1"),
    hfModelId:       get("--hf-model", process.env.HF_MODEL_ID ?? "solanaclawd/solana-tx-foundation-7b"),
    baseModel:       get("--base-model", process.env.BASE_MODEL ?? "Qwen/Qwen2.5-7B-Instruct"),
    datasetSize:     parseInt(get("--dataset-size", process.env.DATASET_SIZE ?? "82169")),
    evalAccuracy:    parseFloat(get("--eval-accuracy", process.env.EVAL_ACCURACY ?? "0.00")),
    termRewardRate:  parseInt(get("--reward-rate", "1000000")),
    keypairPath:     get("--keypair",    process.env.HOME + "/.config/solana/id.json"),
    idlPath:         get("--idl", DEFAULT_IDL_PATH),
    rpcUrl:          get("--rpc-url", ""),
    cluster:         (get("--cluster", "devnet") as "devnet" | "mainnet-beta"),
    dryRun:          argv.includes("--dry-run"),
  };
  if (!Number.isFinite(parsed.datasetSize)) {
    throw new Error(`Invalid --dataset-size: ${parsed.datasetSize}`);
  }
  if (!Number.isFinite(parsed.evalAccuracy)) {
    throw new Error(`Invalid --eval-accuracy: ${parsed.evalAccuracy}`);
  }
  if (!["devnet", "mainnet-beta"].includes(parsed.cluster)) {
    throw new Error(`Invalid --cluster: ${parsed.cluster}. Use devnet or mainnet-beta.`);
  }
  return parsed;
}

(async () => {
  try {
    const args = parseArgs();
    await registerModel(args);
  } catch (err) {
    console.error("Error:", err);
    process.exit(1);
  }
})();
