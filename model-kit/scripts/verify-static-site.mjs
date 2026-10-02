import { access, readFile } from "node:fs/promises";
import path from "node:path";

const root = process.cwd();
const required = [
  "frontend/index.html",
  "frontend/register.html",
  "frontend/app.js",
  "frontend/styles.css",
  "frontend/config.js",
  "frontend/favicon.svg",
  "frontend/assets/solana-ai-model-kit.svg",
];

for (const file of required) {
  await access(path.join(root, file));
}

const app = await readFile(path.join(root, "frontend/app.js"), "utf8");
for (const token of ["api/model-kit/status", "api/register/preview", "api/register", "api/arena/providers", "api/arena/runs", "constitution --strict", ".fly.dev"]) {
  if (!app.includes(token)) {
    throw new Error(`frontend/app.js is missing ${token}`);
  }
}

const config = await readFile(path.join(root, "frontend/config.js"), "utf8");
if (!config.includes("solana-clawd-model-kit.fly.dev")) {
  throw new Error("frontend/config.js is missing the Fly Machines API origin");
}

const index = await readFile(path.join(root, "frontend/index.html"), "utf8");
for (const token of ["data-status-field=\"constitutionGate\"", "id=\"constitutionFiles\""]) {
  if (!index.includes(token)) {
    throw new Error(`frontend/index.html is missing ${token}`);
  }
}

console.log("model-kit static site verified");
