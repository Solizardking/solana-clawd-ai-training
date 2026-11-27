import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { skillMerkleLeafPreimage, modelPreimage, toolPreimage, mcpPreimage } from "../src/hash.mjs";

const dir = path.dirname(fileURLToPath(import.meta.url));
const vectorsPath = path.join(dir, "../formal/vectors/preimages.json");
const load = (name) => JSON.parse(readFileSync(path.join(dir, "../examples", name), "utf8"));

test("Lean-generated preimages match the TypeScript canonical encodings", () => {
  if (!existsSync(vectorsPath)) {
    throw new Error("formal/vectors/preimages.json missing — run `npm run vectors` (lake exe vectors) first");
  }
  const vectors = JSON.parse(readFileSync(vectorsPath, "utf8"));
  const byName = Object.fromEntries(vectors.map((v) => [v.name, v]));

  const skill = load("skillhub-onchain.json").payload;
  const skillHex = skillMerkleLeafPreimage(skill.slug, skill.bundleHash).toString("hex");
  assert.equal(byName.skillhub_onchain_leaf_preimage.preimageHex, skillHex);

  const model = load("hauhau-qwen36-onchain.json").payload;
  assert.equal(byName.hauhau_onchain_model_preimage.preimageHex, modelPreimage(model).toString("hex"));

  const tool = load("cheshire-mcp-tools.json").payload;
  assert.equal(byName.cheshire_tools_preimage.preimageHex, toolPreimage(tool).toString("hex"));

  const mcp = load("cheshire-mcp-server.json").payload;
  assert.equal(byName.cheshire_mcp_preimage.preimageHex, mcpPreimage(mcp).toString("hex"));
});
