import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import {
  skillMerkleLeaf,
  skillMerkleLeafPreimage,
  skillMerkleRoot,
  modelContentHash,
  modelPreimage,
  toolContentHash,
  mcpContentHash,
  uniquenessKey,
  sha256Prefixed,
  stripSha256,
} from "../src/hash.mjs";

const dir = path.dirname(fileURLToPath(import.meta.url));
const load = (name) => JSON.parse(readFileSync(path.join(dir, "../examples", name), "utf8"));

test("SkillHub merkleLeaf encoding matches published skillhub-onchain leaf", () => {
  const skill = load("skillhub-onchain.json").payload;
  assert.equal(skillMerkleLeaf(skill.slug, skill.bundleHash), skill.merkleLeaf);
  const pre = skillMerkleLeafPreimage(skill.slug, skill.bundleHash);
  assert.equal(pre[skill.slug.length], 0);
  assert.equal(sha256Prefixed(pre), skill.merkleLeaf);
});

test("SkillHub merkle root of a single leaf is the leaf itself", () => {
  const leaf = load("skillhub-onchain.json").payload.merkleLeaf;
  assert.equal(skillMerkleRoot([leaf]), leaf);
});

test("model content hashes are stable and differ for the two hauhau repos", () => {
  const a = load("hauhau-qwen36-onchain.json").payload;
  const b = load("hauhau-qwen36-uncensored.json").payload;
  const ha = modelContentHash(a);
  const hb = modelContentHash(b);
  assert.match(ha, /^sha256-[0-9a-f]{64}$/);
  assert.match(hb, /^sha256-[0-9a-f]{64}$/);
  assert.notEqual(ha, hb);
  assert.equal(modelContentHash(a), ha);
  // Sorting files must not change the hash.
  const shuffled = { ...a, files: [...a.files].reverse() };
  assert.equal(modelContentHash(shuffled), ha);
  assert.ok(modelPreimage(a).length > 32);
});

test("uniqueness key is type + hash + version", () => {
  const key = uniquenessKey("model", "ab".repeat(32), "v1");
  assert.equal(key, `model:${"ab".repeat(32)}:v1`);
  assert.throws(() => uniquenessKey("widget", "ab".repeat(32), "v1"));
  assert.throws(() => stripSha256("not-a-hash"));
});

test("tool and mcp hashes bind the SkillHub Google registry manifests", () => {
  const tool = load("cheshire-mcp-tools.json").payload;
  const mcp = load("cheshire-mcp-server.json").payload;
  assert.match(toolContentHash(tool), /^sha256-[0-9a-f]{64}$/);
  assert.match(mcpContentHash(mcp), /^sha256-[0-9a-f]{64}$/);
  assert.notEqual(toolContentHash(tool), mcpContentHash(mcp));
});
