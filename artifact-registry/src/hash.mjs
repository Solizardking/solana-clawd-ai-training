import { createHash } from "node:crypto";

export const ENTRY_TYPES = Object.freeze(["agent", "model", "skill", "tool", "mcp_server"]);
export const ZK_PROGRAM_ID = "4vJ9JU1bJJE96FWSVKmnrL3xFU5jSBSVdk9x4La2vzhn";
export const ZK_STATUS_NOTE =
  "zk-clawd-primitives is not live: the clawd-zk program is not deployed and all verifying-key slots are empty, so every proof is refused until circuits and a key ceremony exist.";

/** SHA-256 hex (no prefix). Accepts string or Uint8Array/Buffer. */
export function sha256Hex(value) {
  return createHash("sha256").update(value).digest("hex");
}

export function sha256Prefixed(value) {
  return `sha256-${sha256Hex(value)}`;
}

export function stripSha256(value) {
  if (typeof value !== "string") throw new Error("hash must be a string");
  const hex = value.startsWith("sha256-") ? value.slice(7) : value;
  if (!/^[0-9a-f]{64}$/.test(hex)) throw new Error(`invalid sha256: ${value}`);
  return hex;
}

/** Stable JSON: sorted object keys, no whitespace. Arrays keep order. */
export function canonicalJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const keys = Object.keys(value).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson(value[k])}`).join(",")}}`;
}

/**
 * SkillHub bundle hash (scripts/build-catalog.mjs hashBundle).
 * bundleFiles: [{ path, content: string|Buffer }] already in catalog order.
 */
export function skillBundleHash(bundleFiles) {
  const hash = createHash("sha256");
  hash.update("skill-bundle-v1\0");
  for (const file of bundleFiles) {
    hash.update(file.path);
    hash.update("\0");
    hash.update(file.content);
    hash.update("\0");
  }
  return `sha256-${hash.digest("hex")}`;
}

/**
 * SkillHub merkle leaf: sha256(`${slug}\\0${bundleHash}`) where bundleHash
 * includes the `sha256-` prefix (exact SkillHub encoding).
 */
export function skillMerkleLeaf(slug, bundleHash) {
  if (!slug) throw new Error("skill slug required");
  stripSha256(bundleHash);
  return sha256Prefixed(`${slug}\0${bundleHash}`);
}

export function skillMerkleLeafPreimage(slug, bundleHash) {
  stripSha256(bundleHash);
  return Buffer.from(`${slug}\0${bundleHash}`, "utf8");
}

/** SkillHub merkle root over `sha256-…` leaves (duplicate last on odd length). */
export function skillMerkleRoot(leaves) {
  if (leaves.length === 0) return sha256Prefixed("");
  let level = leaves.map(stripSha256);
  while (level.length > 1) {
    const next = [];
    for (let i = 0; i < level.length; i += 2) {
      const left = level[i];
      const right = level[i + 1] || left;
      next.push(sha256Hex(Buffer.concat([Buffer.from(left, "hex"), Buffer.from(right, "hex")])));
    }
    level = next;
  }
  return `sha256-${level[0]}`;
}

function preimage(kind, fields) {
  return Buffer.concat([Buffer.from(`${kind}\0`, "utf8"), Buffer.from(canonicalJson(fields), "utf8")]);
}

/** Canonical model preimage + content hash. Files must be sorted by path by the caller or we sort. */
export function modelPreimage(model) {
  const files = [...(model.files || [])]
    .map((f) => ({
      path: f.path,
      sha256: stripSha256(f.sha256),
      size: f.size ?? null,
    }))
    .sort((a, b) => a.path.localeCompare(b.path));
  const fields = {
    hfRepo: model.hfRepo,
    revisionSha: model.revisionSha,
    files,
    license: model.license ?? null,
    provenance: model.provenance ?? null,
  };
  if (!fields.hfRepo || !fields.revisionSha) throw new Error("model requires hfRepo and revisionSha");
  if (!files.length) throw new Error("model requires at least one file hash");
  return preimage("model-v1", fields);
}

export function modelContentHash(model) {
  return sha256Prefixed(modelPreimage(model));
}

export function toolPreimage(tool) {
  const fields = {
    name: tool.name,
    sourceRepo: tool.sourceRepo ?? null,
    commit: tool.commit ?? null,
    manifestHash: stripSha256(tool.manifestHash),
  };
  if (!fields.name) throw new Error("tool requires name");
  return preimage("tool-v1", fields);
}

export function toolContentHash(tool) {
  return sha256Prefixed(toolPreimage(tool));
}

export function mcpPreimage(mcp) {
  const fields = {
    name: mcp.name,
    endpoint: mcp.endpoint ?? null,
    manifestHash: stripSha256(mcp.manifestHash),
    packageVersion: mcp.packageVersion ?? null,
    sourceRepo: mcp.sourceRepo ?? null,
    commit: mcp.commit ?? null,
  };
  if (!fields.name) throw new Error("mcp_server requires name");
  return preimage("mcp_server-v1", fields);
}

export function mcpContentHash(mcp) {
  return sha256Prefixed(mcpPreimage(mcp));
}

export function agentPreimage(agent) {
  const fields = {
    name: agent.name,
    owner: agent.owner,
    asset: agent.asset ?? null,
    handle: agent.handle ?? null,
    endpoint: agent.endpoint ?? null,
  };
  if (!fields.name || !fields.owner) throw new Error("agent requires name and owner");
  return preimage("agent-v1", fields);
}

export function agentContentHash(agent) {
  return sha256Prefixed(agentPreimage(agent));
}

export function contentHashFor(type, payload) {
  switch (type) {
    case "model":
      return modelContentHash(payload);
    case "skill":
      return payload.bundleHash.startsWith("sha256-") ? payload.bundleHash : sha256Prefixed(Buffer.from(stripSha256(payload.bundleHash), "hex"));
    case "tool":
      return toolContentHash(payload);
    case "mcp_server":
      return mcpContentHash(payload);
    case "agent":
      return agentContentHash(payload);
    default:
      throw new Error(`unknown entry type: ${type}`);
  }
}

/** Uniqueness key used by the Lean spec and the TS registry. */
export function uniquenessKey(type, contentHash, version) {
  if (!ENTRY_TYPES.includes(type)) throw new Error(`unknown entry type: ${type}`);
  return `${type}:${stripSha256(contentHash)}:${version}`;
}
