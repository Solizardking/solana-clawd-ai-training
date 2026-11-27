import {
  ENTRY_TYPES,
  ZK_PROGRAM_ID,
  ZK_STATUS_NOTE,
  contentHashFor,
  skillMerkleLeaf,
  stripSha256,
  uniquenessKey,
} from "./hash.mjs";
import nacl from "tweetnacl";
import bs58 from "bs58";

export { ENTRY_TYPES, ZK_PROGRAM_ID, ZK_STATUS_NOTE, uniquenessKey };

export function zkPendingAttestation() {
  return {
    kind: "authority-signature",
    zk: "pending",
    zkProgramId: ZK_PROGRAM_ID,
    vkSlot: null,
    vkSlotsEmpty: true,
    programDeployed: false,
    note: ZK_STATUS_NOTE,
  };
}

/**
 * Build a registry record. `contentHash` must match the canonical hash of `payload`.
 * `authority` is a base58 pubkey. Optional `signature` is an ed25519 sig over the binding message.
 */
export function buildRecord({ type, version, authority, payload, cluster = "devnet", asset = null, collection = null }) {
  if (!ENTRY_TYPES.includes(type)) throw new Error(`unknown entry type: ${type}`);
  if (cluster !== "devnet") throw new Error("this registry client only builds devnet records");
  if (!version || typeof version !== "string") throw new Error("version required");
  if (!authority) throw new Error("authority required");

  const contentHash = contentHashFor(type, payload);
  let skillLeaf = null;
  if (type === "skill") {
    skillLeaf = skillMerkleLeaf(payload.slug, payload.bundleHash);
    if (payload.merkleLeaf && payload.merkleLeaf !== skillLeaf) {
      throw new Error("skill merkleLeaf does not match SkillHub encoding of slug || NUL || bundleHash");
    }
  }

  const attestation = zkPendingAttestation();
  const record = {
    schema: "onchainai-artifact-registry/v1",
    type,
    version,
    cluster,
    authority,
    contentHash,
    uniquenessKey: uniquenessKey(type, contentHash, version),
    payload,
    skillMerkleLeaf: skillLeaf,
    asset,
    collection,
    identity: {
      standard: type === "agent" ? "metaplex-agent-identity-v1" : "metaplex-core-asset+registry-record",
      collection,
      asset,
    },
    attestation,
  };
  return record;
}

export function bindingMessage(record) {
  return [
    "onchainai.fun artifact-registry wants you to attest a registry record.",
    "This signature only proves the registering authority bound this content hash.",
    "It does not authorize any payment. ZK attestation is pending.",
    "",
    `Type: ${record.type}`,
    `Content-Hash: ${record.contentHash}`,
    `Version: ${record.version}`,
    `Cluster: ${record.cluster}`,
    `Authority: ${record.authority}`,
    `Uniqueness-Key: ${record.uniquenessKey}`,
  ].join("\n");
}

export function signRecord(record, secretKey) {
  const message = bindingMessage(record);
  const signature = nacl.sign.detached(new TextEncoder().encode(message), secretKey);
  return {
    ...record,
    attestation: {
      ...record.attestation,
      message,
      signature: bs58.encode(signature),
    },
  };
}

export function verifyRecordSignature(record) {
  const message = record.attestation?.message || bindingMessage(record);
  if (message !== bindingMessage(record)) throw new Error("attestation message does not bind the record fields");
  const signature = record.attestation?.signature;
  if (!signature) throw new Error("missing authority signature");
  const ok = nacl.sign.detached.verify(
    new TextEncoder().encode(message),
    bs58.decode(signature),
    bs58.decode(record.authority),
  );
  if (!ok) throw new Error("invalid authority signature");
  return true;
}

/** In-memory registry used by tests / Lean parity: reject hash mismatch, duplicate keys, wrong authority updates. */
export function createMemoryRegistry() {
  const entries = new Map();
  return {
    entries,
    register(record, { expectedAuthority = null } = {}) {
      stripSha256(record.contentHash);
      const computed = contentHashFor(record.type, record.payload);
      if (computed !== record.contentHash) throw new Error("content hash does not bind payload");
      if (expectedAuthority && record.authority !== expectedAuthority) throw new Error("authority mismatch");
      if (entries.has(record.uniquenessKey)) throw new Error("duplicate registration");
      entries.set(record.uniquenessKey, record);
      return record;
    },
    get(type, contentHash, version) {
      return entries.get(uniquenessKey(type, contentHash, version)) || null;
    },
  };
}

export function metaplexAttributes(record) {
  const attrs = [
    { key: "schema", value: record.schema },
    { key: "type", value: record.type },
    { key: "contentHash", value: record.contentHash },
    { key: "version", value: record.version },
    { key: "cluster", value: record.cluster },
    { key: "authority", value: record.authority },
    { key: "zk", value: record.attestation.zk },
    { key: "zkProgramId", value: record.attestation.zkProgramId },
    { key: "vkSlotsEmpty", value: "true" },
  ];
  if (record.type === "skill") {
    attrs.push({ key: "slug", value: record.payload.slug });
    attrs.push({ key: "bundleHash", value: record.payload.bundleHash });
    attrs.push({ key: "merkleLeaf", value: record.skillMerkleLeaf });
  }
  if (record.type === "model") {
    attrs.push({ key: "hfRepo", value: record.payload.hfRepo });
    attrs.push({ key: "revisionSha", value: record.payload.revisionSha });
    if (record.payload.license) attrs.push({ key: "license", value: record.payload.license });
  }
  return attrs;
}
