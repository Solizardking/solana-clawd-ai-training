import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import nacl from "tweetnacl";
import bs58 from "bs58";
import {
  buildRecord,
  signRecord,
  verifyRecordSignature,
  createMemoryRegistry,
  zkPendingAttestation,
  bindingMessage,
  metaplexAttributes,
} from "../src/record.mjs";
import { ZK_PROGRAM_ID } from "../src/hash.mjs";

const dir = path.dirname(fileURLToPath(import.meta.url));
const load = (name) => JSON.parse(readFileSync(path.join(dir, "../examples", name), "utf8"));

function authority() {
  const kp = nacl.sign.keyPair();
  return { kp, pubkey: bs58.encode(kp.publicKey) };
}

test("zk attestation is pending while VK slots are empty and the program is not deployed", () => {
  const a = zkPendingAttestation();
  assert.equal(a.zk, "pending");
  assert.equal(a.vkSlotsEmpty, true);
  assert.equal(a.programDeployed, false);
  assert.equal(a.vkSlot, null);
  assert.equal(a.zkProgramId, ZK_PROGRAM_ID);
  assert.match(a.note, /not live/i);
});

test("records bind content hash, refuse mainnet, and carry Metaplex attributes", () => {
  const { pubkey } = authority();
  const ex = load("hauhau-qwen36-onchain.json");
  const rec = buildRecord({ type: ex.type, version: ex.version, authority: pubkey, payload: ex.payload });
  assert.equal(rec.cluster, "devnet");
  assert.equal(rec.type, "model");
  assert.equal(rec.payload.hfRepo, "ordlibrary/hauhau-qwen36-onchain");
  assert.match(rec.contentHash, /^sha256-[0-9a-f]{64}$/);
  const attrs = metaplexAttributes(rec);
  assert.ok(attrs.some((x) => x.key === "hfRepo" && x.value === rec.payload.hfRepo));
  assert.ok(attrs.some((x) => x.key === "zk" && x.value === "pending"));
  assert.throws(() => buildRecord({ type: ex.type, version: ex.version, authority: pubkey, payload: ex.payload, cluster: "mainnet-beta" }));
});

test("skill record recomputes and checks the SkillHub merkle leaf", () => {
  const { pubkey } = authority();
  const ex = load("skillhub-onchain.json");
  const rec = buildRecord({ type: ex.type, version: ex.version, authority: pubkey, payload: ex.payload });
  assert.equal(rec.skillMerkleLeaf, ex.payload.merkleLeaf);
  assert.throws(() =>
    buildRecord({
      type: ex.type,
      version: ex.version,
      authority: pubkey,
      payload: { ...ex.payload, merkleLeaf: "sha256-" + "00".repeat(32) },
    }),
  );
});

test("authority signature binds type, hash, version and authority", () => {
  const { kp, pubkey } = authority();
  const ex = load("cheshire-mcp-server.json");
  const rec = signRecord(buildRecord({ type: ex.type, version: ex.version, authority: pubkey, payload: ex.payload }), kp.secretKey);
  assert.equal(verifyRecordSignature(rec), true);
  assert.equal(rec.attestation.message, bindingMessage(rec));
  const tampered = { ...rec, version: "nope", uniquenessKey: rec.uniquenessKey.replace(ex.version, "nope") };
  tampered.attestation = { ...rec.attestation };
  assert.throws(() => verifyRecordSignature(tampered), /does not bind/);
});

test("memory registry enforces uniqueness, hash binding and authority", () => {
  const a = authority();
  const b = authority();
  const ex = load("hauhau-qwen36-onchain.json");
  const rec = signRecord(buildRecord({ type: ex.type, version: ex.version, authority: a.pubkey, payload: ex.payload }), a.kp.secretKey);
  const reg = createMemoryRegistry();
  reg.register(rec, { expectedAuthority: a.pubkey });
  assert.throws(() => reg.register(rec, { expectedAuthority: a.pubkey }), /duplicate/);
  assert.throws(() => reg.register({ ...rec, authority: b.pubkey, uniquenessKey: rec.uniquenessKey + "-x" }, { expectedAuthority: a.pubkey }), /authority mismatch/);
  const broken = { ...rec, contentHash: "sha256-" + "11".repeat(32), uniquenessKey: "model:" + "11".repeat(32) + ":" + ex.version };
  assert.throws(() => reg.register(broken), /does not bind/);
  assert.equal(reg.get(ex.type, rec.contentHash, ex.version).payload.hfRepo, ex.payload.hfRepo);
});
