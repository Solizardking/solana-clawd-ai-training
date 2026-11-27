// Minimal Solana wire-format reader (legacy + v0) used by the browser before signing and by tests.
// Pure JS; base58 encoder injected so the browser and node share it.
function shortvec(bytes, o) { let len = 0, size = 0; for (;;) { const b = bytes[o + size]; len |= (b & 0x7f) << (7 * size); size++; if (!(b & 0x80)) break; } return [len, size]; }
export function inspectWire(wire, b58encode) {
  const bytes = wire instanceof Uint8Array ? wire : new Uint8Array(wire);
  const [nSigs, s0] = shortvec(bytes, 0);
  const sigStart = s0, msgStart = s0 + 64 * nSigs;
  const message = bytes.subarray(msgStart);
  let o = 0; let version = 'legacy';
  if (message[0] & 0x80) { version = message[0] & 0x7f; o = 1; }
  const header = { required: message[o], readonlySigned: message[o + 1], readonlyUnsigned: message[o + 2] }; o += 3;
  const [nKeys, s1] = shortvec(message, o); o += s1;
  const keys = []; for (let i = 0; i < nKeys; i++) { keys.push(b58encode(message.subarray(o, o + 32))); o += 32; }
  const blockhash = b58encode(message.subarray(o, o + 32)); o += 32;
  const [nIx, s2] = shortvec(message, o); o += s2;
  const programs = [];
  for (let i = 0; i < nIx; i++) {
    const programIndex = message[o]; o += 1;
    const [na, sa] = shortvec(message, o); o += sa + na;
    const [nd, sd] = shortvec(message, o); o += sd + nd;
    programs.push(keys[programIndex]);
  }
  const signers = keys.slice(0, header.required);
  return { version, nSigs, sigStart, msgStart, message, header, keys, signers, feePayer: keys[0], blockhash, programs };
}
/** Write a 64-byte signature for `signer` into a copy of the wire. */
export function withSignature(wire, info, signer, signature) {
  const idx = info.signers.indexOf(signer);
  if (idx < 0) throw new Error(`${signer} is not a required signer`);
  const out = new Uint8Array(wire);
  out.set(signature, info.sigStart + 64 * idx);
  return out;
}
