export const SOL = 'So11111111111111111111111111111111111111112';
export const USDC = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v';
export const CHAIN = 'solana:mainnet';
export const SIGN_FEATURE = 'solana:signAndSendTransaction';
const ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';
const ORDER_AMOUNT_MAX = 9223372036854775807n; // DFlow /order uses a signed int64.

export function base58Encode(bytes) {
  let n = 0n;
  for (const b of bytes) n = n * 256n + BigInt(b);
  let encoded = '';
  while (n > 0n) { encoded = ALPHABET[Number(n % 58n)] + encoded; n /= 58n; }
  let zeros = 0;
  while (zeros < bytes.length && bytes[zeros] === 0) zeros++;
  return '1'.repeat(zeros) + encoded;
}

export function base58Decode(text) {
  if (typeof text !== 'string' || !text.length || text.length > 90) throw new Error('Invalid base58 address.');
  let n = 0n;
  for (const char of text) {
    const digit = ALPHABET.indexOf(char);
    if (digit < 0) throw new Error('Invalid base58 address.');
    n = n * 58n + BigInt(digit);
  }
  const bytes = [];
  while (n > 0n) { bytes.unshift(Number(n % 256n)); n /= 256n; }
  let zeros = 0;
  while (text[zeros] === '1') zeros++;
  return new Uint8Array([...Array(zeros).fill(0), ...bytes]);
}

export function validMint(mint) {
  try { return typeof mint === 'string' && mint.length >= 32 && mint.length <= 44 && base58Decode(mint).length === 32; }
  catch { return false; }
}

function checkDecimals(decimals) {
  if (!Number.isInteger(decimals) || decimals < 0 || decimals > 255) throw new Error('Token decimals are unavailable.');
}

export function toAtomic(value, decimals) {
  checkDecimals(decimals);
  if (typeof value !== 'string' || value.length > 280 || !/^(?:0|[1-9]\d*)(?:\.\d+)?$/.test(value)) {
    throw new Error('Enter a positive amount using digits and a decimal point.');
  }
  const [whole, fraction = ''] = value.split('.');
  if (fraction.length > decimals) throw new Error(`This token supports ${decimals} decimal places.`);
  const atomic = BigInt(whole + fraction.padEnd(decimals, '0'));
  if (atomic <= 0n || atomic > ORDER_AMOUNT_MAX) throw new Error('Amount is outside the supported order range.');
  return atomic.toString();
}

export function fromAtomic(value, decimals) {
  checkDecimals(decimals);
  if (!/^\d+$/.test(String(value))) throw new Error('Invalid token amount in quote.');
  const digits = BigInt(value).toString().padStart(decimals + 1, '0');
  if (decimals === 0) return digits;
  const fraction = digits.slice(-decimals).replace(/0+$/, '');
  return digits.slice(0, -decimals) + (fraction ? `.${fraction}` : '');
}

export function shortAddress(value) { return `${value.slice(0, 4)}…${value.slice(-4)}`; }
export function tokenName(mint) { return mint === SOL ? 'SOL' : mint === USDC ? 'USDC' : shortAddress(mint); }

export function compatibleWallet(wallet) {
  return !!wallet?.chains?.includes(CHAIN) && !!wallet.features?.['standard:connect'] &&
    !!wallet.features?.['standard:events'] && !!wallet.features?.[SIGN_FEATURE]?.signAndSendTransaction;
}
export function compatibleAccount(account) {
  return !!account?.chains?.includes(CHAIN) && !!account.features?.includes(SIGN_FEATURE) && validMint(account.address);
}

// Wallet Standard consumes the serialized transaction directly. Inspect its header
// to bind the review to its fee payer without pulling a full RPC SDK into the page.
export function transactionInfo(bytes) {
  if (!(bytes instanceof Uint8Array) || bytes.length < 100 || bytes.length > 1232) throw new Error('Invalid swap transaction.');
  let offset = 0;
  function shortvec() {
    let value = 0;
    for (let i = 0; i < 3; i++) {
      if (offset >= bytes.length) throw new Error('Truncated swap transaction.');
      const b = bytes[offset++];
      value |= (b & 127) << (7 * i);
      if (!(b & 128)) return value;
    }
    throw new Error('Invalid transaction length.');
  }
  const signatures = shortvec();
  if (signatures !== 1) throw new Error('This swap requires an unsupported signer configuration.');
  offset += signatures * 64;
  if (offset >= bytes.length) throw new Error('Truncated swap transaction.');
  let version = 'legacy';
  if (bytes[offset] & 128) {
    version = bytes[offset++] & 127;
    if (version !== 0) throw new Error('Unsupported transaction version.');
  }
  if (bytes[offset] !== 1 || bytes[offset + 1] !== 0) throw new Error('Invalid swap signer.');
  offset += 3;
  const accounts = shortvec();
  if (accounts < 1 || offset + accounts * 32 + 32 > bytes.length) throw new Error('Truncated swap accounts.');
  return { version, feePayer: base58Encode(bytes.slice(offset, offset + 32)) };
}

export function validateQuote(payload, request, now = Date.now()) {
  const order = payload?.order;
  if (!order || order.executionMode !== 'sync' || order.inputMint !== request.inputMint || order.outputMint !== request.outputMint || order.inAmount !== request.amount) {
    throw new Error('The returned quote does not match this swap. Request a new preview.');
  }
  checkDecimals(payload.inputDecimals);
  checkDecimals(payload.outputDecimals);
  if (toAtomic(request.displayAmount, payload.inputDecimals) !== request.amount) throw new Error('Token decimals changed. Request a new preview.');
  for (const field of ['outAmount', 'otherAmountThreshold']) {
    if (typeof order[field] !== 'string' || !/^\d+$/.test(order[field]) || BigInt(order[field]) <= 0n) throw new Error('The quote has no valid output amount.');
  }
  if (BigInt(order.otherAmountThreshold) > BigInt(order.outAmount)) throw new Error('The quote has an invalid minimum output.');
  if (!Number.isInteger(order.slippageBps) || order.slippageBps < 0 || order.slippageBps > 65535) throw new Error('The quote has no valid slippage.');
  if (request.slippageBps !== 'auto' && Number(request.slippageBps) !== order.slippageBps) throw new Error('The quote changed your slippage setting.');
  if (!Number.isFinite(Number(order.priceImpactPct))) throw new Error('The quote has no valid price impact.');
  if (!Number.isSafeInteger(order.lastValidBlockHeight) || order.lastValidBlockHeight <= 0) throw new Error('The transaction has no expiry height.');
  if (!Number.isSafeInteger(order.prioritizationFeeLamports) || order.prioritizationFeeLamports < 0) throw new Error('The quote has no valid priority fee.');
  if (!Number.isFinite(payload.expiresAt) || payload.expiresAt <= now) throw new Error('This quote has expired. Request a new preview.');
  if (typeof order.transaction !== 'string' || !/^[A-Za-z0-9+/]+={0,2}$/.test(order.transaction)) throw new Error('The quote has no valid transaction.');
  return { ...payload, expiresAt: Math.min(payload.expiresAt, now + 30_000), request };
}

export function canApprove(quote, { now = Date.now(), address, revision, busy = false }) {
  return !!quote && !busy && now < quote.expiresAt && quote.request.userPublicKey === address && quote.request.revision === revision;
}

export function confirmationState(status, blockHeight, lastValidBlockHeight) {
  if (status?.err != null) return 'failed';
  if (status?.confirmationStatus === 'confirmed' || status?.confirmationStatus === 'finalized') return 'confirmed';
  // A processed signature may still confirm after the validity window closes.
  // Only absent signatures can be declared expired from a later block height.
  if (status == null && Number.isSafeInteger(blockHeight) && blockHeight > lastValidBlockHeight) return 'expired';
  return 'pending';
}
