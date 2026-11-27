// Shared devnet Wallet Standard helpers for self-serve Core mints (SkillHub page).
import { getWallets } from '@wallet-standard/app';
import nacl from 'tweetnacl';
import bs58 from 'bs58';
import { inspectWire, withSignature } from './tx-inspect.js';

export const CORE = 'CoREENxT6tW1HoK8ypY1SxRMZTcVPm7R94rH4PZNhX7d';
const COMPUTE_BUDGET = 'ComputeBudget111111111111111111111111111111';
export const CHAIN = 'solana:devnet';
export const w = { wallet: null, account: null };

export async function connectDevnetWallet() {
  const wallets = getWallets().get().filter(x => x.features['standard:connect'] && x.chains.includes(CHAIN));
  if (!wallets.length) throw new Error('No devnet-capable Solana wallet found. Install Phantom or Solflare, then reload.');
  const chosen = await new Promise(resolve => {
    const dialog = document.createElement('div'); dialog.id = 'wallet-dialog';
    const panel = document.createElement('div'); panel.className = 'panel';
    panel.innerHTML = '<div class="eyebrow">DEVNET ONLY</div><h2>Connect a wallet</h2><p class="muted">Your wallet keeps its keys. Transactions go to Solana devnet only.</p>';
    for (const x of wallets) { const b = document.createElement('button'); if (x.icon) { const img = document.createElement('img'); img.src = x.icon; img.alt = ''; b.append(img); } b.append(x.name); b.onclick = () => { dialog.remove(); resolve(x); }; panel.append(b); }
    const c = document.createElement('button'); c.textContent = 'Cancel'; c.onclick = () => { dialog.remove(); resolve(null); }; panel.append(c);
    dialog.append(panel); document.body.append(dialog);
  });
  if (!chosen) return null;
  const res = await chosen.features['standard:connect'].connect();
  const account = res.accounts.find(a => a.chains.includes(CHAIN)) || res.accounts.find(a => a.chains.some(c => c.startsWith('solana:')));
  if (!account) throw new Error('Wallet returned no Solana account.');
  if (!chosen.features['solana:signMessage'] || !chosen.features['solana:signTransaction']) throw new Error('This wallet cannot sign messages and transactions.');
  w.wallet = chosen; w.account = account;
  return account.address;
}
export async function signText(message) {
  const bytes = new TextEncoder().encode(message);
  const [signed] = await w.wallet.features['solana:signMessage'].signMessage({ account: w.account, message: bytes });
  if (!nacl.sign.detached.verify(bytes, signed.signature, bs58.decode(w.account.address))) throw new Error('Wallet signature does not verify.');
  return bs58.encode(signed.signature);
}
function includesBytes(hay, needle) { outer: for (let i = 0; i <= hay.length - needle.length; i++) { for (let j = 0; j < needle.length; j++) if (hay[i + j] !== needle[j]) continue outer; return true; } return false; }
/** Check the server-built tx, have the wallet sign it, add the asset signature; returns base64 wire. */
export async function signCoreCreate(b64, assetKp, mustContain) {
  const wire = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const assetAddr = bs58.encode(assetKp.publicKey);
  const info = inspectWire(wire, bs58.encode);
  if (info.feePayer !== w.account.address) throw new Error('Refusing: fee payer is not your wallet.');
  if (!info.programs.includes(CORE) || info.programs.some(p => p !== CORE && p !== COMPUTE_BUDGET)) throw new Error('Refusing: transaction calls a program other than Metaplex Core.');
  if (info.signers.some(s => s !== w.account.address && s !== assetAddr)) throw new Error('Refusing: unexpected signer.');
  if (mustContain && !includesBytes(info.message, new TextEncoder().encode(mustContain))) throw new Error('Refusing: transaction does not carry the hash you reviewed.');
  const [out] = await w.wallet.features['solana:signTransaction'].signTransaction({ account: w.account, transaction: wire, chain: CHAIN });
  const si = inspectWire(out.signedTransaction, bs58.encode);
  if (bs58.encode(si.message) !== bs58.encode(info.message)) throw new Error('Wallet changed the transaction. Nothing was sent.');
  const full = withSignature(out.signedTransaction, si, assetAddr, nacl.sign.detached(si.message, assetKp.secretKey));
  let bin = ''; for (const b of full) bin += String.fromCharCode(b);
  return btoa(bin);
}
