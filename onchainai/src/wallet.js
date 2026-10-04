import { getWallets } from '@wallet-standard/app';
import { address, createTransactionMessage, setTransactionMessageFeePayer, setTransactionMessageLifetimeUsingBlockhash, appendTransactionMessageInstruction, compileTransaction, getTransactionEncoder, getBase64Decoder, getProgramDerivedAddress, getAddressEncoder, AccountRole } from '@solana/kit';
import bs58 from 'bs58';
import { digest } from './datasets.js';
export const MEMO_PROGRAM = 'MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr';
const REGISTRY_PROGRAM = '3dLst2E3djtCSwG19mFS3REHxtZPngjyga7iYZLDL5xj';
let connected;
let account;
export const currentWallet = () => account?.address || null;
export async function connectWallet() {
  const wallets = getWallets().get().filter(w => w.features['standard:connect'] && w.chains.some(c => c.startsWith('solana:')));
  if (!wallets.length) throw new Error('Install a Solana Wallet Standard wallet such as Phantom or Solflare, then reload.');
  const chosen = await new Promise(resolve => {
    const dialog = document.createElement('div'); dialog.id = 'wallet-dialog';
    const panel = document.createElement('div'); panel.className = 'panel'; panel.innerHTML = '<h2>Connect your wallet</h2><p class="muted">Your wallet keeps your signing keys.</p>';
    for (const wallet of wallets) { const button = document.createElement('button'); button.textContent = wallet.name; button.onclick = () => { dialog.remove(); resolve(wallet); }; panel.append(button); }
    const cancel = document.createElement('button'); cancel.textContent = 'Cancel'; cancel.onclick = () => { dialog.remove(); resolve(null); }; panel.append(cancel); dialog.append(panel); document.body.append(dialog);
  });
  if (!chosen) return null;
  const response = await chosen.features['standard:connect'].connect();
  account = response.accounts.find(a => a.chains.some(c => c.startsWith('solana:'))); connected = chosen;
  if (!account) throw new Error('Wallet returned no Solana account');
  chosen.features['standard:events']?.on('change', ({ accounts }) => {
    if (accounts) { account = accounts.find(a => a.chains.some(c => c.startsWith('solana:'))); window.dispatchEvent(new Event('wallet-changed')); }
  });
  window.dispatchEvent(new Event('wallet-changed')); return account.address;
}
export async function disconnectWallet() { await connected?.features['standard:disconnect']?.disconnect(); connected = undefined; account = undefined; window.dispatchEvent(new Event('wallet-changed')); }
export async function authorizePublish(payload) {
  if (!account || !connected.features['solana:signMessage']) throw new Error('Connect a wallet supporting message signing');
  const hash = await digest(JSON.stringify(payload)); const timestamp = Date.now(); const domain = location.hostname;
  const message = `Onchain AI\nDomain: ${domain}\nAction: publish-dataset\nWallet: ${account.address}\nDigest: ${hash}\nTimestamp: ${timestamp}`;
  const [signed] = await connected.features['solana:signMessage'].signMessage({ account, message: new TextEncoder().encode(message) });
  return { wallet: account.address, signature: bs58.encode(signed.signature), timestamp, domain };
}
async function rpc(cluster, method, params) {
  const response = await fetch('/api/rpc', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cluster, method, params }) });
  const data = await response.json(); if (!response.ok) throw new Error(data.error); return data.result;
}
export async function prepareCommitment({ cluster, hash, kind, mode = 'memo', endpoint = '' }) {
  if (!account) throw new Error('Connect your wallet first');
  if (!/^[a-f0-9]{64}$/.test(hash)) throw new Error('Enter a 64-character SHA-256 digest');
  const chain = cluster === 'devnet' ? 'solana:devnet' : 'solana:mainnet';
  if (!account.chains.includes(chain) || !connected.features['solana:signTransaction']) throw new Error(`Wallet must support signing transactions on ${cluster}`);
  const memo = JSON.stringify({ protocol: 'CAAP/1.0', kind, sha256: hash, authority: account.address });
  let instruction = { programAddress: address(MEMO_PROGRAM), data: new TextEncoder().encode(memo) };
  let pda = null, accountRent = 0;
  if (mode === 'registry') {
    const response = await fetch(`/api/onchain/status?cluster=${cluster}`); const gate = await response.json();
    if (!response.ok || !gate.native_registration_enabled) throw new Error('Native registration requires the verified devnet program and matching onchain IDL.');
    if (kind !== 'model') throw new Error('Native registration is for model artifacts. Use memo commitments for datasets and reports.');
    const endpointUrl = new URL(endpoint); if (endpointUrl.protocol !== 'https:' || endpoint.length > 200) throw new Error('Use an HTTPS inference endpoint under 200 characters.');
    [pda] = await getProgramDerivedAddress({ programAddress: address(REGISTRY_PROGRAM), seeds: [new TextEncoder().encode('model'), getAddressEncoder().encode(address(account.address))] });
    const existing = await rpc(cluster, 'getAccountInfo', [pda, { encoding: 'base64', commitment: 'confirmed' }]);
    if (existing.value) throw new Error(`This wallet already has a ModelRegistry PDA (${pda}). The deployed program supports one model per wallet; use a memo commitment for another artifact.`);
    const borshString = text => { const bytes = new TextEncoder().encode(text); const data = new Uint8Array(4 + bytes.length); new DataView(data.buffer).setUint32(0, bytes.length, true); data.set(bytes, 4); return data; };
    const parts = [new Uint8Array([150,196,232,118,138,43,140,244]), borshString(`sha256:${hash}`), new Uint8Array([1]), borshString(endpoint), new Uint8Array(8)];
    const data = new Uint8Array(parts.reduce((n, part) => n + part.length, 0)); let offset = 0; for (const part of parts) { data.set(part, offset); offset += part.length; }
    instruction = { programAddress: address(REGISTRY_PROGRAM), accounts: [{ address: pda, role: AccountRole.WRITABLE }, { address: address(account.address), role: AccountRole.WRITABLE_SIGNER }, { address: address('11111111111111111111111111111111'), role: AccountRole.READONLY }], data };
  }
  const { value: lifetime } = await rpc(cluster, 'getLatestBlockhash', [{ commitment: 'confirmed' }]);
  let message = createTransactionMessage({ version: 0 });
  message = setTransactionMessageFeePayer(address(account.address), message);
  message = setTransactionMessageLifetimeUsingBlockhash({ blockhash: lifetime.blockhash, lastValidBlockHeight: BigInt(lifetime.lastValidBlockHeight) }, message);
  message = appendTransactionMessageInstruction(instruction, message);
  if (mode === 'registry') message = appendTransactionMessageInstruction({ programAddress: address(MEMO_PROGRAM), data: new TextEncoder().encode(memo) }, message);
  const transaction = compileTransaction(message);
  const wire = getTransactionEncoder().encode(transaction); const base64 = getBase64Decoder().decode(wire);
  const [simulation, fee] = await Promise.all([rpc(cluster, 'simulateTransaction', [base64, { encoding: 'base64', sigVerify: false, commitment: 'confirmed', ...(pda ? { accounts: { encoding: 'base64', addresses: [pda] } } : {}) }]), rpc(cluster, 'getFeeForMessage', [getBase64Decoder().decode(transaction.messageBytes), { commitment: 'confirmed' }])]);
  if (simulation.value.err) throw new Error(`Simulation failed: ${JSON.stringify(simulation.value.err)}. Check wallet SOL balance and selected cluster.`);
  if (fee.value === null) throw new Error('Blockhash expired. Preview again.');
  if (pda) accountRent = simulation.value.accounts?.[0]?.lamports || 0;
  return { cluster, chain, hash, kind, mode, pda, account_rent_lamports: accountRent, memo, wire, fee_lamports: fee.value, logs: simulation.value.logs, wallet: account.address, prepared_at: Date.now() };
}
export async function submitCommitment(prepared) {
  if (!account || prepared.wallet !== account.address || Date.now() - prepared.prepared_at > 45000) throw new Error('Wallet changed or preview expired. Simulate again.');
  const [result] = await connected.features['solana:signTransaction'].signTransaction({ account, transaction: prepared.wire, chain: prepared.chain });
  const signature = await rpc(prepared.cluster, 'sendTransaction', [getBase64Decoder().decode(result.signedTransaction), { encoding: 'base64', skipPreflight: false, preflightCommitment: 'confirmed', maxRetries: 3 }]);
  return signature;
}
