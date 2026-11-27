// Wallet Standard test wallet for headless e2e. Signing happens in Node via exposeFunction,
// so the throwaway devnet secret never enters the page.
import nacl from 'tweetnacl';
import bs58 from 'bs58';
import { inspectWire, withSignature } from '../src/tx-inspect.js';
export async function injectTestWallet(page, kp) {
  const address = bs58.encode(kp.publicKey);
  await page.exposeFunction('__signMessage', bytes => Array.from(nacl.sign.detached(Uint8Array.from(bytes), kp.secretKey)));
  await page.exposeFunction('__signTransaction', bytes => {
    const wire = Uint8Array.from(bytes); const info = inspectWire(wire, bs58.encode);
    if (info.feePayer !== address) throw new Error('test wallet: fee payer mismatch');
    return Array.from(withSignature(wire, info, address, nacl.sign.detached(info.message, kp.secretKey)));
  });
  await page.addInitScript(({ address, pub }) => {
    const account = { address, publicKey: Uint8Array.from(pub), chains: ['solana:devnet'], features: ['solana:signMessage', 'solana:signTransaction'] };
    const wallet = {
      version: '1.0.0', name: 'Devnet Test Wallet', chains: ['solana:devnet'], accounts: [],
      icon: 'data:image/svg+xml;base64,' + btoa('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8"><rect width="8" height="8" rx="2" fill="#7c5cff"/></svg>'),
      features: {
        'standard:connect': { version: '1.0.0', connect: async () => { wallet.accounts = [account]; return { accounts: [account] }; } },
        'standard:disconnect': { version: '1.0.0', disconnect: async () => { wallet.accounts = []; } },
        'standard:events': { version: '1.0.0', on: () => () => {} },
        'solana:signMessage': { version: '1.0.0', signMessage: async ({ message }) => [{ signedMessage: message, signature: Uint8Array.from(await window.__signMessage(Array.from(message))) }] },
        'solana:signTransaction': { version: '1.0.0', supportedTransactionVersions: ['legacy', 0], signTransaction: async ({ transaction, chain }) => { if (chain !== 'solana:devnet') throw new Error('test wallet: devnet only'); return [{ signedTransaction: Uint8Array.from(await window.__signTransaction(Array.from(transaction))) }]; } },
      },
    };
    const cb = ({ register }) => register(wallet);
    try { window.dispatchEvent(new CustomEvent('wallet-standard:register-wallet', { detail: cb })); } catch { /* ignore */ }
    window.addEventListener('wallet-standard:app-ready', ({ detail }) => cb(detail));
  }, { address, pub: Array.from(kp.publicKey) });
    return address;

}
