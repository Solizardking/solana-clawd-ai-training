import { getWallets } from '@wallet-standard/app';
import { SOL, USDC, CHAIN, SIGN_FEATURE, base58Encode, validMint, toAtomic, fromAtomic, shortAddress, tokenName, compatibleWallet, compatibleAccount, transactionInfo, validateQuote, canApprove, confirmationState } from './trade-core.mjs';

const root = document.getElementById('trade');
if (root) initTrade(root);

export function initTrade(root) {
  const $ = id => root.querySelector(`#ht-${id}`);
  const el = Object.fromEntries(['form','fields','connect','wallet-label','wallet-dialog','wallet-list','wallet-message','close-wallet','disconnect','amount','input-mint','input-name','output-mint','output-name','output-amount','reverse','slippage','priority','preview','message','review','review-values','expiry','approve','receipt','receipt-title','receipt-message','explorer','check-status','service-mode','fee-cap'].map(id => [id,$(id)]));
  const registry = getWallets();
  const state = { config: null, wallet: null, account: null, offWallet: null, revision: 0, quote: null, busy: false, connecting: false, request: null, receipt: null, polling: false };
  const decimalsCache = new Map();
  let expiryTimer;

  function message(text, tone = '') { el.message.textContent = text; el.message.dataset.tone = tone; }
  function setBusy(value) {
    state.busy = value;
    root.setAttribute('aria-busy', String(value));
    el.fields.disabled = value;
    el.connect.disabled = value;
    updateActions();
  }
  function updateActions() {
    el.preview.disabled = !state.config || !state.account || state.busy;
    el.preview.textContent = state.busy ? 'Working…' : !state.account ? 'Connect wallet to preview' : 'Preview swap';
    el.approve.disabled = !canApprove(state.quote, { address: state.account?.address, revision: state.revision, busy: state.busy });
    el['wallet-label'].textContent = state.account ? `${state.wallet.name} · ${shortAddress(state.account.address)}` : 'Wallet disconnected';
    el['wallet-label'].title = state.account?.address || '';
    el.connect.textContent = state.account ? 'Change wallet' : 'Connect wallet';
    el.disconnect.hidden = !state.wallet;
  }
  function invalidate(reason = '') {
    state.revision++;
    state.quote = null;
    state.request?.abort();
    state.request = null;
    clearInterval(expiryTimer);
    el.review.hidden = true;
    el['output-amount'].textContent = '—';
    updateNames();
    updateActions();
    if (reason) message(reason);
  }
  function updateNames() {
    el['input-name'].textContent = validMint(el['input-mint'].value.trim()) ? tokenName(el['input-mint'].value.trim()) : 'Token';
    el['output-name'].textContent = validMint(el['output-mint'].value.trim()) ? tokenName(el['output-mint'].value.trim()) : 'Token';
    const selected = [el.slippage.selectedOptions[0].textContent, el.priority.selectedOptions[0].textContent];
    root.querySelector('.ht-settings summary span').textContent = selected.every(v => v === 'Auto') ? 'DFlow auto' : `${selected[0]} · ${selected[1]}`;
  }
  async function api(path, options = {}, signal) {
    const timeout = AbortSignal.timeout(22_000);
    const combined = signal && AbortSignal.any ? AbortSignal.any([timeout, signal]) : (signal || timeout);
    const response = await fetch(`/api/trade/${path}`, { ...options, headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers }, signal: combined, cache: 'no-store', credentials: 'same-origin' });
    let payload;
    try { payload = await response.json(); } catch { throw new Error('Trading service returned an unreadable response. Try again.'); }
    if (!response.ok) throw new Error(typeof payload.error === 'string' ? payload.error : payload.error?.message || payload.msg || `Trading service is unavailable (${response.status}).`);
    return payload;
  }
  async function decimals(mint, signal) {
    if (decimalsCache.has(mint)) return decimalsCache.get(mint);
    const token = await api(`token?mint=${encodeURIComponent(mint)}`, {}, signal);
    if (token.mint !== mint || !Number.isInteger(token.decimals) || token.decimals < 0 || token.decimals > 255) throw new Error('Could not verify this token’s decimals.');
    decimalsCache.set(mint, token.decimals);
    return token.decimals;
  }
  function attachWallet(wallet, account) {
    state.offWallet?.();
    state.wallet = wallet;
    state.account = account;
    state.offWallet = wallet?.features['standard:events'].on('change', changes => {
      if (!('accounts' in changes) && !('chains' in changes) && !('features' in changes)) return;
      // Any account event invalidates a reviewed transaction, including a switch
      // that happens while an asynchronous quote request is in flight.
      state.account = compatibleWallet(wallet) ? wallet.accounts.find(a => compatibleAccount(a) && a.address === state.account?.address) || wallet.accounts.find(compatibleAccount) || null : null;
      invalidate('Wallet account changed. Request a new preview.');
    });
    invalidate(account ? 'Wallet connected. Set an amount and preview your swap.' : 'Wallet disconnected.');
  }
  function renderWallets() {
    el['wallet-list'].replaceChildren();
    const wallets = registry.get().filter(compatibleWallet);
    if (!wallets.length) {
      const p = document.createElement('p'); p.className = 'ht-note';
      p.textContent = 'No compatible wallet found. Open this site in your Solana wallet’s browser or enable a Wallet Standard wallet extension, then reopen this chooser.';
      el['wallet-list'].append(p);
    }
    for (const wallet of wallets) {
      const button = document.createElement('button');
      button.className = 'ht-wallet-choice'; button.type = 'button'; button.textContent = `${wallet.name} →`; button.disabled = state.connecting;
      button.addEventListener('click', async () => {
        if (state.connecting) return;
        state.connecting = true; renderWallets(); el['wallet-message'].textContent = `Approve connection in ${wallet.name}…`;
        try {
          const result = await wallet.features['standard:connect'].connect();
          const accounts = result.accounts.filter(compatibleAccount);
          if (!accounts.length) throw new Error('This wallet has no Solana mainnet account with swap support.');
          if (accounts.length === 1) { attachWallet(wallet, accounts[0]); el['wallet-dialog'].close(); }
          else {
            el['wallet-list'].replaceChildren();
            for (const account of accounts) {
              const accountButton = document.createElement('button'); accountButton.className = 'ht-wallet-choice'; accountButton.type = 'button';
              accountButton.textContent = `${account.label || 'Account'} · ${shortAddress(account.address)}`; accountButton.title = account.address;
              accountButton.addEventListener('click', () => { attachWallet(wallet, account); el['wallet-dialog'].close(); });
              el['wallet-list'].append(accountButton);
            }
            el['wallet-message'].textContent = 'Choose the account to use for swaps.';
          }
        } catch (error) { el['wallet-message'].textContent = readableError(error, 'Wallet connection failed.'); renderWallets(); }
        finally { state.connecting = false; el['wallet-list'].querySelectorAll('button').forEach(b => { b.disabled = false; }); }
      });
      el['wallet-list'].append(button);
    }
  }
  el.connect.addEventListener('click', () => { renderWallets(); el['wallet-message'].textContent = ''; el['wallet-dialog'].showModal(); });
  el['close-wallet'].addEventListener('click', () => el['wallet-dialog'].close());
  el.disconnect.addEventListener('click', async () => {
    const wallet = state.wallet;
    attachWallet(null, null); el['wallet-dialog'].close();
    try { await wallet?.features['standard:disconnect']?.disconnect(); } catch { /* Site state is already disconnected. */ }
  });
  registry.on('register', () => { if (el['wallet-dialog'].open && !state.connecting) renderWallets(); });
  registry.on('unregister', (...wallets) => {
    if (wallets.includes(state.wallet)) attachWallet(null, null);
    if (el['wallet-dialog'].open && !state.connecting) renderWallets();
  });
  for (const id of ['amount','input-mint','output-mint','slippage','priority']) {
    el[id].addEventListener(id === 'slippage' || id === 'priority' ? 'change' : 'input', () => invalidate('Settings changed. Preview to update the quote.'));
  }
  for (const button of root.querySelectorAll('[data-ht-token]')) button.addEventListener('click', () => {
    const [side, symbol] = button.dataset.htToken.split(':');
    el[`${side}-mint`].value = symbol === 'SOL' ? SOL : USDC;
    invalidate();
  });
  el.reverse.addEventListener('click', () => {
    [el['input-mint'].value,el['output-mint'].value] = [el['output-mint'].value,el['input-mint'].value];
    el.amount.value = ''; invalidate('Pair reversed. Enter the amount to pay.');
  });
  document.addEventListener('hauhau:trade-token', event => {
    const mint = event.detail?.mint;
    if (!validMint(mint)) return;
    if (state.busy) { message('Finish the current wallet request before selecting another token.'); return; }
    if (el['input-mint'].value.trim() === mint) el['input-mint'].value = mint === SOL ? USDC : SOL;
    el['output-mint'].value = mint;
    // A tape symbol is unverified metadata. The mint remains the display identity.
    invalidate(`Token selected: ${mint}. Enter the amount to pay.`);
    root.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' });
    el.amount.focus({ preventScroll: true });
  });

  el.form.addEventListener('submit', async event => {
    event.preventDefault();
    if (!state.account || !state.config || state.busy) return;
    invalidate();
    const revision = state.revision;
    const account = state.account;
    const inputMint = el['input-mint'].value.trim();
    const outputMint = el['output-mint'].value.trim();
    const displayAmount = el.amount.value.trim();
    const controller = new AbortController(); state.request = controller;
    setBusy(true); message('Verifying tokens and finding your route…');
    try {
      if (!validMint(inputMint) || !validMint(outputMint)) throw new Error('Enter valid Solana mint addresses for both tokens.');
      if (inputMint === outputMint) throw new Error('Choose two different tokens to swap.');
      const inputDecimals = await decimals(inputMint, controller.signal);
      const request = { inputMint, outputMint, userPublicKey: account.address, amount: toAtomic(displayAmount, inputDecimals), slippageBps: el.slippage.value, prioritizationFeeLamports: el.priority.value, displayAmount, revision };
      const { displayAmount: _displayAmount, revision: _revision, ...body } = request;
      const payload = await api('order', { method: 'POST', body: JSON.stringify(body) }, controller.signal);
      if (state.revision !== revision || state.account?.address !== account.address) return;
      const quote = validateQuote(payload, request);
      if (quote.order.prioritizationFeeLamports > state.config.priorityFeeCapLamports) throw new Error('The quoted priority fee exceeds this site’s cap.');
      const transaction = Uint8Array.from(atob(quote.order.transaction), c => c.charCodeAt(0));
      const info = transactionInfo(transaction);
      if (info.feePayer !== account.address) throw new Error('The swap transaction does not match your wallet.');
      if (!state.wallet.features[SIGN_FEATURE].supportedTransactionVersions.includes(info.version)) throw new Error('Your wallet does not support this transaction version.');
      state.quote = { ...quote, transaction };
      showReview(); message('Quote ready. Check the amounts, then approve in your wallet.', 'success');
    } catch (error) {
      if (state.revision === revision) message(readableError(error, 'Could not get a swap quote.'), 'error');
    } finally { if (state.request === controller) state.request = null; setBusy(false); }
  });
  function showReview() {
    const q = state.quote; const o = q.order;
    const input = tokenName(o.inputMint), output = tokenName(o.outputMint);
    const impact = Number(o.priceImpactPct) * 100;
    const rows = [
      ['You pay', `${fromAtomic(o.inAmount, q.inputDecimals)} ${input}`],
      ['Expected receive', `${fromAtomic(o.outAmount, q.outputDecimals)} ${output}`],
      ['Minimum receive', `${fromAtomic(o.otherAmountThreshold, q.outputDecimals)} ${output}`],
      ['Slippage', `${o.slippageBps / 100}%${q.request.slippageBps === 'auto' ? ' (auto)' : ''}`],
      ['Price impact', `${impact.toLocaleString('en-US',{maximumFractionDigits:4})}%`, Math.abs(impact) >= 3 ? 'ht-impact-high' : ''],
      ['Priority fee', `${fromAtomic(o.prioritizationFeeLamports, 9)} SOL`],
      ['Network fee / rent', 'Shown in wallet'],
      ['Wallet', q.request.userPublicKey, 'ht-address'],
      ['Input mint', o.inputMint, 'ht-address'],
      ['Output mint', o.outputMint, 'ht-address'],
    ];
    if (o.platformFee && BigInt(o.platformFee.amount || '0') > 0n) {
      const isInput = o.platformFee.mode === 'inputMint';
      rows.splice(6,0,['Platform fee', `${fromAtomic(o.platformFee.amount, isInput ? q.inputDecimals : q.outputDecimals)} ${isInput ? input : output}`]);
    }
    const venues = [...new Set((o.routePlan || []).map(leg => leg.venue).filter(v => typeof v === 'string'))].slice(0,6);
    if (venues.length) rows.splice(6,0,['Route',venues.join(' → ')]);
    el['review-values'].replaceChildren();
    for (const [label,value,className] of rows) {
      const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = label; dd.textContent = value;
      if (className) dd.className = className; el['review-values'].append(dt,dd);
    }
    el['output-amount'].textContent = fromAtomic(o.outAmount,q.outputDecimals);
    el.review.hidden = false; el.approve.textContent = 'Approve in wallet';
    tickExpiry(); expiryTimer = setInterval(tickExpiry, 500);
  }
  function tickExpiry() {
    if (!state.quote) { clearInterval(expiryTimer); return; }
    const seconds = Math.max(0, Math.ceil((state.quote.expiresAt - Date.now()) / 1000));
    el.expiry.textContent = seconds ? `Fresh for ${seconds}s` : 'Quote expired';
    if (!seconds && !state.busy) { el.approve.textContent = 'Preview again to refresh'; clearInterval(expiryTimer); }
    updateActions();
  }
  el.approve.addEventListener('click', async () => {
    const quote = state.quote, wallet = state.wallet, account = state.account;
    if (!canApprove(quote, { address: account?.address, revision: state.revision, busy: state.busy })) { message('Preview expired or settings changed. Request a fresh preview.', 'error'); return; }
    const liveAccount = wallet.accounts.find(a => a.address === account.address && compatibleAccount(a));
    if (!liveAccount) { invalidate('Wallet account changed. Request a new preview.'); return; }
    setBusy(true); clearInterval(expiryTimer); el.approve.textContent = 'Review in your wallet…';
    message('Approve this swap in your wallet. Waiting for its response…');
    try {
      // This is the only signing call. It runs exclusively from the explicit
      // approval button and delegates both signing and broadcasting to the wallet.
      const results = await wallet.features[SIGN_FEATURE].signAndSendTransaction({
        transaction: quote.transaction, account: liveAccount, chain: CHAIN,
        options: { preflightCommitment: 'confirmed', skipPreflight: false, ...(Number.isSafeInteger(quote.order.contextSlot) ? { minContextSlot: quote.order.contextSlot } : {}) },
      });
      const signatureBytes = results?.[0]?.signature;
      if (!(signatureBytes instanceof Uint8Array) || signatureBytes.length !== 64) throw new Error('Wallet returned no transaction signature. Check wallet activity before retrying.');
      const signature = base58Encode(signatureBytes);
      state.receipt = { signature, lastValidBlockHeight: quote.order.lastValidBlockHeight, wallet: account.address };
      state.quote = null; el.review.hidden = true; el['output-amount'].textContent = '—';
      showReceipt('pending', 'Transaction sent', 'Waiting for Solana confirmation.');
      message('Transaction sent. Checking its on-chain status…');
      await pollReceipt();
    } catch (error) {
      // Some wallet errors can happen after broadcast. Never resubmit the order,
      // even if the wallet reports a timeout instead of a signature.
      state.quote = null; el.review.hidden = true; el['output-amount'].textContent = '—';
      message(`${readableError(error, 'The wallet could not complete the request.')} Check your wallet activity before trying another swap.`, 'error');
    } finally { setBusy(false); }
  });
  function showReceipt(status, title, description) {
    el.receipt.hidden = false; el.receipt.dataset.state = status;
    el['receipt-title'].textContent = title; el['receipt-message'].textContent = description;
    el.explorer.href = `https://orbmarkets.io/tx/${state.receipt.signature}`;
    el.explorer.hidden = false; el.explorer.title = state.receipt.signature;
    el['check-status'].hidden = status !== 'unknown';
  }
  async function pollReceipt() {
    if (!state.receipt || state.polling) return;
    const receipt = state.receipt; state.polling = true;
    setBusy(true);
    el['check-status'].hidden = true;
    const start = Date.now();
    try {
      while (state.receipt === receipt && Date.now() - start < 120_000) {
        try {
          const data = await api('status', { method: 'POST', body: JSON.stringify({ signature: receipt.signature }) });
          if (state.receipt !== receipt) return;
          const status = confirmationState(data.status, data.blockHeight, receipt.lastValidBlockHeight);
          if (status === 'confirmed') { showReceipt(status, 'Swap confirmed', 'Your transaction is confirmed on Solana. Open the transaction to see the settled amounts.'); message('Swap confirmed.', 'success'); return; }
          if (status === 'failed') { showReceipt(status, 'Swap failed on-chain', `Solana reported: ${JSON.stringify(data.status.err).slice(0,400)}. Check the transaction for fees and details.`); message('The swap failed on-chain. Review the transaction before requesting a new quote.', 'error'); return; }
          if (status === 'expired') { showReceipt(status, 'Transaction expired', 'The original block-height window passed and this RPC found no signature. Check the explorer or wallet history before creating a new swap.'); return; }
          showReceipt('pending', 'Transaction sent', data.status?.confirmationStatus === 'processed' ? 'Processed by Solana. Waiting for confirmation…' : 'Waiting for Solana confirmation…');
        } catch { showReceipt('pending', 'Confirmation delayed', 'The status service is temporarily unavailable. Your transaction may still confirm.'); }
        await new Promise(resolve => setTimeout(resolve, 3000));
      }
      if (state.receipt === receipt) showReceipt('unknown','Confirmation unknown','No final result yet. Check the explorer or wallet history before creating another swap. You can check this signature again.');
    } finally { state.polling = false; setBusy(false); }
  }
  el['check-status'].addEventListener('click', () => { void pollReceipt(); });
  function readableError(error, fallback) {
    if (error?.name === 'AbortError' || error?.name === 'TimeoutError') return 'The request timed out. Please try a fresh preview.';
    return typeof error?.message === 'string' ? error.message.slice(0,500) : fallback;
  }
  updateNames(); updateActions();
  api('config').then(config => {
    if (!['mainnet','mainnet-beta','solana:mainnet'].includes(config.network) || !Number.isSafeInteger(config.priorityFeeCapLamports) || config.priorityFeeCapLamports < 0) throw new Error('The trading service has an unsupported network configuration.');
    state.config = config;
    el['service-mode'].textContent = /dev/i.test(config.mode) ? 'DFlow developer endpoint · rate limited' : 'DFlow routing';
    el['fee-cap'].textContent = `Priority fees are capped at ${fromAtomic(config.priorityFeeCapLamports,9)} SOL. The quote shows the resolved fee; your wallet shows network fees and account rent.`;
    message('Connect a wallet to preview a mainnet swap.'); updateActions();
  }).catch(error => { message(readableError(error,'Trading is temporarily unavailable.'),'error'); });
}
