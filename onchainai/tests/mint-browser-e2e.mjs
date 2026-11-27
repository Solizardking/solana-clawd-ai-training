// Headless UI e2e for /mint on devnet. Injects a Wallet Standard test wallet whose signing happens in
// Node via exposeFunction (the throwaway devnet secret never enters the page). Usage:
//   BASE=http://localhost:4199 KEYPAIR=throwaway.json MODEL=deepseek-ai/DeepSeek-R1 SHOTS=/workspace/hf-mint-shots node tests/mint-browser-e2e.mjs
import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';
import nacl from 'tweetnacl';
import bs58 from 'bs58';
import { injectTestWallet } from './_test-wallet.mjs';

const BASE = process.env.BASE || 'http://localhost:4199';
const SHOTS = process.env.SHOTS || '/workspace/hf-mint-shots';
const MODEL = process.env.MODEL || 'deepseek-ai/DeepSeek-R1';
const DO_MINT = !!process.env.KEYPAIR;
const kp = DO_MINT ? nacl.sign.keyPair.fromSecretKey(Uint8Array.from(JSON.parse(readFileSync(process.env.KEYPAIR, 'utf8')))) : nacl.sign.keyPair();
const address = bs58.encode(kp.publicKey);
const browser = await chromium.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', args: ['--no-sandbox'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
page.on('pageerror', e => console.error('pageerror', e.message));
await injectTestWallet(page, kp);
await page.goto(`${BASE}/mint`, { waitUntil: 'networkidle' });
await page.waitForTimeout(800);
await page.screenshot({ path: `${SHOTS}/01-hero-desktop.png` });
await page.fill('#model-input', MODEL);
await page.click('#fetch-btn');
await page.waitForSelector('#browser-verify:has-text("matches")', { timeout: 90000 });
await page.evaluate(() => document.getElementById('preview').scrollIntoView({ block: 'start' }));
await page.waitForTimeout(400);
await page.screenshot({ path: `${SHOTS}/02-model-preview.png` });
await page.screenshot({ path: `${SHOTS}/00-full-page-preview.png`, fullPage: true });
await page.click('#step-wallet-btn');
await page.click('#wallet-dialog button:has-text("Devnet Test Wallet")');
await page.click('#bind-btn');
await page.waitForSelector('#bind-status.ok');
await page.check('#consent');
await page.evaluate(() => document.getElementById('mint').scrollIntoView({ block: 'start' }));
await page.waitForTimeout(300);
await page.screenshot({ path: `${SHOTS}/03-mint-steps-bound.png` });
let asset = null;
if (DO_MINT) {
  await page.click('#mint-btn');
  try { await page.waitForSelector('#result:not([hidden])', { timeout: 90000 }); }
  catch (e) { console.error('mint-status:', await page.$eval('#mint-status', n => n.textContent)); await page.screenshot({ path: `${SHOTS}/04-mint-refused-no-devnet-sol.png` }); if (!process.env.ALLOW_MINT_FAIL) throw e; }
  if (await page.$('#result:not([hidden])')) {
    asset = await page.$eval('#result .kv .mono', n => n.textContent);
    await page.waitForTimeout(800);
    await page.screenshot({ path: `${SHOTS}/04-minted-result.png` });
  }
  if (asset) { await page.waitForTimeout(9000); await page.$eval('#refresh-gallery', b => b.click()); }
  await page.waitForFunction(a => document.getElementById('gallery-grid').textContent.length > 0 && (!a || document.querySelector(`[data-verify="${a}"]`)), asset, { timeout: 60000 }).catch(() => {});
}
await page.evaluate(() => document.getElementById('gallery').scrollIntoView({ block: 'start' }));
const first = await page.$('[data-verify]');
if (first) { await first.evaluate(b => b.click()); await page.waitForFunction(() => /hash matches|✗/.test(document.querySelector('.vres')?.textContent || ''), null, { timeout: 60000 }).catch(() => {}); }
await page.waitForTimeout(300);
await page.screenshot({ path: `${SHOTS}/05-gallery.png` });
await page.screenshot({ path: `${SHOTS}/06-full-page-after.png`, fullPage: true });
const mobile = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 });
await mobile.goto(`${BASE}/mint?model=${encodeURIComponent('ordlibrary/hauhau-qwen36-onchain')}`, { waitUntil: 'networkidle' });
await mobile.waitForSelector('#browser-verify:has-text("matches")', { timeout: 60000 });
await mobile.screenshot({ path: `${SHOTS}/07-mobile.png`, fullPage: true });
const reg = await browser.newPage({ viewport: { width: 1440, height: 900 } });
await reg.goto(`${BASE}/registry`, { waitUntil: 'networkidle' });
await reg.waitForTimeout(800);
await reg.screenshot({ path: `${SHOTS}/08-registry-link.png` });
console.log(JSON.stringify({ ok: true, wallet: address, asset, status: await page.$eval('#mint-status', n => n.textContent) }));
await browser.close();
