import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import bs58 from 'bs58';
const base = process.env.SMOKE_BASE_URL || 'http://127.0.0.1:4177';
const artifacts = new URL('../test-results/', import.meta.url);
await mkdir(artifacts, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on('pageerror', error => errors.push(error.message));
function pdfFixture() {
  const stream = 'BT /F1 12 Tf 30 100 Td (Solana uses program derived addresses.) Tj ET';
  const objects = ['<< /Type /Catalog /Pages 2 0 R >>', '<< /Type /Pages /Kids [3 0 R] /Count 1 >>', '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>', '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>', `<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`];
  let pdf = '%PDF-1.4\n'; const offsets = [0];
  objects.forEach((object, index) => { offsets.push(pdf.length); pdf += `${index + 1} 0 obj\n${object}\nendobj\n`; });
  const start = pdf.length; pdf += 'xref\n0 6\n0000000000 65535 f \n' + offsets.slice(1).map(n => `${String(n).padStart(10, '0')} 00000 n \n`).join('') + `trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${start}\n%%EOF`;
  return Buffer.from(pdf);
}
// Public, funded devnet IDL authority for unsigned simulation only. No private key or signer exists in this fixture.
const address = 'AAqkn72VgkZqFbWggn9SvzjzMRW5zsZrTe5VZKu9DwaM';
await page.addInitScript(({ address, publicKey }) => {
  const account = { address, publicKey: new Uint8Array(publicKey), chains: ['solana:devnet'], features: ['solana:signMessage', 'solana:signTransaction'] };
  const wallet = { version: '1.0.0', name: 'Read-only simulation fixture', icon: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg"/>', accounts: [account], chains: ['solana:devnet'], features: {
    'standard:connect': { version: '1.0.0', connect: async () => ({ accounts: [account] }) },
    'standard:disconnect': { version: '1.0.0', disconnect: async () => {} },
    'standard:events': { version: '1.0.0', on: () => () => {} },
    'solana:signMessage': { version: '1.0.0', signMessage: async () => { throw new Error('Read-only fixture cannot sign'); } },
    'solana:signTransaction': { version: '1.0.0', supportedTransactionVersions: [0], signTransaction: async () => { throw new Error('Read-only fixture cannot sign or broadcast'); } },
  } };
  window.addEventListener('wallet-standard:app-ready', event => event.detail.register(wallet));
  window.dispatchEvent(new CustomEvent('wallet-standard:register-wallet', { detail: api => api.register(wallet) }));
}, { address, publicKey: [...bs58.decode(address)] });
try {
  for (const path of ['/', '/models', '/datasets', '/studio', '/trading-factory', '/chart-agent', '/model-kit', '/register']) {
    await page.goto(base + path); await page.waitForSelector('#content');
    assert.ok(await page.locator('h1').textContent());
    if (path === '/models') { await page.locator('#search').fill('trading-factory'); assert.equal(await page.locator('#catalog-grid .card').count(), 3); }
    if (path === '/trading-factory') { await page.locator('#build-strategy').click(); assert.match(await page.locator('#strategy-output').textContent(), /600/); await page.locator('#weights').fill('60,60'); await page.locator('#build-strategy').click(); assert.match(await page.locator('#strategy-output').textContent(), /totaling 100/); }
    if (path === '/chart-agent') { await page.locator('#candles-file').setInputFiles({ name: 'candles.csv', mimeType: 'text/csv', buffer: Buffer.from('time,open,high,low,close,volume\n1,100,110,90,105,10\n2,105,120,100,115,20\n') }); await page.locator('#inspect-candles').click(); await page.waitForSelector('#export-chart'); assert.match(await page.locator('#chart-output').textContent(), /source_sha256/); }
    console.log('Route passed', path);
  }
  await page.goto(base + '/datasets'); await page.waitForSelector('#content'); assert.equal(await page.locator('#build-data').isDisabled(), true);
  await page.locator('#wallet').click(); await page.getByRole('button', { name: 'Read-only simulation fixture' }).click(); await page.waitForFunction(() => !document.getElementById('build-data').disabled);
  await page.locator('#files').setInputFiles([
    { name: 'examples.jsonl', mimeType: 'application/json', buffer: Buffer.from('{"messages":[{"role":"user","content":"What is a PDA?"},{"role":"assistant","content":"A program-derived address."}]}\n') },
    { name: 'notes.md', mimeType: 'text/markdown', buffer: Buffer.from('Solana accounts contain data and lamports.') },
    { name: 'examples.csv', mimeType: 'text/csv', buffer: Buffer.from('prompt,answer\nWhat signs a transaction?,The wallet\n') },
    { name: 'examples.yaml', mimeType: 'text/yaml', buffer: Buffer.from('- prompt: What is an ALT?\n  answer: An address lookup table\n') },
    { name: 'notebook.ipynb', mimeType: 'application/json', buffer: Buffer.from('{"cells":[{"cell_type":"markdown","source":["Wallet keys stay outside the model."]}]}') },
    { name: 'source.pdf', mimeType: 'application/pdf', buffer: pdfFixture() },
  ]);
  await page.locator('#build-data').click(); await page.waitForSelector('#download-manifest');
  assert.match(await page.locator('#dataset-preview').textContent(), /6 examples/);
  await page.screenshot({ path: new URL('datasets.png', artifacts).pathname, fullPage: true });
  console.log('Wallet-gated local dataset build: 6 formats including PDF passed');
  await page.goto(base + '/register'); await page.waitForSelector('#content');
  await page.locator('#wallet').click(); await page.getByRole('button', { name: 'Read-only simulation fixture' }).click();
  await page.locator('#artifact-hash').fill('a'.repeat(64));
  await page.locator('#preview-tx').click(); await page.waitForFunction(() => /Simulation passed|Simulation failed/.test(document.getElementById('tx-status').textContent), { timeout: 45000 });
  assert.match(await page.locator('#tx-status').textContent(), /Simulation passed/);
  assert.equal(await page.locator('#tx-consent').isChecked(), false);
  await page.locator('#commit-mode').selectOption('registry'); assert.equal(await page.locator('#sign-tx').isDisabled(), true);
  await page.locator('#preview-tx').click(); await page.waitForFunction(() => /Simulation passed|Simulation failed/.test(document.getElementById('tx-status').textContent), { timeout: 45000 });
  assert.match(await page.locator('#tx-status').textContent(), /Simulation passed/);
  assert.match(await page.locator('#tx-preview').textContent(), /model_registry_pda/);
  await page.screenshot({ path: new URL('native-registry-preview.png', artifacts).pathname, fullPage: true });
  console.log('Live unsigned devnet memo and native ModelRegistry simulations passed. No signing or broadcasts.');
  await page.setViewportSize({ width: 390, height: 844 }); await page.goto(base); await page.waitForSelector('.metrics');
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await page.screenshot({ path: new URL('home-mobile.png', artifacts).pathname, fullPage: true });
  assert.deepEqual(errors, []);
} catch (error) {
  console.error('UI failure detail:', await page.locator('#build-status').textContent().catch(() => ''), errors);
  await page.screenshot({ path: new URL('failure.png', artifacts).pathname, fullPage: true });
  throw error;
} finally { await browser.close(); }
