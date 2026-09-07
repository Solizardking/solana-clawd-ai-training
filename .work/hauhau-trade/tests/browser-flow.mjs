// All wallets, signatures, quotes and RPC results in this test are simulated.
// This test cannot broadcast a transaction or access a real wallet.
import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import assert from 'node:assert/strict';
import { SOL, USDC, base58Decode } from '../src/trade-core.mjs';
const here = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(process.env.HAUHAU_PLAYWRIGHT_PACKAGE || '/Users/8bit/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/package.json');
const { chromium } = require('playwright');
const fragment = await readFile(resolve(here,'trade.html'),'utf8');
const server = createServer(async (req,res) => {
  if (req.url === '/') {res.setHeader('Content-Type','text/html');res.end(`<!DOCTYPE html><html><meta name="viewport" content="width=device-width"><link rel="stylesheet" href="/trade.css"><style>body{margin:40px auto;max-width:510px;background:#080b10;color:white;font-family:Arial,sans-serif}</style>${fragment}<script type="module" src="/trade.js"></script></html>`);return;}
  if (['/trade.css','/trade.js'].includes(req.url)) {res.setHeader('Content-Type',req.url.endsWith('css')?'text/css':'text/javascript');res.end(await readFile(resolve(here,req.url.slice(1))));return;}
  res.statusCode=404;res.end();
});
await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
const browser = await chromium.launch({headless:true});
let finished = 0;
const tx = new Uint8Array(160);tx[0]=1;tx[65]=128;tx[66]=1;tx[69]=1;tx.set(base58Decode(SOL),70);
const transaction = Buffer.from(tx).toString('base64');

async function setup(options = {}) {
  const page = await browser.newPage({viewport:{width:560,height:1100}});
  const consoleErrors=[]; page.on('pageerror',error=>consoleErrors.push(error.message));
  await page.addInitScript(({SOL,USDC}) => {
    window.mock = {signCalls:[],reject:false,events:null};
    const account = address => ({address,publicKey:new Uint8Array(32),chains:['solana:mainnet'],features:['solana:signAndSendTransaction']});
    const wallet = {name:'Test Wallet (simulated)',version:'1.0.0',icon:'data:image/svg+xml,<svg/>',chains:['solana:mainnet'],accounts:[account(SOL)],features:{
      'standard:connect':{version:'1.0.0',connect:async()=>({accounts:wallet.accounts})},
      'standard:events':{version:'1.0.0',on:(_event,listener)=>{window.mock.events=listener;return()=>{window.mock.events=null;}}},
      'standard:disconnect':{version:'1.0.0',disconnect:async()=>{}},
      'solana:signAndSendTransaction':{version:'1.0.0',supportedTransactionVersions:[0,'legacy'],signAndSendTransaction:async input=>{window.mock.signCalls.push({chain:input.chain,address:input.account.address,bytes:input.transaction.length});if(window.mock.reject)throw new Error('User rejected the request.');return[{signature:new Uint8Array(64).fill(7)}];}},
    }};
    window.mock.switchAccount=()=>{wallet.accounts=[account(USDC)];window.mock.events?.({accounts:wallet.accounts});};
    window.addEventListener('wallet-standard:app-ready',event=>event.detail.register(wallet));
  },{SOL,USDC});
  const requests=[];
  await page.route('**/api/trade/**',async route=>{
    const request=route.request(),path=new URL(request.url()).pathname;requests.push(path);
    let body={};if(request.method()==='POST')body=request.postDataJSON();
    let payload;
    if(path.endsWith('/config'))payload={network:'solana:mainnet',mode:'developer',priorityFeeCapLamports:5000000};
    else if(path.endsWith('/token'))payload={mint:new URL(request.url()).searchParams.get('mint'),decimals:9};
    else if(path.endsWith('/status'))payload=options.status || {status:{confirmationStatus:'confirmed',err:null},blockHeight:99};
    else {
      if(options.beforeOrder)await options.beforeOrder(page);
      payload={inputDecimals:9,outputDecimals:6,expiresAt:Date.now()+(options.expiresIn??30000),order:{executionMode:'sync',inputMint:body.inputMint,outputMint:body.outputMint,inAmount:body.amount,outAmount:'1234567',otherAmountThreshold:'1200000',slippageBps:body.slippageBps==='auto'?50:Number(body.slippageBps),priceImpactPct:'0.002',lastValidBlockHeight:100,prioritizationFeeLamports:5000,contextSlot:123,routePlan:[{venue:'Test venue'}],transaction,...options.order}};
    }
    await route.fulfill({json:payload}).catch(()=>{});
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.getByText('Connect a wallet to preview a mainnet swap.',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Connect wallet',exact:true}).click();
  await page.getByRole('button',{name:'Test Wallet (simulated) →',exact:true}).click();
  await page.locator('#ht-amount').fill('0.01');
  return {page,requests,consoleErrors};
}
async function preview(page) {
  await page.getByRole('button',{name:'Preview swap',exact:true}).click();
  await page.getByText('Quote ready. Check the amounts, then approve in your wallet.',{exact:true}).waitFor();
}
try {
  {
    const {page,requests,consoleErrors}=await setup();await preview(page);
    assert.equal(await page.evaluate(()=>window.mock.signCalls.length),0);
    assert.match(await page.locator('#ht-review-values').innerText(),/0\.2%/);
    assert.match(await page.locator('#ht-review-values').innerText(),/1\.2 USDC/);
    await page.getByRole('button',{name:'Approve in wallet',exact:true}).click();
    await page.getByText('Swap confirmed',{exact:true}).first().waitFor();
    assert.deepEqual(await page.evaluate(()=>window.mock.signCalls),[{chain:'solana:mainnet',address:SOL,bytes:160}]);
    assert.ok(requests.includes('/api/trade/status'));assert.deepEqual(consoleErrors,[]);
    await page.screenshot({path:resolve(here,'tests/trade-confirmed.png'),fullPage:true});
    await page.close();console.log('PASS: preview is read-only, wallet-approved mainnet send, confirmed receipt');finished++;
  }
  {
    const {page}=await setup();await preview(page);
    await page.locator('#ht-amount').fill('0.02');assert.equal(await page.locator('#ht-review').isVisible(),false);
    assert.equal(await page.locator('#ht-approve').isEnabled(),false);assert.equal(await page.evaluate(()=>window.mock.signCalls.length),0);
    await preview(page);await page.evaluate(()=>window.mock.switchAccount());
    assert.equal(await page.locator('#ht-review').isVisible(),false);assert.equal(await page.locator('#ht-approve').isEnabled(),false);
    await page.close();console.log('PASS: form/account changes revoke reviewed transaction');finished++;
  }
  {
    const {page}=await setup({beforeOrder:async page=>{await page.evaluate(()=>window.mock.switchAccount());}});
    await page.getByRole('button',{name:'Preview swap',exact:true}).click();
    await page.getByText('Wallet account changed. Request a new preview.',{exact:true}).waitFor();
    await page.waitForFunction(()=>!document.getElementById('ht-fields').disabled);
    assert.equal(await page.locator('#ht-review').isVisible(),false);assert.equal(await page.evaluate(()=>window.mock.signCalls.length),0);
    await page.close();console.log('PASS: in-flight quote cannot restore approval after wallet account changes');finished++;
  }
  {
    const {page}=await setup({expiresIn:600});await preview(page);
    await page.getByText('Quote expired',{exact:true}).waitFor();
    assert.equal(await page.locator('#ht-approve').isEnabled(),false);assert.equal(await page.evaluate(()=>window.mock.signCalls.length),0);
    await page.close();console.log('PASS: expired quotes cannot reach the wallet');finished++;
  }
  {
    const {page}=await setup();await preview(page);await page.evaluate(()=>{window.mock.reject=true;});
    await page.getByRole('button',{name:'Approve in wallet',exact:true}).click();
    await page.getByText(/User rejected the request/).waitFor();
    assert.equal(await page.locator('#ht-review').isVisible(),false);assert.equal(await page.evaluate(()=>window.mock.signCalls.length),1);
    await page.close();console.log('PASS: wallet rejection invalidates quote and never retries signing');finished++;
  }
  {
    const {page}=await setup({status:{status:{err:{InstructionError:[1,'Custom']},confirmationStatus:'confirmed'},blockHeight:99}});
    await preview(page);await page.getByRole('button',{name:'Approve in wallet',exact:true}).click();
    await page.getByText('Swap failed on-chain',{exact:true}).waitFor();assert.equal(await page.evaluate(()=>window.mock.signCalls.length),1);
    await page.close();console.log('PASS: on-chain errors produce failed receipts, never success or retry');finished++;
  }
  {
    const {page,requests}=await setup();
    await page.evaluate(()=>document.dispatchEvent(new CustomEvent('hauhau:trade-token',{detail:{mint:'11111111111111111111111111111111',symbol:'FAKE SOL'}})));
    assert.equal(await page.locator('#ht-output-mint').inputValue(),'11111111111111111111111111111111');
    assert.equal(await page.locator('#ht-output-name').innerText(),'1111…1111');
    assert.equal(requests.includes('/api/trade/order'),false);
    await page.close();console.log('PASS: live-tape token selection uses mint identity and does not trade');finished++;
  }
  console.log(`${finished} mocked browser flows passed. No real wallet or RPC was used.`);
} finally { await browser.close();await new Promise(resolve=>server.close(resolve)); }
