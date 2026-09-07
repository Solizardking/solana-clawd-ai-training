const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('/Users/8bit/sol-gpt/node_modules/playwright-core');
const dir = '/Users/8bit/solana-clawd-ai-training/.work/hauhau-ui';
const orig = '/Users/8bit/sol-gpt/hauhau/www';
const server = http.createServer((req, res) => {
  const name = req.url.split('?')[0] === '/' ? 'index.html' : path.basename(req.url.split('?')[0]);
  const file = fs.existsSync(path.join(dir, name)) ? path.join(dir, name) : path.join(orig, name);
  if (!fs.existsSync(file)) { res.writeHead(404); return res.end(); }
  const ext = path.extname(file);
  res.setHeader('Content-Type', ({'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.json':'application/json'})[ext] || 'application/octet-stream');
  res.end(fs.readFileSync(file));
});
let browser;
(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  browser = await chromium.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true });
  const page = await browser.newPage({ viewport: {width:1440,height:1150} });
  const errors=[]; page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(() => {
    window.__sockets=[];
    class Socket {
      constructor(url){ this.url=url; window.__sockets.push(this); setTimeout(()=>this.onopen?.(),0); }
      close(){ this.onclose?.(); }
      emit(data){ this.onmessage?.({data:JSON.stringify(data)}); }
    }
    window.WebSocket=Socket;
    document.addEventListener('hauhau:trade-token', e=>window.__trade=e.detail);
  });
  await page.goto('http://127.0.0.1:'+server.address().port);
  assert.equal(await page.evaluate(()=>window.__sockets.length),1,'one websocket only');
  const mint = '8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump';
  await page.evaluate(mint=>{
    const s=window.__sockets[0];
    s.emit({type:'status',connected:true,totalLaunches:12431,clients:8});
    s.emit({type:'token-launch',mint,name:'Clawd',symbol:'CLAWD',marketCapSol:321.12,time:Date.now()});
    s.emit({type:'token-enriched',mint,name:'<img src=x onerror=alert(1)>',symbol:'CLAWD'});
  },mint);
  await page.waitForFunction(()=>document.querySelectorAll('.token-row').length===1);
  assert.equal(await page.locator('.token-row img').count(),0,'untrusted name cannot create elements');
  assert.equal(await page.locator('.token-name').textContent(),'<img src=x onerror=alert(1)>','enrichment merged safely');
  await page.locator('.token-trade').click();
  assert.deepEqual(await page.evaluate(()=>window.__trade),{mint,symbol:'CLAWD'});
  await page.locator('#stream-search').fill('does-not-match');
  assert.equal(await page.locator('.token-row').count(),0);
  await page.locator('#stream-search').fill('CLAWD');
  assert.equal(await page.locator('.token-row').count(),1);
  await page.locator('#stream-search').fill('');
  await page.locator('#stream-pause').click();
  await page.evaluate(()=>window.__sockets[0].emit({type:'token-launch',mint:'So11111111111111111111111111111111111111112',name:'Wrapped SOL',symbol:'SOL',marketCapSol:99,time:Date.now()}));
  await page.waitForTimeout(300);
  assert.equal(await page.locator('.token-row').count(),1,'paused view frozen');
  await page.locator('#stream-pause').click();
  assert.equal(await page.locator('.token-row').count(),2,'resume catches up');
  await page.locator('#stream-clear').click();
  assert.equal(await page.locator('.token-row').count(),0,'clear removes current feed');
  await page.evaluate(()=>{
    const chars='123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';
    for(let i=0;i<180;i++){
      const mint='1'.repeat(30)+chars[Math.floor(i/58)]+chars[i%58];
      window.__sockets[0].emit({type:'token-launch',mint,name:'Fixture '+i,symbol:'T'+i,marketCapSol:100+i,time:Date.now()});
    }
  });
  await page.waitForFunction(()=>document.querySelectorAll('.token-row').length===160);
  assert.equal(await page.locator('#stream-count').textContent(),'160');
  await page.locator('.token-trade').nth(3).focus();
  const focusedMint=await page.locator('.token-trade').nth(3).getAttribute('data-trade-mint');
  await page.evaluate(()=>window.__sockets[0].emit({type:'token-launch',mint:'So11111111111111111111111111111111111111112',name:'Wrapped SOL',symbol:'SOL',marketCapSol:99,time:Date.now()}));
  await page.waitForTimeout(300);
  assert.equal(await page.evaluate(()=>document.activeElement.dataset.tradeMint),focusedMint,'keyed render retains keyboard focus');
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.screenshot({path: '/Users/8bit/solana-clawd-ai-training/.work/hauhau-desktop.png',fullPage:true});
  for(const width of [390,768,1024,1440]){
    await page.setViewportSize({width,height:950});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth),false,'no horizontal overflow at '+width);
  }
  await page.setViewportSize({width:390,height:900});
  await page.screenshot({path:'/Users/8bit/solana-clawd-ai-training/.work/hauhau-mobile.png',fullPage:true});
  const count=await page.evaluate(()=>window.__sockets.length);
  await page.evaluate(()=>window.__sockets.at(-1).close());
  await page.waitForFunction(count=>window.__sockets.length>count,count);
  assert.equal(await page.locator('.token-row').count(),160,'reconnect retains feed');
  assert.deepEqual(errors,[],'no browser errors');
  console.log('PASS: one websocket, safe enrichment, document trade event, filtering, pause/resume, clear, 160-event cap, keyboard focus, reconnect, 390/768/1024/1440 widths.');
})().catch(error=>{ console.error(error); process.exitCode=1; }).finally(async()=>{ await browser?.close(); server.close(); });
