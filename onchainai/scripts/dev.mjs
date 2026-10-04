import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, resolve } from 'node:path';
import handler from '../api/router.mjs';
const root = resolve(import.meta.dirname, '../dist');
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.woff2': 'font/woff2', '.md': 'text/markdown' };
createServer(async (req, res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  if (pathname.startsWith('/api/')) return handler(req, res);
  if (pathname === '/.well-known/clawd-registry.json') { req.url = '/api/registry'; return handler(req, res); }
  let file = pathname;
  if (pathname === '/kit' || pathname === '/kit/') file = '/kit/index.html';
  else if (['/', '/models', '/datasets', '/data', '/studio', '/studio.html', '/trading-factory', '/model-kit', '/register', '/dashboard', '/chart-agent'].includes(pathname)) file = '/index.html';
  else if (pathname === '/docs' || pathname === '/docs-index') file = '/docs.html';
  try {
    const resolved = resolve(root, '.' + file);
    if (!resolved.startsWith(root + '/')) throw new Error();
    const data = await readFile(resolved); res.setHeader('Content-Type', mime[extname(file)] || 'application/octet-stream'); res.end(data);
  } catch { res.statusCode = 404; res.end('Not found'); }
}).listen(4177, '127.0.0.1', () => console.log('Onchain AI http://127.0.0.1:4177'));
