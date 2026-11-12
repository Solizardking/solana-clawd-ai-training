import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, resolve } from 'node:path';
import handler from '../api/router.mjs';
const root = resolve(import.meta.dirname, '../dist');
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.woff2': 'font/woff2', '.md': 'text/markdown' };
const port = Number(process.env.PORT) || 4177;
const host = process.env.HOST || '0.0.0.0';
createServer(async (req, res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  if (pathname.startsWith('/api/')) return handler(req, res);
  if (pathname === '/.well-known/clawd-registry.json') { req.url = '/api/registry'; return handler(req, res); }
  let file = pathname;
  if (pathname === '/kit' || pathname === '/kit/') file = '/kit/index.html';
  else if (['/', '/models', '/datasets', '/data', '/studio', '/studio.html', '/trading-factory', '/model-kit', '/register', '/dashboard', '/chart-agent'].includes(pathname)) file = '/index.html';
  else if (pathname === '/docs' || pathname === '/docs-index') file = '/docs.html';
  const candidates = [file];
  if (!extname(file)) candidates.push(`${file}.html`, `${file}/index.html`);
  try {
    let data, type;
    for (const candidate of candidates) {
      const resolved = resolve(root, '.' + candidate);
      if (!resolved.startsWith(root + '/')) continue;
      try {
        data = await readFile(resolved);
        type = mime[extname(candidate)] || 'application/octet-stream';
        break;
      } catch { /* try next candidate */ }
    }
    if (!data) throw new Error();
    res.setHeader('Content-Type', type);
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
    res.setHeader('X-Frame-Options', 'SAMEORIGIN');
    res.end(data);
  } catch { res.statusCode = 404; res.end('Not found'); }
}).listen(port, host, () => console.log(`Onchain AI http://${host}:${port}`));
