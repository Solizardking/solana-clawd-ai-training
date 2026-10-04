import { cp, mkdir, readFile, writeFile, access } from 'node:fs/promises';
import { build } from 'esbuild';
import path from 'node:path';
import { createHash } from 'node:crypto';
const root = path.resolve(import.meta.dirname, '..');
const out = path.join(root, 'dist');
await mkdir(out, { recursive: true });
for (const file of ['brand', 'docs', 'data-catalog.json', 'docs-catalog.json', 'hub-catalog.json', 'registry-idl.json']) await cp(path.join(root, file), path.join(out, file), { recursive: true });
let kitSource = path.join(root, '../model-kit/frontend');
try { await access(kitSource); } catch { kitSource = path.join(root, 'kit'); }
await cp(kitSource, path.join(out, 'kit'), { recursive: true });
await writeFile(path.join(out, 'kit/config.js'), 'window.MODEL_KIT_CONFIG = {apiBaseUrl: "", x402Home: "https://x402.wtf", modelsHome: "/models", registerHome: "/register", onchainHome: "/", githubRepo: "https://github.com/Solizardking/solana-clawd-ai-training"};');
const kitAppPath = path.join(out, 'kit/app.js');
let kitApp = await readFile(kitAppPath, 'utf8');
kitApp = kitApp.replace('return "http://127.0.0.1:8787";', 'return window.location.origin;');
const snapshot = JSON.parse(await readFile(path.join(root, 'hub-catalog.json'), 'utf8'));
const fallback = { protocol: 'CAAP/1.0', models: snapshot.models.map(m => ({ repo_id: m.id, url: m.url, status: 'release-snapshot' })), datasets: snapshot.datasets.map(d => ({ repo_id: d.id, url: d.url, status: 'release-snapshot' })), jobs: [], constitution: { ok: false, files: [] } };
kitApp = kitApp.replace('let arenaProviders = [', `Object.assign(fallbackStatus, ${JSON.stringify(fallback)});\nlet arenaProviders = [`);
await writeFile(kitAppPath, kitApp);
const kitIndexPath = path.join(out, 'kit/index.html');
const kitIndex = await readFile(kitIndexPath, 'utf8');
await writeFile(kitIndexPath, kitIndex.replace('<body>', '<body><div style="padding:14px 24px;background:#151c27;color:#c9d9e8;font:14px system-ui">Onchain AI Model Kit · Catalog and CAAP previews use this site’s API. Arena execution and GPU training require your local runtime or configured Model Kit backend. <a style="color:#14f195" href="/studio">Use hosted Core AI in Studio</a> · <a style="color:#14f195" href="/">Back to workspace</a></div>'));
try { await cp(path.join(root, '../onchain.md'), path.join(out, 'docs/onchain.md')); } catch { /* deployed copy is in docs */ }
try { await cp(path.join(root, '../trading_factory/README.md'), path.join(out, 'docs/trading-factory.md')); } catch { /* deployed copy is in docs */ }
const docsHtml = await readFile(path.join(root, 'docs.html'), 'utf8');
await writeFile(path.join(out, 'docs.html'), docsHtml.replace(/<script src="[^\"]*marked[^\"]*"><\/script>/, '<script src="/docs-renderer.js"></script>').replace("content.innerHTML = '<p>failed to load ' + file + ': ' + e.message + '</p>';", "content.textContent = 'Failed to load ' + file + ': ' + e.message;"));
await cp(path.join(root, 'src/index.html'), path.join(out, 'index.html'));
await build({ entryPoints: [path.join(root, 'src/app.js')], bundle: true, format: 'esm', outfile: path.join(out, 'app.js'), minify: true, target: 'es2022' });
await build({ entryPoints: [path.join(root, 'src/docs-renderer.js')], bundle: true, format: 'iife', outfile: path.join(out, 'docs-renderer.js'), minify: true, target: 'es2022' });
await cp(path.join(root, 'src/style.css'), path.join(out, 'style.css'));
try { await cp(path.join(root, 'downloads'), path.join(out, 'downloads'), { recursive: true }); } catch { /* Source download is optional for a minimal fork. */ }
await cp(path.join(root, 'node_modules/pdfjs-dist/build/pdf.worker.min.mjs'), path.join(out, 'pdf.worker.min.mjs'));
const catalog = JSON.parse(await readFile(path.join(root, 'hub-catalog.json'), 'utf8'));
const files = {};
for (const filename of ['app.js', 'style.css', 'kit/app.js', 'docs-renderer.js', 'registry-idl.json']) files[filename] = createHash('sha256').update(await readFile(path.join(out, filename))).digest('hex');
await writeFile(path.join(out, 'release.json'), JSON.stringify({ version: '0.2.0', built_at: new Date().toISOString(), catalog_updated_at: catalog.updated_at, files, routes: ['/', '/models', '/datasets', '/studio', '/trading-factory', '/chart-agent', '/model-kit', '/register', '/docs'] }));
console.log('Built hub, Studio, dataset builder, Trading Factory, onchain flow and canonical Model Kit frontend.');
