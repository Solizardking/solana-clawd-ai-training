import { cp, mkdir, readFile, writeFile, access } from 'node:fs/promises';
import { build } from 'esbuild';
import path from 'node:path';
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
await writeFile(kitAppPath, kitApp);
try { await cp(path.join(root, '../onchain.md'), path.join(out, 'docs/onchain.md')); } catch { /* deployed copy is in docs */ }
try { await cp(path.join(root, '../trading_factory/README.md'), path.join(out, 'docs/trading-factory.md')); } catch { /* deployed copy is in docs */ }
await cp(path.join(root, 'docs.html'), path.join(out, 'docs.html'));
await cp(path.join(root, 'src/index.html'), path.join(out, 'index.html'));
await build({ entryPoints: [path.join(root, 'src/app.js')], bundle: true, format: 'esm', outfile: path.join(out, 'app.js'), minify: true, target: 'es2022' });
await cp(path.join(root, 'src/style.css'), path.join(out, 'style.css'));
await cp(path.join(root, 'node_modules/pdfjs-dist/build/pdf.worker.min.mjs'), path.join(out, 'pdf.worker.min.mjs'));
const catalog = JSON.parse(await readFile(path.join(root, 'hub-catalog.json'), 'utf8'));
await writeFile(path.join(out, 'release.json'), JSON.stringify({ version: '0.2.0', built_at: new Date().toISOString(), catalog_updated_at: catalog.updated_at, routes: ['/', '/models', '/datasets', '/studio', '/trading-factory', '/model-kit', '/register', '/docs'] }));
console.log('Built hub, Studio, dataset builder, Trading Factory, onchain flow and canonical Model Kit frontend.');
