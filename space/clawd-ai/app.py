"""Public Clawd artifact catalog. No signing, credentials, or inference weights."""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import urllib.request

ROOT = Path(__file__).parent
FEATURED = {
    'ordlibrary/clawd-chart-foundation-training', 'ordlibrary/deepsol-clawd-code',
    'ordlibrary/solana-clawd-model-kit', 'solanaclawd/solana-nvidia-trading-factory-8b-GGUF',
    'solanaclawd/solana-clawd-live-data', 'solanaclawd/solana-nvidia-trading-factory-8b',
    'solanaclawd/solana-clawd-repo-corpus', 'ordlibrary/hauhau-qwen36-uncensored',
    'ordlibrary/clawd-agentic-layer-whitepaper',
}
CACHE = {}
LOCK = threading.Lock()


def included(repo_id, author):
    name = repo_id.split('/')[-1].lower()
    return name != 'readme' and (author == 'solanaclawd' or
                                any(word in name for word in ('clawd', 'hauhau', 'deepsol-clawd')))


def fetch_group(author, kind):
    url = f'https://huggingface.co/api/{kind}?author={author}&limit=100&full=true'
    results = []
    for _ in range(20):
        req = urllib.request.Request(url, headers={'User-Agent': 'Clawd-AI-Catalog/1.0'})
        with urllib.request.urlopen(req, timeout=12) as response:
            rows = json.load(response)
            links = response.headers.get('Link', '')
        for row in rows:
            repo_id = row.get('id', '')
            if not included(repo_id, author) or row.get('private'):
                continue
            card = row.get('cardData') or {}
            prefix = '' if kind == 'models' else kind + '/'
            results.append(dict(id=repo_id, kind=kind[:-1],
                                url='https://huggingface.co/' + prefix + repo_id,
                                downloads=row.get('downloads'), likes=row.get('likes', 0),
                                task=row.get('pipeline_tag') or card.get('task_categories') or '',
                                updated=row.get('lastModified'), featured=repo_id in FEATURED,
                                description=card.get('short_description') or '',
                                tags=row.get('tags', []), author=author))
        next_url = None
        for link in links.split(','):
            if 'rel="next"' in link:
                next_url = link.split('<', 1)[1].split('>', 1)[0]
        if not next_url:
            return results
        parsed = urlparse(next_url)
        if parsed.scheme != 'https' or parsed.netloc != 'huggingface.co':
            raise ValueError('Unexpected pagination host')
        url = next_url
    raise ValueError('Pagination limit exceeded')


def catalog():
    with LOCK:
        if CACHE and time.time() - CACHE['fetched_at'] < 60:
            return CACHE['payload']
        old = {r['id']: r for r in CACHE.get('payload', {}).get('repositories', [])}
        rows, errors = [], []
        groups = [(a, k) for a in ('solanaclawd', 'ordlibrary')
                  for k in ('models', 'datasets', 'spaces')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(fetch_group, a, k): (a, k) for a, k in groups}
            for future in concurrent.futures.as_completed(futures):
                a, k = futures[future]
                try:
                    rows.extend(dict(r, stale=False) for r in future.result())
                except Exception:
                    errors.append(a + '/' + k)
                    rows.extend(dict(r, stale=True) for r in old.values()
                                if r['author'] == a and r['kind'] == k[:-1])
        unique = {r['id']: r for r in rows}
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        payload = dict(repositories=sorted(unique.values(), key=lambda r:
                      (not r['featured'], -(r['downloads'] or 0), r['id'])),
                      checked_at=now, refresh_seconds=60, errors=errors,
                      status='partial' if errors else 'ready',
                      stats_as_of=CACHE.get('payload', {}).get('stats_as_of', now) if errors else now,
                      missing_featured=sorted(FEATURED - unique.keys()),
                      metric='Hugging Face downloads in the last 30 days')
        CACHE.update(fetched_at=time.time(), payload=payload)
        return payload


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/catalog':
            body = json.dumps(catalog()).encode()
            content_type = 'application/json'
        elif path == '/healthz':
            body, content_type = b'{"ok":true}', 'application/json'
        elif path in ('/', '/index.html'):
            body, content_type = (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    ThreadingHTTPServer(('0.0.0.0', int(os.environ.get('PORT', '7860'))), Handler).serve_forever()
