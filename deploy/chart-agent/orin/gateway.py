#!/usr/bin/env python3
"""Small Python 3.8-compatible Orin bridge to the private HF chart Space."""
import json
import os
import secrets
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALLOWED = {'/', '/ready', '/live', '/tools', '/detect', '/analyze', '/tokenize', '/model/health', '/model/v1/chat/completions'}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def handle_request(self):
        if self.path == '/health' and self.command == 'GET':
            payload = json.dumps({'service':'clawd-orin-chart-gateway','alive':True,'inference':'hugging-face'}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(payload);return
        if self.path not in ALLOWED:
            self.send_error(404);return
        expected = os.environ['CHART_API_KEY']
        provided = self.headers.get('X-Chart-API-Key') or (self.headers.get('Authorization','')[7:] if self.headers.get('Authorization','').startswith('Bearer ') else '')
        if not secrets.compare_digest(provided, expected):
            self.send_error(401);return
        try:
            length = int(self.headers.get('Content-Length','0'))
        except ValueError:
            self.send_error(400);return
        if length < 0 or length > 8_100_000:
            self.send_error(413);return
        self.connection.settimeout(310)
        body = self.rfile.read(length) if length else None
        headers = {'Authorization':'Bearer '+os.environ['HF_TOKEN'], 'Content-Type':'application/json', 'X-Chart-API-Key':expected}
        if self.path.startswith('/model/'):
            headers['X-Chart-Model-Key'] = os.environ['CHART_MODEL_API_KEY']
        req = urllib.request.Request(os.environ['CHART_SPACE_URL'].rstrip('/')+self.path, data=body, method=self.command, headers=headers)
        try:
            response = urllib.request.urlopen(req, timeout=305)
        except urllib.error.HTTPError as exc:
            response = exc
        except (urllib.error.URLError, TimeoutError):
            self.send_error(503, 'Hugging Face chart service unavailable');return
        with response:
            payload = response.read(16_000_001)
            if len(payload)>16_000_000:
                self.send_error(502);return
            self.send_response(response.status)
            self.send_header('Content-Type',response.headers.get('Content-Type','application/json'))
            self.send_header('Content-Length',str(len(payload)))
            self.end_headers();self.wfile.write(payload)

    do_GET = handle_request
    do_POST = handle_request


if __name__ == '__main__':
    for name in ['CHART_API_KEY','CHART_MODEL_API_KEY','HF_TOKEN','CHART_SPACE_URL']:
        if not os.environ.get(name):raise RuntimeError('Missing '+name)
    if not os.environ['CHART_SPACE_URL'].startswith('https://'):
        raise RuntimeError('HF upstream requires HTTPS')
    ThreadingHTTPServer((os.getenv('CHART_GATEWAY_HOST','127.0.0.1'),int(os.getenv('CHART_GATEWAY_PORT','8092'))),Handler).serve_forever()
