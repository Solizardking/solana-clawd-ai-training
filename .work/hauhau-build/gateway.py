#!/usr/bin/env python3
"""Hauhau UI + DFlow HTTP proxy. Wallets sign and broadcast; server only quotes."""
from __future__ import annotations
import gzip
import http.client
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections import OrderedDict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit, unquote
from urllib.request import Request, urlopen

WWW = Path(os.environ.get('HAUHAU_WWW', Path(__file__).parent / 'www')).resolve()
KEY = os.environ.get('DFLOW_API_KEY', '')
DFLOW = os.environ.get('DFLOW_TRADE_API_URL', 'https://quote-api.dflow.net' if KEY else 'https://dev-quote-api.dflow.net').rstrip('/')
RPC = os.environ.get('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
PORT = int(os.environ.get('PORT', '8080'))
TOKENS: dict[str,int] = {}
TOKEN_TIME = 0.0
TOKEN_LOCK = threading.Lock()
RATE_LOCK = threading.Lock()
RATES: OrderedDict = OrderedDict()
MODEL_PROCESS = None
MAX_BODY = 32768
ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'


def valid_address(value):
    if not isinstance(value, str) or not re.fullmatch(r'[1-9A-HJ-NP-Za-km-z]{32,44}', value):
        return False
    number = 0
    for c in value:
        number = number * 58 + ALPHABET.index(c)
    return len(value) - len(value.lstrip('1')) + (number.bit_length()+7)//8 == 32


def upstream_json(url, payload=None, headers=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, headers={'User-Agent':'Hauhau/1.0','Accept':'application/json', **({'Content-Type':'application/json'} if data else {}), **(headers or {})})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def dflow(path, params=None):
    return upstream_json(DFLOW+path+('?' + urlencode(params) if params else ''), headers={'x-api-key':KEY} if KEY else {})


def decimals(mint):
    # Avoid downloading DFlow's entire, ever-growing mint list for one trade.
    # These two mint accounts have immutable, canonical decimal counts.
    known = {'So11111111111111111111111111111111111111112':9,
             'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v':6}
    if mint in known:return known[mint]
    with TOKEN_LOCK:
        if mint in TOKENS:return TOKENS[mint]
    result=upstream_json(RPC,{'jsonrpc':'2.0','id':1,'method':'getTokenSupply','params':[mint,{'commitment':'confirmed'}]})
    value=result.get('result',{}).get('value',{})
    d=value.get('decimals')
    if type(d) is not int or not 0<=d<=255:
        raise ValueError('Mint metadata is not available yet. Try again after liquidity is available.')
    with TOKEN_LOCK:
        if len(TOKENS)>=4096:TOKENS.pop(next(iter(TOKENS)))
        TOKENS[mint]=d
    return d


def validate_order(body):
    if not isinstance(body,dict): raise ValueError('Expected a JSON object')
    params={}
    for key in ('inputMint','outputMint','userPublicKey'):
        if not valid_address(body.get(key)):raise ValueError('Invalid '+key)
        params[key]=body[key]
    if params['inputMint']==params['outputMint']:raise ValueError('Choose two different tokens')
    amount=body.get('amount')
    if not isinstance(amount,str) or not re.fullmatch(r'[0-9]{1,19}',amount) or not 0<int(amount)<=9223372036854775807:
        raise ValueError('Amount must be positive atomic units within int64 range')
    params['amount']=str(int(amount))
    slip=str(body.get('slippageBps','auto'))
    fee=str(body.get('prioritizationFeeLamports','auto'))
    if slip not in ('auto','10','50','100','300'):raise ValueError('Unsupported slippage setting')
    if fee not in ('auto','medium','high','disabled'):raise ValueError('Unsupported priority fee')
    params.update(slippageBps=slip,prioritizationFeeLamports=fee,prioritizationFeeMaxLamports='5000000')
    return params


def validate_response(order, params):
    if not isinstance(order,dict) or any(str(order.get(k))!=params[k] for k in ('inputMint','outputMint')) or str(order.get('inAmount'))!=params['amount']:
        raise ValueError('Upstream quote does not match the requested trade')
    if order.get('executionMode')!='sync':raise ValueError('This interface supports synchronous spot orders only')
    if not order.get('transaction') or type(order.get('lastValidBlockHeight')) is not int:
        raise ValueError('No signable transaction returned')
    for field in ('outAmount','otherAmountThreshold'):
        if not re.fullmatch(r'[0-9]+',str(order.get(field,''))):raise ValueError('Incomplete quote amounts')
    if int(order['otherAmountThreshold'])<=0:raise ValueError('Quote has no minimum output protection')
    if type(order.get('slippageBps')) is not int or not 0<=order['slippageBps']<=65535:raise ValueError('Invalid slippage from upstream')
    if type(order.get('prioritizationFeeLamports')) is not int or not 0<=order['prioritizationFeeLamports']<=5000000:raise ValueError('Invalid priority fee from upstream')


class Handler(SimpleHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(WWW),**kwargs)
    def setup(self):
        super().setup()
        self.connection.settimeout(30)
    def log_message(self,*_):pass # no wallet addresses, payloads, or secrets in request logs
    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('X-Frame-Options','DENY')
        super().end_headers()
    def reply(self,status,data):
        body=json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)
    def limited(self):
        now=time.monotonic()
        client=self.headers.get('Fly-Client-IP') if os.environ.get('FLY_APP_NAME') else self.client_address[0]
        client=client or self.client_address[0]
        with RATE_LOCK:
            count,start=RATES.get(client,(0,now))
            if now-start>60:count,start=0,now
            RATES[client]=(count+1,start);RATES.move_to_end(client)
            while len(RATES)>4096:RATES.popitem(last=False)
        if count>=90:
            self.reply(429,{'error':'Too many requests. Please wait a minute.'});return True
        return False
    def guard(self,callback):
        try:callback()
        except (ValueError,UnicodeDecodeError) as exc:self.reply(400,{'error':str(exc)})
        except HTTPError as exc:
            code='upstream_error'
            try:
                value=json.loads(exc.read(16384));candidate=value.get('code','upstream_error')
                if isinstance(candidate,str) and re.fullmatch(r'[a-zA-Z0-9_\-]{1,80}',candidate):code=candidate
            except Exception:pass
            self.reply(exc.code if 400<=exc.code<500 else 502,{'error':code,'message':'DFlow could not fulfill this request. Check the pair, amount, and available liquidity.'})
        except (URLError,TimeoutError,OSError):self.reply(502,{'error':'Upstream temporarily unavailable. Please retry.'})
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/health':self.reply(200,{'status':'ok','service':'hauhau-gateway'});return
        if path=='/api/trade/config':self.reply(200,{'network':'solana:mainnet','mode':'production' if KEY else 'developer','priorityFeeCapLamports':5000000});return
        if path=='/api/trade/token':
            if self.limited():return
            def token():
                mint=parse_qs(urlsplit(self.path).query).get('mint',[''])[0]
                if not valid_address(mint):raise ValueError('Invalid token mint')
                self.reply(200,{'mint':mint,'decimals':decimals(mint)})
            self.guard(token);return
        if path=='/api/model/health':
            try:self.reply(200,upstream_json('http://127.0.0.1:8081/health'))
            except Exception:self.reply(503,{'status':'loading' if MODEL_PROCESS and MODEL_PROCESS.poll() is None else 'unavailable'})
            return
        if path.startswith('/v1/'):
            self.proxy_model();return
        if path.startswith('/api/'):
            self.reply(404,{'error':'Not found'});return
        target=(WWW / unquote(path).lstrip('/')).resolve()
        if not target.is_relative_to(WWW):self.send_error(403);return
        if target.is_dir():target=target/'index.html'
        if not target.is_file():self.send_error(404);return
        body=target.read_bytes()
        compress='gzip' in self.headers.get('Accept-Encoding','') and target.suffix in ('.js','.css','.html','.json')
        if compress:body=gzip.compress(body,compresslevel=5)
        self.send_response(200);self.send_header('Content-Type',self.guess_type(str(target)))
        self.send_header('Cache-Control','no-cache' if target.suffix=='.html' else 'public, max-age=300')
        self.send_header('Vary','Accept-Encoding')
        if compress:self.send_header('Content-Encoding','gzip')
        self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def read_body(self,limit=MAX_BODY):
        try:length=int(self.headers.get('Content-Length','0'))
        except ValueError:raise ValueError('Invalid content length')
        if not 0<length<=limit:
            self.close_connection=True
            raise ValueError('Request body is missing or too large')
        return self.rfile.read(length)
    def do_POST(self):
        path=urlsplit(self.path).path
        if path.startswith('/v1/'):
            self.proxy_model();return
        if path not in ('/api/trade/order','/api/trade/status'):
            self.close_connection=True;self.reply(404,{'error':'Not found'});return
        if self.limited():self.close_connection=True;return
        if self.headers.get('Sec-Fetch-Site')=='cross-site':
            self.close_connection=True;self.reply(403,{'error':'Cross-site request refused'});return
        def perform():
            body=json.loads(self.read_body())
            if not isinstance(body,dict):raise ValueError('Expected a JSON object')
            if path.endswith('/order'):
                params=validate_order(body)
                in_dec,out_dec=decimals(params['inputMint']),decimals(params['outputMint'])
                order=dflow('/order',params);validate_response(order,params)
                self.reply(200,{'order':order,'inputDecimals':in_dec,'outputDecimals':out_dec,'expiresAt':int(time.time()*1000)+30000})
            else:
                sig=body.get('signature','')
                if not isinstance(sig,str) or not re.fullmatch(r'[1-9A-HJ-NP-Za-km-z]{64,88}',sig):raise ValueError('Invalid signature')
                result=upstream_json(RPC,{'jsonrpc':'2.0','id':1,'method':'getSignatureStatuses','params':[[sig],{'searchTransactionHistory':True}]})
                if 'error' in result:raise ValueError('Confirmation RPC unavailable; check the transaction in the explorer')
                height=upstream_json(RPC,{'jsonrpc':'2.0','id':2,'method':'getBlockHeight','params':[{'commitment':'confirmed'}]})
                self.reply(200,{'status':result['result']['value'][0],'blockHeight':height.get('result')})
        self.guard(perform)
    def proxy_model(self):
        # No key injection: public model calls retain the caller's Authorization.
        conn=http.client.HTTPConnection('127.0.0.1',8081,timeout=180)
        started=False
        try:
            body=self.read_body(1024*1024) if self.command=='POST' else None
            headers={k:self.headers[k] for k in ('Authorization','Content-Type','Accept') if k in self.headers}
            conn.request(self.command,self.path,body,headers);response=conn.getresponse()
            self.send_response(response.status);self.send_header('Content-Type',response.getheader('Content-Type','application/json'))
            self.send_header('Connection','close');self.end_headers();started=True;self.close_connection=True
            while chunk:=response.read1(65536):self.wfile.write(chunk)
        except ValueError as exc:self.reply(400,{'error':str(exc)})
        except (OSError,http.client.HTTPException):
            if not started:self.reply(503,{'error':'Model is loading or unavailable'})
            self.close_connection=True
        finally:conn.close()


def main():
    global MODEL_PROCESS
    print(f'Hauhau gateway port={PORT} dflow={DFLOW} key_present={bool(KEY)}',flush=True)
    if os.environ.get('HAUHAU_START_MODEL','1')=='1':
        MODEL_PROCESS=subprocess.Popen(['/entrypoint.sh'],env={**os.environ,'PORT':'8081','LLAMA_HOST':'127.0.0.1'})
    server=ThreadingHTTPServer(('0.0.0.0',PORT),Handler)
    def stop(*_):
        if MODEL_PROCESS:MODEL_PROCESS.terminate()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop)
    try:server.serve_forever()
    finally:
        server.server_close()
        if MODEL_PROCESS and MODEL_PROCESS.poll() is None:MODEL_PROCESS.terminate()
if __name__=='__main__':main()
