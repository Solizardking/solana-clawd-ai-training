"""Read-only native tools and a catalog-restricted SOL GPT MCP bridge."""
import json
import os
import re
import time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'


def validate_mint(mint):
    if not isinstance(mint, str) or not 32 <= len(mint) <= 44 or any(c not in ALPHABET for c in mint):
        raise ValueError('Expected a Solana base58 address')
    n = 0
    for c in mint:
        n = n * 58 + ALPHABET.index(c)
    length = (n.bit_length() + 7) // 8 + len(mint) - len(mint.lstrip('1'))
    if length != 32:
        raise ValueError('Solana address must decode to 32 bytes')
    return mint


def catalog():
    text = (ROOT / 'docs/solgpt-chart-tool-catalog.md').read_text()
    return {name: {'name': name, 'core': core == 'yes', 'description': description.strip()}
            for name, core, description in re.findall(r'^\| `([a-z_]+)` \|\s*(yes)?\s*\| (.+?) \|$', text, re.M)}


async def token_market(mint):
    validate_mint(mint)
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get('https://api.dexscreener.com/token-pairs/v1/solana/' + mint)
        response.raise_for_status()
        pairs = response.json()
    if not isinstance(pairs, list):
        raise ValueError('Unexpected market response')
    # Price is quoted for baseToken, not quoteToken. Do not attribute a quote asset's price to the requested mint.
    pairs = [p for p in pairs if p.get('chainId') == 'solana' and p.get('baseToken', {}).get('address') == mint]
    pairs.sort(key=lambda p: (p.get('liquidity') or {}).get('usd') or 0, reverse=True)
    fields = ('pairAddress', 'dexId', 'baseToken', 'quoteToken', 'priceUsd', 'priceNative', 'liquidity', 'volume', 'priceChange', 'txns', 'marketCap', 'fdv')
    return {'source': 'dexscreener', 'fetched_at': time.time(), 'provider_event_time': None,
            'mint': mint, 'pairs': [{k: p.get(k) for k in fields} for p in pairs[:5]]}


class SolGptBridge:
    def __init__(self):
        self.url = os.getenv('SOLGPT_MCP_URL', 'https://solanaclawd.com/api/mcp')
        self.key = os.getenv('SOLGPT_MCP_TOKEN') or os.getenv('SOLGPT_API_KEY')
        self.available = {}
        self.contract = catalog()

    async def rpc(self, method, params):
        if not self.key:
            raise ValueError('SOLGPT_API_KEY with mcp scope or SOLGPT_MCP_TOKEN is required')
        if not self.url.startswith('https://') and not self.url.startswith(('http://127.0.0.1:', 'http://localhost:')):
            raise ValueError('MCP credentials require HTTPS or loopback')
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(self.url, headers={'Authorization': 'Bearer ' + self.key,
                'Accept': 'application/json, text/event-stream', 'MCP-Protocol-Version': '2025-03-26'},
                json={'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params})
            response.raise_for_status()
            payload = response.json()
        if 'error' in payload:
            raise ValueError('SOL GPT MCP rejected request')
        return payload.get('result', {})

    async def discover(self):
        tools, cursor = {}, None
        for _ in range(20):
            result = await self.rpc('tools/list', {'cursor': cursor} if cursor else {})
            for tool in result.get('tools', []):
                if tool['name'] in self.contract:
                    tools[tool['name']] = tool
            cursor = result.get('nextCursor')
            if not cursor:
                self.available = tools
                return
        raise ValueError('MCP catalog pagination exceeded limit')

    async def call(self, name, arguments):
        if name not in self.contract:
            raise ValueError('Tool is outside the supplied 72-tool contract')
        if not self.available:
            await self.discover()
        if name not in self.available:
            raise ValueError('Tool is not available from the connected MCP server')
        return await self.rpc('tools/call', {'name': name, 'arguments': arguments})


def function(name, description, properties, required=()):
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties, 'required': list(required), 'additionalProperties': False}}}


TOOL_DEFS = [
    function('get_token_market', 'Fetch current Solana token price/liquidity snapshots. Requires the exact mint.', {'mint': {'type': 'string'}}, ['mint']),
    function('get_live_tape', 'Recent received Solana launch/trade events; inspect freshness and connection status.', {}),
    function('search_research', 'Find page-cited local chart detection and Solana research.', {'query': {'type': 'string'}}, ['query']),
    function('search_solgpt_tools', 'Search the supplied 72-tool catalog; returns live input schemas when connected.', {'query': {'type': 'string'}}, ['query']),
    function('call_solgpt_tool', 'Call a connected SOL GPT tool using its discovered schema. Swap/transfer tools only prepare unsigned transactions; the user wallet must sign.',
             {'name': {'type': 'string'}, 'arguments': {'type': 'object'}}, ['name', 'arguments']),
]
