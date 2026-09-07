#!/usr/bin/env python3
"""Check public Solana price + OHLCV tools without credentials."""
import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chart_agent.candles import token_candles
from chart_agent.tools import token_market


async def main():
    mint = 'So11111111111111111111111111111111111111112'
    market, candles = await asyncio.gather(token_market(mint), token_candles(mint))
    report = {'market': market, 'candles': candles}
    Path('outputs/chart-agent/market-smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'mint': mint, 'pairs': len(market['pairs']), 'closed_bars': len(candles['ohlcv']),
                      'stale': candles['stale'], 'pool': candles['pool'], 'features': candles['features']}, indent=2))

if __name__ == '__main__':
    asyncio.run(main())
