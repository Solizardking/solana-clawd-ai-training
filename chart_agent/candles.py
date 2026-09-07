"""Public Solana OHLCV with explicit identity, timestamps and closed-bar features."""
import math
import time
import httpx
from .tools import validate_mint


def candle_features(candles):
    if not 2 <= len(candles) <= 500:
        raise ValueError('Expected 2 to 500 candles')
    previous = -1
    for row in candles:
        if len(row) != 6 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in row):
            raise ValueError('Expected finite [timestamp, open, high, low, close, volume] rows')
        timestamp, opening, high, low, close, volume = row
        if timestamp <= previous or low <= 0 or low > min(opening, close) or high < max(opening, close) or volume < 0:
            raise ValueError('Invalid OHLCV ordering or bounds')
        previous = timestamp
    signals = []
    for i, (timestamp, opening, high, low, close, volume) in enumerate(candles):
        span, body = high - low, abs(close - opening)
        if span <= 0:
            continue
        if body / span <= 0.1:
            signals.append({'timestamp': timestamp, 'shape': 'doji'})
        elif min(opening, close) - low >= 2 * body and high - max(opening, close) <= body:
            signals.append({'timestamp': timestamp, 'shape': 'long_lower_wick'})
        if i:
            _, po, _, _, pc, _ = candles[i - 1]
            if close > opening and pc < po and opening <= pc and close >= po:
                signals.append({'timestamp': timestamp, 'shape': 'bullish_body_engulfing'})
            elif close < opening and pc > po and opening >= pc and close <= po:
                signals.append({'timestamp': timestamp, 'shape': 'bearish_body_engulfing'})
    closes = [r[4] for r in candles]
    return {'bars': len(candles), 'window_high': max(r[2] for r in candles), 'window_low': min(r[3] for r in candles),
            'close_change_pct': 100 * (closes[-1] / closes[0] - 1),
            'sma20': sum(closes[-20:]) / 20 if len(closes) >= 20 else None,
            'shapes': signals[-12:], 'interpretation': 'Descriptive candle geometry, not a buy/sell prediction.'}


async def token_candles(mint, timeframe='minute', aggregate=1):
    validate_mint(mint)
    allowed = {'minute': {1, 5, 15}, 'hour': {1, 4, 12}, 'day': {1}}
    if timeframe not in allowed or aggregate not in allowed[timeframe]:
        raise ValueError('Unsupported timeframe or aggregate')
    seconds = {'minute': 60, 'hour': 3600, 'day': 86400}[timeframe] * aggregate
    root = 'https://api.geckoterminal.com/api/v2/networks/solana'
    async with httpx.AsyncClient(timeout=25, headers={'Accept': 'application/json'}) as client:
        response = await client.get(root + '/tokens/' + mint + '/pools', params={'page': 1})
        response.raise_for_status()
        pools = response.json()['data']
        selected = None
        for pool in pools:
            relations = pool['relationships']
            for side in ('base', 'quote'):
                if relations[side + '_token']['data']['id'] == 'solana_' + mint:
                    selected = (pool['attributes']['address'], side)
                    break
            if selected:
                break
        if not selected:
            raise ValueError('No identified pool available for this mint')
        pool, side = selected
        validate_mint(pool)
        response = await client.get(root + '/pools/' + pool + '/ohlcv/' + timeframe,
                                    params={'aggregate': aggregate, 'limit': 100, 'currency': 'usd', 'token': side})
        response.raise_for_status()
        rows = response.json()['data']['attributes']['ohlcv_list']
    now = time.time()
    # In-progress candles cannot be evaluated as closed-bar patterns.
    closed = sorted([r for r in rows if r[0] + seconds <= now], key=lambda r: r[0])
    features = candle_features(closed)
    age = now - (closed[-1][0] + seconds)
    return {'source': 'geckoterminal', 'mint': mint, 'pool': pool, 'token_side': side, 'currency': 'USD',
            'fetched_at': now, 'timeframe': timeframe, 'aggregate': aggregate, 'last_close_age_seconds': age,
            'stale': age > 2 * seconds, 'ohlcv': closed, 'features': features}
