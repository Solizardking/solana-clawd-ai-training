"""Bounded live feed with reconnect and explicit staleness."""
import asyncio
import json
import os
import time
from collections import deque


class Tape:
    def __init__(self):
        self.events = deque(maxlen=200)
        self.connected = False
        self.last_received = None
        self.error = None

    def snapshot(self):
        age = time.time() - self.last_received if self.last_received else None
        return dict(source='clawd-ws', connected=self.connected, age_seconds=age,
                    stale=age is None or age > 30 or not self.connected,
                    error=self.error, events=list(self.events)[-10:])

    async def run(self):
        from websockets.asyncio.client import connect
        delay = 1
        while True:
            try:
                async with connect(os.getenv('CLAWD_WS_URL', 'wss://clawd-ws.fly.dev/ws'),
                                   open_timeout=15, ping_interval=20, ping_timeout=20, max_size=256_000) as ws:
                    self.connected, self.error = True, None
                    delay = 1
                    async for raw in ws:
                        try:
                            event = json.loads(raw)
                        except (ValueError, TypeError):
                            continue
                        if not isinstance(event, dict) or event.get('type') not in {'token-launch', 'trade', 'pump', 'pump-trade', 'new-token', 'token-trade'}:
                            continue
                        self.last_received = time.time()
                        self.events.append({'received_at': self.last_received, 'event': event})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.error = type(exc).__name__
            finally:
                self.connected = False
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
