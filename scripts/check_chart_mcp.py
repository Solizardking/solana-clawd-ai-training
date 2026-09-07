#!/usr/bin/env python3
"""Discover catalog tool schemas without disclosing credentials or invoking tools."""
import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chart_agent.credentials import load_solgpt_env
from chart_agent.tools import SolGptBridge

if __name__ == '__main__':
    if len(sys.argv) > 1:
        load_solgpt_env(sys.argv[1])
    bridge = SolGptBridge()
    try:
        asyncio.run(bridge.discover())
    except Exception as exc:
        print(json.dumps({'connected': False, 'error': type(exc).__name__,
                          'http_status': getattr(getattr(exc, 'response', None), 'status_code', None)}))
        raise SystemExit(1)
    report = {'connected': True, 'connected_count': len(bridge.available), 'tools': sorted(bridge.available),
              'missing': sorted(set(bridge.contract) - set(bridge.available))}
    Path('outputs/chart-agent/mcp-discovery.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
