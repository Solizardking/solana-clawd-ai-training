#!/usr/bin/env python3
"""
Signal Discovery API Server — FastAPI backend for the dashboard.

Exposes the Quantitative Signal Discovery Agent and strategy runner
over HTTP so the React dashboard can poll them.

Start:
    cd ai-training/nvidia/blueprints/signal-discovery
    uvicorn server:app --reload --port 8765

Endpoints:
    GET  /api/health
    GET  /api/signals/{market}?timeframe=1h
    POST /api/scan               body: {"markets":["SOL","BTC"],"timeframe":"1h"}
    GET  /api/report             latest scan report from data/
    POST /api/strategy/launch    body: {"market":"SOL","direction":"long","budget":200,"mode":"paper"}
    GET  /api/strategy/active    list of running strategies
    POST /api/strategy/finalize  body: {"market":"SOL"}
    GET  /api/evolution          last N rows from data/strategy_evolution.jsonl
    GET  /api/training/status    CPT/SFT data counts + output checkpoint state
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────────────

HERE  = Path(__file__).parent
# HERE = .../solana-clawd/ai-training/nvidia/blueprints/signal-discovery
# parents[3] = solana-clawd repo root
ROOT  = HERE.parents[3]
DATA  = ROOT / "ai-training" / "data"
BLUEPRINT_DATA = HERE / "data"
DATA_SOURCES = {
    "blueprint": BLUEPRINT_DATA,
    "training": DATA,
}
FRONTEND = HERE / "frontend"

sys.path.insert(0, str(HERE))

from signals import scan_all, score_signals
from quantitative_signal_agent import (
    QuantitativeSignalAgent, detect_regime, StrategyLauncher,
    AccuracyTracker, build_report, SignalDiscovery,
)

# ── Shared state (single process) ─────────────────────────────────────────────

_agent = QuantitativeSignalAgent(use_llm=False, budget_usdc=200.0, strategy_mode="paper")
_launcher: StrategyLauncher = _agent._launcher

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Clawd Signal Discovery API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=FRONTEND), name="assets")


# ── Models ────────────────────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    markets: list[str] = ["SOL", "BTC", "ETH", "JUP", "JTO"]
    timeframe: str = "1h"


class LaunchRequest(BaseModel):
    market: str
    direction: str = "long"
    budget: float = 200.0
    mode: str = "paper"
    timeframe: str = "1h"


class FinalizeRequest(BaseModel):
    market: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _json_preview(path: Path, max_chars: int = 1800) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            with path.open() as f:
                text = json.dumps(json.load(f), indent=2)
        elif suffix == ".jsonl":
            rows = []
            with path.open() as f:
                for _, line in zip(range(5), f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        rows.append(line)
            text = json.dumps(rows, indent=2)
        elif suffix in {".md", ".txt", ".csv"}:
            with path.open(errors="replace") as f:
                text = f.read(max_chars + 1)
        else:
            return ""
    except Exception as exc:
        return f"preview unavailable: {exc}"

    if len(text) > max_chars:
        return text[:max_chars] + "\n..."
    return text


def _record_count(path: Path, cap: int = 250_000) -> tuple[int | None, bool]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            count = 0
            with path.open() as f:
                for count, _ in enumerate(f, start=1):
                    if count >= cap:
                        return count, True
            return count, False
        if suffix == ".json":
            with path.open() as f:
                data = json.load(f)
            if isinstance(data, list):
                return len(data), False
            if isinstance(data, dict):
                for key in ("discoveries", "rows", "records", "items", "data"):
                    rows = data.get(key)
                    if isinstance(rows, list):
                        return len(rows), False
                return len(data), False
    except Exception:
        return None, False
    return None, False


def _report_path() -> Path | None:
    for base in (BLUEPRINT_DATA, DATA):
        path = base / "signal_discovery_report.json"
        if path.exists():
            return path
    return None


def _signal_results_to_dict(market: str, timeframe: str = "1h") -> dict:
    results = scan_all(market, timeframe)
    direction, strength = score_signals(results)
    rs = detect_regime(results)
    signals = [
        {
            "name": s.name,
            "direction": s.direction,
            "strength": round(s.strength, 4),
            "reason": s.reason,
        }
        for s in results
    ]
    return {
        "timestamp": _now(),
        "market": market,
        "timeframe": timeframe,
        "signals": signals,
        "composite_direction": direction,
        "composite_strength": round(strength, 4),
        "regime": rs.regime,
        "regime_atr_pct": round(rs.atr_pct, 4),
        "regime_adx": round(rs.adx, 2),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

VULCAN_AVAILABLE = shutil.which("vulcan") is not None


@app.get("/")
def dashboard():
    return FileResponse(FRONTEND / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(FRONTEND / "favicon.svg", media_type="image/svg+xml")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "timestamp": _now(),
        "agent": "signal-discovery-v2",
        "vulcan": VULCAN_AVAILABLE,
        "env": os.environ.get("RENDER_SERVICE_NAME", "local"),
    }


@app.get("/api/status")
def status():
    return {
        "timestamp": _now(),
        "vulcan_available": VULCAN_AVAILABLE,
        "vulcan_path": shutil.which("vulcan"),
        "data_dir": str(DATA),
        "data_sources": {name: str(path) for name, path in DATA_SOURCES.items()},
        "python": sys.version,
        "env": {
            "RENDER": os.environ.get("RENDER", ""),
            "SERVICE": os.environ.get("RENDER_SERVICE_NAME", "local"),
        },
    }


@app.get("/api/signals/{market}")
def get_signals(market: str, timeframe: str = "1h"):
    if not VULCAN_AVAILABLE:
        raise HTTPException(status_code=503, detail="vulcan CLI not available")
    try:
        return _signal_results_to_dict(market.upper(), timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scan")
def scan(req: ScanRequest):
    results = []
    for market in req.markets:
        try:
            r = _signal_results_to_dict(market.upper(), req.timeframe)
            # Rule-based verdict
            thresh = _agent._tracker.get_adapted_thresholds(market, r["regime"])
            direction = r["composite_direction"]
            strength  = r["composite_strength"]
            if strength > thresh.get("min_strength", 0.4) and direction != "neutral":
                verdict, confidence = "enter", round(strength, 2)
            else:
                verdict, confidence = "hold", 0.0
            r["verdict"] = verdict
            r["confidence"] = confidence
            r["active_strategy"] = _launcher._active.get(market.upper())
            results.append(r)
        except Exception as e:
            results.append({"market": market.upper(), "error": str(e), "timestamp": _now()})
    return {
        "timestamp": _now(),
        "markets": results,
        "n_markets": len(results),
        "active_strategies": list(_launcher._active.keys()),
    }


@app.get("/api/report")
def get_report():
    report_path = _report_path()
    if report_path is None:
        return {"error": "no report yet", "hint": "POST /api/scan first"}
    with report_path.open() as f:
        report = json.load(f)
    report["_source_path"] = str(report_path)
    return report


@app.get("/api/data/catalog")
def data_catalog():
    files = []
    sources = {}
    for source_name, base in DATA_SOURCES.items():
        sources[source_name] = str(base)
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            stat = path.stat()
            record_count, records_truncated = _record_count(path)
            files.append({
                "source": source_name,
                "path": str(path.relative_to(base)),
                "absolute_path": str(path),
                "extension": path.suffix.lower().lstrip(".") or "file",
                "size_bytes": stat.st_size,
                "modified": _iso_from_timestamp(stat.st_mtime),
                "record_count": record_count,
                "records_truncated": records_truncated,
                "preview": _json_preview(path),
            })

    return {
        "timestamp": _now(),
        "sources": sources,
        "files": files,
        "count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
    }


@app.post("/api/strategy/launch")
def launch_strategy(req: LaunchRequest):
    market = req.market.upper()
    _launcher._mode = req.mode
    _launcher._budget = req.budget

    results = scan_all(market, req.timeframe)
    rs = detect_regime(results)
    thresh = _agent._tracker.get_adapted_thresholds(market, rs.regime)

    signals_dict = [
        {"name": s.name, "direction": s.direction, "strength": s.strength}
        for s in results
    ]

    config = _launcher.build_ta_config(
        market=market,
        direction=req.direction,
        signals=signals_dict,
        regime=rs.regime,
        regime_state=rs,
        thresholds=thresh,
        timeframe=req.timeframe,
    )

    run_id = _launcher.launch(market, config, req.timeframe)
    return {
        "market": market,
        "run_id": run_id,
        "config": config,
        "regime": rs.regime,
        "timestamp": _now(),
    }


@app.get("/api/strategy/active")
def active_strategies():
    strategies = []
    for market, run_id in _launcher._active.items():
        st = _launcher.status(market)
        strategies.append({"market": market, "run_id": run_id, "status": st})
    return {
        "timestamp": _now(),
        "active": strategies,
        "count": len(strategies),
    }


@app.post("/api/strategy/finalize")
def finalize_strategy(req: FinalizeRequest):
    ok = _launcher.finalize(req.market.upper())
    return {"market": req.market.upper(), "ok": ok, "timestamp": _now()}


@app.get("/api/evolution")
def evolution_log(limit: int = 100):
    log_path = BLUEPRINT_DATA / "strategy_evolution.jsonl"
    if not log_path.exists():
        log_path = DATA / "strategy_evolution.jsonl"
    if not log_path.exists():
        return {"rows": [], "total": 0, "error": "no evolution log yet"}
    rows = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    rows = rows[-limit:]
    return {"rows": rows, "total": len(rows), "timestamp": _now()}


@app.get("/api/training/status")
def training_status():
    def count_lines(p: Path) -> int:
        if not p.exists():
            return 0
        return sum(1 for _ in p.open())

    cpt_path = DATA / "tx_foundation_cpt.jsonl"
    sft_path = DATA / "solana_clawd_merged.jsonl"
    jupiter_path = DATA / "jupiter_txs.jsonl"
    eval_path = DATA / "tx_foundation_eval.json"

    cpt_checkpoint = ROOT / "outputs" / "solana-tx-foundation-1.5b" / "cpt"
    sft_checkpoint = ROOT / "outputs" / "solana-tx-foundation-1.5b" / "sft"

    eval_data = {}
    if eval_path.exists():
        try:
            with eval_path.open() as f:
                eval_data = json.load(f)
        except Exception:
            pass

    return {
        "timestamp": _now(),
        "data": {
            "cpt_records":     count_lines(cpt_path),
            "sft_records":     count_lines(sft_path),
            "jupiter_records": count_lines(jupiter_path),
            "cpt_data_mb":     round(cpt_path.stat().st_size / 1e6, 2) if cpt_path.exists() else 0,
        },
        "checkpoints": {
            "cpt_done": cpt_checkpoint.exists(),
            "sft_done": sft_checkpoint.exists(),
            "cpt_path": str(cpt_checkpoint) if cpt_checkpoint.exists() else None,
            "sft_path": str(sft_checkpoint) if sft_checkpoint.exists() else None,
        },
        "eval": eval_data,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8765, reload=True)
