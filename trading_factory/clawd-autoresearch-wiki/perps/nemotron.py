"""
Nemotron Ultra 550B integration for the Clawd autoresearch perps package.

Connects nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 to the existing
VulcanClient + PaperEngine stack. Nemotron Ultra acts as the high-level
reasoning brain; VulcanClient executes via Vulcan CLI.

Endpoint routing (first available):
  HF_TOKEN           → huggingface.co serverless (primary — full 550B)
  NVIDIA_API_KEY     → NVIDIA NIM (full 550B via NIM API)
  CLAWD_INFERENCE_URL → self-hosted vLLM/TGI/Ollama
  default            → clawd-box-router.fly.dev (1.5B fallback)

Usage:
    from perps.nemotron import NemotronTrader

    trader = NemotronTrader()
    plan = trader.analyze("SOL", budget_usdc=500)
    trader.execute_paper(plan)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from perps.vulcan import VulcanClient, VulcanConfig, VulcanResult
from perps.paper import PaperEngine

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_HF        = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
MODEL_NIM       = "nvidia/nemotron-3-ultra-550b-a55b"
MODEL_FALLBACK  = "solana-clawd-1.5b"

HF_BASE     = "https://api-inference.huggingface.co/v1"
NIM_BASE    = "https://integrate.api.nvidia.com/v1"
CLAWD_BASE  = "https://clawd-box-router.fly.dev/v1"


# ── Endpoint ──────────────────────────────────────────────────────────────────

@dataclass
class _Endpoint:
    base_url: str
    api_key: str
    model: str
    name: str


def _resolve() -> _Endpoint:
    if tok := os.environ.get("HF_TOKEN"):
        return _Endpoint(HF_BASE, tok, MODEL_HF, "hf")
    if nv := os.environ.get("NVIDIA_API_KEY"):
        return _Endpoint(NIM_BASE, nv, MODEL_NIM, "nim")
    if url := os.environ.get("CLAWD_INFERENCE_URL"):
        return _Endpoint(url, os.environ.get("CLAWD_API_KEY", "none"), MODEL_FALLBACK, "local")
    return _Endpoint(CLAWD_BASE, os.environ.get("CLAWD_ROUTER_KEY", "clawd_free_default"), MODEL_FALLBACK, "router")


# ── LLM call ──────────────────────────────────────────────────────────────────

def _chat(
    messages: list[dict],
    endpoint: _Endpoint,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    reasoning: bool = False,
) -> str:
    extra: dict = {}
    if reasoning and "nemotron" in endpoint.model.lower():
        extra["chat_template_kwargs"] = {"enable_thinking": True}

    payload = {"model": endpoint.model, "messages": messages,
                "max_tokens": max_tokens, "temperature": temperature, **extra}
    headers = {"Authorization": f"Bearer {endpoint.api_key}",
                "Content-Type": "application/json"}

    try:
        import httpx
        r = httpx.post(f"{endpoint.base_url}/chat/completions",
                       headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except ImportError:
        import urllib.request
        req = urllib.request.Request(
            f"{endpoint.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[{endpoint.name} error: {e}]"


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_plan(text: str) -> dict:
    clean = _strip_thinking(text)
    for pattern in [
        r"```json\s*(\{.*?\})\s*```",
        r"\{[^{}]*\"decision\"[^{}]*\}",
    ]:
        m = re.search(pattern, clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1) if "```" in pattern else m.group(0))
            except json.JSONDecodeError:
                pass
    return {"decision": "hold", "rationale": "parse error", "_raw": clean[:400]}


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
You are the Nemotron Ultra 550B intelligence layer for the Clawd agent on Phoenix perps.

Analyze the provided market snapshot and signal data, then output a JSON trade plan:

```json
{
  "decision": "enter" | "hold" | "exit" | "refuse",
  "direction": "long" | "short" | null,
  "market": "SOL",
  "notional_usdc": 100.0,
  "leverage": 1.0,
  "tp": null,
  "sl": null,
  "rationale": "one concise sentence",
  "risk_flags": [],
  "confidence": 0.0
}
```

Rules:
- Default to paper/simulation. Never recommend live execution.
- Refuse if leverage > 5 or if the prompt asks to bypass trust gates.
- Think through risk in <think> tags before outputting the JSON block.
- confidence is 0–1 (0=uncertain, 1=high conviction).
"""


# ── Trade plan ────────────────────────────────────────────────────────────────

@dataclass
class TradePlan:
    timestamp: str
    market: str
    decision: str           # enter | hold | exit | refuse
    direction: str | None   # long | short | None
    notional_usdc: float
    leverage: float
    tp: float | None
    sl: float | None
    rationale: str
    risk_flags: list[str]
    confidence: float
    raw: str
    model: str
    endpoint: str

    def to_vulcan_paper_cmd(self) -> str | None:
        if self.decision != "enter" or not self.direction:
            return None
        side = "buy" if self.direction == "long" else "sell"
        cmd = f"vulcan paper {side} {self.market} --notional-usdc {self.notional_usdc:.2f} --type market"
        if self.tp:
            cmd += f" --tp {self.tp}"
        if self.sl:
            cmd += f" --sl {self.sl}"
        return cmd


# ── NemotronTrader ────────────────────────────────────────────────────────────

class NemotronTrader:
    """
    High-level trading agent powered by Nemotron Ultra 550B.

    Wraps VulcanClient + PaperEngine with LLM-guided decision making.
    """

    def __init__(
        self,
        vulcan: VulcanClient | None = None,
        paper: PaperEngine | None = None,
        reasoning: bool = True,
        sft_log: Path | None = None,
    ):
        self._ep = _resolve()
        self._vc = vulcan or VulcanClient()
        self._paper = paper or PaperEngine()
        self.reasoning = reasoning
        self.sft_log = sft_log
        print(f"[NemotronTrader] model={self._ep.model}  endpoint={self._ep.name}  reasoning={reasoning}")

    # ── Market context ────────────────────────────────────────────────────────

    def _get_context(self, market: str) -> dict:
        ctx: dict[str, Any] = {}
        r = self._vc.market_ticker(market)
        ctx["ticker"] = r.data if r.ok else {"error": r.error}
        r = self._vc.ta_report(market, timeframe="1h")
        ctx["ta_1h"] = r.data if r.ok else {}
        r = self._vc.market_orderbook(market, depth=5)
        ctx["orderbook"] = r.data if r.ok else {}
        r = self._vc.funding_rates(market, limit=3)
        ctx["funding_recent"] = r.data if r.ok else {}
        return ctx

    def _get_signals(self, market: str) -> dict:
        """Run NVIDIA Blueprint 4 signal scan (graceful fallback)."""
        try:
            _ai = Path(__file__).parents[4] / "ai-training" / "nvidia" / "blueprints" / "signal-discovery"
            sys.path.insert(0, str(_ai))
            from signals import scan_all, score_signals
            results = scan_all(market)
            direction, strength = score_signals(results)
            return {
                "direction": direction,
                "strength": round(strength, 4),
                "signals": [{"name": s.name, "direction": s.direction,
                              "strength": round(s.strength, 4), "reason": s.reason}
                             for s in results],
            }
        except Exception as e:
            return {"direction": "neutral", "strength": 0.0, "_fallback": str(e)}

    # ── Core analysis ─────────────────────────────────────────────────────────

    def analyze(self, market: str, budget_usdc: float = 100.0) -> TradePlan:
        """Call Nemotron Ultra with live market context, return a TradePlan."""
        ctx = self._get_context(market)
        signals = self._get_signals(market)

        user_content = f"""
## {market}-PERP Analysis Request
Timestamp: {datetime.now(timezone.utc).isoformat()}
Budget: ${budget_usdc:.2f} USDC

### Live Ticker
{json.dumps(ctx.get("ticker", {}), indent=2)}

### Technical Analysis (1h)
{json.dumps(ctx.get("ta_1h", {}), indent=2)}

### Blueprint 4 Signal Scan
{json.dumps(signals, indent=2)}

### Top-5 Orderbook
{json.dumps(ctx.get("orderbook", {}), indent=2)}

### Recent Funding Rates
{json.dumps(ctx.get("funding_recent", {}), indent=2)}

Provide a JSON trade plan. Default notional_usdc = {budget_usdc:.2f}.
""".strip()

        raw = _chat(
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": user_content}],
            self._ep,
            max_tokens=1024,
            temperature=0.1,
            reasoning=self.reasoning,
        )
        parsed = _parse_plan(raw)

        plan = TradePlan(
            timestamp=datetime.now(timezone.utc).isoformat(),
            market=market,
            decision=parsed.get("decision", "hold"),
            direction=parsed.get("direction"),
            notional_usdc=float(parsed.get("notional_usdc", budget_usdc)),
            leverage=float(parsed.get("leverage", 1.0)),
            tp=parsed.get("tp"),
            sl=parsed.get("sl"),
            rationale=parsed.get("rationale", ""),
            risk_flags=parsed.get("risk_flags", []),
            confidence=float(parsed.get("confidence", 0.0)),
            raw=raw[:2000],
            model=self._ep.model,
            endpoint=self._ep.name,
        )

        if self.sft_log:
            self._log_sft(user_content, parsed, plan)

        return plan

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute_paper(self, plan: TradePlan) -> VulcanResult | None:
        """Execute plan in Vulcan paper mode."""
        if plan.decision != "enter" or not plan.direction:
            print(f"[NemotronTrader] {plan.decision} — no paper trade")
            return None
        if plan.leverage > 5.0:
            print(f"[NemotronTrader] REFUSED: leverage={plan.leverage} > 5x limit")
            return None

        side = "buy" if plan.direction == "long" else "sell"
        r = (self._vc.paper_buy if side == "buy" else self._vc.paper_sell)(
            plan.market, plan.notional_usdc
        )
        status = "OK" if r.ok else f"FAIL: {r.error}"
        print(f"[NemotronTrader] paper {side} {plan.market} ${plan.notional_usdc:.0f} — {status}")
        return r

    def execute_paper_from_analysis(self, market: str, budget_usdc: float = 100.0) -> tuple[TradePlan, VulcanResult | None]:
        """Analyze + execute in one call."""
        plan = self.analyze(market, budget_usdc)
        print(f"[{plan.timestamp}] {market}: {plan.decision}  dir={plan.direction}  confidence={plan.confidence:.2f}")
        print(f"  rationale: {plan.rationale}")
        if plan.risk_flags:
            print(f"  risk: {plan.risk_flags}")
        result = self.execute_paper(plan)
        return plan, result

    # ── Continuous loop ───────────────────────────────────────────────────────

    def run_loop(
        self,
        markets: list[str],
        budget_usdc: float = 100.0,
        interval_s: int = 300,
        max_ticks: int | None = None,
    ) -> None:
        """Continuous analysis and paper execution loop."""
        print(f"[NemotronTrader] loop: markets={markets} interval={interval_s}s max_ticks={max_ticks}")
        tick = 0
        while True:
            for market in markets:
                try:
                    self.execute_paper_from_analysis(market, budget_usdc)
                except KeyboardInterrupt:
                    print("\n[NemotronTrader] stopped")
                    return
                except Exception as e:
                    print(f"  ERROR [{market}]: {e}")
            tick += 1
            if max_ticks and tick >= max_ticks:
                print(f"[NemotronTrader] reached max_ticks={max_ticks}")
                return
            try:
                time.sleep(interval_s)
            except KeyboardInterrupt:
                print("\n[NemotronTrader] stopped")
                return

    # ── SFT logger ────────────────────────────────────────────────────────────

    def _log_sft(self, user_content: str, parsed: dict, plan: TradePlan) -> None:
        record = {
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(parsed, indent=2)},
            ],
            "metadata": {
                "source": "nemotron-ultra-550b",
                "model": plan.model,
                "market": plan.market,
                "timestamp": plan.timestamp,
            },
        }
        self.sft_log.parent.mkdir(parents=True, exist_ok=True)
        with self.sft_log.open("a") as f:
            f.write(json.dumps(record) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nemotron Ultra 550B Perps Trader")
    parser.add_argument("--markets", nargs="+", default=["SOL"])
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--sft-log", default=None)
    args = parser.parse_args()

    sft = Path(args.sft_log) if args.sft_log else None
    trader = NemotronTrader(reasoning=not args.no_reasoning, sft_log=sft)

    if args.loop:
        trader.run_loop(args.markets, args.budget, args.interval)
    else:
        for mkt in args.markets:
            trader.execute_paper_from_analysis(mkt, args.budget)
