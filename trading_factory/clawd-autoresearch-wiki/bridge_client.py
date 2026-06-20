"""
ClawdBot Bridge Client — Python → TypeScript Agent Communication

Reports training results to the ClawdBot bridge server so the agent
can incorporate Python training insights into its memory vault.

Usage:
    from bridge_client import ClawdBotBridge

    bridge = ClawdBotBridge()
    bridge.report_result(metric=0.145, description="Lowered LR to 3e-4")
    bridge.chat("What was the best val_bpb so far?")
    bridge.trigger_research(max_experiments=5)
"""

import os
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any


BRIDGE_URL = os.environ.get("CLAWDBOT_BRIDGE_URL", "http://localhost:3777")


class ClawdBotBridge:
    """Client for communicating with the ClawdBot TypeScript bridge server."""

    def __init__(self, base_url: str = BRIDGE_URL):
        self.base_url = base_url.rstrip("/")

    def _post(self, endpoint: str, data: dict) -> dict:
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.URLError as e:
            return {"error": str(e), "bridge_available": False}

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.URLError as e:
            return {"error": str(e), "bridge_available": False}

    # ── Agent Operations ─────────────────────────────────────────

    def chat(self, message: str) -> str:
        """Chat with the ClawdBot agent."""
        result = self._post("/api/agent/chat", {"message": message})
        return result.get("response", result.get("error", "No response"))

    def remember(self, content: str, category: str = "inbox") -> dict:
        """Store content in the agent's vault."""
        return self._post("/api/agent/remember", {
            "content": content,
            "opts": {"category": category},
        })

    def recall(self, query: str) -> list:
        """Query the agent's vault memory."""
        result = self._get("/api/agent/recall", {"q": query})
        return result.get("memories", [])

    def observe(self) -> dict:
        """Trigger an OODA observation cycle."""
        return self._post("/api/agent/observe", {})

    def status(self) -> dict:
        """Get agent status and token usage."""
        return self._get("/api/agent/status")

    # ── Python Integration ───────────────────────────────────────

    def report_result(
        self,
        metric: float,
        description: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Report a Python training result to the agent."""
        return self._post("/api/python/result", {
            "metric": metric,
            "description": description,
            "params": params or {},
        })

    def get_results(self) -> dict:
        """Get all Python training results."""
        return self._get("/api/python/results")

    # ── Research ─────────────────────────────────────────────────

    def trigger_research(self, max_experiments: int = 10) -> dict:
        """Start the TS auto-research loop."""
        return self._post("/api/agent/research", {
            "maxExperiments": max_experiments,
        })

    # ── Automation ───────────────────────────────────────────────

    def run_full_cycle(self) -> dict:
        """Trigger full automation: research + training + analysis."""
        return self._post("/api/automate/full", {})

    def health(self) -> dict:
        """Check bridge server health."""
        return self._get("/api/health")

    # ── Convenience ──────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if the bridge server is reachable."""
        result = self.health()
        return result.get("status") == "ok"


# ── Auto-report hook for train.py integration ──────────────────────

def auto_report(val_bpb: float, step: int = 0, description: str = ""):
    """
    Call this from train.py to automatically report results.
    
    Usage in train.py:
        from bridge_client import auto_report
        auto_report(val_bpb=0.145, step=1000, description="baseline run")
    """
    bridge = ClawdBotBridge()
    if bridge.is_available():
        result = bridge.report_result(
            metric=val_bpb,
            description=f"Step {step}: {description}" if step else description,
            params={"val_bpb": val_bpb, "step": step},
        )
        print(f"📡 Reported to ClawdBot: val_bpb={val_bpb:.4f}")
        return result
    else:
        print("⚠️  ClawdBot bridge not available")
        return None


if __name__ == "__main__":
    # Quick test
    bridge = ClawdBotBridge()
    
    if bridge.is_available():
        print("✅ Bridge server is running")
        print(json.dumps(bridge.health(), indent=2))
    else:
        print("❌ Bridge server not available. Start it with: npm run bridge")
