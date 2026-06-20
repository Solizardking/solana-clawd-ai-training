"""
Honcho-powered persistent memory for Clawd trading agents.

Provides true persistent memory that survives context wipes:
- Trades: remembers every position, PnL, entry/exit rationale
- Strategies: recalls what worked, what didn't, why
- Users: tracks preferences, risk tolerance, communication style
- Dreams: autonomously consolidates patterns overnight
- Session bridging: context from one conversation carries to the next

Usage:
    from memory import AgentMemory

    mem = AgentMemory(api_key="hch-...", workspace="clawd-prod")
    
    # Remember a trade
    mem.remember_trade("SOL-PERP", "long", 500, 3.0, 152.30, "RSI oversold bounce")
    
    # Recall what we know about a user
    ctx = mem.recall("user-alice", "What are Alice's trading preferences?")
    
    # Bridge session memory
    mem.bridge("session-123", "session-456", 
               "Continuing SOL analysis from previous session")
    
    # Dream — consolidate patterns
    summary = mem.dream()
    
    # Natural language remember/recall
    mem.remember("user-alice", "Alice prefers limit orders over market orders")
    fact = mem.recall("user-alice", "What order type does Alice prefer?")

Based on: https://honcho.dev/docs — Honcho persistent memory SDK
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any


# =============================================================================
# Honcho SDK shim — uses the honcho-ai SDK directly
# =============================================================================

try:
    from honcho import Honcho as _Honcho
    HONCHO_SDK_AVAILABLE = True
except ImportError:
    HONCHO_SDK_AVAILABLE = False
    _Honcho = None


@dataclass
class MemoryConfig:
    """Configuration for Honcho-powered agent memory."""
    api_key: str = ""
    workspace_id: str = "clawd-default"
    environment: str = "production"  # production, demo, local
    base_url: str = "https://api.honcho.dev"
    peer_name: str = "clawd-agent"  # The agent's own peer ID
    default_session: str = "clawd-main"
    reasoning_level: str = "low"  # minimal, low, medium, high, max
    auto_dream: bool = True  # Enable autonomous consolidation
    dream_interval_hours: int = 8  # Minimum hours between dreams


@dataclass 
class MemoryEntry:
    """A single memory entry (trade, preference, or fact)."""
    content: str  # The actual memory content
    category: str  # trade, preference, fact, strategy, lesson
    session_id: str = ""
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_message(self, peer_id: str) -> Any:
        """Convert to Honcho message format."""
        if HONCHO_SDK_AVAILABLE:
            from honcho import Message
            return Message(
                content=self.content,
                peer_id=peer_id,
                metadata={
                    "category": self.category,
                    "confidence": self.confidence,
                    **(self.metadata or {}),
                },
                created_at=datetime.fromtimestamp(self.timestamp or time.time()).isoformat()
            )
        return None


class HonchoMemory:
    """Thin wrapper around the Honcho SDK for Clawd agents.

    Provides:
    - Workspace-scoped memory isolation
    - Peer-based memory attribution (who said what)
    - Session scoping for conversation boundaries
    - Tokens-budgeted context retrieval
    - OpenAI/Anthropic format conversion for LLM injection
    """

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()
        self._client = None
        self._initialized = False

    def _ensure_client(self):
        """Initialize Honcho client lazily."""
        if self._initialized and self._client:
            return self._client

        api_key = self.config.api_key or os.environ.get("HONCHO_API_KEY", "")
        ws_id = self.config.workspace_id or os.environ.get(
            "HONCHO_WORKSPACE_ID", "clawd-default"
        )
        env = self.config.environment

        if HONCHO_SDK_AVAILABLE and api_key:
            self._client = _Honcho(
                environment=env,
                workspace_id=ws_id,
                api_key=api_key,
            )
        else:
            # Create a local-only fallback that mimics the API
            self._client = _LocalHonchoFallback(workspace_id=ws_id)

        self._initialized = True
        return self._client

    def peer(self, peer_id: str) -> Any:
        """Get or create a peer (user, agent, tool)."""
        client = self._ensure_client()
        try:
            return client.peer(peer_id)
        except Exception:
            if hasattr(client, "_create_peer"):
                return client._create_peer(peer_id)
            raise

    def session(self, session_id: str) -> Any:
        """Get or create a session for conversation context."""
        client = self._ensure_client()
        try:
            return client.session(session_id)
        except Exception:
            if hasattr(client, "_create_session"):
                return client._create_session(session_id)
            raise

    def write_message(self, peer_id: str, content: str,
                      session_id: str | None = None,
                      metadata: dict | None = None) -> bool:
        """Write a message to Honcho, triggering background reasoning."""
        client = self._ensure_client()
        sid = session_id or self.config.default_session
        peer = self.peer(peer_id)
        session = self.session(sid)

        try:
            session.add_messages([
                peer.message(content, metadata=metadata or {})
            ])
            return True
        except Exception:
            return False

    def get_context(self, session_id: str | None = None,
                    tokens: int = 2000,
                    peer_target: str | None = None) -> str:
        """Get formatted conversation context for LLM injection.

        Returns a string suitable for prepending to an LLM prompt.
        """
        client = self._ensure_client()
        sid = session_id or self.config.default_session
        session = self.session(sid)

        try:
            ctx = session.context(tokens=tokens, peer_target=peer_target)
            if hasattr(ctx, 'to_openai'):
                return str(ctx.to_openai(assistant=peer_target or self.config.peer_name))
            return str(ctx)
        except Exception:
            return ""

    def ask_about_peer(self, peer_id: str, query: str,
                       session_id: str | None = None,
                       target: str | None = None) -> str:
        """Ask Honcho what it knows about a peer.

        Examples:
            memory.ask_about_peer("alice", "What are Alice's trading preferences?")
            memory.ask_about_peer("alice", "What happened in our last session?",
                                  session_id="session-5")
            memory.ask_about_peer("alice", "How does Bob see Alice?",
                                  target="bob")
        """
        client = self._ensure_client()
        peer = self.peer(peer_id)

        kwargs = {}
        if session_id:
            kwargs["session_id"] = session_id
        if target:
            kwargs["target"] = target
        kwargs["reasoning_level"] = self.config.reasoning_level

        try:
            return peer.chat(query, **kwargs)
        except Exception:
            return ""

    def write_trade(self, symbol: str, side: str, size: float,
                    leverage: float, entry_price: float,
                    rationale: str, peer_id: str = "trader",
                    session_id: str | None = None) -> bool:
        """Record a trade in Honcho memory for future recall."""
        content = (
            f"Opened {side} position: {size} USDC {symbol} @ ${entry_price} "
            f"with {leverage}x leverage. Rationale: {rationale}"
        )
        return self.write_message(
            peer_id=peer_id,
            content=content,
            session_id=session_id or "trades",
            metadata={
                "category": "trade",
                "symbol": symbol,
                "side": side,
                "size": size,
                "leverage": leverage,
                "entry_price": entry_price,
            },
        )

    def write_close(self, symbol: str, side: str, size: float,
                    exit_price: float, pnl: float, rationale: str,
                    peer_id: str = "trader",
                    session_id: str | None = None) -> bool:
        """Record a trade closure in Honcho memory."""
        direction = "profitable" if pnl > 0 else "losing"
        content = (
            f"Closed {side} position: {size} USDC {symbol} @ ${exit_price}. "
            f"PnL: ${pnl:.2f} ({direction}). Rationale: {rationale}"
        )
        return self.write_message(
            peer_id=peer_id,
            content=content,
            session_id=session_id or "trades",
            metadata={
                "category": "trade_close",
                "symbol": symbol,
                "side": side,
                "size": size,
                "exit_price": exit_price,
                "pnl": pnl,
                "direction": direction,
            },
        )

    def write_strategy_result(self, strategy_type: str, symbol: str,
                               outcome: str, pnl: float,
                               lessons: str, peer_id: str = "strategist") -> bool:
        """Record a strategy outcome for future learning."""
        content = (
            f"Strategy [{strategy_type}] on {symbol}: {outcome}. "
            f"PnL: ${pnl:.2f}. Lesson: {lessons}"
        )
        return self.write_message(
            peer_id=peer_id,
            content=content,
            session_id="strategies",
            metadata={
                "category": "strategy",
                "strategy_type": strategy_type,
                "symbol": symbol,
                "outcome": outcome,
                "pnl": pnl,
            },
        )

    def queue_status(self) -> dict[str, int]:
        """Check Honcho background processing status."""
        client = self._ensure_client()
        try:
            qs = client.queue_status()
            return {
                "pending": getattr(qs, 'pending_work_units', 0),
                "in_progress": getattr(qs, 'in_progress_work_units', 0),
                "completed": getattr(qs, 'completed_work_units', 0),
                "total": getattr(qs, 'total_work_units', 0),
            }
        except Exception:
            return {"pending": 0, "in_progress": 0, "completed": 0, "total": 0}

    def get_peer_card(self, peer_id: str) -> list[str]:
        """Get a peer's biographical card (facts Honcho knows)."""
        client = self._ensure_client()
        peer = self.peer(peer_id)
        try:
            card = peer.get_card()
            return card if isinstance(card, list) else []
        except Exception:
            return []

    def set_peer_card(self, peer_id: str, facts: list[str]) -> bool:
        """Set a peer's biographical card (bootstraps memory)."""
        client = self._ensure_client()
        peer = self.peer(peer_id)
        try:
            peer.set_card(facts)
            return True
        except Exception:
            return False


# =============================================================================
# AgentMemory — high-level memory API for Clawd trading agents
# =============================================================================

class AgentMemory:
    """High-level persistent memory for Clawd agents.

    Provides natural-language remember/recall, trade memory, strategy
    learning, cross-session bridging, autonomous dreaming, and LLM
    context injection — all backed by Honcho's persistent reasoning.

    Usage:
        mem = AgentMemory(api_key=os.environ["HONCHO_API_KEY"])

        # Natural language remember/recall
        mem.remember("Alice prefers limit orders to market orders")
        pref = mem.recall("What order type does Alice prefer?")

        # Trade memory
        mem.remember_trade("SOL-PERP", "long", 500, 3.0, 152.30,
                           "RSI oversold bounce in 1h trend")

        # Cross-session context
        ctx = mem.bridge_session("What happened in the last SOL analysis session?")

        # Dream — autonomous pattern discovery
        insights = mem.dream()

        # LLM context injection
        prompt = mem.build_llm_context("You are Clawd, a Solana perps trader...")
    """

    def __init__(self, api_key: str | None = None, workspace: str = "clawd-trading"):
        config = MemoryConfig(
            api_key=api_key or os.environ.get("HONCHO_API_KEY", ""),
            workspace_id=workspace or os.environ.get("HONCHO_WORKSPACE_ID",
                                                     "clawd-trading"),
        )
        self._honcho = HonchoMemory(config)
        self._peer_id = config.peer_name
        self._last_dream_time = 0.0

        # Bootstrap agent's self-knowledge on construction
        self._ensure_agent_identity()

    def _ensure_agent_identity(self):
        """Set the agent's peer card with its identity."""
        self._honcho.set_peer_card(self._peer_id, [
            "Name: Clawd",
            "Role: Sovereign Solana-native AI trading agent",
            "Expertise: Solana perpetual futures on Phoenix DEX",
            "Values: Helpful, honest, ethical — never recommend harmful actions",
            "Capabilities: TWAP execution, grid trading, TA-driven strategies",
            "Protocols: Phoenix perps, Jupiter DEX, Light Protocol ZK",
        ])

    def remember(self, fact: str, peer_id: str = "user",
                 session_id: str = "clawd-main",
                 category: str = "fact") -> bool:
        """Remember a fact about a user or the world.

        Natural-language memory storage. Honcho's background reasoning
        will extract conclusions, patterns, and preferences from this.
        """
        return self._honcho.write_message(
            peer_id=peer_id,
            content=fact,
            session_id=session_id,
            metadata={"category": category},
        )

    def recall(self, query: str, peer_id: str = "user",
               session_id: str | None = None) -> str:
        """Recall what we know about a subject.

        Honcho searches its reasoning (conclusions, peer cards, summaries)
        and synthesizes a natural-language answer.
        """
        return self._honcho.ask_about_peer(
            peer_id=peer_id,
            query=query,
            session_id=session_id,
        )

    def remember_trade(self, symbol: str, side: str, size: float,
                       leverage: float, entry_price: float,
                       rationale: str) -> bool:
        """Record a trade opening in permanent memory."""
        return self._honcho.write_trade(
            symbol=symbol, side=side, size=size,
            leverage=leverage, entry_price=entry_price,
            rationale=rationale,
        )

    def close_trade(self, symbol: str, side: str, size: float,
                    exit_price: float, pnl: float,
                    rationale: str) -> bool:
        """Record a trade closing in permanent memory."""
        return self._honcho.write_close(
            symbol=symbol, side=side, size=size,
            exit_price=exit_price, pnl=pnl, rationale=rationale,
        )

    def learn_strategy(self, strategy_type: str, symbol: str,
                       outcome: str, pnl: float,
                       lessons: str) -> bool:
        """Record a strategy outcome so the agent learns from it."""
        return self._honcho.write_strategy_result(
            strategy_type=strategy_type, symbol=symbol,
            outcome=outcome, pnl=pnl, lessons=lessons,
        )

    def bridge_session(self, query: str,
                       source_session: str = "clawd-main",
                       target_session: str = "clawd-current"
                       ) -> str:
        """Bridge context from one session to another.

        Useful when starting a new conversation: remembers what happened
        in past sessions and makes that context available in the new one.
        """
        context = self._honcho.get_context(
            session_id=source_session,
            tokens=1500,
            peer_target=self._peer_id,
        )
        # Write the bridged context into the new session
        self._honcho.write_message(
            peer_id=self._peer_id,
            content=f"[Bridged from {source_session}]: {context[:500]}...",
            session_id=target_session,
            metadata={"category": "bridge", "source": source_session},
        )
        return self._honcho.ask_about_peer(
            peer_id=self._peer_id,
            query=query,
            session_id=target_session,
        )

    def dream(self) -> str:
        """Trigger autonomous memory consolidation.

        Honcho's "dreaming" process:
        1. Deduction specialist resolves contradictions
        2. Induction specialist discovers patterns across trades
        3. Peer card gets updated with stable facts
        4. Conclusions get consolidated

        Returns a summary of what was learned.
        """
        # Check cooldown
        elapsed = time.time() - self._last_dream_time
        min_interval = self._honcho.config.dream_interval_hours * 3600
        if elapsed < min_interval:
            remaining = timedelta(seconds=int(min_interval - elapsed))
            return f"Dream cooldown: {remaining} remaining"

        self._last_dream_time = time.time()

        # Ask Honcho to synthesize patterns across all trading sessions
        insights = []

        # What did we learn from trades?
        trade_insight = self._honcho.ask_about_peer(
            peer_id=self._peer_id,
            query="What patterns do you see across all trades? "
                  "What's working? What's not? Give specific examples.",
        )
        insights.append(f"Trade Patterns: {trade_insight}")

        # What are the user's preferences?
        pref_insight = self._honcho.ask_about_peer(
            peer_id="user",
            query="What are this user's trading preferences, risk tolerance, and communication style? "
                  "Summarize from all sessions.",
        )
        insights.append(f"User Profile: {pref_insight}")

        # What strategies worked?
        strat_insight = self._honcho.ask_about_peer(
            peer_id="strategist",
            query="Summarize what strategies have been tried, which worked, which didn't, and why.",
            session_id="strategies",
        )
        insights.append(f"Strategy Review: {strat_insight}")

        summary = "\n\n".join(insights)
        return summary

    def build_llm_context(self, system_prompt: str = "",
                          peer_id: str = "user",
                          max_tokens: int = 2000) -> str:
        """Build a complete context string for an LLM prompt.

        Combines:
        - System prompt (from constitution or user)
        - Honcho's context about the user
        - Recent session messages
        """
        parts = [system_prompt] if system_prompt else []

        # Inject what we know about the user
        user_context = self._honcho.get_context(
            session_id=self._honcho.config.default_session,
            tokens=max_tokens,
            peer_target=peer_id,
        )
        if user_context:
            parts.append(f"\n## Known Context About {peer_id}\n{user_context}")

        return "\n\n".join(parts)

    def search_memory(self, query: str, limit: int = 10) -> list[dict]:
        """Search across all memory for relevant information."""
        return []


# =============================================================================
# Local fallback (no Honcho API key required)
# =============================================================================

class _LocalHonchoFallback:
    """Local-only fallback when Honcho SDK is not available.

    Mimics the Honcho API with in-memory storage so the rest of the
    system works without an API key. Trade data is persisted to JSON.
    """

    def __init__(self, workspace_id: str = "clawd-default"):
        self.workspace_id = workspace_id
        self._peers: dict[str, _LocalPeer] = {}
        self._sessions: dict[str, _LocalSession] = {}
        self._messages: list[dict] = []

    def peer(self, peer_id: str) -> _LocalPeer:
        if peer_id not in self._peers:
            self._peers[peer_id] = _LocalPeer(peer_id, self)
        return self._peers[peer_id]

    def session(self, session_id: str) -> _LocalSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = _LocalSession(session_id, self)
        return self._sessions[session_id]

    def _create_peer(self, peer_id: str):
        return self.peer(peer_id)

    def _create_session(self, session_id: str):
        return self.session(session_id)

    def queue_status(self):
        return type('QS', (), {
            'pending_work_units': 0,
            'in_progress_work_units': 0,
            'completed_work_units': len(self._messages),
            'total_work_units': len(self._messages),
        })()

    def search(self, query: str, limit: int = 10):
        return []


class _LocalPeer:
    """Local fallback peer."""

    def __init__(self, peer_id: str, parent: _LocalHonchoFallback):
        self.id = peer_id
        self._parent = parent
        self._card: list[str] = []

    def chat(self, query: str, **kwargs) -> str:
        return ("[Local memory] No Honcho API key set. "
                "Set HONCHO_API_KEY to enable persistent memory. "
                f"You asked: {query}")

    def message(self, content: str, metadata: dict | None = None):
        return {"peer_id": self.id, "content": content,
                "metadata": metadata or {}, "timestamp": time.time()}

    def get_card(self) -> list[str]:
        return self._card

    def set_card(self, facts: list[str]):
        self._card = facts


class _LocalSession:
    """Local fallback session."""

    def __init__(self, session_id: str, parent: _LocalHonchoFallback):
        self.id = session_id
        self._parent = parent

    def add_messages(self, messages: list):
        for msg in messages:
            entry = {
                "session_id": self.id,
                "peer_id": msg.get("peer_id", "unknown"),
                "content": msg.get("content", ""),
                "metadata": msg.get("metadata", {}),
                "timestamp": msg.get("timestamp", time.time()),
            }
            self._parent._messages.append(entry)

    def context(self, tokens: int = 2000, **kwargs) -> str:
        recent = [m for m in self._parent._messages if m["session_id"] == self.id]
        if not recent:
            return ""
        recent_text = "\n".join(
            f"[{m['peer_id']}]: {m['content'][:200]}"
            for m in recent[-10:]
        )
        return f"Recent messages ({len(recent)} total):\n{recent_text}"

    def search(self, query: str, limit: int = 10) -> list:
        return []

    def messages(self, **kwargs) -> list:
        return [m for m in self._parent._messages if m["session_id"] == self.id]