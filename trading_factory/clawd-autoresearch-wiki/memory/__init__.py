"""
Honcho-powered persistent memory for Clawd agents.

Provides cross-session memory that persists across context wipes:
- Remembers trading decisions, positions, PnL
- Tracks user preferences and communication style
- Recalls past strategies and their outcomes
- Dreams/evolves by consolidating patterns over time

Usage:
    from memory import AgentMemory

    memory = AgentMemory(api_key=HONCHO_API_KEY, workspace="clawd-trading")
    memory.remember("user-1", "Prefers conservative trades, max 2x leverage")
    ctx = memory.recall("user-1", "What are my trading preferences?")
"""
from memory.honcho import HonchoMemory, AgentMemory

__all__ = ["HonchoMemory", "AgentMemory"]