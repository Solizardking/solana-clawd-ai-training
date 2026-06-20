"""
Phoenix Perpetuals trading package for Clawd.

Integrates Vulcan CLI, Rise SDK, paper trading, and Nemotron Ultra 550B.
"""
from perps.vulcan import VulcanClient, VulcanConfig, VulcanResult, OutputFormat
from perps.paper import PaperEngine, PaperAccount, PaperPosition, PaperOrder, PaperFill
from perps.rise import RiseClient, RiseConfig
from perps.nemotron import NemotronTrader, TradePlan
from perps.nvidia_signals import NvidiaSignalBridge, CompositeSignal

__all__ = [
    "VulcanClient", "VulcanConfig", "VulcanResult", "OutputFormat",
    "PaperEngine", "PaperAccount", "PaperPosition", "PaperOrder", "PaperFill",
    "RiseClient", "RiseConfig",
    "NemotronTrader", "TradePlan",
    "NvidiaSignalBridge", "CompositeSignal",
]