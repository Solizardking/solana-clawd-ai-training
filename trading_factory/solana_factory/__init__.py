"""Solana Clawd NVIDIA trading-factory adapters.

This package is intentionally paper-first. It builds reviewable strategy specs,
Phoenix/Rise read plans, and cuFOLIO optimization handoffs; it does not submit
live orders.
"""

from .cufolio_adapter import DEFAULT_ASSETS, build_mean_cvar_handoff, discover_cufolio
from .factory import build_strategy_bundle
from .nvidia_agent import build_nvidia_clawd_agent_plan, write_nvidia_clawd_agent_plan
from .rise_client import RiseReadOnlyClient, RiseReadOnlyConfig, build_rise_data_plan
from .validation import (
    ValidationReport,
    load_strategy_manifest,
    validate_cufolio_handoff,
    validate_nvidia_agent_plan,
    validate_rise_data_plan,
    validate_strategy_bundle,
    validate_vulcan_command,
    validate_vulcan_command_plans,
)
from .vulcan_specs import (
    build_ema_adx_trend_strategy,
    build_macd_adx_trim_strategy,
    build_rsi_mean_reversion_strategy,
    clone_strategy,
    guarded_grid_command,
    guarded_twap_command,
    paper_ta_command,
    validate_ta_strategy_config,
)

__all__ = [
    "DEFAULT_ASSETS",
    "RiseReadOnlyClient",
    "RiseReadOnlyConfig",
    "ValidationReport",
    "build_strategy_bundle",
    "build_mean_cvar_handoff",
    "build_nvidia_clawd_agent_plan",
    "build_rise_data_plan",
    "build_ema_adx_trend_strategy",
    "build_macd_adx_trim_strategy",
    "build_rsi_mean_reversion_strategy",
    "clone_strategy",
    "discover_cufolio",
    "guarded_grid_command",
    "guarded_twap_command",
    "load_strategy_manifest",
    "paper_ta_command",
    "validate_cufolio_handoff",
    "validate_nvidia_agent_plan",
    "validate_rise_data_plan",
    "validate_strategy_bundle",
    "validate_ta_strategy_config",
    "validate_vulcan_command",
    "validate_vulcan_command_plans",
    "write_nvidia_clawd_agent_plan",
]
