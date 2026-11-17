"""Player rule engine for one designated viewer seat."""

from suhou.rules.engine import PlayerRuleEngine
from suhou.rules.extract import PlayerDecision, iter_player_decisions, match_event_action
from suhou.rules.model import (
    AnkanAction,
    ChiAction,
    DiscardAction,
    KakanAction,
    KyushuAction,
    MinkanAction,
    PassAction,
    PonAction,
    RiichiAction,
    RonAction,
    RuleAction,
    RuleDecision,
    RuleDecisionKind,
    RuleError,
    TsumoAction,
    WinContext,
)

__all__ = [
    "AnkanAction",
    "ChiAction",
    "DiscardAction",
    "KakanAction",
    "KyushuAction",
    "MinkanAction",
    "PassAction",
    "PlayerDecision",
    "PlayerRuleEngine",
    "PonAction",
    "RiichiAction",
    "RonAction",
    "RuleAction",
    "RuleDecision",
    "RuleDecisionKind",
    "RuleError",
    "TsumoAction",
    "WinContext",
    "iter_player_decisions",
    "match_event_action",
]
