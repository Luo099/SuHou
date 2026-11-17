"""Observation v1: structured, player-visible decision snapshots."""

from suhou.domain.events import EventType, GameEvent, VisibleEvent
from suhou.domain.visibility import project_visible
from suhou.observation.builder import (
    ObservationError,
    RulesSnapshot,
    RulesState,
    build_observation,
    match_type_from_game_type,
)
from suhou.observation.extract import (
    DecisionPoint,
    LegacyRulesStateFactory,
    RulesStateFactory,
    iter_decision_points,
    legacy_rules_state_factory,
)
from suhou.observation.history import (
    EventHistory,
    HistoryEntry,
    RoundHistory,
)
from suhou.observation.model import (
    DecisionContext,
    DecisionPhase,
    Discard,
    MatchType,
    Meld,
    MeldType,
    Observation,
    PlayerObservation,
    RiichiState,
    SelfObservation,
    TableObservation,
    Tile,
    to_jsonable,
)

__all__ = [
    "DecisionContext",
    "DecisionPhase",
    "DecisionPoint",
    "Discard",
    "EventHistory",
    "EventType",
    "GameEvent",
    "HistoryEntry",
    "LegacyRulesStateFactory",
    "MatchType",
    "Meld",
    "MeldType",
    "Observation",
    "ObservationError",
    "PlayerObservation",
    "RiichiState",
    "RoundHistory",
    "RulesSnapshot",
    "RulesState",
    "RulesStateFactory",
    "SelfObservation",
    "TableObservation",
    "Tile",
    "VisibleEvent",
    "build_observation",
    "iter_decision_points",
    "legacy_rules_state_factory",
    "match_type_from_game_type",
    "project_visible",
    "to_jsonable",
]
