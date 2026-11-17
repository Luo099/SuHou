"""Observation v1: structured, player-visible decision snapshots."""

from suhou.observation.builder import (
    ObservationError,
    RulesSnapshot,
    RulesState,
    build_observation,
    match_type_from_game_type,
)
from suhou.observation.extract import DecisionPoint, RulesStateFactory, iter_decision_points
from suhou.observation.history import HistoryEntry, RoundHistory
from suhou.observation.model import (
    DecisionContext,
    DecisionPhase,
    Discard,
    EventType,
    MatchType,
    Meld,
    MeldType,
    Observation,
    PlayerObservation,
    RiichiState,
    SelfObservation,
    TableObservation,
    Tile,
    VisibleEvent,
    to_jsonable,
)

__all__ = [
    "DecisionContext",
    "DecisionPhase",
    "DecisionPoint",
    "Discard",
    "EventType",
    "HistoryEntry",
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
    "match_type_from_game_type",
    "to_jsonable",
]
