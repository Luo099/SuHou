"""Immutable player-visible observation records.

The field names mirror the Observation v1 contract. Physical tile IDs never
appear here: :class:`~suhou.domain.tiles.Tile` keeps only the kind and the
red-five flag, while the replay keeps the physical copies for flow checks.

The shared enums and the tile model live in ``suhou.domain`` and are re-exported
here so existing observation imports keep working.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from typing import Any

from suhou.domain.events import EventType, VisibleEvent
from suhou.domain.tiles import Tile, Wind
from suhou.domain.types import MatchType, MeldType
from suhou.domain.values import freeze_value as freeze_payload


class RiichiState(StrEnum):
    NONE = "none"
    PENDING = "pending"
    DECLARED = "declared"
    ACCEPTED = "accepted"


class DecisionPhase(StrEnum):
    """Decision context supplied by the rules adapter.

    These values are provisional until the reused rules engine defines its own
    decision phases; the observation contract only requires a stable enum.
    """

    DISCARD = "discard"
    RESPONSE = "response"
    RIICHI = "riichi"
    SELF_KAN = "self_kan"
    TSUMO = "tsumo"


@dataclass(frozen=True, slots=True)
class Discard:
    tile: Tile
    tsumogiri: bool
    riichi_declaration: bool
    called_by: int | None
    seq: int

    def __post_init__(self) -> None:
        if self.called_by is not None and not 0 <= self.called_by < 4:
            raise ValueError(f"called_by must be in 0..3, got {self.called_by}")
        if self.seq < 0:
            raise ValueError(f"Discard seq must be non-negative, got {self.seq}")


@dataclass(frozen=True, slots=True)
class Meld:
    type: MeldType
    tiles: tuple[Tile, ...]
    source_player: int | None
    called_tile: Tile | None
    added_tile: Tile | None
    formed_seq: int
    updated_seq: int

    def __post_init__(self) -> None:
        expected = 4 if self.type in (MeldType.MINKAN, MeldType.ANKAN, MeldType.KAKAN) else 3
        if len(self.tiles) != expected:
            raise ValueError(f"{self.type} must contain {expected} tiles")
        if self.source_player is not None and not 0 <= self.source_player < 4:
            raise ValueError(f"source_player must be in 0..3, got {self.source_player}")
        if self.formed_seq < 0 or self.updated_seq < self.formed_seq:
            raise ValueError("Meld sequence numbers must be ordered and non-negative")
        if self.type == MeldType.ANKAN:
            if self.source_player is not None or self.called_tile is not None:
                raise ValueError("Ankan has no source player or called tile")
            if self.added_tile is not None:
                raise ValueError("Ankan cannot have an added tile")
        elif self.type == MeldType.KAKAN:
            if self.source_player is None or self.called_tile is None:
                raise ValueError("Kakan keeps the original pon source and called tile")
            if self.added_tile is None:
                raise ValueError("Kakan requires an added tile")
        else:
            if self.source_player is None or self.called_tile is None:
                raise ValueError(f"{self.type} requires a source player and called tile")
            if self.added_tile is not None:
                raise ValueError(f"{self.type} cannot have an added tile")
        if self.called_tile is not None and self.called_tile not in self.tiles:
            raise ValueError("called_tile must be one of the meld tiles")
        if self.added_tile is not None and self.added_tile not in self.tiles:
            raise ValueError("added_tile must be one of the meld tiles")


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    seat_wind: Wind
    score: int
    concealed_tile_count: int
    riichi: RiichiState
    discards: tuple[Discard, ...]
    melds: tuple[Meld, ...]

    def __post_init__(self) -> None:
        if self.concealed_tile_count < 0:
            raise ValueError("concealed_tile_count must be non-negative")


@dataclass(frozen=True, slots=True)
class TableObservation:
    match_type: MatchType
    round_wind: Wind
    hand_number: int
    honba: int
    riichi_sticks: int
    dealer: int
    remaining_tiles: int
    dora_indicators: tuple[Tile, ...]

    def __post_init__(self) -> None:
        if not 1 <= self.hand_number <= 4:
            raise ValueError(f"hand_number must be in 1..4, got {self.hand_number}")
        if not 0 <= self.dealer < 4:
            raise ValueError(f"dealer must be in 0..3, got {self.dealer}")
        if self.honba < 0 or self.riichi_sticks < 0:
            raise ValueError("honba and riichi_sticks must be non-negative")
        if self.remaining_tiles < 0:
            raise ValueError("remaining_tiles must be non-negative")


@dataclass(frozen=True, slots=True)
class SelfObservation:
    hand: tuple[Tile, ...]
    drawn_tile: Tile | None
    furiten: bool

    def __post_init__(self) -> None:
        if self.drawn_tile is not None and self.drawn_tile not in self.hand:
            raise ValueError("drawn_tile must be part of the current hand")


@dataclass(frozen=True, slots=True)
class DecisionContext:
    phase: DecisionPhase
    trigger_seq: int

    def __post_init__(self) -> None:
        if self.trigger_seq < 0:
            raise ValueError("trigger_seq must be non-negative")


@dataclass(frozen=True, slots=True)
class Observation:
    viewer: int
    table: TableObservation
    # Named self_view because ``self`` is the method receiver. The JSON
    # contract still uses the key "self" (see to_dict) and the property below.
    self_view: SelfObservation
    players: tuple[PlayerObservation, ...]
    context: DecisionContext
    history: tuple[VisibleEvent, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.viewer < 4:
            raise ValueError(f"viewer must be in 0..3, got {self.viewer}")
        if len(self.players) != 4:
            raise ValueError("players must contain exactly four seats")
        if self.players[self.viewer].concealed_tile_count != len(self.self_view.hand):
            raise ValueError("viewer concealed_tile_count must match self.hand")
        previous = -1
        for event in self.history:
            if event.seq <= previous:
                raise ValueError("history seq values must be strictly increasing")
            previous = event.seq

    @property
    def self(self) -> SelfObservation:
        """The observing player's private view (contract key ``self``)."""
        return self.self_view

    def to_dict(self) -> dict[str, Any]:
        return {
            "viewer": self.viewer,
            "table": to_jsonable(self.table),
            "self": to_jsonable(self.self_view),
            "players": [to_jsonable(player) for player in self.players],
            "context": to_jsonable(self.context),
            "history": [to_jsonable(event) for event in self.history],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True)


def to_jsonable(value: Any) -> Any:
    """Convert observation records to JSON-compatible primitives."""
    if isinstance(value, Observation):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


__all__ = [
    "DecisionContext",
    "DecisionPhase",
    "Discard",
    "EventType",
    "MatchType",
    "Meld",
    "MeldType",
    "Observation",
    "PlayerObservation",
    "RiichiState",
    "SelfObservation",
    "TableObservation",
    "Tile",
    "VisibleEvent",
    "Wind",
    "freeze_payload",
    "to_jsonable",
]
