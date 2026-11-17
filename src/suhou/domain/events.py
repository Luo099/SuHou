"""Immutable internal mahjong events.

``GameEvent`` describes one action that happened at the table. It carries no
Tenhou tag names or raw attributes: the Tenhou adapter owns protocol decoding,
and the source records (``TenhouEvent``) stay available for audit by
``source_index``. Payloads are per-kind frozen dataclasses so no event stacks a
pile of nullable fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from suhou.domain.melds import Meld
from suhou.domain.tiles import Wind, validate_tile
from suhou.domain.types import MeldKind
from suhou.domain.values import freeze_value

SCHEMA_VERSION = 1


class EventType(StrEnum):
    """Every kind of behaviour represented in the unified stream."""

    METADATA = "metadata"
    ROUND_START = "round_start"
    DRAW = "draw"
    DISCARD = "discard"
    CALL = "call"
    RIICHI_DECLARATION = "riichi_declaration"
    RIICHI_ACCEPTED = "riichi_accepted"
    DORA = "dora"
    RESULT = "result"
    CONNECTION = "connection"


# The public event kind and the observation history type are the same enum.
GameEventKind = EventType


class MetadataEntry(StrEnum):
    GAME_SETTINGS = "game_settings"
    ROSTER = "roster"
    GAME_START = "game_start"
    SHUFFLE = "shuffle"


@dataclass(frozen=True, slots=True)
class MetadataPayload:
    """Recognized source metadata, decoded into a whitelisted data mapping."""

    entry: MetadataEntry
    source_tag: str
    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", freeze_value(self.data))


@dataclass(frozen=True, slots=True)
class RoundStartPayload:
    round_index: int
    round_wind: Wind
    hand_number: int
    honba: int
    riichi_sticks: int
    dealer: int
    dice: tuple[int, int]
    scores: tuple[int, int, int, int]
    hands: tuple[tuple[int, ...], ...]
    dora_indicator: int

    def __post_init__(self) -> None:
        if not 0 <= self.round_index < 16:
            raise ValueError("round_index must be in 0..15")
        if not 1 <= self.hand_number <= 4:
            raise ValueError("hand_number must be in 1..4")
        if not 0 <= self.dealer < 4:
            raise ValueError("dealer must be in 0..3")
        if len(self.scores) != 4:
            raise ValueError("scores must contain four seats")
        validate_tile(self.dora_indicator)
        if len(self.hands) != 4 or any(len(hand) != 13 for hand in self.hands):
            raise ValueError("hands must contain four 13-tile hands")
        if any(len(set(hand)) != 13 for hand in self.hands):
            raise ValueError("each hand must contain distinct physical tiles")
        if len(set(tile for hand in self.hands for tile in hand)) != 52:
            raise ValueError("physical tile copies must be unique")


@dataclass(frozen=True, slots=True)
class DrawPayload:
    tile: int
    replacement: bool

    def __post_init__(self) -> None:
        validate_tile(self.tile)


@dataclass(frozen=True, slots=True)
class DiscardPayload:
    tile: int
    tsumogiri: bool
    riichi_declaration: bool

    def __post_init__(self) -> None:
        validate_tile(self.tile)


@dataclass(frozen=True, slots=True)
class CallPayload:
    kind: MeldKind
    # The called player for chi/pon/minkan/kakan; None for ankan/nuki.
    from_seat: int | None
    tiles: tuple[int, ...]
    called_tile: int | None
    consumed_tiles: tuple[int, ...]
    added_tile: int | None
    # Round-local seq of the pon that a kakan extends.
    target_seq: int | None
    # Opaque source encoding preserved for the audit snapshot.
    source_mask: int

    def __post_init__(self) -> None:
        if not 0 <= self.source_mask <= 0xFFFF:
            raise ValueError("call source_mask must be in 0..0xFFFF")
        for tile in self.tiles:
            validate_tile(tile)
        for tile in self.consumed_tiles:
            validate_tile(tile)
        if self.called_tile is not None:
            validate_tile(self.called_tile)
        if self.added_tile is not None:
            validate_tile(self.added_tile)
        if self.kind in (MeldKind.CHI, MeldKind.PON, MeldKind.MINKAN):
            expected_tiles, expected_consumed = (
                (4, 3) if self.kind is MeldKind.MINKAN else (3, 2)
            )
            if len(self.tiles) != expected_tiles or len(self.consumed_tiles) != expected_consumed:
                raise ValueError(f"{self.kind} has an invalid tile count")
            if self.from_seat is None or self.called_tile is None:
                raise ValueError(f"{self.kind} requires a source seat and called tile")
            if self.added_tile is not None or self.target_seq is not None:
                raise ValueError(f"{self.kind} has no added tile or target meld")
        elif self.kind is MeldKind.ANKAN:
            if len(self.tiles) != 4 or len(self.consumed_tiles) != 4:
                raise ValueError("ankan must contain four tiles")
            if self.from_seat is not None or self.added_tile is not None:
                raise ValueError("ankan has no source seat or added tile")
            if self.target_seq is not None:
                raise ValueError("ankan has no target meld")
        elif self.kind is MeldKind.KAKAN:
            if len(self.tiles) != 3 or self.consumed_tiles:
                raise ValueError("kakan keeps the original pon's three tiles")
            if (
                self.from_seat is None
                or self.called_tile is None
                or self.added_tile is None
                or self.target_seq is None
            ):
                raise ValueError("kakan requires source, called tile, added tile, and target")
        else:
            if len(self.tiles) != 1 or self.consumed_tiles:
                raise ValueError("nuki must contain exactly one tile")
            if self.from_seat is not None or self.called_tile is not None:
                raise ValueError("nuki has no source seat or called tile")
            if self.added_tile is not None or self.target_seq is not None:
                raise ValueError("nuki has no added tile or target meld")


@dataclass(frozen=True, slots=True)
class RiichiDeclarationPayload:
    pass


@dataclass(frozen=True, slots=True)
class RiichiAcceptedPayload:
    scores: tuple[int, int, int, int] | None

    def __post_init__(self) -> None:
        if self.scores is not None and len(self.scores) != 4:
            raise ValueError("riichi_accepted scores must contain four seats")


@dataclass(frozen=True, slots=True)
class DoraPayload:
    indicator: int

    def __post_init__(self) -> None:
        validate_tile(self.indicator)


@dataclass(frozen=True, slots=True)
class ConnectionPayload:
    connected: bool


@dataclass(frozen=True, slots=True)
class Yaku:
    id: int
    name: str
    han: int


@dataclass(frozen=True, slots=True)
class AgariPayload:
    group_id: int
    group_final: bool
    who: int
    from_who: int
    deltas: tuple[int, int, int, int]
    scores_before: tuple[int, int, int, int]
    winning_tile: int | None
    hand: tuple[int, ...] | None
    melds: tuple[Meld, ...]
    fu: int | None
    han: int | None
    points: int | None
    limit: int | None
    yaku: tuple[Yaku, ...]
    dora_indicators: tuple[int, ...] | None
    uradora_indicators: tuple[int, ...] | None
    honba_after: int
    riichi_sticks_at_result: int
    final_scores: tuple[int, int, int, int] | None
    final_points: tuple[float, float, float, float] | None

    def __post_init__(self) -> None:
        if self.group_id < 0:
            raise ValueError("agari group_id must be non-negative")
        if not 0 <= self.who < 4 or not 0 <= self.from_who < 4:
            raise ValueError("agari who/from_who must be in 0..3")
        if len(self.deltas) != 4 or len(self.scores_before) != 4:
            raise ValueError("agari scores must contain four seats")
        if self.winning_tile is not None:
            validate_tile(self.winning_tile)
        for tile in self.hand or ():
            validate_tile(tile)
        for meld in self.melds:
            for tile in meld.tiles:
                validate_tile(tile)
        if self.honba_after < 0 or self.riichi_sticks_at_result < 0:
            raise ValueError("agari honba and riichi sticks must be non-negative")
        if self.final_scores is not None and len(self.final_scores) != 4:
            raise ValueError("agari final_scores must contain four seats")
        if self.final_points is not None and len(self.final_points) != 4:
            raise ValueError("agari final_points must contain four seats")


@dataclass(frozen=True, slots=True)
class RyuukyokuPayload:
    group_id: int
    group_final: bool
    reason: str | None
    deltas: tuple[int, int, int, int]
    scores_before: tuple[int, int, int, int]
    honba_after: int
    riichi_sticks: int
    final_scores: tuple[int, int, int, int] | None
    final_points: tuple[float, float, float, float] | None

    def __post_init__(self) -> None:
        if self.group_id < 0:
            raise ValueError("ryuukyoku group_id must be non-negative")
        if len(self.deltas) != 4 or len(self.scores_before) != 4:
            raise ValueError("ryuukyoku scores must contain four seats")
        if self.honba_after < 0 or self.riichi_sticks < 0:
            raise ValueError("ryuukyoku honba and riichi sticks must be non-negative")
        if self.final_scores is not None and len(self.final_scores) != 4:
            raise ValueError("ryuukyoku final_scores must contain four seats")
        if self.final_points is not None and len(self.final_points) != 4:
            raise ValueError("ryuukyoku final_points must contain four seats")


EventPayload = (
    MetadataPayload
    | RoundStartPayload
    | DrawPayload
    | DiscardPayload
    | CallPayload
    | RiichiDeclarationPayload
    | RiichiAcceptedPayload
    | DoraPayload
    | ConnectionPayload
    | AgariPayload
    | RyuukyokuPayload
)

_PAYLOAD_TYPES: dict[EventType, tuple[type[object], ...]] = {
    EventType.METADATA: (MetadataPayload,),
    EventType.ROUND_START: (RoundStartPayload,),
    EventType.DRAW: (DrawPayload,),
    EventType.DISCARD: (DiscardPayload,),
    EventType.CALL: (CallPayload,),
    EventType.RIICHI_DECLARATION: (RiichiDeclarationPayload,),
    EventType.RIICHI_ACCEPTED: (RiichiAcceptedPayload,),
    EventType.DORA: (DoraPayload,),
    EventType.RESULT: (AgariPayload, RyuukyokuPayload),
    EventType.CONNECTION: (ConnectionPayload,),
}


@dataclass(frozen=True, slots=True)
class GameEvent:
    """One applied table action, with round-local ordering.

    ``source_index`` is the XML child index in the source document. ``seq`` is
    round-local and starts at 0 for ``round_start``; metadata outside a round
    has neither ``round_id`` nor ``seq``.
    """

    kind: EventType
    source_index: int
    round_id: int | None
    seq: int | None
    actor: int | None
    payload: EventPayload
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported event schema_version {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        if self.source_index < 0:
            raise ValueError("source_index must be non-negative")
        if self.actor is not None and not 0 <= self.actor < 4:
            raise ValueError(f"actor must be in 0..3, got {self.actor}")
        if self.seq is not None and self.seq < 0:
            raise ValueError("seq must be non-negative")
        if self.round_id is not None and self.round_id < 0:
            raise ValueError("round_id must be non-negative")
        if self.kind is EventType.METADATA:
            if self.round_id is not None or self.seq is not None or self.actor is not None:
                raise ValueError("metadata events have no round_id, seq, or actor")
        else:
            if self.round_id is None or self.seq is None:
                raise ValueError(f"{self.kind} events require round_id and seq")
            if self.kind in (
                EventType.ROUND_START,
                EventType.DORA,
                EventType.RESULT,
            ):
                if self.actor is not None:
                    raise ValueError(f"{self.kind} events have no actor")
            elif self.actor is None:
                raise ValueError(f"{self.kind} events require an actor")
        allowed = _PAYLOAD_TYPES[self.kind]
        if not isinstance(self.payload, allowed):
            names = " or ".join(item.__name__ for item in allowed)
            raise ValueError(
                f"{self.kind} requires payload {names}, "
                f"got {type(self.payload).__name__}"
            )


@dataclass(frozen=True, slots=True)
class VisibleEvent:
    """One event as a specific player could have seen it."""

    seq: int
    type: EventType
    actor: int | None
    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.seq < 0:
            raise ValueError("event seq must be non-negative")
        if self.actor is not None and not 0 <= self.actor < 4:
            raise ValueError(f"actor must be in 0..3, got {self.actor}")
        object.__setattr__(self, "data", freeze_value(self.data))

    def __copy__(self) -> VisibleEvent:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> VisibleEvent:
        return self
