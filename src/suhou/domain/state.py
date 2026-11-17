"""Mutable replay state plus immutable views for consumers.

The replay engine is the only writer of :class:`PlayerState` and
:class:`RoundState`. Every event yields :class:`RoundView` and
:class:`PlayerView` snapshots whose tuples and frozen records cannot change
when the engine continues replaying.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from suhou.domain.melds import Meld
from suhou.domain.tiles import Wind, dora_from_indicator, tile_name
from suhou.domain.types import MatchType, match_type_from_game_type


@dataclass(frozen=True, slots=True)
class DiscardRecord:
    """One discard, retaining the physical tile for flow checks."""

    tile: int
    tsumogiri: bool
    riichi: bool
    called_by: int | None
    seq: int

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "tile": tile_name(self.tile),
            "tile_id": self.tile,
            "riichi": self.riichi,
            "tsumogiri": self.tsumogiri,
            "seq": self.seq,
        }
        if self.called_by is not None:
            data["called_by"] = self.called_by
        return data


def _discard_record(record: Mapping[str, Any] | DiscardRecord) -> DiscardRecord:
    if isinstance(record, DiscardRecord):
        return record
    return DiscardRecord(
        tile=record["tile_id"],
        tsumogiri=bool(record["tsumogiri"]),
        riichi=bool(record["riichi"]),
        called_by=record.get("called_by"),
        seq=record["seq"],
    )


@dataclass(slots=True)
class PlayerState:
    seat: int = 0
    dealer: int = 0
    initial_hand: tuple[int, ...] = ()
    hand: Counter[int] = field(default_factory=Counter)
    drawn: int | None = None
    discards: list[dict[str, Any]] = field(default_factory=list)
    melds: list[Meld] = field(default_factory=list)
    riichi: str = "none"
    connected: bool = True

    @property
    def seat_wind(self) -> Wind:
        """This seat's wind, relative to the hand's dealer."""
        return Wind.for_seat(self.seat, self.dealer)

    def view(self) -> PlayerView:
        return PlayerView(
            seat=self.seat,
            dealer=self.dealer,
            initial_hand=self.initial_hand,
            hand=tuple(sorted(self.hand.elements())),
            drawn=self.drawn,
            discards=tuple(_discard_record(record) for record in self.discards),
            melds=tuple(self.melds),
            riichi=self.riichi,
            connected=self.connected,
        )

    def snapshot(self) -> dict[str, Any]:
        return self.view().snapshot()


@dataclass(frozen=True, slots=True)
class PlayerView:
    seat: int
    dealer: int
    initial_hand: tuple[int, ...]
    hand: tuple[int, ...]
    drawn: int | None
    discards: tuple[DiscardRecord, ...]
    melds: tuple[Meld, ...]
    riichi: str
    connected: bool

    @property
    def seat_wind(self) -> Wind:
        return Wind.for_seat(self.seat, self.dealer)

    def snapshot(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "seat_wind": self.seat_wind.value,
            "initial_hand": [tile_name(tile) for tile in self.initial_hand],
            "hand": [tile_name(tile) for tile in self.hand],
            "drawn": tile_name(self.drawn) if self.drawn is not None else None,
            "discards": [record.to_dict() for record in self.discards],
            "melds": [
                {
                    "kind": getattr(meld.kind, "value", meld.kind),
                    "from_seat": meld.from_seat,
                    "tiles": list(meld.names),
                    "called_tile": tile_name(meld.called_tile),
                    "added_tile": (
                        tile_name(meld.added_tile) if meld.added_tile is not None else None
                    ),
                    "formed_seq": meld.formed_seq,
                    "updated_seq": meld.updated_seq,
                    "mask": meld.raw_mask,
                }
                for meld in self.melds
            ],
            "riichi": self.riichi,
            "connected": self.connected,
        }


@dataclass(slots=True)
class RoundState:
    index: int
    seed: tuple[int, int, int, int, int, int]
    dealer: int
    players: list[PlayerState]
    scores: list[int]
    current_seat: int
    dora_indicators: list[int]
    uradora_indicators: list[int] = field(default_factory=list)
    honba: int = 0
    riichi_sticks: int = 0
    phase: str = "draw"
    pending_riichi: int | None = None
    last_discard: tuple[int, int] | None = None
    awaiting_replacement_draw: int | None = None
    normal_draws: int = 0
    replacement_draws: int = 0
    finished: bool = False
    results: list[dict[str, Any]] = field(default_factory=list)
    final_scores: tuple[int, int, int, int] | None = None
    final_points: tuple[float, float, float, float] | None = None
    # Source index of this round's <INIT>; observation/history sequence numbers
    # are round-local and start at 0 for that event.
    start_index: int = 0
    # <GO type> rule bits. Observation reads the decoded match_type instead.
    game_type: int | None = None
    # Red fives are interpreted according to the parsed game settings.
    red_fives: bool = True
    # Sequential round instance in the game (renchan keeps the same index).
    round_id: int = 0
    _explicit_match_type: MatchType | None = field(default=None, repr=False)

    @property
    def match_type(self) -> MatchType | None:
        """The decoded match type, derived from the game settings when absent."""
        if self._explicit_match_type is not None:
            return self._explicit_match_type
        if self.game_type is None:
            return None
        try:
            return match_type_from_game_type(self.game_type)
        except ValueError:
            # Three-player logs remain outside Observation v1; the observation
            # builder reports the unsupported match type.
            return None

    @match_type.setter
    def match_type(self, value: MatchType | None) -> None:
        self._explicit_match_type = value

    @property
    def name(self) -> str:
        return f"{self.round_wind.value}{self.seed[0] % 4 + 1}局"

    @property
    def round_wind(self) -> Wind:
        """The hand's round wind, from the seed's 0..15 round index."""
        return Wind.from_index(self.seed[0] // 4)

    @property
    def dice(self) -> tuple[int, int]:
        return self.seed[3] + 1, self.seed[4] + 1

    @property
    def wall_remaining(self) -> int:
        return 70 - self.normal_draws - self.replacement_draws

    def event_seq(self, source_event: Any) -> int:
        """Compatibility: round-local index of a source record."""
        return source_event.index - self.start_index

    def view(self) -> RoundView:
        return RoundView(
            index=self.index,
            round_id=self.round_id,
            round_index=self.seed[0],
            round_wind=self.round_wind,
            dealer=self.dealer,
            players=tuple(player.view() for player in self.players),
            scores=list(self.scores),
            current_seat=self.current_seat,
            dora_indicators=tuple(self.dora_indicators),
            uradora_indicators=tuple(self.uradora_indicators),
            honba=self.honba,
            riichi_sticks=self.riichi_sticks,
            phase=self.phase,
            pending_riichi=self.pending_riichi,
            last_discard=self.last_discard,
            awaiting_replacement_draw=self.awaiting_replacement_draw,
            normal_draws=self.normal_draws,
            replacement_draws=self.replacement_draws,
            finished=self.finished,
            results=tuple(copy.deepcopy(result) for result in self.results),
            final_scores=self.final_scores,
            final_points=self.final_points,
            start_index=self.start_index,
            game_type=self.game_type,
            red_fives=self.red_fives,
            match_type=self.match_type,
            dice=self.dice,
        )

    def snapshot(self) -> dict[str, Any]:
        return self.view().snapshot()


@dataclass(frozen=True, slots=True)
class RoundView:
    index: int
    round_id: int
    round_index: int
    round_wind: Wind
    dealer: int
    players: tuple[PlayerView, ...]
    scores: list[int]
    current_seat: int
    dora_indicators: tuple[int, ...]
    uradora_indicators: tuple[int, ...]
    honba: int
    riichi_sticks: int
    phase: str
    pending_riichi: int | None
    last_discard: tuple[int, int] | None
    awaiting_replacement_draw: int | None
    normal_draws: int
    replacement_draws: int
    finished: bool
    results: tuple[dict[str, Any], ...]
    final_scores: tuple[int, int, int, int] | None
    final_points: tuple[float, float, float, float] | None
    start_index: int
    game_type: int | None
    red_fives: bool
    match_type: MatchType | None
    dice: tuple[int, int]

    @property
    def name(self) -> str:
        return f"{self.round_wind.value}{self.round_index % 4 + 1}局"

    @property
    def seed(self) -> tuple[int, int, int, int, int, int]:
        """Synthetic source-style seed for compatibility with older callers."""
        dora = self.dora_indicators[0] if self.dora_indicators else 0
        return (
            self.round_index,
            self.honba,
            self.riichi_sticks,
            self.dice[0] - 1,
            self.dice[1] - 1,
            dora,
        )

    @property
    def wall_remaining(self) -> int:
        return 70 - self.normal_draws - self.replacement_draws

    def event_seq(self, source_event: Any) -> int:
        """Compatibility: round-local index of a source record."""
        return source_event.index - self.start_index

    def snapshot(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "round_id": self.round_id,
            "name": self.name,
            "start_index": self.start_index,
            "round_index": self.round_index,
            "round_wind": self.round_wind.value,
            "dealer": self.dealer,
            "honba": self.honba,
            "riichi_sticks": self.riichi_sticks,
            "dice": list(self.dice),
            "dora_indicators": [tile_name(tile) for tile in self.dora_indicators],
            "dora": [tile_name(dora_from_indicator(tile)) for tile in self.dora_indicators],
            "uradora_indicators": [tile_name(tile) for tile in self.uradora_indicators],
            "scores": list(self.scores),
            "wall_remaining": self.wall_remaining,
            "current_seat": self.current_seat,
            "phase": self.phase,
            "last_discard": (
                {"seat": self.last_discard[0], "tile": tile_name(self.last_discard[1])}
                if self.last_discard is not None
                else None
            ),
            "finished": self.finished,
            "players": [player.snapshot() for player in self.players],
            "results": [copy.deepcopy(result) for result in self.results],
        }
