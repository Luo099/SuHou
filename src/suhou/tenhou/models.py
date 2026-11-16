"""Typed Tenhou protocol records. Scores are represented in points."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from suhou.tenhou.tiles import tile_name


@dataclass(frozen=True, slots=True)
class Player:
    name: str
    dan: int | None = None
    rate: float | None = None
    sex: str | None = None


@dataclass(frozen=True, slots=True)
class Event:
    """One source XML child, in its original document order."""

    index: int
    tag: str
    raw_tag: str
    attributes: MappingProxyType[str, str]
    seat: int | None = None
    tile: int | None = None
    tiles: tuple[int, ...] | None = None
    seed: tuple[int, int, int, int, int, int] | None = None
    oya: int | None = None
    scores: tuple[int, ...] | None = None
    scores_before: tuple[int, int, int, int] | None = None
    deltas: tuple[int, int, int, int] | None = None
    winners: tuple[tuple[int, int], ...] | None = None
    final_scores: tuple[int, int, int, int] | None = None
    final_points: tuple[float, float, float, float] | None = None
    meld: int | None = None
    step: int | None = None
    draw_type: str | None = None

    @property
    def tile_name(self) -> str | None:
        return tile_name(self.tile) if self.tile is not None else None


@dataclass(frozen=True, slots=True)
class GameRecord:
    events: tuple[Event, ...]
    players: tuple[Player, Player, Player, Player] = field(
        default=(Player(""), Player(""), Player(""), Player(""))
    )
    game_type: int | None = None
    lobby: int | None = None
    shuffle: str | None = None
    shuffle_reference: str | None = None
