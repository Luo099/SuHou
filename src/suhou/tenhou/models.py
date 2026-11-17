"""Typed Tenhou protocol records. Scores are represented in points.

``TenhouEvent`` is the source record and keeps the original tag, attributes,
and document order. The old name ``Event`` remains as a compatibility alias.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from urllib.parse import unquote

from suhou.domain.melds import Meld
from suhou.domain.tiles import tile_name
from suhou.domain.types import PlayerInfo

Player = PlayerInfo


@dataclass(frozen=True, slots=True)
class TenhouEvent:
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
    # The N.m bit field decoded once by the parser; downstream shares it.
    meld_record: Meld | None = None
    step: int | None = None
    draw_type: str | None = None

    @property
    def tile_name(self) -> str | None:
        return tile_name(self.tile) if self.tile is not None else None


# Compatibility: callers that imported ``Event`` keep working.
Event = TenhouEvent


@dataclass(frozen=True, slots=True)
class GameRecord:
    events: tuple[TenhouEvent, ...]
    players: tuple[Player, Player, Player, Player] = field(
        default=(Player(""), Player(""), Player(""), Player(""))
    )
    game_type: int | None = None
    lobby: int | None = None
    shuffle: str | None = None
    shuffle_reference: str | None = None


def players_from_attributes(
    attributes: Mapping[str, str],
) -> tuple[Player, Player, Player, Player]:
    """Decode a complete ``<UN>`` roster from raw attributes."""
    names = [unquote(attributes.get(f"n{seat}", ""), encoding="utf-8") for seat in range(4)]

    def optional_values(name: str) -> list[str | None]:
        raw = attributes.get(name)
        if raw is None:
            return [None] * 4
        values = raw.split(",")
        if len(values) != 4:
            raise ValueError(f"'<UN>' '{name}' must contain four comma-separated values")
        return values

    dan = optional_values("dan")
    rate = optional_values("rate")
    sex = optional_values("sx")
    return tuple(
        Player(
            names[seat],
            int(dan[seat]) if dan[seat] not in (None, "") else None,
            float(rate[seat]) if rate[seat] not in (None, "") else None,
            sex[seat],
        )
        for seat in range(4)
    )
