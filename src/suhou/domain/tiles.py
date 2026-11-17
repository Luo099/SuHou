"""Physical tile identifiers, tile kinds, and wind helpers.

The 136-tile physical model and the red-five identity are shared by the
protocol adapter, the replay engine, and the observation projection. Tenhou's
``tenhou.tiles`` module re-exports these names for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

WIND_NAMES = ("东", "南", "西", "北")
DRAGON_NAMES = ("白", "发", "中")

_TILE_KINDS = tuple(
    [f"{number}{suit}" for suit in "mps" for number in range(1, 10)]
    + [*WIND_NAMES, *DRAGON_NAMES]
)
_TILE_KIND_ORDER = {kind: index for index, kind in enumerate(_TILE_KINDS)}
_RED_FIVE_KINDS = frozenset({"5m", "5p", "5s"})


class Wind(StrEnum):
    """A mahjong wind, indexed by the 0..3 order 东南西北."""

    EAST = "东"
    SOUTH = "南"
    WEST = "西"
    NORTH = "北"

    @classmethod
    def from_index(cls, index: int) -> Wind:
        return _WINDS[index % 4]

    @classmethod
    def for_seat(cls, seat: int, dealer: int) -> Wind:
        """Seat wind relative to the dealer (0=dealer=East)."""
        return cls.from_index((seat - dealer + 4) % 4)


_WINDS = (Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH)


def validate_tile(tile: int) -> int:
    """Reject values outside the 136 physical tile identifiers."""
    if not 0 <= tile < 136:
        raise ValueError(f"Tile ID must be in 0..135, got {tile}")
    return tile


def is_red_five(tile: int) -> bool:
    """Return whether a physical tile is one of the three red fives."""
    return tile in (16, 52, 88)


def dora_from_indicator(tile: int) -> int:
    """Return the dora tile kind indicated by a dead-wall indicator."""
    validate_tile(tile)
    kind = tile // 4
    if kind < 27:
        base = kind // 9 * 9
        next_kind = base + (kind - base + 1) % 9
    elif kind < 31:
        next_kind = 27 + (kind - 27 + 1) % 4
    else:
        next_kind = 31 + (kind - 31 + 1) % 3
    return next_kind * 4


def tile_kind(tile: int) -> str:
    """Render the tile kind (1m..9m, 1p..9p, 1s..9s, winds, dragons).

    Red fives keep the ordinary ``5m``/``5p``/``5s`` kind; use
    ``is_red_five`` to distinguish the copy.
    """
    validate_tile(tile)
    kind = tile // 4
    if kind < 27:
        return f"{kind % 9 + 1}{'mps'[kind // 9]}"
    if kind < 31:
        return WIND_NAMES[kind - 27]
    return DRAGON_NAMES[kind - 31]


def tile_name(tile: int) -> str:
    """Render a tile without losing its red-five identity."""
    if is_red_five(tile):
        validate_tile(tile)
        return f"0{'mps'[tile // 36]}"
    return tile_kind(tile)


@dataclass(frozen=True, slots=True)
class Tile:
    """A tile kind plus its red-five identity, without a physical copy ID."""

    kind: str
    red: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _TILE_KIND_ORDER:
            raise ValueError(f"Unknown tile kind: {self.kind!r}")
        if self.red and self.kind not in _RED_FIVE_KINDS:
            raise ValueError(f"Only 5m/5p/5s can be red, got {self.kind!r}")

    @classmethod
    def from_physical(cls, tile: int, red_fives: bool = True) -> Tile:
        """Build a red-five-aware tile from a physical 0..135 identifier.

        ``red_fives`` follows the parsed game settings: a no-red-fives match
        treats the same physical copy as an ordinary five.
        """
        return cls(tile_kind(tile), red_fives and is_red_five(tile))

    @property
    def sort_key(self) -> tuple[int, int]:
        """Suit/kind order, then red before the ordinary copy."""
        return (_TILE_KIND_ORDER[self.kind], 0 if self.red else 1)

    @property
    def name(self) -> str:
        """Display form; red fives use ``0m``/``0p``/``0s``."""
        return f"0{self.kind[1]}" if self.red else self.kind

    def __str__(self) -> str:
        return self.name
