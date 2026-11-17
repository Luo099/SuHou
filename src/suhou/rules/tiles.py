"""34-kind helpers for rule analysis.

Rules use :class:`~suhou.domain.tiles.Tile` (kind plus red-five identity),
while shape analysis needs a dense 0..33 kind index. The index order is the
public order defined by :mod:`suhou.domain.tiles` and matches the physical tile
numbering (``tile // 4``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from suhou.domain.tiles import TILE_KIND_INDEX, TILE_KINDS, Tile

WIND_KIND_INDEX = {"东": 27, "南": 28, "西": 29, "北": 30}
DRAGON_KIND_INDEX = {"白": 31, "发": 32, "中": 33}

# Orphans for kokushi and the green-tile set for ryuuiisou.
ORPHAN_KINDS = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)
GREEN_KINDS = frozenset({19, 20, 21, 23, 25, 32})


def tile_index(tile: Tile) -> int:
    """Return the 0..33 kind index of a rules tile."""
    return TILE_KIND_INDEX[tile.kind]


def physical_index(tile: int) -> int:
    """Return the kind index of a physical 0..135 tile."""
    if not 0 <= tile < 136:
        raise ValueError(f"Tile ID must be in 0..135, got {tile}")
    return tile // 4


def tile_from_index(index: int, *, red: bool = False) -> Tile:
    """Build a rules tile from a 0..33 kind index."""
    if not 0 <= index < 34:
        raise ValueError(f"Tile kind index must be in 0..33, got {index}")
    return Tile(TILE_KINDS[index], red)


def is_honor(index: int) -> bool:
    return index >= 27


def is_terminal(index: int) -> bool:
    return index < 27 and index % 9 in (0, 8)


def is_terminal_or_honor(index: int) -> bool:
    return index >= 27 or index % 9 in (0, 8)


def is_simple(index: int) -> bool:
    return index < 27 and index % 9 not in (0, 8)


def is_suit(index: int) -> bool:
    return index < 27


def suit_of(index: int) -> int:
    """0=m, 1=p, 2=s; -1 for honors."""
    return index // 9 if index < 27 else -1


def counts_from_tiles(tiles: Iterable[Tile]) -> list[int]:
    counts = [0] * 34
    for tile in tiles:
        counts[tile_index(tile)] += 1
    return counts


def counts_from_physical(tiles: Iterable[int]) -> list[int]:
    counts = [0] * 34
    for tile in tiles:
        counts[physical_index(tile)] += 1
    return counts


def tile_copies(tiles: Sequence[Tile], index: int) -> tuple[int, int]:
    """Return ``(ordinary, red)`` copy counts of one kind in a tile list."""
    ordinary = red = 0
    for tile in tiles:
        if tile_index(tile) != index:
            continue
        if tile.red:
            red += 1
        else:
            ordinary += 1
    return ordinary, red


def choose_copies(index: int, ordinary: int, red: int, count: int) -> list[tuple[Tile, ...]]:
    """Enumerate distinct ways to consume ``count`` copies of one kind.

    Red and ordinary copies are semantically different, so
    ``{red, ordinary}`` and ``{ordinary, ordinary}`` are separate choices.
    Results are deterministic: red copies first, matching ``Tile.sort_key``.
    """
    if count < 0 or count > ordinary + red:
        return []
    low = max(0, count - ordinary)
    high = min(count, red)
    options: list[tuple[Tile, ...]] = []
    for red_count in range(high, low - 1, -1):
        ordinary_count = count - red_count
        if ordinary_count < 0 or ordinary_count > ordinary:
            continue
        options.append(
            tuple(tile_from_index(index, red=True) for _ in range(red_count))
            + tuple(tile_from_index(index) for _ in range(ordinary_count))
        )
    return options


def chi_options(hand: Sequence[Tile], called_index: int) -> list[tuple[Tile, Tile]]:
    """Enumerate the two hand tiles for every legal chi on ``called_index``."""
    if not is_suit(called_index):
        return []
    base = (called_index // 9) * 9
    position = called_index - base
    options: set[tuple[Tile, Tile]] = set()
    for start in (position - 2, position - 1, position):
        if not 0 <= start <= 6:
            continue
        needed = [
            base + kind
            for kind in (start, start + 1, start + 2)
            if base + kind != called_index
        ]
        if len(needed) != 2:
            continue
        first = _single_copy_options(hand, needed[0])
        second = _single_copy_options(hand, needed[1])
        for left in first:
            for right in second:
                options.add(tuple(sorted((left, right), key=lambda tile: tile.sort_key)))
    return sorted(options, key=_consumed_sort_key)


def pon_options(hand: Sequence[Tile], called_index: int) -> list[tuple[Tile, Tile]]:
    ordinary, red = tile_copies(hand, called_index)
    return [
        tuple(sorted(choice, key=lambda tile: tile.sort_key))
        for choice in choose_copies(called_index, ordinary, red, 2)
    ]


def minkan_options(hand: Sequence[Tile], called_index: int) -> list[tuple[Tile, Tile, Tile]]:
    ordinary, red = tile_copies(hand, called_index)
    return [
        tuple(sorted(choice, key=lambda tile: tile.sort_key))
        for choice in choose_copies(called_index, ordinary, red, 3)
    ]


def ankan_options(hand: Sequence[Tile]) -> list[tuple[Tile, Tile, Tile, Tile]]:
    counts = Counter(tile_index(tile) for tile in hand)
    options: list[tuple[Tile, Tile, Tile, Tile]] = []
    for index in sorted(counts):
        if counts[index] < 4:
            continue
        ordinary, red = tile_copies(hand, index)
        for choice in choose_copies(index, ordinary, red, 4):
            options.append(tuple(sorted(choice, key=lambda tile: tile.sort_key)))
    return options


def kakan_options(
    hand: Sequence[Tile], pon_kinds: Iterable[tuple[int, int]]
) -> list[tuple[Tile, int]]:
    """Enumerate ``(added_tile, pon_formed_seq)`` for every extendable pon."""
    options: list[tuple[Tile, int]] = []
    for kind_index, formed_seq in pon_kinds:
        ordinary, red = tile_copies(hand, kind_index)
        for choice in choose_copies(kind_index, ordinary, red, 1):
            options.append((choice[0], formed_seq))
    return options


def _single_copy_options(hand: Sequence[Tile], index: int) -> list[Tile]:
    ordinary, red = tile_copies(hand, index)
    return [choice[0] for choice in choose_copies(index, ordinary, red, 1)]


def _consumed_sort_key(tiles: tuple[Tile, ...]) -> tuple[tuple[int, int], ...]:
    return tuple(tile.sort_key for tile in tiles)


__all__ = [
    "DRAGON_KIND_INDEX",
    "GREEN_KINDS",
    "ORPHAN_KINDS",
    "WIND_KIND_INDEX",
    "ankan_options",
    "chi_options",
    "choose_copies",
    "counts_from_physical",
    "counts_from_tiles",
    "is_honor",
    "is_simple",
    "is_suit",
    "is_terminal",
    "is_terminal_or_honor",
    "kakan_options",
    "minkan_options",
    "physical_index",
    "pon_options",
    "suit_of",
    "tile_copies",
    "tile_from_index",
    "tile_index",
]
