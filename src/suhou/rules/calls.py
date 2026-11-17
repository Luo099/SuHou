"""Call combinations and meld bookkeeping for the four-player rules."""

from __future__ import annotations

from collections.abc import Iterable

from suhou.domain.melds import Meld
from suhou.domain.types import MeldKind
from suhou.rules.model import RuleError
from suhou.rules.tiles import physical_index

__all__ = [
    "is_closed",
    "kan_count",
    "meld_kind_index",
    "pon_kinds",
]


def is_closed(melds: Iterable[Meld]) -> bool:
    """A hand with only ankan melds still counts as closed for riichi/yaku."""
    return all(meld.kind is MeldKind.ANKAN for meld in melds)


def kan_count(melds: Iterable[Meld]) -> int:
    return sum(
        1
        for meld in melds
        if meld.kind in (MeldKind.ANKAN, MeldKind.MINKAN, MeldKind.KAKAN)
    )


def meld_kind_index(meld: Meld) -> int:
    """Return the represented kind: a sequence start for chi, else the kind."""
    if meld.kind is MeldKind.CHI:
        indices = sorted(physical_index(tile) for tile in meld.tiles)
        if (
            len(indices) != 3
            or indices[1] != indices[0] + 1
            or indices[2] != indices[0] + 2
            or indices[0] // 9 != indices[2] // 9
        ):
            raise RuleError(f"Invalid chi tiles: {meld.tiles}")
        return indices[0]
    return physical_index(meld.tiles[0])


def pon_kinds(melds: Iterable[Meld]) -> list[tuple[int, int]]:
    """Return ``(kind_index, formed_seq)`` for every pon a kakan can extend."""
    targets: list[tuple[int, int]] = []
    for meld in melds:
        if meld.kind is not MeldKind.PON:
            continue
        if meld.formed_seq is None:
            raise RuleError("A pon meld has no formed_seq; rebuild the replay state")
        targets.append((meld_kind_index(meld), meld.formed_seq))
    return targets
