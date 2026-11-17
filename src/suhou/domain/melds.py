"""The internal call record shared by the replay engine and its consumers.

``raw_mask`` is opaque provenance from the source protocol. It is decoded by
the Tenhou adapter and never interpreted downstream; it stays on the record so
the audit snapshot can preserve the original call encoding.
"""

from __future__ import annotations

from dataclasses import dataclass

from suhou.domain.tiles import tile_name
from suhou.domain.types import MeldKind


@dataclass(frozen=True, slots=True)
class Meld:
    caller: int
    kind: MeldKind
    from_seat: int
    tiles: tuple[int, ...]
    called_tile: int
    consumed_tiles: tuple[int, ...]
    raw_mask: int
    # The tile added by a later kakan. A kakan keeps the original pon's
    # source and called tile, so the two events stay distinguishable.
    added_tile: int | None = None
    # Round-local visible event sequence numbers, filled in by the replay.
    formed_seq: int | None = None
    updated_seq: int | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(tile_name(tile) for tile in self.tiles)
