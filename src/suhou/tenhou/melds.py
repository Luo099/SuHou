"""Decode Tenhou's N.m bit field into physical tiles and seat relations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from suhou.tenhou.tiles import tile_name

MeldKind = Literal["chi", "pon", "minkan", "ankan", "kakan", "nuki"]


@dataclass(frozen=True, slots=True)
class Meld:
    caller: int
    kind: MeldKind
    from_seat: int
    tiles: tuple[int, ...]
    called_tile: int
    consumed_tiles: tuple[int, ...]
    raw_mask: int

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(tile_name(tile) for tile in self.tiles)


def decode_meld(caller: int, mask: int) -> Meld:
    """Decode all four-player call forms; preserve red-five physical IDs."""
    if not 0 <= caller < 4 or not 0 <= mask <= 0xFFFF:
        raise ValueError(f"Invalid call caller={caller}, mask={mask}")
    relative = mask & 0x0003
    from_seat = (caller + relative) % 4

    if mask & 0x0004:
        if relative != 3:
            raise ValueError(f"Chi must call the previous player's tile: mask={mask}")
        encoded = (mask & 0xFC00) >> 10
        called_index = encoded % 3
        base = encoded // 3
        suit, rank = divmod(base, 7)
        if suit >= 3:
            raise ValueError(f"Chi cannot use honor tiles: mask={mask}")
        copies = ((mask & 0x0018) >> 3, (mask & 0x0060) >> 5, (mask & 0x0180) >> 7)
        tiles = tuple((suit * 9 + rank + offset) * 4 + copies[offset] for offset in range(3))
        called = tiles[called_index]
        consumed = tuple(tile for index, tile in enumerate(tiles) if index != called_index)
        return Meld(caller, "chi", from_seat, tiles, called, consumed, mask)

    if mask & 0x0018:
        unused_copy = (mask & 0x0060) >> 5
        encoded = (mask & 0xFE00) >> 9
        called_index = encoded % 3
        kind = encoded // 3
        if kind >= 34:
            raise ValueError(f"Pon/kakan tile kind out of range: mask={mask}")
        triplet = tuple(4 * kind + copy for copy in range(4) if copy != unused_copy)
        if mask & 0x0008:
            if relative == 0:
                raise ValueError(f"Pon must call another player's tile: mask={mask}")
            called = triplet[called_index]
            consumed = tuple(tile for index, tile in enumerate(triplet) if index != called_index)
            return Meld(caller, "pon", from_seat, triplet, called, consumed, mask)
        if mask & 0x0010:
            added_tile = 4 * kind + unused_copy
            return Meld(caller, "kakan", from_seat, triplet, added_tile, (), mask)
        raise ValueError(f"Unrecognized pon/kakan mask: {mask}")

    if mask & 0x0020:
        encoded = (mask & 0xFF00) >> 8
        tile = encoded // 4 * 4 + encoded % 4
        if tile // 4 != 30:
            raise ValueError(f"Nuki must be a north tile: mask={mask}")
        return Meld(caller, "nuki", caller, (tile,), tile, (), mask)

    encoded = (mask & 0xFF00) >> 8
    called_copy = encoded % 4
    kind = encoded // 4
    if kind >= 34:
        raise ValueError(f"Kan tile kind out of range: mask={mask}")
    tiles = tuple(4 * kind + copy for copy in range(4))
    called = 4 * kind + called_copy
    if relative == 0:
        return Meld(caller, "ankan", caller, tiles, called, tiles, mask)
    consumed = tuple(tile for tile in tiles if tile != called)
    return Meld(caller, "minkan", from_seat, tiles, called, consumed, mask)
