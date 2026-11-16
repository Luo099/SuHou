"""Tenhou's physical tile identifiers and display names."""

WIND_NAMES = ("东", "南", "西", "北")
DRAGON_NAMES = ("白", "发", "中")


def validate_tile(tile: int) -> int:
    """Reject values outside Tenhou's 136 physical tile identifiers."""
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


def tile_name(tile: int) -> str:
    """Render a tile without losing its red-five identity."""
    validate_tile(tile)
    if is_red_five(tile):
        return f"0{'mps'[tile // 36]}"
    kind = tile // 4
    if kind < 27:
        return f"{kind % 9 + 1}{'mps'[kind // 9]}"
    if kind < 31:
        return WIND_NAMES[kind - 27]
    return DRAGON_NAMES[kind - 31]
