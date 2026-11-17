"""Tenhou-compatible re-exports of the shared tile model."""

from __future__ import annotations

from suhou.domain.tiles import (
    DRAGON_NAMES,
    WIND_NAMES,
    Wind,
    dora_from_indicator,
    is_red_five,
    tile_kind,
    tile_name,
    validate_tile,
)

__all__ = [
    "DRAGON_NAMES",
    "WIND_NAMES",
    "Wind",
    "dora_from_indicator",
    "is_red_five",
    "tile_kind",
    "tile_name",
    "validate_tile",
]
