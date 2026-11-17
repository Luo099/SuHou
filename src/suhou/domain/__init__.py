"""Common mahjong domain model: types, events, state, and visibility.

This package has no protocol or presentation dependencies. The Tenhou adapter
produces :class:`GameEvent` records; the replay engine owns state; consumers
read immutable views and :func:`project_visible` output.
"""

from suhou.domain.codec import decode_event, encode_event, event_json
from suhou.domain.events import (
    SCHEMA_VERSION,
    AgariPayload,
    CallPayload,
    ConnectionPayload,
    DiscardPayload,
    DoraPayload,
    DrawPayload,
    EventPayload,
    EventType,
    GameEvent,
    GameEventKind,
    MetadataEntry,
    MetadataPayload,
    RiichiAcceptedPayload,
    RiichiDeclarationPayload,
    RoundStartPayload,
    RyuukyokuPayload,
    VisibleEvent,
    Yaku,
)
from suhou.domain.melds import Meld
from suhou.domain.state import (
    DiscardRecord,
    PlayerState,
    PlayerView,
    RoundState,
    RoundView,
)
from suhou.domain.tiles import (
    DRAGON_NAMES,
    WIND_NAMES,
    Tile,
    Wind,
    dora_from_indicator,
    is_red_five,
    tile_kind,
    tile_name,
    validate_tile,
)
from suhou.domain.types import (
    MatchType,
    MeldKind,
    MeldType,
    PlayerInfo,
    match_type_from_game_type,
    red_fives_from_game_type,
)
from suhou.domain.visibility import project_visible

__all__ = [
    "DRAGON_NAMES",
    "SCHEMA_VERSION",
    "WIND_NAMES",
    "AgariPayload",
    "CallPayload",
    "ConnectionPayload",
    "DiscardPayload",
    "DiscardRecord",
    "DoraPayload",
    "DrawPayload",
    "EventPayload",
    "EventType",
    "GameEvent",
    "GameEventKind",
    "MatchType",
    "Meld",
    "MeldKind",
    "MeldType",
    "MetadataEntry",
    "MetadataPayload",
    "PlayerInfo",
    "PlayerState",
    "PlayerView",
    "RiichiAcceptedPayload",
    "RiichiDeclarationPayload",
    "RoundStartPayload",
    "RoundState",
    "RoundView",
    "RyuukyokuPayload",
    "Tile",
    "VisibleEvent",
    "Wind",
    "Yaku",
    "decode_event",
    "dora_from_indicator",
    "encode_event",
    "event_json",
    "is_red_five",
    "match_type_from_game_type",
    "project_visible",
    "red_fives_from_game_type",
    "tile_kind",
    "tile_name",
    "validate_tile",
]
