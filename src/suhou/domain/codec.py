"""Versioned JSON codec for the internal event stream.

``events.jsonl`` is written with :func:`encode_event` and reloaded with
:func:`decode_event`. Decoding is strict: unknown or missing payload keys fail
at the exact event, so a truncated or hand-edited stream cannot silently replay
as a different game.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import StrEnum
from typing import Any

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
    MetadataEntry,
    MetadataPayload,
    RiichiAcceptedPayload,
    RiichiDeclarationPayload,
    RoundStartPayload,
    RyuukyokuPayload,
    Yaku,
)
from suhou.domain.melds import Meld
from suhou.domain.tiles import Wind, validate_tile
from suhou.domain.types import MatchType, MeldKind, PlayerInfo


def encode_value(value: Any) -> Any:
    """Encode frozen dataclasses/enums/sequences as JSON-compatible values."""
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: encode_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(item) for item in value]
    return value


def encode_event(event: GameEvent) -> dict[str, Any]:
    """Encode one internal event, including its schema version."""
    return {
        "schema_version": event.schema_version,
        "kind": event.kind.value,
        "source_index": event.source_index,
        "round_id": event.round_id,
        "seq": event.seq,
        "actor": event.actor,
        "payload": encode_value(event.payload),
    }


def event_json(event: GameEvent) -> str:
    return json.dumps(
        encode_event(event), ensure_ascii=False, allow_nan=False, sort_keys=True
    )


def _check_keys(data: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = expected - set(data)
    extra = set(data) - expected
    if missing:
        raise ValueError(f"{context} lacks {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown fields {sorted(extra)}")


def _ints(value: Any, context: str, length: int | None = None) -> tuple[int, ...]:
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise ValueError(f"{context} must be a list of integers")
    if length is not None and len(value) != length:
        raise ValueError(f"{context} must contain {length} integers")
    return tuple(value)


def _four(value: Any, context: str) -> tuple[int, int, int, int]:
    values = _ints(value, context, 4)
    return values  # type: ignore[return-value]


def _four_floats(value: Any, context: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{context} must contain four numbers")
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        raise ValueError(f"{context} must contain four numbers")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _meld(data: Any) -> Meld:
    if not isinstance(data, Mapping):
        raise ValueError("meld must be an object")
    _check_keys(
        data,
        {
            "caller",
            "kind",
            "from_seat",
            "tiles",
            "called_tile",
            "consumed_tiles",
            "raw_mask",
            "added_tile",
            "formed_seq",
            "updated_seq",
        },
        "meld",
    )
    return Meld(
        caller=int(data["caller"]),
        kind=MeldKind(data["kind"]),
        from_seat=int(data["from_seat"]),
        tiles=tuple(validate_tile(tile) for tile in _ints(data["tiles"], "meld.tiles")),
        called_tile=validate_tile(int(data["called_tile"])),
        consumed_tiles=tuple(
            validate_tile(tile) for tile in _ints(data["consumed_tiles"], "meld.consumed_tiles")
        ),
        raw_mask=int(data["raw_mask"]),
        added_tile=(
            validate_tile(int(data["added_tile"])) if data["added_tile"] is not None else None
        ),
        formed_seq=int(data["formed_seq"]) if data["formed_seq"] is not None else None,
        updated_seq=int(data["updated_seq"]) if data["updated_seq"] is not None else None,
    )


def _metadata(data: Mapping[str, Any]) -> MetadataPayload:
    _check_keys(data, {"entry", "source_tag", "data"}, "metadata payload")
    entry = MetadataEntry(data["entry"])
    raw = data["data"]
    if not isinstance(raw, Mapping):
        raise ValueError("metadata data must be an object")
    decoded = dict(raw)
    if entry is MetadataEntry.GAME_SETTINGS:
        _check_keys(
            raw,
            {"game_type", "lobby", "match_type", "red_fives"},
            "game_settings metadata",
        )
        decoded["match_type"] = (
            MatchType(raw["match_type"]) if raw["match_type"] is not None else None
        )
        decoded["red_fives"] = bool(raw["red_fives"])
    elif entry is MetadataEntry.ROSTER:
        _check_keys(raw, {"players", "complete"}, "roster metadata")
        players = raw["players"]
        if not isinstance(players, list):
            raise ValueError("roster players must be a list")
        decoded["players"] = tuple(
            PlayerInfo(
                name=str(player["name"]),
                dan=player["dan"],
                rate=player["rate"],
                sex=player["sex"],
            )
            for player in players
        )
        decoded["complete"] = bool(raw["complete"])
    elif entry is MetadataEntry.GAME_START:
        _check_keys(raw, {"dealer"}, "game_start metadata")
    else:
        _check_keys(raw, {"seed", "reference"}, "shuffle metadata")
    return MetadataPayload(entry, str(data["source_tag"]), decoded)


def _call(data: Mapping[str, Any]) -> CallPayload:
    _check_keys(
        data,
        {
            "kind",
            "from_seat",
            "tiles",
            "called_tile",
            "consumed_tiles",
            "added_tile",
            "target_seq",
            "source_mask",
        },
        "call payload",
    )
    return CallPayload(
        kind=MeldKind(data["kind"]),
        from_seat=int(data["from_seat"]) if data["from_seat"] is not None else None,
        tiles=tuple(validate_tile(tile) for tile in _ints(data["tiles"], "call.tiles")),
        called_tile=(
            validate_tile(int(data["called_tile"])) if data["called_tile"] is not None else None
        ),
        consumed_tiles=tuple(
            validate_tile(tile) for tile in _ints(data["consumed_tiles"], "call.consumed_tiles")
        ),
        added_tile=(
            validate_tile(int(data["added_tile"])) if data["added_tile"] is not None else None
        ),
        target_seq=int(data["target_seq"]) if data["target_seq"] is not None else None,
        source_mask=int(data["source_mask"]),
    )


def _agari(data: Mapping[str, Any]) -> AgariPayload:
    _check_keys(
        data,
        {
            "group_id",
            "group_final",
            "who",
            "from_who",
            "deltas",
            "scores_before",
            "winning_tile",
            "hand",
            "melds",
            "fu",
            "han",
            "points",
            "limit",
            "yaku",
            "dora_indicators",
            "uradora_indicators",
            "honba_after",
            "riichi_sticks_at_result",
            "final_scores",
            "final_points",
        },
        "agari payload",
    )
    melds = data["melds"]
    if not isinstance(melds, list):
        raise ValueError("agari melds must be a list")
    yaku = data["yaku"]
    if not isinstance(yaku, list):
        raise ValueError("agari yaku must be a list")
    return AgariPayload(
        group_id=int(data["group_id"]),
        group_final=bool(data["group_final"]),
        who=int(data["who"]),
        from_who=int(data["from_who"]),
        deltas=_four(data["deltas"], "agari deltas"),
        scores_before=_four(data["scores_before"], "agari scores_before"),
        winning_tile=(
            validate_tile(int(data["winning_tile"])) if data["winning_tile"] is not None else None
        ),
        hand=(
            tuple(validate_tile(tile) for tile in _ints(data["hand"], "agari hand"))
            if data["hand"] is not None
            else None
        ),
        melds=tuple(_meld(item) for item in melds),
        fu=int(data["fu"]) if data["fu"] is not None else None,
        han=int(data["han"]) if data["han"] is not None else None,
        points=int(data["points"]) if data["points"] is not None else None,
        limit=int(data["limit"]) if data["limit"] is not None else None,
        yaku=tuple(
            Yaku(id=int(item["id"]), name=str(item["name"]), han=int(item["han"]))
            for item in yaku
        ),
        dora_indicators=(
            tuple(validate_tile(tile) for tile in _ints(data["dora_indicators"], "doraHai"))
            if data["dora_indicators"] is not None
            else None
        ),
        uradora_indicators=(
            tuple(validate_tile(tile) for tile in _ints(data["uradora_indicators"], "doraHaiUra"))
            if data["uradora_indicators"] is not None
            else None
        ),
        honba_after=int(data["honba_after"]),
        riichi_sticks_at_result=int(data["riichi_sticks_at_result"]),
        final_scores=(
            _four(data["final_scores"], "agari final_scores")
            if data["final_scores"] is not None
            else None
        ),
        final_points=(
            _four_floats(data["final_points"], "agari final_points")
            if data["final_points"] is not None
            else None
        ),
    )


def _ryuukyoku(data: Mapping[str, Any]) -> RyuukyokuPayload:
    _check_keys(
        data,
        {
            "group_id",
            "group_final",
            "reason",
            "deltas",
            "scores_before",
            "honba_after",
            "riichi_sticks",
            "final_scores",
            "final_points",
        },
        "ryuukyoku payload",
    )
    return RyuukyokuPayload(
        group_id=int(data["group_id"]),
        group_final=bool(data["group_final"]),
        reason=str(data["reason"]) if data["reason"] is not None else None,
        deltas=_four(data["deltas"], "ryuukyoku deltas"),
        scores_before=_four(data["scores_before"], "ryuukyoku scores_before"),
        honba_after=int(data["honba_after"]),
        riichi_sticks=int(data["riichi_sticks"]),
        final_scores=(
            _four(data["final_scores"], "ryuukyoku final_scores")
            if data["final_scores"] is not None
            else None
        ),
        final_points=(
            _four_floats(data["final_points"], "ryuukyoku final_points")
            if data["final_points"] is not None
            else None
        ),
    )


def decode_payload(kind: EventType, data: Mapping[str, Any]) -> EventPayload:
    if not isinstance(data, Mapping):
        raise ValueError(f"{kind} payload must be an object")
    if kind is EventType.METADATA:
        return _metadata(data)
    if kind is EventType.ROUND_START:
        _check_keys(
            data,
            {
                "round_index",
                "round_wind",
                "hand_number",
                "honba",
                "riichi_sticks",
                "dealer",
                "dice",
                "scores",
                "hands",
                "dora_indicator",
            },
            "round_start payload",
        )
        hands = data["hands"]
        if not isinstance(hands, list):
            raise ValueError("round_start hands must be a list")
        return RoundStartPayload(
            round_index=int(data["round_index"]),
            round_wind=Wind(data["round_wind"]),
            hand_number=int(data["hand_number"]),
            honba=int(data["honba"]),
            riichi_sticks=int(data["riichi_sticks"]),
            dealer=int(data["dealer"]),
            dice=_ints(data["dice"], "round_start dice", 2),  # type: ignore[arg-type]
            scores=_four(data["scores"], "round_start scores"),
            hands=tuple(
                tuple(validate_tile(tile) for tile in _ints(hand, "round_start hand"))
                for hand in hands
            ),
            dora_indicator=validate_tile(int(data["dora_indicator"])),
        )
    if kind is EventType.DRAW:
        _check_keys(data, {"tile", "replacement"}, "draw payload")
        return DrawPayload(validate_tile(int(data["tile"])), bool(data["replacement"]))
    if kind is EventType.DISCARD:
        _check_keys(data, {"tile", "tsumogiri", "riichi_declaration"}, "discard payload")
        return DiscardPayload(
            validate_tile(int(data["tile"])),
            bool(data["tsumogiri"]),
            bool(data["riichi_declaration"]),
        )
    if kind is EventType.CALL:
        return _call(data)
    if kind is EventType.RIICHI_DECLARATION:
        _check_keys(data, set(), "riichi_declaration payload")
        return RiichiDeclarationPayload()
    if kind is EventType.RIICHI_ACCEPTED:
        _check_keys(data, {"scores"}, "riichi_accepted payload")
        return RiichiAcceptedPayload(
            None
            if data["scores"] is None
            else _four(data["scores"], "riichi_accepted scores")
        )
    if kind is EventType.DORA:
        _check_keys(data, {"indicator"}, "dora payload")
        return DoraPayload(validate_tile(int(data["indicator"])))
    if kind is EventType.CONNECTION:
        _check_keys(data, {"connected"}, "connection payload")
        return ConnectionPayload(bool(data["connected"]))
    assert kind is EventType.RESULT
    if "who" in data:
        return _agari(data)
    if "reason" in data:
        return _ryuukyoku(data)
    raise ValueError("result payload is neither agari nor ryuukyoku")


def decode_event(data: Mapping[str, Any]) -> GameEvent:
    """Decode one event object and reject unsupported schema versions."""
    _check_keys(
        data,
        {"schema_version", "kind", "source_index", "round_id", "seq", "actor", "payload"},
        "event",
    )
    version = data["schema_version"]
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported event schema_version {version}; expected {SCHEMA_VERSION}"
        )
    kind = EventType(data["kind"])
    payload = data["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("event payload must be an object")
    return GameEvent(
        kind=kind,
        source_index=int(data["source_index"]),
        round_id=int(data["round_id"]) if data["round_id"] is not None else None,
        seq=int(data["seq"]) if data["seq"] is not None else None,
        actor=int(data["actor"]) if data["actor"] is not None else None,
        payload=decode_payload(kind, payload),
        schema_version=int(version),
    )
