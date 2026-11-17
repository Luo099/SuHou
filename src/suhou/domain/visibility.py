"""Viewer-specific projection of internal events.

Only fields in the whitelist below become visible. In particular, opponent
draws carry no tile, the round-start event carries just the viewer's own hand,
and result internals (hand, yaku, ura dora) never appear in history.

``red_fives`` comes from the parsed game settings, so a no-red-fives match
never marks a physical copy as red.
"""

from __future__ import annotations

from suhou.domain.events import (
    AgariPayload,
    CallPayload,
    DiscardPayload,
    DoraPayload,
    DrawPayload,
    EventType,
    GameEvent,
    RiichiAcceptedPayload,
    RoundStartPayload,
    RyuukyokuPayload,
    VisibleEvent,
)
from suhou.domain.tiles import Tile
from suhou.domain.types import MeldKind, MeldType


def _meld_type(kind: object) -> object:
    """Keep the observation-facing enum for the five four-player call forms."""
    try:
        return MeldType(kind)  # type: ignore[arg-type]
    except ValueError:
        return kind


def _tile(value: int, red_fives: bool) -> Tile:
    return Tile.from_physical(value, red_fives)


def project_visible(
    event: GameEvent, viewer: int, *, red_fives: bool = True
) -> VisibleEvent | None:
    """Return what ``viewer`` could see, or ``None`` for private/non-table events."""
    if not 0 <= viewer < 4:
        raise ValueError(f"viewer must be in 0..3, got {viewer}")
    if event.kind is EventType.METADATA:
        return None
    assert event.seq is not None

    payload = event.payload
    if event.kind is EventType.ROUND_START:
        assert isinstance(payload, RoundStartPayload)
        data = {
            "round_wind": payload.round_wind,
            "hand_number": payload.hand_number,
            "honba": payload.honba,
            "riichi_sticks": payload.riichi_sticks,
            "dealer": payload.dealer,
            "scores": payload.scores,
            "dora_indicators": (_tile(payload.dora_indicator, red_fives),),
            "initial_hand": tuple(
                sorted(
                    (_tile(tile, red_fives) for tile in payload.hands[viewer]),
                    key=lambda tile: tile.sort_key,
                )
            ),
        }
    elif event.kind is EventType.DRAW:
        assert isinstance(payload, DrawPayload)
        data = {
            "tile": (
                _tile(payload.tile, red_fives) if event.actor == viewer else None
            ),
            "replacement": payload.replacement,
        }
    elif event.kind is EventType.DISCARD:
        assert isinstance(payload, DiscardPayload)
        data = {
            "tile": _tile(payload.tile, red_fives),
            "tsumogiri": payload.tsumogiri,
            "riichi_declaration": payload.riichi_declaration,
        }
    elif event.kind is EventType.CALL:
        assert isinstance(payload, CallPayload)
        self_call = payload.kind in (MeldKind.ANKAN, MeldKind.NUKI)
        tiles = payload.tiles
        if payload.kind is MeldKind.KAKAN and payload.added_tile is not None:
            # The visible meld is the resulting four-tile set; the payload
            # keeps the original pon triplet and the added tile separately.
            tiles = tuple(sorted((*payload.tiles, payload.added_tile)))
        data = {
            "meld_type": _meld_type(payload.kind),
            "tiles": tuple(_tile(tile, red_fives) for tile in tiles),
            "source_player": None if self_call else payload.from_seat,
            "called_tile": (
                None
                if self_call or payload.called_tile is None
                else _tile(payload.called_tile, red_fives)
            ),
            "added_tile": (
                _tile(payload.added_tile, red_fives)
                if payload.added_tile is not None
                else None
            ),
        }
    elif event.kind is EventType.RIICHI_DECLARATION:
        data = {}
    elif event.kind is EventType.RIICHI_ACCEPTED:
        assert isinstance(payload, RiichiAcceptedPayload)
        data = {} if payload.scores is None else {"scores": payload.scores}
    elif event.kind is EventType.DORA:
        assert isinstance(payload, DoraPayload)
        data = {"indicator": _tile(payload.indicator, red_fives)}
    elif event.kind is EventType.RESULT:
        if isinstance(payload, AgariPayload):
            data = {
                "result": "agari",
                "deltas": payload.deltas,
                "scores": tuple(
                    score + delta
                    for score, delta in zip(
                        payload.scores_before, payload.deltas, strict=True
                    )
                ),
                "winners": ((payload.who, payload.from_who),),
            }
            if payload.winning_tile is not None:
                data["winning_tile"] = _tile(payload.winning_tile, red_fives)
        else:
            assert isinstance(payload, RyuukyokuPayload)
            data = {
                "result": "ryuukyoku",
                "deltas": payload.deltas,
                "scores": tuple(
                    score + delta
                    for score, delta in zip(
                        payload.scores_before, payload.deltas, strict=True
                    )
                ),
            }
            if payload.reason is not None:
                data["reason"] = payload.reason
    else:
        assert event.kind is EventType.CONNECTION
        data = {"connected": bool(payload.connected)}

    return VisibleEvent(event.seq, event.kind, event.actor, data)
