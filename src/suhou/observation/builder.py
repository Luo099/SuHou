"""Build an Observation v1 snapshot from live replay state.

The builder deliberately reads field by field. A replay snapshot contains all
four hands and result internals; exposing that object wholesale would leak
hidden information and future results.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from suhou.observation.model import (
    DecisionContext,
    Discard,
    MatchType,
    Meld,
    MeldType,
    Observation,
    PlayerObservation,
    RiichiState,
    SelfObservation,
    TableObservation,
    Tile,
    VisibleEvent,
)
from suhou.tenhou.melds import Meld as ReplayMeld
from suhou.tenhou.replay import RoundState

# Tenhou <GO type> rule bits (community protocol knowledge): 0x01 vs human,
# 0x02 no red fives, 0x04 no kuitan, 0x08 hanchan, 0x10 sanma, 0x40 soku.
# The bundled sample has type=1 (tonpu); it reaches 南1 through 南入 because
# nobody had 30000 at the end of 東4.
_GO_HANCHAN = 0x08
_GO_SANMA = 0x10


class ObservationError(ValueError):
    """The replay/rules state cannot produce a contract-compliant observation."""


@runtime_checkable
class RulesState(Protocol):
    """The slice of the reused rules engine that observation v1 consumes.

    The full rules engine is integrated later. Until then callers supply an
    adapter for each decision; :class:`RulesSnapshot` is the minimal concrete
    form used by tests and simple callers.
    """

    def furiten(self, viewer: int) -> bool:
        """Return whether ``viewer`` is currently in furiten."""
        ...


@dataclass(frozen=True, slots=True)
class RulesSnapshot:
    """A minimal, explicit rules-engine snapshot for one decision.

    There is deliberately no default: a silent ``False`` would look like a
    real furiten answer. Callers must supply the rules engine's four values.
    """

    furiten_by_seat: tuple[bool, bool, bool, bool]

    def __post_init__(self) -> None:
        if len(self.furiten_by_seat) != 4:
            raise ValueError("furiten_by_seat must contain four seats")

    def furiten(self, viewer: int) -> bool:
        if not 0 <= viewer < 4:
            raise ValueError(f"viewer must be in 0..3, got {viewer}")
        return self.furiten_by_seat[viewer]


def match_type_from_game_type(game_type: int | None) -> MatchType:
    """Decode tonpu/hanchan from the source's <GO type> rule bits."""
    if game_type is None:
        raise ObservationError(
            "Cannot determine tonpu/hanchan: the game has no <GO type> rule bits"
        )
    if game_type & _GO_SANMA:
        raise ObservationError("Three-player logs are outside Observation v1")
    return MatchType.HANCHAN if game_type & _GO_HANCHAN else MatchType.TONPU


def _tile(tile: int) -> Tile:
    return Tile.from_physical(tile)


def _discard(record: dict[str, object]) -> Discard:
    try:
        tile_id = record["tile_id"]
        seq = record["seq"]
    except KeyError as error:
        raise ObservationError(
            f"Replay discard lacks {error.args[0]!r}; rebuild the state with the current replay"
        ) from error
    if not isinstance(tile_id, int) or not isinstance(seq, int):
        raise ObservationError("Replay discard tile_id/seq must be integers")
    tsumogiri = record.get("tsumogiri")
    riichi = record.get("riichi")
    if tsumogiri is None or riichi is None:
        raise ObservationError("Replay discard lacks tsumogiri/riichi")
    called_by = record.get("called_by")
    if called_by is not None and not isinstance(called_by, int):
        raise ObservationError("Replay discard called_by must be an integer or None")
    return Discard(
        tile=_tile(tile_id),
        tsumogiri=bool(tsumogiri),
        riichi_declaration=bool(riichi),
        called_by=called_by,
        seq=seq,
    )


def _meld(meld: ReplayMeld) -> Meld:
    if meld.kind == "nuki":
        raise ObservationError("Nuki (three-player) is outside Observation v1")
    try:
        kind = MeldType(meld.kind)
    except ValueError as error:
        raise ObservationError(f"Unknown replay meld kind: {meld.kind!r}") from error
    if meld.formed_seq is None or meld.updated_seq is None:
        raise ObservationError(
            "Replay meld lacks sequence numbers; rebuild with the current replay"
        )
    if kind == MeldType.ANKAN:
        source = None
        called = None
        added = None
    else:
        if meld.called_tile is None:
            raise ObservationError(f"{kind} lacks a called tile in replay state")
        source = meld.from_seat
        called = _tile(meld.called_tile)
        added = _tile(meld.added_tile) if kind == MeldType.KAKAN else None
        if kind == MeldType.KAKAN and meld.added_tile is None:
            raise ObservationError("Kakan lacks its added tile in replay state")
    return Meld(
        type=kind,
        tiles=tuple(_tile(tile) for tile in meld.tiles),
        source_player=source,
        called_tile=called,
        added_tile=added,
        formed_seq=meld.formed_seq,
        updated_seq=meld.updated_seq,
    )


def _hand(tiles: Iterable[int]) -> tuple[Tile, ...]:
    return tuple(sorted((_tile(tile) for tile in tiles), key=lambda tile: tile.sort_key))


def build_observation(
    replay_state: RoundState,
    rules_state: RulesState,
    visible_history: Sequence[VisibleEvent],
    viewer: int,
    decision_context: DecisionContext,
) -> Observation:
    """Capture the current player's visible table before the decision acts.

    ``visible_history`` must already be the viewer-filtered prefix ending just
    before the decision event (see :meth:`RoundHistory.view`). All returned
    records are independent copies; later replay mutations do not affect them.
    """
    if not 0 <= viewer < 4:
        raise ObservationError(f"viewer must be in 0..3, got {viewer}")

    me = replay_state.players[viewer]
    hand = _hand(me.hand.elements())
    drawn_tile = _tile(me.drawn) if me.drawn is not None else None
    if drawn_tile is not None and drawn_tile not in hand:
        raise ObservationError("Replay drawn tile is not present in the current hand")

    table = TableObservation(
        match_type=match_type_from_game_type(replay_state.game_type),
        round_wind=replay_state.round_wind,
        hand_number=replay_state.seed[0] % 4 + 1,
        honba=replay_state.honba,
        riichi_sticks=replay_state.riichi_sticks,
        dealer=replay_state.dealer,
        remaining_tiles=replay_state.wall_remaining,
        dora_indicators=tuple(_tile(tile) for tile in replay_state.dora_indicators),
    )

    players = tuple(
        PlayerObservation(
            seat_wind=player.seat_wind,
            score=replay_state.scores[seat],
            concealed_tile_count=sum(player.hand.values()),
            riichi=_riichi(player.riichi, seat),
            discards=tuple(_discard(record) for record in player.discards),
            melds=tuple(_meld(meld) for meld in player.melds),
        )
        for seat, player in enumerate(replay_state.players)
    )

    return Observation(
        viewer=viewer,
        table=table,
        self_view=SelfObservation(
            hand=hand,
            drawn_tile=drawn_tile,
            furiten=bool(rules_state.furiten(viewer)),
        ),
        players=players,
        context=decision_context,
        history=tuple(visible_history),
    )


def _riichi(value: str, seat: int) -> RiichiState:
    try:
        return RiichiState(value)
    except ValueError as error:
        raise ObservationError(f"Unknown riichi state {value!r} for seat {seat}") from error


__all__ = [
    "ObservationError",
    "RulesSnapshot",
    "RulesState",
    "build_observation",
    "match_type_from_game_type",
]
