"""Rules-agnostic replay engine over the unified event stream.

Only this module mutates table state. It validates each event against the
current state, applies it, and returns an immutable before/after view. It does
not know XML tags, attribute dictionaries, or call bit encodings.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from suhou.domain.events import (
    AgariPayload,
    CallPayload,
    ConnectionPayload,
    DiscardPayload,
    DoraPayload,
    DrawPayload,
    EventType,
    GameEvent,
    MetadataEntry,
    MetadataPayload,
    RiichiAcceptedPayload,
    RoundStartPayload,
    RyuukyokuPayload,
)
from suhou.domain.melds import Meld
from suhou.domain.state import PlayerState, RoundState, RoundView
from suhou.domain.tiles import tile_name
from suhou.domain.types import MatchType, MeldKind, PlayerInfo


class ReplayError(ValueError):
    """The recorded event stream cannot be applied to the current state."""


def _remove_tiles(hand: Counter[int], tiles: tuple[int, ...], context: str) -> None:
    missing = Counter(tiles) - hand
    if missing:
        raise ReplayError(f"{context}: tile(s) not present: {list(missing.elements())}")
    hand.subtract(tiles)


def _remove_player_tiles(player: PlayerState, tiles: tuple[int, ...], context: str) -> None:
    """Track physical tiles in hand; 'drawn' only remembers the latest draw."""
    _remove_tiles(player.hand, tiles, context)
    if player.drawn is not None and tiles.count(player.drawn):
        player.drawn = None


def _take_last_discard(round_: RoundState, payload: CallPayload, caller: int) -> None:
    if round_.last_discard is None:
        raise ReplayError(f"{payload.kind} requires a preceding opponent discard")
    seat, tile = round_.last_discard
    if seat != payload.from_seat or tile != payload.called_tile:
        raise ReplayError(
            f"Call source/tile ({payload.from_seat}/{payload.called_tile}) does not match "
            f"last discard {round_.last_discard}"
        )
    if not round_.players[seat].discards:
        raise ReplayError("Call has no recorded discard to claim")
    if round_.players[seat].discards[-1]["tile"] != tile_name(tile):
        raise ReplayError("Last discard no longer matches the call")
    round_.players[seat].discards[-1]["called_by"] = caller
    round_.last_discard = None


@dataclass(frozen=True, slots=True)
class Transition:
    """The immutable result of applying one event."""

    event: GameEvent
    before: RoundView | None
    after: RoundView | None
    message: str
    final_scores: tuple[int, int, int, int] | None
    final_points: tuple[float, float, float, float] | None


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    """One event, its state views, and the optional source record."""

    event: GameEvent
    before: RoundView | None
    after: RoundView | None
    message: str
    final_scores: tuple[int, int, int, int] | None
    final_points: tuple[float, float, float, float] | None
    source: Any | None = None


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    """A completed deterministic replay of an internal event stream."""

    events: tuple[GameEvent, ...]
    frames: tuple[ReplayFrame, ...]
    rounds: tuple[RoundView, ...]
    players: tuple[PlayerInfo, ...]
    game_type: int | None
    lobby: int | None
    match_type: MatchType | None
    scores: tuple[int, ...]
    final_scores: tuple[int, int, int, int] | None
    final_points: tuple[float, float, float, float] | None

    def summary(self) -> dict[str, Any]:
        return {
            "source_events": len(self.events),
            "rounds": len(self.rounds),
            "players": [player.name for player in self.players],
            "scores": list(self.scores),
            "final_scores": list(self.final_scores) if self.final_scores else None,
            "final_points": list(self.final_points) if self.final_points else None,
        }


class ReplayEngine:
    """Owns the only mutable table state in the pipeline."""

    def __init__(self) -> None:
        self.players: tuple[PlayerInfo, ...] = (
            PlayerInfo(""),
            PlayerInfo(""),
            PlayerInfo(""),
            PlayerInfo(""),
        )
        self.game_type: int | None = None
        self.lobby: int | None = None
        self.match_type: MatchType | None = None
        self.red_fives = True
        self.shuffle: str | None = None
        self.shuffle_reference: str | None = None
        self.game_start_dealer: int | None = None
        self.round: RoundState | None = None
        self.rounds: list[RoundState] = []
        self.events: list[GameEvent] = []
        self.final_scores: tuple[int, int, int, int] | None = None
        self.final_points: tuple[float, float, float, float] | None = None
        self._last_source_index = -1
        self._last_seq: int | None = None
        self._open_result_group: int | None = None

    def round_view(self) -> RoundView | None:
        return self.round.view() if self.round is not None else None

    def apply(self, event: GameEvent) -> Transition:
        """Validate and apply one event; state changes happen only here."""
        before = self.round_view()
        self._validate_envelope(event)
        message = self._dispatch(event)
        self.events.append(event)
        self._last_source_index = event.source_index
        if event.seq is not None:
            self._last_seq = event.seq
        after = self.round_view()
        return Transition(
            event,
            before,
            after,
            message,
            self.final_scores,
            self.final_points,
        )

    def finish(self, frames: tuple[ReplayFrame, ...] = ()) -> ReplayOutcome:
        """Validate that the stream is a complete game and freeze the result."""
        if self._open_result_group is not None:
            raise ReplayError("The game ends in an unfinished result group")
        if self.round is None:
            raise ReplayError("The game contains no INIT round")
        if self.final_scores is None or self.final_points is None:
            raise ReplayError("The game has no final owari result")
        if not self.round.finished:
            raise ReplayError("The final round has no result")
        return ReplayOutcome(
            events=tuple(self.events),
            frames=frames,
            rounds=tuple(round_.view() for round_ in self.rounds),
            players=self.players,
            game_type=self.game_type,
            lobby=self.lobby,
            match_type=self.match_type,
            scores=tuple(self.round.scores),
            final_scores=self.final_scores,
            final_points=self.final_points,
        )

    # -- envelope ---------------------------------------------------------

    def _validate_envelope(self, event: GameEvent) -> None:
        if event.source_index <= self._last_source_index:
            raise ReplayError(
                f"source_index {event.source_index} does not increase after "
                f"{self._last_source_index}"
            )
        if event.kind is EventType.METADATA:
            if self.final_scores is not None:
                raise ReplayError("Metadata cannot follow final owari")
            if self._open_result_group is not None:
                raise ReplayError("A result group cannot be interrupted by metadata")
            return
        if self.final_scores is not None:
            raise ReplayError("No table event can follow final owari")
        if self.round is None:
            if event.kind is not EventType.ROUND_START:
                raise ReplayError("A round event appeared before INIT")
            if event.round_id != 0:
                raise ReplayError(f"INIT must use round_id 0, got {event.round_id}")
        else:
            if event.kind is EventType.ROUND_START:
                if event.round_id != self.round.round_id + 1:
                    raise ReplayError(
                        f"INIT round_id {event.round_id} does not follow "
                        f"{self.round.round_id}"
                    )
            elif event.round_id != self.round.round_id:
                raise ReplayError(
                    f"Event round_id {event.round_id} differs from current round "
                    f"{self.round.round_id}"
                )
        if self._open_result_group is not None and event.kind is not EventType.RESULT:
            raise ReplayError("A result group cannot be interrupted by another action")
        if event.kind is EventType.ROUND_START:
            if event.seq != 0:
                raise ReplayError(f"INIT seq must be 0, got {event.seq}")
        elif self._last_seq is not None and event.seq <= self._last_seq:
            # Metadata events keep a source index but no round seq, so valid
            # round-local sequence numbers may contain gaps.
            raise ReplayError(
                f"Event seq {event.seq} is out of order; it follows {self._last_seq}"
            )

    # -- dispatch ---------------------------------------------------------

    def _dispatch(self, event: GameEvent) -> str:
        kind = event.kind
        if kind is EventType.METADATA:
            assert isinstance(event.payload, MetadataPayload)
            return self._apply_metadata(event.payload)
        if kind is EventType.ROUND_START:
            assert isinstance(event.payload, RoundStartPayload)
            return self._start_round(event, event.payload)
        round_ = self._require_round()
        if kind is EventType.DRAW:
            assert isinstance(event.payload, DrawPayload)
            return self._apply_draw(round_, event, event.payload)
        if kind is EventType.DISCARD:
            assert isinstance(event.payload, DiscardPayload)
            return self._apply_discard(round_, event, event.payload)
        if kind is EventType.CALL:
            assert isinstance(event.payload, CallPayload)
            return self._apply_call(round_, event, event.payload)
        if kind is EventType.RIICHI_DECLARATION:
            return self._apply_riichi_declaration(round_, event)
        if kind is EventType.RIICHI_ACCEPTED:
            assert isinstance(event.payload, RiichiAcceptedPayload)
            return self._apply_riichi_accepted(round_, event, event.payload)
        if kind is EventType.DORA:
            assert isinstance(event.payload, DoraPayload)
            return self._apply_dora(round_, event, event.payload)
        if kind is EventType.RESULT:
            if isinstance(event.payload, AgariPayload):
                return self._apply_result(round_, event, event.payload)
            assert isinstance(event.payload, RyuukyokuPayload)
            return self._apply_result(round_, event, event.payload)
        assert kind is EventType.CONNECTION and isinstance(
            event.payload, ConnectionPayload
        )
        return self._apply_connection(round_, event, event.payload)

    def _require_round(self) -> RoundState:
        if self.round is None:
            raise ReplayError("A round event appeared before INIT")
        return self.round

    # -- metadata ---------------------------------------------------------

    def _apply_metadata(self, payload: MetadataPayload) -> str:
        data = payload.data
        if payload.entry is MetadataEntry.GAME_SETTINGS:
            self.game_type = data.get("game_type")
            self.lobby = data.get("lobby")
            self.match_type = data.get("match_type")
            self.red_fives = bool(data.get("red_fives", True))
        elif payload.entry is MetadataEntry.ROSTER:
            if data.get("complete"):
                self.players = tuple(data.get("players", ()))
        elif payload.entry is MetadataEntry.GAME_START:
            dealer = data.get("dealer")
            if dealer is not None:
                self.game_start_dealer = int(dealer)
        else:
            self.shuffle = data.get("seed")
            self.shuffle_reference = data.get("reference")
        return f"牌谱元数据 {payload.source_tag}"

    # -- round start ------------------------------------------------------

    def _start_round(self, event: GameEvent, payload: RoundStartPayload) -> str:
        hands = payload.hands
        if self.round is not None and not self.round.finished:
            raise ReplayError("INIT cannot appear before the previous round has finished")
        if self.round is not None and tuple(self.round.scores) != payload.scores:
            raise ReplayError(
                f"INIT scores {payload.scores} differ from previous result {self.round.scores}"
            )
        if self.final_scores is not None:
            raise ReplayError("Cannot begin a new round after final owari")
        seed = (
            payload.round_index,
            payload.honba,
            payload.riichi_sticks,
            payload.dice[0] - 1,
            payload.dice[1] - 1,
            payload.dora_indicator,
        )
        round_ = RoundState(
            index=payload.round_index,
            seed=seed,
            dealer=payload.dealer,
            players=[
                PlayerState(
                    seat=seat, dealer=payload.dealer, hand=Counter(hand), initial_hand=hand
                )
                for seat, hand in enumerate(hands)
            ],
            scores=list(payload.scores),
            current_seat=payload.dealer,
            dora_indicators=[payload.dora_indicator],
            honba=payload.honba,
            riichi_sticks=payload.riichi_sticks,
            start_index=event.source_index,
            game_type=self.game_type,
            red_fives=self.red_fives,
            round_id=event.round_id if event.round_id is not None else len(self.rounds),
        )
        round_.match_type = self.match_type
        self.rounds.append(round_)
        self.round = round_
        self._last_seq = event.seq
        seats = " ".join(
            f"P{seat + 1}={player.name}" for seat, player in enumerate(self.players)
        )
        return (
            f"{round_.name} 开始，庄家 P{round_.dealer + 1}，骰子 {round_.dice}"
            f" | {seats}"
        )

    # -- actions ----------------------------------------------------------

    def _apply_draw(
        self, round_: RoundState, event: GameEvent, payload: DrawPayload
    ) -> str:
        if round_.finished:
            raise ReplayError("Cannot draw after round has finished")
        assert event.actor is not None
        seat = event.actor
        player = round_.players[seat]
        if round_.current_seat != seat:
            raise ReplayError(f"Draw by seat {seat}; expected seat {round_.current_seat}")
        if round_.awaiting_replacement_draw not in (None, seat):
            raise ReplayError("Kan replacement draw belongs to a different seat")
        if round_.phase == "discard" and player.drawn is None and seat == round_.current_seat:
            raise ReplayError("Expected a discard, not another draw")
        if round_.phase == "response" and player.drawn is None:
            round_.phase = "draw"
        if round_.phase != "draw" or player.drawn is not None or round_.pending_riichi is not None:
            raise ReplayError("Draw requires an empty hand and no pending riichi/call")
        replacement = round_.awaiting_replacement_draw == seat
        if payload.replacement != replacement:
            raise ReplayError(
                f"Draw replacement flag {payload.replacement} differs from replay {replacement}"
            )
        player.drawn = payload.tile
        player.hand[payload.tile] += 1
        round_.awaiting_replacement_draw = None
        round_.pending_kakan_tile = None
        if not replacement:
            round_.normal_draws += 1
            if round_.normal_draws > 70:
                raise ReplayError("More than 70 live-wall draws in one round")
        round_.phase = "discard"
        round_.last_discard = None
        suffix = "（岭上）" if replacement else ""
        return f"摸牌 {tile_name(payload.tile)}{suffix}"

    def _apply_discard(
        self, round_: RoundState, event: GameEvent, payload: DiscardPayload
    ) -> str:
        assert event.actor is not None
        seat = event.actor
        player = round_.players[seat]
        if round_.current_seat != seat or round_.phase != "discard":
            raise ReplayError(f"Discard by seat {seat} is not the current discard turn")
        if payload.riichi_declaration != (round_.pending_riichi == seat):
            raise ReplayError(
                "Discard riichi flag differs from the pending riichi state"
            )
        if payload.tsumogiri and player.drawn != payload.tile:
            raise ReplayError("Tsumogiri discard does not match the drawn tile")
        if player.riichi == "accepted" and not payload.tsumogiri:
            raise ReplayError("A riichi player must discard the drawn tile")
        _remove_player_tiles(player, (payload.tile,), "Discard")
        player.discards.append({
            "tile": tile_name(payload.tile),
            "tile_id": payload.tile,
            "riichi": payload.riichi_declaration,
            "tsumogiri": payload.tsumogiri,
            "seq": event.seq,
        })
        player.drawn = None
        round_.last_discard = (seat, payload.tile)
        round_.current_seat = (seat + 1) % 4
        round_.phase = "response"
        if payload.riichi_declaration:
            player.riichi = "declared"
        return (
            f"切牌 {tile_name(payload.tile)}"
            f"{'（立直宣言）' if payload.riichi_declaration else ''}"
        )

    def _apply_call(
        self, round_: RoundState, event: GameEvent, payload: CallPayload
    ) -> str:
        if round_.finished or round_.awaiting_replacement_draw is not None:
            raise ReplayError("Call is not valid after the round or before a replacement draw")
        assert event.actor is not None
        caller = event.actor
        player = round_.players[caller]
        seq = event.seq
        kind = payload.kind
        meld: Meld
        if kind is MeldKind.KAKAN:
            if round_.current_seat != caller or round_.phase != "discard" or player.drawn is None:
                raise ReplayError("Kakan requires the caller's drawn-tile turn")
            if payload.target_seq is None or payload.added_tile is None:
                raise ReplayError("Kakan has no matching previous pon")
            previous = next(
                (
                    item
                    for item in player.melds
                    if item.formed_seq == payload.target_seq and item.kind is MeldKind.PON
                ),
                None,
            )
            if previous is None:
                raise ReplayError("Kakan has no matching previous pon")
            if previous.called_tile != payload.called_tile:
                raise ReplayError("Kakan keeps the original pon's called tile")
            _remove_player_tiles(player, (payload.added_tile,), "Kakan")
            meld = Meld(
                caller,
                MeldKind.KAKAN,
                previous.from_seat,
                tuple(sorted((*previous.tiles, payload.added_tile))),
                previous.called_tile,
                (),
                payload.source_mask,
                added_tile=payload.added_tile,
                formed_seq=previous.formed_seq if previous.formed_seq is not None else seq,
                updated_seq=seq,
            )
            player.melds[player.melds.index(previous)] = meld
            player.drawn = None
            round_.awaiting_replacement_draw = caller
            round_.pending_kakan_tile = payload.added_tile
            round_.phase = "draw"
        elif kind is MeldKind.ANKAN:
            if round_.current_seat != caller or round_.phase != "discard" or player.drawn is None:
                raise ReplayError("Ankan requires the caller's drawn-tile turn")
            _remove_player_tiles(player, payload.tiles, "Ankan")
            meld = Meld(
                caller,
                MeldKind.ANKAN,
                caller,
                payload.tiles,
                payload.called_tile if payload.called_tile is not None else payload.tiles[0],
                payload.consumed_tiles,
                payload.source_mask,
                formed_seq=seq,
                updated_seq=seq,
            )
            player.melds.append(meld)
            player.drawn = None
            round_.awaiting_replacement_draw = caller
            round_.phase = "draw"
        elif kind is MeldKind.NUKI:
            if round_.current_seat != caller or round_.phase != "discard" or player.drawn is None:
                raise ReplayError("Nuki requires the caller's drawn-tile turn")
            _remove_player_tiles(player, payload.tiles, "Nuki")
            meld = Meld(
                caller,
                MeldKind.NUKI,
                caller,
                payload.tiles,
                payload.called_tile if payload.called_tile is not None else payload.tiles[0],
                payload.consumed_tiles,
                payload.source_mask,
                formed_seq=seq,
                updated_seq=seq,
            )
            player.melds.append(meld)
            player.drawn = None
            round_.awaiting_replacement_draw = caller
            round_.phase = "draw"
        else:
            if round_.current_seat == caller and player.drawn is not None:
                raise ReplayError("Cannot chi/pon/minkan on one's own turn")
            _take_last_discard(round_, payload, caller)
            _remove_player_tiles(player, payload.consumed_tiles, kind.value.capitalize())
            assert payload.from_seat is not None and payload.called_tile is not None
            meld = Meld(
                caller,
                kind,
                payload.from_seat,
                payload.tiles,
                payload.called_tile,
                payload.consumed_tiles,
                payload.source_mask,
                formed_seq=seq,
                updated_seq=seq,
            )
            player.melds.append(meld)
            player.drawn = None
            round_.current_seat = caller
            if kind is MeldKind.MINKAN:
                round_.awaiting_replacement_draw = caller
                round_.phase = "draw"
            else:
                round_.phase = "discard"
        if kind in (MeldKind.KAKAN, MeldKind.ANKAN, MeldKind.MINKAN, MeldKind.NUKI):
            round_.replacement_draws += 1
        added = (
            f" + {tile_name(payload.added_tile)}"
            if kind is MeldKind.KAKAN and payload.added_tile is not None
            else ""
        )
        source_seat = payload.from_seat if payload.from_seat is not None else caller
        names = " ".join(tile_name(tile) for tile in payload.tiles)
        return f"{kind.value} {names}{added} ← P{source_seat + 1}"

    def _apply_riichi_declaration(self, round_: RoundState, event: GameEvent) -> str:
        if round_.finished:
            raise ReplayError("Cannot declare riichi after the round has finished")
        assert event.actor is not None
        seat = event.actor
        player = round_.players[seat]
        if round_.current_seat != seat or round_.phase != "discard" or player.drawn is None:
            raise ReplayError("REACH step 1 requires a draw-then-discard turn")
        if round_.pending_riichi is not None or player.riichi != "none":
            raise ReplayError("Duplicate or conflicting riichi declaration")
        if any(meld.kind is not MeldKind.ANKAN for meld in player.melds):
            raise ReplayError("Riichi requires a closed hand")
        round_.pending_riichi = seat
        player.riichi = "pending"
        return "立直宣言（尚未扣点）"

    def _apply_riichi_accepted(
        self, round_: RoundState, event: GameEvent, payload: RiichiAcceptedPayload
    ) -> str:
        assert event.actor is not None
        seat = event.actor
        player = round_.players[seat]
        if round_.pending_riichi != seat or player.riichi != "declared":
            raise ReplayError("REACH step 2 has no matching declared discard")
        if not player.discards or not player.discards[-1]["riichi"]:
            raise ReplayError("REACH step 2 must follow its declaration discard")
        if payload.scores is None:
            raise ReplayError("REACH step 2 requires authoritative ten scores")
        expected = round_.scores.copy()
        expected[seat] -= 1000
        if tuple(expected) != payload.scores:
            raise ReplayError(f"REACH scores {payload.scores} differ from expected {expected}")
        round_.scores = expected
        round_.riichi_sticks += 1
        round_.pending_riichi = None
        player.riichi = "accepted"
        return "立直成立，供托 +1，点数 -1000"

    def _apply_dora(
        self, round_: RoundState, event: GameEvent, payload: DoraPayload
    ) -> str:
        if round_.finished:
            raise ReplayError("DORA cannot appear after the round has finished")
        if payload.indicator in round_.dora_indicators:
            raise ReplayError("DORA duplicates an existing indicator")
        round_.dora_indicators.append(payload.indicator)
        return f"新宝牌指示牌 {tile_name(payload.indicator)}"

    def _apply_connection(
        self, round_: RoundState, event: GameEvent, payload: ConnectionPayload
    ) -> str:
        assert event.actor is not None
        round_.players[event.actor].connected = payload.connected
        return "重连" if payload.connected else "断线"

    # -- results ----------------------------------------------------------

    def _apply_result(
        self,
        round_: RoundState,
        event: GameEvent,
        payload: AgariPayload | RyuukyokuPayload,
    ) -> str:
        if round_.finished:
            raise ReplayError("Result cannot appear after the round has finished")
        if (payload.final_scores is None) != (payload.final_points is None):
            raise ReplayError("Result owari requires both final scores and final points")
        if payload.final_scores is not None and not payload.group_final:
            raise ReplayError("The final result must close its settlement group")
        if self._open_result_group is None:
            self._open_result_group = payload.group_id
        elif self._open_result_group != payload.group_id:
            raise ReplayError("Consecutive results must share one settlement group")
        if tuple(round_.scores) != payload.scores_before:
            raise ReplayError(
                f"Result pre-scores {payload.scores_before} differ from replay {round_.scores}"
            )
        round_.scores = [
            score + delta
            for score, delta in zip(round_.scores, payload.deltas, strict=True)
        ]
        round_.honba = payload.honba_after

        if isinstance(payload, AgariPayload):
            message = self._apply_agari(round_, payload)
        else:
            round_.riichi_sticks = payload.riichi_sticks
            result: dict[str, Any] = {
                "type": "ryuukyoku",
                "reason": payload.reason,
                "deltas": list(payload.deltas),
                "riichi_sticks": payload.riichi_sticks,
                "group_id": payload.group_id,
            }
            message = f"流局 {payload.reason or '荒牌平局'}"
            round_.results.append(result)
            if payload.final_scores is not None and payload.final_points is not None:
                message = self._settle_final(round_, payload, result, message)

        round_.phase = "result"
        round_.pending_riichi = None
        if payload.group_final:
            round_.finished = True
            round_.pending_kakan_tile = None
            self._open_result_group = None
        return message

    def _apply_agari(self, round_: RoundState, payload: AgariPayload) -> str:
        who, from_who = payload.who, payload.from_who
        player = round_.players[who]
        winning_tile = payload.winning_tile
        if who == from_who:
            if player.drawn is None:
                raise ReplayError("Tsumo AGARI requires a current drawn tile")
            if winning_tile is None:
                raise ReplayError("Tsumo AGARI requires machi")
            if player.drawn != winning_tile:
                raise ReplayError("Tsumo machi does not match the current drawn tile")
        else:
            added = round_.pending_kakan_tile
            if added is not None and round_.awaiting_replacement_draw == from_who:
                # The added tile stays pending through the whole settlement group.
                if winning_tile is None:
                    winning_tile = added
                elif winning_tile != added:
                    raise ReplayError("Chankan machi does not match the added kan tile")
            elif round_.last_discard is None:
                raise ReplayError("Ron AGARI requires a winning discard")
            else:
                discard_seat, discard_tile = round_.last_discard
                if discard_seat != from_who:
                    raise ReplayError("AGARI fromWho does not match the winning discard")
                if winning_tile is None:
                    winning_tile = discard_tile
                elif winning_tile != discard_tile:
                    raise ReplayError("Ron machi does not match the winning discard tile")
        if payload.hand is not None:
            recorded_hand = Counter(payload.hand)
            if winning_tile not in recorded_hand:
                raise ReplayError("AGARI machi is absent from the winning hand")
            live_hand = Counter(player.hand)
            if who != from_who:
                live_hand[winning_tile] += 1
            if live_hand != recorded_hand:
                raise ReplayError(
                    f"AGARI hand {sorted(recorded_hand.elements())} differs from replay "
                    f"{sorted(live_hand.elements())}"
                )
        recorded_masks = sorted(meld.raw_mask for meld in payload.melds)
        actual_masks = sorted(meld.raw_mask for meld in player.melds)
        if recorded_masks != actual_masks:
            raise ReplayError(
                f"AGARI meld masks {recorded_masks} differ from replay {actual_masks}"
            )

        kind = "自摸" if who == from_who else "荣和"
        message = f"{kind} P{who + 1} ← P{from_who + 1}"
        result: dict[str, Any] = {
            "type": "agari",
            "who": who,
            "from_who": from_who,
            "deltas": list(payload.deltas),
            "riichi_sticks_at_result": payload.riichi_sticks_at_result,
            "group_id": payload.group_id,
        }
        details: list[str] = []
        if payload.fu is not None and payload.han is not None and payload.points is not None:
            result.update({
                "fu": payload.fu,
                "points": payload.points,
                "limit": payload.limit,
                "han": payload.han,
                "yaku": [
                    {"id": yaku.id, "name": yaku.name, "han": yaku.han}
                    for yaku in payload.yaku
                ],
            })
            details.append(f"{payload.fu}符{payload.han}飜{payload.points}点")
            if payload.yaku:
                details.append(
                    "役:" + " ".join(f"{yaku.name}({yaku.han})" for yaku in payload.yaku)
                )
        if payload.winning_tile is not None:
            result["winning_tile"] = tile_name(payload.winning_tile)
            details.append(f"和了牌 {result['winning_tile']}")
        if payload.hand is not None:
            result["hand"] = [tile_name(tile) for tile in sorted(payload.hand)]
            details.append("手牌 " + " ".join(result["hand"]))
        if payload.dora_indicators is not None:
            if list(payload.dora_indicators) != round_.dora_indicators:
                raise ReplayError(
                    f"AGARI doraHai {payload.dora_indicators} differs from replay "
                    f"{round_.dora_indicators}"
                )
            result["dora_indicators"] = [
                tile_name(tile) for tile in payload.dora_indicators
            ]
        if payload.uradora_indicators is not None:
            round_.uradora_indicators = list(payload.uradora_indicators)
            result["uradora_indicators"] = [
                tile_name(tile) for tile in round_.uradora_indicators
            ]
        round_.riichi_sticks = 0
        round_.results.append(result)
        if details:
            message += " | " + " | ".join(details)
        if payload.final_scores is not None and payload.final_points is not None:
            message = self._settle_final(round_, payload, result, message)
        return message

    def _settle_final(
        self,
        round_: RoundState,
        payload: AgariPayload | RyuukyokuPayload,
        result: dict[str, Any],
        message: str,
    ) -> str:
        assert payload.final_scores is not None and payload.final_points is not None
        expected_scores = list(round_.scores)
        if isinstance(payload, RyuukyokuPayload) and payload.riichi_sticks:
            # At game end, Tenhou awards riichi sticks still on the table to
            # first place. The final <sc> omits this transfer, while <owari>
            # includes it. Ties follow seat order (the log's first dealer is
            # seat 0, so seat order is the tie-break).
            first_place = max(
                range(4), key=lambda seat: (expected_scores[seat], -seat)
            )
            award = 1000 * payload.riichi_sticks
            expected_scores[first_place] += award
            result["terminal_riichi_sticks"] = {
                "seat": first_place,
                "sticks": payload.riichi_sticks,
                "points": award,
            }
            round_.riichi_sticks = 0
            message += f" | 终局供托 {award} 点归 P{first_place + 1}"
        if tuple(expected_scores) != payload.final_scores:
            raise ReplayError(
                f"owari scores {payload.final_scores} differ from replay "
                f"{tuple(expected_scores)}"
            )
        round_.scores = list(payload.final_scores)
        round_.final_scores = payload.final_scores
        round_.final_points = payload.final_points
        self.final_scores = payload.final_scores
        self.final_points = payload.final_points
        return message


def replay_events(events: Iterable[GameEvent]) -> ReplayOutcome:
    """Replay an iterable of :class:`GameEvent` without any XML source."""
    engine = ReplayEngine()
    frames: list[ReplayFrame] = []
    for event in events:
        try:
            transition = engine.apply(event)
        except (ReplayError, ValueError) as error:
            raise ReplayError(
                f"event {event.source_index} ({event.kind}) "
                f"round {event.round_id}: {error}"
            ) from error
        frames.append(
            ReplayFrame(
                event=transition.event,
                before=transition.before,
                after=transition.after,
                message=transition.message,
                final_scores=transition.final_scores,
                final_points=transition.final_points,
            )
        )
    return engine.finish(tuple(frames))
