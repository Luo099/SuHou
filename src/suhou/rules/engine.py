"""Event-driven rule engine for one designated viewer seat.

``PlayerRuleEngine.update(event, after)`` runs after the replay has already
applied one event. It never mutates :class:`RoundView`; it only refreshes its
own derived state (furiten, ippatsu, pending ron opportunities) and returns the
legal actions for the viewer at that boundary. Other seats update the public
table but never produce candidates.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from suhou.domain.events import (
    AgariPayload,
    CallPayload,
    DiscardPayload,
    DrawPayload,
    EventType,
    GameEvent,
    RyuukyokuPayload,
)
from suhou.domain.state import PlayerView, RoundView
from suhou.domain.tiles import Tile
from suhou.domain.types import MeldKind
from suhou.rules.agari import has_yaku, meld_shapes
from suhou.rules.calls import is_closed, kan_count, pon_kinds
from suhou.rules.furiten import FuritenState
from suhou.rules.model import (
    AnkanAction,
    ChiAction,
    DiscardAction,
    KakanAction,
    KyushuAction,
    MinkanAction,
    PassAction,
    PonAction,
    RiichiAction,
    RonAction,
    RuleAction,
    RuleDecision,
    RuleDecisionKind,
    RuleError,
    TsumoAction,
    WinContext,
)
from suhou.rules.shanten import tenpai_discards, winning_tiles
from suhou.rules.tiles import (
    ORPHAN_KINDS,
    ankan_options,
    chi_options,
    counts_from_physical,
    kakan_options,
    minkan_options,
    physical_index,
    pon_options,
    tile_index,
)

_RIICHI_ACTIVE = frozenset({"declared", "accepted"})


@dataclass(slots=True)
class _PendingRon:
    """A legal ron opportunity that the next event may prove was passed."""

    kind: str  # "discard" or "chankan"
    source_seat: int
    riichi: bool
    viewer_won: bool = False


class PlayerRuleEngine:
    """Compute viewer-only legal actions after each applied event."""

    def __init__(self, viewer: int) -> None:
        if not 0 <= viewer < 4:
            raise ValueError(f"viewer must be in 0..3, got {viewer}")
        self.viewer = viewer
        self._round_id: int = -1
        self._view: RoundView | None = None
        self._furiten = FuritenState()
        self._pending: _PendingRon | None = None
        self._ippatsu = False
        self._ippatsu_interrupted = False
        self._riichi_acceptance_pending = False
        self._calls_seen = 0
        self._viewer_draws = 0

    # -- public rules-state protocol ---------------------------------------

    def furiten(self, viewer: int) -> bool:
        if viewer != self.viewer:
            raise ValueError("the rule engine only answers for its own viewer")
        return self._furiten.is_furiten(self._current_waits(), self._discard_kinds())

    # -- event loop --------------------------------------------------------

    def update(self, event: GameEvent, after: RoundView | None) -> RuleDecision | None:
        """Refresh derived state for one applied event and return candidates."""
        if event.kind is EventType.METADATA:
            return None
        if event.kind is EventType.ROUND_START:
            self._reset(event, after)
            return None
        if after is None:
            raise RuleError(f"{event.kind} event has no post-event round view")
        if event.round_id is None or event.seq is None:
            raise RuleError(f"{event.kind} event is not round-local")
        if event.round_id != self._round_id:
            raise RuleError(
                f"event round_id {event.round_id} does not match engine round "
                f"{self._round_id}; a round_start event is missing"
            )

        self._resolve_pending(event)
        self._apply_event_flags(event, after)
        self._view = after
        return self._make_decision(event, after)

    def _reset(self, event: GameEvent, after: RoundView | None) -> None:
        self._round_id = event.round_id if event.round_id is not None else -1
        self._view = after
        self._furiten.reset()
        self._pending = None
        self._ippatsu = False
        self._ippatsu_interrupted = False
        self._riichi_acceptance_pending = False
        self._calls_seen = 0
        self._viewer_draws = 0

    def _apply_event_flags(self, event: GameEvent, after: RoundView) -> None:
        player = after.players[self.viewer]
        if event.kind is EventType.DRAW:
            if event.actor == self.viewer:
                self._furiten.on_draw()
                self._viewer_draws += 1
        elif event.kind is EventType.DISCARD:
            if event.actor == self.viewer:
                payload = event.payload
                assert isinstance(payload, DiscardPayload)
                if payload.riichi_declaration:
                    # The declaration discard has just been made. Ippatsu starts
                    # here; a call before the acceptance event still cancels it.
                    self._ippatsu = not self._ippatsu_interrupted
                elif player.riichi in _RIICHI_ACTIVE:
                    self._ippatsu = False
        elif event.kind is EventType.CALL:
            self._calls_seen += 1
            self._ippatsu = False
            if self._riichi_acceptance_pending:
                self._ippatsu_interrupted = True
        elif event.kind is EventType.RIICHI_DECLARATION:
            if event.actor == self.viewer:
                self._riichi_acceptance_pending = True
                self._ippatsu = False
                self._ippatsu_interrupted = False
        elif event.kind is EventType.RIICHI_ACCEPTED and event.actor == self.viewer:
            self._riichi_acceptance_pending = False
            self._ippatsu = not self._ippatsu_interrupted
            self._ippatsu_interrupted = False

    def _resolve_pending(self, event: GameEvent) -> None:
        pending = self._pending
        if pending is None:
            return
        if event.kind is EventType.RESULT:
            payload = event.payload
            if isinstance(payload, AgariPayload):
                if payload.who == self.viewer and payload.from_who == pending.source_seat:
                    pending.viewer_won = True
                if pending.viewer_won:
                    self._pending = None
                elif payload.group_final:
                    self._mark_missed(pending)
                return
            if isinstance(payload, RyuukyokuPayload) and payload.group_final:
                self._mark_missed(pending)
            return
        if event.kind is EventType.CONNECTION:
            return
        if pending.kind == "chankan" and event.kind is EventType.DORA:
            # The new dora indicator is revealed before the replacement draw;
            # the chankan window is still open.
            return
        self._mark_missed(pending)

    def _mark_missed(self, pending: _PendingRon) -> None:
        self._furiten.on_missed_ron(riichi=pending.riichi)
        self._pending = None

    # -- decision dispatch -------------------------------------------------

    def _make_decision(
        self, event: GameEvent, after: RoundView
    ) -> RuleDecision | None:
        payload = event.payload
        if event.kind is EventType.DRAW and event.actor == self.viewer:
            assert isinstance(payload, DrawPayload)
            return self._turn_decision(after, event.seq, rinshan=payload.replacement)
        if event.kind is EventType.RIICHI_DECLARATION and event.actor == self.viewer:
            return self._riichi_discard_decision(after, event.seq)
        if event.kind is EventType.CALL and event.actor == self.viewer:
            assert isinstance(payload, CallPayload)
            if payload.kind in (MeldKind.CHI, MeldKind.PON):
                return self._after_call_decision(after, payload, event.seq)
            if payload.kind is MeldKind.NUKI:
                raise RuleError("Nuki (three-player) is outside the four-player rules")
            return None
        if event.kind is EventType.DISCARD and event.actor != self.viewer:
            assert isinstance(payload, DiscardPayload)
            return self._response_decision(after, event, event.seq)
        if (
            event.kind is EventType.CALL
            and event.actor != self.viewer
            and isinstance(payload, CallPayload)
            and payload.kind is MeldKind.KAKAN
        ):
            return self._kan_response_decision(after, event, event.seq)
        return None

    # -- turn decisions ----------------------------------------------------

    def _turn_decision(
        self, view: RoundView, trigger_seq: int, *, rinshan: bool
    ) -> RuleDecision | None:
        player = view.players[self.viewer]
        if player.drawn is None:
            raise RuleError("A draw decision requires the viewer's drawn tile")
        hand = _tiles(player.hand, view.red_fives)
        counts = counts_from_physical(player.hand)
        meld_count = len(player.melds)
        closed = is_closed(player.melds)
        riichi_active = player.riichi in _RIICHI_ACTIVE
        drawn_index = physical_index(player.drawn)
        actions: list[RuleAction] = []

        if self._can_tsumo(view, counts, drawn_index, rinshan=rinshan):
            actions.append(TsumoAction())

        if (
            player.riichi == "none"
            and view.pending_riichi is None
            and closed
            and view.scores[self.viewer] >= 1000
            and view.wall_remaining >= 4
            and tenpai_discards(counts, meld_count, closed=closed)
        ):
            actions.append(RiichiAction())

        if self._can_kyushu(view, counts):
            actions.append(KyushuAction())

        actions.extend(
            AnkanAction(consumed)
            for consumed in ankan_options(hand)
            if kan_count(player.melds) < 4
            and view.wall_remaining > 0
            and (not riichi_active or self._ankan_keeps_wait(player, consumed))
        )

        if player.riichi == "none":
            actions.extend(
                KakanAction(added, target_seq)
                for added, target_seq in kakan_options(hand, pon_kinds(player.melds))
                if kan_count(player.melds) < 4 and view.wall_remaining > 0
            )

        if riichi_active:
            actions.append(DiscardAction(Tile.from_physical(player.drawn, view.red_fives)))
        else:
            actions.extend(DiscardAction(tile) for tile in _distinct_tiles(hand))

        return self._finalize(actions, RuleDecisionKind.TURN, trigger_seq)

    def _riichi_discard_decision(
        self, view: RoundView, trigger_seq: int
    ) -> RuleDecision | None:
        player = view.players[self.viewer]
        if player.drawn is None:
            raise RuleError("A riichi declaration decision requires a drawn tile")
        hand = _tiles(player.hand, view.red_fives)
        counts = counts_from_physical(player.hand)
        meld_count = len(player.melds)
        closed = is_closed(player.melds)
        allowed = set(tenpai_discards(counts, meld_count, closed=closed))
        actions = [
            DiscardAction(tile) for tile in _distinct_tiles(hand) if tile_index(tile) in allowed
        ]
        return self._finalize(actions, RuleDecisionKind.RIICHI_DISCARD, trigger_seq)

    def _after_call_decision(
        self, view: RoundView, payload: CallPayload, trigger_seq: int
    ) -> RuleDecision | None:
        player = view.players[self.viewer]
        if payload.called_tile is None:
            raise RuleError("Chi/pon decision has no called tile")
        called_index = physical_index(payload.called_tile)
        hand = _tiles(player.hand, view.red_fives)
        actions = [
            DiscardAction(tile)
            for tile in _distinct_tiles(hand)
            if tile_index(tile) != called_index
        ]
        return self._finalize(actions, RuleDecisionKind.AFTER_CALL, trigger_seq)

    def _response_decision(
        self, view: RoundView, event: GameEvent, trigger_seq: int
    ) -> RuleDecision | None:
        payload = event.payload
        assert isinstance(payload, DiscardPayload) and event.actor is not None
        if view.last_discard != (event.actor, payload.tile):
            raise RuleError("Response decision does not match the current last discard")
        player = view.players[self.viewer]
        hand = _tiles(player.hand, view.red_fives)
        counts = counts_from_physical(player.hand)
        called_index = physical_index(payload.tile)
        riichi = player.riichi in _RIICHI_ACTIVE

        can_ron = self._can_ron(
            view, counts, called_index, houtei=view.wall_remaining == 0
        )
        actions: list[RuleAction] = []
        if can_ron:
            actions.append(RonAction())

        if player.riichi == "none":
            if event.actor == (self.viewer - 1) % 4:
                actions.extend(
                    ChiAction(consumed)
                    for consumed in chi_options(hand, called_index)
                )
            actions.extend(
                PonAction(consumed) for consumed in pon_options(hand, called_index)
            )
            if kan_count(player.melds) < 4 and view.wall_remaining > 0:
                actions.extend(
                    MinkanAction(consumed)
                    for consumed in minkan_options(hand, called_index)
                )

        if actions:
            actions.append(PassAction())
            if can_ron:
                self._pending = _PendingRon("discard", event.actor, riichi=riichi)
            return self._finalize(actions, RuleDecisionKind.RESPONSE, trigger_seq)
        return None

    def _kan_response_decision(
        self, view: RoundView, event: GameEvent, trigger_seq: int
    ) -> RuleDecision | None:
        payload = event.payload
        assert isinstance(payload, CallPayload) and event.actor is not None
        if payload.added_tile is None:
            raise RuleError("Kakan response has no added tile")
        player = view.players[self.viewer]
        counts = counts_from_physical(player.hand)
        winning_index = physical_index(payload.added_tile)
        if not self._can_ron(view, counts, winning_index, chankan=True):
            return None
        actions: tuple[RuleAction, ...] = (RonAction(), PassAction())
        self._pending = _PendingRon(
            "chankan", event.actor, riichi=player.riichi in _RIICHI_ACTIVE
        )
        return self._finalize(actions, RuleDecisionKind.KAN_RESPONSE, trigger_seq)

    # -- winning checks ----------------------------------------------------

    def _can_tsumo(
        self, view: RoundView, counts: Sequence[int], winning_tile: int, *, rinshan: bool
    ) -> bool:
        player = view.players[self.viewer]
        melds = meld_shapes(player.melds)
        closed = is_closed(player.melds)
        riichi = player.riichi in _RIICHI_ACTIVE
        context = WinContext(
            riichi=riichi,
            ippatsu=self._ippatsu and riichi,
            tsumo=True,
            rinshan=rinshan,
            haitei=view.wall_remaining == 0,
            tenhou=(
                self.viewer == view.dealer
                and self._viewer_draws == 1
                and self._calls_seen == 0
                and closed
            ),
            chiihou=(
                self.viewer != view.dealer
                and self._viewer_draws == 1
                and self._calls_seen == 0
                and closed
            ),
        )
        return has_yaku(
            counts,
            melds,
            winning_tile,
            context,
            seat_wind=player.seat_wind,
            round_wind=view.round_wind,
        )

    def _can_ron(
        self,
        view: RoundView,
        counts: Sequence[int],
        winning_tile: int,
        *,
        chankan: bool = False,
        houtei: bool = False,
    ) -> bool:
        player = view.players[self.viewer]
        winning_counts = list(counts)
        winning_counts[winning_tile] += 1
        melds = meld_shapes(player.melds)
        riichi = player.riichi in _RIICHI_ACTIVE
        if not has_yaku(
            winning_counts,
            melds,
            winning_tile,
            WinContext(
                riichi=riichi,
                ippatsu=self._ippatsu and riichi,
                chankan=chankan,
                houtei=houtei,
            ),
            seat_wind=player.seat_wind,
            round_wind=view.round_wind,
        ):
            return False
        return not self._furiten.is_furiten(
            self._current_waits(), self._discard_kinds()
        )

    # -- state helpers -----------------------------------------------------

    def _can_kyushu(self, view: RoundView, counts: Sequence[int]) -> bool:
        player = view.players[self.viewer]
        if (
            self._viewer_draws != 1
            or self._calls_seen != 0
            or player.melds
            or player.riichi != "none"
        ):
            return False
        return sum(1 for index in ORPHAN_KINDS if counts[index] > 0) >= 9

    def _ankan_keeps_wait(
        self, player: PlayerView, consumed: tuple[Tile, ...]
    ) -> bool:
        if player.drawn is None:
            return False
        kind = tile_index(consumed[0])
        before = counts_from_physical(player.hand)
        after = list(before)
        after[kind] -= 4
        if after[kind] < 0:
            return False
        waiting = list(before)
        waiting[physical_index(player.drawn)] -= 1
        meld_count = len(player.melds)
        closed = is_closed(player.melds)
        current = set(winning_tiles(waiting, meld_count, closed=closed))
        return bool(current) and current == set(
            winning_tiles(after, meld_count + 1, closed=closed)
        )

    def _current_waits(self) -> tuple[int, ...]:
        view = self._view
        if view is None:
            return ()
        player = view.players[self.viewer]
        counts = counts_from_physical(player.hand)
        meld_count = len(player.melds)
        closed = is_closed(player.melds)
        waiting_count = 13 - 3 * meld_count
        if sum(counts) == waiting_count:
            return winning_tiles(counts, meld_count, closed=closed)
        if (
            sum(counts) == waiting_count + 1
            and player.drawn is not None
        ):
            # A turn or declaration snapshot has no chosen discard yet. Use the
            # tsumogiri wait (the hand with the drawn tile removed); response
            # decisions always see the exact 13-tile wait, so ron legality is
            # exact where it matters.
            counts[physical_index(player.drawn)] -= 1
            return winning_tiles(counts, meld_count, closed=closed)
        # An after-call 14-tile hand has no drawn tile and no fixed wait;
        # discard furiten is not queried there, only temporary/riichi flags.
        return ()

    def _discard_kinds(self) -> set[int]:
        if self._view is None:
            return set()
        return {
            physical_index(record.tile)
            for record in self._view.players[self.viewer].discards
        }

    def _finalize(
        self,
        actions: Iterable[RuleAction],
        kind: RuleDecisionKind,
        trigger_seq: int,
    ) -> RuleDecision | None:
        unique: list[RuleAction] = []
        seen: set[RuleAction] = set()
        for action in actions:
            if action in seen:
                continue
            seen.add(action)
            unique.append(action)
        if not unique:
            return None
        return RuleDecision(
            viewer=self.viewer,
            round_id=self._round_id,
            trigger_seq=trigger_seq,
            kind=kind,
            actions=tuple(unique),
        )


def _tiles(hand: Iterable[int], red_fives: bool) -> tuple[Tile, ...]:
    return tuple(Tile.from_physical(tile, red_fives) for tile in hand)


def _distinct_tiles(hand: Iterable[Tile]) -> list[Tile]:
    unique = set(hand)
    return sorted(unique, key=lambda tile: tile.sort_key)


__all__ = ["PlayerRuleEngine"]
