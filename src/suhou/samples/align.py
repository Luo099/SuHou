"""Decision-window alignment: settle recorded actions against open windows.

The aligner never touches rule-engine state; it only decides, per event,
which open sample window gets a confirmed label and which response windows
close unlabelled. Anything a legal log records but cannot be attributed to
exactly one open window is an alignment error, not an unlabelled sample.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from suhou.domain.events import (
    AgariPayload,
    CallPayload,
    EventType,
    GameEvent,
    RyuukyokuPayload,
)
from suhou.domain.types import MeldKind
from suhou.rules.extract import PlayerDecision, match_event_action
from suhou.rules.model import KyushuAction, RuleAction, RuleDecisionKind
from suhou.samples.model import DecisionSample, EventRef, SampleAlignmentError, SampleAudit

_SELF_KINDS = frozenset(
    {RuleDecisionKind.TURN, RuleDecisionKind.AFTER_CALL, RuleDecisionKind.RIICHI_DISCARD}
)
_RESPONSE_KINDS = frozenset({RuleDecisionKind.RESPONSE, RuleDecisionKind.KAN_RESPONSE})
_KYUSHU_REASON = "kyushukyuhai"
_RESPONSE_NOT_TAKEN = "response_not_taken"
_KYUSHU_NOT_CONFIRMED = "kyushu_not_confirmed"


@dataclass(slots=True)
class Window:
    """One open decision window plus its mutable sample row."""

    point: PlayerDecision
    sample: DecisionSample
    trigger_actor: int | None
    index: int
    settled: bool = False


class Aligner:
    """Track open windows and settle them as recorded events arrive."""

    def __init__(self, log_id: str) -> None:
        self.log_id = log_id
        self.windows: dict[tuple[int, int, RuleDecisionKind], Window] = {}
        self.samples: list[DecisionSample] = []

    # -- window registry ----------------------------------------------------

    def open_window(self, point: PlayerDecision, *, trigger_actor: int | None) -> None:
        decision = point.decision
        key = (decision.round_id, point.viewer, decision.kind)
        existing = self.windows.get(key)
        if existing is not None and not existing.settled:
            raise self._error(
                decision.viewer,
                decision.trigger_seq,
                point.event_index,
                f"{decision.kind} window already open "
                f"(trigger seq={existing.point.decision.trigger_seq})",
                round_ordinal=point.round_ordinal,
            )
        sample = DecisionSample(
            observation=point.observation,
            legal_actions=decision.actions,
            target_index=None,
            audit=SampleAudit(
                log_id=self.log_id,
                round_ordinal=point.round_ordinal,
                round_index=point.round_index,
                viewer=point.viewer,
                window_type=decision.kind.value,
                trigger=EventRef(
                    seq=decision.trigger_seq,
                    source_index=point.event_index,
                    event_type=point.observation.history[-1].type.value,
                    source_tag=point.event_tag,
                    actor=trigger_actor,
                ),
                label=None,
                unlabelled_reason=None,
            ),
        )
        self.windows[key] = Window(point, sample, trigger_actor, len(self.samples))
        self.samples.append(sample)

    # -- settling -----------------------------------------------------------

    def settle(self, event: GameEvent, red_fives: bool, *, source_tag: str = "") -> None:
        """Apply one recorded event: labels first, then response closures."""
        action = match_event_action(event, red_fives=red_fives)
        if event.kind is EventType.DISCARD:
            assert event.actor is not None
            window = self._own_window(event, _SELF_KINDS)
            self._label(window, event, action, source_tag)
            self._close_responses(event, _RESPONSE_NOT_TAKEN)
        elif event.kind is EventType.CALL:
            assert event.actor is not None and isinstance(event.payload, CallPayload)
            if event.payload.kind in (MeldKind.ANKAN, MeldKind.KAKAN):
                window = self._own_window(event, {RuleDecisionKind.TURN})
                self._label(window, event, action, source_tag)
            else:
                window = self._response_window(event, RuleDecisionKind.RESPONSE)
                self._label(window, event, action, source_tag)
            # Any other open response window loses its chance to this call.
            self._close_responses(event, _RESPONSE_NOT_TAKEN)
        elif event.kind is EventType.RIICHI_DECLARATION:
            assert event.actor is not None
            window = self._own_window(event, {RuleDecisionKind.TURN})
            self._label(window, event, action, source_tag)
        elif event.kind is EventType.DRAW:
            # A replacement draw also proves a pending chankan was passed.
            self._close_responses(event, _RESPONSE_NOT_TAKEN)
        elif event.kind is EventType.RESULT:
            self._settle_result(event, action, source_tag)
        # RIICHI_ACCEPTED / DORA / CONNECTION are mid-window events: no labels,
        # no closures (the chankan window stays open through the new dora).

    def _settle_result(
        self, event: GameEvent, action: RuleAction | None, source_tag: str
    ) -> None:
        payload = event.payload
        if isinstance(payload, AgariPayload):
            if payload.who == payload.from_who:
                window = self._own_window(
                    event, {RuleDecisionKind.TURN}, seat=payload.who
                )
                self._label(window, event, action, source_tag)
            else:
                window = self._ron_window(event, payload)
                self._label(window, event, action, source_tag)
            if payload.group_final:
                self.close_round(event)
        else:
            assert isinstance(payload, RyuukyokuPayload)
            if payload.reason == _KYUSHU_REASON:
                self._settle_kyushu(event, source_tag)
            if payload.group_final:
                self.close_round(event)

    def _ron_window(self, event: GameEvent, payload: AgariPayload) -> Window:
        assert event.round_id is not None and event.seq is not None
        matches = [
            window
            for (round_id, seat, kind), window in self.windows.items()
            if round_id == event.round_id
            and seat == payload.who
            and kind in _RESPONSE_KINDS
            and not window.settled
            and window.trigger_actor == payload.from_who
            and window.point.decision.trigger_seq < event.seq
        ]
        if len(matches) != 1:
            raise self._error(
                payload.who,
                event.seq,
                event.source_index,
                f"ron from seat {payload.from_who} has {len(matches)} open "
                "response/kan_response windows to settle",
            )
        return matches[0]

    def _settle_kyushu(self, event: GameEvent, source_tag: str) -> None:
        assert event.round_id is not None and event.seq is not None
        candidates = [
            window
            for (round_id, _seat, kind), window in self.windows.items()
            if round_id == event.round_id
            and kind is RuleDecisionKind.TURN
            and not window.settled
            and KyushuAction() in window.sample.legal_actions
        ]
        if len(candidates) == 1:
            self._label(candidates[0], event, KyushuAction(), source_tag)
        else:
            for window in candidates:
                self._unlabel(window, _KYUSHU_NOT_CONFIRMED)

    # -- closures -----------------------------------------------------------

    def _close_responses(self, event: GameEvent, reason: str) -> None:
        for window in list(self.windows.values()):
            if window.settled or window.point.decision.kind not in _RESPONSE_KINDS:
                continue
            self._unlabel(window, reason)

    def close_round(self, event: GameEvent) -> None:
        """Close the round after its settlement group completes."""
        assert event.round_id is not None and event.seq is not None
        self._close_responses(event, _RESPONSE_NOT_TAKEN)
        for (round_id, seat, kind), window in self.windows.items():
            if round_id == event.round_id and kind in _SELF_KINDS and not window.settled:
                raise self._error(
                    seat,
                    event.seq,
                    event.source_index,
                    f"{kind} window (trigger seq={window.point.decision.trigger_seq}) "
                    "still open at round end",
                    round_ordinal=window.point.round_ordinal,
                )

    def close_round_defensive(self) -> None:
        """ROUND_START guard: a legal log has no windows left from last round."""
        leftovers = [w for w in self.windows.values() if not w.settled]
        if leftovers:
            window = leftovers[0]
            raise self._error(
                window.point.viewer,
                window.point.decision.trigger_seq,
                window.point.event_index,
                f"{window.point.decision.kind} window from the previous round "
                "is still open at round start",
                round_ordinal=window.point.round_ordinal,
            )

    def finish(self) -> None:
        leftovers = [w for w in self.windows.values() if not w.settled]
        if leftovers:
            window = leftovers[0]
            raise self._error(
                window.point.viewer,
                window.point.decision.trigger_seq,
                window.point.event_index,
                f"{window.point.decision.kind} window still open at end of log",
                round_ordinal=window.point.round_ordinal,
            )

    def _unlabel(self, window: Window, reason: str) -> None:
        assert not window.settled
        window.sample = replace(
            window.sample,
            target_index=None,
            audit=replace(window.sample.audit, label=None, unlabelled_reason=reason),
        )
        window.settled = True
        self._sync(window)

    # -- helpers ------------------------------------------------------------

    def _own_window(
        self,
        event: GameEvent,
        kinds: frozenset[RuleDecisionKind] | set[RuleDecisionKind],
        *,
        seat: int | None = None,
    ) -> Window:
        actor = event.actor if seat is None else seat
        assert actor is not None and event.round_id is not None and event.seq is not None
        matches = [
            window
            for (round_id, window_seat, kind), window in self.windows.items()
            if round_id == event.round_id
            and window_seat == actor
            and kind in kinds
            and not window.settled
            and window.point.decision.trigger_seq < event.seq
        ]
        if len(matches) != 1:
            kinds_text = "/".join(sorted(k.value for k in kinds))
            raise self._error(
                actor,
                event.seq,
                event.source_index,
                f"{event.kind} has {len(matches)} open {kinds_text} windows to settle",
            )
        return matches[0]

    def _response_window(
        self, event: GameEvent, kind: RuleDecisionKind
    ) -> Window:
        """Find the caller's response window; the trigger must be the claimed tile."""
        assert event.actor is not None and event.round_id is not None and event.seq is not None
        assert isinstance(event.payload, CallPayload)
        matches = [
            window
            for (round_id, seat, window_kind), window in self.windows.items()
            if round_id == event.round_id
            and seat == event.actor
            and window_kind is kind
            and not window.settled
            and window.point.decision.trigger_seq < event.seq
        ]
        if len(matches) != 1:
            raise self._error(
                event.actor,
                event.seq,
                event.source_index,
                f"{event.payload.kind} has {len(matches)} open {kind} windows to settle",
            )
        window = matches[0]
        if window.trigger_actor != event.payload.from_seat:
            raise self._error(
                event.actor,
                event.seq,
                event.source_index,
                f"call from seat {event.payload.from_seat} does not match the "
                f"{kind} trigger seat {window.trigger_actor}",
                round_ordinal=window.point.round_ordinal,
            )
        return window

    def _label(
        self,
        window: Window,
        event: GameEvent,
        action: RuleAction | None,
        source_tag: str,
    ) -> None:
        assert event.seq is not None
        seat = window.point.viewer
        if window.settled:
            raise self._error(
                seat,
                event.seq,
                event.source_index,
                f"{window.point.decision.kind} window "
                f"(trigger seq={window.point.decision.trigger_seq}) settled twice",
                round_ordinal=window.point.round_ordinal,
            )
        if action is None:
            raise self._error(
                seat,
                event.seq,
                event.source_index,
                f"{event.kind} is not a candidate action for the open window",
                round_ordinal=window.point.round_ordinal,
            )
        count = window.sample.legal_actions.count(action)
        if count != 1:
            raise self._error(
                seat,
                event.seq,
                event.source_index,
                f"action {action} appears {count} times in window candidates "
                f"(trigger seq={window.point.decision.trigger_seq})",
                round_ordinal=window.point.round_ordinal,
            )
        window.sample = replace(
            window.sample,
            target_index=window.sample.legal_actions.index(action),
            audit=replace(
                window.sample.audit,
                label=EventRef(
                    seq=event.seq,
                    source_index=event.source_index,
                    event_type=event.kind.value,
                    source_tag=source_tag,
                    actor=event.actor,
                ),
                unlabelled_reason=None,
            ),
        )
        window.settled = True
        self._sync(window)

    def _sync(self, window: Window) -> None:
        self.samples[window.index] = window.sample

    def _error(
        self,
        seat: int,
        seq: int | None,
        source_index: int,
        detail: str,
        *,
        round_ordinal: int = -1,
    ) -> SampleAlignmentError:
        return SampleAlignmentError(
            log_id=self.log_id,
            round_ordinal=round_ordinal,
            seat=seat,
            seq=seq if seq is not None else -1,
            source_index=source_index,
            detail=detail,
        )


__all__ = ["Aligner", "Window"]