"""Extract Observation v1 decision points from the unified event stream.

The Tenhou adapter has already decoded every protocol detail. This module only
selects decision events, freezes result-group state, and asks
``project_visible`` for the viewer-filtered history prefix.

The rules engine is not part of this branch yet. Callers supply a
``rules_state_factory`` per decision; it is the seam where the reused rules
engine will provide furiten (and later legal actions).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from suhou.domain.events import (
    AgariPayload,
    CallPayload,
    EventType,
    GameEvent,
    RyuukyokuPayload,
)
from suhou.domain.state import RoundView
from suhou.domain.types import MeldKind
from suhou.observation.builder import ObservationError, RulesState, build_observation
from suhou.observation.history import EventHistory
from suhou.observation.model import (
    DecisionContext,
    DecisionPhase,
    Observation,
)
from suhou.tenhou.adapter import iter_replay
from suhou.tenhou.models import GameRecord, TenhouEvent

RulesStateFactory = Callable[[RoundView, GameEvent, int], RulesState]
LegacyRulesStateFactory = Callable[[Any, TenhouEvent, int], RulesState]

_DRAW_KINDS = frozenset({MeldKind.CHI, MeldKind.PON, MeldKind.MINKAN})
_SELF_KAN_KINDS = frozenset({MeldKind.ANKAN, MeldKind.KAKAN})


@dataclass(frozen=True, slots=True)
class DecisionPoint:
    """One pre-action observation plus the source event that acts next.

    The observation is the deliverable; ``event_*`` fields identify the action
    for auditing and are not part of the observation contract.
    """

    viewer: int
    context: DecisionContext
    observation: Observation
    # Sequential round instance in the game. Unlike ``round_index`` (seed[0]),
    # this stays unique when a dealer repeats the same hand (renchan).
    round_ordinal: int
    round_index: int
    event_index: int
    event_seq: int
    event_tag: str


@dataclass(frozen=True, slots=True)
class _Decision:
    viewer: int
    phase: DecisionPhase
    trigger_seq: int


class _LegacyRulesFactory:
    """Mark an old source-event factory so the extractor passes TenhouEvent."""

    uses_source_event = True

    def __init__(self, factory: LegacyRulesStateFactory) -> None:
        self._factory = factory

    def __call__(self, view: Any, event: TenhouEvent, viewer: int) -> RulesState:
        return self._factory(view, event, viewer)


def legacy_rules_state_factory(factory: LegacyRulesStateFactory) -> RulesStateFactory:
    """Wrap the old ``(state, TenhouEvent, viewer)`` factory explicitly.

    New factories receive the unified :class:`GameEvent`; this wrapper is only
    for code that still inspects source tags or attributes.
    """
    return _LegacyRulesFactory(factory)  # type: ignore[return-value]


class _SequenceTracker:
    """Track the events that trigger each kind of decision."""

    __slots__ = ("last_turn_seq", "last_discard_seq")

    def __init__(self) -> None:
        self.last_turn_seq = 0
        self.last_discard_seq: int | None = None

    def reset(self) -> None:
        self.last_turn_seq = 0
        self.last_discard_seq = None

    def observe(self, event: GameEvent) -> None:
        if event.seq is None:
            return
        if event.kind is EventType.DRAW:
            self.last_turn_seq = event.seq
        elif event.kind is EventType.DISCARD:
            self.last_discard_seq = event.seq
        elif event.kind is EventType.CALL:
            # Any call ends the previous response window and gives the caller
            # the next turn; a kan is followed by its replacement draw.
            self.last_turn_seq = event.seq


def iter_decision_points(
    source: str | Path | GameRecord,
    rules_state_factory: RulesStateFactory,
) -> Iterator[DecisionPoint]:
    """Replay ``source`` and yield one observation per recorded decision.

    Decisions are the actual recorded actions: draws are automatic, and passes
    are not in the log, so no pass samples are invented. A riichi declaration
    and the discard it requires are separate source events and both yield a
    point; the sample layer may group them later.

    The replay runs eagerly; the returned iterator holds already-built,
    immutable observations.
    """
    frames = tuple(iter_replay(source))
    points: list[DecisionPoint] = []
    history = EventHistory()
    tracker = _SequenceTracker()
    round_ordinal = -1
    result_group: int | None = None
    result_view: RoundView | None = None
    result_cutoff: int | None = None

    for frame in frames:
        event = frame.event
        if event.kind is EventType.ROUND_START:
            assert event.round_id is not None
            round_ordinal = event.round_id
            history = EventHistory(
                red_fives=frame.after.red_fives if frame.after is not None else True
            )
            tracker.reset()
            result_group = None
            result_view = None
            result_cutoff = None
            history.append(event)
            continue
        if event.round_id is None or event.seq is None:
            continue
        view = frame.before
        if view is None:
            raise ObservationError("A round event has no pre-event state")
        cutoff = event.seq
        active_view = view
        if event.kind is EventType.RESULT:
            assert isinstance(event.payload, (AgariPayload, RyuukyokuPayload))
            group_id = event.payload.group_id
            if result_group != group_id:
                result_group = group_id
                result_view = view
                result_cutoff = event.seq
            assert result_view is not None and result_cutoff is not None
            active_view = result_view
            cutoff = result_cutoff
        else:
            result_group = None
            result_view = None
            result_cutoff = None

        decision = _decision_for(event, tracker)
        if decision is not None:
            context = DecisionContext(decision.phase, decision.trigger_seq)
            event_for_rules: Any = (
                frame.source
                if getattr(rules_state_factory, "uses_source_event", False)
                else event
            )
            rules_state = rules_state_factory(
                active_view, event_for_rules, decision.viewer
            )
            observation = build_observation(
                active_view,
                rules_state,
                history.view(decision.viewer, cutoff=cutoff),
                decision.viewer,
                context,
            )
            points.append(
                DecisionPoint(
                    viewer=decision.viewer,
                    context=context,
                    observation=observation,
                    round_ordinal=round_ordinal,
                    round_index=active_view.index,
                    event_index=event.source_index,
                    event_seq=event.seq,
                    event_tag=frame.source.tag if frame.source is not None else "",
                )
            )
        history.append(event)
        tracker.observe(event)

    return iter(points)


def _decision_for(event: GameEvent, tracker: _SequenceTracker) -> _Decision | None:
    if event.kind is EventType.DISCARD:
        assert event.actor is not None
        return _Decision(event.actor, DecisionPhase.DISCARD, tracker.last_turn_seq)
    if event.kind is EventType.CALL:
        assert isinstance(event.payload, CallPayload) and event.actor is not None
        if event.payload.kind in _DRAW_KINDS:
            if tracker.last_discard_seq is None:
                raise ObservationError("Call decision has no preceding discard")
            return _Decision(
                event.actor, DecisionPhase.RESPONSE, tracker.last_discard_seq
            )
        if event.payload.kind in _SELF_KAN_KINDS:
            return _Decision(event.actor, DecisionPhase.SELF_KAN, tracker.last_turn_seq)
        raise ObservationError("Nuki (three-player) is outside Observation v1")
    if event.kind is EventType.RIICHI_DECLARATION:
        assert event.actor is not None
        return _Decision(event.actor, DecisionPhase.RIICHI, tracker.last_turn_seq)
    if event.kind is EventType.RESULT and isinstance(event.payload, AgariPayload):
        who, from_who = event.payload.who, event.payload.from_who
        if who == from_who:
            return _Decision(who, DecisionPhase.TSUMO, tracker.last_turn_seq)
        if tracker.last_discard_seq is None:
            raise ObservationError("Ron decision has no preceding discard")
        return _Decision(who, DecisionPhase.RESPONSE, tracker.last_discard_seq)
    return None


__all__ = [
    "DecisionPoint",
    "LegacyRulesStateFactory",
    "RulesStateFactory",
    "iter_decision_points",
    "legacy_rules_state_factory",
]
