"""Rules-driven decision extraction for one viewer.

Unlike the compatibility extractor, this path:

* runs :class:`PlayerRuleEngine` over every applied event;
* yields a decision only when the viewer has at least one legal action;
* builds the observation from the post-event state and a history that already
  contains the triggering event;
* keeps candidate actions outside the Observation v1 JSON contract.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from suhou.domain.events import (
    AgariPayload,
    CallPayload,
    DiscardPayload,
    EventType,
    GameEvent,
)
from suhou.domain.tiles import Tile
from suhou.domain.types import MeldKind
from suhou.observation.builder import ObservationError, build_observation
from suhou.observation.history import EventHistory
from suhou.observation.model import (
    DecisionContext,
    DecisionPhase,
    Observation,
)
from suhou.replay.engine import ReplayFrame
from suhou.rules.engine import PlayerRuleEngine
from suhou.rules.model import (
    AnkanAction,
    ChiAction,
    DiscardAction,
    KakanAction,
    MinkanAction,
    PonAction,
    RiichiAction,
    RonAction,
    RuleAction,
    RuleDecision,
    RuleDecisionKind,
    RuleError,
    TsumoAction,
)
from suhou.tenhou.adapter import iter_replay
from suhou.tenhou.models import GameRecord

_PHASES = {
    RuleDecisionKind.TURN: DecisionPhase.TURN,
    RuleDecisionKind.AFTER_CALL: DecisionPhase.AFTER_CALL,
    RuleDecisionKind.RIICHI_DISCARD: DecisionPhase.RIICHI_DISCARD,
    RuleDecisionKind.RESPONSE: DecisionPhase.RESPONSE,
    RuleDecisionKind.KAN_RESPONSE: DecisionPhase.KAN_RESPONSE,
}


@dataclass(frozen=True, slots=True)
class PlayerDecision:
    """One viewer decision plus observation and source-event audit fields."""

    viewer: int
    decision: RuleDecision
    observation: Observation
    round_ordinal: int
    round_index: int
    event_index: int
    event_seq: int
    event_tag: str


def apply_decision_frame(
    engine: PlayerRuleEngine,
    history: EventHistory,
    frame: ReplayFrame,
) -> PlayerDecision | None:
    """Advance one seat's engine/history by one frame; freeze a decision if any.

    The caller owns ROUND_START resets (rebuild history, engine.update) and
    METADATA skipping; this helper handles exactly the post-event decision
    path shared by iter_player_decisions and the sample exporter.
    """
    event = frame.event
    if frame.after is None:
        raise ObservationError("A round event has no post-event state")
    decision = engine.update(event, frame.after)
    history.append(event)
    if decision is None:
        return None
    assert event.round_id is not None and event.seq is not None
    context = DecisionContext(_PHASES[decision.kind], decision.trigger_seq)
    observation = build_observation(
        frame.after,
        engine,
        history.view(engine.viewer),
        engine.viewer,
        context,
    )
    return PlayerDecision(
        viewer=engine.viewer,
        decision=decision,
        observation=observation,
        round_ordinal=event.round_id,
        round_index=frame.after.index,
        event_index=event.source_index,
        event_seq=event.seq,
        event_tag=frame.source.tag if frame.source is not None else "",
    )


def iter_player_decisions(
    source: str | Path | GameRecord, viewer: int
) -> Iterator[PlayerDecision]:
    """Replay ``source`` and yield every viewer decision window.

    Legal but unused response windows are included; the log cannot tell an
    active pass from a window cut short by another call, so no pass labels are
    invented here.
    """
    if not 0 <= viewer < 4:
        raise ValueError(f"viewer must be in 0..3, got {viewer}")
    frames = tuple(iter_replay(source))
    engine = PlayerRuleEngine(viewer)
    history = EventHistory()
    points: list[PlayerDecision] = []

    for frame in frames:
        event = frame.event
        if event.kind is EventType.METADATA:
            continue
        if event.kind is EventType.ROUND_START:
            assert frame.after is not None
            history = EventHistory(red_fives=frame.after.red_fives)
            engine.update(event, frame.after)
            history.append(event)
            continue
        point = apply_decision_frame(engine, history, frame)
        if point is not None:
            points.append(point)
    return iter(points)


def match_event_action(event: GameEvent, *, red_fives: bool = True) -> RuleAction | None:
    """Map one recorded event to the structured action it represents.

    This is the audit helper for acceptance tests and the future sample layer.
    ``red_fives`` must match the parsed game settings, just like the rule
    engine and ``project_visible``. ``None`` means the event is not an action
    the rules model predicts (draws, dora reveals, results that are not wins,
    and so on).
    """
    if event.kind is EventType.DISCARD:
        payload = event.payload
        assert isinstance(payload, DiscardPayload)
        return DiscardAction(Tile.from_physical(payload.tile, red_fives))
    if event.kind is EventType.CALL:
        payload = event.payload
        assert isinstance(payload, CallPayload)
        if payload.kind is MeldKind.CHI:
            return ChiAction(_sorted_tiles(payload.consumed_tiles, red_fives))
        if payload.kind is MeldKind.PON:
            return PonAction(_sorted_tiles(payload.consumed_tiles, red_fives))
        if payload.kind is MeldKind.MINKAN:
            return MinkanAction(_sorted_tiles(payload.consumed_tiles, red_fives))
        if payload.kind is MeldKind.ANKAN:
            return AnkanAction(_sorted_tiles(payload.tiles, red_fives))
        if payload.kind is MeldKind.KAKAN:
            if payload.added_tile is None or payload.target_seq is None:
                raise RuleError("Kakan event lacks an added tile or target meld")
            return KakanAction(
                Tile.from_physical(payload.added_tile, red_fives),
                payload.target_seq,
            )
        raise RuleError("Nuki (three-player) is outside the four-player rules")
    if event.kind is EventType.RIICHI_DECLARATION:
        return RiichiAction()
    if event.kind is EventType.RESULT and isinstance(event.payload, AgariPayload):
        if event.payload.who == event.payload.from_who:
            return TsumoAction()
        return RonAction()
    return None


def _sorted_tiles(tiles: tuple[int, ...], red_fives: bool) -> tuple[Tile, ...]:
    return tuple(
        sorted(
            (Tile.from_physical(tile, red_fives) for tile in tiles),
            key=lambda tile: tile.sort_key,
        )
    )


__all__ = [
    "PlayerDecision",
    "apply_decision_frame",
    "iter_player_decisions",
    "match_event_action",
]
