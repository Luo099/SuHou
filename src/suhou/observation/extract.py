"""Extract Observation v1 decision points from a Tenhou replay.

The rules engine is not part of this branch yet. Callers therefore supply a
``rules_state_factory`` per decision; it is the seam where the reused rules
engine will provide furiten (and later legal actions). The event-to-phase
mapping below is a provisional adapter, not a rules implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from suhou.observation.builder import ObservationError, RulesState, build_observation
from suhou.observation.history import HistoryEntry, RoundHistory
from suhou.observation.model import (
    DecisionContext,
    DecisionPhase,
    EventType,
    MeldType,
    Observation,
    Tile,
)
from suhou.tenhou.melds import Meld, decode_meld
from suhou.tenhou.models import Event, GameRecord
from suhou.tenhou.replay import RoundState, replay_tenhou
from suhou.tenhou.tiles import Wind

RulesStateFactory = Callable[[RoundState, Event, int], RulesState]

_ROUND_TAGS = frozenset(
    {"T", "D", "N", "REACH", "DORA", "AGARI", "RYUUKYOKU", "BYE", "RESUME"}
)
_DRAW_KINDS = frozenset({"chi", "pon", "minkan"})
_SELF_KAN_KINDS = frozenset({"ankan", "kakan"})


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


class _SequenceTracker:
    """Track the events that trigger each kind of decision."""

    __slots__ = ("last_turn_seq", "last_discard_seq")

    def __init__(self) -> None:
        self.last_turn_seq = 0
        self.last_discard_seq: int | None = None

    def reset(self) -> None:
        self.last_turn_seq = 0
        self.last_discard_seq = None

    def observe(self, event: Event, round_: RoundState) -> None:
        seq = round_.event_seq(event)
        if event.tag == "T":
            self.last_turn_seq = seq
        elif event.tag == "D":
            self.last_discard_seq = seq
        elif event.tag == "N":
            # Any call ends the previous response window and gives the caller
            # the next turn; a kan is followed by its replacement draw.
            self.last_turn_seq = seq


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
    points: list[DecisionPoint] = []
    history = RoundHistory()
    tracker = _SequenceTracker()
    round_ordinal = -1

    def before_event(event: Event, round_: RoundState | None) -> None:
        nonlocal history, round_ordinal
        if event.tag == "INIT":
            round_ordinal += 1
            history = RoundHistory()
            tracker.reset()
            history.append(_round_start_entry(event))
            return
        if round_ is None or event.tag not in _ROUND_TAGS:
            return

        seq = round_.event_seq(event)
        decision = _decision_for(event, round_, tracker)
        if decision is not None:
            context = DecisionContext(decision.phase, decision.trigger_seq)
            rules_state = rules_state_factory(round_, event, decision.viewer)
            observation = build_observation(
                round_,
                rules_state,
                history.view(decision.viewer, cutoff=seq),
                decision.viewer,
                context,
            )
            points.append(
                DecisionPoint(
                    viewer=decision.viewer,
                    context=context,
                    observation=observation,
                    round_ordinal=round_ordinal,
                    round_index=round_.index,
                    event_index=event.index,
                    event_seq=seq,
                    event_tag=event.tag,
                )
            )
        history.append(_history_entry(event, round_, seq))
        tracker.observe(event, round_)

    replay_tenhou(source, before_event=before_event)
    return iter(points)


def _decision_for(
    event: Event,
    round_: RoundState,
    tracker: _SequenceTracker,
) -> _Decision | None:
    if event.tag == "D":
        assert event.seat is not None
        return _Decision(event.seat, DecisionPhase.DISCARD, tracker.last_turn_seq)
    if event.tag == "N":
        assert event.seat is not None and event.meld is not None
        meld = decode_meld(event.seat, event.meld)
        if meld.kind in _DRAW_KINDS:
            if tracker.last_discard_seq is None:
                raise ObservationError("Call decision has no preceding discard")
            return _Decision(meld.caller, DecisionPhase.RESPONSE, tracker.last_discard_seq)
        if meld.kind in _SELF_KAN_KINDS:
            return _Decision(meld.caller, DecisionPhase.SELF_KAN, tracker.last_turn_seq)
        raise ObservationError("Nuki (three-player) is outside Observation v1")
    if event.tag == "REACH" and event.step == 1:
        assert event.seat is not None
        return _Decision(event.seat, DecisionPhase.RIICHI, tracker.last_turn_seq)
    if event.tag == "AGARI":
        if not event.winners:
            raise ObservationError("AGARI has no winner")
        who, from_who = event.winners[0]
        if who == from_who:
            return _Decision(who, DecisionPhase.TSUMO, tracker.last_turn_seq)
        if tracker.last_discard_seq is None:
            raise ObservationError("Ron decision has no preceding discard")
        return _Decision(who, DecisionPhase.RESPONSE, tracker.last_discard_seq)
    return None


def _round_start_entry(event: Event) -> HistoryEntry:
    if event.seed is None or event.oya is None or event.scores is None or event.tiles is None:
        raise ObservationError("INIT is missing round-start fields")
    hands = tuple(
        tuple(
            sorted(
                (Tile.from_physical(tile) for tile in event.tiles[seat * 13:(seat + 1) * 13]),
                key=lambda tile: tile.sort_key,
            )
        )
        for seat in range(4)
    )
    return HistoryEntry(
        seq=0,
        type=EventType.ROUND_START,
        actor=None,
        data={
            "round_wind": Wind.from_index(event.seed[0] // 4),
            "hand_number": event.seed[0] % 4 + 1,
            "honba": event.seed[1],
            "riichi_sticks": event.seed[2],
            "dealer": event.oya,
            "scores": event.scores,
            "dora_indicators": (Tile.from_physical(event.seed[5]),),
            "hands": hands,
        },
    )


def _history_entry(event: Event, round_: RoundState, seq: int) -> HistoryEntry:
    assert event.tag in _ROUND_TAGS
    actor = event.seat
    if event.tag == "T":
        assert event.tile is not None
        data = {
            "tile": Tile.from_physical(event.tile),
            "replacement": round_.awaiting_replacement_draw == event.seat,
        }
        kind = EventType.DRAW
    elif event.tag == "D":
        assert event.seat is not None and event.tile is not None
        player = round_.players[event.seat]
        data = {
            "tile": Tile.from_physical(event.tile),
            "tsumogiri": event.attributes.get("tsumogiri") == "1" or player.drawn == event.tile,
            "riichi_declaration": round_.pending_riichi == event.seat,
        }
        kind = EventType.DISCARD
    elif event.tag == "N":
        data = _call_data(event, round_)
        kind = EventType.CALL
    elif event.tag == "REACH":
        assert event.step in (1, 2)
        kind = (
            EventType.RIICHI_DECLARATION if event.step == 1 else EventType.RIICHI_ACCEPTED
        )
        data = {} if event.scores is None else {"scores": event.scores}
    elif event.tag == "DORA":
        assert event.tile is not None
        actor = None
        kind = EventType.DORA
        data = {"indicator": Tile.from_physical(event.tile)}
    elif event.tag == "AGARI":
        kind = EventType.RESULT
        actor = None
        data = _result_data(event, round_, "agari")
    elif event.tag == "RYUUKYOKU":
        kind = EventType.RESULT
        actor = None
        data = _result_data(event, round_, "ryuukyoku")
    else:  # BYE / RESUME
        kind = EventType.CONNECTION
        data = {"connected": event.tag == "RESUME"}
    return HistoryEntry(seq=seq, type=kind, actor=actor, data=data)


def _call_data(event: Event, round_: RoundState) -> dict[str, object]:
    assert event.seat is not None and event.meld is not None
    meld = decode_meld(event.seat, event.meld)
    if meld.kind == "nuki":
        raise ObservationError("Nuki (three-player) is outside Observation v1")
    if meld.kind == "ankan":
        return {
            "meld_type": MeldType.ANKAN,
            "tiles": tuple(Tile.from_physical(tile) for tile in meld.tiles),
            "source_player": None,
            "called_tile": None,
            "added_tile": None,
        }
    if meld.kind == "kakan":
        previous = _previous_pon(round_, meld)
        if previous is None:
            raise ObservationError("Kakan has no matching previous pon")
        return {
            "meld_type": MeldType.KAKAN,
            "tiles": tuple(
                Tile.from_physical(tile)
                for tile in sorted((*meld.tiles, meld.called_tile))
            ),
            "source_player": meld.from_seat,
            "called_tile": _require_tile(previous.called_tile),
            "added_tile": Tile.from_physical(meld.called_tile),
        }
    return {
        "meld_type": MeldType(meld.kind),
        "tiles": tuple(Tile.from_physical(tile) for tile in meld.tiles),
        "source_player": meld.from_seat,
        "called_tile": Tile.from_physical(meld.called_tile),
        "added_tile": None,
    }


def _previous_pon(round_: RoundState, meld: Meld) -> Meld | None:
    player = round_.players[meld.caller]
    return next(
        (
            item
            for item in player.melds
            if item.kind == "pon"
            and item.from_seat == meld.from_seat
            and item.tiles == meld.tiles
        ),
        None,
    )


def _require_tile(tile: int | None) -> Tile:
    if tile is None:
        raise ObservationError("Meld lacks a physical tile")
    return Tile.from_physical(tile)


def _result_data(event: Event, round_: RoundState, result: str) -> dict[str, object]:
    deltas = event.deltas or (0, 0, 0, 0)
    data: dict[str, object] = {
        "result": result,
        "deltas": deltas,
        "scores": tuple(
            score + delta for score, delta in zip(round_.scores, deltas, strict=True)
        ),
    }
    if event.winners is not None:
        data["winners"] = event.winners
    if event.draw_type is not None:
        data["reason"] = event.draw_type
    if "machi" in event.attributes:
        data["winning_tile"] = Tile.from_physical(int(event.attributes["machi"]))
    return data


__all__ = ["DecisionPoint", "RulesStateFactory", "iter_decision_points"]
