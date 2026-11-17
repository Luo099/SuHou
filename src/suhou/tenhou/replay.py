"""Compatibility wrapper around the unified replay engine.

``replay_tenhou`` keeps its public shape (steps, snapshots, summary, and the
``before_event`` seam) but parses and applies the unified event stream
internally. New consumers should use ``iter_replay`` or ``replay_events``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from suhou.domain.events import GameEvent
from suhou.domain.state import PlayerState, PlayerView, RoundState, RoundView
from suhou.replay.engine import ReplayError
from suhou.tenhou.adapter import replay_frames
from suhou.tenhou.models import GameRecord, TenhouEvent
from suhou.tenhou.parser import parse_tenhou


@dataclass(frozen=True, slots=True)
class ReplayStep:
    event: TenhouEvent
    snapshot: dict[str, Any]
    message: str


@dataclass(slots=True)
class ReplayResult:
    game: GameRecord
    steps: list[ReplayStep]
    rounds: list[RoundView]
    players: tuple[str, ...]
    scores: tuple[int, ...]
    final_scores: tuple[int, int, int, int] | None = None
    final_points: tuple[float, float, float, float] | None = None
    # Versioned internal events, in the same order as ``steps``.
    events: tuple[GameEvent, ...] = field(default=())

    def summary(self) -> dict[str, Any]:
        return {
            "source_events": len(self.game.events),
            "rounds": len(self.rounds),
            "players": list(self.players),
            "scores": list(self.scores),
            "final_scores": list(self.final_scores) if self.final_scores else None,
            "final_points": list(self.final_points) if self.final_points else None,
        }


def parse_replay(path: str | Path) -> GameRecord:
    """Public input seam shared by API replay and the CLI."""
    return parse_tenhou(path)


BeforeEvent = Callable[[TenhouEvent, RoundView | None], None]


def replay_tenhou(
    source: str | Path | GameRecord,
    *,
    before_event: BeforeEvent | None = None,
) -> ReplayResult:
    """Consume each source event exactly once and return all post-event states.

    ``before_event`` observes the table before an event is applied. It is the
    compatibility seam for callers that still key off source tags; it must not
    mutate the replay.
    """
    game = parse_replay(source) if isinstance(source, (str, Path)) else source
    frames, outcome = replay_frames(game, before_event=before_event)
    steps: list[ReplayStep] = []
    for frame in frames:
        assert frame.source is not None
        after = frame.after
        steps.append(
            ReplayStep(
                frame.source,
                {
                    "event": {
                        "index": frame.source.index,
                        "tag": frame.source.raw_tag,
                        "kind": frame.source.tag,
                        "seat": frame.source.seat,
                        "tile": frame.source.tile_name,
                        "attributes": dict(frame.source.attributes),
                    },
                    "message": frame.message,
                    "round": after.snapshot() if after is not None else None,
                    "final_scores": (
                        list(frame.final_scores) if frame.final_scores else None
                    ),
                    "final_points": (
                        list(frame.final_points) if frame.final_points else None
                    ),
                },
                frame.message,
            )
        )
    return ReplayResult(
        game=game,
        steps=steps,
        rounds=list(outcome.rounds),
        players=tuple(player.name for player in game.players),
        scores=outcome.scores,
        final_scores=outcome.final_scores,
        final_points=outcome.final_points,
        events=outcome.events,
    )


__all__ = [
    "PlayerState",
    "PlayerView",
    "ReplayError",
    "ReplayResult",
    "ReplayStep",
    "RoundState",
    "RoundView",
    "parse_replay",
    "replay_tenhou",
]
