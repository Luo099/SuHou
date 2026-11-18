"""Decision-sample v1 records, action serialization, and alignment failures.

A sample freezes one rule-engine window: the observation right after the
triggering event, the ordered legal actions, an optional confirmed label, and
audit fields that locate everything back in the source log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from suhou.observation.model import Observation, to_jsonable
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
    TsumoAction,
)


class SampleAlignmentError(ValueError):
    """A recorded action cannot be aligned to any open decision window."""

    def __init__(
        self,
        *,
        log_id: str,
        round_ordinal: int,
        seat: int,
        seq: int,
        source_index: int,
        detail: str,
    ) -> None:
        self.log_id = log_id
        self.round_ordinal = round_ordinal
        self.seat = seat
        self.seq = seq
        self.source_index = source_index
        self.detail = detail
        super().__init__(str(self))

    def __str__(self) -> str:
        return (
            f"{self.log_id} round={self.round_ordinal} seat={self.seat} "
            f"seq={self.seq} source_index={self.source_index}: {self.detail}"
        )


@dataclass(frozen=True, slots=True)
class EventRef:
    """One source-event location for audit back-references."""

    seq: int
    source_index: int
    event_type: str
    source_tag: str
    actor: int | None


@dataclass(frozen=True, slots=True)
class SampleAudit:
    """Where a sample came from and how its label was confirmed."""

    log_id: str
    round_ordinal: int
    round_index: int
    viewer: int
    window_type: str
    trigger: EventRef
    label: EventRef | None
    unlabelled_reason: str | None


@dataclass(frozen=True, slots=True)
class DecisionSample:
    """One JSONL row: observation + legal actions + optional label + audit."""

    observation: Observation
    legal_actions: tuple[RuleAction, ...]
    target_index: int | None
    audit: SampleAudit
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observation": self.observation.to_dict(),
            "legal_actions": [action_json(action) for action in self.legal_actions],
            "target_index": self.target_index,
            "audit": to_jsonable(self.audit),
        }


def action_json(action: RuleAction) -> dict[str, Any]:
    """Serialize one structured action to its fixed v1 JSON shape."""
    if isinstance(action, DiscardAction):
        return {"type": "discard", "tile": to_jsonable(action.tile)}
    if isinstance(action, ChiAction):
        return {"type": "chi", "consumed_tiles": to_jsonable(action.consumed_tiles)}
    if isinstance(action, PonAction):
        return {"type": "pon", "consumed_tiles": to_jsonable(action.consumed_tiles)}
    if isinstance(action, MinkanAction):
        return {"type": "minkan", "consumed_tiles": to_jsonable(action.consumed_tiles)}
    if isinstance(action, AnkanAction):
        return {"type": "ankan", "consumed_tiles": to_jsonable(action.consumed_tiles)}
    if isinstance(action, KakanAction):
        return {
            "type": "kakan",
            "added_tile": to_jsonable(action.added_tile),
            "target_meld_seq": action.target_meld_seq,
        }
    if isinstance(action, RiichiAction):
        return {"type": "riichi"}
    if isinstance(action, TsumoAction):
        return {"type": "tsumo"}
    if isinstance(action, RonAction):
        return {"type": "ron"}
    if isinstance(action, KyushuAction):
        return {"type": "kyushu"}
    if isinstance(action, PassAction):
        return {"type": "pass"}
    raise ValueError(f"unknown rule action type: {type(action).__name__}")


__all__ = [
    "DecisionSample",
    "EventRef",
    "SampleAlignmentError",
    "SampleAudit",
    "action_json",
]