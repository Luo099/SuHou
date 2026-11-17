"""Per-round visible event history with viewer filtering.

The full round sequence is kept once. A sample stores only the observer and the
cutoff position, and :meth:`RoundHistory.view` regenerates the visible prefix.
This is the storage optimisation allowed by the Observation v1 contract; the
resulting observation content is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from suhou.observation.model import (
    EventType,
    VisibleEvent,
    freeze_payload,
)


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One round event before viewer-specific redaction.

    The payload may contain hidden information (an opponent's drawn tile or all
    four initial hands). :meth:`visible_to` is the only supported way to turn
    it into observation history.
    """

    seq: int
    type: EventType
    actor: int | None
    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.seq < 0:
            raise ValueError("history seq must be non-negative")
        if self.actor is not None and not 0 <= self.actor < 4:
            raise ValueError(f"history actor must be in 0..3, got {self.actor}")
        object.__setattr__(self, "data", freeze_payload(self.data))

    def visible_to(self, viewer: int) -> VisibleEvent:
        if not 0 <= viewer < 4:
            raise ValueError(f"viewer must be in 0..3, got {viewer}")
        data = dict(self.data)
        if self.type == EventType.ROUND_START:
            # Never retain the other seats' initial hands, even accidentally.
            hands = data.pop("hands", ())
            if len(hands) == 4:
                data["initial_hand"] = hands[viewer]
        elif self.type == EventType.DRAW and self.actor != viewer:
            # Everyone sees that a draw happened; only the drawer sees the tile.
            data["tile"] = None
        return VisibleEvent(self.seq, self.type, self.actor, data)


class RoundHistory:
    """Append-only round sequence shared by every viewer of that round."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: list[HistoryEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[HistoryEntry, ...]:
        return tuple(self._entries)

    def append(self, entry: HistoryEntry) -> None:
        if self._entries and entry.seq <= self._entries[-1].seq:
            raise ValueError(
                f"history seq must increase, got {entry.seq} after {self._entries[-1].seq}"
            )
        self._entries.append(entry)

    def view(self, viewer: int, cutoff: int | None = None) -> tuple[VisibleEvent, ...]:
        """Return events with ``seq < cutoff`` for ``viewer``.

        ``cutoff`` is the sequence number of the decision event, so the event
        being predicted is never part of history.
        """
        if not 0 <= viewer < 4:
            raise ValueError(f"viewer must be in 0..3, got {viewer}")
        if cutoff is not None and cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        return tuple(
            entry.visible_to(viewer)
            for entry in self._entries
            if cutoff is None or entry.seq < cutoff
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> RoundHistory:
        clone = RoundHistory()
        clone._entries = list(self._entries)
        return clone
