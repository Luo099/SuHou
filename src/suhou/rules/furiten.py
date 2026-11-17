"""Discard, temporary, and riichi furiten state.

Discard furiten is derived from the current wait set and the player's own
discard pile each time it is queried. Temporary furiten is set when a legal
ron opportunity was passed and cleared by the player's next draw. Riichi
furiten is permanent for the round once a riichi player passes on a ron.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(slots=True)
class FuritenState:
    temporary: bool = False
    riichi: bool = False

    def reset(self) -> None:
        self.temporary = False
        self.riichi = False

    def on_draw(self) -> None:
        """The player's own draw clears temporary furiten only."""
        self.temporary = False

    def on_missed_ron(self, *, riichi: bool) -> None:
        self.temporary = True
        if riichi:
            self.riichi = True

    def is_furiten(self, waits: Iterable[int], discarded_kinds: Iterable[int]) -> bool:
        if self.temporary or self.riichi:
            return True
        wait_set = set(waits)
        return bool(wait_set) and not wait_set.isdisjoint(discarded_kinds)


__all__ = ["FuritenState"]
