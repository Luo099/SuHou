"""Rule-engine decision records and structured legal actions.

Actions are the rules layer's canonical output. They are defined with
:class:`~suhou.domain.tiles.Tile` (kind plus red-five identity) and meld
sequence numbers, never with Tenhou bit masks or a model-specific action index.
A separate action codec can later map these records onto a fixed vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from suhou.domain.tiles import Tile


class RuleError(ValueError):
    """An event or view cannot be interpreted by the player rule engine."""


class RuleDecisionKind(StrEnum):
    """What just happened and which decision window is now open."""

    TURN = "turn"
    AFTER_CALL = "after_call"
    RIICHI_DISCARD = "riichi_discard"
    RESPONSE = "response"
    KAN_RESPONSE = "kan_response"


@dataclass(frozen=True, slots=True)
class DiscardAction:
    tile: Tile


@dataclass(frozen=True, slots=True)
class ChiAction:
    consumed_tiles: tuple[Tile, ...]

    def __post_init__(self) -> None:
        _require_count("chi", self.consumed_tiles, 2)


@dataclass(frozen=True, slots=True)
class PonAction:
    consumed_tiles: tuple[Tile, ...]

    def __post_init__(self) -> None:
        _require_count("pon", self.consumed_tiles, 2)


@dataclass(frozen=True, slots=True)
class MinkanAction:
    consumed_tiles: tuple[Tile, ...]

    def __post_init__(self) -> None:
        _require_count("minkan", self.consumed_tiles, 3)


@dataclass(frozen=True, slots=True)
class AnkanAction:
    consumed_tiles: tuple[Tile, ...]

    def __post_init__(self) -> None:
        _require_count("ankan", self.consumed_tiles, 4)


@dataclass(frozen=True, slots=True)
class KakanAction:
    added_tile: Tile
    target_meld_seq: int

    def __post_init__(self) -> None:
        if self.target_meld_seq < 0:
            raise ValueError("kakan target_meld_seq must be non-negative")


@dataclass(frozen=True, slots=True)
class RiichiAction:
    pass


@dataclass(frozen=True, slots=True)
class TsumoAction:
    pass


@dataclass(frozen=True, slots=True)
class RonAction:
    pass


@dataclass(frozen=True, slots=True)
class KyushuAction:
    pass


@dataclass(frozen=True, slots=True)
class PassAction:
    pass


RuleAction = (
    DiscardAction
    | ChiAction
    | PonAction
    | MinkanAction
    | AnkanAction
    | KakanAction
    | RiichiAction
    | TsumoAction
    | RonAction
    | KyushuAction
    | PassAction
)


@dataclass(frozen=True, slots=True)
class RuleDecision:
    """The legal actions that belong to one viewer at one event boundary."""

    viewer: int
    round_id: int
    trigger_seq: int
    kind: RuleDecisionKind
    actions: tuple[RuleAction, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.viewer < 4:
            raise ValueError(f"viewer must be in 0..3, got {self.viewer}")
        if self.round_id < 0:
            raise ValueError("round_id must be non-negative")
        if self.trigger_seq < 0:
            raise ValueError("trigger_seq must be non-negative")
        if not self.actions:
            raise ValueError("a rule decision must contain at least one action")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("rule decision actions must be unique")


@dataclass(frozen=True, slots=True)
class WinContext:
    """Situational yaku inputs for one winning-shape check."""

    riichi: bool = False
    ippatsu: bool = False
    tsumo: bool = False
    chankan: bool = False
    rinshan: bool = False
    haitei: bool = False
    houtei: bool = False
    tenhou: bool = False
    chiihou: bool = False


def _require_count(name: str, tiles: tuple[Tile, ...], expected: int) -> None:
    if len(tiles) != expected:
        raise ValueError(f"{name} requires {expected} consumed tiles, got {len(tiles)}")


__all__ = [
    "AnkanAction",
    "ChiAction",
    "DiscardAction",
    "KakanAction",
    "KyushuAction",
    "MinkanAction",
    "PassAction",
    "PonAction",
    "RiichiAction",
    "RonAction",
    "RuleAction",
    "RuleDecision",
    "RuleDecisionKind",
    "RuleError",
    "TsumoAction",
    "WinContext",
]
