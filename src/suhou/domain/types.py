"""Common mahjong enums and player identity shared by every consumer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# Tenhou <GO type> rule bits (community protocol knowledge): 0x08 hanchan,
# 0x10 sanma. Decoding lives here so the adapter and the state model share a
# single interpretation instead of each consumer reading the bits again.
_GO_HANCHAN = 0x08
_GO_SANMA = 0x10
_GO_NO_RED_FIVES = 0x02


class MatchType(StrEnum):
    """The match length declared by the source game settings."""

    TONPU = "tonpu"
    HANCHAN = "hanchan"


class MeldType(StrEnum):
    """The call forms represented in observations (nuki is out of scope)."""

    CHI = "chi"
    PON = "pon"
    MINKAN = "minkan"
    ANKAN = "ankan"
    KAKAN = "kakan"


class MeldKind(StrEnum):
    """Every internal call form, including three-player nuki."""

    CHI = "chi"
    PON = "pon"
    MINKAN = "minkan"
    ANKAN = "ankan"
    KAKAN = "kakan"
    NUKI = "nuki"


@dataclass(frozen=True, slots=True)
class PlayerInfo:
    """A player's source-visible identity and ranked-match metadata."""

    name: str
    dan: int | None = None
    rate: float | None = None
    sex: str | None = None


def match_type_from_game_type(game_type: int | None) -> MatchType:
    """Decode tonpu/hanchan from the source's rule bits.

    This is the only rule-bit interpretation. Raises ``ValueError`` for
    three-player logs and for logs without rule bits; callers that need a
    different error type wrap it.
    """
    if game_type is None:
        raise ValueError(
            "Cannot determine tonpu/hanchan: the game has no <GO type> rule bits"
        )
    if game_type & _GO_SANMA:
        raise ValueError("Three-player logs are outside Observation v1")
    return MatchType.HANCHAN if game_type & _GO_HANCHAN else MatchType.TONPU


def red_fives_from_game_type(game_type: int | None) -> bool:
    """Return whether red fives exist under the declared game settings."""
    if game_type is None:
        return True
    return not bool(game_type & _GO_NO_RED_FIVES)
