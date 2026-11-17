"""Winning-shape and yaku legality for the four-player rules.

The engine only needs to know whether a hand has at least one yaku; it does
not score the hand. Dora are deliberately absent from the result, so a hand
whose only value comes from dora is not a legal ron/tsumo.

``yaku_names`` is also useful for auditing against a recorded Tenhou AGARI.
Names are stable snake-case identifiers, not display strings.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from suhou.domain.melds import Meld
from suhou.domain.tiles import Wind
from suhou.domain.types import MeldKind
from suhou.rules.model import RuleError, WinContext
from suhou.rules.shanten import (
    is_agari_shape,
    is_chiitoitsu_complete,
    is_kokushi_complete,
)
from suhou.rules.tiles import (
    GREEN_KINDS,
    WIND_KIND_INDEX,
    is_honor,
    is_simple,
    is_terminal,
    is_terminal_or_honor,
    physical_index,
)

DRAGON_KINDS = frozenset({31, 32, 33})
WIND_KINDS = frozenset({27, 28, 29, 30})


@dataclass(frozen=True, slots=True)
class MeldShape:
    """One meld as seen by shape analysis."""

    kind: str
    tiles: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.kind not in ("chi", "pon", "minkan", "ankan", "kakan"):
            raise ValueError(f"Unknown meld shape kind: {self.kind!r}")
        if self.kind == "chi":
            if not (
                self.tiles[0] < 27
                and self.tiles[1] == self.tiles[0] + 1
                and self.tiles[2] == self.tiles[0] + 2
                and self.tiles[0] // 9 == self.tiles[2] // 9
            ):
                raise ValueError(f"Invalid chi shape: {self.tiles}")
        elif not (self.tiles[0] == self.tiles[1] == self.tiles[2]):
            raise ValueError(f"Invalid triplet shape: {self.tiles}")

    @property
    def is_sequence(self) -> bool:
        return self.kind == "chi"

    @property
    def is_open(self) -> bool:
        return self.kind != "ankan"

    @property
    def is_kan(self) -> bool:
        return self.kind in ("minkan", "ankan", "kakan")


@dataclass(frozen=True, slots=True)
class _Group:
    kind: str
    tiles: tuple[int, int, int]
    open: bool
    kan: bool


def meld_shapes(melds: Iterable[Meld]) -> tuple[MeldShape, ...]:
    """Convert replay melds to shape records, rejecting three-player nuki."""
    shapes: list[MeldShape] = []
    for meld in melds:
        if meld.kind is MeldKind.CHI:
            indices = tuple(sorted(physical_index(tile) for tile in meld.tiles))
            if len(indices) != 3:
                raise RuleError(f"Chi meld has {len(indices)} tiles")
            shapes.append(
                MeldShape("chi", (indices[0], indices[1], indices[2]))
            )
            continue
        if meld.kind is MeldKind.NUKI:
            raise RuleError("Nuki (three-player) is outside the four-player rules")
        index = physical_index(meld.tiles[0])
        if any(physical_index(tile) != index for tile in meld.tiles):
            raise RuleError(f"{meld.kind} meld contains mixed tile kinds")
        shapes.append(MeldShape(meld.kind.value, (index, index, index)))
    return tuple(shapes)


def is_closed(melds: Sequence[MeldShape]) -> bool:
    return all(not meld.is_open for meld in melds)


def yaku_names(
    counts: Sequence[int],
    melds: Sequence[MeldShape],
    winning_tile: int,
    context: WinContext,
    *,
    seat_wind: Wind | str | int,
    round_wind: Wind | str | int,
) -> frozenset[str]:
    """Return every yaku detected for the winning hand (dora excluded)."""
    if len(counts) != 34:
        raise ValueError("counts must contain 34 tile kinds")
    closed = is_closed(melds)
    if not is_agari_shape(counts, len(melds), closed=closed):
        return frozenset()

    seat = _wind_index(seat_wind)
    round_ = _wind_index(round_wind)
    names = set(_situational_names(context, closed))

    if closed and not melds:
        if is_chiitoitsu_complete(counts):
            names.add("chiitoitsu")
            names.update(_chiitoi_names(counts))
        if is_kokushi_complete(counts):
            names.add("kokushi")
        if _is_chuuren(counts):
            names.add("chuuren")

    meld_groups = [_group_from_meld(meld) for meld in melds]
    for pair, concealed in _decompositions(counts, 4 - len(melds)):
        groups = [*meld_groups, *concealed]
        names.update(
            _standard_names(
                groups,
                pair,
                winning_tile,
                context,
                seat,
                round_,
                closed,
            )
        )
    return frozenset(names)


def has_yaku(
    counts: Sequence[int],
    melds: Sequence[MeldShape],
    winning_tile: int,
    context: WinContext,
    *,
    seat_wind: Wind | str | int,
    round_wind: Wind | str | int,
) -> bool:
    return bool(
        yaku_names(
            counts,
            melds,
            winning_tile,
            context,
            seat_wind=seat_wind,
            round_wind=round_wind,
        )
    )


def _situational_names(context: WinContext, closed: bool) -> set[str]:
    names: set[str] = set()
    if context.riichi:
        names.add("riichi")
    if context.ippatsu and context.riichi:
        names.add("ippatsu")
    if context.tsumo and closed:
        names.add("menzen_tsumo")
    if context.chankan:
        names.add("chankan")
    if context.rinshan:
        names.add("rinshan")
    if context.haitei:
        names.add("haitei")
    if context.houtei:
        names.add("houtei")
    if context.tenhou:
        names.add("tenhou")
    if context.chiihou:
        names.add("chiihou")
    return names


def _chiitoi_names(counts: Sequence[int]) -> set[str]:
    indices = _expanded_indices(counts)
    names: set[str] = set()
    if all(is_simple(index) for index in indices):
        names.add("tanyao")
    if all(is_terminal_or_honor(index) for index in indices):
        names.add("honroutou")
    if all(is_terminal(index) for index in indices):
        names.add("chinroutou")
    if all(is_honor(index) for index in indices):
        names.add("tsuuiisou")
    if all(index in GREEN_KINDS for index in indices):
        names.add("ryuuiisou")
    names.update(_flush_names(indices))
    return names


def _standard_names(
    groups: Sequence[_Group],
    pair: int,
    winning_tile: int,
    context: WinContext,
    seat_wind: int,
    round_wind: int,
    closed: bool,
) -> set[str]:
    names: set[str] = set()
    indices = [tile for group in groups for tile in group.tiles] + [pair, pair]
    sequences = [group for group in groups if group.kind == "sequence"]
    triplets = [group for group in groups if group.kind == "triplet"]

    if all(is_simple(index) for index in indices):
        names.add("tanyao")
    if (
        closed
        and len(sequences) == len(groups)
        and not _is_value_pair(pair, seat_wind, round_wind)
        and _has_ryanmen_wait(groups, winning_tile)
    ):
        names.add("pinfu")
    if any(
        group.tiles[0] in DRAGON_KINDS or group.tiles[0] in (seat_wind, round_wind)
        for group in triplets
    ):
        names.add("yakuhai")
    if len(triplets) == len(groups):
        names.add("toitoi")

    concealed_triplets = _concealed_triplet_count(groups, winning_tile, context.tsumo)
    if concealed_triplets >= 3:
        names.add("sanankou")
    if concealed_triplets == 4:
        names.add("suuankou")

    kans = sum(1 for group in groups if group.kan)
    if kans >= 3:
        names.add("sankantsu")
    if kans == 4:
        names.add("suukantsu")

    if closed:
        sequence_counts = Counter(
            (group.tiles[0] // 9, group.tiles[0] % 9) for group in sequences
        )
        if any(value >= 2 for value in sequence_counts.values()):
            names.add("iipeikou")
        if sum(value // 2 for value in sequence_counts.values()) >= 2:
            names.add("ryanpeikou")

    starts = {(group.tiles[0] // 9, group.tiles[0] % 9) for group in sequences}
    if any(
        {(0, start), (1, start), (2, start)} <= starts for start in range(7)
    ):
        names.add("sanshoku_doujun")
    triplet_kinds = {group.tiles[0] for group in triplets}
    if any({index, index + 9, index + 18} <= triplet_kinds for index in range(9)):
        names.add("sanshoku_doukou")
    if any({(suit, 0), (suit, 3), (suit, 6)} <= starts for suit in range(3)):
        names.add("ittsu")

    all_contain_terminal = all(
        _group_contains_terminal_or_honor(group) for group in groups
    ) and is_terminal_or_honor(pair)
    if all_contain_terminal and sequences:
        if any(is_honor(index) for index in indices):
            names.add("chanta")
        else:
            names.add("junchan")

    if all(is_terminal_or_honor(index) for index in indices):
        names.add("honroutou")
    if all(is_honor(index) for index in indices):
        names.add("tsuuiisou")
    if all(is_terminal(index) for index in indices):
        names.add("chinroutou")
    if all(index in GREEN_KINDS for index in indices):
        names.add("ryuuiisou")

    dragon_triplets = sum(
        1 for group in triplets if group.tiles[0] in DRAGON_KINDS
    )
    if dragon_triplets == 3:
        names.add("daisangen")
    elif dragon_triplets == 2 and pair in DRAGON_KINDS:
        names.add("shousangen")

    wind_triplets = sum(1 for group in triplets if group.tiles[0] in WIND_KINDS)
    if wind_triplets == 4:
        names.add("daisuushii")
    elif wind_triplets == 3 and pair in WIND_KINDS:
        names.add("shousuushii")

    names.update(_flush_names(indices))
    return names


def _concealed_triplet_count(
    groups: Sequence[_Group], winning_tile: int, tsumo: bool
) -> int:
    count = 0
    for group in groups:
        if group.kind != "triplet" or group.open:
            continue
        if not tsumo and winning_tile in group.tiles:
            # A ron-completed triplet is not a concealed triplet. This is the
            # conservative reading (hand 4444 + sequence can still be
            # interpreted as keeping the triplet concealed; such rare hands
            # normally have another yaku).
            continue
        count += 1
    return count


def _group_contains_terminal_or_honor(group: _Group) -> bool:
    if group.kind == "sequence":
        return group.tiles[0] % 9 == 0 or group.tiles[2] % 9 == 8
    return is_terminal_or_honor(group.tiles[0])


def _has_ryanmen_wait(groups: Sequence[_Group], winning_tile: int) -> bool:
    for group in groups:
        if group.kind != "sequence" or winning_tile not in group.tiles:
            continue
        start = group.tiles[0]
        if winning_tile == start and start % 9 != 6:
            return True
        if winning_tile == start + 2 and start % 9 != 0:
            return True
    return False


def _is_value_pair(pair: int, seat_wind: int, round_wind: int) -> bool:
    return pair in DRAGON_KINDS or pair in (seat_wind, round_wind)


def _flush_names(indices: Iterable[int]) -> set[str]:
    values = list(indices)
    suits = {index // 9 for index in values if index < 27}
    honors = any(index >= 27 for index in values)
    names: set[str] = set()
    if len(suits) == 1:
        if honors:
            names.add("honitsu")
        else:
            names.add("chinitsu")
    return names


def _is_chuuren(counts: Sequence[int]) -> bool:
    suits = [suit for suit in range(3) if any(counts[suit * 9 + offset] for offset in range(9))]
    if len(suits) != 1 or any(counts[index] for index in range(27) if index // 9 != suits[0]):
        return False
    if any(counts[index] for index in range(27, 34)):
        return False
    base = suits[0] * 9
    if counts[base] < 3 or counts[base + 8] < 3:
        return False
    if any(counts[base + offset] < 1 for offset in range(1, 8)):
        return False
    return sum(counts[base : base + 9]) == 14


def _wind_index(value: Wind | str | int) -> int:
    if isinstance(value, Wind):
        return WIND_KIND_INDEX[value.value]
    if isinstance(value, str):
        if value not in WIND_KIND_INDEX:
            raise ValueError(f"Unknown wind: {value!r}")
        return WIND_KIND_INDEX[value]
    if not 27 <= value <= 30:
        raise ValueError(f"Wind kind index must be in 27..30, got {value}")
    return value


def _group_from_meld(meld: MeldShape) -> _Group:
    if meld.is_sequence:
        return _Group("sequence", meld.tiles, open=True, kan=False)
    return _Group("triplet", meld.tiles, open=meld.is_open, kan=meld.is_kan)


def _decompositions(
    counts: Sequence[int], set_count: int
) -> Iterator[tuple[int, tuple[_Group, ...]]]:
    values = list(counts)
    for pair in range(34):
        if values[pair] < 2:
            continue
        values[pair] -= 2
        yield from (
            (pair, groups) for groups in _form_sets(values, 0, set_count, ())
        )
        values[pair] += 2


def _form_sets(
    values: list[int],
    start: int,
    remaining: int,
    prefix: tuple[_Group, ...],
) -> Iterator[tuple[_Group, ...]]:
    while start < 34 and values[start] == 0:
        start += 1
    if remaining == 0:
        if start == 34:
            yield prefix
        return
    if start == 34:
        return

    if values[start] >= 3:
        values[start] -= 3
        group = _Group("triplet", (start, start, start), open=False, kan=False)
        yield from _form_sets(values, start, remaining - 1, (*prefix, group))
        values[start] += 3

    if (
        start < 27
        and start % 9 <= 6
        and values[start + 1]
        and values[start + 2]
    ):
        values[start] -= 1
        values[start + 1] -= 1
        values[start + 2] -= 1
        group = _Group(
            "sequence", (start, start + 1, start + 2), open=False, kan=False
        )
        yield from _form_sets(values, start, remaining - 1, (*prefix, group))
        values[start] += 1
        values[start + 1] += 1
        values[start + 2] += 1


def _expanded_indices(counts: Sequence[int]) -> list[int]:
    return [index for index, count in enumerate(counts) for _ in range(count)]


__all__ = [
    "MeldShape",
    "has_yaku",
    "is_closed",
    "meld_shapes",
    "yaku_names",
]
