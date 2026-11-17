"""Shanten, tenpai, and winning-tile analysis.

The standard-form search treats four complete sets plus one pair as the goal.
Open and annotated kan melds each count as one complete set; the concealed
hand therefore contains ``13 - 3 * melds`` tiles before a draw and
``14 - 3 * melds`` after one. Chiitoitsu and kokushi require a fully closed
hand with no melds at all.
"""

from __future__ import annotations

from collections.abc import Sequence

from suhou.rules.tiles import ORPHAN_KINDS


def standard_shanten(counts: Sequence[int], melds: int = 0) -> int:
    """Return the standard-form shanten (``-1`` is a complete hand)."""
    if len(counts) != 34:
        raise ValueError("counts must contain 34 tile kinds")
    if not 0 <= melds <= 4:
        raise ValueError("melds must be in 0..4")
    values = list(counts)
    best = 8
    kinds = len(values)

    def evaluate(sets: int, partials: int, pair: bool) -> None:
        nonlocal best
        total_sets = melds + sets
        if total_sets > 4:
            return
        effective_partials = min(partials, 4 - total_sets)
        shanten = 8 - 2 * total_sets - effective_partials - (1 if pair else 0)
        if shanten < best:
            best = shanten

    def search(start: int, sets: int, partials: int, pair: bool) -> None:
        nonlocal best
        if best == -1:
            return
        while start < kinds and values[start] == 0:
            start += 1
        evaluate(sets, partials, pair)
        if start == kinds:
            return
        index = start

        if values[index] >= 3 and melds + sets < 4:
            values[index] -= 3
            search(index, sets + 1, partials, pair)
            values[index] += 3

        if (
            index < 27
            and index % 9 <= 6
            and values[index + 1]
            and values[index + 2]
            and melds + sets < 4
        ):
            values[index] -= 1
            values[index + 1] -= 1
            values[index + 2] -= 1
            search(index, sets + 1, partials, pair)
            values[index] += 1
            values[index + 1] += 1
            values[index + 2] += 1

        if values[index] >= 2 and not pair:
            values[index] -= 2
            search(index, sets, partials, True)
            values[index] += 2

        if values[index] >= 2 and melds + sets + partials < 4:
            values[index] -= 2
            search(index, sets, partials + 1, pair)
            values[index] += 2

        if (
            index < 27
            and index % 9 <= 7
            and values[index + 1]
            and melds + sets + partials < 4
        ):
            values[index] -= 1
            values[index + 1] -= 1
            search(index, sets, partials + 1, pair)
            values[index] += 1
            values[index + 1] += 1

        if (
            index < 27
            and index % 9 <= 6
            and values[index + 2]
            and melds + sets + partials < 4
        ):
            values[index] -= 1
            values[index + 2] -= 1
            search(index, sets, partials + 1, pair)
            values[index] += 1
            values[index + 2] += 1

        values[index] -= 1
        search(index, sets, partials, pair)
        values[index] += 1

    search(0, 0, 0, False)
    return best


def chiitoitsu_shanten(counts: Sequence[int]) -> int:
    if len(counts) != 34:
        raise ValueError("counts must contain 34 tile kinds")
    pairs = sum(1 for value in counts if value >= 2)
    kinds = sum(1 for value in counts if value > 0)
    return 6 - pairs + max(0, 7 - kinds)


def kokushi_shanten(counts: Sequence[int]) -> int:
    if len(counts) != 34:
        raise ValueError("counts must contain 34 tile kinds")
    orphans = sum(1 for index in ORPHAN_KINDS if counts[index] > 0)
    pair = any(counts[index] >= 2 for index in ORPHAN_KINDS)
    return 13 - orphans - (1 if pair else 0)


def shanten(counts: Sequence[int], melds: int = 0) -> int:
    """Return the best shanten across the supported four-player forms."""
    result = standard_shanten(counts, melds)
    if melds == 0:
        result = min(result, chiitoitsu_shanten(counts), kokushi_shanten(counts))
    return result


def is_standard_complete(counts: Sequence[int], melds: int = 0) -> bool:
    """Whether the counts form four sets and a pair for the given meld count."""
    if len(counts) != 34:
        raise ValueError("counts must contain 34 tile kinds")
    if not 0 <= melds <= 4:
        raise ValueError("melds must be in 0..4")
    if sum(counts) != 14 - 3 * melds:
        return False
    values = list(counts)
    for pair in range(34):
        if values[pair] < 2:
            continue
        values[pair] -= 2
        if _forms_sets(values, 0, 4 - melds):
            values[pair] += 2
            return True
        values[pair] += 2
    return False


def is_chiitoitsu_complete(counts: Sequence[int]) -> bool:
    return len(counts) == 34 and sum(counts) == 14 and all(
        value in (0, 2) for value in counts
    )


def is_kokushi_complete(counts: Sequence[int]) -> bool:
    if len(counts) != 34 or sum(counts) != 14:
        return False
    if any(counts[index] == 0 for index in ORPHAN_KINDS):
        return False
    if any(counts[index] for index in range(34) if index not in ORPHAN_KINDS):
        return False
    return sum(1 for index in ORPHAN_KINDS if counts[index] == 2) == 1


def is_agari_shape(counts: Sequence[int], melds: int = 0, *, closed: bool = True) -> bool:
    """Whether these 14 - 3*melds tiles form any supported winning shape."""
    if is_standard_complete(counts, melds):
        return True
    if closed and melds == 0:
        return is_chiitoitsu_complete(counts) or is_kokushi_complete(counts)
    return False


def winning_tiles(
    counts: Sequence[int], melds: int = 0, *, closed: bool = True
) -> tuple[int, ...]:
    """Return sorted kinds that complete a ``13 - 3*melds`` waiting hand."""
    if len(counts) != 34:
        raise ValueError("counts must contain 34 tile kinds")
    values = list(counts)
    waits: list[int] = []
    for index in range(34):
        if values[index] >= 4:
            continue
        values[index] += 1
        if is_agari_shape(values, melds, closed=closed):
            waits.append(index)
        values[index] -= 1
    return tuple(waits)


def is_tenpai(counts: Sequence[int], melds: int = 0, *, closed: bool = True) -> bool:
    return bool(winning_tiles(counts, melds, closed=closed))


def tenpai_discards(
    counts: Sequence[int], melds: int = 0, *, closed: bool = True
) -> tuple[int, ...]:
    """Return sorted kinds whose discard leaves a ``13 - 3*melds`` tenpai hand."""
    if len(counts) != 34:
        raise ValueError("counts must contain 34 tile kinds")
    values = list(counts)
    discards: list[int] = []
    for index in range(34):
        if values[index] == 0:
            continue
        values[index] -= 1
        if is_tenpai(values, melds, closed=closed):
            discards.append(index)
        values[index] += 1
    return tuple(discards)


def _forms_sets(values: list[int], start: int, remaining: int) -> bool:
    while start < 34 and values[start] == 0:
        start += 1
    if remaining == 0:
        return start == 34
    if start == 34:
        return False

    if values[start] >= 3:
        values[start] -= 3
        if _forms_sets(values, start, remaining - 1):
            values[start] += 3
            return True
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
        if _forms_sets(values, start, remaining - 1):
            values[start] += 1
            values[start + 1] += 1
            values[start + 2] += 1
            return True
        values[start] += 1
        values[start + 1] += 1
        values[start + 2] += 1
    return False


__all__ = [
    "chiitoitsu_shanten",
    "is_agari_shape",
    "is_chiitoitsu_complete",
    "is_kokushi_complete",
    "is_standard_complete",
    "is_tenpai",
    "kokushi_shanten",
    "shanten",
    "standard_shanten",
    "tenpai_discards",
    "winning_tiles",
]
