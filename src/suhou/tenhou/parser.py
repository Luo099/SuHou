"""Read Tenhou mjlog 2.3 without changing its event order."""

from __future__ import annotations

import gzip
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from types import MappingProxyType

from suhou.tenhou.models import GameRecord, Player, TenhouEvent, players_from_attributes
from suhou.tenhou.tiles import validate_tile

_DRAW_SEATS = dict(zip("TUVW", range(4), strict=True))
_DISCARD_SEATS = dict(zip("DEFG", range(4), strict=True))
_TILE_TAG = re.compile(r"([TUVWDEFG])(\d+)$")
_KNOWN_TAGS = {
    "SHUFFLE", "GO", "UN", "TAIKYOKU", "INIT", "N", "REACH", "DORA", "AGARI", "RYUUKYOKU",
    "BYE", "RESUME",
}


class TenhouParseError(ValueError):
    """An invalid or unsupported Tenhou log, with source context."""


def _int_values(element: ET.Element, name: str, length: int | None = None) -> tuple[int, ...]:
    value = element.get(name)
    if value is None:
        raise TenhouParseError(f"<{element.tag}> lacks required '{name}'")
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise TenhouParseError(f"<{element.tag}> has non-integer '{name}': {value!r}") from error
    if length is not None and len(values) != length:
        raise TenhouParseError(f"<{element.tag}> '{name}' must contain {length} integers")
    return values


def _tile_values(element: ET.Element, name: str) -> tuple[int, ...]:
    try:
        return tuple(validate_tile(value) for value in _int_values(element, name))
    except ValueError as error:
        raise TenhouParseError(f"<{element.tag}> invalid '{name}': {error}") from error


def _score_pairs(
    element: ET.Element,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    values = _int_values(element, "sc", 8)
    return (
        tuple(values[0::2][seat] * 100 for seat in range(4)),
        tuple(values[1::2][seat] * 100 for seat in range(4)),
    )


def _final_result(
    element: ET.Element,
) -> tuple[tuple[int, int, int, int], tuple[float, ...]] | None:
    raw = element.get("owari")
    if raw is None:
        return None
    values = raw.split(",")
    if len(values) != 8:
        raise TenhouParseError(f"<{element.tag}> 'owari' must contain eight values")
    try:
        return (
            tuple(int(float(values[seat * 2]) * 100) for seat in range(4)),
            tuple(float(values[seat * 2 + 1]) for seat in range(4)),
        )
    except ValueError as error:
        raise TenhouParseError(f"<{element.tag}> has invalid 'owari'") from error


def _parse_event(element: ET.Element, index: int) -> TenhouEvent:
    tag = element.tag
    match = _TILE_TAG.fullmatch(tag)
    attributes = MappingProxyType(dict(element.attrib))
    common: dict[str, object] = {"index": index, "raw_tag": tag, "attributes": attributes}
    try:
        if match:
            prefix, raw_tile = match.groups()
            seat = _DRAW_SEATS.get(prefix.upper(), _DISCARD_SEATS.get(prefix.upper()))
            common.update(tag="T" if prefix.upper() in _DRAW_SEATS else "D", seat=seat)
            if seat is None:
                raise TenhouParseError(f"Unsupported tile event <{tag}>")
            return TenhouEvent(tile=validate_tile(int(raw_tile)), **common)
        if tag not in _KNOWN_TAGS:
            raise TenhouParseError(f"Unsupported Tenhou tag <{tag}>")
        if tag == "INIT":
            seed = _int_values(element, "seed", 6)
            if not 0 <= seed[0] < 16:
                raise TenhouParseError("<INIT> seed round index must be in 0..15")
            validate_tile(seed[5])
            oya = int(element.get("oya", "-1"))
            if not 0 <= oya < 4:
                raise TenhouParseError("<INIT> oya must be in 0..3")
            hands = tuple(_tile_values(element, f"hai{seat}") for seat in range(4))
            if any(len(hand) != 13 for hand in hands):
                raise TenhouParseError("<INIT> each player must have exactly 13 tiles")
            all_tiles = tuple(tile for hand in hands for tile in hand)
            if len(set(all_tiles)) != 52:
                raise TenhouParseError("<INIT> physical tile copies must be unique")
            return TenhouEvent(
                tag=tag,
                seed=seed,
                oya=oya,
                scores=tuple(value * 100 for value in _int_values(element, "ten", 4)),
                tiles=all_tiles,
                **common,
            )

        if tag == "N":
            seat = int(element.get("who", "-1"))
            if not 0 <= seat < 4:
                raise TenhouParseError("<N> who must be in 0..3")
            mask = int(element.get("m", "-1"))
            from suhou.tenhou.melds import decode_meld
            decoded = decode_meld(seat, mask)
            return TenhouEvent(
                tag=tag, seat=seat, meld=mask, tiles=decoded.tiles,
                meld_record=decoded, **common,
            )
        if tag == "REACH":
            seat = int(element.get("who", "-1"))
            step = int(element.get("step", "-1"))
            if not 0 <= seat < 4 or step not in (1, 2):
                raise TenhouParseError("<REACH> requires who in 0..3 and step 1 or 2")
            scores = None
            if "ten" in element.attrib:
                scores = tuple(value * 100 for value in _int_values(element, "ten", 4))
            return TenhouEvent(tag=tag, seat=seat, step=step, scores=scores, **common)
        if tag == "DORA":
            return TenhouEvent(tag=tag, tile=validate_tile(int(element.get("hai", "-1"))), **common)
        if tag in ("BYE", "RESUME"):
            seat = int(element.get("who", "-1"))
            if not 0 <= seat < 4:
                raise TenhouParseError(f"<{tag}> who must be in 0..3")
            return TenhouEvent(tag=tag, seat=seat, **common)
        if tag in ("AGARI", "RYUUKYOKU"):
            scores_before, deltas = _score_pairs(element)
            final = _final_result(element)
            winners = None
            if tag == "AGARI":
                who = int(element.get("who", "-1"))
                from_who = int(element.get("fromWho", "-1"))
                if not 0 <= who < 4 or not 0 <= from_who < 4:
                    raise TenhouParseError("<AGARI> who/fromWho must be in 0..3")
                winners = ((who, from_who),)
            return TenhouEvent(
                tag=tag,
                scores_before=scores_before,
                deltas=deltas,
                winners=winners,
                final_scores=final[0] if final else None,
                final_points=final[1] if final else None,
                draw_type=element.get("type") if tag == "RYUUKYOKU" else None,
                **common,
            )
        return TenhouEvent(tag=tag, **common)
    except (ValueError, TypeError) as error:
        if isinstance(error, TenhouParseError):
            raise TenhouParseError(f"Event {index} <{tag}>: {error}") from error
        raise TenhouParseError(f"Event {index} <{tag}> has invalid attributes: {error}") from error


def _reject_doctype(data: bytes) -> bytes:
    """XML internal subsets/entities are not part of Tenhou's protocol."""
    if b"<!DOCTYPE" in data.upper():
        raise TenhouParseError("DOCTYPE is not allowed in Tenhou logs")
    return data


def parse_tenhou(path: str | Path) -> GameRecord:
    """Parse an ordinary or gzip-compressed Tenhou XML log."""
    source = Path(path)
    try:
        data = source.read_bytes()
        if data.startswith(b"\x1f\x8b"):
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as archive:
                data = archive.read()
        data = _reject_doctype(data)
        root = ET.fromstring(data)
    except TenhouParseError:
        raise
    except (OSError, EOFError, gzip.BadGzipFile, ET.ParseError) as error:
        raise TenhouParseError(f"Cannot read Tenhou XML {source}: {error}") from error
    if root.tag != "mjloggm" or root.get("ver") != "2.3":
        raise TenhouParseError("Expected <mjloggm ver=\"2.3\"> root element")

    events: list[TenhouEvent] = []
    players: tuple[Player, Player, Player, Player] | None = None
    game_type: int | None = None
    lobby: int | None = None
    shuffle: str | None = None
    shuffle_reference: str | None = None
    for index, element in enumerate(root):
        event = _parse_event(element, index)
        events.append(event)
        if event.tag == "UN":
            # Later <UN> tags can be partial score updates without player names;
            # only the initial roster carries n0..n3. Don't let a partial update
            # overwrite the authoritative player list.
            if all(f"n{seat}" in element.attrib for seat in range(4)):
                try:
                    players = players_from_attributes(element.attrib)
                except ValueError as error:
                    raise TenhouParseError(f"Event {index} <UN>: {error}") from error
        elif event.tag == "GO":
            try:
                game_type = int(element.get("type", "0"))
                lobby = int(element.get("lobby", "0"))
            except ValueError as error:
                raise TenhouParseError(f"Event {index} <GO> has invalid type/lobby") from error
        elif event.tag == "SHUFFLE":
            shuffle = element.get("seed")
            shuffle_reference = element.get("ref")
    return GameRecord(
        tuple(events),
        players or (Player(""), Player(""), Player(""), Player("")),
        game_type,
        lobby,
        shuffle,
        shuffle_reference,
    )
