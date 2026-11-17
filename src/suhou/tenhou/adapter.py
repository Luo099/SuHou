"""Tenhou protocol adapter and the XML-to-internal-event replay driver.

All tag names, raw attributes, and N.m call decoding stop here. The adapter
reads the current read-only round view for context fields (tsumogiri,
replacement draw, riichi declaration discard, and the pon extended by a kakan)
and emits immutable :class:`GameEvent` records.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from suhou.domain.events import (
    AgariPayload,
    CallPayload,
    ConnectionPayload,
    DiscardPayload,
    DoraPayload,
    DrawPayload,
    EventType,
    GameEvent,
    MetadataEntry,
    MetadataPayload,
    RiichiAcceptedPayload,
    RiichiDeclarationPayload,
    RoundStartPayload,
    RyuukyokuPayload,
    Yaku,
)
from suhou.domain.state import RoundView
from suhou.domain.tiles import Wind
from suhou.domain.types import (
    MeldKind,
    match_type_from_game_type,
    red_fives_from_game_type,
)
from suhou.replay.engine import ReplayEngine, ReplayError, ReplayFrame, ReplayOutcome
from suhou.tenhou.melds import decode_meld
from suhou.tenhou.models import GameRecord, TenhouEvent, players_from_attributes
from suhou.tenhou.parser import parse_tenhou
from suhou.tenhou.yaku import yaku_name

ROUND_TAGS = frozenset(
    {"T", "D", "N", "REACH", "DORA", "AGARI", "RYUUKYOKU", "BYE", "RESUME"}
)
_METADATA_TAGS = frozenset({"GO", "UN", "TAIKYOKU", "SHUFFLE"})


@dataclass(frozen=True, slots=True)
class ResultGroup:
    """Settlement grouping of consecutive result records."""

    group_id: int
    final: bool


@dataclass(frozen=True, slots=True)
class AdaptContext:
    """Read-only context the adapter uses for one source event."""

    round_id: int | None
    seq: int | None
    round: RoundView | None
    result_group: ResultGroup | None = None


BeforeSourceEvent = Callable[[TenhouEvent, RoundView | None], None]


class TenhouAdapter:
    """Translate one source record without mutating any state."""

    def adapt(self, source: TenhouEvent, context: AdaptContext) -> GameEvent:
        tag = source.tag
        if tag in _METADATA_TAGS:
            return self._metadata(source)
        if tag == "INIT":
            return self._round_start(source, context)
        if tag == "T":
            return self._draw(source, context)
        if tag == "D":
            return self._discard(source, context)
        if tag == "N":
            return self._call(source, context)
        if tag == "REACH":
            return self._riichi(source, context)
        if tag == "DORA":
            return self._dora(source, context)
        if tag in ("AGARI", "RYUUKYOKU"):
            return self._result(source, context)
        if tag in ("BYE", "RESUME"):
            return self._connection(source, context)
        raise ValueError(f"Unsupported Tenhou tag <{tag}>")

    # -- helpers ----------------------------------------------------------

    def _round_context(self, source: TenhouEvent, context: AdaptContext) -> tuple[int, int]:
        if context.round_id is None or context.seq is None or context.round is None:
            raise ValueError(f"<{source.tag}> appears before INIT")
        return context.round_id, context.seq

    def _event(
        self,
        source: TenhouEvent,
        context: AdaptContext,
        *,
        kind: EventType,
        payload: object,
        actor: int | None,
    ) -> GameEvent:
        round_id, seq = self._round_context(source, context)
        return GameEvent(
            kind=kind,
            source_index=source.index,
            round_id=round_id,
            seq=seq,
            actor=actor,
            payload=payload,  # type: ignore[arg-type]
        )

    def _metadata(self, source: TenhouEvent) -> GameEvent:
        tag = source.tag
        attributes = source.attributes
        if tag == "GO":
            game_type = int(attributes.get("type", "0"))
            lobby = int(attributes.get("lobby", "0"))
            try:
                match_type = match_type_from_game_type(game_type)
            except ValueError:
                # Keep three-player logs replayable; observation rejects them.
                match_type = None
            entry = MetadataEntry.GAME_SETTINGS
            data = {
                "game_type": game_type,
                "lobby": lobby,
                "match_type": match_type,
                "red_fives": red_fives_from_game_type(game_type),
            }
        elif tag == "UN":
            complete = all(f"n{seat}" in attributes for seat in range(4))
            entry = MetadataEntry.ROSTER
            data = {
                "players": players_from_attributes(attributes) if complete else (),
                "complete": complete,
            }
        elif tag == "TAIKYOKU":
            entry = MetadataEntry.GAME_START
            oya = attributes.get("oya")
            data = {"dealer": int(oya) if oya is not None else None}
        else:
            entry = MetadataEntry.SHUFFLE
            data = {
                "seed": attributes.get("seed"),
                "reference": attributes.get("ref"),
            }
        return GameEvent(
            kind=EventType.METADATA,
            source_index=source.index,
            round_id=None,
            seq=None,
            actor=None,
            payload=MetadataPayload(entry, tag, data),
        )

    # -- round events -----------------------------------------------------

    def _round_start(self, source: TenhouEvent, context: AdaptContext) -> GameEvent:
        if source.seed is None or source.scores is None or source.oya is None:
            raise ValueError("INIT is missing seed, scores, or oya")
        if source.tiles is None or len(source.tiles) != 52:
            raise ValueError("INIT must provide exactly four 13-tile hands")
        round_id = context.round_id
        if round_id is None:
            raise ValueError("<INIT> requires a round id")
        seed = source.seed
        payload = RoundStartPayload(
            round_index=seed[0],
            round_wind=Wind.from_index(seed[0] // 4),
            hand_number=seed[0] % 4 + 1,
            honba=seed[1],
            riichi_sticks=seed[2],
            dealer=source.oya,
            dice=(seed[3] + 1, seed[4] + 1),
            scores=source.scores,
            hands=tuple(
                source.tiles[seat * 13:(seat + 1) * 13] for seat in range(4)
            ),
            dora_indicator=seed[5],
        )
        return GameEvent(
            kind=EventType.ROUND_START,
            source_index=source.index,
            round_id=round_id,
            seq=0,
            actor=None,
            payload=payload,
        )

    def _draw(self, source: TenhouEvent, context: AdaptContext) -> GameEvent:
        assert source.seat is not None and source.tile is not None
        assert context.round is not None
        replacement = context.round.awaiting_replacement_draw == source.seat
        return self._event(
            source,
            context,
            kind=EventType.DRAW,
            payload=DrawPayload(source.tile, replacement),
            actor=source.seat,
        )

    def _discard(self, source: TenhouEvent, context: AdaptContext) -> GameEvent:
        assert source.seat is not None and source.tile is not None
        assert context.round is not None
        player = context.round.players[source.seat]
        tsumogiri = source.attributes.get("tsumogiri") == "1" or player.drawn == source.tile
        riichi = context.round.pending_riichi == source.seat
        return self._event(
            source,
            context,
            kind=EventType.DISCARD,
            payload=DiscardPayload(source.tile, tsumogiri, riichi),
            actor=source.seat,
        )

    def _call(self, source: TenhouEvent, context: AdaptContext) -> GameEvent:
        assert source.seat is not None
        assert context.round is not None
        if source.meld_record is None:
            raise ValueError("<N> lacks a decoded call record")
        meld = source.meld_record
        if meld.kind is MeldKind.KAKAN:
            previous = next(
                (
                    item
                    for item in context.round.players[meld.caller].melds
                    if item.kind is MeldKind.PON
                    and item.from_seat == meld.from_seat
                    and item.tiles == meld.tiles
                ),
                None,
            )
            payload = CallPayload(
                kind=meld.kind,
                from_seat=meld.from_seat,
                tiles=meld.tiles,
                called_tile=previous.called_tile if previous is not None else meld.called_tile,
                consumed_tiles=(),
                added_tile=meld.called_tile,
                target_seq=previous.formed_seq if previous is not None else None,
                source_mask=meld.raw_mask,
            )
        elif meld.kind is MeldKind.ANKAN:
            payload = CallPayload(
                kind=meld.kind,
                from_seat=None,
                tiles=meld.tiles,
                called_tile=meld.called_tile,
                consumed_tiles=meld.consumed_tiles,
                added_tile=None,
                target_seq=None,
                source_mask=meld.raw_mask,
            )
        elif meld.kind is MeldKind.NUKI:
            payload = CallPayload(
                kind=meld.kind,
                from_seat=None,
                tiles=meld.tiles,
                called_tile=None,
                consumed_tiles=(),
                added_tile=None,
                target_seq=None,
                source_mask=meld.raw_mask,
            )
        else:
            payload = CallPayload(
                kind=meld.kind,
                from_seat=meld.from_seat,
                tiles=meld.tiles,
                called_tile=meld.called_tile,
                consumed_tiles=meld.consumed_tiles,
                added_tile=None,
                target_seq=None,
                source_mask=meld.raw_mask,
            )
        return self._event(
            source,
            context,
            kind=EventType.CALL,
            payload=payload,
            actor=meld.caller,
        )

    def _riichi(self, source: TenhouEvent, context: AdaptContext) -> GameEvent:
        assert source.seat is not None
        if source.step == 1:
            return self._event(
                source,
                context,
                kind=EventType.RIICHI_DECLARATION,
                payload=RiichiDeclarationPayload(),
                actor=source.seat,
            )
        scores = (
            tuple(source.scores)
            if source.scores is not None and len(source.scores) == 4
            else None
        )
        return self._event(
            source,
            context,
            kind=EventType.RIICHI_ACCEPTED,
            payload=RiichiAcceptedPayload(scores),
            actor=source.seat,
        )

    def _dora(self, source: TenhouEvent, context: AdaptContext) -> GameEvent:
        assert source.tile is not None
        return self._event(
            source,
            context,
            kind=EventType.DORA,
            payload=DoraPayload(source.tile),
            actor=None,
        )

    def _connection(self, source: TenhouEvent, context: AdaptContext) -> GameEvent:
        assert source.seat is not None
        return self._event(
            source,
            context,
            kind=EventType.CONNECTION,
            payload=ConnectionPayload(source.tag == "RESUME"),
            actor=source.seat,
        )

    def _result(self, source: TenhouEvent, context: AdaptContext) -> GameEvent:
        if context.result_group is None:
            raise ValueError("Result event has no settlement group")
        if source.scores_before is None or source.deltas is None:
            raise ValueError("Result event is missing score pairs")
        honba_after, riichi_sticks = self._result_counts(source)
        if source.tag == "AGARI":
            return self._agari(source, context, honba_after, riichi_sticks)
        return self._event(
            source,
            context,
            kind=EventType.RESULT,
            payload=RyuukyokuPayload(
                group_id=context.result_group.group_id,
                group_final=context.result_group.final,
                reason=source.draw_type,
                deltas=source.deltas,
                scores_before=source.scores_before,
                honba_after=honba_after,
                riichi_sticks=riichi_sticks,
                final_scores=source.final_scores,
                final_points=source.final_points,
            ),
            actor=None,
        )

    @staticmethod
    def _result_counts(source: TenhouEvent) -> tuple[int, int]:
        ba = source.attributes.get("ba", "0,0").split(",")
        if len(ba) != 2:
            raise ValueError(f"Invalid result ba={ba}")
        try:
            return int(ba[0]), int(ba[1])
        except ValueError as error:
            raise ValueError(f"Invalid result ba={ba}") from error

    def _agari(
        self,
        source: TenhouEvent,
        context: AdaptContext,
        honba_after: int,
        riichi_sticks: int,
    ) -> GameEvent:
        assert source.winners is not None
        who, from_who = source.winners[0]
        attributes = source.attributes
        fu = han = points = limit = None
        yaku: tuple[Yaku, ...] = ()
        if "ten" in attributes:
            try:
                fu_text, points_text, limit_text = attributes["ten"].split(",")
                fu, points, limit = int(fu_text), int(points_text), int(limit_text)
            except (ValueError, TypeError) as error:
                raise ValueError("AGARI has invalid ten") from error
            yaku_text = attributes.get("yaku", "")
            yaku_values = [int(value) for value in yaku_text.split(",") if value]
            if len(yaku_values) % 2:
                raise ValueError("AGARI yaku must contain id,han pairs")
            yaku = tuple(
                Yaku(yaku_values[index], yaku_name(yaku_values[index]), yaku_values[index + 1])
                for index in range(0, len(yaku_values), 2)
            )
            han = sum(item.han for item in yaku)
        winning_tile = int(attributes["machi"]) if "machi" in attributes else None
        hand = (
            tuple(int(tile) for tile in attributes["hai"].split(","))
            if "hai" in attributes
            else None
        )
        melds = tuple(
            decode_meld(who, int(mask))
            for mask in attributes.get("m", "").split(",")
            if mask
        )
        dora_indicators = (
            tuple(int(tile) for tile in attributes["doraHai"].split(","))
            if "doraHai" in attributes
            else None
        )
        uradora_indicators = (
            tuple(int(tile) for tile in attributes["doraHaiUra"].split(","))
            if attributes.get("doraHaiUra")
            else None
        )
        assert context.result_group is not None
        return self._event(
            source,
            context,
            kind=EventType.RESULT,
            payload=AgariPayload(
                group_id=context.result_group.group_id,
                group_final=context.result_group.final,
                who=who,
                from_who=from_who,
                deltas=source.deltas,  # type: ignore[arg-type]
                scores_before=source.scores_before,  # type: ignore[arg-type]
                winning_tile=winning_tile,
                hand=hand,
                melds=melds,
                fu=fu,
                han=han,
                points=points,
                limit=limit,
                yaku=yaku,
                dora_indicators=dora_indicators,
                uradora_indicators=uradora_indicators,
                honba_after=honba_after,
                riichi_sticks_at_result=riichi_sticks,
                final_scores=source.final_scores,
                final_points=source.final_points,
            ),
            actor=None,
        )


def replay_frames(
    source: str | Path | GameRecord,
    *,
    before_event: BeforeSourceEvent | None = None,
) -> tuple[tuple[ReplayFrame, ...], ReplayOutcome]:
    """Adapt and replay every source event once, returning frames and outcome."""
    game = parse_tenhou(source) if isinstance(source, (str, Path)) else source
    adapter = TenhouAdapter()
    engine = ReplayEngine()
    events = game.events
    frames: list[ReplayFrame] = []
    round_id = -1
    start_index: int | None = None
    group_counter = -1
    previous_tag: str | None = None
    for position, source_event in enumerate(events):
        tag = source_event.tag
        if tag == "INIT":
            round_id += 1
            start_index = source_event.index
        if tag in ROUND_TAGS and start_index is not None:
            seq: int | None = source_event.index - start_index
        else:
            seq = None
        result_group: ResultGroup | None = None
        if tag == "AGARI":
            if previous_tag != "AGARI":
                group_counter += 1
            final = (
                position + 1 >= len(events) or events[position + 1].tag != "AGARI"
            )
            result_group = ResultGroup(group_counter, final)
        elif tag == "RYUUKYOKU":
            group_counter += 1
            result_group = ResultGroup(group_counter, True)
        before = engine.round_view()
        if before_event is not None:
            before_event(source_event, before)
        context = AdaptContext(
            round_id=(
                round_id
                if tag == "INIT" or (tag in ROUND_TAGS and round_id >= 0)
                else None
            ),
            seq=seq,
            round=before,
            result_group=result_group,
        )
        try:
            event = adapter.adapt(source_event, context)
            transition = engine.apply(event)
        except (ReplayError, ValueError) as error:
            location = f"event {source_event.index} <{source_event.raw_tag}>"
            if before is not None:
                location = f"{before.name} {location}"
            raise ReplayError(f"{location}: {error}") from error
        frames.append(
            ReplayFrame(
                event=transition.event,
                before=transition.before,
                after=transition.after,
                message=transition.message,
                final_scores=transition.final_scores,
                final_points=transition.final_points,
                source=source_event,
            )
        )
        previous_tag = tag
    outcome = engine.finish(tuple(frames))
    return tuple(frames), outcome


def iter_replay(source: str | Path | GameRecord) -> Iterator[ReplayFrame]:
    """Replay XML and return immutable frames in source order."""
    frames, _ = replay_frames(source)
    return iter(frames)


__all__ = [
    "ROUND_TAGS",
    "AdaptContext",
    "ResultGroup",
    "TenhouAdapter",
    "iter_replay",
    "replay_frames",
]
