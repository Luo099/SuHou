"""Replay Tenhou's recorded actions, maintaining one auditable table state."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from suhou.tenhou.melds import Meld, decode_meld
from suhou.tenhou.models import Event, GameRecord
from suhou.tenhou.parser import parse_tenhou
from suhou.tenhou.tiles import Wind, dora_from_indicator, tile_name
from suhou.tenhou.yaku import yaku_name


class ReplayError(ValueError):
    """The recorded event stream cannot be applied to the current state."""


@dataclass(slots=True)
class PlayerState:
    seat: int = 0
    dealer: int = 0
    initial_hand: tuple[int, ...] = ()
    hand: Counter[int] = field(default_factory=Counter)
    drawn: int | None = None
    discards: list[dict[str, Any]] = field(default_factory=list)
    melds: list[Meld] = field(default_factory=list)
    riichi: str = "none"
    connected: bool = True

    @property
    def seat_wind(self) -> Wind:
        """This seat's wind, relative to the hand's dealer."""
        return Wind.for_seat(self.seat, self.dealer)

    def snapshot(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "seat_wind": self.seat_wind.value,
            "initial_hand": [tile_name(tile) for tile in self.initial_hand],
            "hand": [tile_name(tile) for tile in sorted(self.hand.elements())],
            "drawn": tile_name(self.drawn) if self.drawn is not None else None,
            "discards": copy.deepcopy(self.discards),
            "melds": [
                {
                    "kind": meld.kind,
                    "from_seat": meld.from_seat,
                    "tiles": list(meld.names),
                    "called_tile": tile_name(meld.called_tile),
                    "mask": meld.raw_mask,
                }
                for meld in self.melds
            ],
            "riichi": self.riichi,
            "connected": self.connected,
        }


@dataclass(slots=True)
class RoundState:
    index: int
    seed: tuple[int, int, int, int, int, int]
    dealer: int
    players: list[PlayerState]
    scores: list[int]
    current_seat: int
    dora_indicators: list[int]
    uradora_indicators: list[int] = field(default_factory=list)
    honba: int = 0
    riichi_sticks: int = 0
    phase: str = "draw"
    pending_riichi: int | None = None
    last_discard: tuple[int, int] | None = None
    awaiting_replacement_draw: int | None = None
    normal_draws: int = 0
    replacement_draws: int = 0
    finished: bool = False
    results: list[dict[str, Any]] = field(default_factory=list)
    final_scores: tuple[int, int, int, int] | None = None
    final_points: tuple[float, float, float, float] | None = None

    @property
    def name(self) -> str:
        return f"{self.round_wind.value}{self.seed[0] % 4 + 1}局"

    @property
    def round_wind(self) -> Wind:
        """The hand's round wind, from the seed's 0..15 round index."""
        return Wind.from_index(self.seed[0] // 4)

    @property
    def dice(self) -> tuple[int, int]:
        return self.seed[3] + 1, self.seed[4] + 1

    @property
    def wall_remaining(self) -> int:
        return 70 - self.normal_draws - self.replacement_draws

    def snapshot(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "round_index": self.seed[0],
            "round_wind": self.round_wind.value,
            "dealer": self.dealer,
            "honba": self.honba,
            "riichi_sticks": self.riichi_sticks,
            "dice": list(self.dice),
            "dora_indicators": [tile_name(tile) for tile in self.dora_indicators],
            "dora": [tile_name(dora_from_indicator(tile)) for tile in self.dora_indicators],
            "uradora_indicators": [tile_name(tile) for tile in self.uradora_indicators],
            "scores": self.scores.copy(),
            "wall_remaining": self.wall_remaining,
            "current_seat": self.current_seat,
            "phase": self.phase,
            "last_discard": (
                {"seat": self.last_discard[0], "tile": tile_name(self.last_discard[1])}
                if self.last_discard is not None else None
            ),
            "finished": self.finished,
            "players": [player.snapshot() for player in self.players],
            "results": copy.deepcopy(self.results),
        }


@dataclass(frozen=True, slots=True)
class ReplayStep:
    event: Event
    snapshot: dict[str, Any]
    message: str


@dataclass(slots=True)
class ReplayResult:
    game: GameRecord
    steps: list[ReplayStep]
    rounds: list[RoundState]
    players: tuple[str, ...]
    scores: tuple[int, int, int, int]
    final_scores: tuple[int, int, int, int] | None = None
    final_points: tuple[float, float, float, float] | None = None

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


def _hand_parts(event: Event) -> tuple[tuple[int, ...], ...]:
    if event.tiles is None or len(event.tiles) != 52:
        raise ReplayError("INIT must provide exactly four 13-tile hands")
    return tuple(event.tiles[seat * 13:(seat + 1) * 13] for seat in range(4))


def _remove_tiles(hand: Counter[int], tiles: tuple[int, ...], context: str) -> None:
    missing = Counter(tiles) - hand
    if missing:
        raise ReplayError(f"{context}: tile(s) not present: {list(missing.elements())}")
    hand.subtract(tiles)


def _remove_player_tiles(player: PlayerState, tiles: tuple[int, ...], context: str) -> None:
    """Track physical tiles in hand; 'drawn' only remembers the latest draw."""
    _remove_tiles(player.hand, tiles, context)
    if player.drawn is not None and tiles.count(player.drawn):
        player.drawn = None


def _take_last_discard(round_: RoundState, meld: Meld) -> None:
    if round_.last_discard is None:
        raise ReplayError(f"{meld.kind} requires a preceding opponent discard")
    seat, tile = round_.last_discard
    if seat != meld.from_seat or tile != meld.called_tile:
        raise ReplayError(
            f"Call mask source/tile ({meld.from_seat}/{meld.called_tile}) does not match "
            f"last discard {round_.last_discard}"
        )
    if not round_.players[seat].discards:
        raise ReplayError("Call has no recorded discard to claim")
    if round_.players[seat].discards[-1]["tile"] != tile_name(tile):
        raise ReplayError("Last discard no longer matches the call")
    round_.players[seat].discards[-1]["called_by"] = meld.caller
    round_.last_discard = None


def _start_round(state: _ReplayState, event: Event) -> RoundState:
    assert event.seed is not None and event.scores is not None and event.oya is not None
    hands = _hand_parts(event)
    if state.round is not None and not state.round.finished:
        raise ReplayError("INIT cannot appear before the previous round has finished")
    if state.round is not None and tuple(state.round.scores) != event.scores:
        raise ReplayError(
            f"INIT scores {event.scores} differ from previous result {state.round.scores}"
        )
    if state.final_scores is not None:
        raise ReplayError("Cannot begin a new round after final owari")
    round_ = RoundState(
        index=event.seed[0],
        seed=event.seed,
        dealer=event.oya,
        players=[
            PlayerState(seat=seat, dealer=event.oya, hand=Counter(hand), initial_hand=hand)
            for seat, hand in enumerate(hands)
        ],
        scores=list(event.scores),
        current_seat=event.oya,
        dora_indicators=[event.seed[5]],
        honba=event.seed[1],
        riichi_sticks=event.seed[2],
    )
    state.rounds.append(round_)
    state.round = round_
    return round_


@dataclass(slots=True)
class _ReplayState:
    game: GameRecord
    round: RoundState | None = None
    rounds: list[RoundState] = field(default_factory=list)
    final_scores: tuple[int, int, int, int] | None = None
    final_points: tuple[float, float, float, float] | None = None

    def require_round(self) -> RoundState:
        if self.round is None:
            raise ReplayError("A round event appeared before INIT")
        return self.round


def _apply_draw(round_: RoundState, event: Event) -> str:
    if round_.finished:
        raise ReplayError("Cannot draw after round has finished")
    assert event.seat is not None and event.tile is not None
    player = round_.players[event.seat]
    if round_.current_seat != event.seat:
        raise ReplayError(f"Draw by seat {event.seat}; expected seat {round_.current_seat}")
    if round_.awaiting_replacement_draw not in (None, event.seat):
        raise ReplayError("Kan replacement draw belongs to a different seat")
    if round_.phase == "discard" and player.drawn is None and event.seat == round_.current_seat:
        raise ReplayError("Expected a discard, not another draw")
    if round_.phase == "response" and player.drawn is None:
        round_.phase = "draw"
    if round_.phase != "draw" or player.drawn is not None or round_.pending_riichi is not None:
        raise ReplayError("Draw requires an empty hand and no pending riichi/call")
    player.drawn = event.tile
    player.hand[event.tile] += 1
    replacement = round_.awaiting_replacement_draw == event.seat
    round_.awaiting_replacement_draw = None
    if not replacement:
        round_.normal_draws += 1
        if round_.normal_draws > 70:
            raise ReplayError("More than 70 live-wall draws in one round")
    round_.phase = "discard"
    round_.last_discard = None
    suffix = "（岭上）" if replacement else ""
    return f"摸牌 {tile_name(event.tile)}{suffix}"


def _apply_discard(round_: RoundState, event: Event) -> str:
    assert event.seat is not None and event.tile is not None
    player = round_.players[event.seat]
    if round_.current_seat != event.seat or round_.phase != "discard":
        raise ReplayError(f"Discard by seat {event.seat} is not the current discard turn")
    tsumogiri = event.attributes.get("tsumogiri") == "1" or player.drawn == event.tile
    if player.riichi == "accepted" and not tsumogiri:
        raise ReplayError("A riichi player must discard the drawn tile")
    _remove_player_tiles(player, (event.tile,), "Discard")
    riichi = round_.pending_riichi == event.seat
    player.discards.append({
        "tile": tile_name(event.tile),
        "riichi": riichi,
        "tsumogiri": tsumogiri,
    })
    player.drawn = None
    round_.last_discard = (event.seat, event.tile)
    round_.current_seat = (event.seat + 1) % 4
    round_.phase = "response"
    if riichi:
        player.riichi = "declared"
    return f"切牌 {tile_name(event.tile)}{'（立直宣言）' if riichi else ''}"


def _apply_call(round_: RoundState, event: Event) -> str:
    assert event.seat is not None and event.meld is not None
    if round_.finished or round_.awaiting_replacement_draw is not None:
        raise ReplayError("Call is not valid after the round or before a replacement draw")
    meld = decode_meld(event.seat, event.meld)
    player = round_.players[meld.caller]
    if meld.kind == "kakan":
        if round_.current_seat != meld.caller or round_.phase != "discard" or player.drawn is None:
            raise ReplayError("Kakan requires the caller's drawn-tile turn")
        _remove_player_tiles(player, (meld.called_tile,), "Kakan")
        previous = next(
            (item for item in player.melds
             if item.kind == "pon" and item.from_seat == meld.from_seat
             and item.tiles == meld.tiles),
            None,
        )
        if previous is None:
            raise ReplayError("Kakan has no matching previous pon")
        player.melds[player.melds.index(previous)] = Meld(
            meld.caller, "kakan", meld.from_seat,
            tuple(sorted((*meld.tiles, meld.called_tile))),
            meld.called_tile, (), meld.raw_mask,
        )
        player.drawn = None
        round_.awaiting_replacement_draw = meld.caller
        round_.phase = "draw"
    elif meld.kind == "ankan":
        if round_.current_seat != meld.caller or round_.phase != "discard" or player.drawn is None:
            raise ReplayError("Ankan requires the caller's drawn-tile turn")
        _remove_player_tiles(player, meld.consumed_tiles, "Ankan")
        player.melds.append(meld)
        player.drawn = None
        round_.awaiting_replacement_draw = meld.caller
        round_.phase = "draw"
    elif meld.kind == "nuki":
        if round_.current_seat != meld.caller or round_.phase != "discard" or player.drawn is None:
            raise ReplayError("Nuki requires the caller's drawn-tile turn")
        _remove_player_tiles(player, (meld.called_tile,), "Nuki")
        player.melds.append(meld)
        player.drawn = None
        round_.awaiting_replacement_draw = meld.caller
        round_.phase = "draw"
    else:
        if round_.current_seat == meld.caller and player.drawn is not None:
            raise ReplayError("Cannot chi/pon/minkan on one's own turn")
        _take_last_discard(round_, meld)
        _remove_player_tiles(player, meld.consumed_tiles, meld.kind.capitalize())
        player.melds.append(meld)
        player.drawn = None
        round_.current_seat = meld.caller
        if meld.kind == "minkan":
            round_.awaiting_replacement_draw = meld.caller
            round_.phase = "draw"
        else:
            round_.phase = "discard"
    if meld.kind in ("kakan", "ankan", "minkan", "nuki"):
        round_.replacement_draws += 1
    added = f" + {tile_name(meld.called_tile)}" if meld.kind == "kakan" else ""
    return f"{meld.kind} {' '.join(meld.names)}{added} ← P{meld.from_seat + 1}"


def _apply_riichi(round_: RoundState, event: Event) -> str:
    assert event.seat is not None and event.step is not None
    if round_.finished:
        raise ReplayError("Cannot declare riichi after the round has finished")
    player = round_.players[event.seat]
    if event.step == 1:
        if round_.current_seat != event.seat or round_.phase != "discard" or player.drawn is None:
            raise ReplayError("REACH step 1 requires a draw-then-discard turn")
        if round_.pending_riichi is not None or player.riichi != "none":
            raise ReplayError("Duplicate or conflicting riichi declaration")
        if any(meld.kind != "ankan" for meld in player.melds):
            raise ReplayError("Riichi requires a closed hand")
        round_.pending_riichi = event.seat
        player.riichi = "pending"
        return "立直宣言（尚未扣点）"
    if round_.pending_riichi != event.seat or player.riichi != "declared":
        raise ReplayError("REACH step 2 has no matching declared discard")
    if not player.discards or not player.discards[-1]["riichi"]:
        raise ReplayError("REACH step 2 must follow its declaration discard")
    if event.scores is None:
        raise ReplayError("REACH step 2 requires authoritative ten scores")
    expected = round_.scores.copy()
    expected[event.seat] -= 1000
    if tuple(expected) != event.scores:
        raise ReplayError(f"REACH scores {event.scores} differ from expected {expected}")
    round_.scores = expected
    round_.riichi_sticks += 1
    round_.pending_riichi = None
    player.riichi = "accepted"
    return "立直成立，供托 +1，点数 -1000"


def _apply_result(round_: RoundState, event: Event) -> str:
    assert event.scores_before is not None and event.deltas is not None
    if tuple(round_.scores) != event.scores_before:
        raise ReplayError(
            f"Result pre-scores {event.scores_before} differ from replay {round_.scores}"
        )
    round_.scores = [
        score + delta for score, delta in zip(round_.scores, event.deltas, strict=True)
    ]
    ba = event.attributes.get("ba", "0,0").split(",")
    try:
        round_.honba, recorded_sticks = (int(value) for value in ba)
    except (ValueError, TypeError) as error:
        raise ReplayError(f"Invalid result ba={ba}") from error
    if event.tag == "AGARI":
        assert event.winners is not None
        who, from_who = event.winners[0]
        player = round_.players[who]
        winning_tile = int(event.attributes["machi"]) if "machi" in event.attributes else None
        if who == from_who:
            if player.drawn is None:
                raise ReplayError("Tsumo AGARI requires a current drawn tile")
            if winning_tile is None:
                raise ReplayError("Tsumo AGARI requires machi")
            if player.drawn != winning_tile:
                raise ReplayError("Tsumo machi does not match the current drawn tile")
        else:
            if round_.last_discard is None:
                raise ReplayError("Ron AGARI requires a winning discard")
            discard_seat, discard_tile = round_.last_discard
            if discard_seat != from_who:
                raise ReplayError("AGARI fromWho does not match the winning discard")
            if winning_tile is None:
                winning_tile = discard_tile
            elif winning_tile != discard_tile:
                raise ReplayError("Ron machi does not match the winning discard tile")
        if "hai" in event.attributes:
            recorded_hand = Counter(int(tile) for tile in event.attributes["hai"].split(","))
            if winning_tile not in recorded_hand:
                raise ReplayError("AGARI machi is absent from the winning hand")
            live_hand = Counter(player.hand)
            if who != from_who:
                live_hand[winning_tile] += 1
            if live_hand != recorded_hand:
                raise ReplayError(
                    f"AGARI hand {sorted(recorded_hand.elements())} differs from replay "
                    f"{sorted(live_hand.elements())}"
                )
        recorded_masks = [
            int(mask) for mask in event.attributes.get("m", "").split(",") if mask
        ]
        actual_masks = [meld.raw_mask for meld in player.melds]
        if sorted(recorded_masks) != sorted(actual_masks):
            raise ReplayError(
                f"AGARI meld masks {recorded_masks} differ from replay {actual_masks}"
            )
        kind = "自摸" if who == from_who else "荣和"
        message = f"{kind} P{who + 1} ← P{from_who + 1}"
        result: dict[str, Any] = {
            "type": "agari",
            "who": who,
            "from_who": from_who,
            "deltas": list(event.deltas),
            "riichi_sticks_at_result": recorded_sticks,
            "raw": dict(event.attributes),
        }
        details: list[str] = []
        if "ten" in event.attributes:
            try:
                fu_text, points_text, limit_text = event.attributes["ten"].split(",")
                fu, points, limit = int(fu_text), int(points_text), int(limit_text)
            except (ValueError, TypeError) as error:
                raise ReplayError("AGARI has invalid ten") from error
            yaku_text = event.attributes.get("yaku", "")
            yaku_values = [int(value) for value in yaku_text.split(",") if value]
            if len(yaku_values) % 2:
                raise ReplayError("AGARI yaku must contain id,han pairs")
            yaku = [
                {"id": yaku_values[index], "name": yaku_name(yaku_values[index]),
                 "han": yaku_values[index + 1]}
                for index in range(0, len(yaku_values), 2)
            ]
            han = sum(item["han"] for item in yaku)
            result.update({"fu": fu, "points": points, "limit": limit, "han": han, "yaku": yaku})
            details.append(f"{fu}符{han}飜{points}点")
            if yaku:
                details.append("役:" + " ".join(f"{item['name']}({item['han']})" for item in yaku))
        if "machi" in event.attributes:
            result["winning_tile"] = tile_name(int(event.attributes["machi"]))
            details.append(f"和了牌 {result['winning_tile']}")
        if "hai" in event.attributes:
            hand_ids = sorted(int(tile) for tile in event.attributes["hai"].split(","))
            result["hand"] = [tile_name(tile) for tile in hand_ids]
            details.append("手牌 " + " ".join(result["hand"]))
        if "doraHai" in event.attributes:
            recorded_dora = [int(tile) for tile in event.attributes["doraHai"].split(",")]
            if recorded_dora != round_.dora_indicators:
                raise ReplayError(
                    f"AGARI doraHai {recorded_dora} differs from replay "
                    f"{round_.dora_indicators}"
                )
            result["dora_indicators"] = [tile_name(tile) for tile in recorded_dora]
        if event.attributes.get("doraHaiUra"):
            round_.uradora_indicators = [
                int(tile) for tile in event.attributes["doraHaiUra"].split(",")
            ]
            result["uradora_indicators"] = [
                tile_name(tile) for tile in round_.uradora_indicators
            ]
        round_.riichi_sticks = 0
        if details:
            message += " | " + " | ".join(details)
    else:
        round_.riichi_sticks = recorded_sticks
        message = f"流局 {event.draw_type or '荒牌平局'}"
        result = {
            "type": "ryuukyoku",
            "reason": event.draw_type,
            "deltas": list(event.deltas),
            "riichi_sticks": recorded_sticks,
            "raw": dict(event.attributes),
        }
    round_.results.append(result)
    round_.finished = True
    round_.phase = "result"
    round_.pending_riichi = None
    if event.final_scores is not None and event.final_points is not None:
        expected_scores = list(round_.scores)
        if event.tag == "RYUUKYOKU" and recorded_sticks:
            # At game end, Tenhou awards riichi sticks still on the table to
            # first place. The final <sc> omits this transfer, while <owari>
            # includes it. Ties follow seat order (the log's first dealer is
            # seat 0, so seat order is the tie-break).
            first_place = max(
                range(4), key=lambda seat: (expected_scores[seat], -seat)
            )
            award = 1000 * recorded_sticks
            expected_scores[first_place] += award
            result["terminal_riichi_sticks"] = {
                "seat": first_place,
                "sticks": recorded_sticks,
                "points": award,
            }
            round_.riichi_sticks = 0
            message += f" | 终局供托 {award} 点归 P{first_place + 1}"
        if tuple(expected_scores) != event.final_scores:
            raise ReplayError(
                f"owari scores {event.final_scores} differ from replay "
                f"{tuple(expected_scores)}"
            )
        round_.scores = list(event.final_scores)
        round_.final_scores = event.final_scores
        round_.final_points = event.final_points
    return message


def replay_tenhou(source: str | Path | GameRecord) -> ReplayResult:
    """Consume each source event exactly once and return all post-event states."""
    game = parse_replay(source) if isinstance(source, (str, Path)) else source
    state = _ReplayState(game)
    steps: list[ReplayStep] = []
    for event in game.events:
        try:
            if event.tag == "INIT":
                round_ = _start_round(state, event)
                seats = " ".join(
                    f"P{seat + 1}={player.name}" for seat, player in enumerate(state.game.players)
                )
                message = (
                    f"{round_.name} 开始，庄家 P{round_.dealer + 1}，骰子 {round_.dice}"
                    f" | {seats}"
                )
            elif event.tag in ("T", "D"):
                round_ = state.require_round()
                if event.tag == "T":
                    message = _apply_draw(round_, event)
                else:
                    message = _apply_discard(round_, event)
            elif event.tag == "N":
                round_ = state.require_round()
                message = _apply_call(round_, event)
            elif event.tag == "REACH":
                round_ = state.require_round()
                message = _apply_riichi(round_, event)
            elif event.tag == "DORA":
                round_ = state.require_round()
                if round_.finished:
                    raise ReplayError("DORA cannot appear after the round has finished")
                assert event.tile is not None
                if event.tile in round_.dora_indicators:
                    raise ReplayError("DORA duplicates an existing indicator")
                round_.dora_indicators.append(event.tile)
                message = f"新宝牌指示牌 {tile_name(event.tile)}"
            elif event.tag in ("AGARI", "RYUUKYOKU"):
                round_ = state.require_round()
                message = _apply_result(round_, event)
                if event.final_scores is not None:
                    state.final_scores = event.final_scores
                    state.final_points = event.final_points
            elif event.tag in ("BYE", "RESUME"):
                round_ = state.require_round()
                assert event.seat is not None
                round_.players[event.seat].connected = event.tag == "RESUME"
                message = "断线" if event.tag == "BYE" else "重连"
            else:
                if state.final_scores is not None:
                    raise ReplayError("Metadata cannot follow final owari")
                message = f"牌谱元数据 {event.raw_tag}"
        except (ReplayError, ValueError) as error:
            location = f"event {event.index} <{event.raw_tag}>"
            if state.round is not None:
                location = f"{state.round.name} {location}"
            raise ReplayError(f"{location}: {error}") from error
        steps.append(ReplayStep(event, {
            "event": {
                "index": event.index,
                "tag": event.raw_tag,
                "kind": event.tag,
                "seat": event.seat,
                "tile": event.tile_name,
                "attributes": dict(event.attributes),
            },
            "message": message,
            "round": state.round.snapshot() if state.round else None,
            "final_scores": list(state.final_scores) if state.final_scores else None,
            "final_points": list(state.final_points) if state.final_points else None,
        }, message))
    if state.round is None:
        raise ReplayError("The game contains no INIT round")
    if state.final_scores is None or state.final_points is None:
        raise ReplayError("The game has no final owari result")
    if not state.round.finished:
        raise ReplayError("The final round has no result")
    return ReplayResult(
        game=game,
        steps=steps,
        rounds=state.rounds,
        players=tuple(player.name for player in game.players),
        scores=tuple(state.round.scores),
        final_scores=state.final_scores,
        final_points=state.final_points,
    )
