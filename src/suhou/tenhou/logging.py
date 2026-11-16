"""Deterministic logs for a Tenhou replay, independent of the reference viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from suhou.tenhou.replay import ReplayResult, ReplayStep


def replay_json(result: ReplayResult) -> str:
    """Serialize the entire event stream, including post-event table states."""
    return "".join(
        json.dumps(step.snapshot, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
        for step in result.steps
    )


def _format_line(step: ReplayStep) -> str:
    snapshot: dict[str, Any] = step.snapshot
    round_ = snapshot["round"]
    event = step.event
    prefix = f"#{event.index:04d} {event.raw_tag}"
    if event.seat is not None:
        prefix += f" P{event.seat + 1}"
    if round_ is not None:
        prefix += f" [{round_['name']}]"
    suffix = ""
    if round_ is not None and event.tag in ("REACH", "AGARI", "RYUUKYOKU"):
        suffix = (
            f" | 点数 {'/'.join(str(score) for score in round_['scores'])}"
            f" | 本场 {round_['honba']} 供托 {round_['riichi_sticks']}"
        )
    return f"{prefix} {step.message}{suffix}"


def replay_text(result: ReplayResult) -> str:
    return "\n".join(_format_line(step) for step in result.steps) + "\n"


def write_replay(result: ReplayResult, directory: str | Path) -> Path:
    """Write the same replay in JSONL, readable text, and final summary forms."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "replay.jsonl").write_text(replay_json(result), encoding="utf-8", newline="\n")
    (output / "replay.log").write_text(replay_text(result), encoding="utf-8", newline="\n")
    (output / "summary.json").write_text(
        json.dumps(result.summary(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output
