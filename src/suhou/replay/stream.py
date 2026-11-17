"""Read and write the versioned internal event stream (``events.jsonl``)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from suhou.domain.codec import decode_event, event_json
from suhou.domain.events import GameEvent


class EventStreamError(ValueError):
    """A serialized event stream cannot be decoded."""


def write_event_stream(events: Iterable[GameEvent], path: str | Path) -> Path:
    """Write one event object per line, in replay order."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(event_json(event) + "\n" for event in events),
        encoding="utf-8",
        newline="\n",
    )
    return target


def read_event_stream(path: str | Path) -> tuple[GameEvent, ...]:
    """Read ``events.jsonl`` and reject malformed or version-mismatched lines."""
    source = Path(path)
    events: list[GameEvent] = []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EventStreamError(f"Cannot read event stream {source}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as error:
            raise EventStreamError(
                f"{source}: line {line_number}: invalid JSON: {error}"
            ) from error
        if not isinstance(data, dict):
            raise EventStreamError(f"{source}: line {line_number}: event must be an object")
        try:
            events.append(decode_event(data))
        except (ValueError, TypeError, KeyError) as error:
            raise EventStreamError(
                f"{source}: line {line_number}: invalid event: {error}"
            ) from error
    if not events:
        raise EventStreamError(f"{source}: event stream is empty")
    return tuple(events)
