"""Four-seat decision-sample iteration and atomic JSONL export."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from suhou.domain.events import EventType
from suhou.observation.history import EventHistory
from suhou.rules.engine import PlayerRuleEngine
from suhou.rules.extract import apply_decision_frame
from suhou.samples.align import Aligner
from suhou.samples.model import DecisionSample
from suhou.tenhou.adapter import iter_replay
from suhou.tenhou.models import GameRecord


def _log_id(source: str | Path | GameRecord) -> str:
    if isinstance(source, GameRecord):
        return ""
    name = Path(source).name.removesuffix(".gz")
    return Path(name).stem


def iter_decision_samples(
    source: str | Path | GameRecord, viewer: int | None = None
) -> Iterator[DecisionSample]:
    """Replay ``source`` once and yield labelled/unlabelled decision samples.

    ``viewer`` only filters the output rows: all four engines always run so a
    single-seat export is byte-identical to filtering the full export.
    """
    if viewer is not None and not 0 <= viewer < 4:
        raise ValueError(f"viewer must be in 0..3 or None, got {viewer}")
    frames = tuple(iter_replay(source))
    log_id = _log_id(source)
    engines = [PlayerRuleEngine(seat) for seat in range(4)]
    histories = [EventHistory() for _ in range(4)]
    aligner = Aligner(log_id)
    for frame in frames:
        event = frame.event
        if event.kind is EventType.METADATA:
            continue
        if event.kind is EventType.ROUND_START:
            assert frame.after is not None
            aligner.close_round_defensive()
            for seat in range(4):
                histories[seat] = EventHistory(red_fives=frame.after.red_fives)
                engines[seat].update(event, frame.after)
                histories[seat].append(event)
            continue
        assert frame.after is not None
        source_tag = frame.source.tag if frame.source is not None else ""
        aligner.settle(event, frame.after.red_fives, source_tag=source_tag)
        for seat in range(4):
            point = apply_decision_frame(engines[seat], histories[seat], frame)
            if point is not None:
                aligner.open_window(point, trigger_actor=event.actor)
    aligner.finish()
    samples = sorted(
        aligner.samples,
        key=lambda s: (s.audit.round_ordinal, s.audit.trigger.seq, s.audit.viewer),
    )
    return iter([s for s in samples if viewer is None or s.audit.viewer == viewer])


def export_samples(
    source: str | Path | GameRecord, output: str | Path, viewer: int | None = None
) -> dict:
    """Write samples atomically: temp file, full validation, os.replace.

    Alignment errors raise before any file is touched; a failed run never
    modifies an existing target file.
    """
    target = Path(output)
    samples = list(iter_decision_samples(source, viewer))
    lines = [
        json.dumps(s.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True)
        for s in samples
    ]
    for line, sample in zip(lines, samples, strict=True):
        data = json.loads(line)
        assert set(data) == {
            "schema_version",
            "observation",
            "legal_actions",
            "target_index",
            "audit",
        }
        if sample.target_index is not None:
            assert 0 <= sample.target_index < len(sample.legal_actions)
        else:
            assert sample.audit.unlabelled_reason
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return _stats(samples)


def _stats(samples: list[DecisionSample]) -> dict:
    by_type = Counter(s.audit.window_type for s in samples)
    labelled_by_type = Counter(
        s.audit.window_type for s in samples if s.target_index is not None
    )
    labelled = sum(labelled_by_type.values())
    return {
        "total": len(samples),
        "labelled": labelled,
        "unlabelled": len(samples) - labelled,
        "by_type": dict(by_type),
        "labelled_by_type": dict(labelled_by_type),
    }


__all__ = ["export_samples", "iter_decision_samples"]