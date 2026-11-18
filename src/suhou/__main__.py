"""Command-line entry point for SuHou."""

from __future__ import annotations

import argparse
from pathlib import Path

from suhou import __version__
from suhou.observation.builder import ObservationError
from suhou.samples import SampleAlignmentError, export_samples
from suhou.tenhou.logging import replay_text, write_replay
from suhou.tenhou.parser import TenhouParseError
from suhou.tenhou.replay import ReplayError, replay_tenhou


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="suhou", description="SuHou riichi mahjong tools")
    subcommands = parser.add_subparsers(dest="command")
    replay = subcommands.add_parser("tenhou-replay", help="Parse and replay a Tenhou mjlog XML")
    replay.add_argument("input", type=Path, help="Path to plain or gzip-compressed mjlog XML")
    replay.add_argument("--output", "-o", type=Path, required=True, help="Replay log directory")
    replay.add_argument(
        "--step", type=int, default=None, metavar="N",
        help="Print only the first N events (zero-based, inclusive); output remains complete",
    )
    export = subcommands.add_parser("export-samples", help="Export decision samples as JSONL")
    export.add_argument("input", type=Path, help="Path to plain or gzip-compressed mjlog XML")
    export.add_argument("--output", "-o", type=Path, required=True, help="Samples JSONL file")
    export.add_argument(
        "--viewer", type=int, choices=range(4), default=None,
        help="Export only this seat (0..3); default all four",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command is None:
        print(f"SuHou {__version__}")
        return 0
    if args.command == "tenhou-replay":
        if args.step is not None and args.step < 0:
            _parser().error("--step must be non-negative")
        try:
            result = replay_tenhou(args.input)
            output = write_replay(result, args.output)
        except (TenhouParseError, ReplayError, OSError) as error:
            print(f"tenhou-replay: {error}")
            return 2
        lines = replay_text(result).splitlines()
        end = len(lines) if args.step is None else min(args.step + 1, len(lines))
        for line in lines[:end]:
            print(line)
        print(f"已重放 {len(result.steps)} 个事件，{len(result.rounds)} 局。日志：{output}")
        return 0
    if args.command == "export-samples":
        try:
            stats = export_samples(args.input, args.output, viewer=args.viewer)
        except (
            TenhouParseError,
            ReplayError,
            SampleAlignmentError,
            ObservationError,
            OSError,
        ) as error:
            print(f"export-samples: {error}")
            return 2
        kinds = " ".join(f"{k}={v}" for k, v in sorted(stats["by_type"].items()))
        print(
            f"窗口 {stats['total']}（{kinds}）"
            f"已标注 {stats['labelled']} 未标注 {stats['unlabelled']}"
        )
        print(f"样本：{args.output}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
