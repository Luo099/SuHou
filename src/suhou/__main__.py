"""Command-line entry point for SuHou."""

from __future__ import annotations

import argparse
from pathlib import Path

from suhou import __version__
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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
