"""LoreWeaver command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loreweaver import __version__
from loreweaver.config import load_config
from loreweaver.logging import configure_logging, new_run_id


PIPELINE_COMMANDS = (
    "ingest",
    "windows",
    "extract",
    "index",
    "graph",
    "retrieve",
    "ask",
    "eval",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loreweaver",
        description="LoreWeaver M1 command line interface.",
    )
    parser.add_argument("--version", action="version", version=f"LoreWeaver {__version__}")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the LoreWeaver config file.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser("status", help="Show project bootstrap status.")
    status_parser.set_defaults(func=_status)

    for command in PIPELINE_COMMANDS:
        command_parser = subparsers.add_parser(
            command,
            help=f"M1 command placeholder: {command}.",
        )
        command_parser.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="Arguments reserved for later M1 stages.",
        )
        command_parser.set_defaults(func=_placeholder)

    return parser


def _status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run_id = new_run_id("status")
    sample_path = config.sample_source_path

    print(f"run_id: {run_id}")
    print(f"version: {__version__}")
    print(f"config: {config.path}")
    print(f"stage: {config.values.get('project', {}).get('stage', 'unknown')}")
    print(f"data_dir: {config.data_dir}")

    if sample_path is None:
        print("sample: not configured")
    else:
        status = "found" if sample_path.exists() else "missing"
        size = sample_path.stat().st_size if sample_path.exists() else 0
        print(f"sample: {sample_path} ({status}, {size} bytes)")

    required_dirs = [
        config.data_dir / "raw",
        config.data_dir / "normalized",
        config.data_dir / "runs",
        config.data_dir / "indexes",
        config.data_dir / "eval",
    ]
    missing_dirs = [str(path) for path in required_dirs if not Path(path).exists()]
    if missing_dirs:
        print("missing_dirs:")
        for path in missing_dirs:
            print(f"  - {path}")
        return 1

    print("bootstrap: ok")
    return 0


def _placeholder(args: argparse.Namespace) -> int:
    run_id = new_run_id(args.command)
    print(f"run_id: {run_id}")
    print(f"command: {args.command}")
    print("status: placeholder")
    print("message: This command surface is ready; implementation begins in later M1 substages.")
    if args.args:
        print("received_args:")
        for item in args.args:
            print(f"  - {item}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = sys.argv[1:] if argv is None else argv
    args, unknown_args = parser.parse_known_args(raw_argv)
    configure_logging(args.verbose)

    if not hasattr(args, "func"):
        if unknown_args:
            parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
        parser.print_help()
        return 0

    if unknown_args and hasattr(args, "args"):
        command_index = raw_argv.index(args.command)
        args.args = raw_argv[command_index + 1 :]
    elif unknown_args:
        parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
