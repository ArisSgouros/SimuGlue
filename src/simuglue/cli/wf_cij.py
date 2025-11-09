# src/simuglue/cli/wf_cij.py
from __future__ import annotations

import argparse
from typing import Sequence

from simuglue.workflow.cij.run import (
    init_cij,
    run_cij,
    parse_cij,
)
from simuglue.workflow.cij.post import post_cij


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Cij workflow: prepare, run, parse and post-process elastic constant "
            "calculations."
        ),
    )
    sub = p.add_subparsers(dest="action", required=True)

    # Common -c/--config option factory
    def add_config_arg(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "-c",
            "--config",
            default="cij.yaml",
            help="Path to Cij workflow YAML config (default: cij.yaml).",
        )

    # sgl wf cij init -c cij.yaml
    p_init = sub.add_parser(
        "init",
        help=(
            "Initialize Cij workflow: create case directories, "
            "generate deformed structures and input files."
        ),
    )
    add_config_arg(p_init)

    # sgl wf cij run -c cij.yaml
    p_run = sub.add_parser(
        "run",
        help=(
            "Run all prepared Cij cases whose jobs are not yet marked done. "
            "Assumes 'init' has been executed."
        ),
    )
    add_config_arg(p_run)

    # sgl wf cij parse -c cij.yaml
    p_parse = sub.add_parser(
        "parse",
        help=(
            "Parse finished Cij cases (with .done present) and write per-case "
            "result.json files."
        ),
    )
    add_config_arg(p_parse)

    # sgl wf cij post -c cij.yaml -o cij.json
    p_post = sub.add_parser(
        "post",
        help=(
            "Post-process result.json files to assemble the Cij tensor. "
            "Backend-agnostic."
        ),
    )
    add_config_arg(p_post)
    p_post.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON file name (overrides output.cij_json in config).",
    )

    # sgl wf cij all -c cij.yaml -o cij.json
    p_all = sub.add_parser(
        "all",
        help=(
            "Run full Cij workflow locally: init → run → parse → post. "
            "Mainly for small/test systems."
        ),
    )
    add_config_arg(p_all)
    p_all.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON file name for final Cij "
             "(overrides output.cij_json in config).",
    )

    return p


def main(
    argv: Sequence[str] | None = None,
    prog: str | None = None,
) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(list(argv) if argv is not None else None)

    action = args.action

    if action == "init":
        init_cij(args.config)
        return 0

    if action == "run":
        run_cij(args.config)
        return 0

    if action == "parse":
        parse_cij(args.config)
        return 0

    if action == "post":
        post_cij(args.config, outfile=args.output)
        return 0

    if action == "all":
        init_cij(args.config)
        run_cij(args.config)
        parse_cij(args.config)
        post_cij(args.config, outfile=args.output)
        return 0

    # Should be unreachable because subparsers are required.
    parser.error(f"Unknown action: {action!r}")
    return 2

