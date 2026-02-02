# src/simuglue/cli/wf_deform.py
from __future__ import annotations

import argparse
from typing import Sequence

# Import from your new deform workflow package
from simuglue.workflow.deform.workflow import (
    init_deformation,
    run_deformation,
)
from simuglue.workflow.deform.post_deformation import post_deformation


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Deformation workflow: prepare, run, and post-process "
            "stress-strain curve calculations."
        ),
    )
    sub = p.add_subparsers(dest="action", required=True)

    def add_config_arg(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "-c",
            "--config",
            default="config.yaml",
            help="Path to workflow YAML config (default: config.yaml).",
        )

    # sgl wf deform init
    p_init = sub.add_parser("init", help="Initialize deformation cases.")
    add_config_arg(p_init)

    # sgl wf deform run
    p_run = sub.add_parser("run", help="Run prepared cases with failure detection.")
    add_config_arg(p_run)

    # sgl wf deform post
    p_post = sub.add_parser("post", help="Assemble stress-strain curve JSON.")
    add_config_arg(p_post)
    p_post.add_argument("-o", "--output", default="stress_strain_curve.json")

    # sgl wf deform all
    p_all = sub.add_parser("all", help="Run full workflow: init -> run -> post.")
    add_config_arg(p_all)
    p_all.add_argument("-o", "--output", default="stress_strain_curve.json")

    return p


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(list(argv) if argv is not None else None)

    action = args.action

    if action == "init":
        init_deformation(args.config)
        return 0

    if action == "run":
        run_deformation(args.config)
        return 0

    if action == "post":
        post_deformation(args.config, outfile=args.output)
        return 0

    if action == "all":
        init_deformation(args.config)
        run_deformation(args.config)
        post_deformation(args.config, outfile=args.output)
        return 0

    return 2
