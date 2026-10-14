#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import logging

# ==============================================================================
# Logging
# ==============================================================================

import logging
from pathlib import Path


def setup_logging(command: str, fix_dir: Path, level: str):

    log_dir = fix_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()

    root.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        "%(asctime)s "
        "[%(levelname)s] "
        "%(name)s: "
        "%(message)s"
    )

    # console output
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    # global log
    global_file = logging.FileHandler(
        log_dir / "fixrabies.log"
    )
    global_file.setFormatter(formatter)
    root.addHandler(global_file)

    # command-specific log
    command_file = logging.FileHandler(
        log_dir / f"{command}.log"
    )
    command_file.setFormatter(formatter)
    root.addHandler(command_file)

# ==============================================================================
# Formatter
# ==============================================================================

class FixFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter,
):
    """Custom argparse formatter."""
    pass


# ==============================================================================
# Lazy command dispatchers
# ==============================================================================

def run_prepare_command(args):
    from fixrabies.prepare import run_prepare
    run_prepare(args)


def run_label_command(args):
    from fixrabies.label import run_label
    run_label(args)


def run_train_command(args):
    from fixrabies.train import run_train
    run_train(args)


def run_classify_command(args):
    from fixrabies.classify import run_classify
    run_classify(args)


def run_review_command(args):
    from fixrabies.review import run_review
    run_report(args)


def run_cleanup_command(args):
    from fixrabies.cleanup import run_cleanup
    run_report(args)


def run_analyze_command(args):
    from fixrabies.analyze import run_analyze
    run_analyze(args)


# ==============================================================================
# Common arguments
# ==============================================================================

def build_common_parser():

    common = argparse.ArgumentParser(add_help=False)

    common.add_argument(
        "-fix_dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""
FixRABIES working directory.

All outputs, models, logs and reports will
be saved below this directory.
""",
    )

    common.add_argument(
        "--log-level",
        default="INFO",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ],
        help="Logging verbosity."
    )

    return common


# ==============================================================================
# Prepare
# ==============================================================================

def add_prepare_parser(subparsers, common):

    parser = subparsers.add_parser(
        "prepare",
        parents=[common],
        formatter_class=FixFormatter,
        help="Prepare RABIES outputs for FIX.",
        description="""
Prepare a dataset for ICA-FIX classification.

This command gathers preprocessing outputs,
extracts metadata, computes QC metrics and
creates a FixRABIES working directory.
""",
    )

    req = parser.add_argument_group("Required arguments")

    req.add_argument(
        "-species",
        required=True,
        choices=[
            "mouseDSURQE",
            "mouseAIBS",
            "ratSIGMA",
        ],
        help="""
Species/template space.

mouseDSURQE
mouseAIBS
ratSIGMA
""",
    )

    req.add_argument(
        "-preprocess_dir",
        required=True,
        type=Path,
        metavar="PATH",
        help="Root preprocessing directory.",
    )

    prep = parser.add_argument_group("Preprocessing options")

    prep.add_argument(
        "-preprocess_software",
        choices=["rabies"],
        default="rabies",
        help="""
Preprocessing software.

Currently only RABIES is supported.
""",
    )

    prep.add_argument(
        "-tr",
        type=float,
        metavar="SECONDS",
        help="""
Repetition time.

If omitted, FixRABIES will attempt to infer
TR from preprocessing outputs.
""",
    )

    patch = parser.add_argument_group(
        "Patch denoising (requires [patch])"
    )

    patch.add_argument(
        "-patch_denoise",
        type=json.loads,
        metavar="JSON",
        default={
            "apply": False,
            "method": "optimal-fro",
            "patch_shape": [11, 11, 11, -1],
            "patch_overlap": [5, 5, 5, -1],
        },
        help="""
Patch denoising configuration.

Example:

-patch_denoise '{
    "apply": true,
    "method": "optimal-fro"
}'

Methods requiring a noise mask 
are not yet implemented. 

CPU execution only. 
""",
    )

    parser.set_defaults(func=run_prepare_command)


# ==============================================================================
# Label
# ==============================================================================

def add_label_parser(subparsers, common):

    parser = subparsers.add_parser(
        "label",
        parents=[common],
        formatter_class=FixFormatter,
        help="Label ICA components.",
    )

    label = parser.add_argument_group("Label selection")

    label.add_argument(
        "-select_label",
        type=float,
        default=20,
        metavar="PERCENT",
        help="""
Percentage of scans to include.

100 = all scans

0 = randomly choose one scan

Default = 20
""",
    )

    parser.set_defaults(func=run_label_command)


# ==============================================================================
# Train
# ==============================================================================

def add_train_parser(subparsers, common):

    parser = subparsers.add_parser(
        "train",
        parents=[common],
        formatter_class=FixFormatter,
        help="Train a FIX model.",
    )

    req = parser.add_argument_group("Required arguments")

    req.add_argument(
        "-fix_model",
        required=True,
        metavar="MODEL",
        help="""
Name of model to create.

Example:

-fix_model mouse_awake_v1
""",
    )

    parser.set_defaults(func=run_train_command)


# ==============================================================================
# Classify
# ==============================================================================

def add_classify_parser(subparsers, common):

    parser = subparsers.add_parser(
        "classify",
        parents=[common],
        formatter_class=FixFormatter,
        help="Apply a FIX classifier.",
    )

    req = parser.add_argument_group("Required arguments")

    req.add_argument(
        "-fix_model",
        required=True,
        metavar="MODEL",
        help="""
FIX model to use.

Examples:

mouseD_awake
ratS_stdrat
my_custom_model
""",
    )

    req.add_argument(
        "-threshold",
        type=float,
        default=20,
        metavar="VALUE",
        help="""
FIX threshold.

Valid range:
0 - 100
""",
    )


# ==============================================================================
# Report
# ==============================================================================

def add_review_parser(subparsers, common):

    parser = subparsers.add_parser(
        "review",
        parents=[common],
        formatter_class=FixFormatter,
        help="Generate QC reports.",
    )

    parser.set_defaults(func=run_review_command)

# ==============================================================================
# Cleanup
# ==============================================================================

def add_cleanup_parser(subparsers, common):

    parser = subparsers.add_parser(
        "cleanup",
        parents=[common],
        formatter_class=FixFormatter,
        help="Apply a FIX denoising.",
    )

    cleanup = parser.add_argument_group("Cleanup arguments")


    cleanup.add_argument(
        "-highpass",
        type=float,
        metavar="SECONDS",
        default=None,
        help="""
High-pass filtering.

omit
    no filtering

-1
    no filtering

0
    detrending only

>0
    high-pass filtering

Value corresponds to
full-width (2*sigma) in seconds.
""",
    )

    cleanup.add_argument(
        "-smooth",
        type=float,
        metavar="MM",
        default=0,
        help="""
Smoothing kernel (mm)
applied after FIX cleanup.
""",
    )

    cleanup.add_argument(
        "-FD_threshold",
        type=float,
        default=0,
        metavar="VALUE",
        help="Framewise displacement threshold.",
    )

    cleanup.add_argument(
        "-DVARS_threshold",
        type=float,
        default=0,
        metavar="VALUE",
        help="DVARS threshold.",
    )

    parser.set_defaults(func=run_cleanup_command)


# ==============================================================================
# Analyze
# ==============================================================================

def add_analyze_parser(subparsers, common):

    parser = subparsers.add_parser(
        "analyze",
        parents=[common],
        formatter_class=FixFormatter,
        help="Run connectivity analyses.",
    )

    dr = parser.add_argument_group(
        "Dual regression"
    )

    dr.add_argument(
        "-DR_template",
        type=Path,
        metavar="/path/to/template.nii.gz",
        help="""
ICA atlas.

If omitted,
a species-specific atlas is used.
""",
    )

    dr.add_argument(
        "--norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply variance normalization.",
    )

    seed = parser.add_argument_group(
        "Seed analysis"
    )

    seed.add_argument(
        "-seed",
        type=json.loads,
        metavar="JSON",
        default={
            "apply": False,
            "seeds": [],
        },
        help="""
Seed based analysis.

Example:

-seed '{
    "apply": true,
    "seeds": [
        "/path/to/acc.nii.gz",
        "/path/to/thalamus.nii.gz"
    ]
}'
""",
    )

    nipype = parser.add_argument_group(
        "Nipype execution"
    )

    nipype.add_argument(
        "-nipype",
        type=json.loads,
        metavar="JSON",
        default={
            "plugin": "MultiProc",
            "n_procs": -1,
            "memory_gb": -1,
        },
        help="""
Nipype execution settings.

Example:

-nipype '{
    "plugin": "MultiProc",
    "n_procs": 4,
    "memory_gb": 16
}'
""",
    )

    parser.set_defaults(func=run_analyze_command)


# ==============================================================================
# Parser builder
# ==============================================================================

def build_parser():

    common = build_common_parser()

    parser = argparse.ArgumentParser(
        prog="fixrabies",
        formatter_class=FixFormatter,
        description="""
FixRABIES, a modern workflow 
to  implement FIX in rodents.
""",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
    )

    add_prepare_parser(subparsers, common)
    add_label_parser(subparsers, common)
    add_train_parser(subparsers, common)
    add_classify_parser(subparsers, common)
    add_review_parser(subparsers, common)
    add_cleanup_parser(subparsers, common)
    add_analyze_parser(subparsers, common)

    return parser


# ==============================================================================
# Main entrypoint
# ==============================================================================

def main():

    parser = build_parser()
    args = parser.parse_args()

    setup_logging(
        command=args.command,
        fix_dir=args.fix_dir,
        level=args.log_level,
    )
    
    args.func(args)


if __name__ == "__main__":
    main()
