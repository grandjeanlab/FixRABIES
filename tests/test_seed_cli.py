import argparse
from pathlib import Path

import pytest

from fixrabies import cli
from fixrabies.interfaces.validate_inputs import validate_seed_files
from fixrabies.workflows.analyze_wf import create_analyze_workflow


@pytest.fixture
def analyze_parser():
    parser = argparse.ArgumentParser(prog="fixrabies")
    subparsers = parser.add_subparsers(dest="command", required=True)
    cli.add_analyze_parser(subparsers, cli.build_common_parser())
    return parser


@pytest.fixture
def seed_paths(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    relative_seed = work / "relative seed.nii.gz"
    home_seed = home / "home seed.nii.gz"
    absolute_seed = tmp_path / "absolute seed.nii.gz"
    for path in (relative_seed, home_seed, absolute_seed):
        path.write_bytes(b"synthetic seed")
    return {
        "values": [
            "./relative seed.nii.gz",
            "~/home seed.nii.gz",
            str(absolute_seed),
        ],
        "resolved": [
            relative_seed.resolve(),
            home_seed.resolve(),
            absolute_seed.resolve(),
        ],
    }


def test_analyze_seed_paths_resolve(analyze_parser, tmp_path, seed_paths):
    args = analyze_parser.parse_args(
        [
            "analyze",
            "-fix_dir",
            str(tmp_path),
            "-seed",
            *seed_paths["values"],
        ]
    )

    assert args.seed == seed_paths["resolved"]
    assert isinstance(args.seed, list)
    assert all(isinstance(path, Path) and path.is_absolute() for path in args.seed)
    assert args.func is cli.run_analyze_command
    assert validate_seed_files(args.seed) == [
        str(path) for path in seed_paths["resolved"]
    ]


@pytest.mark.parametrize(
    "seed_arguments", [[], ["-seed"]], ids=["default", "empty-list"]
)
def test_analyze_seed_disabled(analyze_parser, tmp_path, seed_arguments):
    args = analyze_parser.parse_args(
        ["analyze", "-fix_dir", str(tmp_path), *seed_arguments]
    )

    assert args.seed == []
    assert isinstance(args.seed, list)
    assert validate_seed_files(args.seed) == []


@pytest.mark.parametrize("value", ["", "   ", "\u0000"])
def test_analyze_seed_rejects_invalid_paths(analyze_parser, tmp_path, capsys, value):
    with pytest.raises(SystemExit) as error:
        analyze_parser.parse_args(
            ["analyze", "-fix_dir", str(tmp_path), "-seed", value]
        )

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "error:" in stderr
    assert "-seed" in stderr
    assert "Traceback" not in stderr


def test_parse_seed_path_expands_and_resolves(seed_paths):
    assert [cli.parse_seed_path(value) for value in seed_paths["values"]] == seed_paths[
        "resolved"
    ]


@pytest.mark.parametrize("value", ["", "   ", "\u0000"])
def test_parse_seed_path_raises_argument_type_error(value):
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_seed_path(value)


def test_analyze_seed_parser_does_not_require_existing_files(analyze_parser, tmp_path):
    missing = tmp_path / "missing.nii.gz"
    args = analyze_parser.parse_args(
        ["analyze", "-fix_dir", str(tmp_path), "-seed", str(missing)]
    )

    assert args.seed == [missing.resolve()]
    with pytest.raises(ValueError, match="Seed image"):
        validate_seed_files(args.seed)


@pytest.mark.parametrize("path_type", [str, Path])
def test_validate_seed_files_expands_and_resolves(seed_paths, path_type):
    result = validate_seed_files([path_type(value) for value in seed_paths["values"]])

    assert result == [str(path) for path in seed_paths["resolved"]]
    assert all(isinstance(path, str) and Path(path).is_absolute() for path in result)


def test_validate_seed_files_accepts_mixed_path_types(seed_paths):
    values = seed_paths["values"]

    assert validate_seed_files([values[0], Path(values[1]), Path(values[2])]) == [
        str(path) for path in seed_paths["resolved"]
    ]


def test_validate_seed_files_empty_list():
    assert validate_seed_files([]) == []


@pytest.mark.parametrize(
    "existing_directory", [False, True], ids=["missing-file", "directory"]
)
def test_validate_seed_files_requires_files(tmp_path, existing_directory):
    seed = tmp_path / "not-a-file.nii.gz"
    if existing_directory:
        seed.mkdir()

    with pytest.raises(ValueError, match="Seed image"):
        validate_seed_files([seed])


@pytest.mark.parametrize("run_dual_regression", [True, False])
@pytest.mark.parametrize("seed_mode", ["default", "empty", "nonempty"])
def test_analyze_workflow_seed_branch_matches_cli_selection(
    analyze_parser, tmp_path, seed_paths, run_dual_regression, seed_mode
):
    arguments = ["analyze", "-fix_dir", str(tmp_path)]
    if seed_mode == "empty":
        arguments.append("-seed")
    elif seed_mode == "nonempty":
        arguments.extend(["-seed", *seed_paths["values"]])
    args = analyze_parser.parse_args(arguments)
    seed_files = validate_seed_files(args.seed)
    scan_dir = tmp_path / "data" / "scan"
    scan_dir.mkdir(parents=True)
    workflow = create_analyze_workflow(
        fix_dir=args.fix_dir,
        result_dir=args.result_dir,
        scan_list=[str(scan_dir)],
        dr_template=None,
        seed_files=seed_files,
        norm=args.norm,
        demean=args.demean,
        dice_threshold=args.dice_threshold,
        extract_threshold=args.extract_threshold,
        run_dual_regression=run_dual_regression,
    )
    seed_enabled = seed_mode == "nonempty"
    seed_nodes = {
        "resampled_seeds",
        "seed_outputs",
        "image_meants",
        "seed_glm",
        "seed_correlation",
    }
    node_names = {node.name for node in workflow._graph.nodes}

    assert node_names & seed_nodes == (seed_nodes if seed_enabled else set())
    assert (
        workflow.get_node("analysis_directories").inputs.seed_analysis is seed_enabled
    )
    assert ("dual_regression" in node_names) is run_dual_regression
    if seed_enabled:
        assert workflow.get_node("resampled_seeds").inputs.seed_files == seed_files
        for source, target in (
            ("resampled_seeds", "seed_outputs"),
            ("resampled_seeds", "image_meants"),
            ("image_meants", "seed_glm"),
            ("seed_glm", "seed_correlation"),
        ):
            assert workflow._graph.has_edge(
                workflow.get_node(source), workflow.get_node(target)
            )
