import pytest

from fixrabies.cli import build_parser

COMMANDS = [
    "prepare",
    "label",
    "train",
    "classify",
    "review",
    "retrain",
    "cleanup",
    "analyze",
    "analyze_group",
    "import",
]


@pytest.mark.parametrize("command", [None, *COMMANDS])
def test_help_exits_without_running_workflows(command, capsys):
    arguments = ["-h"] if command is None else [command, "-h"]

    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(arguments)

    assert error.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("command", COMMANDS)
def test_commands_require_fix_directory(command):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args([command])

    assert error.value.code == 2


def test_analyze_expands_working_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    args = build_parser().parse_args(["analyze", "-fix_dir", "~/working"])

    assert args.fix_dir == tmp_path / "working"
    assert args.nipype == {"plugin": "Linear"}
    assert args.seed == []
    assert args.norm is True
    assert args.demean is True


@pytest.mark.parametrize("value, expected", [("True", True), ("false", False)])
def test_analyze_boolean_options(value, expected, tmp_path):
    args = build_parser().parse_args(
        ["analyze", "-fix_dir", str(tmp_path), "-norm", value, "-demean", value]
    )

    assert args.norm is expected
    assert args.demean is expected


def test_analyze_rejects_invalid_boolean(tmp_path):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(
            ["analyze", "-fix_dir", str(tmp_path), "-norm", "sometimes"]
        )

    assert error.value.code == 2
