from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fixrabies.interfaces.validate_inputs import (
    pyfix_model_exists,
    validate_data_directory,
    validate_feature_files,
    validate_hand_label_files,
    validate_select_label,
    validate_trained_model,
    validate_training_scans,
    validate_training_set,
)
from fixrabies.label import get_matching_training_set
from fixrabies.train import run_train


def test_validate_label_inputs():
    fix_dir = Path("~/test").expanduser()
    data_dir = Path(validate_data_directory(fix_dir))
    scan_list = sorted(
        str(path)
        for path in data_dir.iterdir()
        if path.is_dir()
    )

    assert validate_feature_files(scan_list) == scan_list
    assert validate_select_label(0) == 0
    assert validate_select_label(100) == 100
    assert get_matching_training_set(
        fix_dir,
        len(scan_list),
        100,
    ) == (fix_dir / "model" / "training_set.txt").read_text(
        encoding="utf-8"
    ).splitlines()


def test_validate_training_inputs_and_run_training():
    fix_dir = Path("~/test").expanduser()
    data_dir = validate_data_directory(fix_dir)
    training_scans = validate_training_scans(
        data_dir,
        validate_training_set(fix_dir),
    )

    assert validate_feature_files(training_scans) == training_scans
    assert validate_hand_label_files(training_scans) == training_scans

    args = SimpleNamespace(
        fix_dir=fix_dir,
        fix_model="test_model",
        species="mouseDSURQE",
    )
    with (
        patch("fixrabies.interfaces.pyfix.Training") as training,
        patch(
            "fixrabies.interfaces.validate_inputs.validate_trained_model"
        ) as validate_model,
    ):
        run_train(args)

    training.assert_called_once_with(
        mel_icas=training_scans,
        model_stem=str(fix_dir / "model" / "test_model"),
        species="mouseDSURQE",
    )
    training.return_value.run.assert_called_once_with()
    validate_model.assert_called_once_with(fix_dir, "test_model")


def test_validate_trained_model():
    fix_dir = Path("~/test").expanduser()
    model_file = fix_dir / "model" / "test_model.pyfix_model"
    model_file.touch()
    try:
        assert validate_trained_model(fix_dir, "test_model") == str(model_file)
    finally:
        model_file.unlink()


def test_run_train_rejects_existing_pyfix_model():
    fix_dir = Path("~/test").expanduser()

    assert pyfix_model_exists("Standard")
    with pytest.raises(
        ValueError,
        match="FIX model already exists in PyFIX: Standard.pyfix_model",
    ):
        run_train(SimpleNamespace(fix_dir=fix_dir, fix_model="Standard"))
