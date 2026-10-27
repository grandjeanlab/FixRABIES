from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fixrabies.interfaces.validate_inputs import (
    pyfix_model_exists,
    validate_classification_outputs,
    validate_data_directory,
    validate_feature_files,
    validate_hand_label_files,
    validate_select_label,
    validate_trained_model,
    validate_training_scans,
    validate_training_set,
)
from fixrabies.classify import run_classify
from fixrabies.label import get_matching_training_set
from fixrabies.nodes.classify import classifier_node
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
    model_file = fix_dir / "model" / "test_v1.pyfix_model"

    assert validate_trained_model(fix_dir, "test_v1") == str(model_file)


def test_run_train_rejects_existing_pyfix_model():
    fix_dir = Path("~/test").expanduser()

    assert pyfix_model_exists("Standard")
    with pytest.raises(
        ValueError,
        match="FIX model already exists in PyFIX: Standard.pyfix_model",
    ):
        run_train(SimpleNamespace(fix_dir=fix_dir, fix_model="Standard"))


def test_classifier_node_uses_current_fix_command():
    model_file = Path("~/test/model/test_v1.pyfix_model").expanduser()
    node = classifier_node(str(model_file), 20)

    assert node.interface.cmd == "fix -c"
    assert node.inputs.trained_wts_file == str(model_file)
    assert node.inputs.thresh == 20


def test_validate_classification_outputs():
    fix_dir = Path("~/test").expanduser()
    scan_list = sorted(
        str(path)
        for path in (fix_dir / "data").iterdir()
        if path.is_dir()
    )

    assert validate_classification_outputs(
        scan_list,
        "test_v1",
        20,
    ) == scan_list


def test_run_classify_creates_iterable_workflow():
    fix_dir = Path("~/test").expanduser()
    args = SimpleNamespace(
        fix_dir=fix_dir,
        fix_model="test_v1",
        threshold=20,
        nipype={"plugin": "Linear"},
    )
    with patch(
        "fixrabies.workflows.classify_wf.create_classify_workflow"
    ) as create_workflow, patch(
        "fixrabies.interfaces.validate_inputs.validate_classification_outputs"
    ) as validate_outputs:
        run_classify(args)

    create_workflow.assert_called_once()
    workflow_args = create_workflow.call_args.kwargs
    assert workflow_args["fix_dir"] == fix_dir
    assert workflow_args["trained_wts_file"] == str(
        fix_dir / "model" / "test_v1.pyfix_model"
    )
    assert workflow_args["thresh"] == 20
    assert workflow_args["scan_list"]
    create_workflow.return_value.run.assert_called_once_with(
        plugin="Linear",
        plugin_args={},
    )
    validate_outputs.assert_called_once_with(
        workflow_args["scan_list"],
        "test_v1",
        20,
    )
