from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fixrabies.interfaces.validate_inputs import (
    pyfix_model_exists,
    validate_classification_outputs,
    validate_cleanup_files,
    validate_data_directory,
    validate_feature_files,
    validate_hand_label_files,
    validate_select_label,
    validate_trained_model,
    validate_training_scans,
    validate_training_set,
)
from fixrabies.interfaces.pyfix import Cleanup
from fixrabies.classify import run_classify
from fixrabies.label import get_matching_training_set
from fixrabies.nodes.classify import classifier_node
from fixrabies.nodes.cleanup import (
    cleanup_node,
    get_kept_volumes,
    save_exclusion_file,
)
from fixrabies.review import run_review
from fixrabies.train import run_train
from fixrabies.workflows.review_wf import get_review_scan_info
from fixrabies.cleanup import run_cleanup


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


def test_get_review_scan_info_uses_classifier_artifact_lists():
    fix_dir = Path("~/test").expanduser()
    scan_list = sorted(
        str(path)
        for path in (fix_dir / "data").iterdir()
        if path.is_dir()
    )

    scans = get_review_scan_info(scan_list, "test_v1", 20)

    assert set(scans) == {Path(scan).name for scan in scan_list}
    for scan in scans.values():
        assert scan["noise_list"].endswith(
            "fix4melview_test_v1_thr20.txt"
        )
        assert Path(scan["noise_list"]).is_file()
        assert scan["n_ic"] > 0


def test_run_review_creates_metadata_workflow():
    fix_dir = Path("~/test").expanduser()
    args = SimpleNamespace(
        fix_dir=fix_dir,
        fix_model="test_v1",
        threshold=20,
        nipype={"plugin": "Linear"},
    )
    with patch(
        "fixrabies.workflows.review_wf.create_review_workflow"
    ) as create_workflow:
        run_review(args)

    workflow_args = create_workflow.call_args.kwargs
    assert workflow_args["fix_dir"] == fix_dir
    assert workflow_args["scan_list"]
    assert workflow_args["fix_model"] == "test_v1"
    assert workflow_args["threshold"] == 20
    create_workflow.return_value.run.assert_called_once_with(
        plugin="Linear",
        plugin_args={},
    )


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


def test_cleanup_node_uses_requested_flags():
    node = cleanup_node(
        motionhp=True,
        aggressive=True,
        highpass=100,
    )

    assert node.interface.cmd == "fix -a"
    assert node.inputs.motionhp is True
    assert node.inputs.aggressive is True
    assert node.inputs.highpass == 100


def test_cleanup_interface_command():
    artifact_list_file = next(
        Path("~/test/data").expanduser().glob(
            "*/fix4melview_test_v1_thr20.txt"
        )
    )
    cleanup = Cleanup(
        artifact_list_file=str(artifact_list_file),
        motionhp=True,
        aggressive=True,
        highpass=100,
    )

    assert cleanup.cmdline == (
        f"fix -a {artifact_list_file} -m -h 100.0 -A"
    )


def test_run_cleanup_creates_iterable_workflow():
    fix_dir = Path("~/test").expanduser()
    args = SimpleNamespace(
        fix_dir=fix_dir,
        fix_model="test_v1",
        threshold=20,
        motionhp=True,
        aggressive=True,
        highpass=100,
        smooth=0,
        DVARS_threshold=2,
        FD_threshold=0.1,
        nipype={"plugin": "Linear"},
    )
    with patch(
        "fixrabies.workflows.cleanup_wf.create_cleanup_workflow"
    ) as create_workflow, patch(
        "fixrabies.workflows.cleanup_wf.create_post_cleanup_workflow"
    ) as create_post_cleanup_workflow:
        run_cleanup(args)

    workflow_args = create_workflow.call_args.kwargs
    assert workflow_args["fix_dir"] == fix_dir
    assert workflow_args["scan_list"]
    assert workflow_args["fix_model"] == "test_v1"
    assert workflow_args["threshold"] == 20
    assert workflow_args["motionhp"] is True
    assert workflow_args["aggressive"] is True
    assert workflow_args["highpass"] == 100
    create_workflow.return_value.run.assert_called_once_with(
        plugin="Linear",
        plugin_args={},
    )
    create_post_cleanup_workflow.assert_called_once_with(
        fix_dir=fix_dir,
        scan_list=workflow_args["scan_list"],
        dvars_threshold=2,
        fd_threshold=0.1,
        smooth=0,
    )
    create_post_cleanup_workflow.return_value.run.assert_called_once_with(
        plugin="Linear",
        plugin_args={},
    )


def test_get_kept_volumes_excludes_fd_or_dvars_outliers():
    assert get_kept_volumes(
        dvars=[0, 3, 0, 3],
        mean_fd=[0, 0, 0.2, 0.2],
        dvars_threshold=2,
        fd_threshold=0.1,
    ) == [0]


def test_zero_thresholds_do_not_exclude_volumes():
    assert get_kept_volumes(
        dvars=[0, 3],
        mean_fd=[0, 0.2],
        dvars_threshold=0,
        fd_threshold=0,
    ) == [0, 1]


def test_save_exclusion_file():
    scan_dir = next(Path("~/test/data").expanduser().iterdir())
    output_file = scan_dir / "exclude.csv"
    original = output_file.read_bytes() if output_file.exists() else None
    try:
        save_exclusion_file(
            dvars=[0, 3, 0, 3],
            mean_fd=[0, 0, 0.2, 0.2],
            dvars_threshold=2,
            fd_threshold=0.1,
            scan_dir=scan_dir,
        )
        assert output_file.read_text(encoding="utf-8").splitlines() == [
            "DVARS,MeanFD,DVARS_selection,FD_selection",
            "0,0.0,0,0",
            "3,0.0,1,0",
            "0,0.2,0,1",
            "3,0.2,1,1",
        ]
    finally:
        if original is None:
            output_file.unlink(missing_ok=True)
        else:
            output_file.write_bytes(original)


def test_validate_cleanup_files():
    fix_dir = Path("~/test").expanduser()
    scan_list = sorted(
        str(path)
        for path in (fix_dir / "data").iterdir()
        if path.is_dir()
    )

    assert validate_cleanup_files(scan_list) == scan_list
