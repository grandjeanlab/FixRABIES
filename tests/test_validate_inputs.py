from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fixrabies.analyze import run_analyze
from fixrabies.classify import run_classify
from fixrabies.cleanup import run_cleanup
from fixrabies.interfaces.pyfix import Cleanup
from fixrabies.interfaces.validate_inputs import (
    get_pending_dual_regression_scans,
    pyfix_model_exists,
    validate_analysis_files,
    validate_analysis_image,
    validate_classification_outputs,
    validate_cleanup_files,
    validate_data_directory,
    validate_dice_threshold,
    validate_extract_threshold,
    validate_feature_files,
    validate_hand_label_files,
    validate_select_label,
    validate_trained_model,
    validate_training_scans,
    validate_training_set,
)
from fixrabies.label import get_matching_training_set
from fixrabies.nodes.analyze import (
    get_seed_output_files,
    save_dual_regression_outcomes,
    save_seed_correlation,
)
from fixrabies.nodes.classify import classifier_node
from fixrabies.nodes.cleanup import (
    cleanup_node,
    get_kept_volumes,
    save_exclusion_file,
)
from fixrabies.review import run_review
from fixrabies.train import run_train
from fixrabies.workflows.analyze_wf import create_analyze_workflow
from fixrabies.workflows.review_wf import get_review_scan_info


def test_validate_label_inputs():
    fix_dir = Path("~/test").expanduser()
    data_dir = Path(validate_data_directory(fix_dir))
    scan_list = sorted(str(path) for path in data_dir.iterdir() if path.is_dir())

    assert validate_feature_files(scan_list) == scan_list
    assert validate_select_label(0) == 0
    assert validate_select_label(100) == 100
    assert (
        get_matching_training_set(
            fix_dir,
            len(scan_list),
            100,
        )
        == (fix_dir / "model" / "training_set.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )


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
        str(path) for path in (fix_dir / "data").iterdir() if path.is_dir()
    )

    assert (
        validate_classification_outputs(
            scan_list,
            "test_v1",
            20,
        )
        == scan_list
    )


def test_get_review_scan_info_uses_classifier_artifact_lists():
    fix_dir = Path("~/test").expanduser()
    scan_list = sorted(
        str(path) for path in (fix_dir / "data").iterdir() if path.is_dir()
    )

    scans = get_review_scan_info(scan_list, "test_v1", 20)

    assert set(scans) == {Path(scan).name for scan in scan_list}
    for scan in scans.values():
        assert scan["noise_list"].endswith("fix4melview_test_v1_thr20.txt")
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
    with (
        patch(
            "fixrabies.workflows.classify_wf.create_classify_workflow"
        ) as create_workflow,
        patch(
            "fixrabies.interfaces.validate_inputs.validate_classification_outputs"
        ) as validate_outputs,
    ):
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
        Path("~/test/data").expanduser().glob("*/fix4melview_test_v1_thr20.txt")
    )
    cleanup = Cleanup(
        artifact_list_file=str(artifact_list_file),
        motionhp=True,
        aggressive=True,
        highpass=100,
    )

    assert cleanup.cmdline == (f"fix -a {artifact_list_file} -m -h 100.0 -A")


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
    with (
        patch(
            "fixrabies.workflows.cleanup_wf.create_cleanup_workflow"
        ) as create_workflow,
        patch(
            "fixrabies.workflows.cleanup_wf.create_post_cleanup_workflow"
        ) as create_post_cleanup_workflow,
    ):
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
        str(path) for path in (fix_dir / "data").iterdir() if path.is_dir()
    )

    assert validate_cleanup_files(scan_list) == scan_list


def test_validate_analysis_inputs():
    fix_dir = Path("~/test").expanduser()
    scan_list = sorted(
        str(path) for path in (fix_dir / "data").iterdir() if path.is_dir()
    )
    template = Path("src/fixrabies/templates/mouseDSURQE/ica.nii.gz").resolve()

    assert validate_analysis_files(scan_list) == scan_list
    assert validate_analysis_image(
        template,
        "Dual regression template",
    ) == str(template)
    assert validate_extract_threshold("90p") == "90p"
    assert validate_extract_threshold("7.5z") == "7.5z"
    assert validate_dice_threshold(90) == 90
    with pytest.raises(ValueError, match="between 0 and 100"):
        validate_extract_threshold("101p")
    with pytest.raises(ValueError, match="percentage"):
        validate_extract_threshold("90x")
    with pytest.raises(ValueError, match="dice_threshold"):
        validate_dice_threshold(101)


def test_analyze_workflow_configures_dual_regression():
    fix_dir = Path("~/test").expanduser()
    scan_list = [str(next((fix_dir / "data").iterdir()))]
    template = str(Path("src/fixrabies/templates/mouseDSURQE/ica.nii.gz").resolve())

    workflow = create_analyze_workflow(
        fix_dir=fix_dir,
        scan_list=scan_list,
        dr_template=template,
        seed_files=[],
        norm=True,
        dice_threshold=90,
        extract_threshold="15z",
        run_dual_regression=True,
    )

    dual_regression = workflow.get_node("dual_regression")
    assert dual_regression.inputs.des_norm is True
    assert dual_regression.inputs.one_sample_group_mean is True
    assert dual_regression.inputs.n_perm == 0
    outcomes = workflow.get_node("outcomes")
    assert outcomes.inputs.dice_threshold == 90
    assert outcomes.inputs.extract_threshold == "15z"
    assert workflow.get_node("infosource").iterables == [("scan_dir", scan_list)]


def test_analyze_workflow_uses_scan_ica_without_template():
    fix_dir = Path("~/test").expanduser()
    scan_list = [str(next((fix_dir / "data").iterdir()))]

    workflow = create_analyze_workflow(
        fix_dir=fix_dir,
        scan_list=scan_list,
        dr_template=None,
        seed_files=[],
        norm=False,
        dice_threshold=90,
        extract_threshold="15z",
        run_dual_regression=True,
    )

    assert workflow.get_node("dual_regression") is not None
    assert workflow.get_node("ica_file") is not None


def test_analyze_outcomes_workflow_skips_dual_regression():
    fix_dir = Path("~/test").expanduser()
    scan_list = [str(next((fix_dir / "data").iterdir()))]

    workflow = create_analyze_workflow(
        fix_dir=fix_dir,
        scan_list=scan_list,
        dr_template=None,
        seed_files=[],
        norm=True,
        dice_threshold=90,
        extract_threshold="15z",
        run_dual_regression=False,
    )

    assert workflow.get_node("dual_regression") is None
    assert workflow.get_node("outcomes") is not None


def test_seed_analysis_paths_and_correlation():
    fix_dir = Path("~/test").expanduser()
    scan_dir = next((fix_dir / "data").iterdir())
    seed_dir = fix_dir / "results" / scan_dir.name / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_file = Path(
        "~/test/data/sub-jgrAesMEDISOc11L_ses-1_task-rest_run-1/aca_r.nii.gz"
    ).expanduser()
    time_file = seed_dir / "aca_r.txt"
    corr_file = seed_dir / "corr.txt"
    original = corr_file.read_bytes() if corr_file.exists() else None
    try:
        (
            seed_time_files,
            beta_files,
            z_files,
            generated_corr_file,
            seed_basenames,
        ) = get_seed_output_files([str(seed_file)], str(seed_dir))
        assert seed_time_files == [str(time_file)]
        assert beta_files == [str(seed_dir / "aca_r.nii.gz")]
        assert z_files == [str(seed_dir / "aca_r_Z.nii.gz")]
        assert generated_corr_file == str(corr_file)
        assert seed_basenames == ["aca_r"]

        time_file.write_text("1\n2\n3\n", encoding="utf-8")
        save_seed_correlation(
            seed_time_files,
            beta_files,
            generated_corr_file,
            seed_basenames,
        )
        assert corr_file.read_text(encoding="utf-8").splitlines() == [
            "\taca_r",
            "aca_r\t1.0",
        ]
    finally:
        time_file.unlink(missing_ok=True)
        if original is None:
            corr_file.unlink(missing_ok=True)
        else:
            corr_file.write_bytes(original)


def test_analyze_workflow_configures_seed_analysis():
    fix_dir = Path("~/test").expanduser()
    scan_list = [str(next((fix_dir / "data").iterdir()))]
    seed_file = str(
        Path(
            "~/test/data/sub-jgrAesMEDISOc11L_ses-1_task-rest_run-1/aca_r.nii.gz"
        ).expanduser()
    )

    workflow = create_analyze_workflow(
        fix_dir=fix_dir,
        scan_list=scan_list,
        dr_template=None,
        seed_files=[seed_file],
        norm=True,
        dice_threshold=90,
        extract_threshold="15z",
        run_dual_regression=False,
    )

    image_meants = workflow.get_node("image_meants")
    glm = workflow.get_node("seed_glm")
    assert image_meants.interface.cmd == "fslmeants"
    assert glm.interface.inputs.dat_norm is True
    assert glm.interface.inputs.demean is True
    assert glm.interface.inputs.des_norm is True
    seed_correlation = workflow.get_node("seed_correlation")
    assert seed_correlation is not None
    assert workflow._graph.has_edge(glm, seed_correlation)


def test_save_dual_regression_outcomes():
    fix_dir = Path("~/test").expanduser()
    scan_dir = next((fix_dir / "data").iterdir())
    dr_dir = fix_dir / "results" / scan_dir.name / "dr"
    output_file = dr_dir / "outcomes.csv"
    original = output_file.read_bytes() if output_file.exists() else None
    try:
        save_dual_regression_outcomes(
            scan_dir / "ica.nii.gz",
            scan_dir / "mask.nii.gz",
            dr_dir,
            90,
            "7.5z",
        )
        outcomes = output_file.read_text(encoding="utf-8").splitlines()
        assert outcomes[0] == "beta_dice,z_dice,beta_value,z_value"
        assert len(outcomes) == 19
    finally:
        if original is None:
            output_file.unlink(missing_ok=True)
        else:
            output_file.write_bytes(original)


def test_get_pending_dual_regression_scans():
    fix_dir = Path("~/test").expanduser()
    scan_list = sorted(
        str(path) for path in (fix_dir / "data").iterdir() if path.is_dir()
    )

    assert get_pending_dual_regression_scans(fix_dir, scan_list) == []


def test_run_analyze_creates_workflow():
    fix_dir = Path("~/test").expanduser()
    template = Path("src/fixrabies/templates/mouseDSURQE/ica.nii.gz").resolve()
    args = SimpleNamespace(
        fix_dir=fix_dir,
        DR_template=template,
        seed={"apply": False, "seeds": []},
        norm=True,
        dice_threshold=90,
        extract_threshold="15z",
        nipype={"plugin": "Linear"},
    )
    with (
        patch(
            "fixrabies.workflows.analyze_wf.create_analyze_workflow"
        ) as create_workflow,
        patch(
            "fixrabies.interfaces.validate_inputs.get_pending_dual_regression_scans",
            side_effect=lambda _, scans: scans,
        ),
    ):
        run_analyze(args)

    workflow_args = create_workflow.call_args.kwargs
    assert workflow_args["fix_dir"] == fix_dir
    assert workflow_args["scan_list"]
    assert workflow_args["dr_template"] == str(template)
    assert workflow_args["seed_files"] == []
    assert workflow_args["norm"] is True
    assert workflow_args["dice_threshold"] == 90
    assert workflow_args["extract_threshold"] == "15z"
    assert workflow_args["run_dual_regression"] is True
    create_workflow.return_value.run.assert_called_once_with(
        plugin="Linear",
        plugin_args={},
    )


def test_run_analyze_runs_outcomes_for_completed_scans():
    fix_dir = Path("~/test").expanduser()
    args = SimpleNamespace(
        fix_dir=fix_dir,
        DR_template=None,
        seed={"apply": False, "seeds": []},
        norm=True,
        dice_threshold=90,
        extract_threshold="15z",
        nipype={"plugin": "Linear"},
    )
    with patch(
        "fixrabies.workflows.analyze_wf.create_analyze_workflow"
    ) as create_workflow:
        run_analyze(args)

    workflow_args = create_workflow.call_args.kwargs
    assert workflow_args["scan_list"]
    assert workflow_args["run_dual_regression"] is False
    create_workflow.return_value.run.assert_called_once_with(
        plugin="Linear",
        plugin_args={},
    )
