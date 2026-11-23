from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fixrabies import cli
from fixrabies.analyze import run_analyze
from fixrabies.classify import run_classify
from fixrabies.cleanup import run_cleanup
from fixrabies.interfaces.pyfix import Cleanup, Training
from fixrabies.interfaces.validate_inputs import (
    copy_existing_pyfix_model,
    get_pending_dual_regression_scans,
    pyfix_model_exists,
    validate_analysis_files,
    validate_analysis_ica_files,
    validate_analysis_image,
    validate_analysis_mask_files,
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
from fixrabies.nodes.cleanup import cleanup_node, get_kept_volumes, save_exclusion_file
from fixrabies.nodes.prepare_node import censor_frames_by_fd
from fixrabies.train import run_train


@pytest.fixture
def fix_tree(tmp_path):
    for name in ("scan-b", "scan-a"):
        scan = tmp_path / "data" / name
        (scan / "fix").mkdir(parents=True)
        for filename in (
            "fix/features.csv",
            "hand_labels_noise.txt",
            "filtered_func_data_clean.nii.gz",
            "filtered_func_data_clean_smoothed.nii.gz",
            "fd.csv",
            "ica.nii.gz",
            "mask.nii.gz",
            "fix4melview_test_v1_thr20.txt",
        ):
            (scan / filename).touch()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "test_v1.pyfix_model").write_bytes(b"placeholder model")
    (model_dir / "training_set.txt").write_text("scan-a\n\nscan-b\n", encoding="utf-8")
    return tmp_path


def test_censor_frames_by_fd_keeps_bold_and_fd_rows_aligned(tmp_path):
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    image_file = scan_dir / "filtered_func_data.nii.gz"
    image_data = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 4)
    nib.Nifti1Image(image_data, np.eye(4)).to_filename(image_file)
    pd.DataFrame(
        {
            "MeanFD": [0.05, 0.2, 0.1, 0.4],
            "Other": [1, 2, 3, 4],
        }
    ).to_csv(scan_dir / "fd.csv", index=False)
    motion_dir = scan_dir / "mc"
    motion_dir.mkdir()
    pd.DataFrame(
        {
            0: [1, 2, 3, 4],
            1: [5, 6, 7, 8],
        }
    ).to_csv(
        motion_dir / "prefiltered_func_data_mcf.par",
        sep="\t",
        header=False,
        index=False,
    )
    (scan_dir / "design.fsf").write_text(
        "set fmri(tr) 1\nset fmri(npts) 4\n",
        encoding="utf-8",
    )

    censor_frames_by_fd(image_file, scan_dir, 0.1)

    assert nib.load(image_file).shape == (2, 2, 2, 2)
    assert np.array_equal(
        nib.load(image_file).get_fdata(),
        image_data[..., [0, 2]],
    )
    assert pd.read_csv(scan_dir / "fd.csv").to_dict("list") == {
        "MeanFD": [0.05, 0.1],
        "Other": [1, 3],
    }
    assert pd.read_csv(
        motion_dir / "prefiltered_func_data_mcf.par",
        sep="\t",
        header=None,
    ).to_dict("list") == {0: [1.0, 3.0], 1: [5.0, 7.0]}
    assert pd.read_csv(scan_dir / "prepare_exclude.csv").to_dict("list") == {
        "Frame": [0, 1, 2, 3],
        "MeanFD": [0.05, 0.2, 0.1, 0.4],
        "Other": [1, 2, 3, 4],
        "FD_selection": [0, 1, 0, 1],
    }
    assert (scan_dir / "design.fsf").read_text(encoding="utf-8") == (
        "set fmri(tr) 1\nset fmri(npts) 2\n"
    )


def test_run_classify_creates_iterable_workflow(fix_tree):
    fix_dir = fix_tree
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
    assert workflow_args["scan_list"] == [
        str(fix_dir / "data" / name) for name in ("scan-a", "scan-b")
    ]
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


def test_run_cleanup_creates_iterable_workflow(fix_tree):
    fix_dir = fix_tree
    args = SimpleNamespace(
        fix_dir=fix_dir,
        fix_model="test_v1",
        threshold=20,
        motionhp=True,
        aggressive=True,
        highpass=100,
        smooth=0,
        mask=True,
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
    assert workflow_args["scan_list"] == [
        str(fix_dir / "data" / name) for name in ("scan-a", "scan-b")
    ]
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
        mask=True,
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


def test_run_analyze_creates_workflow(fix_tree):
    fix_dir = fix_tree
    template = fix_dir / "template.nii.gz"
    template.touch()
    args = SimpleNamespace(
        fix_dir=fix_dir,
        result_dir=Path("results"),
        DR_template=template,
        seed=[],
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
            side_effect=lambda _, __, scans: scans,
        ),
    ):
        run_analyze(args)

    workflow_args = create_workflow.call_args.kwargs
    assert workflow_args["fix_dir"] == fix_dir
    assert workflow_args["result_dir"] == Path("results")
    assert workflow_args["scan_list"] == [
        str(fix_dir / "data" / name) for name in ("scan-a", "scan-b")
    ]
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


@pytest.mark.parametrize("state", ["missing", "empty", "file-only"])
def test_validate_data_directory_rejects_no_scans(tmp_path, state):
    if state != "missing":
        (tmp_path / "data").mkdir()
    if state == "file-only":
        (tmp_path / "data" / "not-a-scan").touch()
    with pytest.raises(ValueError, match="Data directory"):
        validate_data_directory(tmp_path)


def test_validate_data_directory(fix_tree):
    assert validate_data_directory(fix_tree) == str(fix_tree / "data")


@pytest.mark.parametrize(
    "validator,filename",
    [
        (validate_feature_files, "fix/features.csv"),
        (validate_hand_label_files, "hand_labels_noise.txt"),
        (validate_cleanup_files, "filtered_func_data_clean.nii.gz"),
        (validate_cleanup_files, "fd.csv"),
        (validate_analysis_files, "filtered_func_data_clean_smoothed.nii.gz"),
        (validate_analysis_ica_files, "ica.nii.gz"),
        (validate_analysis_mask_files, "mask.nii.gz"),
    ],
)
def test_required_scan_files(fix_tree, validator, filename):
    scans = [str(fix_tree / "data" / name) for name in ("scan-a", "scan-b")]
    assert validator(scans) == scans
    missing = Path(scans[1]) / filename
    missing.unlink()
    with pytest.raises(ValueError) as error:
        validator(scans)
    assert str(missing) in str(error.value)


def test_validate_classification_outputs_uses_model_and_threshold(fix_tree):
    scans = [str(fix_tree / "data" / "scan-a")]
    assert validate_classification_outputs(scans, "test_v1", 20) == scans
    for model, threshold in (("other", 20), ("test_v1", 21)):
        with pytest.raises(ValueError, match=f"fix4melview_{model}_thr{threshold}"):
            validate_classification_outputs(scans, model, threshold)


@pytest.mark.parametrize("validator", [validate_select_label, validate_dice_threshold])
@pytest.mark.parametrize("value", [0, 50, 100, -1, 101])
def test_percentage_threshold_bounds(validator, value):
    if 0 <= value <= 100:
        assert validator(value) == value
    else:
        with pytest.raises(ValueError, match="between 0 and 100"):
            validator(value)


@pytest.mark.parametrize("value", ["0p", "100p", "90p", "7.5z", "101z"])
def test_validate_extract_threshold(value):
    assert validate_extract_threshold(value) == value


@pytest.mark.parametrize("value", ["101p", "90x", "-1z", "", "90"])
def test_validate_extract_threshold_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="extract_threshold"):
        validate_extract_threshold(value)


def test_validate_analysis_image(tmp_path):
    image = tmp_path / "template.nii.gz"
    with pytest.raises(ValueError, match="Template does not exist"):
        validate_analysis_image(image, "Template")
    image.touch()
    assert validate_analysis_image(image, "Template") == str(image.resolve())


def test_training_list_and_scan_paths(fix_tree):
    assert validate_training_set(fix_tree) == ["scan-a", "scan-b"]
    scans = ["scan-a", str(fix_tree / "data" / "scan-b")]
    assert validate_training_scans(fix_tree / "data", scans) == [
        str(fix_tree / "data" / "scan-a"),
        str(fix_tree / "data" / "scan-b"),
    ]
    with pytest.raises(ValueError, match="not in the data directory"):
        validate_training_scans(fix_tree / "data", [str(fix_tree / "scan-a")])
    with pytest.raises(ValueError, match="does not exist"):
        validate_training_scans(fix_tree / "data", ["missing"])


@pytest.mark.parametrize("state", ["missing", "empty"])
def test_training_list_requires_nonempty_file(fix_tree, state):
    training_file = fix_tree / "model" / "training_set.txt"
    if state == "missing":
        training_file.unlink()
    else:
        training_file.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Training set file"):
        validate_training_set(fix_tree)


def test_validate_trained_model(fix_tree):
    assert validate_trained_model(fix_tree, "test_v1") == str(
        fix_tree / "model" / "test_v1.pyfix_model"
    )
    with pytest.raises(ValueError, match="Trained model was not created"):
        validate_trained_model(fix_tree, "missing")


@pytest.mark.parametrize("select_label,expected", [(0, 1), (1, 1), (50, 2), (100, 3)])
def test_matching_training_set_counts(tmp_path, select_label, expected):
    assert get_matching_training_set(tmp_path, 3, select_label) is None
    (tmp_path / "model").mkdir()
    training_file = tmp_path / "model" / "training_set.txt"
    selected = [f"scan-{index}" for index in range(expected)]
    training_file.write_text("\n".join(selected) + "\n\n", encoding="utf-8")
    assert get_matching_training_set(tmp_path, 3, select_label) == selected
    training_file.write_text("\n".join(selected + ["extra"]), encoding="utf-8")
    assert get_matching_training_set(tmp_path, 3, select_label) is None


def test_run_train_uses_validated_training_scans(fix_tree):
    args = SimpleNamespace(
        fix_dir=fix_tree, fix_model="new_model", species="mouseDSURQE"
    )
    with (
        patch(
            "fixrabies.interfaces.validate_inputs.copy_existing_pyfix_model",
            return_value=None,
        ),
        patch("fixrabies.interfaces.pyfix.Training") as training,
        patch(
            "fixrabies.interfaces.validate_inputs.validate_trained_model"
        ) as validate_model,
    ):
        run_train(args)
    training.assert_called_once_with(
        mel_icas=[str(fix_tree / "data" / name) for name in ("scan-a", "scan-b")],
        model_stem=str(fix_tree / "model" / "new_model"),
        species="mouseDSURQE",
    )
    training.return_value.run.assert_called_once_with()
    validate_model.assert_called_once_with(fix_tree, "new_model")


@pytest.mark.parametrize("with_suffix", [True, False])
def test_run_train_copies_model_without_training(tmp_path, with_suffix):
    source = tmp_path / "source.pyfix_model"
    source.write_bytes(b"placeholder model")
    destination = tmp_path / "working"
    model_argument = str(source if with_suffix else source.with_suffix(""))
    with patch("fixrabies.interfaces.pyfix.Training") as training:
        run_train(SimpleNamespace(fix_dir=destination, fix_model=model_argument))
    assert (destination / "model" / source.name).read_bytes() == source.read_bytes()
    training.assert_not_called()


def test_packaged_model_lookup_and_copy_are_isolated(tmp_path):
    packaged = tmp_path / "package"
    models = packaged / "resources" / "models"
    models.mkdir(parents=True)
    (models / "Standard.pyfix_model").write_bytes(b"placeholder packaged model")
    with patch(
        "fixrabies.interfaces.validate_inputs.files", return_value=packaged
    ) as resources:
        assert pyfix_model_exists("Standard")
        assert not pyfix_model_exists("missing")
        assert copy_existing_pyfix_model(tmp_path / "working", "missing") is None
        assert copy_existing_pyfix_model(tmp_path / "working", "Standard") == "Standard"
    resources.assert_called_with("pyfix")
    assert (
        tmp_path / "working/model/Standard.pyfix_model"
    ).read_bytes() == b"placeholder packaged model"


def test_copy_model_rejects_missing_path(tmp_path):
    with pytest.raises(ValueError, match="existing .pyfix_model"):
        copy_existing_pyfix_model(tmp_path, str(tmp_path / "missing.pyfix_model"))


@pytest.mark.parametrize(
    "command", ["train", "classify", "review", "cleanup", "analyze"]
)
def test_cli_dispatch_is_mocked(command, tmp_path):
    args = SimpleNamespace(fix_dir=tmp_path)
    with patch(f"fixrabies.{command}.run_{command}") as runner:
        getattr(cli, f"run_{command}_command")(args)
    runner.assert_called_once_with(args)


def test_command_construction_does_not_execute_fix(fix_tree):
    scan = fix_tree / "data" / "scan-a"
    model = fix_tree / "model" / "test_v1.pyfix_model"
    classifier = classifier_node(str(model), 20)
    assert classifier.interface.cmd == "fix -c"
    assert classifier.inputs.trained_wts_file == str(model)
    assert classifier.inputs.thresh == 20
    artifact = scan / "fix4melview_test_v1_thr20.txt"
    cleanup = Cleanup(
        artifact_list_file=str(artifact), motionhp=True, aggressive=True, highpass=100
    )
    assert cleanup.cmdline == f"fix -a {artifact} -m -h 100.0 -A"
    training = Training(
        species="mouseDSURQE",
        model_stem=str(model.with_suffix("")),
        mel_icas=[str(scan)],
    )
    assert training.cmdline == f"fix -t -s mouseDSURQE {model.with_suffix('')} {scan}"


def test_pending_dual_regression_requires_both_outputs(fix_tree):
    scans = [str(fix_tree / "data" / name) for name in ("scan-a", "scan-b")]
    dr_dir = fix_tree / "results" / "scan-a" / "dr"
    dr_dir.mkdir(parents=True)
    assert get_pending_dual_regression_scans(fix_tree, Path("results"), scans) == scans
    (dr_dir / "dr_stage2_subject00000.nii.gz").touch()
    assert get_pending_dual_regression_scans(fix_tree, Path("results"), scans) == scans
    (dr_dir / "dr_stage2_subject00000_Z.nii.gz").touch()
    assert (
        get_pending_dual_regression_scans(fix_tree, Path("results"), scans) == scans[1:]
    )


def test_save_exclusion_file(tmp_path):
    save_exclusion_file(
        dvars=[0, 3, 0, 3],
        mean_fd=[0, 0, 0.2, 0.2],
        dvars_threshold=2,
        fd_threshold=0.1,
        scan_dir=tmp_path,
    )
    assert pd.read_csv(tmp_path / "exclude.csv")["Combined_selection"].tolist() == [
        0,
        1,
        1,
        1,
    ]


def test_seed_paths_and_correlation(tmp_path):
    times, betas, z_files, corr_file, names = get_seed_output_files(
        [str(tmp_path / "seed.nii.gz")], str(tmp_path)
    )
    assert times == [str(tmp_path / "seed.txt")]
    assert betas == [str(tmp_path / "seed.nii.gz")]
    assert z_files == [str(tmp_path / "seed_Z.nii.gz")]
    assert names == ["seed"]
    Path(times[0]).write_text("1\n2\n3\n", encoding="utf-8")
    assert save_seed_correlation(times, betas, corr_file, names) == corr_file
    assert pd.read_csv(corr_file, sep="\t", index_col=0).loc["seed", "seed"] == 1


@pytest.mark.parametrize("extract_threshold", ["50p", "4z"])
def test_dual_regression_outcomes_on_tiny_images(tmp_path, extract_threshold):
    data = np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1)
    for filename, values in (
        ("ica.nii.gz", data),
        ("mask.nii.gz", np.ones((2, 2, 2), dtype=np.float32)),
        ("dr_stage2_subject00000.nii.gz", data * 2),
        ("dr_stage2_subject00000_Z.nii.gz", data * 3),
    ):
        nib.Nifti1Image(values, np.eye(4)).to_filename(tmp_path / filename)
    output = save_dual_regression_outcomes(
        tmp_path / "ica.nii.gz",
        tmp_path / "mask.nii.gz",
        tmp_path,
        50,
        extract_threshold,
    )
    outcomes = pd.read_csv(output)
    assert list(outcomes) == ["beta_dice", "z_dice", "beta_value", "z_value"]
    np.testing.assert_allclose(outcomes.iloc[0], [1, 1, 11, 16.5])


def test_run_analyze_dispatches_pending_and_completed_scans(fix_tree):
    scans = [str(fix_tree / "data" / name) for name in ("scan-a", "scan-b")]
    args = SimpleNamespace(
        fix_dir=fix_tree,
        result_dir=Path("results"),
        DR_template=None,
        seed=[],
        norm=False,
        demean=False,
        dice_threshold=90,
        extract_threshold="15z",
        nipype={"plugin": "MultiProc", "n_procs": 2},
    )
    with (
        patch(
            "fixrabies.workflows.analyze_wf.create_analyze_workflow"
        ) as create_workflow,
        patch(
            "fixrabies.interfaces.validate_inputs.get_pending_dual_regression_scans",
            side_effect=[scans[:1], []],
        ),
    ):
        run_analyze(args)
    assert create_workflow.call_count == 2
    pending, completed = [call.kwargs for call in create_workflow.call_args_list]
    assert pending["scan_list"] == scans[:1]
    assert pending["run_dual_regression"] is True
    assert completed["scan_list"] == scans
    assert completed["run_dual_regression"] is False
    assert completed["demean"] is False
    assert completed["norm"] is False
    assert create_workflow.return_value.run.call_count == 2
    create_workflow.return_value.run.assert_called_with(
        plugin="MultiProc", plugin_args={"n_procs": 2}
    )
    assert args.nipype == {"plugin": "MultiProc", "n_procs": 2}


def test_classify_missing_features_stops_before_workflow(fix_tree):
    (fix_tree / "data/scan-a/fix/features.csv").unlink()
    args = SimpleNamespace(fix_dir=fix_tree, fix_model="test_v1", threshold=20)
    with (
        patch(
            "fixrabies.workflows.classify_wf.create_classify_workflow"
        ) as create_workflow,
        pytest.raises(ValueError, match="Missing FIX feature files"),
    ):
        run_classify(args)
    create_workflow.assert_not_called()
