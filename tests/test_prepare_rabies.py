from pathlib import Path

import pytest

from fixrabies.interfaces.prepare_rabies import prepare_rabies, validate_scans


@pytest.fixture
def preprocessing_tree(tmp_path):
    scan_info = "_scan_info_sub-001_ses-01_task-rest_run-1"
    run_dir = "_run_1"
    files = {
        "bold_file": tmp_path
        / "bold_datasink/commonspace_bold"
        / scan_info
        / run_dir
        / "sub-001_ses-01_task-rest_run-1_resampled.nii.gz",
        "fd": tmp_path / "motion_datasink/FD_csv" / scan_info / run_dir / "fd.csv",
        "motion": tmp_path
        / "motion_datasink/motion_params_csv"
        / scan_info
        / run_dir
        / "motion.csv",
        "bg": tmp_path
        / "bold_datasink/commonspace_resampled_template/commonspace_template.nii.gz",
        "mask": tmp_path
        / "bold_datasink/commonspace_mask/commonspace_template_brain_mask_resampled.nii.gz",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    files["highres"] = files["bg"]
    return tmp_path, files


def test_prepare_rabies_discovers_scan_paths(preprocessing_tree):
    preprocess_dir, files = preprocessing_tree
    scans = prepare_rabies(preprocess_dir, 1.2)

    assert scans == {
        "sub-001_ses-01_task-rest_run-1": {
            **{key: str(path) for key, path in files.items()},
            "tr": 1.2,
        }
    }
    assert validate_scans(scans) is True


def test_prepare_rabies_expands_home(preprocessing_tree, monkeypatch):
    preprocess_dir, _ = preprocessing_tree
    monkeypatch.setenv("HOME", str(preprocess_dir.parent))

    assert prepare_rabies(f"~/{preprocess_dir.name}", 1.2) == prepare_rabies(
        preprocess_dir, 1.2
    )


@pytest.mark.parametrize("exists", [False, True], ids=["missing", "empty"])
def test_prepare_rabies_rejects_missing_inputs(tmp_path, exists):
    preprocess_dir = tmp_path / "preprocessing"
    if exists:
        preprocess_dir.mkdir()

    with pytest.raises(AssertionError):
        prepare_rabies(preprocess_dir, 1.2)


@pytest.mark.parametrize("key", ["bold_file", "fd", "motion", "bg", "highres", "mask"])
@pytest.mark.parametrize("failure", ["missing-key", "none", "missing-file"])
def test_validate_scans_rejects_missing_files(preprocessing_tree, key, failure):
    preprocess_dir, _ = preprocessing_tree
    scans = prepare_rabies(preprocess_dir, 1.2)
    scan = next(iter(scans.values()))
    if failure == "missing-key":
        del scan[key]
    elif failure == "none":
        scan[key] = None
    else:
        scan[key] = str(Path(preprocess_dir) / "missing.nii.gz")

    with pytest.raises(AssertionError, match=key):
        validate_scans(scans)
