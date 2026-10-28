import re
from importlib.resources import files
from pathlib import Path


def validate_scans(scans):
    """
    Validate that all files referenced in the scans dict exist.

    Raises
    ------
    AssertionError
        If a required file is missing.
    """

    required_keys = [
        "bold_file",
        "fd",
        "motion",
        "bg",
        "highres",
        "mask",
    ]

    for scan_id, scan in scans.items():

        for key in required_keys:
            assert key in scan, f"{scan_id}: missing key '{key}'"

            assert scan[key] is not None, f"{scan_id}: '{key}' is None"

            path = Path(scan[key])

            assert path.exists(), f"{scan_id}: '{key}' does not exist\n" f"  {path}"

    return True


def validate_data_directory(fix_dir):
    data_dir = Path(fix_dir) / "data"
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    if not any(path.is_dir() for path in data_dir.iterdir()):
        raise ValueError(f"Data directory contains no scan folders: {data_dir}")

    return str(data_dir)


def validate_feature_files(scan_list):
    missing_features = [
        str(Path(scan) / "fix" / "features.csv")
        for scan in scan_list
        if not (Path(scan) / "fix" / "features.csv").is_file()
    ]
    if missing_features:
        raise ValueError(
            "Missing FIX feature files:\n" + "\n".join(missing_features)
        )

    return scan_list


def validate_select_label(select_label):
    if not 0 <= select_label <= 100:
        raise ValueError(
            f"select_label must be between 0 and 100, got {select_label}"
        )

    return select_label


def validate_training_set(fix_dir):
    training_set_file = Path(fix_dir) / "model" / "training_set.txt"
    if not training_set_file.is_file():
        raise ValueError(
            f"Training set file does not exist: {training_set_file}"
        )

    scan_list = [
        line
        for line in training_set_file.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not scan_list:
        raise ValueError(f"Training set file is empty: {training_set_file}")

    return scan_list


def validate_training_scans(data_dir, scan_list):
    data_dir = Path(data_dir).resolve()
    validated_scans = []
    for scan in scan_list:
        scan_path = Path(scan)
        expected_scan_path = data_dir / scan_path.name
        if scan_path.is_absolute() and scan_path.resolve() != expected_scan_path:
            raise ValueError(
                f"Training scan is not in the data directory: {scan_path}"
            )
        if not expected_scan_path.is_dir():
            raise ValueError(
                f"Training scan does not exist: {expected_scan_path}"
            )
        validated_scans.append(str(expected_scan_path))

    return validated_scans


def validate_hand_label_files(scan_list):
    missing_labels = [
        str(Path(scan) / "hand_labels_noise.txt")
        for scan in scan_list
        if not (Path(scan) / "hand_labels_noise.txt").is_file()
    ]
    if missing_labels:
        raise ValueError(
            "Missing hand label files:\n" + "\n".join(missing_labels)
        )

    return scan_list


def pyfix_model_exists(fix_model):
    model_file = files("pyfix").joinpath(
        "resources",
        "models",
        f"{fix_model}.pyfix_model",
    )
    return model_file.is_file()


def validate_trained_model(fix_dir, fix_model):
    model_file = Path(fix_dir) / "model" / f"{fix_model}.pyfix_model"
    if not model_file.is_file():
        raise ValueError(f"Trained model was not created: {model_file}")

    return str(model_file)


def validate_classification_outputs(scan_list, fix_model, threshold):
    artifact_filename = f"fix4melview_{fix_model}_thr{threshold}.txt"
    missing_artifact_lists = [
        str(Path(scan) / artifact_filename)
        for scan in scan_list
        if not (Path(scan) / artifact_filename).is_file()
    ]
    if missing_artifact_lists:
        raise ValueError(
            "Missing classification artifact lists:\n"
            + "\n".join(missing_artifact_lists)
        )

    return scan_list


def validate_cleanup_files(scan_list):
    required_files = [
        "filtered_func_data_clean.nii.gz",
        "fd.csv",
    ]
    missing_files = [
        str(Path(scan) / required_file)
        for scan in scan_list
        for required_file in required_files
        if not (Path(scan) / required_file).is_file()
    ]
    if missing_files:
        raise ValueError(
            "Missing cleanup files:\n" + "\n".join(missing_files)
        )

    return scan_list


def validate_analysis_files(scan_list):
    missing_files = [
        str(Path(scan) / "filtered_func_data_clean_smoothed.nii.gz")
        for scan in scan_list
        if not (
            Path(scan) / "filtered_func_data_clean_smoothed.nii.gz"
        ).is_file()
    ]
    if missing_files:
        raise ValueError(
            "Missing cleaned and smoothed files:\n"
            + "\n".join(missing_files)
        )

    return scan_list


def validate_analysis_ica_files(scan_list):
    missing_files = [
        str(Path(scan) / "ica.nii.gz")
        for scan in scan_list
        if not (Path(scan) / "ica.nii.gz").is_file()
    ]
    if missing_files:
        raise ValueError(
            "Missing per-scan ICA maps:\n" + "\n".join(missing_files)
        )

    return scan_list


def validate_analysis_mask_files(scan_list):
    missing_files = [
        str(Path(scan) / "mask.nii.gz")
        for scan in scan_list
        if not (Path(scan) / "mask.nii.gz").is_file()
    ]
    if missing_files:
        raise ValueError(
            "Missing per-scan masks:\n" + "\n".join(missing_files)
        )

    return scan_list


def validate_analysis_image(image_file, image_type):
    image_file = Path(image_file)
    if not image_file.is_file():
        raise ValueError(f"{image_type} does not exist: {image_file}")

    return str(image_file.resolve())


def validate_seed_files(seed):
    if not seed.get("apply", False):
        return []

    seed_files = seed.get("seeds", [])
    if not seed_files:
        raise ValueError("Seed analysis requires at least one seed image")

    return [
        validate_analysis_image(seed_file, "Seed image")
        for seed_file in seed_files
    ]


def validate_extract_threshold(extract_threshold):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([pz])", extract_threshold)
    if match is None:
        raise ValueError(
            "extract_threshold must be a percentage such as '90p' or "
            "a z-score such as '7.5z'"
        )

    value = float(match.group(1))
    threshold_type = match.group(2)
    if threshold_type == "p" and not 0 <= value <= 100:
        raise ValueError(
            f"Percentage extract_threshold must be between 0 and 100, "
            f"got {extract_threshold}"
        )

    return extract_threshold


def validate_dice_threshold(dice_threshold):
    if not 0 <= dice_threshold <= 100:
        raise ValueError(
            f"dice_threshold must be between 0 and 100, got {dice_threshold}"
        )

    return dice_threshold


def get_pending_dual_regression_scans(fix_dir, scan_list):
    results_dir = Path(fix_dir) / "results"
    pending_scans = []
    for scan in scan_list:
        dr_dir = results_dir / Path(scan).name / "dr"
        beta_file = dr_dir / "dr_stage2_subject00000.nii.gz"
        z_file = dr_dir / "dr_stage2_subject00000_Z.nii.gz"
        if not (beta_file.is_file() and z_file.is_file()):
            pending_scans.append(scan)

    return pending_scans
