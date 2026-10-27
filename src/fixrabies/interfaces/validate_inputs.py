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
