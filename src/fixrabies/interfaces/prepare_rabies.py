import re
from pathlib import Path


def prepare_rabies(preprocess_dir):
    """
    Returns

    {
        "sub-001_ses-01_run-1": {
            "bold_file": "...",
            "fd": "...",
            "motion": "...",
            "bg": "...",
            "highres": "...",
            "mask": "..."
        }
    }
    """

    preprocess_dir = Path(preprocess_dir).expanduser().resolve()

    assert (
        preprocess_dir.exists()
    ), f"Preprocessing directory does not exist: {preprocess_dir}"

    bold_files = sorted(
        preprocess_dir.glob(
            "bold_datasink/commonspace_bold/_scan_info*/_run_*/*_resampled.nii*"
        )
    )

    assert len(bold_files) > 0, f"No bold_resampled files found in {preprocess_dir}"

    scans = {}

    for bold_file in bold_files:
        text = str(bold_file)

        entities = []

        for pattern in (
            r"sub-[^_/]+",
            r"ses-[^_/]+",
            r"task-[^_/]+",
            r"run-[^_/]+",
        ):
            match = re.search(pattern, text)
            if match:
                entities.append(match.group(0))

        scan_id = "_".join(entities)

        scan_info_dir = bold_file.parent.parent.name
        run_dir = bold_file.parent.name

        fd_matches = list(
            preprocess_dir.glob(
                f"motion_datasink/FD_csv/{scan_info_dir}/{run_dir}/*.csv"
            )
        )

        motion_matches = list(
            preprocess_dir.glob(
                f"motion_datasink/motion_params_csv/{scan_info_dir}/{run_dir}/*.csv"
            )
        )

        tr = get_tr(preprocess_dir)

        scans[scan_id] = {
            "bold_file": str(bold_file),
            "fd": str(fd_matches[0]) if fd_matches else None,
            "motion": str(motion_matches[0]) if motion_matches else None,
            "bg": str(
                preprocess_dir
                / "bold_datasink"
                / "commonspace_resampled_template"
                / "commonspace_template.nii.gz"
            ),
            "highres": str(
                preprocess_dir
                / "bold_datasink"
                / "commonspace_resampled_template"
                / "commonspace_template.nii.gz"
            ),
            "mask": str(
                preprocess_dir
                / "bold_datasink"
                / "commonspace_mask"
                / "commonspace_template_brain_mask_resampled.nii.gz"
            ),
            "tr": tr,
        }

    return scans


def get_tr(preprocess_dir):
    """
    Read TR from:

        {preprocess_dir}/rabies_preprocess.log

    looking for lines such as:

        TR = 1.2
        -> TR = 1.2

    Returns
    -------
    float
    """

    log_file = Path(preprocess_dir) / "rabies_preprocess.log"

    assert log_file.exists(), f"Missing RABIES log file: {log_file}"

    with open(log_file) as f:
        for line in f:
            match = re.search(r"->\s*TR\s*=\s*([0-9.]+)", line)
            if match:
                return float(match.group(1))

    raise ValueError(f"Could not find TR in {log_file}")


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
