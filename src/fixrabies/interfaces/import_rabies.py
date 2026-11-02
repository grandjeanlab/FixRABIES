import re
from pathlib import Path


def import_rabies(preprocess_dir, confound_dir):
    """
    Returns

    {
        "sub-001_ses-01_run-1": {
            "clean_file": "...",
            "mask": "..."
        }
    }
    """

    preprocess_dir = Path(preprocess_dir).expanduser().resolve()
    confound_dir = Path(confound_dir).expanduser().resolve()

    if not preprocess_dir.is_dir():
        raise ValueError(f"Preprocessing directory does not exist: {preprocess_dir}")
    if not confound_dir.is_dir():
        raise ValueError(f"Confound directory does not exist: {confound_dir}")

    clean_files = sorted(
        confound_dir.glob(
            "confound_correction_datasink/cleaned_timeseries/_split_name*/*_cleaned.nii*"
        )
    )

    if not clean_files:
        raise ValueError(f"No cleaned BOLD files found in {confound_dir}")

    scans = {}

    for clean_file in clean_files:
        text = str(clean_file)

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
        if not scan_id:
            raise ValueError(f"Could not determine scan entities from: {clean_file}")
        if scan_id in scans:
            raise ValueError(f"Duplicate scan identifier: {scan_id}")

        scans[scan_id] = {
            "clean_file": str(clean_file),
            "mask": str(
                preprocess_dir
                / "bold_datasink"
                / "commonspace_mask"
                / "commonspace_template_brain_mask_resampled.nii.gz"
            ),
        }

    return scans


def validate_scans(scans):
    """
    Validate that all files referenced in the scans dict exist.

    Raises ValueError if a required file is missing.
    """

    required_keys = [
        "clean_file",
        "mask",
    ]

    for scan_id, scan in scans.items():

        for key in required_keys:
            if key not in scan:
                raise ValueError(f"{scan_id}: missing key '{key}'")
            if scan[key] is None:
                raise ValueError(f"{scan_id}: '{key}' is None")

            path = Path(scan[key])

            if not path.is_file():
                raise ValueError(f"{scan_id}: '{key}' does not exist\n  {path}")

    return True
