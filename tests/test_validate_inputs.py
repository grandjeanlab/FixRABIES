from pathlib import Path

from fixrabies.interfaces.validate_inputs import (
    validate_data_directory,
    validate_feature_files,
    validate_select_label,
)
from fixrabies.label import get_matching_training_set


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
