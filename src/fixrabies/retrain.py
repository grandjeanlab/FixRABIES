import logging
import re
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger("fixrabies.retrain")

NOISE_LIST_PATTERN = re.compile(r"^\s*\[(?:\d+\s*(?:,\s*\d+\s*)*)?\]\s*$")


def save_reviewed_labels(scan_list, old_fix_model, old_threshold):
    artifact_filename = f"fix4melview_{old_fix_model}_thr{old_threshold}.txt"
    for scan in scan_list:
        artifact_list = Path(scan) / artifact_filename
        component_lines = [
            line.strip()
            for line in artifact_list.read_text(encoding="utf-8").splitlines()
            if NOISE_LIST_PATTERN.fullmatch(line)
        ]
        if len(component_lines) != 1:
            raise ValueError(f"Expected exactly one component list in: {artifact_list}")
        (Path(scan) / "hand_labels_noise.txt").write_text(
            component_lines[0] + "\n",
            encoding="utf-8",
        )


def save_training_set(fix_dir, scan_list):
    training_set_file = (
        Path(fix_dir).expanduser().resolve() / "model" / "training_set.txt"
    )
    training_set_file.parent.mkdir(parents=True, exist_ok=True)
    training_set_file.write_text(
        "\n".join(scan_list) + "\n",
        encoding="utf-8",
    )
    return str(training_set_file)


def run_retrain(args):
    from fixrabies.interfaces.validate_inputs import (
        validate_classification_outputs,
        validate_data_directory,
        validate_feature_files,
        validate_hand_label_files,
        validate_select_label,
        validate_training_scans,
    )
    from fixrabies.train import run_train

    logger.info("Starting retraining")
    if args.old_fix_model == args.new_fix_model:
        raise ValueError("old_fix_model and new_fix_model must differ")
    if Path(args.new_fix_model).name != args.new_fix_model:
        raise ValueError("new_fix_model must be a model name, not a path")

    data_dir = validate_data_directory(args.fix_dir)
    if args.scan_list is None:
        scan_list = [str(path) for path in Path(data_dir).iterdir() if path.is_dir()]
    else:
        if not isinstance(args.scan_list, list) or not all(
            isinstance(scan, str) for scan in args.scan_list
        ):
            raise ValueError("scan_list must be a JSON array of scan paths")
        scan_list = args.scan_list
    scan_list = validate_training_scans(data_dir, scan_list)

    threshold = validate_select_label(args.old_threshold)
    validate_feature_files(scan_list)
    validate_classification_outputs(
        scan_list,
        args.old_fix_model,
        threshold,
    )
    save_reviewed_labels(scan_list, args.old_fix_model, threshold)
    validate_hand_label_files(scan_list)
    save_training_set(args.fix_dir, scan_list)
    run_train(
        SimpleNamespace(
            fix_dir=args.fix_dir,
            fix_model=args.new_fix_model,
            species=args.species,
        )
    )

    logger.info("Finished retraining")
