import logging
from pathlib import Path

logger = logging.getLogger("fixrabies.train")


def run_train(args):
    from fixrabies.interfaces.validate_inputs import (
        validate_data_directory,
        validate_feature_files,
        validate_hand_label_files,
        validate_training_scans,
        validate_training_set,
        validate_trained_model,
        pyfix_model_exists,
    )
    from fixrabies.interfaces.pyfix import Training

    logger.info("Starting training")
    logger.debug("Fix directory: %s", args.fix_dir)
    logger.debug("Model name: %s", args.fix_model)

    data_dir = validate_data_directory(args.fix_dir)
    scan_list = validate_training_set(args.fix_dir)
    scan_list = validate_training_scans(data_dir, scan_list)
    validate_feature_files(scan_list)
    validate_hand_label_files(scan_list)

    model_name = Path(args.fix_model)
    if model_name.name != args.fix_model:
        raise ValueError("fix_model must be a model name, not a path")
    if pyfix_model_exists(args.fix_model):
        raise ValueError(
            f"FIX model already exists in PyFIX: {args.fix_model}.pyfix_model"
        )

    model_stem = Path(args.fix_dir) / "model" / model_name
    Training(
        species=args.species,
        model_stem=str(model_stem),
        mel_icas=scan_list,
    ).run()
    validate_trained_model(args.fix_dir, args.fix_model)

    logger.info("Finished training")
