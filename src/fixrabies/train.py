import logging
from pathlib import Path

logger = logging.getLogger("fixrabies.train")


def run_train(args):
    from fixrabies.interfaces.pyfix import Training
    from fixrabies.interfaces.validate_inputs import (
        copy_existing_pyfix_model,
        validate_data_directory,
        validate_feature_files,
        validate_hand_label_files,
        validate_trained_model,
        validate_training_scans,
        validate_training_set,
    )

    logger.info("Starting training")
    logger.debug("Fix directory: %s", args.fix_dir)
    logger.debug("Model name: %s", args.fix_model)

    copied_model_name = copy_existing_pyfix_model(args.fix_dir, args.fix_model)
    if copied_model_name is not None:
        logger.info(
            "Copied existing PyFIX model to the working directory: %s",
            copied_model_name,
        )
        return

    data_dir = validate_data_directory(args.fix_dir)
    scan_list = validate_training_set(args.fix_dir)
    scan_list = validate_training_scans(data_dir, scan_list)
    validate_feature_files(scan_list)
    validate_hand_label_files(scan_list)

    model_stem = Path(args.fix_dir) / "model" / args.fix_model
    Training(
        species=args.species,
        model_stem=str(model_stem),
        mel_icas=scan_list,
    ).run()
    validate_trained_model(args.fix_dir, args.fix_model)

    logger.info("Finished training")
