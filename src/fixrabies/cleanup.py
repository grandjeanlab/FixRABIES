import logging
from pathlib import Path

logger = logging.getLogger("fixrabies.cleanup")


def run_cleanup(args):
    from fixrabies.interfaces.validate_inputs import (
        validate_classification_outputs,
        validate_data_directory,
        validate_feature_files,
        validate_select_label,
        validate_trained_model,
    )
    from fixrabies.workflows.cleanup_wf import create_cleanup_workflow

    logger.info("Starting cleanup")
    logger.debug("Fix directory: %s", args.fix_dir)
    logger.debug("Model name: %s", args.fix_model)
    logger.debug("Threshold: %s", args.threshold)

    data_dir = Path(validate_data_directory(args.fix_dir))
    scan_list = sorted(
        str(path)
        for path in data_dir.iterdir()
        if path.is_dir()
    )
    validate_feature_files(scan_list)
    validate_trained_model(args.fix_dir, args.fix_model)
    threshold = validate_select_label(args.threshold)
    validate_classification_outputs(scan_list, args.fix_model, threshold)

    workflow = create_cleanup_workflow(
        fix_dir=args.fix_dir,
        scan_list=scan_list,
        fix_model=args.fix_model,
        threshold=threshold,
        motionhp=args.motionhp,
        aggressive=args.aggressive,
        highpass=args.highpass,
    )
    nipype_args = args.nipype.copy()
    plugin = nipype_args.pop("plugin")
    workflow.run(plugin=plugin, plugin_args=nipype_args)

    logger.info("Finished cleanup")
