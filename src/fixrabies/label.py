import logging
from pathlib import Path

logger = logging.getLogger("fixrabies.label")


def run_label(args):
    from fixrabies.interfaces.validate_inputs import (
        validate_data_directory,
        validate_feature_files,
        validate_select_label,
    )

    logger.info("Starting label")
    logger.debug("Fix directory: %s", args.fix_dir)
    logger.debug("Label selection: %s%%", args.select_label)

    data_dir = Path(validate_data_directory(args.fix_dir))
    scan_list = sorted(
        str(path)
        for path in data_dir.iterdir()
        if path.is_dir()
    )
    validate_feature_files(scan_list)
    select_label = validate_select_label(args.select_label)

    from fixrabies.workflows.label_wf import create_label_workflow

    workflow = create_label_workflow(
        fix_dir=args.fix_dir,
        scan_list=scan_list,
        select_label=select_label,
    )
    nipype_args = args.nipype.copy()
    plugin = nipype_args.pop("plugin")
    workflow.run(plugin=plugin, plugin_args=nipype_args)

    logger.info("Finished label")
