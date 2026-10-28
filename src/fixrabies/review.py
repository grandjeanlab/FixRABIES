import logging
from pathlib import Path

logger = logging.getLogger("fixrabies.review")


def run_review(args):
    from fixrabies.interfaces.validate_inputs import (
        validate_classification_outputs,
        validate_data_directory,
        validate_feature_files,
        validate_select_label,
    )
    from fixrabies.workflows.review_wf import create_review_workflow

    logger.info("Starting review")
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
    threshold = validate_select_label(args.threshold)
    validate_classification_outputs(scan_list, args.fix_model, threshold)

    workflow = create_review_workflow(
        fix_dir=args.fix_dir,
        scan_list=scan_list,
        fix_model=args.fix_model,
        threshold=threshold,
    )
    nipype_args = args.nipype.copy()
    plugin = nipype_args.pop("plugin")
    execution_graph = workflow.run(plugin=plugin, plugin_args=nipype_args)
    scan_info_node = next(
        node
        for node in execution_graph.nodes()
        if node.name == "review_scan_info"
    )

    from fixrabies.gui import launch_component_labeler

    launch_component_labeler(scan_info_node.result.outputs.scans)

    logger.info("Finished review")
