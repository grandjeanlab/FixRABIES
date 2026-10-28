import logging
import math
from pathlib import Path

logger = logging.getLogger("fixrabies.label")


def get_matching_training_set(fix_dir, scan_count, select_label):
    expected_count = (
        1
        if select_label == 0
        else math.ceil(scan_count * select_label / 100)
    )
    training_set_file = Path(fix_dir) / "model" / "training_set.txt"
    if not training_set_file.is_file():
        return None

    selected_scans = [
        line
        for line in training_set_file.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(selected_scans) == expected_count:
        return selected_scans

    return None


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
    selected_scans = get_matching_training_set(
        args.fix_dir,
        len(scan_list),
        select_label,
    )
    if selected_scans is not None:
        logger.info("Using existing training set")

    from fixrabies.workflows.label_wf import create_label_workflow

    workflow = create_label_workflow(
        fix_dir=args.fix_dir,
        scan_list=scan_list,
        select_label=select_label,
        selected_scans=selected_scans,
    )
    nipype_args = args.nipype.copy()
    plugin = nipype_args.pop("plugin")
    execution_graph = workflow.run(plugin=plugin, plugin_args=nipype_args)
    scan_info_node = next(
        node
        for node in execution_graph.nodes()
        if node.name == "selected_scan_info"
    )

    from fixrabies.gui import launch_component_labeler

    launch_component_labeler(scan_info_node.result.outputs.scans)

    logger.info("Finished label")
