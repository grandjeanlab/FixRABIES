import logging
import re
from pathlib import Path

logger = logging.getLogger("fixrabies.review")

NOISE_LIST_PATTERN = re.compile(r"^\s*\[(?:\d+\s*(?:,\s*\d+\s*)*)?\]\s*$")


def keep_noise_component_lines(scan_list, fix_model, threshold):
    artifact_filename = f"fix4melview_{fix_model}_thr{threshold}.txt"
    for scan in scan_list:
        artifact_list = Path(scan) / artifact_filename
        component_lines = [
            line
            for line in artifact_list.read_text(encoding="utf-8").splitlines()
            if NOISE_LIST_PATTERN.fullmatch(line)
        ]
        if len(component_lines) != 1:
            raise ValueError(f"Expected exactly one component list in: {artifact_list}")
        artifact_list.write_text(
            component_lines[0].strip() + "\n",
            encoding="utf-8",
        )


def rebuild_classification_outputs(scans):
    for scan in scans.values():
        noise_list = Path(str(scan["noise_list"]))
        component_lines = [
            line.strip()
            for line in noise_list.read_text(encoding="utf-8").splitlines()
            if NOISE_LIST_PATTERN.fullmatch(line)
        ]
        if len(component_lines) != 1:
            raise ValueError(f"Expected exactly one component list in: {noise_list}")
        noise_components = {
            int(component)
            for component in component_lines[0][1:-1].split(",")
            if component.strip()
        }
        n_ic = int(scan["n_ic"])
        invalid_components = sorted(
            component
            for component in noise_components
            if component < 1 or component > n_ic
        )
        if invalid_components:
            raise ValueError(
                f"Invalid noise components in {noise_list}: " f"{invalid_components}"
            )

        classification_lines = [
            "filtered_func_data.ica",
            *(
                f"{component}, "
                f"{'Noise, True' if component in noise_components else 'Signal, False'}, "
                "1"
                for component in range(1, n_ic + 1)
            ),
            component_lines[0],
        ]
        noise_list.write_text(
            "\n".join(classification_lines) + "\n",
            encoding="utf-8",
        )


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
    scan_list = sorted(str(path) for path in data_dir.iterdir() if path.is_dir())
    validate_feature_files(scan_list)
    threshold = validate_select_label(args.threshold)
    validate_classification_outputs(scan_list, args.fix_model, threshold)
    keep_noise_component_lines(scan_list, args.fix_model, threshold)

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
        node for node in execution_graph.nodes() if node.name == "review_scan_info"
    )

    from fixrabies.gui import launch_component_labeler

    launch_component_labeler(scan_info_node.result.outputs.scans)
    rebuild_classification_outputs(scan_info_node.result.outputs.scans)

    logger.info("Finished review")
