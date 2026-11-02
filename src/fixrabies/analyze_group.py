import logging
from pathlib import Path

logger = logging.getLogger("fixrabies.analyze_group")


def run_analyze_group(args):
    from fixrabies.interfaces.validate_inputs import (
        validate_data_directory,
        validate_group_analysis_outputs,
    )
    from fixrabies.workflows.analyze_group_wf import (
        create_analyze_group_workflow,
    )

    logger.info("Starting group analysis")
    data_dir = Path(validate_data_directory(args.fix_dir))
    scan_names = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    validate_group_analysis_outputs(
        args.fix_dir,
        scan_names,
        args.dr,
        args.seed,
    )

    workflow = create_analyze_group_workflow(
        fix_dir=args.fix_dir,
        scan_names=scan_names,
        dr=args.dr,
        seed=args.seed,
    )
    nipype_args = args.nipype.copy()
    plugin = nipype_args.pop("plugin")
    workflow.run(plugin=plugin, plugin_args=nipype_args)
    logger.info("Finished group analysis")
