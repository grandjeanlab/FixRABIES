import logging
from pathlib import Path

logger = logging.getLogger("fixrabies.analyze")


def run_analyze(args):
    from fixrabies.interfaces.validate_inputs import (
        validate_analysis_files,
        validate_analysis_ica_files,
        validate_analysis_image,
        validate_analysis_mask_files,
        validate_data_directory,
        validate_dice_threshold,
        validate_extract_threshold,
        get_pending_dual_regression_scans,
        validate_seed_files,
    )
    from fixrabies.workflows.analyze_wf import create_analyze_workflow

    logger.info("Starting analysis")
    logger.debug("Fix directory: %s", args.fix_dir)

    data_dir = Path(validate_data_directory(args.fix_dir))
    scan_list = sorted(
        str(path)
        for path in data_dir.iterdir()
        if path.is_dir()
    )
    validate_analysis_files(scan_list)
    validate_analysis_mask_files(scan_list)
    dr_template = (
        validate_analysis_image(args.DR_template, "Dual regression template")
        if args.DR_template is not None
        else None
    )
    if dr_template is None:
        validate_analysis_ica_files(scan_list)
    seed_files = validate_seed_files(args.seed)
    dice_threshold = validate_dice_threshold(args.dice_threshold)
    extract_threshold = validate_extract_threshold(args.extract_threshold)
    pending_scans = get_pending_dual_regression_scans(
        args.fix_dir,
        scan_list,
    )
    nipype_args = args.nipype.copy()
    plugin = nipype_args.pop("plugin")
    if pending_scans:
        workflow = create_analyze_workflow(
            fix_dir=args.fix_dir,
            scan_list=pending_scans,
            dr_template=dr_template,
            seed_files=seed_files,
            norm=args.norm,
            dice_threshold=dice_threshold,
            extract_threshold=extract_threshold,
            run_dual_regression=True,
        )
        workflow.run(plugin=plugin, plugin_args=nipype_args)

    completed_scans = [
        scan for scan in scan_list if scan not in pending_scans
    ]
    if completed_scans:
        validate_analysis_ica_files(completed_scans)
        outcomes_workflow = create_analyze_workflow(
            fix_dir=args.fix_dir,
            scan_list=completed_scans,
            dr_template=dr_template,
            seed_files=seed_files,
            norm=args.norm,
            dice_threshold=dice_threshold,
            extract_threshold=extract_threshold,
            run_dual_regression=False,
        )
        outcomes_workflow.run(plugin=plugin, plugin_args=nipype_args)

    logger.info("Finished analysis")
