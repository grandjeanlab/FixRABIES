import logging

logger = logging.getLogger("fixrabies.prepare")


def run_prepare(args):
    from fixrabies.interfaces.prepare_rabies import (
        prepare_rabies,
        validate_scans,
    )
    from fixrabies.workflows.prepare_wf import create_prepare_fix_workflow

    logger.info("Starting prepare")

    logger.debug(
        "Preprocess directory: %s",
        args.preprocess_dir,
    )
    logger.debug(
        "Fix directory: %s",
        args.fix_dir,
    )

    if args.preprocess_software != "rabies":
        raise ValueError(
            f"Unsupported preprocessing software: {args.preprocess_software}"
        )

    scans = prepare_rabies(args.preprocess_dir)
    validate_scans(scans)

    workflow = create_prepare_fix_workflow(
        scans=scans,
        fix_dir=args.fix_dir,
        patch_denoise=args.patch_denoise,
        species=args.species,
    )
    nipype_args = args.nipype.copy()
    plugin = nipype_args.pop("plugin")
    workflow.run(plugin=plugin, plugin_args=nipype_args)

    logger.info("Finished prepare")
