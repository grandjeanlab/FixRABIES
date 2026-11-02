import logging

logger = logging.getLogger("fixrabies.import")


def run_import(args):
    from fixrabies.interfaces.import_rabies import (
        import_rabies,
        validate_scans,
    )
    from fixrabies.workflows.import_wf import create_import_workflow

    logger.info("Starting import")

    logger.debug(
        "Preprocess directory: %s",
        args.preprocess_dir,
    )
    logger.debug(
        "Confound directory: %s",
        args.confound_dir,
    )
    logger.debug(
        "Fix directory: %s",
        args.fix_dir,
    )
    logger.debug(
        "Species: %s",
        args.species,
    )

    if args.preprocess_software != "rabies":
        raise ValueError(
            f"Unsupported preprocessing software: {args.preprocess_software}"
        )

    scans = import_rabies(args.preprocess_dir, args.confound_dir)
    validate_scans(scans)

    workflow = create_import_workflow(
        scans=scans,
        fix_dir=args.fix_dir,
        species=args.species,
    )
    nipype_args = args.nipype.copy()
    plugin = nipype_args.pop("plugin")
    workflow.run(plugin=plugin, plugin_args=nipype_args)

    logger.info("Finished import")
