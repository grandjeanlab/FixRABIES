import logging

logger = logging.getLogger("fixrabies.prepare")


def run_prepare(args):
    logger.info("Starting prepare")

    logger.debug(
        "Preprocess directory: %s",
        args.preprocess_dir,
    )
    logger.debug(
        "Fix directory: %s",
        args.fix_dir,
    )

    logger.info("Finished prepare")
