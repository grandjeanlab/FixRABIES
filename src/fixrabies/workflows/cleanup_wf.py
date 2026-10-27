from pathlib import Path

from nipype import Node, Workflow
from nipype.interfaces.utility import Function, IdentityInterface

from fixrabies.nodes.cleanup import (
    cleanup_node,
    estimate_dvars,
    get_cleaned_file,
    get_artifact_list_file,
    get_fd_file,
    get_kept_volumes,
    get_mean_fd,
    save_exclusion_file,
    save_censored_image,
    smooth_image,
)


def create_cleanup_workflow(
    fix_dir,
    scan_list,
    fix_model,
    threshold,
    motionhp,
    aggressive,
    highpass,
):
    infosource = Node(
        IdentityInterface(fields=["scan_dir"]),
        name="infosource",
    )
    infosource.iterables = [("scan_dir", scan_list)]

    artifact_list = Node(
        Function(
            input_names=["scan_dir", "fix_model", "threshold"],
            output_names=["artifact_list_file"],
            function=get_artifact_list_file,
        ),
        name="artifact_list",
    )
    artifact_list.inputs.fix_model = fix_model
    artifact_list.inputs.threshold = threshold

    cleanup = cleanup_node(motionhp, aggressive, highpass)

    workflow = Workflow(
        name="cleanup",
        base_dir=str(Path(fix_dir) / "workflows"),
    )
    workflow.connect(
        infosource,
        "scan_dir",
        artifact_list,
        "scan_dir",
    )
    workflow.connect(
        artifact_list,
        "artifact_list_file",
        cleanup,
        "artifact_list_file",
    )

    return workflow


def create_post_cleanup_workflow(
    fix_dir,
    scan_list,
    dvars_threshold,
    fd_threshold,
    smooth,
):
    infosource = Node(
        IdentityInterface(fields=["scan_dir"]),
        name="infosource",
    )
    infosource.iterables = [("scan_dir", scan_list)]

    cleaned_file = Node(
        Function(
            input_names=["scan_dir"],
            output_names=["cleaned_file"],
            function=get_cleaned_file,
        ),
        name="cleaned_file",
    )
    fd_file = Node(
        Function(
            input_names=["scan_dir"],
            output_names=["fd_file"],
            function=get_fd_file,
        ),
        name="fd_file",
    )
    dvars = Node(
        Function(
            input_names=["cleaned_file"],
            output_names=["dvars"],
            function=estimate_dvars,
        ),
        name="dvars",
    )
    mean_fd = Node(
        Function(
            input_names=["fd_file"],
            output_names=["mean_fd"],
            function=get_mean_fd,
        ),
        name="mean_fd",
    )
    kept_volumes = Node(
        Function(
            input_names=[
                "dvars",
                "mean_fd",
                "dvars_threshold",
                "fd_threshold",
            ],
            output_names=["kept_volumes"],
            function=get_kept_volumes,
        ),
        name="kept_volumes",
    )
    kept_volumes.inputs.dvars_threshold = dvars_threshold
    kept_volumes.inputs.fd_threshold = fd_threshold

    exclusion_file = Node(
        Function(
            input_names=[
                "dvars",
                "mean_fd",
                "dvars_threshold",
                "fd_threshold",
                "scan_dir",
            ],
            output_names=["exclusion_file"],
            function=save_exclusion_file,
        ),
        name="exclusion_file",
    )
    exclusion_file.inputs.dvars_threshold = dvars_threshold
    exclusion_file.inputs.fd_threshold = fd_threshold

    smoothed_image = Node(
        Function(
            input_names=["cleaned_file", "smooth"],
            output_names=["smoothed_image"],
            function=smooth_image,
        ),
        name="smoothed_image",
    )
    smoothed_image.inputs.smooth = smooth
    censored_image = Node(
        Function(
            input_names=[
                "smoothed_image",
                "kept_volumes",
                "scan_dir",
            ],
            output_names=["censored_file"],
            function=save_censored_image,
        ),
        name="censored_image",
    )

    workflow = Workflow(
        name="post_cleanup",
        base_dir=str(Path(fix_dir) / "workflows"),
    )
    workflow.connect(
        infosource,
        "scan_dir",
        cleaned_file,
        "scan_dir",
    )
    workflow.connect(
        infosource,
        "scan_dir",
        fd_file,
        "scan_dir",
    )
    workflow.connect(
        cleaned_file,
        "cleaned_file",
        dvars,
        "cleaned_file",
    )
    workflow.connect(
        fd_file,
        "fd_file",
        mean_fd,
        "fd_file",
    )
    workflow.connect(
        dvars,
        "dvars",
        kept_volumes,
        "dvars",
    )
    workflow.connect(
        mean_fd,
        "mean_fd",
        kept_volumes,
        "mean_fd",
    )
    workflow.connect(
        dvars,
        "dvars",
        exclusion_file,
        "dvars",
    )
    workflow.connect(
        mean_fd,
        "mean_fd",
        exclusion_file,
        "mean_fd",
    )
    workflow.connect(
        infosource,
        "scan_dir",
        exclusion_file,
        "scan_dir",
    )
    workflow.connect(
        cleaned_file,
        "cleaned_file",
        smoothed_image,
        "cleaned_file",
    )
    workflow.connect(
        smoothed_image,
        "smoothed_image",
        censored_image,
        "smoothed_image",
    )
    workflow.connect(
        kept_volumes,
        "kept_volumes",
        censored_image,
        "kept_volumes",
    )
    workflow.connect(
        infosource,
        "scan_dir",
        censored_image,
        "scan_dir",
    )

    return workflow
