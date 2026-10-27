from pathlib import Path

from nipype import Node, Workflow
from nipype.interfaces.utility import Function, IdentityInterface

from fixrabies.nodes.cleanup import (
    cleanup_node,
    get_artifact_list_file,
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
