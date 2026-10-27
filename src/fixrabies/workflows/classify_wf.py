from pathlib import Path

from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface

from fixrabies.nodes.classify import classifier_node


def create_classify_workflow(
    fix_dir,
    scan_list,
    trained_wts_file,
    thresh,
):
    infosource = Node(
        IdentityInterface(fields=["scan_dir"]),
        name="infosource",
    )
    infosource.iterables = [("scan_dir", scan_list)]

    classifier = classifier_node(
        trained_wts_file=trained_wts_file,
        thresh=thresh,
    )

    workflow = Workflow(
        name="classify",
        base_dir=str(Path(fix_dir) / "workflows"),
    )
    workflow.connect(
        infosource,
        "scan_dir",
        classifier,
        "mel_ica",
    )

    return workflow
