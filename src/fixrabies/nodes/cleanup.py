from nipype import Node

from fixrabies.interfaces.pyfix import Cleanup


def cleanup_node(motionhp, aggressive, highpass):
    node = Node(
        Cleanup(),
        name="cleanup",
    )
    node.inputs.motionhp = motionhp
    node.inputs.aggressive = aggressive
    if highpass is not None:
        node.inputs.highpass = highpass
    return node


def get_artifact_list_file(scan_dir, fix_model, threshold):
    from pathlib import Path

    return str(
        Path(scan_dir)
        / f"fix4melview_{fix_model}_thr{threshold}.txt"
    )
