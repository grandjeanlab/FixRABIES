from pathlib import Path

from nipype import Node, Workflow
from nipype.interfaces.utility import Function

from fixrabies.nodes.analyze_group import aggregate_group_analysis


def create_analyze_group_workflow(fix_dir, scan_names, dr, seed):
    aggregate = Node(
        Function(
            input_names=["fix_dir", "scan_names", "dr", "seed"],
            output_names=["group_dir"],
            function=aggregate_group_analysis,
        ),
        name="aggregate",
    )
    aggregate.inputs.fix_dir = str(Path(fix_dir).expanduser().resolve())
    aggregate.inputs.scan_names = scan_names
    aggregate.inputs.dr = dr
    aggregate.inputs.seed = seed

    workflow = Workflow(
        name="analyze_group",
        base_dir=str(Path(fix_dir).expanduser().resolve() / "workflows"),
    )
    workflow.add_nodes([aggregate])
    return workflow
