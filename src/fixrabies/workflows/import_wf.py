from pathlib import Path

from nipype import Node, Workflow
from nipype.interfaces.utility import Function, IdentityInterface

from fixrabies.nodes.import_node import import_cleaned_scan
from fixrabies.nodes.prepare_node import get_scan_info


def create_import_workflow(scans, fix_dir, species):
    infosource = Node(
        IdentityInterface(fields=["scan_id"]),
        name="infosource",
    )
    infosource.iterables = [("scan_id", sorted(scans))]

    lookup = Node(
        Function(
            input_names=["scan_id", "scans"],
            output_names=["scan_info"],
            function=get_scan_info,
        ),
        name="lookup",
    )
    lookup.inputs.scans = scans

    import_scan = Node(
        Function(
            input_names=["scan_id", "scan_info", "fix_dir", "species"],
            output_names=["scan_dir"],
            function=import_cleaned_scan,
        ),
        name="import_scan",
    )
    import_scan.inputs.fix_dir = str(Path(fix_dir).expanduser().resolve())
    import_scan.inputs.species = species

    workflow = Workflow(
        name="import",
        base_dir=str(Path(fix_dir).expanduser().resolve() / "workflows"),
    )
    workflow.connect(infosource, "scan_id", lookup, "scan_id")
    workflow.connect(infosource, "scan_id", import_scan, "scan_id")
    workflow.connect(lookup, "scan_info", import_scan, "scan_info")

    return workflow
