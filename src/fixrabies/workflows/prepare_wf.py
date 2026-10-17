from pathlib import Path

from nipype import Node, Workflow
from nipype.interfaces.utility import Function, IdentityInterface
from fixrabies.nodes.prepare_node import get_bold_file, get_scan_info, prepare_fix_scan
from fixrabies.nodes.denoise_node import denoise_input_node, denoise_tofile_node

def create_prepare_fix_workflow(scans, fix_dir, patch_denoise):

    scan_ids = sorted(scans.keys())

    infosource = Node(
        IdentityInterface(fields=["scan_id"]),
        name="infosource",
    )

    infosource.iterables = [("scan_id", scan_ids)]

    lookup = Node(
        Function(
            input_names=[
                "scan_id",
                "scans",
            ],
            output_names=[
                "scan_info",
            ],
            function=get_scan_info,
        ),
        name="lookup",
    )

    lookup.inputs.scans = scans

    bold_file = Node(
        Function(
            input_names=[
                "scan_info",
            ],
            output_names=[
                "bold_file",
            ],
            function=get_bold_file,
        ),
        name="bold_file",
    )

    prepare = Node(
        Function(
            input_names=[
                "scan_id",
                "scan_info",
                "fix_dir",
                "patch_denoise",
            ],
            output_names=[
                "outdir",
            ],
            function=prepare_fix_scan,
        ),
        name="prepare_fix",
    )

    prepare.inputs.fix_dir = str(fix_dir)
    prepare.inputs.patch_denoise = patch_denoise

    wf = Workflow(
        name="prepare",
        base_dir=str(Path(fix_dir) / "workflows"),
    )

    wf.connect(
        infosource,
        "scan_id",
        lookup,
        "scan_id",
    )

    wf.connect(
        infosource,
        "scan_id",
        prepare,
        "scan_id",
    )

    wf.connect(
        lookup,
        "scan_info",
        prepare,
        "scan_info",
    )

    if patch_denoise.get("apply", False):
        denoise = denoise_input_node(patch_denoise)
        save_denoised = denoise_tofile_node()

        wf.connect(
            lookup,
            "scan_info",
            bold_file,
            "scan_info",
        )

        wf.connect(
            bold_file,
            "bold_file",
            denoise,
            "in_mag",
        )

        wf.connect(
            denoise,
            "denoised_file",
            save_denoised,
            "input_file",
        )

        wf.connect(
            prepare,
            "outdir",
            save_denoised,
            "scan_dir",
        )

    return wf
