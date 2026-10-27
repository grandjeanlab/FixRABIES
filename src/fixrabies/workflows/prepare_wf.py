from pathlib import Path

from nipype import Node, Workflow
from nipype.interfaces.fsl.fix import FeatureExtractor
from nipype.interfaces.utility import Function, IdentityInterface
from fixrabies.nodes.prepare_node import get_bold_file, get_scan_info, prepare_fix_scan
from fixrabies.nodes.denoise_node import (
    denoise_input_node,
    denoise_tofile_node,
    noise_std_map_node,
)
from fixrabies.nodes.melodic import (
    get_melodic_metadata,
    get_melodic_out_dir,
    hand_labels_noise_node,
    melodic_node,
)

def create_prepare_fix_workflow(
    scans,
    fix_dir,
    patch_denoise,
    species,
):

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
                "species",
            ],
            output_names=[
                "outdir",
                "filtered_func_data",
            ],
            function=prepare_fix_scan,
        ),
        name="prepare_fix",
    )

    prepare.inputs.fix_dir = str(fix_dir)
    prepare.inputs.patch_denoise = patch_denoise
    prepare.inputs.species = species

    melodic_metadata = Node(
        Function(
            input_names=[
                "scan_info",
            ],
            output_names=[
                "bg_image",
                "tr_sec",
            ],
            function=get_melodic_metadata,
        ),
        name="melodic_metadata",
    )

    melodic_out_dir = Node(
        Function(
            input_names=[
                "scan_dir",
            ],
            output_names=[
                "out_dir",
            ],
            function=get_melodic_out_dir,
        ),
        name="melodic_out_dir",
    )

    melodic = melodic_node()
    hand_labels_noise = hand_labels_noise_node()
    feature_extractor = Node(
        FeatureExtractor(command='fix'),
        name="feature_extractor",
    )

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

    wf.connect(
        lookup,
        "scan_info",
        melodic_metadata,
        "scan_info",
    )

    wf.connect(
        melodic_metadata,
        "bg_image",
        melodic,
        "bg_image",
    )

    wf.connect(
        melodic_metadata,
        "tr_sec",
        melodic,
        "tr_sec",
    )

    wf.connect(
        prepare,
        "outdir",
        melodic_out_dir,
        "scan_dir",
    )

    wf.connect(
        melodic_out_dir,
        "out_dir",
        melodic,
        "out_dir",
    )

    wf.connect(
        melodic,
        "out_dir",
        hand_labels_noise,
        "melodic_out_dir",
    )

    wf.connect(
        prepare,
        "outdir",
        hand_labels_noise,
        "scan_dir",
    )

    wf.connect(
        prepare,
        "outdir",
        feature_extractor,
        "mel_ica",
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

        if patch_denoise["method"] == "nordic":
            noise_std_map = noise_std_map_node()

            wf.connect(
                bold_file,
                "bold_file",
                noise_std_map,
                "noise_map_file",
            )

            wf.connect(
                noise_std_map,
                "noise_std_map",
                denoise,
                "noise_std_map",
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
        wf.connect(
            save_denoised,
            "filtered_func_data",
            melodic,
            "in_files",
        )
    else:
        wf.connect(
            prepare,
            "filtered_func_data",
            melodic,
            "in_files",
        )

    return wf
