from pathlib import Path
import shutil

from nipype import Node
from nipype.interfaces.utility import Function

from patch_denoise.bindings.nipype import (
    PatchDenoise,
    NoiseStdMap,
)


def save_filtered_func_data(
    input_file,
    scan_dir,
):
    """
    Save the denoised output into the
    FIX-compatible location.
    """

    output_file = (
        Path(scan_dir)
        / "filtered_func_data.nii.gz"
    )

    shutil.copyfile(
        input_file,
        output_file,
    )

    return str(output_file)


def denoise_input_node(
    patch_denoise,
):

    node = Node(
        PatchDenoise(),
        name="patch_denoise",
    )

    node.inputs.method = (
        patch_denoise["method"]
    )

    node.inputs.patch_shape = (
        patch_denoise["patch_shape"]
    )

    node.inputs.patch_overlap = (
        patch_denoise["patch_overlap"]
    )

    node.inputs.mask_threshold = 1

    node.inputs.recombination = (
        "weighted"
    )

    return node



def denoise_tofile_node():

    return Node(
        Function(
            input_names=[
                "input_file",
                "scan_dir",
            ],
            output_names=[
                "filtered_func_data",
            ],
            function=save_filtered_func_data,
        ),
        name="save_filtered_func",
    )

