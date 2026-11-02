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

    return str(Path(scan_dir) / f"fix4melview_{fix_model}_thr{threshold}.txt")


def get_cleaned_file(scan_dir):
    from pathlib import Path

    return str(Path(scan_dir) / "filtered_func_data_clean.nii.gz")


def get_fd_file(scan_dir):
    from pathlib import Path

    return str(Path(scan_dir) / "fd.csv")


def get_mask_file(scan_dir):
    from pathlib import Path

    return str(Path(scan_dir) / "mask.nii.gz")


def mask_smoothed_image(smoothed_image, mask_file):
    from pathlib import Path

    import nibabel as nib
    import numpy as np

    cleaned_image = (
        nib.load(smoothed_image)
        if isinstance(smoothed_image, (str, Path))
        else smoothed_image
    )
    mask_image = nib.load(mask_file)
    cleaned_data = cleaned_image.get_fdata(dtype="float32")
    mask_data = mask_image.get_fdata() > 0
    if cleaned_data.ndim != 4 or mask_data.ndim != 3:
        raise ValueError("Cleaned image must be 4D and mask image must be 3D")
    if cleaned_data.shape[:3] != mask_data.shape:
        raise ValueError(
            "Cleaned image and mask do not share spatial dimensions: "
            f"{cleaned_data.shape[:3]} and {mask_data.shape}"
        )
    return nib.Nifti1Image(
        cleaned_data * mask_data[..., np.newaxis],
        cleaned_image.affine,
        cleaned_image.header,
    )


def estimate_dvars(cleaned_file):
    import numpy as np
    from nilearn.image import load_img

    image = load_img(cleaned_file)
    data = image.get_fdata(dtype="float32")
    if data.ndim != 4:
        raise ValueError(f"Cleaned image is not 4D: {cleaned_file}")

    temporal_difference = np.diff(data, axis=-1)
    dvars = np.sqrt(np.mean(temporal_difference**2, axis=(0, 1, 2)))
    return [0.0, *dvars.tolist()]


def get_mean_fd(fd_file):
    import pandas as pd

    return pd.read_csv(fd_file)["MeanFD"].tolist()


def get_kept_volumes(
    dvars,
    mean_fd,
    dvars_threshold,
    fd_threshold,
):
    import numpy as np

    dvars = np.asarray(dvars)
    mean_fd = np.asarray(mean_fd)
    if len(dvars) != len(mean_fd):
        raise ValueError(
            "DVARS and MeanFD must have the same number of volumes, got "
            f"{len(dvars)} and {len(mean_fd)}"
        )

    dvars_selection = (
        dvars > dvars_threshold
        if dvars_threshold > 0
        else np.zeros(len(dvars), dtype=bool)
    )
    fd_selection = (
        mean_fd > fd_threshold
        if fd_threshold > 0
        else np.zeros(len(mean_fd), dtype=bool)
    )
    excluded = dvars_selection | fd_selection
    return np.flatnonzero(~excluded).tolist()


def save_exclusion_file(
    dvars,
    mean_fd,
    dvars_threshold,
    fd_threshold,
    scan_dir,
):
    from pathlib import Path

    import numpy as np
    import pandas as pd

    dvars = np.asarray(dvars)
    mean_fd = np.asarray(mean_fd)
    if len(dvars) != len(mean_fd):
        raise ValueError(
            "DVARS and MeanFD must have the same number of volumes, got "
            f"{len(dvars)} and {len(mean_fd)}"
        )

    dvars_selection = (
        dvars > dvars_threshold
        if dvars_threshold > 0
        else np.zeros(len(dvars), dtype=bool)
    )
    fd_selection = (
        mean_fd > fd_threshold
        if fd_threshold > 0
        else np.zeros(len(mean_fd), dtype=bool)
    )
    combined_selection = dvars_selection | fd_selection
    output_file = Path(scan_dir) / "exclude.csv"
    pd.DataFrame(
        {
            "DVARS": dvars,
            "MeanFD": mean_fd,
            "DVARS_selection": dvars_selection.astype(int),
            "FD_selection": fd_selection.astype(int),
            "Combined_selection": combined_selection.astype(int),
        }
    ).to_csv(output_file, index=False)
    return str(output_file)


def smooth_image(cleaned_file, smooth):
    from nilearn.image import smooth_img

    return smooth_img(cleaned_file, fwhm=smooth)


def save_censored_image(smoothed_image, kept_volumes, scan_dir):
    from pathlib import Path

    from nilearn.image import index_img

    output_file = Path(scan_dir) / "filtered_func_data_clean_smoothed.nii.gz"
    index_img(smoothed_image, kept_volumes).to_filename(output_file)
    return str(output_file)
