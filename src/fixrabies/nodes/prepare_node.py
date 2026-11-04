def prepare_fix_scan(
    scan_id,
    scan_info,
    fix_dir,
    patch_denoise,
    species,
):
    import shutil
    from importlib.resources import as_file, files
    from pathlib import Path

    import nibabel as nib
    import pandas as pd
    from nilearn.image import resample_to_img

    outdir = Path(fix_dir) / "data" / scan_id

    mcdir = outdir / "mc"
    regdir = outdir / "reg"

    outdir.mkdir(parents=True, exist_ok=True)
    mcdir.mkdir(exist_ok=True)
    regdir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Copy files
    # ------------------------------------------------------------------
    filtered_func_data = outdir / "filtered_func_data.nii.gz"

    shutil.copyfile(
        scan_info["bold_file"],
        filtered_func_data,
    )

    shutil.copyfile(
        scan_info["fd"],
        outdir / "fd.csv",
    )

    shutil.copyfile(
        scan_info["bg"],
        outdir / "bg.nii.gz",
    )

    shutil.copyfile(
        scan_info["mask"],
        outdir / "mask.nii.gz",
    )

    shutil.copyfile(
        scan_info["highres"],
        regdir / "highres.nii.gz",
    )

    template = (
        files("fixrabies")
        .joinpath("templates")
        .joinpath(species)
        .joinpath("highres_pveseg.nii.gz")
    )

    with as_file(template) as template_file:
        resample_to_img(
            template_file,
            outdir / "bg.nii.gz",
            interpolation="nearest",
        ).to_filename(regdir / "highres_pveseg.nii.gz")

    template = (
        files("fixrabies")
        .joinpath("templates")
        .joinpath(species)
        .joinpath("ica.nii.gz")
    )

    with as_file(template) as template_file:
        resample_to_img(
            template_file,
            outdir / "bg.nii.gz",
            interpolation="nearest",
        ).to_filename(outdir / "ica.nii.gz")

    # ------------------------------------------------------------------
    # Convert motion file
    # ------------------------------------------------------------------

    df = pd.read_csv(scan_info["motion"])

    motion = df.iloc[:, 1:7].astype("float64")

    motion.to_csv(
        mcdir / "prefiltered_func_data_mcf.par",
        sep="\t",
        header=False,
        index=False,
        float_format="%.10f",
    )

    # ----------------------------------------------------------
    # highres2example_func.mat
    # ----------------------------------------------------------

    with open(
        regdir / "highres2example_func.mat",
        "w",
        newline="\n",
    ) as f:
        f.write("1  0  0  0  \n" "0  1  0  0  \n" "0  0  1  0  \n" "0  0  0  1  \n")

    # ----------------------------------------------------------
    # design.fsf
    # ----------------------------------------------------------

    img = nib.load(scan_info["bold_file"])

    npts = img.shape[3]

    tr = scan_info["tr"]

    with open(outdir / "design.fsf", "w") as f:
        f.write(
            f"set fmri(tr) {tr}\n" f"set fmri(npts) {npts}\n" "set fmri(temphp_yn) 0\n"
        )

    return str(outdir), str(filtered_func_data)


def censor_frames_by_fd(filtered_func_data, scan_dir, fd_threshold):
    from pathlib import Path

    import nibabel as nib
    import numpy as np
    import pandas as pd

    scan_dir = Path(scan_dir)
    fd_file = scan_dir / "fd.csv"
    motion_file = scan_dir / "mc" / "prefiltered_func_data_mcf.par"
    fd_data = pd.read_csv(fd_file)
    if "MeanFD" not in fd_data:
        raise ValueError(f"Missing MeanFD column in: {fd_file}")
    if fd_threshold < 0:
        raise ValueError(
            f"FD_threshold must be greater than or equal to 0, got {fd_threshold}"
        )

    image = nib.load(filtered_func_data)
    image_data = image.get_fdata(dtype="float32")
    if image_data.ndim != 4:
        raise ValueError(f"Prepared functional image is not 4D: {filtered_func_data}")
    if len(fd_data) != image_data.shape[3]:
        raise ValueError(
            "FD rows and functional-image volumes must match, got "
            f"{len(fd_data)} and {image_data.shape[3]}"
        )
    motion_data = pd.read_csv(motion_file, sep="\t", header=None)
    if len(motion_data) != image_data.shape[3]:
        raise ValueError(
            "Motion rows and functional-image volumes must match, got "
            f"{len(motion_data)} and {image_data.shape[3]}"
        )

    fd_selection = (
        fd_data["MeanFD"].to_numpy() > fd_threshold
        if fd_threshold > 0
        else np.zeros(len(fd_data), dtype=bool)
    )
    kept_volumes = np.flatnonzero(~fd_selection)
    if not len(kept_volumes):
        raise ValueError(
            f"FD_threshold {fd_threshold} excludes every volume in {scan_dir}"
        )

    prepare_exclude = fd_data.copy()
    prepare_exclude.insert(0, "Frame", np.arange(len(fd_data)))
    prepare_exclude["FD_selection"] = fd_selection.astype(int)
    prepare_exclude.to_csv(scan_dir / "prepare_exclude.csv", index=False)
    fd_data.loc[~fd_selection].to_csv(fd_file, index=False)
    motion_data.loc[~fd_selection].to_csv(
        motion_file,
        sep="\t",
        header=False,
        index=False,
        float_format="%.10f",
    )

    output_file = Path(filtered_func_data)
    nib.Nifti1Image(
        image_data[..., kept_volumes],
        image.affine,
        image.header,
    ).to_filename(output_file)
    design_file = scan_dir / "design.fsf"
    if design_file.is_file():
        design_file.write_text(
            "\n".join(
                (
                    f"set fmri(npts) {len(kept_volumes)}"
                    if line.startswith("set fmri(npts)")
                    else line
                )
                for line in design_file.read_text(encoding="utf-8").splitlines()
            )
            + "\n",
            encoding="utf-8",
        )

    return str(output_file)


def get_scan_info(scan_id, scans):
    return scans[scan_id]


def get_bold_file(scan_info):
    return scan_info["bold_file"]
