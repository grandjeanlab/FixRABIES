def prepare_fix_scan(
    scan_id,
    scan_info,
    fix_dir,
    patch_denoise,
    species,
):
    from importlib.resources import as_file, files
    import shutil
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
        ).to_filename(regdir / "ica.nii.gz")

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


def get_scan_info(scan_id, scans):
    return scans[scan_id]


def get_bold_file(scan_info):
    return scan_info["bold_file"]
