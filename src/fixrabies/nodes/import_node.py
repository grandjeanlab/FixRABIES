def import_cleaned_scan(scan_id, scan_info, fix_dir, species):
    from importlib.resources import as_file, files
    from pathlib import Path
    from shutil import copyfile

    from nilearn.image import resample_to_img

    scan_dir = Path(fix_dir) / "data" / scan_id
    scan_dir.mkdir(parents=True, exist_ok=True)

    cleaned_file = scan_dir / "filtered_func_data_clean_smoothed.nii.gz"
    mask_file = scan_dir / "mask.nii.gz"
    ica_file = scan_dir / "ica.nii.gz"
    copyfile(scan_info["clean_file"], cleaned_file)
    copyfile(scan_info["mask"], mask_file)

    template = (
        files("fixrabies")
        .joinpath("templates")
        .joinpath(species)
        .joinpath("ica.nii.gz")
    )
    with as_file(template) as template_file:
        resample_to_img(
            template_file,
            cleaned_file,
            interpolation="nearest",
        ).to_filename(ica_file)

    return str(scan_dir)
