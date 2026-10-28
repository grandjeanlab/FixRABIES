def get_cleaned_file(scan_dir):
    from pathlib import Path

    return str(Path(scan_dir) / "filtered_func_data_clean_smoothed.nii.gz")


def get_ica_file(scan_dir):
    from pathlib import Path

    return str(Path(scan_dir) / "ica.nii.gz")


def get_mask_file(scan_dir):
    from pathlib import Path

    return str(Path(scan_dir) / "mask.nii.gz")


def save_dual_regression_outcomes(
    ica_file,
    mask_file,
    dr_dir,
    dice_threshold,
    extract_threshold,
):
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from nilearn.image import load_img

    dr_dir = Path(dr_dir)
    beta_file = dr_dir / "dr_stage2_subject00000.nii.gz"
    z_file = dr_dir / "dr_stage2_subject00000_Z.nii.gz"
    required_files = [ica_file, mask_file, beta_file, z_file]
    missing_files = [str(path) for path in required_files if not Path(path).is_file()]
    if missing_files:
        raise ValueError(
            "Missing dual regression outcome files:\n" + "\n".join(missing_files)
        )

    ica_data = load_img(ica_file).get_fdata(dtype="float32")
    beta_data = load_img(beta_file).get_fdata(dtype="float32")
    z_data = load_img(z_file).get_fdata(dtype="float32")
    mask_data = load_img(mask_file).get_fdata() > 0
    if (
        ica_data.ndim != 4
        or beta_data.ndim != 4
        or z_data.ndim != 4
        or ica_data.shape != beta_data.shape
        or ica_data.shape != z_data.shape
        or ica_data.shape[:3] != mask_data.shape
    ):
        raise ValueError("ICA, dual regression, and mask dimensions do not match")
    if not np.any(mask_data):
        raise ValueError(f"Mask contains no voxels: {mask_file}")

    ica_values = ica_data[mask_data]
    beta_values = beta_data[mask_data]
    z_values = z_data[mask_data]
    dice_cutoffs = np.percentile(
        ica_values,
        dice_threshold,
        axis=0,
    )
    ica_dice_masks = ica_values >= dice_cutoffs
    beta_dice_masks = beta_values >= np.percentile(
        beta_values,
        dice_threshold,
        axis=0,
    )
    z_dice_masks = z_values >= np.percentile(
        z_values,
        dice_threshold,
        axis=0,
    )

    extract_value = float(extract_threshold[:-1])
    if extract_threshold.endswith("p"):
        extract_masks = ica_values >= np.percentile(
            ica_values,
            extract_value,
            axis=0,
        )
    else:
        extract_masks = ica_values >= extract_value

    def dice_score(first, second):
        intersection = np.count_nonzero(first & second)
        size = np.count_nonzero(first) + np.count_nonzero(second)
        return 1.0 if size == 0 else (2 * intersection) / size

    outcomes = pd.DataFrame(
        {
            "beta_dice": [
                dice_score(ica_mask, beta_mask)
                for ica_mask, beta_mask in zip(
                    ica_dice_masks.T,
                    beta_dice_masks.T,
                    strict=True,
                )
            ],
            "z_dice": [
                dice_score(ica_mask, z_mask)
                for ica_mask, z_mask in zip(
                    ica_dice_masks.T,
                    z_dice_masks.T,
                    strict=True,
                )
            ],
            "beta_value": [
                values[extract_mask].mean() if np.any(extract_mask) else np.nan
                for values, extract_mask in zip(
                    beta_values.T,
                    extract_masks.T,
                    strict=True,
                )
            ],
            "z_value": [
                values[extract_mask].mean() if np.any(extract_mask) else np.nan
                for values, extract_mask in zip(
                    z_values.T,
                    extract_masks.T,
                    strict=True,
                )
            ],
        }
    )
    output_file = dr_dir / "outcomes.csv"
    outcomes.to_csv(output_file, index=False)
    return str(output_file)


def resample_template(template_file, cleaned_file, scan_dir):
    from pathlib import Path

    from nilearn.image import resample_to_img

    output_file = Path(scan_dir) / "ica.nii.gz"
    resample_to_img(
        template_file,
        cleaned_file,
        interpolation="continuous",
    ).to_filename(output_file)
    return str(output_file)


def resample_seeds(seed_files, cleaned_file, scan_dir):
    from pathlib import Path

    from nilearn.image import resample_to_img

    scan_dir = Path(scan_dir)
    resampled_seeds = []
    for seed_file in seed_files:
        output_file = scan_dir / Path(seed_file).name
        resample_to_img(
            seed_file,
            cleaned_file,
            interpolation="nearest",
        ).to_filename(output_file)
        resampled_seeds.append(str(output_file))

    return resampled_seeds


def get_seed_output_files(resampled_seeds, seed_dir):
    from pathlib import Path

    seed_dir = Path(seed_dir)
    seed_basenames = [
        Path(seed_file).name.removesuffix(".nii.gz").removesuffix(".nii")
        for seed_file in resampled_seeds
    ]
    return (
        [str(seed_dir / f"{seed_basename}.txt") for seed_basename in seed_basenames],
        [
            str(seed_dir / f"{seed_basename}.nii.gz")
            for seed_basename in seed_basenames
        ],
        [
            str(seed_dir / f"{seed_basename}_Z.nii.gz")
            for seed_basename in seed_basenames
        ],
        str(seed_dir / "corr.txt"),
        seed_basenames,
    )


def save_seed_correlation(
    seed_time_files,
    glm_files,
    corr_file,
    seed_basenames,
):
    import numpy as np
    import pandas as pd

    time_series = {
        seed_basename: np.loadtxt(seed_time_file)
        for seed_basename, seed_time_file in zip(
            seed_basenames,
            seed_time_files,
            strict=True,
        )
    }
    pd.DataFrame(time_series).corr().to_csv(corr_file, sep="\t")
    return corr_file


def create_analysis_directories(scan_dir, results_dir, seed_analysis):
    from pathlib import Path

    scan_results_dir = Path(results_dir) / Path(scan_dir).name
    dr_dir = scan_results_dir / "dr"
    dr_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = scan_results_dir / "seed"
    if seed_analysis:
        seed_dir.mkdir(parents=True, exist_ok=True)

    return str(dr_dir), str(seed_dir)
