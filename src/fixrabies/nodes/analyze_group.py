def aggregate_group_analysis(fix_dir, scan_names, dr, seed):
    from pathlib import Path

    import nibabel as nib
    import numpy as np
    import pandas as pd
    from scipy.stats import ttest_1samp

    results_dir = Path(fix_dir) / "results"
    group_dir = Path(fix_dir) / "results_group"

    def save_group_maps(files, output_dir, output_stem, z_map=False):
        images = [nib.load(file) for file in files]
        data = [image.get_fdata(dtype="float32") for image in images]
        reference_shape = data[0].shape
        if any(image_data.shape != reference_shape for image_data in data[1:]):
            raise ValueError(f"Input images do not share dimensions for {output_stem}")
        if len(reference_shape) != 3:
            raise ValueError(
                f"Expected 3D images for {output_stem}, got {reference_shape}"
            )

        stacked_data = np.stack(data, axis=-1)
        mean_data = np.mean(stacked_data, axis=-1)
        std_data = np.std(stacked_data, axis=-1, ddof=1)
        t_data, p_data = ttest_1samp(
            stacked_data,
            popmean=0,
            axis=-1,
            nan_policy="omit",
        )
        p_data = 1 - p_data
        affine = images[0].affine
        header = images[0].header
        output_dir.mkdir(parents=True, exist_ok=True)

        z_suffix = "Z" if z_map else ""
        output_files = {
            statistic: output_dir / f"{output_stem}_{statistic}{z_suffix}.nii.gz"
            for statistic in ("concat", "mean", "std", "tmap", "pmap")
        }
        nib.Nifti1Image(stacked_data, affine, header).to_filename(
            output_files["concat"]
        )
        nib.Nifti1Image(mean_data, affine, header).to_filename(output_files["mean"])
        nib.Nifti1Image(std_data, affine, header).to_filename(output_files["std"])
        nib.Nifti1Image(t_data, affine, header).to_filename(output_files["tmap"])
        nib.Nifti1Image(p_data, affine, header).to_filename(output_files["pmap"])

    if dr:
        dr_dir = group_dir / "dr"
        dr_dir.mkdir(parents=True, exist_ok=True)
        (dr_dir / "scan_list.txt").write_text(
            "\n".join(scan_names) + "\n",
            encoding="utf-8",
        )
        beta_images = [
            results_dir / scan_name / "dr" / "dr_stage2_subject00000.nii.gz"
            for scan_name in scan_names
        ]
        z_images = [
            results_dir / scan_name / "dr" / "dr_stage2_subject00000_Z.nii.gz"
            for scan_name in scan_names
        ]
        beta_data = nib.load(beta_images[0]).shape
        z_data = nib.load(z_images[0]).shape
        if (
            len(beta_data) != 4
            or beta_data != z_data
            or any(nib.load(file).shape != beta_data for file in beta_images[1:])
            or any(nib.load(file).shape != z_data for file in z_images[1:])
        ):
            raise ValueError("Dual-regression images do not share 4D dimensions")

        for component in range(beta_data[3]):
            component_index = component + 1
            component_dir = dr_dir / f"ic{component_index}"
            component_dir.mkdir(parents=True, exist_ok=True)
            beta_component_files = []
            z_component_files = []
            for scan_name, beta_image, z_image in zip(
                scan_names,
                beta_images,
                z_images,
                strict=True,
            ):
                beta_component_file = (
                    component_dir / f".{scan_name}_dr_stage2_ic{component_index}.nii.gz"
                )
                z_component_file = (
                    component_dir
                    / f".{scan_name}_dr_stage2_ic{component_index}_Z.nii.gz"
                )
                beta_image_data = nib.load(beta_image)
                z_image_data = nib.load(z_image)
                nib.Nifti1Image(
                    beta_image_data.dataobj[..., component],
                    beta_image_data.affine,
                    beta_image_data.header,
                ).to_filename(beta_component_file)
                nib.Nifti1Image(
                    z_image_data.dataobj[..., component],
                    z_image_data.affine,
                    z_image_data.header,
                ).to_filename(z_component_file)
                beta_component_files.append(beta_component_file)
                z_component_files.append(z_component_file)

            try:
                save_group_maps(
                    beta_component_files,
                    component_dir,
                    f"dr_stage2_ic{component_index}",
                )
                save_group_maps(
                    z_component_files,
                    component_dir,
                    f"dr_stage2_ic{component_index}",
                    z_map=True,
                )
            finally:
                for component_file in beta_component_files + z_component_files:
                    component_file.unlink(missing_ok=True)

        outcomes = [
            pd.read_csv(results_dir / scan_name / "dr" / "outcomes.csv")
            for scan_name in scan_names
        ]
        outcome_columns = outcomes[0].columns
        if any(
            not outcome.columns.equals(outcome_columns)
            or len(outcome) != len(outcomes[0])
            for outcome in outcomes[1:]
        ):
            raise ValueError("Dual-regression outcome files do not match")
        for component in range(len(outcomes[0])):
            component_outcomes = pd.DataFrame(
                [
                    {"scan": scan_name, **outcome.iloc[component].to_dict()}
                    for scan_name, outcome in zip(
                        scan_names,
                        outcomes,
                        strict=True,
                    )
                ]
            )
            component_outcomes.to_csv(
                dr_dir / f"ic{component + 1}" / f"outcomes_ic{component + 1}.csv",
                index=False,
            )

    if seed:
        seed_dir = group_dir / "seed"
        seed_files_by_name = {}
        correlations = []
        for scan_name in scan_names:
            scan_seed_dir = results_dir / scan_name / "seed"
            for seed_file in sorted(scan_seed_dir.glob("*.nii.gz")):
                seed_files_by_name.setdefault(seed_file.name, []).append(seed_file)

            correlation = pd.read_csv(
                scan_seed_dir / "corr.txt",
                sep="\t",
                index_col=0,
            )
            correlations.append((scan_name, correlation))

        for seed_name, seed_files in seed_files_by_name.items():
            if len(seed_files) != len(scan_names):
                raise ValueError(
                    f"Seed image {seed_name} is not available for every scan"
                )
            save_group_maps(
                seed_files,
                seed_dir / Path(seed_name).name.removesuffix(".nii.gz"),
                Path(seed_name).name.removesuffix(".nii.gz"),
            )

        correlation_columns = None
        correlation_rows = []
        for scan_name, correlation in correlations:
            flattened = {
                f"{row}_{column}": correlation.loc[row, column]
                for row in correlation.index
                for column in correlation.columns
            }
            if correlation_columns is None:
                correlation_columns = list(flattened)
            elif list(flattened) != correlation_columns:
                raise ValueError("Seed correlation matrices do not match")
            correlation_rows.append({"scan": scan_name, **flattened})
        pd.DataFrame(correlation_rows).to_csv(
            seed_dir / "corr.csv",
            index=False,
        )

    return str(group_dir)
