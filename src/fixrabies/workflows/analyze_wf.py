from pathlib import Path

from nipype import MapNode, Node, Workflow
from nipype.interfaces.fsl.model import GLM, DualRegression
from nipype.interfaces.fsl.utils import ImageMeants
from nipype.interfaces.utility import Function, IdentityInterface

from fixrabies.nodes.analyze import (
    create_analysis_directories,
    get_cleaned_file,
    get_ica_file,
    get_mask_file,
    get_seed_output_files,
    resample_seeds,
    resample_template,
    save_dual_regression_outcomes,
    save_seed_correlation,
)


def create_analyze_workflow(
    fix_dir,
    scan_list,
    dr_template,
    seed_files,
    norm,
    dice_threshold,
    extract_threshold,
    run_dual_regression,
):
    infosource = Node(
        IdentityInterface(fields=["scan_dir"]),
        name="infosource",
    )
    infosource.iterables = [("scan_dir", scan_list)]

    cleaned_file = Node(
        Function(
            input_names=["scan_dir"],
            output_names=["cleaned_file"],
            function=get_cleaned_file,
        ),
        name="cleaned_file",
    )
    analysis_directories = Node(
        Function(
            input_names=["scan_dir", "results_dir", "seed_analysis"],
            output_names=["dr_dir", "seed_dir"],
            function=create_analysis_directories,
        ),
        name="analysis_directories",
    )
    analysis_directories.inputs.results_dir = str(Path(fix_dir) / "results")
    analysis_directories.inputs.seed_analysis = bool(seed_files)

    workflow = Workflow(
        name=("analyze" if run_dual_regression else "analyze_outcomes"),
        base_dir=str(Path(fix_dir) / "workflows"),
    )
    workflow.connect(
        infosource,
        "scan_dir",
        cleaned_file,
        "scan_dir",
    )
    ica_file = Node(
        Function(
            input_names=["scan_dir"],
            output_names=["ica_file"],
            function=get_ica_file,
        ),
        name="ica_file",
    )
    mask_file = Node(
        Function(
            input_names=["scan_dir"],
            output_names=["mask_file"],
            function=get_mask_file,
        ),
        name="mask_file",
    )
    workflow.connect(infosource, "scan_dir", ica_file, "scan_dir")
    workflow.connect(infosource, "scan_dir", mask_file, "scan_dir")
    workflow.connect(
        infosource,
        "scan_dir",
        analysis_directories,
        "scan_dir",
    )

    outcomes = Node(
        Function(
            input_names=[
                "ica_file",
                "mask_file",
                "dr_dir",
                "dice_threshold",
                "extract_threshold",
            ],
            output_names=["outcomes_file"],
            function=save_dual_regression_outcomes,
        ),
        name="outcomes",
    )
    outcomes.inputs.dice_threshold = dice_threshold
    outcomes.inputs.extract_threshold = extract_threshold
    workflow.connect(ica_file, "ica_file", outcomes, "ica_file")
    workflow.connect(mask_file, "mask_file", outcomes, "mask_file")
    if run_dual_regression:
        dual_regression = Node(
            DualRegression(
                des_norm=norm,
                one_sample_group_mean=True,
                n_perm=0,
            ),
            name="dual_regression",
        )
        workflow.connect(
            cleaned_file,
            "cleaned_file",
            dual_regression,
            "in_files",
        )
        workflow.connect(
            analysis_directories,
            "dr_dir",
            dual_regression,
            "out_dir",
        )
        workflow.connect(
            dual_regression,
            "out_dir",
            outcomes,
            "dr_dir",
        )
    else:
        workflow.connect(
            analysis_directories,
            "dr_dir",
            outcomes,
            "dr_dir",
        )

    if run_dual_regression and dr_template is not None:
        resampled_template = Node(
            Function(
                input_names=["template_file", "cleaned_file", "scan_dir"],
                output_names=["resampled_template"],
                function=resample_template,
            ),
            name="resampled_template",
        )
        resampled_template.inputs.template_file = dr_template

        workflow.connect(
            infosource,
            "scan_dir",
            resampled_template,
            "scan_dir",
        )
        workflow.connect(
            cleaned_file,
            "cleaned_file",
            resampled_template,
            "cleaned_file",
        )
        workflow.connect(
            resampled_template,
            "resampled_template",
            dual_regression,
            "group_IC_maps_4D",
        )
    elif run_dual_regression:
        workflow.connect(
            ica_file,
            "ica_file",
            dual_regression,
            "group_IC_maps_4D",
        )

    if seed_files:
        resampled_seeds = Node(
            Function(
                input_names=["seed_files", "cleaned_file", "scan_dir"],
                output_names=["resampled_seeds"],
                function=resample_seeds,
            ),
            name="resampled_seeds",
        )
        resampled_seeds.inputs.seed_files = seed_files
        workflow.connect(
            infosource,
            "scan_dir",
            resampled_seeds,
            "scan_dir",
        )
        workflow.connect(
            cleaned_file,
            "cleaned_file",
            resampled_seeds,
            "cleaned_file",
        )
        seed_outputs = Node(
            Function(
                input_names=["resampled_seeds", "seed_dir"],
                output_names=[
                    "seed_time_files",
                    "beta_files",
                    "z_files",
                    "corr_file",
                    "seed_basenames",
                ],
                function=get_seed_output_files,
            ),
            name="seed_outputs",
        )
        image_meants = MapNode(
            ImageMeants(),
            iterfield=["mask", "out_file"],
            name="image_meants",
        )
        glm = MapNode(
            GLM(
                dat_norm=norm,
                demean=True,
                des_norm=True,
            ),
            iterfield=["design", "out_file", "out_z_name"],
            name="seed_glm",
        )
        seed_correlation = Node(
            Function(
                input_names=[
                    "seed_time_files",
                    "glm_files",
                    "corr_file",
                    "seed_basenames",
                ],
                output_names=["corr_file"],
                function=save_seed_correlation,
            ),
            name="seed_correlation",
        )
        workflow.connect(
            resampled_seeds,
            "resampled_seeds",
            seed_outputs,
            "resampled_seeds",
        )
        workflow.connect(
            analysis_directories,
            "seed_dir",
            seed_outputs,
            "seed_dir",
        )
        workflow.connect(
            cleaned_file,
            "cleaned_file",
            image_meants,
            "in_file",
        )
        workflow.connect(
            resampled_seeds,
            "resampled_seeds",
            image_meants,
            "mask",
        )
        workflow.connect(
            seed_outputs,
            "seed_time_files",
            image_meants,
            "out_file",
        )
        workflow.connect(
            cleaned_file,
            "cleaned_file",
            glm,
            "in_file",
        )
        workflow.connect(mask_file, "mask_file", glm, "mask")
        workflow.connect(
            image_meants,
            "out_file",
            glm,
            "design",
        )
        workflow.connect(seed_outputs, "beta_files", glm, "out_file")
        workflow.connect(seed_outputs, "z_files", glm, "out_z_name")
        workflow.connect(
            image_meants,
            "out_file",
            seed_correlation,
            "seed_time_files",
        )
        workflow.connect(glm, "out_file", seed_correlation, "glm_files")
        workflow.connect(
            seed_outputs,
            "corr_file",
            seed_correlation,
            "corr_file",
        )
        workflow.connect(
            seed_outputs,
            "seed_basenames",
            seed_correlation,
            "seed_basenames",
        )

    return workflow
