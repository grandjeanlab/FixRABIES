from pathlib import Path

from nipype import Node, Workflow
from nipype.interfaces.utility import Function


def get_review_scan_info(scan_list, fix_model, threshold):
    from pathlib import Path

    scans = {}
    for scan_path in scan_list:
        scan_dir = Path(scan_path)
        melodic_icstats = (
            scan_dir
            / "filtered_func_data.ica"
            / "melodic_ICstats"
        )
        if not melodic_icstats.is_file():
            raise ValueError(
                f"Missing melodic IC statistics file: {melodic_icstats}"
            )

        with melodic_icstats.open(encoding="utf-8") as file:
            n_ic = sum(1 for _ in file)

        scans[scan_dir.name] = {
            "scan_path": str(scan_dir),
            "noise_list": str(
                scan_dir
                / f"fix4melview_{fix_model}_thr{threshold}.txt"
            ),
            "n_ic": n_ic,
        }

    return scans


def create_review_workflow(
    fix_dir,
    scan_list,
    fix_model,
    threshold,
):
    review_scan_info = Node(
        Function(
            input_names=["scan_list", "fix_model", "threshold"],
            output_names=["scans"],
            function=get_review_scan_info,
        ),
        name="review_scan_info",
    )
    review_scan_info.inputs.scan_list = scan_list
    review_scan_info.inputs.fix_model = fix_model
    review_scan_info.inputs.threshold = threshold

    workflow = Workflow(
        name="review",
        base_dir=str(Path(fix_dir) / "workflows"),
    )
    workflow.add_nodes([review_scan_info])

    return workflow
