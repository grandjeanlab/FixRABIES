from pathlib import Path

from nipype import Node, Workflow
from nipype.interfaces.utility import Function


def create_model_directory(fix_dir, scan_list):
    from pathlib import Path

    model_dir = Path(fix_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    return str(model_dir)


def select_training_scans(scan_list, select_label, model_dir):
    import math
    import random
    from pathlib import Path

    if select_label == 0:
        selected_scans = random.sample(scan_list, k=1)
    elif select_label == 100:
        selected_scans = scan_list
    else:
        selected_count = math.ceil(len(scan_list) * select_label / 100)
        selected_scans = random.sample(scan_list, k=selected_count)

    training_set_file = Path(model_dir) / "training_set.txt"
    training_set_file.write_text(
        "\n".join(selected_scans) + "\n",
        encoding="utf-8",
    )

    return selected_scans


def get_selected_scan_info(scan_list):
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
            "noise_list": str(scan_dir / "hand_labels_noise.txt"),
            "n_ic": n_ic,
        }

    return scans


def create_label_workflow(fix_dir, scan_list, select_label):
    model_directory = Node(
        Function(
            input_names=["fix_dir", "scan_list"],
            output_names=["model_dir"],
            function=create_model_directory,
        ),
        name="model_directory",
    )
    model_directory.inputs.fix_dir = str(fix_dir)
    model_directory.inputs.scan_list = scan_list

    select_scans = Node(
        Function(
            input_names=["scan_list", "select_label", "model_dir"],
            output_names=["selected_scans"],
            function=select_training_scans,
        ),
        name="select_scans",
    )
    select_scans.inputs.scan_list = scan_list
    select_scans.inputs.select_label = select_label

    selected_scan_info = Node(
        Function(
            input_names=["scan_list"],
            output_names=["scans"],
            function=get_selected_scan_info,
        ),
        name="selected_scan_info",
    )

    workflow = Workflow(
        name="label",
        base_dir=str(Path(fix_dir) / "workflows"),
    )
    workflow.connect(
        model_directory,
        "model_dir",
        select_scans,
        "model_dir",
    )
    workflow.connect(
        select_scans,
        "selected_scans",
        selected_scan_info,
        "scan_list",
    )

    return workflow
