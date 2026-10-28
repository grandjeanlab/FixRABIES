from nipype import Node
from nipype.interfaces.fsl.model import MELODIC
from nipype.interfaces.utility import Function


def melodic_node():
    node = Node(
        MELODIC(),
        name="melodic",
    )

    node.inputs.approach = "tica"
    node.inputs.no_bet = True
    node.inputs.no_mask = False
    node.inputs.report = True

    return node


def get_melodic_metadata(scan_info):
    return scan_info["bg"], scan_info["tr"], scan_info["mask"]


def get_melodic_out_dir(scan_dir):
    from pathlib import Path

    return str(Path(scan_dir) / "filtered_func_data.ica")


def write_hand_labels_noise(melodic_out_dir, scan_dir):
    import warnings
    from pathlib import Path

    melodic_out_dir = Path(melodic_out_dir)
    stats_file = melodic_out_dir / "melodic_ICstats"

    with open(stats_file) as f:
        component_count = sum(1 for _ in f)

    if component_count == 0:
        warnings.warn(
            f"No ICA components found in {stats_file}",
            stacklevel=2,
        )

    hand_labels_noise = Path(scan_dir) / "hand_labels_noise.txt"
    hand_labels_noise.write_text(
        f"{list(range(1, component_count + 1))}\n"
    )

    return str(hand_labels_noise)


def hand_labels_noise_node():
    return Node(
        Function(
            input_names=[
                "melodic_out_dir",
                "scan_dir",
            ],
            output_names=[
                "hand_labels_noise",
            ],
            function=write_hand_labels_noise,
        ),
        name="hand_labels_noise",
    )
