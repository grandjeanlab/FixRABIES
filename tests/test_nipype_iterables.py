from nipype import Workflow, Node, MapNode
from nipype.interfaces.io import DataFinder
from nipype.interfaces.utility import Function
from pathlib import Path

fix_dir=Path("~/test/fix_test").expanduser().resolve()
preprocess_dir=Path("~/test/preprocessing").expanduser().resolve()


def build_scan_id(bold_file):
    import re
    from pathlib import Path

    path = Path(bold_file)

    text = str(path)

    entities = []

    for pattern in [
        r"sub-[^_/]+",
        r"ses-[^_/]+",
        r"run-[^_/]+",
    ]:
        match = re.search(pattern, text)
        if match:
            entities.append(match.group(0))

    scan_id = "_".join(entities)

    print(f"{bold_file} -> {scan_id}")

    return scan_id

def print_file(bold_file):
    print(bold_file)
    return bold_file

def create_fix_dir(scan_id, fix_dir):
    from pathlib import Path

    outdir = fix_dir / "data" / scan_id
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir)


finder = Node(
    DataFinder(
        root_paths=preprocess_dir,
        match_regex=r".*/bold_datasink/commonspace_bold/_scan_info.*/_run_.*/.*bold_resampled\.nii(\.gz)?$",
    ),
    name="finder",
)

printer = MapNode(
    Function(
        input_names=["bold_file"],
        output_names=["bold_file"],
        function=print_file,
    ),
    iterfield=["bold_file"],
    name="printer",
)

wf = Workflow(name="test")
wf.connect(finder, "out_paths", printer, "bold_file")
wf.run()
