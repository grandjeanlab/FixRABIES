from pathlib import Path

from nipype import Node, Workflow
from nipype.interfaces.utility import Function, IdentityInterface


def prepare_fix_scan(scan_id, scan_info, fix_dir):
    import shutil
    from pathlib import Path

    import nibabel as nib
    import pandas as pd

    outdir = Path(fix_dir) / "data" / scan_id

    mcdir = outdir / "mc"
    regdir = outdir / "reg"

    outdir.mkdir(parents=True, exist_ok=True)
    mcdir.mkdir(exist_ok=True)
    regdir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Copy files
    # ------------------------------------------------------------------

    shutil.copyfile(
        scan_info["bold_file"],
        outdir / "filtered_func_data.nii.gz",
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

    return str(outdir)


def get_scan_info(scan_id, scans):
    return scans[scan_id]


def create_prepare_fix_workflow(scans, fix_dir):

    scan_ids = sorted(scans.keys())

    infosource = Node(
        IdentityInterface(fields=["scan_id"]),
        name="infosource",
    )

    infosource.iterables = [("scan_id", scan_ids)]

    lookup = Node(
        Function(
            input_names=[
                "scan_id",
                "scans",
            ],
            output_names=[
                "scan_info",
            ],
            function=get_scan_info,
        ),
        name="lookup",
    )

    lookup.inputs.scans = scans

    prepare = Node(
        Function(
            input_names=[
                "scan_id",
                "scan_info",
                "fix_dir",
            ],
            output_names=[
                "outdir",
            ],
            function=prepare_fix_scan,
        ),
        name="prepare_fix",
    )

    prepare.inputs.fix_dir = str(fix_dir)

    wf = Workflow(
        name="prepare",
        base_dir=str(Path(fix_dir) / "workflows"),
    )

    wf.connect(
        infosource,
        "scan_id",
        lookup,
        "scan_id",
    )

    wf.connect(
        infosource,
        "scan_id",
        prepare,
        "scan_id",
    )

    wf.connect(
        lookup,
        "scan_info",
        prepare,
        "scan_info",
    )

    return wf
