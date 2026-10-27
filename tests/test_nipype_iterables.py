from fixrabies.interfaces.prepare_rabies import prepare_rabies
from fixrabies.workflows.prepare_wf import create_prepare_fix_workflow


scans = prepare_rabies("/home/traaffneu/joagra/test/preprocessing")

wf = create_prepare_fix_workflow(
    scans=scans,
    fix_dir="/home/traaffneu/joagra/test/fix",
    #make patch_denoise a dict with apply=True, method="optimal-fro"
    patch_denoise={"apply": False, "method": "nordic"},
    species="mouseDSURQE",
)

