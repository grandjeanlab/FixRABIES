from fixrabies.interfaces.prepare_rabies import prepare_rabies
from fixrabies.workflows.prepare_wf import create_prepare_fix_workflow

scans = prepare_rabies("/home/gjoanes/test/preprocessing")

wf = create_prepare_fix_workflow(
    scans=scans,
    fix_dir="/home/gjoanes/test/fix",
)

wf.run()
