from simuglue.workflow.cij.run import run_cij
from simuglue.workflow.cij.post import post_cij
cases = ["cij_2d.yaml", "cij_3d.yaml"]
for case_ in cases:
    print(f"case: {case_}")
    run_cij(config_path=case_)
    post_cij(config_path=case_)
