from simuglue.workflow.cij.runner import run_cij
cases = ["cij_2d.yaml", "cij_3d.yaml"]
for case_ in cases:
    print(f"Running case: {case_}")
    run_cij(config_path=case_)
