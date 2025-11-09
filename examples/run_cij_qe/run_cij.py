from simuglue.workflow.cij.run import init_cij, run_cij, parse_cij
from simuglue.workflow.cij.post import post_cij
cases = ["cij_2d.yaml"]
for case_ in cases:
    print(f"case: {case_}")
    init_cij(config_path=case_)
    run_cij(config_path=case_)
    parse_cij(config_path=case_)
    post_cij(config_path=case_)
