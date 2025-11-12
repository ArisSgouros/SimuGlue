from simuglue.workflow.cij.workflow import init_cij, run_cij, parse_cij
from simuglue.workflow.cij.post import post_cij
case_ = "cfg.yaml"
init_cij(config_path=case_)
run_cij(config_path=case_)
parse_cij(config_path=case_)
post_cij(config_path=case_)
