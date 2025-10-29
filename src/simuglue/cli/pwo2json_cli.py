from simuglue.quantum_espresso import pwo2json
from ._pw2json_cli_base import run
def main():  # console_script
    raise SystemExit(run(pwo2json, prog="sglue-qe-pwo2json"))
