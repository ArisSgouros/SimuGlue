# src/simuglue/cli/pwo2json_cli.py
from simuglue.quantum_espresso import pwo2json
from ._qe_pw2json_base import run

def main(argv=None, prog: str | None = None) -> int:
    return run(pwo2json, argv=argv, prog=prog)

if __name__ == "__main__":
    raise SystemExit(main())
