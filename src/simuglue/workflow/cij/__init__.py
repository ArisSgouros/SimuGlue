# src/simuglue/workflows/cij/__init__.py
from .runner import run_cij  # public API

# auto-register workflow-local backends
from .backends import lammps
#from .backends import qe
