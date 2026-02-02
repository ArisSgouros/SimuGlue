# src/simuglue/workflows/cij/__init__.py
"""
Cij workflow package.

Public API:
- run_cij: generate and run deformation cases
- post_cij: collect finished cases and compute elastic constants
"""

from .config import Config, load_config
from .workflow import init_deformation, run_deformation
from .post_deformation import post_deformation

# auto-register workflow-local backends
from . import backends  # noqa: F401  (e.g. backends.lammps)

