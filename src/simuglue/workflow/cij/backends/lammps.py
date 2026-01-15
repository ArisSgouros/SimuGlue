from __future__ import annotations

import json
import os, shlex, subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from ase import units
from ase.calculators.lammps.unitconvert import convert
from simuglue.ase_patches.lammpsdata import read_lammps_data, write_lammps_data
from simuglue.io.lammps_cell import lammps_box_to_ase_cell

from ..config import Config
from ..registry import (
    register_backend,
    Backend,
    RelaxResult,
    is_done,
    is_running,
)

from ..markers import prepare_run, finalize_success, finalize_failure

def _to_argv(x) -> list[str]:
    """Accept list/tuple/str/None and return a list of argv tokens."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(t) for t in x]
    if isinstance(x, str):
        return shlex.split(x)
    return [str(x)]

THERMO_JSON_PRINT_LINE = (
    "print '{\"pxx\":$(pxx), \"pyy\":$(pyy), \"pzz\":$(pzz), "
    "\"pyz\":$(pyz), \"pxz\":$(pxz), \"pxy\":$(pxy), \"pe\":$(pe),"
    "\"lx\":$(lx), \"ly\":$(ly), \"lz\":$(lz), \"xy\":$(xy), \"xz\":$(xz), \"yz\":$(yz) }' "
    "file thermo.json"
)

def generate_min_template_from_cfg(cfg: Any) -> str:
    """
    Generate a LAMMPS minimization input template from cfg.lammps.

    Expected keys under cfg.lammps (all optional):
      - units: str                         (default: "metal")
      - atom_style: str                    (default: "full")
      - potential_file: str                (default: "../potential.mod")
      - etol: float/str                    (default: 0.0)
      - ftol: float/str                    (default: 1.0e-10)
      - maxiter: int/str                   (default: 1000)
      - maxeval: int/str                   (default: 10000)   # note: "maxeval"
      - dmax: float/str                    (default: 1.0e-2)  # kept from your template
      - box_tilt: str                      (default: "large")
    """
    lmp: Mapping[str, Any] = getattr(cfg, "lammps", {}) or {}

    units = lmp.get("units", "metal")
    atom_style = lmp.get("atom_style", "full")

    potential_file = lmp.get("potential_file", "potential.mod")

    etol = lmp.get("etol", 0.0)
    ftol = lmp.get("ftol", 1.0e-10)
    maxiter = lmp.get("maxiter", 1000)
    maxeval = lmp.get("maxeval", 10000)
    dmax = lmp.get("dmax", 1.0e-2)

    box_tilt = lmp.get("box_tilt", "large")

    # Format numerics robustly (accept numbers or strings)
    def _fmt(v: Any) -> str:
        return str(v)

    tpl = "\n".join(
        [
            "# SimuGlue auto-generated template",
            f"units           {units}",
            f"atom_style      {atom_style}",
            "",
            "# Simulation variables",
            f"variable etol equal {_fmt(etol)}",
            f"variable ftol equal {_fmt(ftol)}",
            f"variable maxiter equal {_fmt(maxiter)}",
            f"variable maxeval equal {_fmt(maxeval)}",
            f"variable dmax equal {_fmt(dmax)}",
            "",
            f"box tilt {box_tilt}",
            f"read_data str.data",
            "",
            "# Include coefficients and potential parameters",
            f"include ../{potential_file}",
            "",
            "# Energy minimization",
            "minimize ${etol} ${ftol} ${maxiter} ${maxeval}",
            "",
            THERMO_JSON_PRINT_LINE,
            "",
        ]
    )
    return tpl

# ---------- LAMMPS backend (skeleton) ----------
@register_backend("lammps")
class LAMMPSBackend(Backend):

    def read_data(self, cfg: Config):
        units_ = cfg.lammps.get("units", "metal")
        atom_style = cfg.lammps.get("atom_style", "atomic")
        data_file = cfg.lammps.get("data_file", None)
        return read_lammps_data(data_file, units=units_, atom_style=atom_style)

    def write_data(self, path: Path, atoms, cfg: Config) -> None:
        units_ = cfg.lammps.get("units", "metal")
        atom_style = cfg.lammps.get("atom_style", "atomic")

        write_lammps_data(
            path,
            atoms,
            atom_style=atom_style,
            units=units_,
            preserve_atom_types=True,
            masses=True,
            force_skew=True,
        )

    def prepare_case(self, case_dir: Path, atoms, cfg: Config) -> None:
        if is_done(case_dir):
            return

        units_ = cfg.lammps.get("units", "metal")
        atom_style = cfg.lammps.get("atom_style", "atomic")

        data_path = case_dir / "str.data"

        write_lammps_data(
            case_dir / "str.data",
            atoms,
            atom_style=atom_style,
            units=units_,
            preserve_atom_types=True,
            masses=True,
            force_skew=True,
       )

        dst = case_dir / "in.min"
        if not dst.exists():
            tpl = generate_min_template_from_cfg(cfg)
            dst.write_text(tpl, encoding="utf-8")


    def run_case(self, case_dir: Path, cfg: Config) -> None:
        if is_done(case_dir) or is_running(case_dir):
            return

        """Run a single QE case with shared markers helper."""
        prepare_run(case_dir)

        log = case_dir / "log.lammps"
        log.parent.mkdir(parents=True, exist_ok=True)

        exe  = cfg.lammps.get("exe", "lmp")
        para = cfg.lammps.get("para", None)   # e.g., "mpirun -np 8"
        post = cfg.lammps.get("post", None)
        env  = dict(os.environ)
        env.update(cfg.lammps.get("env", {}))

        cmd = [*_to_argv(para), *_to_argv(exe), "-in", "in.min", *_to_argv(post)]

        try:
            with log.open("w") as f:
                f.write(f"# cwd: {case_dir}\n# cmd: {' '.join(shlex.quote(c) for c in cmd)}\n\n")
                f.flush()
                subprocess.run(
                    cmd,
                    cwd=case_dir,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

            # sanity check for expected output
            if (case_dir / "thermo.json").is_file():
                finalize_success(case_dir)
            else:
                finalize_failure(case_dir, "thermo.json missing; check your thermo output.")

        except subprocess.CalledProcessError as e:
            finalize_failure(case_dir, f"LAMMPS failed with exit code {e.returncode}")
        except Exception as e:
            finalize_failure(case_dir, f"Unexpected failure: {e}")

    def parse_case(self, case_dir: Path, cfg: Config) -> RelaxResult:
        if not is_done(case_dir):
            raise ValueError("Attempted to parse case that is not marked done.")

        data = json.loads((case_dir / "thermo.json").read_text(encoding="utf-8"))

        lammps_units = cfg.lammps.get("units", "metal")

        # pressures: LAMMPS p (compression +) = -σ, convert to ASE [eV/Angstrom^3]
        p_evA3 = convert(1.0, "pressure", lammps_units, "ASE")

        S = -np.array(
            [
                [data["pxx"], data["pxy"], data["pxz"]],
                [data["pxy"], data["pyy"], data["pyz"]],
                [data["pxz"], data["pyz"], data["pzz"]],
            ],
            dtype=float,
        ) * p_evA3

        # energy
        e_eV = convert(1.0, "energy", lammps_units, "ASE")
        pe = float(data["pe"] * e_eV)

        # cell
        cell = lammps_box_to_ase_cell(data['lx'], data['ly'], data['lz'], data['xy'], data['xz'], data['yz'])

        return RelaxResult(energy=pe, stress=S, cell=cell)

