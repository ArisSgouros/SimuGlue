from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from ase import units
from ase.calculators.lammps.unitconvert import convert
from ase.io.lammpsdata import read_lammps_data, write_lammps_data

from ..config import Config
from ..registry import (
    register_backend,
    Backend,
    RelaxResult,
    is_done,
    mark_done,
)

# ---------- LAMMPS backend (skeleton) ----------
@register_backend("lammps")
class LAMMPSBackend(Backend):

    def read_data(self, path: Path, cfg: Config):
        units_ = cfg.lammps.get("units", "metal")
        atom_style = cfg.lammps.get("atom_style", "atomic")
        return read_lammps_data(path, units=units_, atom_style=atom_style)

    def write_data(self, path: Path, atoms, cfg: Config) -> None:
        units_ = cfg.lammps.get("units", "metal")
        atom_style = cfg.lammps.get("atom_style", "atomic")
        write_lammps_data(
            path,
            atoms,
            atom_style=atom_style,
            units=units_,
            force_skew=True,
        )

    def prepare_case(self, case_dir: Path, atoms, cfg: Config) -> None:
        if is_done(case_dir):
            return

        units_ = cfg.lammps.get("units", "metal")
        atom_style = cfg.lammps.get("atom_style", "atomic")

        write_lammps_data(
            case_dir / "str.data",
            atoms,
            atom_style=atom_style,
            units=units_,
            force_skew=True,
        )

        tpl = Path(cfg.lammps["input_template"]).read_text(encoding="utf-8")
        tpl = tpl.replace("${datafile}", "str.data")

        print_line = (
            "print '{\"pxx\":$(pxx), \"pyy\":$(pyy), \"pzz\":$(pzz), "
            "\"pyz\":$(pyz), \"pxz\":$(pxz), \"pxy\":$(pxy), \"pe\":$(pe)}' "
            "file thermo.json"
        )
        if "thermo.json" not in tpl:
            tpl = tpl.rstrip() + "\n" + print_line + "\n"

        dst = case_dir / "in.min"
        if not dst.exists():
            dst.write_text(tpl, encoding="utf-8")

    def run_case(self, case_dir: Path, atoms, cfg: Config) -> None:
        if is_done(case_dir):
            return
        exe = cfg.lammps.get("exe", "lmp")
        log = case_dir / "log.lammps"
        with log.open("w") as f:
            subprocess.run(
                [exe, "-in", "in.min"],
                cwd=case_dir,
                check=True,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        if not (case_dir / "thermo.json").is_file():
            raise RuntimeError(
                "LAMMPS finished but thermo.json missing; "
                "check template/thermo line."
            )
        mark_done(case_dir)

    def parse_case(self, case_dir: Path, atoms, cfg: Config) -> RelaxResult:
        if not is_done(case_dir):
            raise ValueError("Attempted to parse case that is not marked done.")

        data = json.loads((case_dir / "thermo.json").read_text(encoding="utf-8"))

        lammps_units = cfg.lammps.get("units", "metal")

        # pressures: LAMMPS p (compression +) = -σ, convert to GPa
        p_evA3 = convert(1.0, "pressure", lammps_units, "ASE")
        p_GPa = p_evA3 / units.GPa

        S = -np.array(
            [
                [data["pxx"], data["pxy"], data["pxz"]],
                [data["pxy"], data["pyy"], data["pyz"]],
                [data["pxz"], data["pyz"], data["pzz"]],
            ],
            dtype=float,
        ) * p_GPa

        # energy
        e_eV = convert(1.0, "energy", lammps_units, "ASE")
        pe = float(data["pe"] * e_eV)

        return RelaxResult(atoms=atoms, energy=pe, stress=S)

