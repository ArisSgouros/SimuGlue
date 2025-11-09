from __future__ import annotations
from pathlib import Path
import shutil, subprocess, json, numpy as np
from ase.calculators.lammps.unitconvert import convert
from ase import units
from ase.io import write # REFACTOR: rm
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ..runner import register_backend, Backend, RelaxResult, is_done, mark_done

# ---------- LAMMPS backend (skeleton) ----------
@register_backend("lammps")
class LAMMPSBackend(Backend):

    def read_data(self, path: Path, cfg: Config):
        units = cfg.lammps.get("units", "metal")
        atom_style = cfg.lammps.get("atom_style", "atomic")
        atoms = read_lammps_data(cfg.data_file, units=units, atom_style=atom_style)
        return atoms

    def prepare_case(self, case_dir: Path, atoms, cfg: Config):

        # If already done, do nothing.
        if is_done(case_dir):
            return

        # Minimal: write a data file with ASE; let template refer to it.
        units = cfg.lammps.get("units", "metal")
        atom_style = cfg.lammps.get("atom_style", "atomic")
        write(case_dir / "str.data", atoms, format="lammps-data", units=units, atom_style=atom_style, force_skew="False")
 
        # Render the user template (keep it simple: {datafile} placeholder)
        tpl = Path(cfg.lammps["input_template"]).read_text()

        # set datafile
        tpl = tpl.replace("${datafile}", "str.data")

        # append json thermo to end of file
        print_line = "print '{\"pxx\":$(pxx), \"pyy\":$(pyy), \"pzz\":$(pzz), \"pyz\":$(pyz), \"pxz\":$(pxz), \"pxy\":$(pxy), \"pe\":$(pe)}' file thermo.json"
        if "thermo.json" not in tpl:               # only append once
            tpl = tpl.rstrip() + "\n" + print_line + "\n"

        # write file
        dst = case_dir / "in.min"
        if not is_done(case_dir) and not dst.exists():
            dst.write_text(tpl, encoding="utf-8")

        # Copy include files if listed
        for p in cfg.lammps.get("include_files", []):
            src = Path(p)
            dst = case_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy(src, dst)

    def run_case(self, case_dir: Path, atoms, cfg: Config) -> RelaxResult:
        if is_done(case_dir):
            return
        exe = cfg.lammps.get("exe", "lmp")
        log = (case_dir / "log.lammps")
        subprocess.run([exe, "-in", "in.min"], cwd=case_dir, check=True, stdout=log.open("w"), stderr=subprocess.STDOUT)
        text = log.read_text(errors="ignore")
        status = (case_dir / "thermo.json").is_file()
        if not status:
            raise RuntimeError("LAMMPS finished but thermo.json missing; check template/thermo line.")
        mark_done(case_dir)
        return

    def parse_case(self, case_dir: Path, atoms, cfg: Config) -> RelaxResult:

        if not is_done(case_dir):
            raise ValueError(f"Attempted to parse a running/not done case")

        data = json.loads((case_dir / "thermo.json").read_text())

        # convert lammps units to ase
        lammps_units = cfg.lammps.get("units", "metal")
        p_evA3 = convert(1.0, "pressure", lammps_units, "ASE")
        p_GPa  = p_evA3 / units.GPa
        e_eV = convert(1.0, "energy", lammps_units, "ASE")

        pe = data["pe"]*e_eV

        # LAMMPS pressure tensor is +p (compression positive) = -σ
        S = -np.array([
            [data["pxx"], data["pxy"], data["pxz"]],
            [data["pxy"], data["pyy"], data["pyz"]],
            [data["pxz"], data["pyz"], data["pzz"]],
        ], dtype=float) * p_GPa

        return RelaxResult(atoms, pe, S)
