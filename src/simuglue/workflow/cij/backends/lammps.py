from __future__ import annotations

import json
import os, shlex, subprocess
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
)

def _to_argv(x) -> list[str]:
    """Accept list/tuple/str/None and return a list of argv tokens."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(t) for t in x]
    if isinstance(x, str):
        return shlex.split(x)
    return [str(x)]

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


    def run_case(self, case_dir: Path, cfg: Config) -> None:
        """Run a single LAMMPS case with .running / .done / .failed markers."""
        running = case_dir / ".running"
        done    = case_dir / ".done"
        failed  = case_dir / ".failed"
        log     = case_dir / "log.lammps"

        # --- skip conditions ---
        if done.exists():
            return
        if running.exists():
            print(f"[skip] {case_dir} is already running.")
            return

        log.parent.mkdir(parents=True, exist_ok=True)

        # --- build command ---
        exe  = cfg.lammps.get("exe", "lmp")
        para = cfg.lammps.get("para", None)   # optional: "mpirun -np 8"
        post = cfg.lammps.get("post", None)   # optional trailing args
        env  = dict(os.environ)
        env.update(cfg.lammps.get("env", {}))

        para_argv = _to_argv(para)
        exe_argv  = _to_argv(exe)
        post_argv = _to_argv(post)
        cmd = [*para_argv, *exe_argv, "-in", "in.min", *post_argv]

        # --- mark as running ---
        running.write_text("running\n", encoding="utf-8")
        if failed.exists():
            failed.unlink()  # remove old failure marker

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

            # --- post-run sanity check ---
            thermo_json = case_dir / "thermo.json"
            if thermo_json.is_file():
                done.write_text("done\n", encoding="utf-8")
            else:
                msg = "LAMMPS finished but thermo.json missing; check template/thermo line."
                failed.write_text(msg + "\n", encoding="utf-8")
                print(f"[warn] {msg}")

        except subprocess.CalledProcessError as e:
            msg = f"LAMMPS failed with exit code {e.returncode}\n"
            failed.write_text(msg, encoding="utf-8")
            print(f"[error] {msg.strip()}")

        except Exception as e:
            failed.write_text(str(e), encoding="utf-8")
            print(f"[error] Unexpected failure in {case_dir}: {e}")

        finally:
            running.unlink(missing_ok=True)

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

        return RelaxResult(energy=pe, stress=S)

