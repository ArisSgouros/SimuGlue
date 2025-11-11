from __future__ import annotations

import json
import os, shlex, subprocess
from pathlib import Path, PurePosixPath

import numpy as np
from ase import units
from ase.calculators.lammps.unitconvert import convert
from ase.io import read, write

from ..config import Config
from ..markers import prepare_run, finalize_success, finalize_failure
from ..registry import (
    register_backend,
    Backend,
    RelaxResult,
    is_done,
)

from simuglue.quantum_espresso.pwi_update import update_qe_input
from simuglue.mechanics.voigt import voigt6_to_stress_tensor

def _to_argv(x) -> list[str]:
    """Accept list/tuple/str/None and return a list of argv tokens."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(t) for t in x]
    if isinstance(x, str):
        return shlex.split(x)
    return [str(x)]

# ---------- QE backend (skeleton) ----------
@register_backend("qe")
class QEBackend(Backend):

    def read_data(self, cfg: Config):
        qe_input = cfg.qe.get("input", None)
        return read(qe_input, format='espresso-in', index=0)

    def write_data(self, path: Path, atoms, cfg: Config, case_tag: str | None) -> None:
        input_path = Path(cfg.qe.get("input", None))
        input_text = input_path.read_text(encoding="utf-8")

        prefix = cfg.qe.get("prefix", None)
        outdir_base = cfg.qe.get("outdir", None)

        outdir_for_qe = None
        if outdir_base:
            # Build per-case outdir: <outdir_base>/<prefix>/<case_tag>
            outdir_path = Path(outdir_base)
            if prefix:
                outdir_path = outdir_path / prefix
            if case_tag:
                outdir_path = outdir_path / case_tag

            outdir_path = outdir_path.resolve()
            outdir_path.mkdir(parents=True, exist_ok=True)
            outdir_for_qe = outdir_path.as_posix()  # POSIX slashes for QE

        qe_text = update_qe_input(
            input_text,
            cell=atoms.get_cell().array,
            positions=atoms.get_positions(),
            symbols=atoms.get_chemical_symbols(),
            prefix=prefix,
            outdir=outdir_for_qe,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(qe_text, encoding="utf-8")

    def prepare_case(self, case_dir: Path, atoms, cfg: Config) -> None:
        if is_done(case_dir):
            return
        case_tag = case_dir.name or "case"
        self.write_data(case_dir / "qe.in", atoms, cfg, case_tag=case_tag)

    def run_case(self, case_dir: Path, cfg: Config) -> None:
        """Run a single QE case with shared markers helper."""
        if not prepare_run(case_dir):
            return

        log = case_dir / "qe.out"
        log.parent.mkdir(parents=True, exist_ok=True)

        exe  = cfg.qe.get("exe", "pw.x")
        para = cfg.qe.get("para", None)
        post = cfg.qe.get("post", None)
        env  = dict(os.environ)
        env.update(cfg.qe.get("env", {}))

        cmd = [*_to_argv(para), *_to_argv(exe), "-in", "qe.in", *_to_argv(post)]

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

            txt = log.read_text(errors="ignore")
            if "JOB DONE." in txt or "convergence has been achieved" in txt:
                finalize_success(case_dir)
            else:
                finalize_failure(case_dir, "not converged")

        except subprocess.CalledProcessError as e:
            finalize_failure(case_dir, f"QE failed with exit code {e.returncode}")
        except Exception as e:
            finalize_failure(case_dir, f"Unexpected failure: {e}")


    def parse_case(self, case_dir: Path, cfg: Config) -> RelaxResult:
        if not is_done(case_dir):
            raise ValueError(f"Case not marked done: {case_dir}")

        out_path = case_dir / cfg.qe.get("outfile", "qe.out")
        if not out_path.is_file():
            raise FileNotFoundError(f"QE output not found: {out_path}")

        # Last step from QE output
        atoms = read(out_path, format="espresso-out", index=-1)

        # Energy in eV (ASE handles units)
        pe = float(atoms.get_potential_energy())

        # Stress in ASE units: eV/Å^3
        try:
            S = np.array(atoms.get_stress(voigt=False), float)
        except TypeError:
            # Fallback: Voigt -> 3x3 using your helper
            s6 = np.array(atoms.get_stress(voigt=True), float)
            S = voigt6_to_stress_tensor(s6)

        # Enforce symmetry numerically
        S = 0.5 * (S + S.T)

        return RelaxResult(energy=pe, stress=S)
