from __future__ import annotations

import argparse
import gzip
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from simuglue.ase_patches.lammpsdata import read_lammps_data
from simuglue.phonon.dispersion import compute_phonon_dispersion

LOG = logging.getLogger("sgl.phonon_dispersion")


def load_kpoints(path: str | Path) -> NDArray[np.float64]:
    k = np.loadtxt(path, dtype=np.float64, comments="#", ndmin=2)
    if k.shape[1] == 2:
        k = np.column_stack([k, np.zeros((k.shape[0],), dtype=np.float64)])
    if k.shape[1] < 3:
        raise ValueError(f"k-point file {path} must have at least 2 or 3 columns; got shape {k.shape}")
    if k.shape[1] > 3:
        LOG.warning("k-point file has %d columns; using the first 3.", k.shape[1])
        k = k[:, :3]
    return k


def load_dynmat(path: str | Path, expected_size: int) -> NDArray[np.float64]:
    """
    Load a 3N x 3N dynamical matrix from:
      - .npy
      - .npz (uses key 'dynmat' or first array)
      - ASCII text (optionally .gz)
    """
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".npy"):
        dyn = np.asarray(np.load(path), dtype=np.float64)
    elif suffixes.endswith(".npz"):
        data = np.load(path)
        key = "dynmat" if "dynmat" in data.files else data.files[0]
        dyn = np.asarray(data[key], dtype=np.float64)
    else:
        opener = gzip.open if suffixes.endswith(".gz") else open
        with opener(path, "rt") as f:
            dyn = np.loadtxt(f, dtype=np.float64, comments="#")

    dyn = np.asarray(dyn, dtype=np.float64).reshape(-1)
    if dyn.size != expected_size:
        raise ValueError(f"Dynamical matrix in {path} has {dyn.size} numbers; expected {expected_size}")

    n = int(round(np.sqrt(dyn.size)))
    if n * n != dyn.size:
        raise ValueError(f"Dynamical matrix size {dyn.size} is not a perfect square")

    return dyn.reshape(n, n)


def write_dispersion(
    out_path: str | Path,
    kpoints: NDArray[np.float64],
    freqs: NDArray[np.float64],
    *,
    output_format: str = "with_k",  # "with_k" or "freq_only"
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "freq_only":
        np.savetxt(out, freqs, fmt="%15.7e", delimiter="\t")
        return

    nmodes = freqs.shape[1]
    data = np.column_stack([kpoints, freqs])
    header = "kx\tky\tkz\t" + "\t".join([f"w{m+1}" for m in range(nmodes)])
    np.savetxt(out, data, fmt="%15.7e", delimiter="\t", header=header)


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "sgl lmp phonon-dispersion",
        description="Compute phonon dispersion from a LAMMPS dynamical matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("cells", type=str, help="Supercell replications as 'Na,Nb,Nc' (e.g. 20,10,1).")

    p.add_argument("-s", "--structure", default="o.struct.nc", help="LAMMPS data file (atom_style full).")
    p.add_argument("-d", "--dyn-mat", dest="dyn_mat", default="o.dynmat.gz", help="Dynamical matrix file (.gz/.txt or .npy/.npz).")
    p.add_argument("-k", "--k-point", dest="k_point", default="i.k_point.dat", help="k-point list file.")
    p.add_argument("-u", "--freq-units", dest="freq_units", default="Thz", help="Output frequency units: Thz or cm-1.")
    p.add_argument("-o", "--output", default="1.phonon_disp.tsv", help="Output file (TSV).")

    p.add_argument("--k-format", default="2pi_over_a", choices=["2pi_over_a", "reciprocal"],
                   help="Interpretation of k-point columns.")
    p.add_argument("--a-lat", type=float, default=None,
                   help="Lattice parameter a (Å) for k_format=2pi_over_a. Default: |a1| from primitive cell.")

    p.add_argument("--output-format", choices=["with_k", "freq_only"], default="with_k",
                   help="Write kx ky kz columns or only frequencies (legacy-style).")

    p.add_argument("--imag", choices=["clip", "signed"], default="clip",
                   help="How to treat negative eigenvalues (imaginary modes).")
    p.add_argument("--no-hermitian-project", action="store_true",
                   help="Disable Hermitian projection D=(D+D†)/2 (not recommended).")

    p.add_argument("--save-eigenvectors", action="store_true", help="Save eigenvectors as .npz (can be large).")
    p.add_argument("--eigenvectors-out", default="1.phonon_eigvecs.npz", help="Eigenvector output path (when enabled).")

    p.add_argument("--debug", action="store_true", help="Verbose logging.")
    return p


def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s: %(message)s")

    try:
        Na, Nb, Nc = (int(x) for x in args.cells.split(","))
    except Exception as e:
        raise ValueError("cells must be formatted as 'Na,Nb,Nc'") from e

    atoms = read_lammps_data(args.structure, atom_style="full", units="metal")
    natom = len(atoms)
    expected_size = (3 * natom) * (3 * natom)

    kpoints = load_kpoints(args.k_point)
    dynmat = load_dynmat(args.dyn_mat, expected_size=expected_size)

    res = compute_phonon_dispersion(
        atoms,
        dynmat,
        kpoints,
        cells=(Na, Nb, Nc),
        freq_units=args.freq_units,
        k_format=args.k_format,
        a_lat=args.a_lat,
        imag_policy=args.imag,
        hermitian_project=(not args.no_hermitian_project),
        return_eigenvectors=bool(args.save_eigenvectors),
    )

    write_dispersion(args.output, res.kpoints, res.frequencies, output_format=args.output_format)
    LOG.info("Wrote dispersion to %s", args.output)

    if res.eigenvectors is not None:
        np.savez_compressed(args.eigenvectors_out,
                            eigenvectors=res.eigenvectors,
                            kpoints=res.kpoints,
                            frequencies=res.frequencies)
        LOG.info("Wrote eigenvectors to %s", args.eigenvectors_out)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
