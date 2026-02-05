from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from ase import Atoms
except Exception:  # pragma: no cover
    Atoms = object  # type: ignore[misc,assignment]


KFormat = Literal["2pi_over_a", "reciprocal"]
ImagPolicy = Literal["clip", "signed"]
ReorderPolicy = Literal["auto", "none"]


@dataclass(frozen=True)
class DispersionResult:
    """Return container for a dispersion calculation."""
    kpoints: NDArray[np.float64]               # (nk, 3) in user input convention
    frequencies: NDArray[np.float64]           # (nk, nmodes)
    eigenvectors: NDArray[np.complex128] | None  # (nk, nmodes, nmodes) if requested

    # Useful diagnostics / reproducibility
    atom_order: NDArray[np.int64]              # (natom,) permutation applied to dynmat/atoms
    nbasis: int
    ncell: int
    H_prim: NDArray[np.float64]                # (3,3) primitive cell vectors as columns (Å)
    a_lat: float                               # used for k_format="2pi_over_a"
    freq_units: str

def validate_canonical_atom_order(
    cell_lin: NDArray[np.int64],
    types: NDArray[np.int64],
    u: NDArray[np.float64],
    *,
    ncell: int,
    nbasis: int,
    order_decimals: int = 6,
) -> None:
    """
    Validate that the *current* atom order is already canonical:
      - atoms grouped by cell_lin = 0,0,...,1,1,...,ncell-1
      - within each cell, basis ordering is consistent across all cells
        (same 'type' sequence AND same fractional coords up to rounding)

    This is required for reorder="none" because the tensor reshape assumes
    [cell][basis][xyz] layout.
    """
    natom = cell_lin.size
    if natom != ncell * nbasis:
        raise RuntimeError(
            f"Canonical-order check failed: natom={natom} but ncell*nbasis={ncell*nbasis}."
        )

    expected = np.repeat(np.arange(ncell, dtype=np.int64), nbasis)
    if not np.array_equal(cell_lin, expected):
        raise RuntimeError(
            "reorder='none' requires atoms already grouped by cell in canonical order.\n"
            "Expected cell_lin = [0..0, 1..1, ..., ncell-1..ncell-1] (each repeated nbasis)."
        )

    types_by_cell = types.reshape(ncell, nbasis)
    u_by_cell = u.reshape(ncell, nbasis, 3)
    u0 = np.round(u_by_cell[0], decimals=order_decimals)
    t0 = types_by_cell[0].copy()

    # basis must be consistent across all cells
    if not np.all(types_by_cell == t0[None, :]):
        raise RuntimeError(
            "reorder='none' requires identical basis type ordering in every cell.\n"
            "At least one cell has a different type sequence than cell 0."
        )

    u_round = np.round(u_by_cell, decimals=order_decimals)
    if not np.all(u_round == u0[None, :, :]):
        raise RuntimeError(
            "reorder='none' requires identical basis fractional coordinates in every cell.\n"
            "At least one cell differs from cell 0 (after rounding)."
        )



def wrap01(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap to [0,1)."""
    return x - np.floor(x)


def infer_cell_and_basis(
    positions: NDArray[np.float64],
    cell_H: NDArray[np.float64],
    N_a: int,
    N_b: int,
    N_c: int,
    eps: float = 1e-6,
) -> Tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    """
    Infer primitive-cell indices (a,b,c), within-cell fractional coordinates u,
    and linear cell index l=a+Na*(b+Nb*c).
    """
    if N_a <= 0 or N_b <= 0 or N_c <= 0:
        raise ValueError("Replication counts Na,Nb,Nc must be positive integers")

    H_prim = cell_H @ np.diag([1.0 / N_a, 1.0 / N_b, 1.0 / N_c])
    Hinv = np.linalg.inv(H_prim)

    s = (Hinv @ positions.T).T
    cell = np.floor(s + eps).astype(np.int64)
    u = wrap01(s - cell)

    cell[:, 0] %= N_a
    cell[:, 1] %= N_b
    cell[:, 2] %= N_c

    cell_lin = cell[:, 0] + N_a * (cell[:, 1] + N_b * cell[:, 2])
    return cell, u, cell_lin


def atom_order_cell_basis_type(
    types: NDArray[np.int64],
    u: NDArray[np.float64],
    cell_lin: NDArray[np.int64],
    decimals: int = 6,
) -> NDArray[np.int64]:
    """Permutation that sorts atoms by (cell_lin, type, u)."""
    if types.ndim != 1:
        raise ValueError("types must be 1D")
    if u.shape != (types.size, 3):
        raise ValueError("u must have shape (natom,3)")
    if cell_lin.shape != (types.size,):
        raise ValueError("cell_lin must have shape (natom,)")

    ur = np.round(u, decimals=decimals)
    return np.lexsort((ur[:, 2], ur[:, 1], ur[:, 0], types, cell_lin))


def permute_dynmat(
    dyn_mat: NDArray[np.float64],
    atom_order: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Apply an atom permutation to a 3N x 3N dynamical matrix (3 DOF per atom)."""
    atom_order = np.asarray(atom_order, dtype=np.int64)
    natom = atom_order.size

    dyn_mat = np.asarray(dyn_mat)
    if dyn_mat.shape != (3 * natom, 3 * natom):
        raise ValueError(f"dyn_mat has shape {dyn_mat.shape}, expected {(3*natom, 3*natom)}")

    dof = np.empty(3 * natom, dtype=np.int64)
    dof[0::3] = 3 * atom_order + 0
    dof[1::3] = 3 * atom_order + 1
    dof[2::3] = 3 * atom_order + 2

    return dyn_mat[np.ix_(dof, dof)]


def wrap_mi(d: np.ndarray, N: int) -> np.ndarray:
    """
    Minimum-image wrap of integer displacements.

    Important: for even N, +/-N/2 are both minimum-image solutions.
    We preserve the sign from the raw displacement so that:
        d(l,L) = -d(L,l)
    which guarantees phase antisymmetry and keeps D(k) Hermitian
    when the real-space dynmat is symmetric.
    """
    if N <= 0:
        raise ValueError("N must be positive")

    d = np.asarray(d, dtype=np.int64)
    raw = d.copy()

    half = N // 2
    wrapped = (d + half) % N - half

    if N % 2 == 0:
        # Tie case: wrapped == -N/2 could have come from raw == +N/2.
        # Flip those to +N/2 to preserve antisymmetry.
        wrapped = np.where((wrapped == -half) & (raw > 0), half, wrapped)

    return wrapped



def build_cell_displacements(
    N_a: int, N_b: int, N_c: int
) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return minimum-image integer displacements (da, db, dc) for all cell pairs."""
    ncell = N_a * N_b * N_c
    l = np.arange(ncell, dtype=np.int64)

    ca = l % N_a
    cb = (l // N_a) % N_b
    cc = (l // (N_a * N_b)) % N_c

    da = ca[None, :] - ca[:, None]
    db = cb[None, :] - cb[:, None]
    dc = cc[None, :] - cc[:, None]

    da = wrap_mi(da, N_a)
    db = wrap_mi(db, N_b)
    dc = wrap_mi(dc, N_c) if N_c > 1 else np.zeros_like(dc)

    return da, db, dc


def frequency_conversion(freq_units: str) -> float:
    """
    Conversion factor from sqrt([eV]/([g/mol]*Å^2)) to the requested unit.
    LAMMPS 'metal' base: eV, Å, g/mol.
    """
    N_A = 6.02214076e23
    eV = 1.602176634e-19
    kg_per_g = 1e-3
    m_per_A = 1e-10

    conv_s2 = eV / ((kg_per_g / N_A) * (m_per_A**2))  # s^-2

    u = freq_units.strip().lower()
    if u == "thz":
        return np.sqrt(conv_s2) / (2.0 * np.pi) * 1e-12
    if u in ("cm", "cm-1", "cm^-1"):
        return np.sqrt(conv_s2) / (2.0 * np.pi) * 1e-12 * 33.35640951981521  # THz->cm^-1
    raise ValueError(f"Unsupported freq_units: {freq_units!r} (use 'Thz' or 'cm-1')")


def k_to_cartesian(
    k_vec: NDArray[np.float64],
    H_prim: NDArray[np.float64],
    *,
    k_format: KFormat,
    a_lat: float,
) -> NDArray[np.float64]:
    """Convert a user k-vector to Cartesian (1/Å)."""
    if k_format == "2pi_over_a":
        return (2.0 * np.pi / a_lat) * np.asarray(k_vec, dtype=np.float64)

    # k_format == "reciprocal"
    B = 2.0 * np.pi * np.linalg.inv(H_prim).T  # reciprocal basis as columns
    return B @ np.asarray(k_vec, dtype=np.float64)


def build_Dk(
    dyn_tensor: NDArray[np.complex128],
    phase: NDArray[np.complex128],
    ncell: int,
) -> NDArray[np.complex128]:
    """Compute D_k from the 6D supercell tensor and a (ncell,ncell) phase matrix."""
    D_k4 = np.einsum("laiLbj,lL->aibj", dyn_tensor, phase, optimize=True)
    D_k = D_k4.reshape(dyn_tensor.shape[1] * 3, dyn_tensor.shape[4] * 3)
    D_k /= ncell
    return D_k


def eigen_frequencies(
    D_k: NDArray[np.complex128],
    w_conv: float,
    *,
    imag_policy: ImagPolicy = "clip",
    hermitian_project: bool = True,
) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Diagonalize D_k and return frequencies (+ eigenvectors)."""
    if hermitian_project:
        D_k = 0.5 * (D_k + D_k.conj().T)

    evals, evecs = np.linalg.eigh(D_k)
    lam = evals.real

    if imag_policy == "clip":
        w = np.sqrt(np.clip(lam, 0.0, None)) * w_conv
    elif imag_policy == "signed":
        w = np.sign(lam) * np.sqrt(np.abs(lam)) * w_conv
    else:
        raise ValueError(f"Unknown imag_policy: {imag_policy!r} (use 'clip' or 'signed')")

    return w.astype(np.float64, copy=False), evecs


def validate_cell_occupancy(
    cell_lin: NDArray[np.int64],
    types: NDArray[np.int64],
    ncell: int,
) -> Tuple[int, NDArray[np.int64]]:
    """Check each cell has exactly one atom per type; return (nbasis, unique_types)."""
    unique_types = np.unique(types)
    nbasis = unique_types.size
    counts = np.bincount(cell_lin, minlength=ncell)

    if counts.min() != nbasis or counts.max() != nbasis:
        raise RuntimeError(
            "Unexpected cell occupancy.\n"
            f"  min/mean/max atoms per cell = {counts.min()}/{counts.mean():.3f}/{counts.max()}\n"
            f"  expected exactly nbasis = {nbasis} atoms per cell (one per type)\n"
            "Check that (Na,Nb,Nc) match the structure and that the basis is encoded by LAMMPS 'type'."
        )

    return nbasis, unique_types


def compute_phonon_dispersion(
    atoms: Atoms,
    dyn_mat: NDArray[np.float64],
    kpoints: NDArray[np.float64],
    *,
    cells: Tuple[int, int, int],
    freq_units: str = "Thz",
    k_format: KFormat = "2pi_over_a",
    a_lat: float | None = None,
    imag_policy: ImagPolicy = "clip",
    hermitian_project: bool = True,
    return_eigenvectors: bool = False,
    order_decimals: int = 6,
    reorder: ReorderPolicy = "auto",
) -> DispersionResult:
    """
    Pure computation:
      - takes already-loaded structure + dynmat + kpoints
      - returns frequencies (and optionally eigenvectors) as arrays
    """
    N_a, N_b, N_c = (int(cells[0]), int(cells[1]), int(cells[2]))
    if N_a <= 0 or N_b <= 0 or N_c <= 0:
        raise ValueError("cells must be positive integers (Na,Nb,Nc)")

    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    types = np.asarray(atoms.arrays["type"], dtype=np.int64)
    cell_H = np.transpose(np.asarray(atoms.cell.array, dtype=np.float64))  # columns

    natom = types.size
    ncell = N_a * N_b * N_c


    cell, u, cell_lin = infer_cell_and_basis(positions, cell_H, N_a, N_b, N_c)
    nbasis, unique_types = validate_cell_occupancy(cell_lin, types, ncell)

    if natom != ncell * nbasis:
        raise RuntimeError(f"Inconsistent counts: natom={natom} but ncell*nbasis={ncell*nbasis}")

    if reorder == "auto":
        # canonicalize by (cell, type, u) to ensure consistent basis ordering
        order = atom_order_cell_basis_type(types, u, cell_lin, decimals=order_decimals)

    elif reorder == "none":
        # trust user ordering but validate it is canonical for tensor reshape
        # (no atom reorder, no dynmat permute)
        validate_canonical_atom_order(
            cell_lin=cell_lin,
            types=types,
            u=u,
            ncell=ncell,
            nbasis=nbasis,
            order_decimals=order_decimals,
        )
        order = np.arange(natom, dtype=np.int64)

    else:
        raise ValueError(f"Unknown reorder policy: {reorder!r} (use 'auto' or 'none')")

    # permute dynmat to match ordering (identity if reorder='none')
    dyn_real = permute_dynmat(np.asarray(dyn_mat, dtype=np.float64), order)
    dyn = dyn_real.astype(np.complex128, copy=False)
    dyn_tensor = dyn.reshape(ncell, nbasis, 3, ncell, nbasis, 3)

    # Optional: extra sanity check for auto mode
    if reorder == "auto":
        types_ord = types[order]
        types_by_cell = types_ord.reshape(ncell, nbasis)
        if not np.all(np.sort(types_by_cell, axis=1) == unique_types[None, :]):
            raise RuntimeError(
                "After auto reordering, at least one cell does not contain exactly one of each type.\n"
                "This suggests the basis is not uniquely encoded by 'type' or cell inference failed."
            )

    # primitive cell vectors
    H_prim = cell_H @ np.diag([1.0 / N_a, 1.0 / N_b, 1.0 / N_c])
    a1 = H_prim[:, 0]
    a_lat_use = float(np.linalg.norm(a1)) if a_lat is None else float(a_lat)
    if a_lat_use <= 0:
        raise ValueError("a_lat must be positive")

    # precompute minimum-image displacements
    da, db, dc = build_cell_displacements(N_a, N_b, N_c)

    k_list = np.asarray(kpoints, dtype=np.float64)
    if k_list.ndim != 2 or k_list.shape[1] != 3:
        raise ValueError(f"kpoints must have shape (nk,3); got {k_list.shape}")
    nk = k_list.shape[0]

    w_conv = frequency_conversion(freq_units)
    nmodes = 3 * nbasis

    freqs = np.empty((nk, nmodes), dtype=np.float64)
    eigvecs = np.empty((nk, nmodes, nmodes), dtype=np.complex128) if return_eigenvectors else None

    for ik, k_vec in enumerate(k_list):
        k_cart = k_to_cartesian(k_vec, H_prim, k_format=k_format, a_lat=a_lat_use)

        ka1 = float(np.dot(k_cart, a1))
        ka2 = float(np.dot(k_cart, H_prim[:, 1]))
        ka3 = float(np.dot(k_cart, H_prim[:, 2]))

        phase = np.exp(-1j * (da * ka1 + db * ka2 + dc * ka3))
        D_k = build_Dk(dyn_tensor, phase, ncell=ncell)

        w, evec = eigen_frequencies(
            D_k,
            w_conv=w_conv,
            imag_policy=imag_policy,
            hermitian_project=hermitian_project,
        )

        freqs[ik, :] = w
        if eigvecs is not None:
            eigvecs[ik, :, :] = evec

    return DispersionResult(
        kpoints=k_list,
        frequencies=freqs,
        eigenvectors=eigvecs,
        atom_order=order,
        nbasis=nbasis,
        ncell=ncell,
        H_prim=H_prim,
        a_lat=a_lat_use,
        freq_units=freq_units,
    )

