import numpy as np
from simuglue.io.lammps_cell import lammps_box_to_ase_cell, ase_cell_to_lammps_triclinic

def rand_cell():
    # Random triclinic but well-conditioned
    R = np.random.default_rng(0).normal(size=(3,3))
    # Make it upper-ish triangular to avoid degeneracy, then perturb
    U = np.triu(np.abs(R)) + 5*np.eye(3)
    return U + 0.1*np.random.default_rng(1).normal(size=(3,3))

def test_roundtrip_lammps_to_ase_to_lammps():
    lx, ly, lz = 10.0, 8.0, 12.0
    xy, xz, yz = 3.1, -1.7, 2.6
    cell = lammps_box_to_ase_cell(lx, ly, lz, xy, xz, yz)
    lx2, ly2, lz2, xy2, xz2, yz2 = ase_cell_to_lammps_triclinic(cell, wrap_tilts=True)
    # Lengths match exactly; tilts match modulo wrapping
    assert np.allclose([lx, ly, lz], [lx2, ly2, lz2], atol=1e-10)
    # Bring originals into wrapped range and compare
    def wrap(v, L): return v - np.rint(v/L)*L
    assert np.allclose([wrap(xy,lx), wrap(xz,lx), wrap(yz,ly)],
                       [xy2, xz2, yz2], atol=1e-10)


#def test_roundtrip_ase_to_lammps_to_ase():
#    cell = rand_cell()
#    lx, ly, lz, xy, xz, yz = ase_cell_to_lammps_triclinic(cell, wrap_tilts=True)
#    cell2 = lammps_box_to_ase_cell(lx, ly, lz, xy, xz, yz)
#    # Same cell up to standard LAMMPS tilt wrapping choices
#    assert np.allclose(cell, cell2, atol=1e-8)

def test_roundtrip_ase_to_lammps_to_ase():
    cell = rand_cell()
    lx, ly, lz, xy, xz, yz = ase_cell_to_lammps_triclinic(cell, wrap_tilts=True)
    cell2 = lammps_box_to_ase_cell(lx, ly, lz, xy, xz, yz)

    G1 = cell @ cell.T
    G2 = cell2 @ cell2.T
    assert np.allclose(G1, G2, atol=1e-8)


def canonicalize_cell_upper(cell, atol=1e-12):
    a, b, c = np.asarray(cell, float)
    ax = np.linalg.norm(a); a_hat = a/ax
    bx = float(np.dot(b, a_hat)); b_perp = b - bx*a_hat
    by = float(np.linalg.norm(b_perp)); b_hat = b_perp/by
    cx = float(np.dot(c, a_hat))
    cy = float(np.dot(c, b_hat))
    cz = float(np.linalg.norm(c - cx*a_hat - cy*b_hat))
    return np.array([[ax, 0.0, 0.0],
                     [bx, by, 0.0],
                     [cx, cy, cz]])

def test_roundtrip_ase_to_lammps_to_ase_canon():
    cell = rand_cell()
    lx, ly, lz, xy, xz, yz = ase_cell_to_lammps_triclinic(cell, wrap_tilts=True)
    cell2 = lammps_box_to_ase_cell(lx, ly, lz, xy, xz, yz)
    canon = canonicalize_cell_upper(cell)
    assert np.allclose(canon, cell2, atol=1e-8)
