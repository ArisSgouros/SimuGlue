import numpy as np

def lammps_box_to_ase_cell(lx, ly, lz, xy=0.0, xz=0.0, yz=0.0):
    a = np.array([lx, 0.0, 0.0])
    b = np.array([xy, ly, 0.0])
    c = np.array([xz, yz, lz])
    return np.vstack([a, b, c])   # shape (3,3); rows are a,b,c

import numpy as np

def ase_cell_to_lammps_triclinic(cell, *, wrap_tilts=True, atol=1e-12):
    """
    Convert an arbitrary ASE-style cell (3x3, rows are a,b,c) to the
    LAMMPS triclinic box parameters (lx, ly, lz, xy, xz, yz).

    Mapping (LAMMPS convention):
        a = (lx,  0,  0)
        b = (xy, ly,  0)
        c = (xz, yz, lz)

    Parameters
    ----------
    cell : (3,3) array-like
        Rows are the lattice vectors a,b,c in Å.
    wrap_tilts : bool, default True
        If True, wrap tilts into the preferred LAMMPS ranges:
            xy, xz in (-0.5*lx, 0.5*lx]
            yz in (-0.5*ly, 0.5*ly]
    atol : float
        Tolerance for degenerate checks.

    Returns
    -------
    lx, ly, lz, xy, xz, yz : floats
        LAMMPS triclinic parameters in Å.
    """
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("cell must be shape (3,3) with rows a,b,c")

    a = cell[0]
    b = cell[1]
    c = cell[2]

    ax = np.linalg.norm(a)
    if ax < atol:
        raise ValueError("Degenerate cell: |a| ≈ 0")

    # Unit vector along a
    a_hat = a / ax

    # Decompose b into components parallel/perpendicular to a
    bx = float(np.dot(b, a_hat))
    b_perp = b - bx * a_hat
    by = float(np.linalg.norm(b_perp))
    if by < atol:
        raise ValueError("Degenerate cell: a and b nearly collinear (by≈0)")

    # Decompose c: first along a (cx), then extract the component along
    # the unit vector normal to a within the a–b plane to get cy, finally cz.
    cx = float(np.dot(c, a_hat))

    # Unit vector along b_perp
    b_perp_hat = b_perp / by
    # cy = projection of c onto b_perp_hat
    cy = float(np.dot(c, b_perp_hat))

    # Remaining component gives lz
    c_rem = c - cx * a_hat - cy * b_perp_hat
    cz = float(np.linalg.norm(c_rem))
    if cz < atol:
        # Perfectly planar cell (2D slab) is fine; cz≈0 is allowed in 2D,
        # but raise for true 3D triclinic usage
        # You can downgrade to a warning if you expect 2D cells.
        pass

    # LAMMPS parameters:
    lx, ly, lz = ax, by, cz
    xy, xz, yz = bx, cx, cy

    if wrap_tilts:
        # Wrap into preferred LAMMPS ranges:
        #   xy,xz in (-0.5*lx, 0.5*lx],  yz in (-0.5*ly, 0.5*ly]
        def wrap(v, L):
            return v - np.rint(v / L) * L if L > atol else v
        xy = wrap(xy, lx)
        xz = wrap(xz, lx)
        yz = wrap(yz, ly)

    return lx, ly, lz, xy, xz, yz

