# --- Coefficients (Type indices must match in.data) ---

# Pair Interactions
pair_coeff     * *

# Bond-stretching (Morse)
bond_coeff     1 5.35 1.897 1.4518

# Angle-bending (Quartic)
angle_coeff    1 120.0 2.685 -1.78 0.0
angle_coeff    2 120.0 2.685 -1.78 0.0

## Dihedral (Harmonic)
#dihedral_coeff 1 0.109 -1 2
#dihedral_coeff 2 0.066 -1 2

# --- Setup Neighbor Style ---
neighbor       1.0 nsq
neigh_modify   once no every 1 delay 0 check yes

# --- Setup Minimization Style ---
min_style      cg
min_modify     dmax ${dmax} line quadratic

# --- Setup Output ---
thermo         1
thermo_style   custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
thermo_modify  norm no
