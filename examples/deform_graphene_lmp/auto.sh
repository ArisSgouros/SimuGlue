#!/bin/bash
# 1. Convert basis to ASE format
sgl io aseconv -i in.basis.dat --iformat lammps-data -o tmp.basis.xyz

# 2. Build supercell
sgl build supercell -i tmp.basis.xyz -o tmp.supercell.xyz --repl 10 10 1

# 3. Convert back to LAMMPS data
sgl io aseconv -i tmp.supercell.xyz --oformat lammps-data -o tmp.supercell.dat

# 4. Generate Topology (Bonds and Angles)
sgl build lmp-topo \
    -i tmp.supercell.dat \
    -o tmp.supercell_topo.dat \
    --rc 1.42 \
    --drc 0.1 \
    --bonds --angles \
    --types-out tmp.types \
    --type-delimiter ' ' \

# 5. Run the updated deformation workflow
sgl wf deform all -c deform_2d.yaml
