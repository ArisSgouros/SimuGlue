#!/bin/bash
sgl io aseconv -i in.basis.dat --iformat lammps-data -o tmp.basis.xyz

sgl build supercell -i tmp.basis.xyz -o tmp.supercell.xyz --repl 2 2 1

sgl io aseconv -i tmp.supercell.xyz --oformat lammps-data -o tmp.supercell.dat

sgl build lmp-topo \
    -i tmp.supercell.dat \
    -o tmp.supercell_topo.dat \
    --rc 1.44568507405082 \
    --drc 0.1 \
    --bonds --angles \
    --types-out tmp.types \
    --type-delimiter ' ' \

sgl wf cij all --config cij_2d.yaml
