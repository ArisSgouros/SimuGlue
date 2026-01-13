#!/bin/bash
sgl io aseconv -i in.basis_MX1X2_tri.dat --iformat lammps-data -o o.basis.xyz --overwrite

sgl build supercell -i o.basis.xyz -o o.supercell.xyz --repl 3 3 1 --overwrite --order abc

sgl io aseconv -i o.supercell.xyz --oformat lammps-data -o o.supercell.dat --overwrite

sgl build lmp-topo \
    -i o.supercell.dat \
    -o o.supercell_topo.dat \
    --types-out o.types \
    --rc "0.763762616" \
    --drc 0.1 \
    --bonds --angles \
    --angle-symmetry \
    --type-delimiter ' ' \

python ../apply_type_tag.py  o.supercell_topo.dat o.types o.supercell_topo_tagged.dat
python ../compare.py o.supercell_topo_tagged.dat ref.supercell_topo.dat --compare-tags all --verbose
