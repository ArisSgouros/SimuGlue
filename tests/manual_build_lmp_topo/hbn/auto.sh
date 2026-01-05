#!/bin/bash
sgl io aseconv -i in.basis_hbn.dat --iformat lammps-data -o o.basis.xyz --overwrite

sgl build supercell -i o.basis.xyz -o o.supercell_4x4x4.xyz --repl 4 4 2 --overwrite --order abc

sgl io aseconv -i o.supercell_4x4x4.xyz --oformat lammps-data -o o.supercell_4x4x4.dat --overwrite

sgl build lmp-topo o.supercell_4x4x4.dat \
    --file_pos="o.supercell_4x4x4_topo.dat" \
    --rc="1.44568507405082" \
    --drc=0.1 \
    --bond=1 \
    --angle=1 \
    --dihed=1 \
    --cis_trans=0 \
    --type_delimeter=' '

python ../compare.py o.supercell_4x4x4_topo.dat ref.supercell_4x4x4_topo.dat --compare-tags all --verbose

sgl build lmp-topo o.supercell_4x4x4.dat \
    --file_pos="o.supercell_4x4x4_topo_cistrans.dat" \
    --rc="1.44568507405082" \
    --drc=0.1 \
    --bond=1 \
    --angle=1 \
    --dihed=1 \
    --cis_trans=1 \
    --file_types="o.types_cistrans" \
    --type_delimeter=' '

python ../compare.py o.supercell_4x4x4_topo_cistrans.dat ref.supercell_4x4x4_topo_cistrans.dat --compare-tags all --verbose
