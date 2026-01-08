#!/bin/bash
sgl io aseconv -i in.basis_hbn.dat --iformat lammps-data -o o.basis.xyz --overwrite

sgl build supercell -i o.basis.xyz -o o.supercell.xyz --repl 4 4 2 --overwrite --order abc

sgl io aseconv -i o.supercell.xyz --oformat lammps-data -o o.supercell.dat --overwrite

sgl build lmp-topo o.supercell.dat \
    --file_pos="o.supercell_topo.dat" \
    --rc="1.44568507405082" \
    --drc=0.1 \
    --bond=1 \
    --angle=1 \
    --dihed=1 \
    --cis_trans=1 \
    --file_types="o.types" \
    --type_delimeter=' ' \

python ../compare.py deprec.o.supercell_topo.dat o.supercell_topo.dat --verbose
python ../apply_type_tag.py  o.supercell_topo.dat o.types o.supercell_topo_tagged.dat
python ../compare.py o.supercell_topo_tagged.dat ref.supercell_topo.dat --compare-tags all --verbose
