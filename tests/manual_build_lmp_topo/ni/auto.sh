#!/bin/bash
sgl io aseconv -i in.basis_ni.dat --iformat lammps-data -o o.basis_ni.xyz --overwrite

sgl build supercell -i o.basis_ni.xyz -o o.supercell.xyz --repl 4 4 4 --overwrite --order abc

sgl io aseconv -i o.supercell.xyz --oformat lammps-data -o o.supercell.dat --overwrite

sgl build lmp-topo o.supercell.dat --file_pos="o.supercell_topo.dat" --rc="2.49" --bond=1 > o.log

python ../compare.py o.supercell_topo.dat ref.supercell_topo.dat --verbose
