#!/bin/bash
sgl io aseconv -i in.basis_ni.dat --iformat lammps-data -o o.basis_ni.xyz --overwrite

sgl build crystalbuilder -i o.basis_ni.xyz -o o.supercell_4x4x4.xyz --repl 4 4 4

sgl io aseconv -i o.supercell_4x4x4.xyz --oformat lammps-data -o o.supercell_4x4x4.dat --overwrite
