#!/bin/bash
CMD="sgl build crystalbuilder"

$CMD \
  in.basis_nacl.dat "5,5,5" --file_pos="o.nacl.dat" --file_dump="o.nacl.lammpstrj" --file_xyz="o.nacl.xyz" --rc="2.8201" --bond=1 > o.log
