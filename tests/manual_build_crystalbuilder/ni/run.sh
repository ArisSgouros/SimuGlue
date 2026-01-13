#!/bin/bash
CMD="sgl build crystalbuilder"

$CMD \
  in.basis_ni.dat "4,4,4" --file_pos="o.ni.dat" --file_dump="o.ni.lammpstrj" --file_xyz="o.ni.xyz" --rc="2.49" --bond=1 > o.log
