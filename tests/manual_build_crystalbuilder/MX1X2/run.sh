#!/bin/bash
CMD="sgl build crystalbuilder"
$CMD \
  in.basis_MX1X2_tri.dat "3,3,1" \
  --file_pos="o.pos_MX1X2.dat" \
  --file_dump="o.MX1X2.lammpstrj" \
  --file_xyz="o.MX1X2.xyz" \
  --rc="0.763762616" \
  --drc=0.1 \
  --bond=1 \
  --angle=1 \
  --angle_symmetry 1 \
  --file_types="o.types.dat" \
  --type_delimeter=" " \
   > o.log
