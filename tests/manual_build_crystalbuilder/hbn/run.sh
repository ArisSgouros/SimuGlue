#!/bin/bash
CMD="sgl build crystalbuilder"

$CMD \
  in.basis_hbn.dat "4,4,2" \
  --file_pos="o.hbn.dat" \
  --file_dump="o.hbn.lammpstrj" \
  --file_xyz="o.hbn.xyz" \
  --rc="1.44568507405082" \
  --drc=0.1 \
  --bond=1 \
  --angle=1 \
  --dihed=1 \
  --cis_trans=0 \
   > o.log

$CMD \
  in.basis_hbn.dat "4,4,2" \
  --file_pos="o.hbn_cis_trans.dat" \
  --file_dump="o.hbn_cis_trans.lammpstrj" \
  --file_xyz="o.hbn_cis_trans.xyz" \
  --rc="1.44568507405082" \
  --drc=0.1 \
  --bond=1 \
  --angle=1 \
  --dihed=1 \
  --cis_trans=1 \
   > o.log_cis_trans
