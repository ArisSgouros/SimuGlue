#!/bin/bash
CMD="sgl build crystalbuilder"
$CMD in.basis_graphene.dat "20,10,1" --file_pos="o.20_10.dat" --file_dump="o.20_10.lammpstrj" --file_xyz="o.20_10.xyz" --rc="1.5" --drc=0.1 --bond=1 --angle=1 --dihed=1 > o.20_10.log
$CMD in.basis_graphene.dat "20,10,1" --file_pos="o.20_10xy.dat" --file_dump="o.20_10xy.lammpstrj" --file_xyz="o.20_10xy.xyz" --rc="1.5" --drc=0.1 --bond=1 --angle=1 --dihed=1 --grid="xy" > o.20_10xy.log
$CMD in.basis_graphene.dat "30,4,1"  --file_pos="o.30_4.dat"  --file_dump="o.30_4.lammpstrj"  --file_xyz="o.30_4.xyz"  --rc="1.5" --drc=0.1 --bond=1 --angle=1 --dihed=1 > o.30_4.log
