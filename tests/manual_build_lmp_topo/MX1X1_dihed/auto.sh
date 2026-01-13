#!/bin/bash
sgl build lmp-topo \
    -i i.supercell.dat \
    -o o.supercell_topo.dat \
    --types-out o.types \
    --type-delimiter '-' \
    --rc "0.763762616" \
    --drc 0.1 \
    --bonds --angles --dihedrals \
    --diff-bond-len \
    --angle-symmetry --diff-angle-theta --angle-theta-fmt %.4f \
    --diff-dihed-theta --dihed-theta-fmt %.5f \
