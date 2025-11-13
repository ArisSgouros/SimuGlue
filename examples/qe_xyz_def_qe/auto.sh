## Example Script

#The whole pipeline can be run with a small helper script, for example:

#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 1: Strain (Voigt) → Deformation Gradient (F) ==="
echo "Input strain (engineering, Voigt notation):"
cat i.strain_voigt
echo

sgl mech defgrad \
    -i i.strain_voigt \
    --measure engineering \
    -o o.defgrad

echo "Computed deformation gradient F (saved to o.defgrad):"
cat o.defgrad
echo

echo "=== Step 2: QE input → extxyz structure ==="
sgl io aseconv \
    -i i.hbn.in \
    --oformat extxyz \
    -o o.hbn.xyz

echo "Wrote undeformed structure to o.hbn.xyz"
echo

echo "=== Step 3: Apply deformation gradient to structure ==="
sgl xform xyz \
    -i o.hbn.xyz \
    --F '2 0 0; 0 1 0; 0 0 1' \
    -o o.hbn_def.xyz

echo "Wrote deformed structure to o.hbn_def.xyz"
echo

echo "=== Step 4: Build QE input for deformed configuration ==="
sgl qe xyz2qe \
    --pwi i.hbn.in \
    --xyz o.hbn_def.xyz \
    --prefix NEW_PREF \
    --outdir NEW_OUTDIR \
    -o o.hbn_def.in

echo "Done. New QE input for the deformed structure: o.hbn_def.in"

