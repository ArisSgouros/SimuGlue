# SIMUGLUE Example: From Strain to Deformed QE Input

This example shows how to use **SIMUGLUE** to:

1. Convert an engineering strain (in Voigt notation) to a deformation gradient tensor.
2. Convert a Quantum ESPRESSO input file to an `extxyz` structure.
3. Apply a prescribed deformation gradient to the structure.
4. Regenerate a new Quantum ESPRESSO input file for the deformed configuration.

The workflow is driven by the following script:

```bash
#!/bin/bash
cat i.strain_voigt
sgl mech defgrad -i i.strain_voigt --measure engineering -o o.defgrad
cat o.defgrad

sgl io aseconv -i i.hbn.in --oformat extxyz -o o.hbn.xyz

sgl xform xyz -i o.hbn.xyz --F '2 0 0; 0 1 0; 0 0 1' -o o.hbn_def.xyz

sgl qe xyz2qe --pwi i.hbn.in --xyz o.hbn_def.xyz --prefix NEW_PREF --outdir NEW_OUTDIR -o o.hbn_def.in

