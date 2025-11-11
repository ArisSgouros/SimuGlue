# SIMUGLUE
**SIMUlation GLUE — General Lab Utility Engine**

SIMUGLUE provides a unified set of lightweight tools and scripts
for preparing, running, and analyzing materials simulations.

It focuses on the practical links between major codes (Quantum ESPRESSO,
LAMMPS, ASE, NEP, etc.), offering a consistent way to convert data,
launch jobs, and extract physical quantities.

### Features
- File and structure converters
- Job setup and submission helpers
- Logfile and trajectory analyzers
- Reusable command-line utilities and Python modules
- Minimal dependencies, easy to extend

**SIMUGLUE:** the glue that keeps your simulation workflows together.

### Installation in conda environment
conda create -n simuglue python=3.11
conda activate simuglue

# install dev version
tools/dev-install.sh

# intall argcomplete
argcompletion-install.sh
