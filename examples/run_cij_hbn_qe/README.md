### Cij Workflow (Quantum ESPRESSO + SIMUGLUE)

1. **Fetch a relaxed quantum espresso input:**

2. **Verify relaxation mode:**
   ```fortran
   calculation = 'relax'
   ```

3. **Edit config file:**  
   Adjust `cfg.yaml` (Voigt components, strain amplitudes, backend, paths, etc.).

4. **Run workflow:**
   ```bash
   sgl wf cij init  -c cfg.yaml    # Initialize deformation cases
   sgl wf cij run   -c cfg.yaml    # Run QE calculations
   sgl wf cij parse -c cfg.yaml    # Parse QE outputs
   sgl wf cij post  -c cfg.yaml    # Post-process to get Cij
   ```

5. **Results:**  
   Final stiffness matrix is saved in `cij.json`; completed cases are marked with `.done`.
