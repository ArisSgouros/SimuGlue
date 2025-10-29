#!/usr/bin/env python3
from __future__ import annotations

import sys
import re
import json

kRyToEv = 13.6057039763
kRyToJoule = 2.1798741E-18
kBohrToAngstrom = 0.529177
kEvToJoule = 1.60218e-19
kPaToAtm = 1.0 / 101325
kAngstromToMeter = 1e-10
kRy_Bohr3ToPa = kRyToJoule / (kBohrToAngstrom * kAngstromToMeter) ** 3
kRy_Bohr3ToAtm = kRy_Bohr3ToPa * kPaToAtm

def pwo2json(file_path: str | Path) -> dict | None:
    """Parse a Quantum ESPRESSO *.out file into a Python dict."""
    line_stress = None
    line_forces = None
    line_energy = None
    line_atomic_positions = None
    line_cartesian_axes = None
    line_cell_parameters = None
    line_crystal_axes = None
    line_unit_cell_volume = None
    line_alat = None
    is_job_done = False

    lines: list[str] = []
    file_path = str(file_path)

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.rstrip("\n") for line in f]

        for ii, line in enumerate(lines):
            if "CELL_PARAMETERS" in line:
                line_cell_parameters = ii
            if "crystal axes:" in line:
                line_crystal_axes = ii + 1
            if "ATOMIC_POSITIONS" in line:
                line_atomic_positions = ii + 1
            if "Cartesian axes" in line:
                line_cartesian_axes = ii + 3
            if "total" in line and "stress" in line and "Ry/bohr**3" in line:
                line_stress = ii + 1
            if "unit-cell volume" in line:
                line_unit_cell_volume = ii
            if "!    total energy" in line:
                line_energy = ii
            if "Forces acting on atoms" in line:
                line_forces = ii + 2
            if "lattice parameter (alat)" in line:
                line_alat = ii
            if "JOB DONE" in line:
                is_job_done = True

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An error occurred while parsing the file '{file_path}': {e}", file=sys.stderr)
        return None

    data = {"num_atoms": None, "cell_data": {}, "info": {}, "atoms_data": []}

    # Parse alat
    alat_bohr = None
    if line_alat is not None:
        try:
            lsplit = lines[line_alat].split()
            alat_bohr = float(lsplit[4])
            # print('alat (bohr): ', alat_bohr)
        except Exception:
            print("Warning: error parsing alat", file=sys.stderr)

    # Parse energy
    if line_energy is not None:
        try:
            lsplit = lines[line_energy].split()
            energy_ry = float(lsplit[4])
            data["info"]["energy"] = energy_ry * kRyToEv
        except Exception:
            print("Warning: error parsing total energy", file=sys.stderr)

    # Parse cell_volume
    unit_cell_volume_bohr3 = None
    if line_unit_cell_volume is not None:
        try:
            lsplit = lines[line_unit_cell_volume].split()
            offset = 4 if "new" in lsplit else 3
            unit_cell_volume_bohr3 = float(lsplit[offset])
            # print('unit_cell_volume (bohr^3): ', unit_cell_volume_bohr3)
        except Exception:
            print("Warning: error parsing unit cell volume", file=sys.stderr)

    # Parse CELL_PARAMETERS or crystal axes
    lattice_vectors = None
    if line_cell_parameters is not None:
        lsplit = lines[line_cell_parameters + 1].split()
        ax, ay, az = float(lsplit[0]), float(lsplit[1]), float(lsplit[2])
        lsplit = lines[line_cell_parameters + 2].split()
        bx, by, bz = float(lsplit[0]), float(lsplit[1]), float(lsplit[2])
        lsplit = lines[line_cell_parameters + 3].split()
        cx, cy, cz = float(lsplit[0]), float(lsplit[1]), float(lsplit[2])
        lattice_vectors = [ax, ay, az, bx, by, bz, cx, cy, cz]
        if "alat" in lines[line_cell_parameters]:
            lattice_vectors = [val * alat_bohr * kBohrToAngstrom for val in lattice_vectors]
    elif line_crystal_axes is not None and alat_bohr is not None:
        # Fallback: initial crystal axes, convert from alat to Å
        offset = 3
        lsplit = lines[line_crystal_axes + 0].split()
        ax, ay, az = [float(lsplit[offset + i]) for i in range(3)]
        lsplit = lines[line_crystal_axes + 1].split()
        bx, by, bz = [float(lsplit[offset + i]) for i in range(3)]
        lsplit = lines[line_crystal_axes + 2].split()
        cx, cy, cz = [float(lsplit[offset + i]) for i in range(3)]
        lattice_vectors_bohr = [ax, ay, az, bx, by, bz, cx, cy, cz]
        lattice_vectors = [val * alat_bohr * kBohrToAngstrom for val in lattice_vectors_bohr]
    else:
        print("Warning: could not determine lattice vectors", file=sys.stderr)

    if lattice_vectors is not None:
        data["cell_data"] = lattice_vectors

    # Parse stress
    if (line_stress is not None) and (unit_cell_volume_bohr3 is not None):
        lsplit = lines[line_stress + 0].split()
        sxx, sxy, sxz = float(lsplit[0]), float(lsplit[1]), float(lsplit[2])
        lsplit = lines[line_stress + 1].split()
        syx, syy, syz = float(lsplit[0]), float(lsplit[1]), float(lsplit[2])
        lsplit = lines[line_stress + 2].split()
        szx, szy, szz = float(lsplit[0]), float(lsplit[1]), float(lsplit[2])

        stress_tensor_Ry_bohr3 = [sxx, sxy, sxz, syx, syy, syz, szx, szy, szz]
        virial_tensor_ev = [sij * unit_cell_volume_bohr3 * kRyToEv for sij in stress_tensor_Ry_bohr3]
        stress_tensor_atm = [sij * kRy_Bohr3ToAtm for sij in stress_tensor_Ry_bohr3]

        data["info"]["virial"] = virial_tensor_ev
        data["info"]["stress"] = stress_tensor_atm

    # Parse ATOMIC_POSITIONS
    if line_atomic_positions is not None:
        for ii in range(100000):
            line = lines[line_atomic_positions + ii]
            if not line.strip():
                # Not final coordinates (likely input block)
                # print("Warning: these are not final coordinates. Ignore if this is input file", file=sys.stderr)
                break
            if "End final coordinates" in line:
                break
            lsplit = line.split()
            symbols = lsplit[0]
            x, y, z = float(lsplit[1]), float(lsplit[2]), float(lsplit[3])
            data["atoms_data"].append({"symbols": symbols, "positions": [x, y, z]})
    elif line_cartesian_axes is not None:
        for ii in range(100000):
            lsplit = lines[line_cartesian_axes + ii].split()
            if not lsplit:
                break
            symbols = lsplit[1]
            offset = 6
            x, y, z = float(lsplit[offset + 0]), float(lsplit[offset + 1]), float(lsplit[offset + 2])
            if alat_bohr is None:
                print("Warning: missing alat for Cartesian axes conversion", file=sys.stderr)
                break
            pos_angstrom = [coord * alat_bohr * kBohrToAngstrom for coord in (x, y, z)]
            data["atoms_data"].append({"symbols": symbols, "positions": pos_angstrom})

    # Number of atoms
    data["num_atoms"] = len(data["atoms_data"])

    # Parse forces
    if (line_forces is not None) and (line_atomic_positions is not None or line_cartesian_axes is not None):
        for ii in range(data["num_atoms"]):
            lsplit = lines[line_forces + ii].split()
            fx, fy, fz = float(lsplit[6]), float(lsplit[7]), float(lsplit[8])
            ff_eV_angstrom = [fi * kRyToEv / kBohrToAngstrom for fi in (fx, fy, fz)]
            data["atoms_data"][ii]["forces"] = ff_eV_angstrom

    data["info"]["job_done"] = is_job_done
    return data
