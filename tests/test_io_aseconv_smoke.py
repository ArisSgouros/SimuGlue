from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np

import pytest
from ase.io import read


# ---------- helpers ----------

def run_aseconv(tmp_path: Path, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    """
    Run `sgl io aseconv` inside tmp_path.
    If stdin is provided, it's passed to the process.
    """
    cmd = ["sgl", "io", "aseconv", *args]
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        input=stdin,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


# ---------- tests ----------

def test_extxyz_roundtrip(tmp_path: Path):
    """extxyz -> extxyz should be structurally identical."""
    in_xyz = tmp_path / "in.xyz"
    out_xyz = tmp_path / "out.xyz"

    in_xyz.write_text(
        "2\n"
        'Lattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3\n'
        "H 0.0 0.0 0.0\n"
        "He 1.0 1.0 1.0\n",
        encoding="utf-8",
    )

    run_aseconv(tmp_path, ["-i", str(in_xyz), "-o", str(out_xyz)])

    a_in = read(in_xyz)
    a_out = read(out_xyz)

    assert len(a_in) == len(a_out)
    assert a_in.get_chemical_symbols() == a_out.get_chemical_symbols()
    assert a_in.get_volume() == pytest.approx(a_out.get_volume())
    assert a_in.get_positions() == pytest.approx(a_out.get_positions())


def test_extxyz_to_lammps_data_and_back(tmp_path: Path):
    """extxyz -> lammps-data -> extxyz: preserve natoms + symbols (via Masses+specorder)."""
    in_xyz = tmp_path / "in.xyz"
    data_file = tmp_path / "out.data"
    back_xyz = tmp_path / "back.xyz"

    in_xyz.write_text(
        "4\n"
        'Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3\n'
        "Bi 0 0 0\n"
        "Bi 1 0 0\n"
        "Se 0 1 0\n"
        "Se 0 0 1\n",
        encoding="utf-8",
    )

    # extxyz -> lammps-data
    run_aseconv(tmp_path, [
        "-i", str(in_xyz),
        "--iformat", "extxyz",
        "-o", str(data_file),
        "--oformat", "lammps-data",
        "--lammps-style", "atomic",
        "--overwrite",
    ])

    assert data_file.exists()

    # lammps-data -> extxyz
    run_aseconv(tmp_path, [
        "-i", str(data_file),
        "--iformat", "lammps-data",
        "--lammps-style", "atomic",
        "-o", str(back_xyz),
        "--oformat", "extxyz",
        "--overwrite",
    ])

    a_in = read(in_xyz)
    a_back = read(back_xyz)

    assert len(a_in) == len(a_back)
    # same multiset of symbols (order may differ depending on writer)
    assert sorted(a_in.get_chemical_symbols()) == sorted(a_back.get_chemical_symbols())


def test_multiframe_extxyz_traj_roundtrip(tmp_path: Path):
    """Multi-frame extxyz -> traj -> extxyz: preserve natoms and frame count."""
    in_xyz = tmp_path / "multi.xyz"
    traj_file = tmp_path / "out.traj"
    back_xyz = tmp_path / "back.xyz"

    # 2 frames, 2 atoms each
    in_xyz.write_text(
        "2\n"
        'Lattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3\n'
        "H 0 0 0\n"
        "He 1 1 1\n"
        "2\n"
        'Lattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3\n'
        "H 0.1 0 0\n"
        "He 1.1 1 1\n",
        encoding="utf-8",
    )

    # extxyz -> traj
    run_aseconv(tmp_path, [
        "-i", str(in_xyz),
        "--iformat", "extxyz",
        "-o", str(traj_file),
        "--oformat", "traj",
        "--overwrite",
    ])
    assert traj_file.exists()

    # traj -> extxyz
    run_aseconv(tmp_path, [
        "-i", str(traj_file),
        "--iformat", "traj",
        "-o", str(back_xyz),
        "--oformat", "extxyz",
        "--overwrite",
    ])

    frames_in = list(read(in_xyz, index=":"))
    frames_back = list(read(back_xyz, index=":"))

    assert len(frames_in) == len(frames_back)
    assert all(len(a) == len(b) for a, b in zip(frames_in, frames_back))


def test_lammps_dump_to_extxyz(tmp_path: Path):
    """lammps-dump-text -> extxyz: basic sanity on atom count and coords."""
    dump = tmp_path / "in.dump"
    out_xyz = tmp_path / "out.xyz"

    dump.write_text(
        "ITEM: TIMESTEP\n"
        "0\n"
        "ITEM: NUMBER OF ATOMS\n"
        "2\n"
        "ITEM: BOX BOUNDS pp pp pp\n"
        "0.0 10.0\n"
        "0.0 10.0\n"
        "0.0 10.0\n"
        "ITEM: ATOMS id type x y z\n"
        "1 1 0.0 0.0 0.0\n"
        "2 1 1.0 0.0 0.0\n",
        encoding="utf-8",
    )

    run_aseconv(tmp_path, [
        "-i", str(dump),
        "--iformat", "lammps-dump-text",
        "-o", str(out_xyz),
        "--oformat", "extxyz",
    ])

    a = read(out_xyz, format="extxyz")
    assert len(a) == 2
    xs = a.get_positions()[:, 0]
    assert xs.min() == pytest.approx(0.0)
    assert xs.max() == pytest.approx(1.0)

def test_stdin_stdout_extxyz(tmp_path: Path):
    """extxyz -> extxyz via stdin/stdout: ensure streaming path works."""
    extxyz_text = (
        "2\n"
        'Lattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3\n'
        "H 0.0 0.0 0.0\n"
        "He 1.0 1.0 1.0\n"
    )

    # Run through aseconv with stdin/stdout
    result = run_aseconv(
        tmp_path,
        ["-i", "-", "--iformat", "extxyz", "-o", "-", "--oformat", "extxyz"],
        stdin=extxyz_text,
    )

    out_xyz = tmp_path / "pipe.xyz"
    out_xyz.write_text(result.stdout, encoding="utf-8")

    in_xyz = tmp_path / "in_from_text.xyz"
    in_xyz.write_text(extxyz_text, encoding="utf-8")

    a_in = read(in_xyz)
    a_out = read(out_xyz)

    assert len(a_in) == len(a_out)
    assert a_in.get_chemical_symbols() == a_out.get_chemical_symbols()

def test_espresso_in_to_extxyz(tmp_path: Path):
    """QE espresso-in -> extxyz: parse cell + positions + symbols."""
    qe_in = tmp_path / "qe.in"
    out_xyz = tmp_path / "out.xyz"

    qe_in.write_text(
        """ &control
    calculation = 'scf'
    restart_mode='from_scratch',
    prefix='myprefix',
    tstress = .true.
    tprnfor = .true.
    etot_conv_thr=0.1,
    forc_conv_thr=0.2
    pseudo_dir = '/psuedo_path/',
    outdir='/outdir_path/'
 /
 &system
    ibrav=  0, celldm(1) =3.77945225090701, nat=  3, ntyp= 3,
    ecutwfc = 100.0, ecutrho=200.0
 /
 &electrons
    diagonalization='david'
    electron_maxstep = 100
    mixing_mode = 'plain'
    mixing_beta = 0.7
    conv_thr = 1d-10
 /
ATOMIC_SPECIES
 Mo  1.0     Mo.UPF
 S   2.0     S.UPF
 Se  3.0     Se.UPF
CELL_PARAMETERS {alat}
0.50000000      0.00000000      0.00000000
0.00000000      0.50000000      0.00000000
0.00000000      0.00000000      5.00000000
ATOMIC_POSITIONS {angstrom}
Mo      0.110000000     0.120000000     0.130000000
S       0.210000000     0.220000000     0.230000000
Se      0.310000000     0.320000000     0.330000000
K_POINTS automatic
24 24 1 0 0 0
""",
        encoding="utf-8",
    )

    # espresso-in -> extxyz
    run_aseconv(tmp_path, [
        "-i", str(qe_in),
        "--iformat", "espresso-in",
        "-o", str(out_xyz),
        "--oformat", "extxyz",
        "--overwrite",
    ])

    a = read(out_xyz, format="extxyz")

    # 1) Correct number of atoms
    assert len(a) == 3

    # 2) Correct species order
    assert a.get_chemical_symbols() == ["Mo", "S", "Se"]

    # 3) Positions match ATOMIC_POSITIONS
    pos = a.get_positions()
    assert pos.shape == (3, 3)
    assert pos[0, 0] == pytest.approx(0.110000000)
    assert pos[0, 1] == pytest.approx(0.120000000)
    assert pos[0, 2] == pytest.approx(0.130000000)
    assert pos[1, 0] == pytest.approx(0.210000000)
    assert pos[1, 1] == pytest.approx(0.220000000)
    assert pos[1, 2] == pytest.approx(0.230000000)
    assert pos[2, 0] == pytest.approx(0.310000000)
    assert pos[2, 1] == pytest.approx(0.320000000)
    assert pos[2, 2] == pytest.approx(0.330000000)

    # 4) Cell from celldm(1) + CELL_PARAMETERS {alat}:
    # celldm(1) in Bohr; alat ≈ 2.0 Å; so diag ~ (1, 1, 10) Å
    cell = a.get_cell()
    assert cell[0, 0] == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert cell[1, 1] == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert cell[2, 2] == pytest.approx(10.0, rel=1e-6, abs=1e-6)


def test_lammps_triclinic_data_to_extxyz(tmp_path: Path):
    """lammps-data (triclinic) -> extxyz: verify tilted cell is preserved."""
    data_file = tmp_path / "triclinic.data"
    out_xyz = tmp_path / "triclinic.xyz"

    data_file.write_text(
        """# Triclinic test box
4 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 11.0 ylo yhi
0.0 12.0 zlo zhi
2.0 3.0 4.0 xy xz yz

Masses

1 1.0

Atoms # atomic

1 1 0.0 0.0 0.0
2 1 1.0 0.0 0.0
3 1 0.0 1.0 0.0
4 1 0.0 0.0 1.0
""",
        encoding="utf-8",
    )

    run_aseconv(
        tmp_path,
        [
            "-i",
            str(data_file),
            "--iformat",
            "lammps-data",
            "--lammps-style",
            "atomic",
            "-o",
            str(out_xyz),
            "--oformat",
            "extxyz",
            "--overwrite",
        ],
    )

    a = read(out_xyz, format="extxyz")

    # 4 atoms: make sure it actually read them
    assert len(a) == 4

    # Expected triclinic cell:
    # a = (10, 0, 0)
    # b = ( 2,10, 0)
    # c = ( 3, 4,10)
    cell = a.get_cell().array

    expected = np.array([
        [10.0, 0.0, 0.0],
        [2.0, 11.0, 0.0],
        [3.0, 4.0, 12.0],
    ])

    assert cell == pytest.approx(expected, rel=1e-12, abs=1e-12)

def test_lammps_data_full_to_extxyz(tmp_path: Path):
    """lammps-data (Atoms # full) -> extxyz: verify we parse style=full correctly."""
    data_file = tmp_path / "full.data"
    out_xyz = tmp_path / "full.xyz"

    # LAMMPS data in 'full' atom_style:
    # Atoms # full: id mol type q x y z
    data_file.write_text(
        """# Full style LAMMPS data
3 atoms
3 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 95.95  # Mo
2 32.06  # S
3 78.96  # Se

Atoms # full

1 1 1  0.10  0.0 0.0 0.0
2 1 2 -0.20  1.0 0.0 0.0
3 2 3  0.30  0.0 1.0 0.0
""",
        encoding="utf-8",
    )

    run_aseconv(
        tmp_path,
        [
            "-i",
            str(data_file),
            "--iformat",
            "lammps-data",
            "--lammps-style",
            "full",
            "-o",
            str(out_xyz),
            "--oformat",
            "extxyz",
            "--overwrite",
        ],
    )

    a = read(out_xyz, format="extxyz")

    # 1) Correct number of atoms
    assert len(a) == 3

    # 2) Symbols inferred from Masses comments via ASE
    assert sorted(a.get_chemical_symbols()) == ["Mo", "S", "Se"]

    # 3) Positions preserved
    pos = a.get_positions()
    assert pos[0] == pytest.approx([0.0, 0.0, 0.0])
    assert pos[1] == pytest.approx([1.0, 0.0, 0.0])
    assert pos[2] == pytest.approx([0.0, 1.0, 0.0])

    # 4) Cell preserved
    cell = a.get_cell().array
    expected = np.array([
        [10.0, 0.0, 0.0],
        [0.0,10.0, 0.0],
        [0.0, 0.0,10.0],
    ])
    assert cell == pytest.approx(expected, rel=1e-12, abs=1e-12)

def test_lammps_dump_select_first_two_frames(tmp_path: Path):
    """lammps-dump-text with 3 frames -> select first 2 via --frames and check."""

    dump = tmp_path / "three_frames.dump"
    out_xyz = tmp_path / "two_frames.xyz"

    # 3 timesteps, 2 atoms, simple positions to distinguish frames
    dump.write_text(
        # frame 0
        "ITEM: TIMESTEP\n"
        "0\n"
        "ITEM: NUMBER OF ATOMS\n"
        "2\n"
        "ITEM: BOX BOUNDS pp pp pp\n"
        "0.0 10.0\n"
        "0.0 10.0\n"
        "0.0 10.0\n"
        "ITEM: ATOMS id type x y z\n"
        "1 1 0.0 0.0 0.0\n"
        "2 1 1.0 0.0 0.0\n"
        # frame 1
        "ITEM: TIMESTEP\n"
        "1\n"
        "ITEM: NUMBER OF ATOMS\n"
        "2\n"
        "ITEM: BOX BOUNDS pp pp pp\n"
        "0.0 10.0\n"
        "0.0 10.0\n"
        "0.0 10.0\n"
        "ITEM: ATOMS id type x y z\n"
        "1 1 0.0 0.0 0.0\n"
        "2 1 2.0 0.0 0.0\n"
        # frame 2
        "ITEM: TIMESTEP\n"
        "2\n"
        "ITEM: NUMBER OF ATOMS\n"
        "2\n"
        "ITEM: BOX BOUNDS pp pp pp\n"
        "0.0 10.0\n"
        "0.0 10.0\n"
        "0.0 10.0\n"
        "ITEM: ATOMS id type x y z\n"
        "1 1 0.0 0.0 0.0\n"
        "2 1 3.0 0.0 0.0\n",
        encoding="utf-8",
    )

    # Select only the first two frames: indices 0 and 1.
    # Using slice syntax "0:2" works with our _parse_index.
    run_aseconv(
        tmp_path,
        [
            "-i",
            str(dump),
            "--iformat",
            "lammps-dump-text",
            "--frames",
            "0:2",
            "-o",
            str(out_xyz),
            "--oformat",
            "extxyz",
            "--overwrite",
        ],
    )

    # Read all frames from the output
    frames = list(read(out_xyz, format="extxyz", index=":"))

    # We expect exactly 2 frames
    assert len(frames) == 2

    # Frame 0: second atom at x = 1.0
    assert frames[0].get_positions()[1, 0] == pytest.approx(1.0)

    # Frame 1: second atom at x = 2.0
    assert frames[1].get_positions()[1, 0] == pytest.approx(2.0)

def test_lammps_data_small_skew_without_force_skew(tmp_path: Path):
    """
    extxyz with a very small skew (1e-8) -> lammps-data without --lammps-force-skew.

    Expectation:
      The box is treated as orthorhombic (no triclinic tilt line "xy xz yz"),
      i.e. tiny skew is ignored unless force_skew is enabled.
    """
    in_xyz = tmp_path / "small_skew.xyz"
    out_data = tmp_path / "small_skew.data"

    skew = 1e-8  # tiny tilt in the b_x component

    # Lattice:
    #   a = (10,      0,  0)
    #   b = (1e-8,  10,  0)
    #   c = ( 0,      0, 10)
    in_xyz.write_text(
        "2\n"
        f'Lattice="10 0 0 {skew} 10 0 0 0 10" Properties=species:S:1:pos:R:3\n'
        "H  0.0 0.0 0.0\n"
        "He 1.0 0.0 0.0\n",
        encoding="utf-8",
    )

    # extxyz -> lammps-data WITHOUT --lammps-force-skew
    run_aseconv(
        tmp_path,
        [
            "-i",
            str(in_xyz),
            "--iformat",
            "extxyz",
            "-o",
            str(out_data),
            "--oformat",
            "lammps-data",
            "--lammps-style",
            "atomic",
            "--overwrite",
        ],
    )

    text = out_data.read_text(encoding="utf-8")

    # For an orthorhombic box, ASE's lammps-data writer should emit 3 box lines:
    #   xlo xhi
    #   ylo yhi
    #   zlo zhi
    # and NO extra "xy xz yz" tilt line.
    assert "xy xz yz" not in text

def test_lammps_data_small_skew_with_force_skew(tmp_path: Path):
    """
    extxyz with a very small skew (1e-8) -> lammps-data WITH --lammps-force-skew.

    Expectation:
      The box is written as triclinic (tilt line 'xy xz yz' present),
      and the xy tilt is ~1e-8 (i.e. not dropped).
    """
    in_xyz = tmp_path / "small_skew_force.xyz"
    out_data = tmp_path / "small_skew_force.data"

    skew = 1e-8  # tiny tilt in b_x

    # Same lattice as previous test:
    #   a = (10,      0,  0)
    #   b = (1e-8,  10,  0)
    #   c = ( 0,      0, 10)
    in_xyz.write_text(
        "2\n"
        f'Lattice="10 0 0 {skew} 10 0 0 0 10" Properties=species:S:1:pos:R:3\n'
        "H  0.0 0.0 0.0\n"
        "He 1.0 0.0 0.0\n",
        encoding="utf-8",
    )

    # extxyz -> lammps-data WITH --lammps-force-skew
    run_aseconv(
        tmp_path,
        [
            "-i",
            str(in_xyz),
            "--iformat",
            "extxyz",
            "-o",
            str(out_data),
            "--oformat",
            "lammps-data",
            "--lammps-style",
            "atomic",
            "--lammps-force-skew",
            "--overwrite",
        ],
    )

    text = out_data.read_text(encoding="utf-8")

    # We expect a triclinic tilt line with 'xy xz yz'
    lines = [ln.strip() for ln in text.splitlines()]
    tilt_lines = [ln for ln in lines if ln.endswith("xy xz yz")]
    assert tilt_lines, "Expected triclinic tilt line 'xy xz yz' not found."

    # Optionally: check that xy is ~1e-8 and xz,yz ~ 0
    parts = tilt_lines[0].split()
    # format: xy xz yz xy-label xz-label yz-label
    xy, xz, yz = map(float, parts[0:3])
    assert xy == pytest.approx(skew, rel=0, abs=1e-12)
    assert xz == pytest.approx(0.0, abs=1e-12)
    assert yz == pytest.approx(0.0, abs=1e-12)

