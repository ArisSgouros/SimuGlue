from __future__ import annotations

import re

from simuglue.topology.topo import Topo
from simuglue.topology.export import ExportTypesTopo


def test_export_types_topo_writes_tables_and_masses(tmp_path):
    topo = Topo(
        bonds=[(0, 1)],
        angles=[(0, 1, 2)],
        dihedrals=[(0, 1, 2, 3)],
        bond_types=[1],
        angle_types=[1],
        dihedral_types=[2],
    )
    topo.meta["bond_type_table"] = {1: {"tag": "H O"}}
    topo.meta["angle_type_table"] = {1: {"tag": "H O H"}}
    topo.meta["dihedral_type_table"] = {2: {"tag": "C C C C trans"}}

    masses = {
        "n_types": 2,
        "mass": {1: 10.811, 2: 14.0067},
        "tag": {1: "B", 2: "N"},
    }

    out = tmp_path / "o.types"
    ExportTypesTopo(str(out), topo, masses)

    txt = out.read_text()

    # header counts
    assert re.search(r"^1 bond types$", txt, flags=re.M) is not None
    assert re.search(r"^1 angle types$", txt, flags=re.M) is not None
    assert re.search(r"^1 dihedral types$", txt, flags=re.M) is not None
    assert re.search(r"^2 atom types$", txt, flags=re.M) is not None

    # masses section
    assert "Masses" in txt
    assert re.search(r"^1\s+10\.811\s+#\s+B$", txt, flags=re.M)
    assert re.search(r"^2\s+14\.0067\s+#\s+N$", txt, flags=re.M)

    # type tables
    assert re.search(r"^Bond Types\s*$", txt, flags=re.M)
    assert re.search(r"^1\s+H O$", txt, flags=re.M)

    assert re.search(r"^Angle Types\s*$", txt, flags=re.M)
    assert re.search(r"^1\s+H O H$", txt, flags=re.M)

    assert re.search(r"^Dihedral Types\s*$", txt, flags=re.M)
    assert re.search(r"^2\s+C C C C trans$", txt, flags=re.M)
