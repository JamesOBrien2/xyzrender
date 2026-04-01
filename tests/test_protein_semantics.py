"""Tests for graph-first protein semantics extraction and fallback gating."""

from __future__ import annotations

from textwrap import dedent

import numpy as np
import pytest

from xyzrender import load, render
from xyzrender.types import ProteinConfidence


def test_extxyz_canonical_semantics(tmp_path):
    p = tmp_path / "prot.xyz"
    p.write_text(
        dedent(
            """\
            8
            Properties=species:S:1:pos:R:3:atom_name:S:1:res_name:S:1:res_seq:I:1:chain_id:S:1:ss_type:S:1
            N 0.000 0.000 0.000 N ALA 1 A H
            C 1.456 0.000 0.000 CA ALA 1 A H
            C 1.930 0.000 1.463 C ALA 1 A H
            O 1.160 0.000 2.421 O ALA 1 A H
            N 3.241 0.000 1.742 N GLY 2 A H
            C 3.690 0.000 3.127 CA GLY 2 A H
            C 5.228 0.000 3.127 C GLY 2 A H
            O 5.902 0.000 2.098 O GLY 2 A H
            """
        )
    )
    mol = load(p)
    assert mol.protein_semantics is not None
    assert mol.protein_semantics.confidence_tier == ProteinConfidence.FULL_RIBBON
    assert "A" in mol.protein_semantics.chains


def test_mol2_semantics_extraction(tmp_path):
    p = tmp_path / "prot.mol2"
    p.write_text(
        dedent(
            """\
            @<TRIPOS>MOLECULE
            prot
             8 7 0 0 0
            SMALL
            NO_CHARGES
            @<TRIPOS>ATOM
              1 N   0.000 0.000 0.000 N.3  1 ALA1_A  0.0
              2 CA  1.456 0.000 0.000 C.3  1 ALA1_A  0.0
              3 C   1.930 0.000 1.463 C.2  1 ALA1_A  0.0
              4 O   1.160 0.000 2.421 O.2  1 ALA1_A  0.0
              5 N   3.241 0.000 1.742 N.3  2 GLY2_A  0.0
              6 CA  3.690 0.000 3.127 C.3  2 GLY2_A  0.0
              7 C   5.228 0.000 3.127 C.2  2 GLY2_A  0.0
              8 O   5.902 0.000 2.098 O.2  2 GLY2_A  0.0
            @<TRIPOS>BOND
             1 1 2 1
             2 2 3 1
             3 3 4 1
             4 3 5 1
             5 5 6 1
             6 6 7 1
             7 7 8 1
            """
        )
    )
    mol = load(p)
    assert mol.protein_semantics is not None
    assert mol.protein_semantics.confidence_tier == ProteinConfidence.FULL_RIBBON
    assert len(mol.protein_semantics.chains["A"].residues) == 2


def test_mol2_ligand_not_promoted_to_ribbon(tmp_path):
    p = tmp_path / "prot_lig.mol2"
    p.write_text(
        dedent(
            """\
            @<TRIPOS>MOLECULE
            prot_lig
             10 8 0 0 0
            SMALL
            NO_CHARGES
            @<TRIPOS>ATOM
              1 N   0.000 0.000 0.000 N.3  1 ALA1_A  0.0
              2 CA  1.456 0.000 0.000 C.3  1 ALA1_A  0.0
              3 C   1.930 0.000 1.463 C.2  1 ALA1_A  0.0
              4 O   1.160 0.000 2.421 O.2  1 ALA1_A  0.0
              5 N   3.241 0.000 1.742 N.3  2 GLY2_A  0.0
              6 CA  3.690 0.000 3.127 C.3  2 GLY2_A  0.0
              7 C   5.228 0.000 3.127 C.2  2 GLY2_A  0.0
              8 O   5.902 0.000 2.098 O.2  2 GLY2_A  0.0
              9 C1  8.000 0.000 0.000 C.3  3 LIG3_A  0.0
             10 O1  9.200 0.000 0.000 O.2  3 LIG3_A  0.0
            @<TRIPOS>BOND
             1 1 2 1
             2 2 3 1
             3 3 4 1
             4 3 5 1
             5 5 6 1
             6 6 7 1
             7 7 8 1
             8 9 10 1
            """
        )
    )
    mol = load(p)
    sem = mol.protein_semantics
    assert sem is not None
    protein_atoms = {i for ch in sem.chains.values() for r in ch.residues for i in r.atom_indices}
    assert sem.ligand_indices == {8, 9}
    assert 8 not in protein_atoms
    assert 9 not in protein_atoms

    # In protein mode, backbone protein atoms are hidden but ligand atoms should stay visible.
    svg = str(render(mol, protein=True, orient=False, gradient=False, config="flat"))
    assert svg.count("<circle") >= 2


def test_graph_only_trace_fallback(tmp_path):
    p = tmp_path / "trace.xyz"
    p.write_text(
        dedent(
            """\
            16
            trace
            N 0.000 0.000 0.000
            C 1.456 0.000 0.000
            C 1.930 0.000 1.463
            O 1.160 0.000 2.421
            N 3.241 0.000 1.742
            C 3.690 0.000 3.127
            C 5.228 0.000 3.127
            O 5.902 0.000 2.098
            N 5.898 0.000 4.287
            C 7.353 0.000 4.287
            C 7.828 0.000 5.750
            O 7.058 0.001 6.709
            N 9.148 0.000 6.029
            C 9.602 0.000 7.414
            C 11.140 0.000 7.414
            O 11.814 0.000 6.384
            """
        )
    )
    mol = load(p)
    assert mol.protein_semantics is None  # no metadata extracted at load-time
    svg = str(render(mol, protein=True, orient=False, gradient=False, config="flat"))
    # Trace fallback uses path strokes (not helix/sheet polygons).
    assert "<path" in svg


def test_cif_annotations_prevent_graph_only_fallback(tmp_path, monkeypatch):
    pytest.importorskip("xyzgraph.protein", reason="xyzgraph protein semantics module required")
    ase = pytest.importorskip("ase", reason="ase required")
    from ase import Atoms

    p = tmp_path / "prot.cif"
    p.write_text("data_fake\n")

    structure = Atoms(
        symbols=["N", "C", "C", "O", "N", "C", "C", "O"],
        positions=np.zeros((8, 3), dtype=float),
        cell=np.eye(3),
        pbc=True,
    )
    structure.set_array("atomtypes", np.array(["N", "CA", "C", "O", "N", "CA", "C", "O"], dtype=object))
    structure.set_array(
        "residuenames",
        np.array(["ALA", "ALA", "ALA", "ALA", "GLY", "GLY", "GLY", "GLY"], dtype=object),
    )
    structure.set_array("residuenumbers", np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=int))
    structure.set_array("chainids", np.array(["A", "A", "A", "A", "A", "A", "A", "A"], dtype=object))

    def fake_read(path, format=None, **kwargs):
        return structure

    monkeypatch.setattr(ase.io, "read", fake_read)

    mol = load(p)
    sem = mol.protein_semantics
    assert sem is not None
    assert sem.confidence_tier == ProteinConfidence.FULL_RIBBON
    assert any("annotations:.cif annotations parsed" in reason for reason in sem.confidence_reasons)
    assert not any("graph-only heuristic" in reason for reason in sem.confidence_reasons)


def test_cif_struct_conf_and_sheet_range_labels(tmp_path, monkeypatch):
    pytest.importorskip("xyzgraph.protein", reason="xyzgraph protein semantics module required")
    ase = pytest.importorskip("ase", reason="ase required")
    from ase import Atoms

    p = tmp_path / "prot_ss.cif"
    p.write_text("data_fake\n")

    n_res = 6
    symbols: list[str] = []
    positions: list[tuple[float, float, float]] = []
    atomtypes: list[str] = []
    residuenames: list[str] = []
    residuenumbers: list[int] = []
    chainids: list[str] = []
    groups: list[str] = []
    serial = 0
    for r in range(1, n_res + 1):
        for atom_name, sym, dx in (("N", "N", -1.1), ("CA", "C", 0.0), ("C", "C", 1.1), ("O", "O", 1.8)):
            symbols.append(sym)
            positions.append((float(r * 1.5 + dx), 0.0, 0.0))
            atomtypes.append(atom_name)
            residuenames.append("ALA")
            residuenumbers.append(r)
            chainids.append("A")
            groups.append("ATOM")
            serial += 1

    structure = Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float), cell=np.eye(3), pbc=True)
    structure.set_array("atomtypes", np.asarray(atomtypes, dtype=object))
    structure.set_array("residuenames", np.asarray(residuenames, dtype=object))
    structure.set_array("residuenumbers", np.asarray(residuenumbers, dtype=int))
    structure.set_array("chainids", np.asarray(chainids, dtype=object))
    structure.set_array("tags", np.arange(serial, dtype=int))
    structure.info["_atom_site.group_pdb"] = groups
    structure.info["_struct_conf.conf_type_id"] = ["HELX_P"]
    structure.info["_struct_conf.beg_auth_asym_id"] = ["A"]
    structure.info["_struct_conf.end_auth_asym_id"] = ["A"]
    structure.info["_struct_conf.beg_auth_seq_id"] = [1]
    structure.info["_struct_conf.end_auth_seq_id"] = [3]
    structure.info["_struct_sheet_range.beg_auth_asym_id"] = ["A"]
    structure.info["_struct_sheet_range.end_auth_asym_id"] = ["A"]
    structure.info["_struct_sheet_range.beg_auth_seq_id"] = [4]
    structure.info["_struct_sheet_range.end_auth_seq_id"] = [6]

    def fake_read(path, format=None, **kwargs):
        assert kwargs.get("store_tags") is True
        return structure

    monkeypatch.setattr(ase.io, "read", fake_read)

    mol = load(p)
    sem = mol.protein_semantics
    assert sem is not None
    assert sem.confidence_tier == ProteinConfidence.FULL_RIBBON
    chain = sem.chains["A"]
    labels = {res.res_seq: res.ss_type for res in chain.residues}
    assert labels[1] == "H"
    assert labels[2] == "H"
    assert labels[3] == "H"
    assert labels[4] == "E"
    assert labels[5] == "E"
    assert labels[6] == "E"
