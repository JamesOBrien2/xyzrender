"""Tests for protein ribbon rendering."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import networkx as nx
import pytest

from xyzrender import load, render
from xyzrender.colors import resolve_color
from xyzrender.parsers import parse_pdb
from xyzrender.readers import filter_ligand_protein_nci
from xyzrender.types import ChainData, ProteinData, ResidueData

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


def _require_protein_data(data: ProteinData | None) -> ProteinData:
    assert data is not None
    return data


# ---------------------------------------------------------------------------
# Minimal synthetic PDB fixtures
# ---------------------------------------------------------------------------

# 4-residue helix (HELIX record + 4 ATOM records with CA/N/C/O)
_HELIX_PDB = dedent("""\
    HELIX    1   1 ALA A    1  ALA A    4  1                                   4
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1       1.456   0.000   0.000  1.00  0.00           C
    ATOM      3  C   ALA A   1       1.930   0.000   1.463  1.00  0.00           C
    ATOM      4  O   ALA A   1       1.160   0.000   2.421  1.00  0.00           O
    ATOM      5  N   ALA A   2       3.241   0.000   1.742  1.00  0.00           N
    ATOM      6  CA  ALA A   2       3.690   0.000   3.127  1.00  0.00           C
    ATOM      7  C   ALA A   2       5.228   0.000   3.127  1.00  0.00           C
    ATOM      8  O   ALA A   2       5.902   0.000   2.098  1.00  0.00           O
    ATOM      9  N   ALA A   3       5.898   0.000   4.287  1.00  0.00           N
    ATOM     10  CA  ALA A   3       7.353   0.000   4.287  1.00  0.00           C
    ATOM     11  C   ALA A   3       7.828   0.000   5.750  1.00  0.00           C
    ATOM     12  O   ALA A   3       7.058   0.001   6.709  1.00  0.00           O
    ATOM     13  N   ALA A   4       9.148   0.000   6.029  1.00  0.00           N
    ATOM     14  CA  ALA A   4       9.602   0.000   7.414  1.00  0.00           C
    ATOM     15  C   ALA A   4      11.140   0.000   7.414  1.00  0.00           C
    ATOM     16  O   ALA A   4      11.814   0.000   6.384  1.00  0.00           O
    END
""")

# 4-residue strand (SHEET record)
_SHEET_PDB = dedent("""\
    SHEET    1   A 1 ALA A   1  ALA A   4  0
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1       1.456   0.000   0.000  1.00  0.00           C
    ATOM      3  C   ALA A   1       1.930   0.000   1.463  1.00  0.00           C
    ATOM      4  O   ALA A   1       1.160   0.000   2.421  1.00  0.00           O
    ATOM      5  N   ALA A   2       3.241   0.000   1.742  1.00  0.00           N
    ATOM      6  CA  ALA A   2       3.690   0.000   3.127  1.00  0.00           C
    ATOM      7  C   ALA A   2       5.228   0.000   3.127  1.00  0.00           C
    ATOM      8  O   ALA A   2       5.902   0.000   2.098  1.00  0.00           O
    ATOM      9  N   ALA A   3       5.898   0.000   4.287  1.00  0.00           N
    ATOM     10  CA  ALA A   3       7.353   0.000   4.287  1.00  0.00           C
    ATOM     11  C   ALA A   3       7.828   0.000   5.750  1.00  0.00           C
    ATOM     12  O   ALA A   3       7.058   0.001   6.709  1.00  0.00           O
    ATOM     13  N   ALA A   4       9.148   0.000   6.029  1.00  0.00           N
    ATOM     14  CA  ALA A   4       9.602   0.000   7.414  1.00  0.00           C
    ATOM     15  C   ALA A   4      11.140   0.000   7.414  1.00  0.00           C
    ATOM     16  O   ALA A   4      11.814   0.000   6.384  1.00  0.00           O
    END
""")

# PDB with HETATM ligand + protein ATOM records
_HETATM_PDB = dedent("""\
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1       1.456   0.000   0.000  1.00  0.00           C
    ATOM      3  C   ALA A   1       1.930   0.000   1.463  1.00  0.00           C
    ATOM      4  O   ALA A   1       1.160   0.000   2.421  1.00  0.00           O
    HETATM    5  C1  LIG A 101       5.000   5.000   0.000  1.00  0.00           C
    HETATM    6  O1  LIG A 101       6.000   5.000   0.000  1.00  0.00           O
    END
""")

_HETATM_CLASS_PDB = dedent("""\
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1       1.456   0.000   0.000  1.00  0.00           C
    HETATM    3  O   HOH A 201       3.000   0.000   0.000  1.00  0.00           O
    HETATM    4 NA   NA  A 202       4.000   0.000   0.000  1.00  0.00          NA
    HETATM    5  C1  LIG A 203       5.000   0.000   0.000  1.00  0.00           C
    END
""")

# Two-chain PDB
_TWO_CHAIN_PDB = dedent("""\
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1       1.456   0.000   0.000  1.00  0.00           C
    ATOM      3  C   ALA A   1       1.930   0.000   1.463  1.00  0.00           C
    ATOM      4  O   ALA A   1       1.160   0.000   2.421  1.00  0.00           O
    ATOM      5  N   ALA A   2       3.241   0.000   1.742  1.00  0.00           N
    ATOM      6  CA  ALA A   2       3.690   0.000   3.127  1.00  0.00           C
    ATOM      7  C   ALA A   2       5.228   0.000   3.127  1.00  0.00           C
    ATOM      8  O   ALA A   2       5.902   0.000   2.098  1.00  0.00           O
    ATOM      9  N   GLY B   1      10.000  10.000   0.000  1.00  0.00           N
    ATOM     10  CA  GLY B   1      11.456  10.000   0.000  1.00  0.00           C
    ATOM     11  C   GLY B   1      11.930  10.000   1.463  1.00  0.00           C
    ATOM     12  O   GLY B   1      11.160  10.000   2.421  1.00  0.00           O
    ATOM     13  N   GLY B   2      13.241  10.000   1.742  1.00  0.00           N
    ATOM     14  CA  GLY B   2      13.690  10.000   3.127  1.00  0.00           C
    ATOM     15  C   GLY B   2      15.228  10.000   3.127  1.00  0.00           C
    ATOM     16  O   GLY B   2      15.902  10.000   2.098  1.00  0.00           O
    END
""")


@pytest.fixture
def helix_pdb(tmp_path):
    p = tmp_path / "helix.pdb"
    p.write_text(_HELIX_PDB)
    return p


@pytest.fixture
def sheet_pdb(tmp_path):
    p = tmp_path / "sheet.pdb"
    p.write_text(_SHEET_PDB)
    return p


@pytest.fixture
def hetatm_pdb(tmp_path):
    p = tmp_path / "hetatm.pdb"
    p.write_text(_HETATM_PDB)
    return p


@pytest.fixture
def two_chain_pdb(tmp_path):
    p = tmp_path / "two_chain.pdb"
    p.write_text(_TWO_CHAIN_PDB)
    return p


@pytest.fixture
def hetatm_class_pdb(tmp_path):
    p = tmp_path / "hetatm_class.pdb"
    p.write_text(_HETATM_CLASS_PDB)
    return p


# ---------------------------------------------------------------------------
# Phase 1: Parser tests
# ---------------------------------------------------------------------------


def test_parse_pdb_extracts_chain(tmp_path, helix_pdb):
    data = parse_pdb(helix_pdb)
    pd = _require_protein_data(data.protein_data)
    assert "A" in pd.chains


def test_parse_pdb_residue_count(helix_pdb):
    data = parse_pdb(helix_pdb)
    pd = _require_protein_data(data.protein_data)
    chain_a = pd.chains["A"]
    assert len(chain_a.residues) == 4


def test_parse_pdb_ca_index(helix_pdb):
    data = parse_pdb(helix_pdb)
    pd = _require_protein_data(data.protein_data)
    for res in pd.chains["A"].residues:
        assert res.ca_index is not None


def test_parse_pdb_ss_helix(helix_pdb):
    """HELIX record should set ss_type='H' on all residues."""
    data = parse_pdb(helix_pdb)
    pd = _require_protein_data(data.protein_data)
    for res in pd.chains["A"].residues:
        assert res.ss_type == "H", f"residue {res.res_seq} expected H got {res.ss_type}"


def test_parse_pdb_ss_sheet(sheet_pdb):
    """SHEET record should set ss_type='E' on all residues."""
    data = parse_pdb(sheet_pdb)
    pd = _require_protein_data(data.protein_data)
    for res in pd.chains["A"].residues:
        assert res.ss_type == "E", f"residue {res.res_seq} expected E got {res.ss_type}"


def test_parse_pdb_ss_default_loop():
    """Without HELIX/SHEET records, ss_type defaults to 'C'."""
    data = parse_pdb(STRUCTURES / "ala_phe_ala.pdb")
    pd = data.protein_data
    if pd is not None:
        for chain in pd.chains.values():
            for res in chain.residues:
                assert res.ss_type == "C"


def test_parse_pdb_hetatm_separation(hetatm_pdb):
    data = parse_pdb(hetatm_pdb)
    pd = _require_protein_data(data.protein_data)
    # Indices 4 and 5 are HETATM (0-indexed)
    assert 4 in pd.hetatm_indices
    assert 5 in pd.hetatm_indices
    # Backbone atoms 0-3 should not be HETATM
    assert 0 not in pd.hetatm_indices


def test_parse_pdb_backbone_indices(helix_pdb):
    data = parse_pdb(helix_pdb)
    pd = data.protein_data
    assert pd is not None
    # N, CA, C, O for 4 residues = 16 backbone atoms (all atoms in this PDB)
    assert len(pd.backbone_indices) == 16


def test_parse_pdb_hetatm_ligand_water_ion_indices(hetatm_class_pdb):
    data = parse_pdb(hetatm_class_pdb)
    pd = data.protein_data
    assert pd is not None
    # 0-indexed from records above: HOH=2, NA=3, LIG=4
    assert pd.water_indices == {2}
    assert pd.ion_indices == {3}
    assert pd.ligand_indices == {4}


def test_parse_pdb_two_chains(two_chain_pdb):
    data = parse_pdb(two_chain_pdb)
    pd = data.protein_data
    assert pd is not None
    assert set(pd.chains.keys()) == {"A", "B"}
    assert len(pd.chains["A"].residues) == 2
    assert len(pd.chains["B"].residues) == 2


def test_load_attaches_protein_data(helix_pdb):
    mol = load(helix_pdb)
    pd = _require_protein_data(mol.protein_data)
    assert "A" in pd.chains


def test_load_water_pdb_loads_without_error():
    """Simple water PDB loads without error (has chain A, so protein_data may be set)."""
    mol = load(STRUCTURES / "water.pdb")
    # water.pdb has chain 'A' so protein_data is populated, which is fine
    assert mol.graph.number_of_nodes() == 3


# ---------------------------------------------------------------------------
# Phase 2: Ribbon geometry tests
# ---------------------------------------------------------------------------


def test_ribbon_items_returned(helix_pdb):
    """ribbon_svg_items() returns non-empty list for a helix structure."""
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    mol = load(helix_pdb)
    pd = mol.protein_data
    assert pd is not None

    cfg = RenderConfig(protein=True)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])

    items = ribbon_svg_items(pd, cfg, pos, scale=50.0, cx=0.0, cy=0.0, canvas_w=800, canvas_h=800)
    assert len(items) > 0


def test_ribbon_helix_produces_polygons(helix_pdb):
    """Helix segments produce polygon SVG elements."""
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    mol = load(helix_pdb)
    cfg = RenderConfig(protein=True)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])
    pd = _require_protein_data(mol.protein_data)
    items = ribbon_svg_items(pd, cfg, pos, 50.0, 0.0, 0.0, 800, 800)
    all_svg = " ".join(line for _, lines in items for line in lines)
    assert "<polygon" in all_svg


def test_adaptive_spline_steps_is_deterministic():
    """Internal adaptive tessellation should choose stable step counts."""
    import numpy as np

    from xyzrender.ribbon import _adaptive_spline_steps

    small = np.array([[float(i), 0.0, 0.0] for i in range(100)], dtype=float)
    large = np.array([[float(i), 0.0, 0.0] for i in range(2500)], dtype=float)

    s1 = _adaptive_spline_steps(small, scale=80.0)
    s2 = _adaptive_spline_steps(small, scale=80.0)
    s3 = _adaptive_spline_steps(large, scale=80.0)

    assert s1 == s2
    assert s1 >= s3


def test_ribbon_sheet_produces_polygons(sheet_pdb):
    """Sheet segments produce polygon elements (including arrowhead)."""
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    mol = load(sheet_pdb)
    cfg = RenderConfig(protein=True)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])
    pd = _require_protein_data(mol.protein_data)
    items = ribbon_svg_items(pd, cfg, pos, 50.0, 0.0, 0.0, 800, 800)
    all_svg = " ".join(line for _, lines in items for line in lines)
    assert "<polygon" in all_svg


def test_ribbon_two_chains_distinct_colors(two_chain_pdb):
    """Two chains get distinct colours from the palette."""
    from xyzrender.ribbon import assign_chain_colors
    from xyzrender.types import RenderConfig

    mol = load(two_chain_pdb)
    cfg = RenderConfig(protein=True)
    pd = _require_protein_data(mol.protein_data)
    colors = assign_chain_colors(cfg, list(pd.chains.keys()))
    assert len(set(colors.values())) == 2, "Chains A and B should have different colours"


def test_ribbon_chain_color_override(two_chain_pdb):
    """Explicit chain_colors override palette."""
    from xyzrender.colors import resolve_color
    from xyzrender.ribbon import assign_chain_colors
    from xyzrender.types import RenderConfig

    mol = load(two_chain_pdb)
    cfg = RenderConfig(protein=True, chain_colors={"A": resolve_color("steelblue")})
    pd = _require_protein_data(mol.protein_data)
    colors = assign_chain_colors(cfg, list(pd.chains.keys()))
    assert colors["A"] == resolve_color("steelblue")


def test_ribbon_items_sorted_by_z(helix_pdb):
    """ribbon_svg_items() returns items sorted ascending by z_depth."""
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    mol = load(helix_pdb)
    cfg = RenderConfig(protein=True)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])

    pd = _require_protein_data(mol.protein_data)
    items = ribbon_svg_items(pd, cfg, pos, 50.0, 0.0, 0.0, 800, 800)
    depths = [z for z, _ in items]
    assert depths == sorted(depths)


# ---------------------------------------------------------------------------
# Phase 3: Renderer integration tests
# ---------------------------------------------------------------------------


def test_protein_flag_renders(helix_pdb):
    """render(mol, protein=True) produces an SVG without error."""
    mol = load(helix_pdb)
    svg = str(render(mol, protein=True, orient=False))
    assert "<svg" in svg


def test_backbone_atoms_hidden(helix_pdb):
    """In protein mode, backbone atom circles should not appear in SVG."""
    mol = load(helix_pdb)
    pd = mol.protein_data
    assert pd is not None

    svg_protein = str(render(mol, protein=True, orient=False, gradient=False))
    svg_normal = str(render(mol, orient=False, gradient=False))

    # Protein mode should produce fewer atom circles (backbone suppressed)
    assert svg_protein.count("<circle") < svg_normal.count("<circle")


def test_hetatm_atoms_visible_in_protein_mode(hetatm_pdb):
    """HETATM atoms remain visible as ball-and-stick in protein mode."""
    mol = load(hetatm_pdb)
    svg = str(render(mol, protein=True, orient=False, gradient=False))
    # There should be at least some circles (HETATM atoms)
    assert "<circle" in svg


def test_ribbon_polygons_in_svg(helix_pdb):
    """render with protein=True contains ribbon polygon elements."""
    mol = load(helix_pdb)
    svg = str(render(mol, protein=True, orient=False))
    assert "<polygon" in svg


@pytest.mark.parametrize("style", ["cartoon", "plastic", "matte", "pymol", "illustrative"])
def test_removed_protein_styles_raise(helix_pdb, style):
    """Removed legacy styles should fail fast with a validation error."""
    mol = load(helix_pdb)
    with pytest.raises(ValueError, match="unknown style"):
        render(mol, protein=style, orient=False)


def test_protein_gloss_respects_transparent_background(helix_pdb):
    """Gloss style must not force an opaque or black background."""
    mol = load(helix_pdb)
    svg = str(render(mol, protein="gloss", orient=False, transparent=True))
    assert 'style="background:transparent"' in svg
    assert 'fill="#000000"' not in svg


def test_protein_invalid_style_raises(helix_pdb):
    """Unknown protein style strings should raise a clear ValueError."""
    mol = load(helix_pdb)
    with pytest.raises(ValueError, match="unknown style"):
        render(mol, protein="not-a-style", orient=False)


def test_sidechain_flag(helix_pdb):
    """--sidechain flag results in more atoms visible than backbone-only mode."""
    mol = load(helix_pdb)
    # The helix fixture has no sidechain atoms (only N/CA/C/O), so this mostly
    # confirms the flag is accepted without error.
    svg = str(render(mol, protein=True, sidechain=True, orient=False))
    assert "<svg" in svg


def test_ligand_highlight_recolors_ligands(hetatm_pdb):
    """Ligand-highlight recolors ligands without needing manual indices."""
    mol = load(hetatm_pdb)
    svg_plain = str(render(mol, protein=True, orient=False, gradient=False, config="flat"))
    svg_lig_a = str(
        render(
            mol,
            config="flat",
            protein=True,
            ligand_highlight=True,
            ligand_color="#123abc",
            orient=False,
            gradient=False,
        )
    )
    svg_lig_b = str(
        render(
            mol,
            config="flat",
            protein=True,
            ligand_highlight=True,
            ligand_color="#ff0000",
            orient=False,
            gradient=False,
        )
    )
    assert svg_lig_a != svg_plain
    assert svg_lig_b != svg_plain
    assert svg_lig_a != svg_lig_b


def test_ligand_halo_selector_uses_ligand_color(hetatm_pdb):
    """halo='ligand' should target semantic ligands and reuse ligand_color when provided."""
    mol = load(hetatm_pdb)
    svg = str(
        render(
            mol,
            config="flat",
            halo="ligand",
            ligand_color="#12ab34",
            halo_blur=False,
            orient=False,
            gradient=False,
        )
    )
    halos = [ln for ln in svg.splitlines() if "fill-opacity" in ln and 'stroke="none"' in ln]
    assert len(halos) == 2  # two ligand HETATM atoms in fixture
    assert resolve_color("#12ab34") in svg


def test_protein_mode_keeps_nci_dotted_edges_when_backbone_hidden():
    """Protein mode should still render dotted NCI edges with hidden protein endpoints."""
    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))  # protein backbone atom (hidden)
    g.add_node(1, symbol="O", position=(1.4, 0.0, 0.0))  # ligand atom (visible)
    g.add_edge(0, 1, NCI=True)

    pd = ProteinData(
        chains={
            "A": ChainData(
                chain_id="A",
                residues=[
                    ResidueData(
                        res_name="ALA",
                        res_seq=1,
                        chain_id="A",
                        atom_indices=[0],
                        ca_index=0,
                        c_index=None,
                        o_index=None,
                        n_index=None,
                        ss_type="C",
                    )
                ],
            )
        },
        hetatm_indices={1},
        backbone_indices={0},
        sidechain_indices=set(),
        helix_spans=[],
        sheet_spans=[],
        ligand_indices={1},
        water_indices=set(),
        ion_indices=set(),
    )
    from xyzrender.api import Molecule

    mol = Molecule(graph=g, protein_data=pd)
    svg = str(render(mol, protein=True, orient=False))
    assert "stroke-dasharray" in svg


def test_filter_ligand_protein_nci_keeps_ligand_protein_components():
    """NCI filter keeps only NCI components containing both ligand and protein context."""
    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))  # protein
    g.add_node(1, symbol="N", position=(1.0, 0.0, 0.0))  # protein
    g.add_node(2, symbol="C", position=(2.0, 0.0, 0.0))  # ligand
    g.add_node(3, symbol="O", position=(3.0, 0.0, 0.0))  # ligand
    g.add_node(4, symbol="*", position=(2.5, 0.5, 0.0))  # centroid in kept component
    g.add_node(5, symbol="*", position=(3.5, 0.5, 0.0))  # centroid in dropped component
    g.add_node(6, symbol="C", position=(4.0, 0.0, 0.0))  # other
    g.add_node(7, symbol="C", position=(5.0, 0.0, 0.0))  # other
    g.add_edge(0, 1, NCI=True)  # protein-protein component -> drop
    g.add_edge(2, 4, NCI=True)  # ligand-centroid -> keep (bridges to protein below)
    g.add_edge(4, 1, NCI=True)  # centroid-protein -> keep
    g.add_edge(3, 5, NCI=True)  # ligand-only component -> drop
    g.add_edge(6, 7, NCI=True)  # other-only component -> drop
    g.add_edge(3, 6)  # non-NCI edge should remain untouched

    pd = ProteinData(
        chains={
            "A": ChainData(
                chain_id="A",
                residues=[
                    ResidueData(
                        res_name="ALA",
                        res_seq=1,
                        chain_id="A",
                        atom_indices=[0, 1],
                        ca_index=None,
                        c_index=None,
                        o_index=None,
                        n_index=None,
                        ss_type="C",
                    )
                ],
            )
        },
        hetatm_indices={2, 3},
        backbone_indices=set(),
        sidechain_indices=set(),
        helix_spans=[],
        sheet_spans=[],
        ligand_indices={2, 3},
        water_indices=set(),
        ion_indices=set(),
    )
    fg = filter_ligand_protein_nci(g, pd)
    nci_edges = {(min(i, j), max(i, j)) for i, j, d in fg.edges(data=True) if d.get("NCI", False)}
    assert nci_edges == {(0, 1), (1, 4), (2, 4)}
    assert 4 in fg.nodes(), "centroid in kept component should remain"
    assert 5 not in fg.nodes(), "orphan centroid should be removed"
    assert fg.has_edge(3, 6), "non-NCI edges should not be removed by NCI filtering"


def test_demo_fixture_has_ligand_filtered_nci_edges():
    """The committed protein+ligand demo fixture should retain ligand-filtered NCI edges."""
    if not (STRUCTURES / "protein_ligand_demo.pdb").exists():
        pytest.skip("protein_ligand_demo.pdb fixture not present in this checkout")
    mol = load(STRUCTURES / "protein_ligand_demo.pdb", nci_ligand_protein_only=True)
    n_nci = sum(1 for _, _, data in mol.graph.edges(data=True) if data.get("NCI", False))
    assert n_nci > 0


def test_demo_fixture_renders_dotted_nci_in_protein_mode():
    """Protein-mode render for the demo fixture should include dotted NCI overlays."""
    if not (STRUCTURES / "protein_ligand_demo.pdb").exists():
        pytest.skip("protein_ligand_demo.pdb fixture not present in this checkout")
    mol = load(STRUCTURES / "protein_ligand_demo.pdb", nci_ligand_protein_only=True)
    svg = str(render(mol, protein=True, nci_ligand_protein_only=True, orient=False))
    assert "stroke-dasharray" in svg


# ---------------------------------------------------------------------------
# Secondary-structure inference accuracy (regression floor)
# ---------------------------------------------------------------------------

_8UWL = STRUCTURES / "8UWL.pdb"


def _records_ss(pdb_text: str) -> dict[tuple[str, int], str]:
    """Per-residue H/E labels from a PDB's HELIX/SHEET records."""
    truth: dict[tuple[str, int], str] = {}
    for ln in pdb_text.splitlines():
        if ln.startswith("HELIX "):
            for r in range(int(ln[21:25]), int(ln[33:37]) + 1):
                truth[(ln[19], r)] = "H"
        elif ln.startswith("SHEET "):
            for r in range(int(ln[22:26]), int(ln[33:37]) + 1):
                truth.setdefault((ln[21], r), "E")
    return truth


def _assigned_ss(path: Path) -> dict[tuple[str, int], str]:
    from xyzrender.protein_semantics import extract_protein_semantics

    mol = load(path)
    sem = mol.protein_semantics or extract_protein_semantics(mol.graph, source_path=str(path), protein_requested=True)
    assert sem is not None
    return {(cid, res.res_seq): res.ss_type for cid, chain in sem.chains.items() for res in chain.residues}


def test_ss_records_take_precedence_over_inference():
    """With HELIX/SHEET present, every record-labelled residue keeps its record label."""
    if not _8UWL.exists():
        pytest.skip("8UWL.pdb fixture not present in this checkout")
    truth = _records_ss(_8UWL.read_text())
    got = _assigned_ss(_8UWL)
    common = set(truth) & set(got)
    assert len(common) > 500, "expected a substantial record overlap"
    assert all(truth[k] == got[k] for k in common)


def test_ss_geometry_inference_accuracy_floor(tmp_path):
    """Blind test: strip HELIX/SHEET, infer from CA geometry, score against the records.

    Guards the inference constants in ``xyzgraph.protein``.  Coil recall is
    the weak axis: extended coil partly satisfies the strand test.  The
    strand-pairing filter in xyzgraph took coil recall from 63.6% to 71.8%
    and overall agreement from 83.3% to 85.8%.
    """
    if not _8UWL.exists():
        pytest.skip("8UWL.pdb fixture not present in this checkout")
    text = _8UWL.read_text()
    blind = tmp_path / "8UWL_noss.pdb"
    blind.write_text("\n".join(ln for ln in text.splitlines() if not ln.startswith(("HELIX ", "SHEET "))))

    truth_he = _records_ss(text)
    got = _assigned_ss(blind)
    truth = {k: truth_he.get(k, "C") for k in got}

    overall = sum(truth[k] == got[k] for k in got) / len(got)
    recall = {}
    for label in "HEC":
        n = sum(1 for k in got if truth[k] == label)
        recall[label] = sum(1 for k in got if truth[k] == label and got[k] == label) / n

    assert overall >= 0.85, f"overall SS agreement regressed to {overall:.1%}"
    assert recall["H"] >= 0.93, f"helix recall regressed to {recall['H']:.1%}"
    assert recall["E"] >= 0.92, f"sheet recall regressed to {recall['E']:.1%}"
    assert recall["C"] >= 0.71, f"coil recall regressed to {recall['C']:.1%}"


# ---------------------------------------------------------------------------
# Swept-cartoon geometry contract
#
# These replace an earlier suite that asserted the retired implementation's
# artifacts: whole-run <polygon> strips (`poly_count <= 2`), stroked <path>
# loops, and `rg_h_*` / `rg_e_*` gradient ids.  The cartoon is now a swept
# surface emitted one depth-sorted quad at a time with per-quad Lambert
# shading, so those assertions described bugs rather than requirements.
# ---------------------------------------------------------------------------


def _ribbon_items(pdb, cfg=None, **cfg_kw):
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    mol = load(pdb)
    pd = _require_protein_data(mol.protein_data)
    cfg = cfg or RenderConfig(protein=True, **cfg_kw)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])
    return ribbon_svg_items(pd, cfg, pos, 50.0, 0.0, 0.0, 800, 800)


def test_ribbon_emits_per_quad_items(helix_pdb):
    """Depth granularity is per quad, not per secondary-structure run.

    One item per run was the root cause of helices painting as flat
    self-overlapping pancakes: a whole helix sorted at a single centroid z
    cannot occlude its own turns.
    """
    items = _ribbon_items(helix_pdb)
    assert len(items) > 10
    assert all(len(lines) == 1 for _, lines in items)
    assert all("<polygon" in lines[0] for _, lines in items)


def test_ribbon_items_are_depth_sorted(helix_pdb):
    """Items ascend in z so the renderer's drain loop interleaves correctly."""
    zs = [z for z, _ in _ribbon_items(helix_pdb)]
    assert zs == sorted(zs)


def test_ribbon_quads_are_individually_shaded(helix_pdb):
    """Per-quad normals must yield a range of fills, not one flat colour."""
    import re

    fills = {
        m.group(1)
        for _, lines in _ribbon_items(helix_pdb)
        for m in [re.search(r'fill="(#[0-9a-fA-F]{6})"', lines[0])]
        if m
    }
    assert len(fills) >= 3


def test_ribbon_emits_no_gradient_defs(helix_pdb):
    """Shading is baked into solid fills; no <linearGradient> is needed.

    The old objectBoundingBox gradients ramped across each polygon's screen
    bounding box rather than across the ribbon's local width.
    """
    from xyzrender.ribbon import ribbon_gradient_defs, ribbon_style_uses_gradients

    assert ribbon_gradient_defs({"A": "#4682b4"}) == []
    assert ribbon_style_uses_gradients("gloss") is False
    svg = str(render(load(helix_pdb), protein=True, orient=False))
    assert "rg_h_" not in svg
    assert "rg_e_" not in svg


def test_ribbon_default_width_matches_helix_pitch():
    """Default tape width is near the ~5.4 A helix rise, so turns nearly close up."""
    from xyzrender.types import RenderConfig

    assert RenderConfig().ribbon_width == pytest.approx(4.5)


def test_chain_color_reaches_the_ribbon(two_chain_pdb):
    """An explicit chain colour must appear in that chain's quad fills."""
    from xyzrender.colors import Color

    svg = str(render(load(two_chain_pdb), protein=True, chain_colors={"A": "steelblue"}, orient=False))
    base = Color.from_str("steelblue")
    # Fills are Lambert-tinted, so look for the hue rather than an exact hex.
    assert any(
        abs(Color.from_hex(m).to_hls()[0] - base.to_hls()[0]) < 0.03
        for m in __import__("re").findall(r'fill="(#[0-9a-fA-F]{6})"', svg)
    )


def test_exclude_chain_removes_its_geometry(two_chain_pdb):
    """Excluding a chain drops its quads (and shrinks the emitted SVG)."""
    kw = {"protein": True, "chain_colors": {"A": "steelblue", "B": "red"}, "orient": False}
    svg_all = str(render(load(two_chain_pdb), **kw))
    svg_excl = str(render(load(two_chain_pdb), exclude_chains="B", **kw))
    assert svg_excl.count("<polygon") < svg_all.count("<polygon")


def test_named_styles_render(helix_pdb):
    """Both named styles still produce cartoon geometry."""
    for style in ("gloss", "illustration"):
        svg = str(render(load(helix_pdb), protein=style, orient=False))
        assert svg.count("<polygon") > 5


def test_illustration_style_is_slimmer_and_outlined():
    """Illustration is the flatter, contoured variant of the same geometry."""
    from xyzrender.ribbon import ribbon_style_profile

    il, gl = ribbon_style_profile("illustration"), ribbon_style_profile("gloss")
    assert il.width_scale < gl.width_scale
    assert il.thickness_scale < gl.thickness_scale
    assert il.shade_gain < gl.shade_gain
    assert il.outline_px > 0


def test_ss_transition_is_continuous():
    """A single-residue H/C/H alternation still sweeps one connected surface.

    The old code emitted an unsplined straight quad at a third width across
    every SS junction, which read as a notch.
    """
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    pd = ProteinData(
        chains={
            "A": ChainData(
                chain_id="A",
                residues=[
                    ResidueData("ALA", 1, "A", [0, 3], ca_index=0, c_index=None, o_index=3, n_index=None, ss_type="H"),
                    ResidueData("GLY", 2, "A", [1, 4], ca_index=1, c_index=None, o_index=4, n_index=None, ss_type="C"),
                    ResidueData("SER", 3, "A", [2, 5], ca_index=2, c_index=None, o_index=5, n_index=None, ss_type="H"),
                ],
            )
        },
        hetatm_indices=set(),
        backbone_indices={0, 1, 2, 3, 4, 5},
        sidechain_indices=set(),
        helix_spans=[],
        sheet_spans=[],
    )
    pos = np.array(
        [[0.0, 0, 0], [3.8, 0, 0], [7.6, 0, 0], [0.5, 1.2, 0], [4.3, 1.2, 0], [8.1, 1.2, 0]],
        dtype=float,
    )
    items = ribbon_svg_items(pd, RenderConfig(protein=True), pos, 50.0, 0.0, 0.0, 800, 800)
    assert len(items) > 5


def test_frame_is_flip_corrected():
    """Successive ribbon normals never invert.

    Carbonyls alternate by ~180 degrees along a strand; without sign
    propagation the tape would flip on every residue.
    """
    import numpy as np

    from xyzrender.ribbon import _central_tangents, _frame_normals

    n = 12
    ca = np.stack([np.arange(n) * 3.8, np.zeros(n), np.zeros(n)], axis=1).astype(float)
    # Alternating carbonyl directions, as in a real beta strand.
    o = [ca[i] + np.array([0.0, 1.2 * (-1) ** i, 0.3]) for i in range(n)]
    normals = _frame_normals(ca, o, _central_tangents(ca), ["E"] * n)
    dots = np.einsum("ij,ij->i", normals[:-1], normals[1:])
    assert bool((dots > 0).all()), f"frame flips at {np.where(dots <= 0)[0]}"


def test_helix_frame_tracks_the_axis():
    """On an ideal helix the tape's width runs along the helix axis.

    This is the Carson-Bugg construction.  Using the curvature vector instead
    (which points at the axis) makes the tape a fin spiralling around it.
    """
    import numpy as np

    from xyzrender.ribbon import _central_tangents, _frame_normals

    # Ideal alpha helix: r=2.3 A, rise 1.5 A/residue, 100 deg/residue.
    n = 20
    t = np.arange(n) * np.deg2rad(100.0)
    ca = np.stack([2.3 * np.cos(t), 2.3 * np.sin(t), np.arange(n) * 1.5], axis=1)
    # Carbonyl points roughly along +z (the axis) in a real helix.
    o = [ca[i] + np.array([0.3 * np.cos(t[i]), 0.3 * np.sin(t[i]), 1.1]) for i in range(n)]
    normals = _frame_normals(ca, o, _central_tangents(ca), ["H"] * n)
    axis = np.array([0.0, 0.0, 1.0])
    # Interior residues only; the ends have one-sided tangents.
    align = np.abs(normals[3:-3] @ axis)
    assert align.mean() > 0.5, f"mean |n . axis| = {align.mean():.2f}"


def test_centripetal_spline_hits_control_points():
    """The spline interpolates its control points (it is not an approximation)."""
    import numpy as np

    from xyzrender.ribbon import _catmull_rom_3d

    pts = np.array([[0.0, 0, 0], [1.0, 2, 0], [3.0, 1, 1], [5.0, 3, 0]])
    curve = _catmull_rom_3d(pts, steps=8)
    for p in pts:
        assert np.min(np.linalg.norm(curve - p, axis=1)) < 1e-6


def test_centripetal_spline_does_not_overshoot():
    """No cusp or overshoot at a tight turn, which uniform Catmull-Rom shows."""
    import numpy as np

    from xyzrender.ribbon import _catmull_rom_3d

    # Hairpin: three nearly-coincident points then a sharp reversal.
    pts = np.array([[0.0, 0, 0], [3.8, 0, 0], [4.2, 3.0, 0], [0.4, 3.2, 0]])
    curve = _catmull_rom_3d(pts, steps=12)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    margin = 0.35 * float(np.linalg.norm(hi - lo))
    assert (curve >= lo - margin).all()
    assert (curve <= hi + margin).all()


def test_strand_gets_an_arrowhead(sheet_pdb):
    """A strand's tape widens to a flange then tapers, as one swept surface."""
    import numpy as np

    from xyzrender.ribbon import _sample_widths

    ss = ["E"] * 8
    res_t = np.linspace(0, 7, 60)
    hw, _, _ = _sample_widths(ss, res_t, 2.0, 2.0, 0.4, 0.3, 0.3, 0.4)
    body = hw[res_t <= 5.0].mean()
    assert hw.max() > body * 1.5, "no flange"
    assert hw[-1] < body * 0.5, "strand does not taper to a point"


def test_short_strand_still_renders(tmp_path):
    """A 2-residue SHEET record is legal PDB and must not be dropped.

    The old code had _MIN_SHEET_RES=2 and made such a strand entirely
    arrowhead; the taper now handles it as a short tapered strand.
    """
    import numpy as np

    from xyzrender.ribbon import _sample_widths

    hw, _, _ = _sample_widths(["E", "E"], np.linspace(0, 1, 16), 2.0, 2.0, 0.4, 0.3, 0.3, 0.4)
    assert np.isfinite(hw).all()
    assert hw.max() > 0
