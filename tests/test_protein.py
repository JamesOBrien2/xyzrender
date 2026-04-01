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


def test_ribbon_helix_is_solid_strip_by_default(helix_pdb):
    """Default protein style renders helix runs as near-solid strip polygons."""
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    mol = load(helix_pdb)
    pd = mol.protein_data
    assert pd is not None
    cfg = RenderConfig(protein=True)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])

    items = ribbon_svg_items(pd, cfg, pos, 50.0, 0.0, 0.0, 800, 800)
    all_svg = " ".join(line for _, lines in items for line in lines)
    poly_count = all_svg.count("<polygon")
    assert poly_count <= 2


def test_ribbon_surface_panels_are_fill_only(helix_pdb):
    """Ribbon sweeps should remain near-solid (no per-facet panel explosion)."""
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    mol = load(helix_pdb)
    cfg = RenderConfig(protein=True)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])
    pd = _require_protein_data(mol.protein_data)
    items = ribbon_svg_items(pd, cfg, pos, 50.0, 0.0, 0.0, 800, 800)
    all_svg = " ".join(line for _, lines in items for line in lines)

    assert all_svg.count("<polygon") <= 3


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


def test_ribbon_default_width_is_thicker():
    """Protein ribbon defaults keep the tuned width target."""
    from xyzrender.types import RenderConfig

    cfg = RenderConfig()
    assert cfg.ribbon_width == pytest.approx(2.2)


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


def test_ribbon_sheet_has_reduced_panel_count(sheet_pdb):
    """Sheet sweeps should avoid excessive panelization/striping."""
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    mol = load(sheet_pdb)
    cfg = RenderConfig(protein=True)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])
    pd = _require_protein_data(mol.protein_data)
    items = ribbon_svg_items(pd, cfg, pos, 50.0, 0.0, 0.0, 800, 800)
    all_svg = " ".join(line for _, lines in items for line in lines)
    poly_count = all_svg.count("<polygon")
    assert poly_count <= 12


def test_ribbon_styles_both_use_solid_helix_strips(helix_pdb):
    """Both gloss and illustration styles should avoid faceted helix panels."""
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    mol = load(helix_pdb)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])
    pd = _require_protein_data(mol.protein_data)
    gloss_items = ribbon_svg_items(pd, RenderConfig(protein=True), pos, 50.0, 0.0, 0.0, 800, 800)
    illustration_items = ribbon_svg_items(
        pd,
        RenderConfig(protein=True, protein_style="illustration"),
        pos,
        50.0,
        0.0,
        0.0,
        800,
        800,
    )
    gloss_svg = " ".join(line for _, lines in gloss_items for line in lines)
    illustration_svg = " ".join(line for _, lines in illustration_items for line in lines)
    assert gloss_svg.count("<polygon") <= 2
    assert illustration_svg.count("<polygon") <= 2


def test_ribbon_loop_produces_path():
    """Loop regions produce <path> elements."""
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    # ala_phe_ala has no HELIX/SHEET → all coil
    mol = load(STRUCTURES / "ala_phe_ala.pdb")
    pd = mol.protein_data
    if pd is None:
        pytest.skip("ala_phe_ala has no chain info")
    cfg = RenderConfig(protein=True)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])

    items = ribbon_svg_items(pd, cfg, pos, 50.0, 0.0, 0.0, 800, 800)
    all_svg = " ".join(line for _, lines in items for line in lines)
    assert "<path" in all_svg


def test_loop_items_use_single_stroke_segments():
    """Loop connectors should render as one darker stroke (not dual-stroke tubes)."""
    import numpy as np

    from xyzrender.ribbon import _loop_items, ribbon_style_profile

    ca_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.2, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    style = ribbon_style_profile("illustration")
    items = _loop_items(
        ca_pos,
        "#7aa9b2",
        stroke_w=4.0,
        style=style,
        scale=80.0,
        cx=0.0,
        cy=0.0,
        canvas_w=800.0,
        canvas_h=800.0,
    )
    assert items
    assert all(len(lines) == 1 for _, lines in items)


def test_ribbon_ss_transitions_are_continuous():
    """Single-residue H/C/H transitions still emit connector geometry."""
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    # Residue 1 (H), 2 (C), 3 (H): old segmentation would drop all runs
    # because each run length is 1. Continuity-aware rendering should still
    # draw transition connector quads.
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
        ligand_indices=set(),
        water_indices=set(),
        ion_indices=set(),
    )
    pos = np.array(
        [
            [0.0, 0.0, 0.0],   # CA1
            [1.5, 0.0, 0.0],   # CA2
            [3.0, 0.0, 0.0],   # CA3
            [0.0, 0.8, 0.0],   # O1
            [1.5, 0.8, 0.0],   # O2
            [3.0, 0.8, 0.0],   # O3
        ],
        dtype=float,
    )
    cfg = RenderConfig(protein=True)
    items = ribbon_svg_items(pd, cfg, pos, 80.0, 0.0, 0.0, 800, 800)
    all_svg = " ".join(line for _, lines in items for line in lines)
    assert "<polygon" in all_svg
    assert 'stroke="none"' not in all_svg


def test_sheet_arrowheads_have_contour_strokes(sheet_pdb):
    """Sheet arrowhead triangles should include contour stroke for consistency."""
    import numpy as np

    from xyzrender.ribbon import ribbon_svg_items
    from xyzrender.types import RenderConfig

    mol = load(sheet_pdb)
    cfg = RenderConfig(protein=True)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])

    pd = _require_protein_data(mol.protein_data)
    items = ribbon_svg_items(pd, cfg, pos, 50.0, 0.0, 0.0, 800, 800)
    triangle_polys = [
        line
        for _, lines in items
        for line in lines
        if "<polygon" in line and line.count(",") == 3
    ]
    assert triangle_polys
    assert all('stroke="none"' not in line for line in triangle_polys)


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


def test_protein_chain_color_in_svg(two_chain_pdb):
    """Explicit chain colour is wired into the ribbon gradient for the correct chain."""
    from xyzrender.colors import resolve_color

    mol = load(two_chain_pdb)
    svg = str(render(mol, protein=True, chain_colors={"A": "steelblue"}, orient=False))
    # Ribbon gradients are keyed by chain colour hex in helix/sheet ids.
    hex_color = resolve_color("steelblue")[1:]  # strip '#'
    assert f"rg_h_{hex_color}" in svg
    assert f"rg_e_{hex_color}" in svg


def test_protein_helix_gradient_refs_are_valid(helix_pdb):
    """Helix surfaces should reference rg_h_* gradients (never rg_*)."""
    import re

    mol = load(helix_pdb)
    svg = str(render(mol, protein=True, orient=False))
    assert "rg_h_" in svg
    assert re.search(r'url\(#x\d+rg_h_[0-9A-Fa-f]{6}\)', svg) is not None
    assert re.search(r'url\(#x\d+rg_[0-9A-Fa-f]{6}\)', svg) is None


@pytest.mark.parametrize("style", ["illustration", "gloss"])
def test_protein_named_styles_render_with_ribbon_gradients(helix_pdb, style):
    """Supported named styles should emit ribbon gradient defs."""
    mol = load(helix_pdb)
    svg = str(render(mol, protein=style, orient=False))
    assert "rg_h_" in svg
    assert "rg_e_" in svg


def test_protein_gloss_style_explicit_string(helix_pdb):
    """protein='gloss' remains equivalent to protein=True behavior."""
    import re

    mol = load(helix_pdb)
    svg = str(render(mol, protein="gloss", orient=False))
    assert "rg_h_" in svg
    assert re.search(r'url\(#x\d+rg_h_[0-9A-Fa-f]{6}\)', svg) is not None


@pytest.mark.parametrize("style", ["cartoon", "plastic", "matte", "pymol", "illustrative"])
def test_removed_protein_styles_raise(helix_pdb, style):
    """Removed legacy styles should fail fast with a validation error."""
    mol = load(helix_pdb)
    with pytest.raises(ValueError, match="unknown style"):
        render(mol, protein=style, orient=False)


def test_illustration_profile_is_slimmer_than_gloss():
    """Illustration style should be materially slimmer than gloss defaults."""
    from xyzrender.ribbon import ribbon_style_profile

    il = ribbon_style_profile("illustration")
    gl = ribbon_style_profile("gloss")
    assert il.ribbon_scale / gl.ribbon_scale <= 0.70
    assert il.ribbon_scale / gl.ribbon_scale >= 0.50


def test_loop_width_ratio_target_locked():
    """Illustration defaults keep loop tube width at ~45% of local ribbon width."""
    from xyzrender.ribbon import _LOOP_WIDTH_RATIO

    assert _LOOP_WIDTH_RATIO == pytest.approx(0.45)


def test_illustration_style_uses_round_loop_paths(two_chain_pdb):
    """Illustration style should render coil runs as rounded tube paths."""
    mol = load(two_chain_pdb)
    svg = str(render(mol, protein="illustration", orient=False))
    assert "<path" in svg


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


def test_protein_exclude_chain_removes_chain_gradient(two_chain_pdb):
    """Excluded chains should not emit ribbon gradient defs for that chain."""
    mol = load(two_chain_pdb)
    svg_all = str(render(mol, protein=True, chain_colors={"A": "steelblue", "B": "red"}, orient=False))
    svg_excl = str(
        render(
            mol,
            protein=True,
            chain_colors={"A": "steelblue", "B": "red"},
            exclude_chains="B",
            orient=False,
        )
    )
    assert "rg_h_ff0000" in svg_all
    assert "rg_h_ff0000" not in svg_excl


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
    mol = load(STRUCTURES / "protein_ligand_demo.pdb", nci_ligand_protein_only=True)
    n_nci = sum(1 for _, _, data in mol.graph.edges(data=True) if data.get("NCI", False))
    assert n_nci > 0


def test_demo_fixture_renders_dotted_nci_in_protein_mode():
    """Protein-mode render for the demo fixture should include dotted NCI overlays."""
    mol = load(STRUCTURES / "protein_ligand_demo.pdb", nci_ligand_protein_only=True)
    svg = str(render(mol, protein=True, nci_ligand_protein_only=True, orient=False))
    assert "stroke-dasharray" in svg
