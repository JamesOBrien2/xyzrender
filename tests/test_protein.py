"""Tests for protein ribbon rendering."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import networkx as nx
import pytest

from xyzrender import load, render
from xyzrender.colors import resolve_color
from xyzrender.parsers import parse_pdb
from xyzrender.protein_semantics import xyzgraph_protein_available
from xyzrender.readers import filter_ligand_protein_nci
from xyzrender.types import ChainData, ProteinData, ResidueData

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"

# Ribbon rendering hard-depends on `xyzgraph.protein`, which older xyzgraph
# releases do not ship.  Skip rather than fail so the suite stays green on an
# environment that predates it; see protein_semantics.xyzgraph_protein_available.
pytestmark = pytest.mark.skipif(
    not xyzgraph_protein_available(),
    reason="installed xyzgraph does not provide xyzgraph.protein",
)


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


# A ligand belonging to chain B, appended to _TWO_CHAIN_PDB.
_LIGAND_ON_B = dedent("""\
    HETATM   17  C1  LIG B 101      20.000  20.000   0.000  1.00  0.00           C
    HETATM   18  O1  LIG B 101      21.000  20.000   0.000  1.00  0.00           O
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
def sidechain_pdb(tmp_path):
    """Backbone with CB/CG sidechains, and CONECT so bonds exist without rebuild."""
    lines = []
    serial = 1
    conect = []
    for k in range(6):
        base = k * 3.8
        ids = {}
        for name, off, elem in (
            ("N", (0.0, 0.0, 0.0), "N"),
            ("CA", (1.5, 0.0, 0.0), "C"),
            ("C", (2.5, 0.0, 0.0), "C"),
            ("O", (3.0, 0.9, 0.0), "O"),
            ("CB", (1.2, 0.0, 1.5), "C"),
            ("CG", (1.0, 0.0, 3.0), "C"),
        ):
            lines.append(_atom_line(serial, name, "ALA", "A", k + 1, (base + off[0], off[1], off[2]), elem=elem))
            ids[name] = serial
            serial += 1
        conect += [
            (ids["N"], ids["CA"]),
            (ids["CA"], ids["C"]),
            (ids["C"], ids["O"]),
            (ids["CA"], ids["CB"]),
            (ids["CB"], ids["CG"]),
        ]
    for a, b in conect:
        lines.append(f"CONECT{a:>5}{b:>5}")
    p_ = tmp_path / "sidechain.pdb"
    p_.write_text("\n".join(lines) + "\n")
    return p_


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


def _glow_circles(svg: str) -> list[str]:
    import re

    return [m.group(0) for m in re.finditer(r"<circle[^>]*filter=\"url\(#\w*glow\d+\)\"[^>]*/>", svg)]


def test_ligand_glow_selector_uses_ligand_color(hetatm_pdb):
    """glow='ligand' should target semantic ligands and reuse ligand_color when provided."""
    mol = load(hetatm_pdb)
    svg = str(
        render(
            mol,
            config="flat",
            glow="ligand",
            ligand_color="#12ab34",
            orient=False,
            gradient=False,
            fog=False,
        )
    )
    circles = _glow_circles(svg)
    assert len(circles) == 2  # two ligand HETATM atoms in fixture
    assert all(resolve_color("#12ab34") in c for c in circles)


def test_glow_semantic_tokens_resolve_against_semantics(hetatm_class_pdb):
    """'water' and 'ion' pick out their own HETATM classes, not the whole file."""
    mol = load(hetatm_class_pdb)
    svg = str(render(mol, config="flat", glow=[("water", "blue"), ("ion", "red")], orient=False, fog=False))
    circles = _glow_circles(svg)
    fills = [resolve_color(c) for c in ("blue", "red")]
    assert len(circles) == 2
    assert sorted(f for f in fills if any(f in c for c in circles)) == sorted(fills)


def test_glow_semantic_token_warns_without_semantics(caplog):
    """A semantic token on a plain molecule warns and selects nothing."""
    mol = load(STRUCTURES / "caffeine.xyz")
    with caplog.at_level("WARNING", logger="xyzrender.api"):
        svg = str(render(mol, glow="ligand", orient=False))
    assert _glow_circles(svg) == []
    assert "no matching protein semantics are available" in caplog.text


def test_glow_element_selector_still_works_on_a_protein(hetatm_pdb):
    """Semantic tokens are additive: the element grammar must still resolve."""
    mol = load(hetatm_pdb)
    svg = str(render(mol, config="flat", glow="O", orient=False, fog=False))
    assert len(_glow_circles(svg)) == 2  # backbone O + ligand O1


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
    # One quad per item; silhouette strokes ride along in the same item so they
    # paint over their own quad but under anything in front of it.
    assert all(sum("<polygon" in ln for ln in lines) == 1 for _, lines in items)
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


def test_outline_strokes_the_silhouette_not_the_mesh(helix_pdb):
    """Outlines follow the surface boundary, not every interior quad edge.

    Stroking the quads themselves made --protein illustration render as a
    wireframe dump of the swept mesh.
    """
    items = _ribbon_items(helix_pdb, protein_style="illustration")
    quads = sum(sum("<polygon" in ln for ln in lines) for _, lines in items)
    strokes = sum(sum("<line" in ln for ln in lines) for _, lines in items)
    assert strokes > 0
    # A wireframe would stroke all four edges of every quad.
    assert strokes < 2 * quads


def test_gloss_outlines_the_ribbon(helix_pdb):
    """The default style carries a hairline contour, not just illustration."""
    items = _ribbon_items(helix_pdb, protein_style="gloss")
    assert any("<line" in ln for _, lines in items for ln in lines)


def test_ribbon_shading_reaches_both_ends_of_the_ramp(helix_pdb):
    """Shadowed faces must actually darken, or overlapping strands merge."""
    import re

    def lum(hex_str):
        r, g, b = (int(hex_str[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    fills = [
        m.group(1)
        for _, lines in _ribbon_items(helix_pdb)
        for m in [re.search(r'fill="(#[0-9a-fA-F]{6})"', lines[0])]
        if m
    ]
    lums = [lum(f) for f in fills]
    assert max(lums) - min(lums) > 0.25


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


def test_exclude_chain_removes_its_ligand(tmp_path):
    """A chain's heteroatoms go with its ribbon, not float on alone.

    Ligand/water/ion indices carry no chain of their own, so exclusion has to
    reach them through ProteinSemantics.het_chains.
    """
    p = tmp_path / "two_chain_lig.pdb"
    p.write_text(_TWO_CHAIN_PDB.replace("END\n", "") + _LIGAND_ON_B + "END\n")
    mol = load(p)
    kw = {"protein": True, "orient": False}
    assert str(render(mol, **kw)).count("<circle") == 2  # the two LIG atoms
    assert str(render(mol, exclude_chains="B", **kw)).count("<circle") == 0


def test_exclude_chain_does_not_recolour_the_others(two_chain_pdb):
    """B keeps palette[1] when A is excluded, rather than sliding to palette[0].

    Excluding the *first* chain is the discriminating case: filtering before
    allocation renumbered everything after it, so the same chain changed colour
    between an overview and a close-up.
    """
    import re

    kw = {"protein": True, "orient": False}
    mol = load(two_chain_pdb)

    def fills(svg):
        return set(re.findall(r'<polygon[^>]*fill="(#[0-9a-fA-F]{6})"', svg))

    # The fixture's chains are translated copies, so their shade sets differ
    # only by base colour.  Filtering before allocation gave whichever chain
    # survived palette[0], making these two renders identical.
    assert fills(str(render(mol, exclude_chains="A", **kw))) != fills(str(render(mol, exclude_chains="B", **kw)))


def test_lone_amino_acid_ligand_is_still_drawn(tmp_path):
    """A single residue on its own chain must not vanish.

    Backbone names are collected before the peptide test diverts a residue to
    ligand, so its N/CA/C/O stayed in backbone_indices -- which the renderer
    hides -- while its chain was too short to earn a ribbon.
    """
    p = tmp_path / "docked.pdb"
    lig = "".join(
        f"ATOM  {9 + i:>5} {name:^4} GLY L   1    {9.0 + i:>8.3f}{9.0:>8.3f}{0.0:>8.3f}"
        f"{1.0:>6.2f}{0.0:>6.2f}          {elem:>2}\n"
        for i, (name, elem) in enumerate([("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")])
    )
    p.write_text(_TWO_CHAIN_PDB.replace("END\n", "") + lig + "END\n")
    sem = load(p).protein_semantics
    assert len(sem.ligand_indices) == 4
    assert not (sem.ligand_indices & sem.backbone_indices)
    assert str(render(load(p), protein=True, orient=False)).count("<circle") == 4


def test_placeholder_cryst1_draws_no_cell():
    """PDB mandates CRYST1, so non-crystallographic entries carry a 1 A sentinel."""
    from xyzrender.parsers import parse_pdb

    assert parse_pdb(STRUCTURES / "8UWL.pdb").pbc_cell is None
    assert parse_pdb(STRUCTURES / "water_cryst.pdb").pbc_cell is not None


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
    dims = {"H": (2.0, 0.3), "E": (2.0, 0.3), "C": (0.4, 0.4)}
    hw, _, _ = _sample_widths(ss, res_t, dims)
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

    dims = {"H": (2.0, 0.3), "E": (2.0, 0.3), "C": (0.4, 0.4)}
    hw, _, _ = _sample_widths(["E", "E"], np.linspace(0, 1, 16), dims)
    assert np.isfinite(hw).all()
    assert hw.max() > 0


# ---------------------------------------------------------------------------
# PDB record handling: MODEL / altLoc / insertion codes / tempFactor
# ---------------------------------------------------------------------------


def _atom_line(serial, name, res_name, chain, res_seq, xyz, *, altloc=" ", icode=" ", bfac=0.0, elem=None):
    """One fixed-column ATOM record."""
    elem = elem or name[0]
    x, y, z = xyz
    return (
        f"ATOM  {serial:>5} {name:<4}{altloc}{res_name:>3} {chain}{res_seq:>4}{icode}   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{bfac:>6.2f}          {elem:>2}"
    )


def test_multi_model_pdb_uses_first_model_only(tmp_path):
    """An NMR ensemble must not concatenate every model into one structure.

    Without this the models are overlaid *and* res_seq restarts per model, so
    the backbone is shredded into fragments at every model boundary.
    """
    lines = []
    for model in (1, 2, 3):
        lines.append(f"MODEL     {model:>4}")
        for i in range(3):
            lines.append(_atom_line(i + 1, "CA", "ALA", "A", i + 1, (i * 3.8, model * 10.0, 0.0), elem="C"))
        lines.append("ENDMDL")
    p = tmp_path / "nmr.pdb"
    p.write_text("\n".join(lines) + "\n")

    data = parse_pdb(p)
    assert len(data.atoms) == 3, f"expected 1 model (3 atoms), got {len(data.atoms)}"
    # Model 1 was the one kept.
    assert data.atoms[0][1][1] == pytest.approx(10.0)


def test_altloc_keeps_one_conformer(tmp_path):
    """Alternate conformers share a residue key; only altLoc blank/A is kept."""
    lines = [
        _atom_line(1, "N", "SER", "A", 1, (0.0, 0.0, 0.0), elem="N"),
        _atom_line(2, "CA", "SER", "A", 1, (1.5, 0.0, 0.0), altloc="A", elem="C"),
        _atom_line(3, "CA", "SER", "A", 1, (1.5, 0.9, 0.0), altloc="B", elem="C"),
        _atom_line(4, "C", "SER", "A", 1, (2.5, 0.0, 0.0), elem="C"),
    ]
    p = tmp_path / "altloc.pdb"
    p.write_text("\n".join(lines) + "\n")

    data = parse_pdb(p)
    assert len(data.atoms) == 3, "altLoc B should have been dropped"
    pd = _require_protein_data(data.protein_data)
    res = pd.chains["A"].residues[0]
    assert data.atoms[res.ca_index][1][1] == pytest.approx(0.0), "kept the wrong conformer"


def test_insertion_codes_do_not_merge_residues(tmp_path):
    """Residues 100, 100A, 100B are distinct even when res_name repeats.

    Without the insertion code in the residue key they collide, and ca_index
    is overwritten by whichever CA was parsed last -- a silent jump in the
    backbone trace.
    """
    lines = []
    serial = 1
    for k, icode in enumerate((" ", "A", "B")):
        for name, off in (("N", 0.0), ("CA", 1.5), ("C", 2.5)):
            lines.append(
                _atom_line(
                    serial,
                    name,
                    "SER",
                    "A",
                    100,
                    (k * 3.8 + off, 0.0, 0.0),
                    icode=icode,
                    elem=name[0],
                )
            )
            serial += 1
    p = tmp_path / "icode.pdb"
    p.write_text("\n".join(lines) + "\n")

    pd = _require_protein_data(parse_pdb(p).protein_data)
    residues = pd.chains["A"].residues
    assert len(residues) == 3, f"insertion codes merged: {len(residues)} residue(s)"
    assert [r.i_code for r in residues] == ["", "A", "B"]
    assert len({r.ca_index for r in residues}) == 3


def test_insertion_codes_do_not_split_the_ribbon(tmp_path):
    """An insertion repeats res_seq; that is a continuation, not a chain break."""
    import numpy as np

    from xyzrender.ribbon import _split_backbone_segments

    pd = _require_protein_data(
        parse_pdb(
            _write_icode_chain(tmp_path),
        ).protein_data
    )
    residues = pd.chains["A"].residues
    pos = np.array([[i * 3.8, 0.0, 0.0] for i in range(sum(len(r.atom_indices) for r in residues) + 5)])
    for r in residues:
        pos[r.ca_index] = [residues.index(r) * 3.8, 0.0, 0.0]
    segs = _split_backbone_segments(residues, pos)
    assert len(segs) == 1, f"insertion split the backbone into {len(segs)} segments"


def _write_icode_chain(tmp_path):
    lines = []
    serial = 1
    seq_icodes = [(99, " "), (100, " "), (100, "A"), (100, "B"), (101, " ")]
    for k, (seq, icode) in enumerate(seq_icodes):
        for name, off in (("N", 0.0), ("CA", 1.5), ("C", 2.5)):
            lines.append(
                _atom_line(serial, name, "SER", "A", seq, (k * 3.8 + off, 0.0, 0.0), icode=icode, elem=name[0])
            )
            serial += 1
    p = tmp_path / "icode_chain.pdb"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_temperature_factor_is_parsed(tmp_path):
    """B-factor / pLDDT is read from cols 61-66 onto the residue."""
    lines = [
        _atom_line(1, "N", "ALA", "A", 1, (0.0, 0.0, 0.0), bfac=11.0, elem="N"),
        _atom_line(2, "CA", "ALA", "A", 1, (1.5, 0.0, 0.0), bfac=42.5, elem="C"),
        _atom_line(3, "C", "ALA", "A", 1, (2.5, 0.0, 0.0), bfac=13.0, elem="C"),
    ]
    p = tmp_path / "bfac.pdb"
    p.write_text("\n".join(lines) + "\n")

    pd = _require_protein_data(parse_pdb(p).protein_data)
    # The CA value is the per-residue B-factor (pLDDT is reported on CA).
    assert pd.chains["A"].residues[0].b_factor == pytest.approx(42.5)


# ---------------------------------------------------------------------------
# Sidechain attachment and selection
# ---------------------------------------------------------------------------


def test_sidechain_anchor_lies_outside_the_ribbon(sidechain_pdb):
    """The attachment point clears the swept surface rather than sitting at CA.

    CA is the tape's centreline, so a bond starting there is buried inside the
    solid; the anchor is where the CA->CB ray exits the surface.
    """
    import numpy as np

    from xyzrender.ribbon import sidechain_anchors
    from xyzrender.types import RenderConfig

    mol = load(sidechain_pdb)
    pd = _require_protein_data(mol.protein_data)
    pos = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])
    cfg = RenderConfig(protein=True, show_sidechain=True)

    anchors = sidechain_anchors(pd, cfg, pos)
    assert anchors, "no sidechain anchors produced"
    for ca_idx, pt in anchors.items():
        assert np.linalg.norm(pt - pos[ca_idx]) > 1e-3, "anchor collapsed onto CA"


def test_sidechain_bond_survives_hidden_ca(sidechain_pdb):
    """--sidechain must emit the CA-CB bond even though CA is hidden.

    Regression for the original defect: renderer.py dropped every bond with a
    hidden endpoint, so sidechains floated unattached to the ribbon.
    """
    mol = load(sidechain_pdb)
    plain = str(render(mol, protein=True, orient=False))
    with_sc = str(render(mol, protein=True, sidechain=True, orient=False))
    assert with_sc.count("<line") > plain.count("<line")


def test_select_sidechain_atoms_by_residue_number(two_chain_pdb):
    """Residue selectors accept numbers, ranges and chain qualifiers."""
    from xyzrender.ribbon import select_sidechain_atoms

    pd = _require_protein_data(load(two_chain_pdb).protein_data)
    all_seqs = {r.res_seq for c in pd.chains.values() for r in c.residues}
    first = min(all_seqs)

    everything = select_sidechain_atoms(pd, None)
    just_one = select_sidechain_atoms(pd, str(first))
    assert just_one
    assert just_one < everything

    chain_a = next(iter(pd.chains))
    qualified = select_sidechain_atoms(pd, f"{chain_a}:{first}")
    assert qualified <= just_one


def test_select_sidechain_atoms_rejects_garbage(two_chain_pdb):
    from xyzrender.ribbon import select_sidechain_atoms

    pd = _require_protein_data(load(two_chain_pdb).protein_data)
    with pytest.raises(ValueError, match="invalid residue selector"):
        select_sidechain_atoms(pd, "not-a-residue")


def test_sidechain_selector_restricts_what_is_drawn(sidechain_pdb):
    """A restricted selector draws fewer atoms than bare --sidechain."""
    mol = load(sidechain_pdb)
    everything = str(render(mol, protein=True, sidechain=True, orient=False))
    restricted = str(render(mol, protein=True, sidechain="1", orient=False))
    assert restricted.count("<circle") < everything.count("<circle")


# ---------------------------------------------------------------------------
# Colour modes
# ---------------------------------------------------------------------------


def _fills(svg: str) -> set[str]:
    import re

    return set(re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg))


def test_color_by_modes_are_validated():
    from xyzrender.ribbon import COLOR_BY_MODES, normalize_color_by

    assert normalize_color_by(None) == "chain"
    for mode in COLOR_BY_MODES:
        assert normalize_color_by(mode.upper()) == mode
    with pytest.raises(ValueError, match="unknown color-by mode"):
        normalize_color_by("bogus")


def _residues(pdb):
    pd = _require_protein_data(load(pdb).protein_data)
    return [r for c in pd.chains.values() for r in c.residues]


def test_color_by_ss_uses_distinct_colors_per_structure(helix_pdb, sheet_pdb):
    """Helix, sheet and coil each get their own colour."""
    from xyzrender.ribbon import _residue_colors

    h = _residue_colors(_residues(helix_pdb), "ss", "#888888", b_range=None)
    e = _residue_colors(_residues(sheet_pdb), "ss", "#888888", b_range=None)
    assert len(set(h)) == 1
    assert len(set(e)) == 1
    assert set(h).isdisjoint(e), "helix and sheet share a colour under color_by='ss'"


def test_color_by_rainbow_ramps_n_to_c(sheet_pdb):
    """Rainbow walks a spectral ramp from the N terminus to the C terminus."""
    from xyzrender.ribbon import _residue_colors

    residues = _residues(sheet_pdb)
    colors = _residue_colors(residues, "rainbow", "#888888", b_range=None)
    assert len(colors) == len(residues)
    assert len(set(colors)) == len(residues), "rainbow repeated a colour"
    assert colors[0] != colors[-1]


def test_color_by_reaches_the_rendered_output(sheet_pdb):
    """Switching mode actually changes the emitted SVG."""
    mol = load(sheet_pdb)
    by_chain = str(render(mol, protein=True, color_by="chain", orient=False))
    by_ss = str(render(mol, protein=True, color_by="ss", orient=False))
    assert _fills(by_chain) != _fills(by_ss)


def test_color_by_bfactor_tracks_the_temperature_factor(tmp_path):
    """Residues with different B-factors get different colours."""
    lines = []
    serial = 1
    for k in range(6):
        for name, off, elem in (("N", 0.0, "N"), ("CA", 1.5, "C"), ("C", 2.5, "C"), ("O", 3.0, "O")):
            lines.append(
                _atom_line(serial, name, "ALA", "A", k + 1, (k * 3.8 + off, 0.0, 0.0), bfac=10.0 * k, elem=elem)
            )
            serial += 1
    p = tmp_path / "bfac_chain.pdb"
    p.write_text("\n".join(lines) + "\n")

    mol = load(p)
    flat = _fills(str(render(mol, protein=True, color_by="chain", orient=False)))
    graded = _fills(str(render(mol, protein=True, color_by="bfactor", orient=False)))
    assert len(graded) > len(flat)


def test_color_by_bfactor_falls_back_when_uniform(helix_pdb, caplog):
    """A structure with no B-factor variation warns and keeps chain colours."""
    import logging

    with caplog.at_level(logging.WARNING):
        svg = str(render(load(helix_pdb), protein=True, color_by="bfactor", orient=False))
    assert svg.count("<polygon") > 0
    assert any("same B-factor" in r.message for r in caplog.records)


def test_bfactor_range_ignores_excluded_chains(two_chain_pdb):
    from xyzrender.ribbon import bfactor_range

    pd = _require_protein_data(load(two_chain_pdb).protein_data)
    # All fixture B-factors are 0.00, so the range is degenerate -> None.
    assert bfactor_range(pd, set()) is None


def test_chain_palette_wrap_is_distinguishable():
    """Chain 9 must not be identical to chain 1 when the palette wraps."""
    from xyzrender.ribbon import assign_chain_colors
    from xyzrender.types import RenderConfig

    cfg = RenderConfig(protein=True)
    n = len(cfg.protein_palette)
    ids = [f"C{i}" for i in range(n * 2)]
    colors = assign_chain_colors(cfg, ids)
    assert colors[ids[0]] != colors[ids[n]], "palette wrapped to an identical colour"
