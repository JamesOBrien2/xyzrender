"""Tests for --hl (highlight) atom group coloring."""

from pathlib import Path

import pytest

from xyzrender import load, render
from xyzrender.colors import Color, resolve_color

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


@pytest.fixture(scope="module")
def caffeine():
    return load(STRUCTURES / "caffeine.xyz")


# ---------------------------------------------------------------------------
# Single-group highlight (backward compat)
# ---------------------------------------------------------------------------


def test_highlight_color_in_svg(caffeine):
    """Orchid (default palette[0]) appears as fill on highlighted atoms."""
    svg = str(render(caffeine, highlight=[1, 2, 3], gradient=False, fog=False, orient=False))
    assert resolve_color("orchid") in svg


def test_highlight_custom_color(caffeine):
    """Explicit color via tuple form."""
    svg = str(render(caffeine, highlight=[([1, 2, 3], "steelblue")], gradient=False, fog=False, orient=False))
    assert resolve_color("steelblue") in svg
    assert resolve_color("orchid") not in svg


def test_highlight_bond_darkened(caffeine):
    """Bonds between highlighted atoms get darkened color."""
    svg = str(render(caffeine, highlight=[1, 2, 3], gradient=False, fog=False, orient=False))
    orchid = Color.from_str(resolve_color("orchid"))
    dark = orchid.blend(Color(0, 0, 0), 0.3).hex
    assert dark in svg


def test_highlight_no_mutation(caffeine):
    """Molecule graph is not mutated by highlight rendering."""
    pos_before = {n: caffeine.graph.nodes[n]["position"] for n in caffeine.graph.nodes()}
    render(caffeine, highlight=[1, 2, 3, 4, 5], orient=False)
    for n in caffeine.graph.nodes():
        assert caffeine.graph.nodes[n]["position"] == pos_before[n]


def test_highlight_string_api(caffeine):
    """Python API accepts 1-indexed string."""
    svg = str(render(caffeine, highlight="1-3", gradient=False, fog=False, orient=False))
    assert resolve_color("orchid") in svg


def test_highlight_list_api(caffeine):
    """Python API accepts 1-indexed list."""
    svg = str(render(caffeine, highlight=[1, 2, 3], gradient=False, fog=False, orient=False))
    assert resolve_color("orchid") in svg


# ---------------------------------------------------------------------------
# Multi-group highlight
# ---------------------------------------------------------------------------


def test_multi_group_explicit_colors(caffeine):
    """Two groups with explicit colors — both colors appear in SVG."""
    svg = str(
        render(
            caffeine,
            highlight=[([1, 2, 3], "blue"), ([5, 6, 7, 8], "red")],
            gradient=False,
            fog=False,
            orient=False,
        )
    )
    assert resolve_color("blue") in svg
    assert resolve_color("red") in svg


def test_multi_group_auto_color(caffeine):
    """Multiple groups without colors get palette colors."""
    svg = str(
        render(
            caffeine,
            highlight=[[1, 2, 3], [5, 6, 7, 8]],
            gradient=False,
            fog=False,
            orient=False,
        )
    )
    # First group gets palette[0] = orchid, second gets palette[1] = mediumseagreen
    assert resolve_color("orchid") in svg
    assert resolve_color("mediumseagreen") in svg


def test_multi_group_string_specs(caffeine):
    """Multi-group using string atom specs."""
    svg = str(
        render(
            caffeine,
            highlight=["1-3", "5-8"],
            gradient=False,
            fog=False,
            orient=False,
        )
    )
    assert resolve_color("orchid") in svg
    assert resolve_color("mediumseagreen") in svg


def test_multi_group_tuple_string_specs(caffeine):
    """Multi-group using string atom specs with explicit colors."""
    svg = str(
        render(
            caffeine,
            highlight=[("1-3", "blue"), ("5-8", "red")],
            gradient=False,
            fog=False,
            orient=False,
        )
    )
    assert resolve_color("blue") in svg
    assert resolve_color("red") in svg


def test_multi_group_overlap_raises(caffeine):
    """Overlapping atoms between groups should raise ValueError."""
    with pytest.raises(ValueError, match="appear in multiple highlight groups"):
        render(caffeine, highlight=[([1, 2, 3], "blue"), ([3, 4, 5], "red")], orient=False)


def test_multi_group_cross_bonds_not_highlighted(caffeine):
    """Bonds between atoms in different groups keep default color."""
    # Atoms 3 and 4 are bonded in caffeine; put them in separate groups
    svg = str(
        render(
            caffeine,
            highlight=[([3], "blue"), ([4], "red")],
            gradient=False,
            fog=False,
            orient=False,
        )
    )
    blue_dark = Color.from_str(resolve_color("blue")).blend(Color(0, 0, 0), 0.3).hex
    red_dark = Color.from_str(resolve_color("red")).blend(Color(0, 0, 0), 0.3).hex
    # Neither group's darkened bond color should appear (bond between groups)
    assert blue_dark not in svg
    assert red_dark not in svg


def test_multi_group_same_group_bonds(caffeine):
    """Bonds between atoms in the same group get that group's darkened color."""
    svg = str(
        render(
            caffeine,
            highlight=[([1, 2, 3], "steelblue")],
            gradient=False,
            fog=False,
            orient=False,
        )
    )
    dark = Color.from_str(resolve_color("steelblue")).blend(Color(0, 0, 0), 0.3).hex
    assert dark in svg


# ---------------------------------------------------------------------------
# Flat atom color
# ---------------------------------------------------------------------------


def test_mol_color_overrides_cpk(caffeine):
    """mol_color paints all atoms and bonds a single color."""
    svg = str(render(caffeine, mol_color="gray", gradient=False, fog=False, orient=False))
    gray = resolve_color("gray")
    dark = Color.from_str(gray).blend(Color(0, 0, 0), 0.3).hex
    assert gray in svg
    assert dark in svg  # bonds get darkened mol_color


def test_mol_color_with_highlight(caffeine):
    """Highlight paints on top of mol_color."""
    svg = str(
        render(
            caffeine,
            mol_color="gray",
            highlight=[1, 2, 3],
            gradient=False,
            fog=False,
            orient=False,
        )
    )
    # Both gray (base) and orchid (highlight) should appear
    assert resolve_color("gray") in svg
    assert resolve_color("orchid") in svg


# ---------------------------------------------------------------------------
# Atom halos
# ---------------------------------------------------------------------------


def test_halo_circles_in_svg(caffeine):
    """Halo circles (fill-opacity, stroke=none) appear for each halo atom."""
    svg = str(render(caffeine, halo=[1, 2, 3], gradient=False, fog=False, orient=False))
    halos = [ln for ln in svg.splitlines() if "fill-opacity" in ln and 'stroke="none"' in ln]
    assert len(halos) == 3


def test_halo_default_color(caffeine):
    """Default halo color is palette[0] (orchid)."""
    svg = str(render(caffeine, halo="1-3", gradient=False, fog=False, orient=False))
    assert resolve_color("orchid") in svg


def test_halo_custom_color(caffeine):
    """Explicit halo color via tuple form."""
    svg = str(render(caffeine, halo=[([1, 2, 3], "hotpink")], gradient=False, fog=False, orient=False))
    assert resolve_color("hotpink") in svg


def test_halo_blur_filter_in_svg(caffeine):
    """SVG contains halo blur filter def when blur is enabled (default)."""
    svg = str(render(caffeine, halo=[1], orient=False))
    assert "halo_blur" in svg


def test_halo_no_blur_omits_filter(caffeine):
    """No blur filter def when halo_blur=False."""
    svg = str(render(caffeine, halo=[1], halo_blur=False, orient=False))
    assert "halo_blur" not in svg
    # Halo circle still present but without filter attribute
    halos = [ln for ln in svg.splitlines() if "fill-opacity" in ln and 'stroke="none"' in ln]
    assert len(halos) == 1
    assert "filter=" not in halos[0]


def test_halo_opacity_respected(caffeine):
    """Custom opacity value appears in halo circle SVG."""
    svg = str(render(caffeine, halo="1", halo_opacity=0.77, halo_blur=False, orient=False))
    halos = [ln for ln in svg.splitlines() if "fill-opacity" in ln and 'stroke="none"' in ln]
    assert len(halos) == 1
    assert 'fill-opacity="0.77"' in halos[0]


def test_halo_scale_affects_radius(caffeine):
    """Larger halo_scale produces a larger radius circle than smaller scale."""
    svg_small = str(render(caffeine, halo="1", halo_scale=1.5, halo_blur=False, orient=False))
    svg_large = str(render(caffeine, halo="1", halo_scale=3.0, halo_blur=False, orient=False))
    import re

    def _halo_r(svg: str) -> float:
        for ln in svg.splitlines():
            if "fill-opacity" in ln and 'stroke="none"' in ln:
                m = re.search(r'r="([\d.]+)"', ln)
                if m:
                    return float(m.group(1))
        return 0.0

    assert _halo_r(svg_large) > _halo_r(svg_small)


def test_halo_multi_group(caffeine):
    """Two halo groups with different colors both appear."""
    svg = str(
        render(
            caffeine,
            halo=[("1-3", "blue"), ("5-8", "red")],
            gradient=False,
            fog=False,
            orient=False,
        )
    )
    assert resolve_color("blue") in svg
    assert resolve_color("red") in svg
    halos = [ln for ln in svg.splitlines() if "fill-opacity" in ln and 'stroke="none"' in ln]
    assert len(halos) == 7  # 3 + 4 atoms


def test_halo_independent_of_highlight(caffeine):
    """Halo and highlight can be combined; atom keeps highlight color, halo is separate circle."""
    svg = str(
        render(
            caffeine,
            highlight=[1, 2, 3],
            halo=[1, 2, 3],
            gradient=False,
            fog=False,
            orient=False,
        )
    )
    # Highlight changes atom fill color; halo adds a separate translucent circle
    halos = [ln for ln in svg.splitlines() if "fill-opacity" in ln and 'stroke="none"' in ln]
    assert len(halos) == 3
    assert resolve_color("orchid") in svg  # highlight color on atom


def test_halo_ligand_selector_warns_without_semantics(caffeine, caplog):
    """'ligand' halo selector should warn and skip when semantics are unavailable."""
    with caplog.at_level("WARNING", logger="xyzrender.api"):
        svg = str(render(caffeine, halo="ligand", orient=False))
    halos = [ln for ln in svg.splitlines() if "fill-opacity" in ln and 'stroke="none"' in ln]
    assert len(halos) == 0
    assert "selector 'ligand' requested but no ligand semantics are available" in caplog.text
