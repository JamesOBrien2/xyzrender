"""Tests for gif.py — rotation axis parsing and GIF rendering."""

from pathlib import Path

import numpy as np
import pytest

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


# ---------------------------------------------------------------------------
# _rotation_axis — unit tests (no I/O)
# ---------------------------------------------------------------------------


def test_rotation_axis_single():
    from xyzrender.gif import _rotation_axis

    ax, sign = _rotation_axis("x")
    assert np.allclose(ax, [1, 0, 0])
    assert sign == 1.0

    ax, sign = _rotation_axis("y")
    assert np.allclose(ax, [0, 1, 0])

    ax, sign = _rotation_axis("z")
    assert np.allclose(ax, [0, 0, 1])


def test_rotation_axis_negative():
    from xyzrender.gif import _rotation_axis

    ax, sign = _rotation_axis("-y")
    assert np.allclose(ax, [0, 1, 0])
    assert sign == -1.0


def test_rotation_axis_diagonal():
    from xyzrender.gif import _rotation_axis

    ax, _sign = _rotation_axis("xy")
    assert np.allclose(np.linalg.norm(ax), 1.0)
    assert ax[2] == pytest.approx(0.0)

    ax2, _ = _rotation_axis("yx")
    assert not np.allclose(ax, ax2)  # different diagonal


def test_rotation_axis_crystallographic():
    from xyzrender.gif import _rotation_axis

    lat = np.eye(3) * 5.0  # cubic lattice
    ax, _sign = _rotation_axis("111", lattice=lat)
    assert np.allclose(np.linalg.norm(ax), 1.0)
    assert np.allclose(ax, np.array([1, 1, 1]) / np.sqrt(3))


def test_rotation_axis_crystallographic_requires_lattice():
    from xyzrender.gif import _rotation_axis

    with pytest.raises(ValueError, match="lattice"):
        _rotation_axis("110")


# ---------------------------------------------------------------------------
# render_rotation_gif — integration (requires cairosvg)
# ---------------------------------------------------------------------------


def test_render_rotation_gif(tmp_path):
    pytest.importorskip("cairosvg", reason="cairosvg required")
    from xyzrender.gif import render_rotation_gif
    from xyzrender.readers import load_molecule
    from xyzrender.types import RenderConfig

    graph, _ = load_molecule(str(STRUCTURES / "caffeine.xyz"))
    cfg = RenderConfig(auto_orient=False)
    out = str(tmp_path / "rot.gif")
    render_rotation_gif(graph, cfg, out, n_frames=4, fps=5)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


# ---------------------------------------------------------------------------
# trj_bonds — per-frame bond detection
# ---------------------------------------------------------------------------


def test_orient_frames_preserves_extra_keys():
    """_orient_frames must keep graph / bond_opacities / hull_opacity_factor."""
    from xyzrender.gif import _orient_frames

    frame = {
        "symbols": ["C", "H"],
        "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "graph": object(),
        "bond_opacities": {(0, 1): 0.5},
        "hull_opacity_factor": 0.8,
    }
    out = _orient_frames([frame], np.eye(3))
    assert out[0]["graph"] is frame["graph"]
    assert out[0]["bond_opacities"] == {(0, 1): 0.5}
    assert out[0]["hull_opacity_factor"] == 0.8


def test_rotate_frames_preserves_extra_keys():
    """_rotate_frames must keep graph / bond_opacities / hull_opacity_factor."""
    from xyzrender.gif import _rotate_frames

    frame = {
        "symbols": ["C", "H"],
        "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "graph": object(),
        "bond_opacities": {(0, 1): 0.5},
        "hull_opacity_factor": 0.8,
    }
    out = _rotate_frames([frame], np.eye(3))
    assert out[0]["graph"] is frame["graph"]
    assert out[0]["bond_opacities"] == {(0, 1): 0.5}
    assert out[0]["hull_opacity_factor"] == 0.8


def test_render_trajectory_gif_trj_bonds(tmp_path):
    """trj_bonds=True must rebuild graphs per frame, and those graphs must
    survive the orient/rotate transforms and reach the worker."""
    from unittest.mock import patch

    from xyzrender.gif import render_trajectory_gif
    from xyzrender.readers import load_trajectory_frames
    from xyzrender.types import RenderConfig

    frames = load_trajectory_frames(STRUCTURES / "sn2.v000.xyz")
    captured: dict = {}

    def _spy(graph, frames, config, **_):
        captured["frames"] = frames
        return [b""] * len(frames)

    with (
        patch("xyzrender.gif._render_frames", side_effect=_spy),
        patch("xyzrender.gif._stitch_gif"),
    ):
        render_trajectory_gif(frames=frames, config=RenderConfig(), output=str(tmp_path / "x.gif"), trj_bonds=True)

    seen = captured["frames"]
    assert all("graph" in f for f in seen), "trj_bonds frames must each carry a 'graph'"
    bond_counts = [f["graph"].number_of_edges() for f in seen]
    # sn2.v000.xyz: bond forms across the SN2 reaction so counts must vary
    assert len(set(bond_counts)) > 1, f"expected varying per-frame bond counts, got {bond_counts}"
    # The graphs must be distinct objects (not all aliased to the last-frame graph)
    assert len({id(f["graph"]) for f in seen}) == len(seen)


# ---------------------------------------------------------------------------
# render_gif API — rotation GIF via public API
# ---------------------------------------------------------------------------


def test_api_render_gif_rotation(tmp_path):
    pytest.importorskip("cairosvg", reason="cairosvg required")
    from xyzrender import render_gif
    from xyzrender.api import GIFResult

    out = str(tmp_path / "caffeine.gif")
    result = render_gif(
        STRUCTURES / "caffeine.xyz",
        output=out,
        gif_rot="y",
        rot_frames=4,
        gif_fps=5,
        orient=False,
    )
    assert isinstance(result, GIFResult)
    assert result.path.exists()


def test_api_render_gif_bounce(tmp_path):
    pytest.importorskip("cairosvg", reason="cairosvg required")
    from xyzrender import render_gif
    from xyzrender.api import GIFResult

    out = str(tmp_path / "bounce.gif")
    result = render_gif(
        STRUCTURES / "caffeine.xyz",
        output=out,
        gif_bounce=30.0,
        rot_frames=4,
        gif_fps=5,
        orient=False,
    )
    assert isinstance(result, GIFResult)
    assert result.path.exists()
    assert result.path.stat().st_size > 0


def test_api_render_gif_bounce_axis_tuple(tmp_path):
    pytest.importorskip("cairosvg", reason="cairosvg required")
    from xyzrender import render_gif

    out = str(tmp_path / "bounce_x.gif")
    result = render_gif(
        STRUCTURES / "caffeine.xyz",
        output=out,
        gif_bounce=(30.0, "x"),
        rot_frames=4,
        gif_fps=5,
        orient=False,
    )
    assert result.path.exists()


def test_render_gif_bounce_invalid(tmp_path):
    from xyzrender import render_gif

    with pytest.raises(ValueError, match="gif_bounce must be > 0"):
        render_gif(
            STRUCTURES / "caffeine.xyz",
            output=str(tmp_path / "x.gif"),
            gif_bounce=0.0,
        )

    with pytest.raises(ValueError, match="gif_bounce must be > 0"):
        render_gif(
            STRUCTURES / "caffeine.xyz",
            output=str(tmp_path / "x.gif"),
            gif_bounce=-10.0,
        )


def test_render_gif_bounce_gif_rot_conflict(tmp_path):
    from xyzrender import render_gif

    with pytest.raises(ValueError, match="gif_bounce and gif_rot are mutually exclusive"):
        render_gif(
            STRUCTURES / "caffeine.xyz",
            output=str(tmp_path / "x.gif"),
            gif_bounce=(30.0, "x"),
            gif_rot="y",
        )


def test_render_gif_bounce_invalid_axis(tmp_path):
    from xyzrender import render_gif

    with pytest.raises(ValueError, match="invalid gif_bounce axis"):
        render_gif(
            STRUCTURES / "caffeine.xyz",
            output=str(tmp_path / "x.gif"),
            gif_bounce=(30.0, "qq"),
        )


def test_gifresult_save(tmp_path):
    pytest.importorskip("cairosvg", reason="cairosvg required")
    from xyzrender import render_gif

    src = str(tmp_path / "src.gif")
    result = render_gif(
        STRUCTURES / "caffeine.xyz",
        output=src,
        gif_rot="y",
        rot_frames=4,
        gif_fps=5,
        orient=False,
    )
    dest = tmp_path / "copy.gif"
    result.save(dest)
    assert dest.exists()
    assert dest.read_bytes() == Path(src).read_bytes()


# ---------------------------------------------------------------------------
# render_gif — gif_rot branch must forward CLI overrides to the renderer
# ---------------------------------------------------------------------------


def _tiny_molecule():
    import networkx as nx

    from xyzrender.api import Molecule

    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))
    g.add_node(1, symbol="H", position=(1.0, 0.0, 0.0))
    g.add_edge(0, 1, bond_order=1.0)
    return Molecule(graph=g)


def _capture_rotation_cfg():
    """Patch render_rotation_gif and return (context-manager, captured-dict)."""
    from unittest.mock import patch

    captured: dict = {}

    def _spy(graph=None, config=None, output=None, **_):
        captured["cfg"] = config

    return patch("xyzrender.gif.render_rotation_gif", side_effect=_spy), captured


def test_render_gif_rot_applies_vector_color(tmp_path):
    """--vector-color must reach the rotation renderer's config."""
    from xyzrender.api import render_gif
    from xyzrender.colors import resolve_color

    cm, captured = _capture_rotation_cfg()
    with cm:
        render_gif(_tiny_molecule(), gif_rot="y", vector_color="red", output=str(tmp_path / "x.gif"))
    assert captured["cfg"].vector_color == resolve_color("red")


def test_render_gif_rot_applies_surface_overrides(tmp_path):
    """--mo-pos-color / --mo-neg-color / --mo-upsample / --flat-mo / --dens-color
    must reach build_surface_params when gif_rot is the only mode."""
    from unittest.mock import patch

    from xyzrender.api import render_gif

    mol = _tiny_molecule()
    mol.cube_data = object()  # truthy; consumers are patched below

    captured: dict = {}

    def _capture(cfg, overrides, **_):
        captured["overrides"] = overrides
        return (None, None, None, None)

    cm_rot, _ = _capture_rotation_cfg()
    with (
        patch("xyzrender.config.build_surface_params", side_effect=_capture),
        cm_rot,
    ):
        render_gif(
            mol,
            gif_rot="y",
            mo=True,
            mo_pos_color="cyan",
            mo_neg_color="magenta",
            mo_upsample=2,
            flat_mo=True,
            dens_color="grey",
            output=str(tmp_path / "x.gif"),
        )
    o = captured["overrides"]
    assert o.mo_pos_color == "cyan"
    assert o.mo_neg_color == "magenta"
    assert o.mo_upsample == 2
    assert o.flat_mo is True
    assert o.dens_color == "grey"


def test_render_gif_rot_applies_overlay_config_without_overlay_kwarg(tmp_path):
    """`overlay_config=` must update cfg.overlay even when `overlay=` is not passed
    (regression for PR #126).

    Mirrors render()'s behaviour at api.py:1118 — `if overlay_config is not None: cfg.overlay = overlay_config`
    runs unconditionally. PR #126 nested the same block under `if overlay is not None:` in render_gif's
    rotation branch, silently dropping the kwarg for callers who override a preset's overlay block
    without re-passing the overlay structure itself.
    """
    from xyzrender.api import render_gif
    from xyzrender.types import OverlayConfig

    cm_rot, captured = _capture_rotation_cfg()
    with cm_rot:
        render_gif(
            _tiny_molecule(),
            gif_rot="y",
            overlay_config=OverlayConfig(color="#abcdef"),
            output=str(tmp_path / "x.gif"),
        )
    assert captured["cfg"].overlay.color == "#abcdef", (
        "overlay_config kwarg silently dropped — render_gif's overlay-block writes "
        "are nested under `if overlay is not None:` instead of running unconditionally "
        "(parity with render())"
    )


def test_render_gif_rot_applies_auto_align_without_overlay_kwarg(tmp_path):
    """`auto_align=False` must update cfg.auto_align even when `overlay=` is not passed
    (regression for PR #126). Same root cause as the overlay_config test above."""
    from xyzrender.api import render_gif
    from xyzrender.types import RenderConfig

    mol = _tiny_molecule()
    cfg = RenderConfig()
    cfg.auto_align = True  # default-true; verify False overrides

    cm_rot, captured = _capture_rotation_cfg()
    with cm_rot:
        render_gif(
            mol,
            config=cfg,
            gif_rot="y",
            auto_align=False,
            output=str(tmp_path / "x.gif"),
        )
    assert captured["cfg"].auto_align is False, (
        "auto_align=False kwarg silently dropped without overlay= — render_gif "
        "nests the write under `if overlay is not None:`"
    )


def test_render_gif_ts_does_not_double_load_molecule(tmp_path):
    """`render_gif(path, gif_ts=True)` must not call `load_molecule` an extra time
    just to populate `ref_graph` (regression for PR #126).

    PR #126 hoisted `ref_graph, _ = load_molecule(str(mol_path))` to run before the
    dispatch switch. The `gif_ts` branch passes `mol_path` straight to
    `render_vibration_gif` and never reads `ref_graph`, so the extra load is wasted
    work — non-trivial for files with cube data.

    Expected call count for `xyzrender.readers.load_molecule`:
      - 1: the initial `load(molecule)` at api.py:1445 (legitimate, needed for cfg setup)
      - 2 (BUG): the extra load at api.py:1593 (the regression — should be lazy)
    """
    from unittest.mock import patch

    from xyzrender.api import render_gif

    src = STRUCTURES / "caffeine.xyz"

    with (
        patch("xyzrender.gif.render_vibration_gif") as mock_vib,
        patch(
            "xyzrender.readers.load_molecule",
            wraps=__import__("xyzrender.readers", fromlist=["load_molecule"]).load_molecule,
        ) as mock_load,
    ):
        render_gif(src, gif_ts=True, output=str(tmp_path / "x.gif"))

    mock_vib.assert_called_once()
    assert mock_load.call_count == 1, (
        f"load_molecule called {mock_load.call_count} times for gif_ts — the second call "
        "(api.py:1593) is wasted because render_vibration_gif uses mol_path directly. "
        "Move the ref_graph load into the rotation/diffuse/trajectory branches that need it."
    )


# ---------------------------------------------------------------------------
# render_gif — additional cfg-kwarg propagation through the gif_rot branch.
# These pin down the parts of cfg most likely to silently break under future
# refactors (Bug-2-shaped regressions).
# ---------------------------------------------------------------------------


def test_render_gif_rot_applies_hull_color(tmp_path):
    """`hull=True, hull_color=X` must propagate through gif_rot — PR #126 routed all
    hull setup through `_apply_hull_pore_workflow`; verify the cfg the rotation
    renderer receives still carries the hull color."""
    from xyzrender.api import render_gif

    cm_rot, captured = _capture_rotation_cfg()
    with cm_rot:
        render_gif(
            _tiny_molecule(),
            gif_rot="y",
            hull=True,
            hull_color="#deadbe",
            output=str(tmp_path / "x.gif"),
        )
    assert captured["cfg"].show_convex_hull is True
    assert "#deadbe" in captured["cfg"].hull_colors


def test_render_gif_rot_applies_radius_scale(tmp_path):
    """`radius_scale=[...]` must reach cfg.radius_scale on the gif_rot branch."""
    from xyzrender.api import render_gif

    cm_rot, captured = _capture_rotation_cfg()
    with cm_rot:
        render_gif(
            _tiny_molecule(),
            gif_rot="y",
            radius_scale=[("H", 0.5)],
            output=str(tmp_path / "x.gif"),
        )
    assert captured["cfg"].radius_scale == [("H", 0.5)]


def test_render_gif_rot_applies_glow(tmp_path):
    """`glow=[1]` (1-indexed) must reach cfg.glow_indices as 0-indexed on the gif_rot branch."""
    from xyzrender.api import render_gif

    cm_rot, captured = _capture_rotation_cfg()
    with cm_rot:
        render_gif(
            _tiny_molecule(),
            gif_rot="y",
            glow=[1, 2],
            output=str(tmp_path / "x.gif"),
        )
    assert captured["cfg"].glow_indices == [0, 1]


def test_render_gif_rot_respects_auto_align_false(tmp_path):
    """An explicit auto_align=False must override a config-level True
    even on the gif_rot branch (overlay path)."""
    from unittest.mock import patch

    from xyzrender.api import render_gif
    from xyzrender.types import RenderConfig

    mol = _tiny_molecule()
    cfg = RenderConfig()
    cfg.auto_align = True

    cm_rot, captured = _capture_rotation_cfg()
    with patch("xyzrender.api._apply_overlay", return_value=mol), cm_rot:
        render_gif(
            mol,
            config=cfg,
            gif_rot="y",
            overlay=mol,
            auto_align=False,
            output=str(tmp_path / "x.gif"),
        )
    assert captured["cfg"].auto_align is False


def test_render_gif_rot_accepts_align_atoms_kwarg(tmp_path):
    """render_gif must accept `align_atoms=[…]` and propagate it to `_apply_overlay`
    (parity with render() — see api.py:1132)."""
    from unittest.mock import patch

    from xyzrender.api import render_gif

    captured_align: dict = {}

    def _spy_overlay(_base, _ov, _cfg, _ov_arg, **kwargs):
        captured_align["align_atoms"] = kwargs.get("align_atoms")
        return _base

    cm_rot, _ = _capture_rotation_cfg()
    mol = _tiny_molecule()
    with patch("xyzrender.api._apply_overlay", side_effect=_spy_overlay), cm_rot:
        render_gif(
            mol,
            gif_rot="y",
            overlay=mol,
            output=str(tmp_path / "x.gif"),
            align_atoms=[1, 2],
        )
    assert captured_align["align_atoms"] == [1, 2], (
        f"render_gif accepted align_atoms but didn't propagate to _apply_overlay: "
        f"got {captured_align.get('align_atoms')!r}"
    )


# ---------------------------------------------------------------------------
# gif_append — crossfade helpers (no I/O)
# ---------------------------------------------------------------------------


def _solid_png(color: tuple[int, int, int, int], size: int = 4) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGBA", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


def test_concat_with_crossfade_counts():
    pytest.importorskip("PIL")
    from xyzrender.gif import _concat_with_crossfade

    red, green = _solid_png((255, 0, 0, 255)), _solid_png((0, 255, 0, 255))
    segments = [[red] * 3, [green] * 3]
    out = _concat_with_crossfade(segments, 2)
    # 2 segments x 3 frames + 2 boundaries (incl. loop seam) x 2 fade frames
    assert len(out) == 6 + 4


def test_concat_with_crossfade_hard_cut():
    from xyzrender.gif import _concat_with_crossfade

    segments = [[b"a"] * 3, [b"b"] * 2]
    assert _concat_with_crossfade(segments, 0) == [b"a"] * 3 + [b"b"] * 2


def test_concat_with_crossfade_single_segment():
    from xyzrender.gif import _concat_with_crossfade

    # One segment — no boundaries, fade frames must not be inserted
    assert _concat_with_crossfade([[b"a"] * 3], 5) == [b"a"] * 3


def test_blend_pngs_midpoint():
    pytest.importorskip("PIL")
    from io import BytesIO

    from PIL import Image

    from xyzrender.gif import _blend_pngs

    black, white = _solid_png((0, 0, 0, 255)), _solid_png((255, 255, 255, 255))
    blended = _blend_pngs(black, white, 1)
    assert len(blended) == 1
    px = Image.open(BytesIO(blended[0])).convert("RGBA").getpixel((0, 0))
    assert abs(px[0] - 127) <= 2  # ~50% blend


# ---------------------------------------------------------------------------
# gif_append — rendering + API validation
# ---------------------------------------------------------------------------


def _write_conformers_xyz(path, n_frames: int = 3) -> None:
    """Tiny 3-frame water 'conformer' trajectory."""
    base = [("O", 0.0, 0.0, 0.0), ("H", 0.96, 0.0, 0.0), ("H", -0.24, 0.93, 0.0)]
    lines = []
    for k in range(n_frames):
        lines.append("3")
        lines.append(f"conf {k}")
        for s, x, y, z in base:
            lines.append(f"{s} {x + 0.01 * k:.4f} {y:.4f} {z:.4f}")
    Path(path).write_text("\n".join(lines) + "\n")


def test_render_append_gif_frame_count(tmp_path):
    pytest.importorskip("resvg_py", reason="PNG rendering backend required")
    from PIL import Image

    from xyzrender import render_gif

    xyz = tmp_path / "confs.xyz"
    _write_conformers_xyz(xyz)
    out = tmp_path / "append.gif"
    result = render_gif(xyz, output=str(out), gif_rot="y", gif_append=True, rot_frames=4, gif_fps=5, append_fade=1)
    assert result.path.exists()
    # 3 segments x 4 frames + 3 boundaries (incl. loop seam) x 1 fade frame
    assert Image.open(out).n_frames == 12 + 3


def test_render_append_gif_bounce(tmp_path):
    pytest.importorskip("resvg_py", reason="PNG rendering backend required")
    from PIL import Image

    from xyzrender import render_gif

    xyz = tmp_path / "confs.xyz"
    _write_conformers_xyz(xyz, n_frames=2)
    out = tmp_path / "append_bounce.gif"
    render_gif(xyz, output=str(out), gif_bounce=20.0, gif_append=True, rot_frames=4, gif_fps=5, append_fade=0)
    assert Image.open(out).n_frames == 8  # hard cut: 2 x 4


def test_render_append_gif_mixed_atom_counts(tmp_path):
    pytest.importorskip("resvg_py", reason="PNG rendering backend required")
    from PIL import Image

    from xyzrender import render_gif

    xyz = tmp_path / "mixed.xyz"
    xyz.write_text(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n2\nhydrogen\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n"
    )
    out = tmp_path / "mixed.gif"
    render_gif(xyz, output=str(out), gif_rot="y", gif_append=True, rot_frames=4, gif_fps=5, append_fade=0)
    assert Image.open(out).n_frames == 8


def test_render_gif_append_requires_mode(tmp_path):
    from xyzrender import render_gif

    with pytest.raises(ValueError, match="gif_append requires"):
        render_gif(STRUCTURES / "caffeine.xyz", output=str(tmp_path / "x.gif"), gif_append=True)


def test_render_gif_append_trj_conflict(tmp_path):
    from xyzrender import render_gif

    with pytest.raises(ValueError, match="mutually exclusive with gif_ts / gif_trj"):
        render_gif(
            STRUCTURES / "caffeine.xyz",
            output=str(tmp_path / "x.gif"),
            gif_rot="y",
            gif_append=True,
            gif_trj=True,
        )


def test_render_gif_append_molecule_input_error(tmp_path):
    from xyzrender import load, render_gif

    mol = load(STRUCTURES / "caffeine.xyz")
    with pytest.raises(ValueError, match="pass a file path"):
        render_gif(mol, output=str(tmp_path / "x.gif"), gif_rot="y", gif_append=True)


def test_render_gif_append_negative_fade(tmp_path):
    from xyzrender import render_gif

    with pytest.raises(ValueError, match="append_fade must be >= 0"):
        render_gif(
            STRUCTURES / "caffeine.xyz",
            output=str(tmp_path / "x.gif"),
            gif_rot="y",
            gif_append=True,
            append_fade=-1,
        )


def test_render_gif_append_fade_requires_append(tmp_path):
    from xyzrender import render_gif

    with pytest.raises(ValueError, match="append_fade requires gif_append"):
        render_gif(
            STRUCTURES / "caffeine.xyz",
            output=str(tmp_path / "x.gif"),
            gif_rot="y",
            append_fade=2,
        )


def test_load_variable_xyz_frames(tmp_path):
    from xyzrender.readers import load_trajectory_frames

    xyz = tmp_path / "var.xyz"
    xyz.write_text(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n2\nhydrogen\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n"
    )
    frames = load_trajectory_frames(str(xyz))
    assert len(frames) == 2
    assert frames[0]["symbols"] == ["O", "H", "H"]
    assert frames[1]["symbols"] == ["H", "H"]
    assert frames[1]["positions"][1] == [0.74, 0.0, 0.0]


# ---------------------------------------------------------------------------
# gif_append — multi-conformer ensemble (RDKit butan-1-ol, 6 frames x 15 atoms)
# ---------------------------------------------------------------------------

BUTANOL_CONFS = STRUCTURES / "butanol_confs.xyz"


def test_load_butanol_ensemble_frames():
    from xyzrender.readers import load_trajectory_frames

    frames = load_trajectory_frames(str(BUTANOL_CONFS))
    assert len(frames) == 6
    assert all(len(f["symbols"]) == 15 for f in frames)
    assert frames[0]["symbols"].count("O") == 1


def test_render_append_gif_butanol_ensemble(tmp_path):
    pytest.importorskip("resvg_py", reason="PNG rendering backend required")
    from PIL import Image

    from xyzrender import render_gif

    # First 4 conformers (17 lines per frame: count + comment + 15 atoms)
    lines = BUTANOL_CONFS.read_text().splitlines()
    subset = tmp_path / "subset.xyz"
    subset.write_text("\n".join(lines[: 4 * 17]) + "\n")

    out = tmp_path / "butanol.gif"
    result = render_gif(subset, output=str(out), gif_rot="y", gif_append=True, rot_frames=3, gif_fps=5, append_fade=1)
    assert result.path.exists()
    # 4 segments x 3 frames + 4 boundaries (incl. loop seam) x 1 fade frame
    assert Image.open(out).n_frames == 12 + 4
