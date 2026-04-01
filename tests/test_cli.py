"""Tests for CLI helpers."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from xyzrender.cli import _basename, _parse_pairs

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


def test_basename_from_xyz():
    assert _basename("molecule.xyz", from_stdin=False) == "molecule"


def test_basename_from_path():
    assert _basename("/path/to/caffeine.xyz", from_stdin=False) == "caffeine"


def test_basename_from_out_file():
    assert _basename("calc.out", from_stdin=False) == "calc"


def test_basename_stdin():
    assert _basename(None, from_stdin=True) == "graphic"


def test_basename_stdin_overrides_input():
    assert _basename("molecule.xyz", from_stdin=True) == "graphic"


def test_basename_none_not_stdin():
    assert _basename(None, from_stdin=False) == "graphic"


# ---------------------------------------------------------------------------
# _parse_pairs
# ---------------------------------------------------------------------------


def test_parse_pairs_single():
    assert _parse_pairs("1-6") == [(0, 5)]


def test_parse_pairs_multiple():
    assert _parse_pairs("1-6,3-4") == [(0, 5), (2, 3)]


def test_parse_pairs_empty():
    assert _parse_pairs("") == []
    assert _parse_pairs("   ") == []


def _dummy_molecule():
    from xyzrender.api import Molecule

    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))
    return Molecule(graph=g)


def _dummy_cell_molecule():
    from xyzrender.api import Molecule
    from xyzrender.types import CellData

    g = nx.Graph()
    g.add_node(0, symbol="C", position=(0.0, 0.0, 0.0))
    return Molecule(graph=g, cell_data=CellData(lattice=np.eye(3), cell_origin=np.zeros(3)))


def test_cli_protein_without_style_defaults_to_gloss(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--protein"])

    cli.main()
    assert captured["protein"] == "gloss"


def test_cli_protein_without_ghost_flag_defaults_ghosts_off(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_cell_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--protein"])

    cli.main()
    assert captured["ghosts"] is False


def test_cli_protein_with_ghosts_flag_overrides_default(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_cell_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--protein", "--ghosts"])

    cli.main()
    assert captured["ghosts"] is True


def test_cli_non_protein_cell_input_keeps_ghosts_default_on(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_cell_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb")])

    cli.main()
    assert captured["ghosts"] is True


def test_cli_protein_gloss_style(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--protein", "gloss"])

    cli.main()
    assert captured["protein"] == "gloss"


def test_cli_protein_illustration_style(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--protein", "illustration"])

    cli.main()
    assert captured["protein"] == "illustration"


def test_cli_protein_removed_style_fails(monkeypatch):
    import sys

    from xyzrender import cli
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--protein", "cartoon"])
    with pytest.raises(SystemExit):
        cli.main()


def test_cli_highlight_ligand_flag_wires_to_render(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--highlight-ligand"])

    cli.main()
    assert captured["ligand_highlight"] is True


def test_cli_ligand_highlight_flag_removed(monkeypatch):
    import sys

    from xyzrender import cli

    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--ligand-highlight"])
    with pytest.raises(SystemExit):
        cli.main()


def test_cli_halo_ligand_wires_to_render(monkeypatch):
    import sys

    from xyzrender import cli

    captured: dict = {}

    def _fake_render(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("xyzrender.api.render", _fake_render)
    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_molecule())
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--halo", "ligand"])

    cli.main()
    assert captured["halo"] == [("ligand",)]


def test_cli_nci_ligand_implies_nci_detect_on_load(monkeypatch):
    import sys

    from xyzrender import cli

    captured_load: dict = {}

    def _fake_load(*args, **kwargs):
        captured_load.update(kwargs)
        return _dummy_molecule()

    monkeypatch.setattr("xyzrender.api.load", _fake_load)
    monkeypatch.setattr("xyzrender.api.render", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["xyzrender", str(STRUCTURES / "water.pdb"), "--nci-ligand"])

    cli.main()
    assert captured_load["nci_detect"] is True
    assert captured_load["nci_ligand_protein_only"] is True


def test_cli_nci_ligand_implies_nci_detect_for_gif(monkeypatch, tmp_path):
    import sys

    from xyzrender import cli

    captured_gif: dict = {}

    def _fake_render_gif(*args, **kwargs):
        captured_gif.update(kwargs)
        return None

    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_molecule())
    monkeypatch.setattr("xyzrender.api.render", lambda *args, **kwargs: None)
    monkeypatch.setattr("xyzrender.api.render_gif", _fake_render_gif)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xyzrender",
            str(STRUCTURES / "water.pdb"),
            "--nci-ligand",
            "--gif-rot",
            "--gif-output",
            str(tmp_path / "out.gif"),
        ],
    )

    cli.main()
    assert captured_gif["detect_nci"] is True
    assert captured_gif["nci_ligand_protein_only"] is True


def test_cli_nci_ligand_protein_only_flag_removed(monkeypatch):
    import sys

    from xyzrender import cli

    monkeypatch.setattr(
        sys,
        "argv",
        ["xyzrender", str(STRUCTURES / "water.pdb"), "--nci-ligand-protein-only"],
    )
    with pytest.raises(SystemExit):
        cli.main()


def test_cli_halo_ligand_wires_to_render_gif(monkeypatch, tmp_path):
    import sys

    from xyzrender import cli

    captured_gif: dict = {}

    def _fake_render_gif(*args, **kwargs):
        captured_gif.update(kwargs)
        return None

    monkeypatch.setattr("xyzrender.api.load", lambda *args, **kwargs: _dummy_molecule())
    monkeypatch.setattr("xyzrender.api.render", lambda *args, **kwargs: None)
    monkeypatch.setattr("xyzrender.api.render_gif", _fake_render_gif)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xyzrender",
            str(STRUCTURES / "water.pdb"),
            "--halo",
            "ligand",
            "--gif-rot",
            "--gif-output",
            str(tmp_path / "out.gif"),
        ],
    )

    cli.main()
    assert captured_gif["halo"] == [("ligand",)]
