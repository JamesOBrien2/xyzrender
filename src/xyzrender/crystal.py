"""Crystal structure support (optional — requires xyzrender[crystal] / phonopy).

This module contains all phonopy-dependent functionality for loading periodic
crystal structures and generating periodic image atoms for rendering.  It is
intentionally separated from ``io.py`` so that the optional ``phonopy``
dependency is not imported at all unless crystal loading is actually requested.

Public API
----------
load_crystal
    Load a VASP/QE/... crystal structure file and return a molecular graph
    together with its ``CellData`` (lattice matrix + cell origin).
add_crystal_images
    Populate a crystal graph with ghost atoms from the 26 neighbouring unit
    cells so that bonds crossing cell boundaries are visible.
"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import numpy as np
from xyzgraph import DATA, build_graph
from xyzgraph.parameters import BondThresholds

from xyzrender.types import CellData

_bond_thresholds = BondThresholds()

if TYPE_CHECKING:
    from pathlib import Path

    import networkx as nx

logger = logging.getLogger(__name__)


def _is_bonded(sym_i: str, sym_j: str, dist: float) -> bool:
    """Return True if two atoms at *dist* Å apart are likely bonded.

    Uses xyzgraph's VDW radii (DATA.vdw) and the same type-specific distance
    thresholds as xyzgraph's BondThresholds defaults, so ghost-bond detection
    is consistent with main-cell bond detection.  Note: xyzgraph also applies
    geometric pruning (bond angles, valence) which is not replicated here.
    """
    ri = DATA.vdw.get(sym_i, 2.0)
    rj = DATA.vdw.get(sym_j, 2.0)
    metals = DATA.metals
    hi, hj = sym_i == "H", sym_j == "H"
    mi, mj = sym_i in metals, sym_j in metals
    if hi and hj:
        t = _bond_thresholds.threshold_h_h
    elif hi or hj:
        t = _bond_thresholds.threshold_h_metal if (mi or mj) else _bond_thresholds.threshold_h_nonmetal
    elif mi and mj:
        t = _bond_thresholds.threshold_metal_metal_self
    elif mi or mj:
        t = _bond_thresholds.threshold_metal_ligand
    else:
        t = _bond_thresholds.threshold_nonmetal_nonmetal
    return dist < t * (ri + rj)


def load_crystal(
    path: str | Path,
    interface_mode: str,
) -> tuple[nx.Graph, CellData]:
    """Load a periodic crystal structure using phonopy.

    Parameters
    ----------
    path:
        Path to the crystal structure input file (POSCAR/CONTCAR for VASP,
        ``*.in`` / ``pw.in`` for Quantum ESPRESSO, etc.).
    interface_mode:
        Phonopy interface identifier: ``"vasp"``, ``"qe"``, ``"abinit"``, etc.

    Returns
    -------
    tuple[nx.Graph, CellData]
        Molecular graph with atoms as nodes and ``CellData`` containing the
        3x3 lattice matrix (rows = a, b, c in Å).
    """
    logger.info("Loading %s", path)
    try:
        from phonopy.interface.calculator import get_calculator_physical_units, read_crystal_structure
    except ImportError:
        msg = "Crystal structure loading requires phonopy: pip install 'xyzrender[crystal]'"
        raise ImportError(msg) from None

    unitcell, _ = read_crystal_structure(str(path), interface_mode=interface_mode)
    if unitcell is None:
        msg = f"Failed to read crystal structure from {path!r} (interface_mode={interface_mode!r})"
        raise ValueError(msg)
    # Convert native units → Angstrom.
    factor: float = get_calculator_physical_units(interface_mode).distance_to_A
    symbols: list[str] = list(unitcell.symbols)
    positions = unitcell.positions * factor  # ndarray, shape (N, 3), in Å
    lattice = np.array(unitcell.cell) * factor  # shape (3, 3), rows = a, b, c in Å

    atoms: list[tuple[str, tuple[float, float, float]]] = [
        (sym, (float(pos[0]), float(pos[1]), float(pos[2]))) for sym, pos in zip(symbols, positions, strict=True)
    ]
    graph = build_graph(atoms, charge=0, multiplicity=None, kekule=False, quick=True)
    logger.info(
        "Crystal graph: %d atoms, %d bonds, lattice=%s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        lattice.diagonal().round(3),
    )
    graph.graph["lattice"] = lattice
    graph.graph["lattice_origin"] = np.zeros(3)
    return graph, CellData(lattice=lattice)


def add_crystal_images(graph: nx.Graph, crystal_data: CellData) -> int:
    """Add periodic image atoms that are bonded to cell atoms.

    For each of the 26 neighbouring unit cells, adds image copies of cell
    atoms that form at least one bond with an atom inside the cell.  Image
    nodes carry ``image=True`` and ``source=<cell_atom_id>`` attributes;
    image bonds carry ``image_bond=True``.

    Returns the number of image atoms added.
    """
    lattice = crystal_data.lattice  # (3, 3)
    a, b, c = lattice[0], lattice[1], lattice[2]

    cell_ids = list(graph.nodes())
    if not cell_ids:
        return 0

    cell_syms = {i: graph.nodes[i]["symbol"] for i in cell_ids}
    cell_pos = {i: np.array(graph.nodes[i]["position"]) for i in cell_ids}

    next_id = max(cell_ids) + 1
    n_added = 0

    shifts = [(dx, dy, dz) for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3) if (dx, dy, dz) != (0, 0, 0)]

    for dx, dy, dz in shifts:
        offset = dx * a + dy * b + dz * c
        for src_id in cell_ids:
            sym_i = cell_syms[src_id]
            img_pos = cell_pos[src_id] + offset

            bonded_to: list[int] = [
                j for j in cell_ids if _is_bonded(sym_i, cell_syms[j], float(np.linalg.norm(img_pos - cell_pos[j])))
            ]

            if not bonded_to:
                continue

            img_id = next_id
            next_id += 1
            n_added += 1
            graph.add_node(
                img_id,
                symbol=sym_i,
                position=(float(img_pos[0]), float(img_pos[1]), float(img_pos[2])),
                image=True,
                source=src_id,
            )
            for j in bonded_to:
                graph.add_edge(img_id, j, bond_order=1.0, image_bond=True)

    logger.debug("Added %d image atoms", n_added)
    return n_added


def expand_supercell(
    graph: nx.Graph,
    crystal_data: CellData,
    supercell: int | tuple[int, int, int],
) -> int:
    """Add all atoms from neighbouring unit cells to form an expanded supercell view.

    Unlike :func:`add_crystal_images` (which adds only atoms bonded to the central
    cell), this function populates the graph with **every** atom from each expanded
    cell so that complete molecules are visible across cell boundaries.

    Parameters
    ----------
    supercell:
        Number of extra cells to show in each direction.  An integer *N* is
        shorthand for ``(N, N, N)``.  A 3-tuple ``(na, nb, nc)`` sets per-axis
        expansion independently: the cell is expanded ±na along **a**, ±nb along
        **b**, and ±nc along **c**.  E.g. ``supercell=1`` → 3×3×3 = 27 cells;
        ``supercell=(1, 1, 0)`` → 3×3×1 = 9 cells.

    Returns
    -------
    int
        Number of image atoms added to the graph.

    Notes
    -----
    Added nodes carry ``image=True``, ``source=<original_cell_atom_id>``, and
    ``shift=(dx, dy, dz)`` (fractional offsets).  All added edges carry
    ``image_bond=True``.  Intra-cell bonds are replicated for each copy of the
    cell.  Cross-cell bonds between adjacent expanded cells are detected with the
    same :func:`_is_bonded` logic used by :func:`add_crystal_images`.
    """
    # Normalise supercell parameter
    if isinstance(supercell, int):
        na, nb, nc = supercell, supercell, supercell
    else:
        na, nb, nc = int(supercell[0]), int(supercell[1]), int(supercell[2])

    if any(n < 0 for n in (na, nb, nc)):
        msg = f"supercell values must be non-negative integers, got ({na}, {nb}, {nc})"
        raise ValueError(msg)

    if na == 0 and nb == 0 and nc == 0:
        return 0

    lattice = crystal_data.lattice  # (3, 3)
    a, b, c = lattice[0], lattice[1], lattice[2]

    cell_ids = list(graph.nodes())
    if not cell_ids:
        return 0

    cell_syms = {i: graph.nodes[i]["symbol"] for i in cell_ids}
    cell_pos = {i: np.array(graph.nodes[i]["position"]) for i in cell_ids}
    cell_edges = list(graph.edges(data=True))

    next_id = max(cell_ids) + 1
    n_added = 0

    # Map from shift → {original_id: new_node_id}
    # Central cell (0,0,0) maps to itself.
    shift_to_nodemap: dict[tuple[int, int, int], dict[int, int]] = {
        (0, 0, 0): {i: i for i in cell_ids},
    }

    all_shifts = [
        (dx, dy, dz)
        for dx in range(-na, na + 1)
        for dy in range(-nb, nb + 1)
        for dz in range(-nc, nc + 1)
    ]
    image_shifts = [s for s in all_shifts if s != (0, 0, 0)]

    # --- Step 1: add all image atoms and replicate intra-cell bonds ---
    for shift in image_shifts:
        dx, dy, dz = shift
        offset = dx * a + dy * b + dz * c
        nodemap: dict[int, int] = {}

        for src_id in cell_ids:
            img_pos = cell_pos[src_id] + offset
            img_id = next_id
            next_id += 1
            n_added += 1
            graph.add_node(
                img_id,
                symbol=cell_syms[src_id],
                position=(float(img_pos[0]), float(img_pos[1]), float(img_pos[2])),
                image=True,
                source=src_id,
                shift=shift,
            )
            nodemap[src_id] = img_id

        shift_to_nodemap[shift] = nodemap

        # Replicate all intra-cell bonds within this shifted copy
        for ai, aj, edge_data in cell_edges:
            new_ai = nodemap[ai]
            new_aj = nodemap[aj]
            graph.add_edge(new_ai, new_aj, bond_order=edge_data.get("bond_order", 1.0), image_bond=True)

    # --- Step 2: cross-cell bonds between adjacent shifts ---
    # Build a position cache so we don't reconstruct numpy arrays in the inner loop.
    all_node_pos: dict[int, np.ndarray] = {
        nid: np.array(attrs["position"]) for nid, attrs in graph.nodes(data=True)
    }

    # Two shifts are "adjacent" when their Chebyshev distance is exactly 1
    # (i.e. they share a face, edge, or corner).  Iterate ordered pairs only.
    for idx_a, shift_a in enumerate(all_shifts):
        for shift_b in all_shifts[idx_a + 1 :]:
            if max(abs(shift_a[k] - shift_b[k]) for k in range(3)) != 1:
                # Cells more than one step apart cannot share bonds.
                continue

            nodemap_a = shift_to_nodemap[shift_a]
            nodemap_b = shift_to_nodemap[shift_b]

            for src_a, node_a in nodemap_a.items():
                pos_a = all_node_pos[node_a]
                sym_a = cell_syms[src_a]

                for src_b, node_b in nodemap_b.items():
                    if graph.has_edge(node_a, node_b):
                        continue
                    dist = float(np.linalg.norm(pos_a - all_node_pos[node_b]))
                    if _is_bonded(sym_a, cell_syms[src_b], dist):
                        graph.add_edge(node_a, node_b, bond_order=1.0, image_bond=True)

    logger.debug("expand_supercell: added %d image atoms (supercell=%s)", n_added, (na, nb, nc))
    return n_added
