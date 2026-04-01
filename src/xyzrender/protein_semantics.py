"""Protein semantics extraction for graph-first ribbon rendering.

Builds a unified semantics object from parser metadata (PDB/MOL2/extXYZ) or
graph-only heuristics, with explicit confidence tiers for rendering decisions.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from xyzrender.types import ChainData, ProteinConfidence, ProteinSemantics, ResidueData

if TYPE_CHECKING:
    import networkx as nx

    from xyzrender.parsers import MolData
    from xyzrender.types import ProteinData

logger = logging.getLogger(__name__)

_WATER_RESNAMES: frozenset[str] = frozenset({"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"})
_ION_RESNAMES: frozenset[str] = frozenset(
    {"NA", "K", "CA", "MG", "ZN", "CL", "FE", "CU", "MN", "CO", "NI", "SO4", "PO4"}
)


def from_protein_data(data: "ProteinData", *, provenance: str = "pdb") -> ProteinSemantics:
    """Promote legacy :class:`ProteinData` into :class:`ProteinSemantics`."""
    return ProteinSemantics(
        chains=data.chains,
        hetatm_indices=set(data.hetatm_indices),
        backbone_indices=set(data.backbone_indices),
        sidechain_indices=set(data.sidechain_indices),
        helix_spans=list(data.helix_spans),
        sheet_spans=list(data.sheet_spans),
        ligand_indices=set(data.ligand_indices),
        water_indices=set(data.water_indices),
        ion_indices=set(data.ion_indices),
        confidence_tier=ProteinConfidence.FULL_RIBBON,
        confidence_reasons=[f"semantic metadata extracted from {provenance}"],
        provenance=[provenance],
        trace_chains={},
    )


def _map_confidence_tier(value: Any) -> ProteinConfidence:
    raw = value.value if hasattr(value, "value") else str(value)
    raw = str(raw).lower().strip()
    if raw == ProteinConfidence.FULL_RIBBON.value:
        return ProteinConfidence.FULL_RIBBON
    if raw == ProteinConfidence.TRACE_ONLY.value:
        return ProteinConfidence.TRACE_ONLY
    return ProteinConfidence.INSUFFICIENT


def _to_xyzrender_semantics(xg_sem: Any) -> ProteinSemantics:
    """Convert xyzgraph protein semantics payload/dataclasses into xyzrender types."""
    chains_in = getattr(xg_sem, "chains", None)
    if chains_in is None and isinstance(xg_sem, dict):
        chains_in = xg_sem.get("chains", {})
    chains: dict[str, ChainData] = {}

    if isinstance(chains_in, dict):
        for cid, chain_obj in chains_in.items():
            residues_obj = getattr(chain_obj, "residues", None)
            if residues_obj is None and isinstance(chain_obj, dict):
                residues_obj = chain_obj.get("residues", [])
            residues: list[ResidueData] = []
            if residues_obj is None:
                residues_obj = []
            for r in residues_obj:
                get = r.get if isinstance(r, dict) else lambda k, default=None, _r=r: getattr(_r, k, default)
                residues.append(
                    ResidueData(
                        res_name=str(get("res_name", "UNK")),
                        res_seq=int(get("res_seq", 0)),
                        chain_id=str(get("chain_id", cid)),
                        atom_indices=[int(i) for i in get("atom_indices", [])],
                        ca_index=(None if get("ca_index", None) is None else int(get("ca_index"))),
                        c_index=(None if get("c_index", None) is None else int(get("c_index"))),
                        o_index=(None if get("o_index", None) is None else int(get("o_index"))),
                        n_index=(None if get("n_index", None) is None else int(get("n_index"))),
                        ss_type=str(get("ss_type", "C")),
                    )
                )
            chain_id = str(
                getattr(chain_obj, "chain_id", None)
                or (chain_obj.get("chain_id") if isinstance(chain_obj, dict) else cid)
                or cid
            )
            chains[str(cid)] = ChainData(chain_id=chain_id, residues=residues)

    get_top = xg_sem.get if isinstance(xg_sem, dict) else lambda k, default=None: getattr(xg_sem, k, default)
    return ProteinSemantics(
        chains=chains,
        hetatm_indices={int(i) for i in get_top("hetatm_indices", set())},
        backbone_indices={int(i) for i in get_top("backbone_indices", set())},
        sidechain_indices={int(i) for i in get_top("sidechain_indices", set())},
        helix_spans=[(str(c), int(s), int(e)) for c, s, e in get_top("helix_spans", [])],
        sheet_spans=[(str(c), int(s), int(e)) for c, s, e in get_top("sheet_spans", [])],
        ligand_indices={int(i) for i in get_top("ligand_indices", set())},
        water_indices={int(i) for i in get_top("water_indices", set())},
        ion_indices={int(i) for i in get_top("ion_indices", set())},
        confidence_tier=_map_confidence_tier(get_top("confidence_tier", ProteinConfidence.INSUFFICIENT.value)),
        confidence_reasons=[str(r) for r in get_top("confidence_reasons", [])],
        provenance=[str(p) for p in get_top("provenance", [])],
        trace_chains={str(cid): [int(i) for i in idxs] for cid, idxs in dict(get_top("trace_chains", {})).items()},
    )


def _parse_extxyz_annotation_rows(path: Path) -> list[dict[str, object]] | None:
    """Parse canonical protein annotation keys from extXYZ Properties, if present."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        return None
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        return None
    props = _parse_extxyz_properties(lines[1])
    if props is None:
        return None
    required = ("atom_name", "res_name", "res_seq", "chain_id", "ss_type")
    if any(k not in props for k in required):
        return None

    rows: list[dict[str, object]] = []
    for ln in lines[2 : 2 + n_atoms]:
        parts = ln.split()
        if len(parts) < 4:
            return None

        def _get(name: str, _parts: list[str] = parts) -> str:
            offset, count = props[name]
            if offset >= len(_parts):
                return ""
            return " ".join(_parts[offset : offset + count]).strip()

        res_seq_raw = _get("res_seq")
        try:
            res_seq = int(res_seq_raw)
        except ValueError:
            return None
        rows.append(
            {
                "record_type": "ATOM",
                "atom_name": _get("atom_name"),
                "res_name": _get("res_name"),
                "res_seq": res_seq,
                "chain_id": _get("chain_id"),
                "ss_type": _get("ss_type"),
            }
        )
    return rows or None


def _parse_mol2_annotation_rows(path: Path) -> list[dict[str, object]] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    in_atom = False
    rows: list[dict[str, object]] = []
    for ln in lines:
        s = ln.strip()
        up = s.upper()
        if up.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if up.startswith("@<TRIPOS>") and in_atom:
            break
        if not in_atom or not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 6:
            continue
        atom_name = parts[1]
        if len(parts) > 6:
            try:
                subst_id = int(parts[6])
            except ValueError:
                subst_id = 0
        else:
            subst_id = 0
        subst_name = parts[7] if len(parts) > 7 else "RES"
        m_chain = re.search(r"[:_]([A-Za-z0-9])$", subst_name)
        chain_id = m_chain.group(1) if m_chain else "A"
        m_res = re.match(r"([A-Za-z]{3})", subst_name)
        res_name = m_res.group(1).upper() if m_res else "RES"
        m_seq = re.search(r"(-?\d+)", subst_name)
        res_seq = int(m_seq.group(1)) if m_seq else subst_id
        rows.append(
            {
                "record_type": "ATOM",
                "atom_name": atom_name,
                "res_name": res_name,
                "res_seq": res_seq,
                "chain_id": chain_id,
                "ss_type": "C",
            }
        )
    return rows or None


def _canonical_annotations(
    *,
    moldata: "MolData | None",
    source_path: str | Path | None,
    format_hint: str | None,
) -> list[dict[str, object]] | None:
    if moldata is not None:
        rows = getattr(moldata, "atom_annotations", None)
        if rows:
            return rows
    path = Path(source_path) if source_path is not None else None
    fmt = (format_hint or (path.suffix.lower() if path is not None else "")).lower()
    if path is not None and fmt == ".mol2":
        return _parse_mol2_annotation_rows(path)
    if path is not None and fmt == ".xyz":
        return _parse_extxyz_annotation_rows(path)
    return None


def _extract_from_xyzgraph(
    graph: "nx.Graph",
    *,
    moldata: "MolData | None",
    source_path: str | Path | None,
    protein_requested: bool,
    format_hint: str | None,
) -> ProteinSemantics | None:
    try:
        from xyzgraph.protein import annotate_protein_semantics, protein_semantics_from_dict
    except Exception:
        return None

    annotations = _canonical_annotations(moldata=moldata, source_path=source_path, format_hint=format_hint)
    report = annotate_protein_semantics(
        graph,
        atom_annotations=annotations,
        format_hint=format_hint,
        protein_requested=protein_requested,
    )
    if report is None:
        return None

    payload = graph.graph.get("protein_semantics")
    if isinstance(payload, dict):
        return _to_xyzrender_semantics(protein_semantics_from_dict(payload))
    return _to_xyzrender_semantics(report.semantics)


def _residue_key(chain_id: str, res_seq: int, res_name: str) -> tuple[str, int, str]:
    return chain_id, int(res_seq), res_name


def _build_semantics_from_atom_meta(
    atom_meta: list[dict[str, object]],
    *,
    provenance: str,
    allow_ss: bool = True,
) -> ProteinSemantics | None:
    """Build semantics from normalized per-atom metadata rows."""
    if not atom_meta:
        return None

    residues: dict[tuple[str, int, str], list[int]] = {}
    residue_order: list[tuple[str, int, str]] = []
    chains: dict[str, list[ResidueData]] = {}
    hetatm_indices: set[int] = set()
    ligand_indices: set[int] = set()
    water_indices: set[int] = set()
    ion_indices: set[int] = set()
    backbone_indices: set[int] = set()
    sidechain_indices: set[int] = set()
    atom_name_by_idx: dict[int, str] = {}
    ss_by_residue: dict[tuple[str, int, str], str] = {}
    residue_atom_names: dict[tuple[str, int, str], set[str]] = {}

    for idx, row in enumerate(atom_meta):
        rec = str(row.get("record_type", "ATOM")).upper()
        atom_name = str(row.get("atom_name", "")).strip()
        res_name = str(row.get("res_name", "RES")).strip() or "RES"
        chain_id = str(row.get("chain_id", "A")).strip() or "A"
        res_seq = int(row.get("res_seq", 0) or 0)
        ss_type = str(row.get("ss_type", "C")).strip().upper() if allow_ss else "C"
        if ss_type not in {"H", "E", "C"}:
            ss_type = "C"
        atom_name_by_idx[idx] = atom_name

        if rec == "HETATM":
            hetatm_indices.add(idx)
            if res_name in _WATER_RESNAMES:
                water_indices.add(idx)
            elif res_name in _ION_RESNAMES:
                ion_indices.add(idx)
            else:
                ligand_indices.add(idx)
            continue

        key = _residue_key(chain_id, res_seq, res_name)
        if key not in residues:
            residues[key] = []
            residue_order.append(key)
        residues[key].append(idx)
        residue_atom_names.setdefault(key, set()).add(atom_name.upper())
        ss_by_residue[key] = ss_type

    for chain_id, res_seq, res_name in residue_order:
        idxs = residues[(chain_id, res_seq, res_name)]
        atom_names = residue_atom_names.get((chain_id, res_seq, res_name), set())
        # MOL2/extXYZ metadata often marks all atoms as "ATOM". Treat residues
        # as protein only when they look peptide-like (CA with N/C backbone).
        is_protein_like = "CA" in atom_names and ("N" in atom_names or "C" in atom_names)
        if not is_protein_like:
            for idx in idxs:
                hetatm_indices.add(idx)
                if res_name in _WATER_RESNAMES:
                    water_indices.add(idx)
                elif res_name in _ION_RESNAMES:
                    ion_indices.add(idx)
                else:
                    ligand_indices.add(idx)
            continue

        ca_index = c_index = o_index = n_index = None
        for idx in idxs:
            an = atom_name_by_idx[idx].upper()
            if an == "CA":
                ca_index = idx
                backbone_indices.add(idx)
            elif an == "C":
                c_index = idx
                backbone_indices.add(idx)
            elif an == "N":
                n_index = idx
                backbone_indices.add(idx)
            elif an in {"O", "OXT"}:
                if an == "O" and o_index is None:
                    o_index = idx
                backbone_indices.add(idx)
            else:
                sidechain_indices.add(idx)
        chains.setdefault(chain_id, []).append(
            ResidueData(
                res_name=res_name,
                res_seq=res_seq,
                chain_id=chain_id,
                atom_indices=list(idxs),
                ca_index=ca_index,
                c_index=c_index,
                o_index=o_index,
                n_index=n_index,
                ss_type=ss_by_residue[(chain_id, res_seq, res_name)],
            )
        )

    if not chains:
        return None

    chain_data = {cid: ChainData(chain_id=cid, residues=res) for cid, res in chains.items()}
    has_ca_trace = any(r.ca_index is not None for ch in chain_data.values() for r in ch.residues)
    has_ss = any(r.ss_type in {"H", "E"} for ch in chain_data.values() for r in ch.residues)
    confidence = ProteinConfidence.FULL_RIBBON if has_ca_trace else ProteinConfidence.TRACE_ONLY
    reasons = [f"{provenance} residue/chain metadata parsed"]
    if not has_ca_trace:
        reasons.append("CA atoms missing; downgraded to TRACE_ONLY")
    if not has_ss:
        reasons.append("no explicit helix/sheet labels; loops-only ribbon")
    return ProteinSemantics(
        chains=chain_data,
        hetatm_indices=hetatm_indices,
        backbone_indices=backbone_indices,
        sidechain_indices=sidechain_indices,
        helix_spans=[],
        sheet_spans=[],
        ligand_indices=ligand_indices,
        water_indices=water_indices,
        ion_indices=ion_indices,
        confidence_tier=confidence,
        confidence_reasons=reasons,
        provenance=[provenance],
        trace_chains={},
    )


def _extract_from_mol2(path: Path) -> ProteinSemantics | None:
    """Extract residue/chain semantics from MOL2 ATOM records."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    in_atom = False
    rows: list[dict[str, object]] = []
    for ln in lines:
        s = ln.strip()
        up = s.upper()
        if up.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if up.startswith("@<TRIPOS>") and in_atom:
            break
        if not in_atom or not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 6:
            continue
        atom_name = parts[1]
        subst_id = 0
        subst_name = "RES"
        if len(parts) > 6:
            try:
                subst_id = int(parts[6])
            except ValueError:
                subst_id = 0
        if len(parts) > 7:
            subst_name = parts[7]
        # Heuristic parsing of subst_name: e.g. ALA12_A, ALA12:A, ALA12
        chain_id = "A"
        m_chain = re.search(r"[:_]([A-Za-z0-9])$", subst_name)
        if m_chain:
            chain_id = m_chain.group(1)
        m_res = re.match(r"([A-Za-z]{3})", subst_name)
        res_name = m_res.group(1).upper() if m_res else "RES"
        m_seq = re.search(r"(-?\d+)", subst_name)
        res_seq = int(m_seq.group(1)) if m_seq else subst_id
        rows.append(
            {
                "record_type": "ATOM",
                "atom_name": atom_name,
                "res_name": res_name,
                "chain_id": chain_id,
                "res_seq": res_seq,
                "ss_type": "C",
            }
        )
    return _build_semantics_from_atom_meta(rows, provenance="mol2", allow_ss=False)


def _parse_extxyz_properties(comment: str) -> dict[str, tuple[int, int]] | None:
    m = re.search(r"Properties=([^\s]+)", comment)
    if m is None:
        return None
    spec = m.group(1)
    toks = spec.split(":")
    if len(toks) < 3:
        return None
    props: dict[str, tuple[int, int]] = {}
    col = 0
    i = 0
    while i + 2 < len(toks):
        name = toks[i]
        count = int(toks[i + 2])
        props[name] = (col, count)
        col += count
        i += 3
    return props


def _extract_from_extxyz(path: Path) -> ProteinSemantics | None:
    """Extract canonical protein metadata from extXYZ atom properties."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        return None
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        return None
    comment = lines[1]
    props = _parse_extxyz_properties(comment)
    if props is None:
        return None
    required = ["atom_name", "res_name", "res_seq", "chain_id", "ss_type"]
    if any(k not in props for k in required):
        return None

    atom_lines = lines[2 : 2 + n_atoms]
    rows: list[dict[str, object]] = []
    for ln in atom_lines:
        parts = ln.split()
        if not parts:
            continue
        if len(parts) < 4:
            return None
        def _get(name: str, _parts: list[str] = parts) -> str:
            offset, count = props[name]
            # extxyz tokens include species + xyz first (4 columns)
            # our offset is against properties vector, so shift by 0 here as species/pos are part of properties.
            idx = offset
            if idx >= len(_parts):
                return ""
            return " ".join(_parts[idx : idx + count]).strip()

        atom_name = _get("atom_name")
        res_name = _get("res_name")
        res_seq_raw = _get("res_seq")
        chain_id = _get("chain_id")
        ss_type = _get("ss_type")
        try:
            res_seq = int(res_seq_raw)
        except ValueError:
            return None
        rows.append(
            {
                "record_type": "ATOM",
                "atom_name": atom_name,
                "res_name": res_name,
                "chain_id": chain_id,
                "res_seq": res_seq,
                "ss_type": ss_type,
            }
        )
    return _build_semantics_from_atom_meta(rows, provenance="extxyz", allow_ss=True)


def _greedy_trace(indices: list[int], pos: np.ndarray) -> list[int]:
    if len(indices) < 2:
        return indices
    remaining = set(indices)
    # deterministic seed: lowest x then y then z
    start = min(indices, key=lambda i: (float(pos[i][0]), float(pos[i][1]), float(pos[i][2]), i))
    trace = [start]
    remaining.remove(start)
    current = start
    while remaining:
        nxt = min(remaining, key=lambda j: (float(np.linalg.norm(pos[current] - pos[j])), j))
        trace.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return trace


def _extract_heuristic(graph: "nx.Graph") -> ProteinSemantics | None:
    """Graph-only fallback semantics for trace rendering."""
    node_ids = list(graph.nodes())
    if not node_ids:
        return None
    symbols = {i: str(graph.nodes[i].get("symbol", "")) for i in node_ids}
    pos = np.array([graph.nodes[i].get("position", (0.0, 0.0, 0.0)) for i in node_ids], dtype=float)
    idx_map = {nid: k for k, nid in enumerate(node_ids)}
    peptide_like = [nid for nid in node_ids if symbols[nid] in {"C", "N", "O", "S"}]
    if len(peptide_like) < 12:
        return None
    ca_like: list[int] = []
    for nid in peptide_like:
        if symbols[nid] != "C":
            continue
        nbr_syms = {symbols.get(nb, "") for nb in graph.neighbors(nid)}
        if "N" in nbr_syms and ("C" in nbr_syms or "O" in nbr_syms):
            ca_like.append(nid)
    if len(ca_like) < 4:
        return None
    ca_indices = [idx_map[n] for n in ca_like]
    trace = _greedy_trace(ca_indices, pos)
    # Build synthetic single-chain residues using trace order
    residues: list[ResidueData] = []
    for k, ai in enumerate(trace, start=1):
        residues.append(
            ResidueData(
                res_name="UNK",
                res_seq=k,
                chain_id="A",
                atom_indices=[ai],
                ca_index=ai,
                c_index=None,
                o_index=None,
                n_index=None,
                ss_type="C",
            )
        )
    chains = {"A": ChainData(chain_id="A", residues=residues)}
    return ProteinSemantics(
        chains=chains,
        hetatm_indices=set(),
        backbone_indices=set(trace),
        sidechain_indices={idx_map[n] for n in peptide_like if idx_map[n] not in set(trace)},
        helix_spans=[],
        sheet_spans=[],
        ligand_indices=set(),
        water_indices=set(),
        ion_indices=set(),
        confidence_tier=ProteinConfidence.TRACE_ONLY,
        confidence_reasons=["graph-only heuristic backbone trace"],
        provenance=["heuristic"],
        trace_chains={"A": trace},
    )


def extract_protein_semantics(
    graph: "nx.Graph",
    *,
    moldata: "MolData | None" = None,
    source_path: str | Path | None = None,
    protein_requested: bool = False,
    format_hint: str | None = None,
) -> ProteinSemantics | None:
    """Extract unified protein semantics from metadata and optional heuristics."""
    sem = _extract_from_xyzgraph(
        graph,
        moldata=moldata,
        source_path=source_path,
        protein_requested=protein_requested,
        format_hint=format_hint,
    )
    if sem is not None:
        return sem

    # Compatibility fallback: support older xyzgraph versions that do not yet
    # consume canonical atom annotations.
    annotations = _canonical_annotations(moldata=moldata, source_path=source_path, format_hint=format_hint)
    if annotations:
        path = Path(source_path) if source_path is not None else None
        fmt = (format_hint or (path.suffix.lower() if path is not None else "")).lower()
        provenance = f"annotations:{fmt}" if fmt else "annotations"
        sem = _build_semantics_from_atom_meta(annotations, provenance=provenance, allow_ss=True)
        if sem is not None:
            return sem

    # Legacy fallback path (kept for compatibility with older xyzgraph versions).
    if moldata is not None and getattr(moldata, "protein_data", None) is not None:
        return from_protein_data(moldata.protein_data, provenance="pdb")

    path = Path(source_path) if source_path is not None else None
    fmt = (format_hint or (path.suffix.lower() if path is not None else "")).lower()

    sem = None
    if path is not None and fmt == ".mol2":
        sem = _extract_from_mol2(path)
    elif path is not None and fmt == ".xyz":
        sem = _extract_from_extxyz(path)

    if sem is not None:
        return sem

    if protein_requested:
        heuristic = _extract_heuristic(graph)
        if heuristic is not None:
            return heuristic
        return ProteinSemantics(
            chains={},
            hetatm_indices=set(),
            backbone_indices=set(),
            sidechain_indices=set(),
            helix_spans=[],
            sheet_spans=[],
            ligand_indices=set(),
            water_indices=set(),
            ion_indices=set(),
            confidence_tier=ProteinConfidence.INSUFFICIENT,
            confidence_reasons=["insufficient metadata and weak graph-only signal"],
            provenance=["none"],
            trace_chains={},
        )
    return None
