"""Protein ribbon geometry and SVG generation.

Produces depth-sorted ``(z_depth, svg_lines)`` tuples consumed by the main
renderer via the same drain pattern as NCI patches.

Secondary structure elements:
- Helix  (ss_type="H") — wide flat ribbon, swept-quad panels
- Sheet  (ss_type="E") — wide flat ribbon with triangular arrowhead at C-terminus
- Loop   (ss_type="C") — thin rounded tube rendered as a stroked SVG path
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xyzrender.types import ProteinData, ProteinSemantics, RenderConfig, ResidueData

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default Catmull-Rom interpolation steps between each CA pair
_SPLINE_STEPS = 8
# Arrowhead width multiplier for last strand residues
_ARROW_FLANGE = 1.8
# Arrowhead tip extension beyond last CA (fraction of average CA-CA distance)
_ARROW_TIP_FRAC = 0.6
# Minimum residues to attempt ribbon rendering for a segment
_MIN_HELIX_RES = 2
_MIN_SHEET_RES = 2
_MIN_LOOP_RES = 2
# Backbone discontinuity thresholds (metadata + geometry)
_MAX_SEQ_STEP = 1
_MAX_CA_GAP = 5.0  # Å
_EPS = 1e-8

# Projected full-width clamps (px) for visual consistency across zoom levels
_HELIX_WIDTH_PX = (6.0, 22.0)
_SHEET_WIDTH_PX = (5.5, 20.0)
_TRANSITION_WIDTH_PX = (4.5, 14.0)
_LOOP_WIDTH_PX = (2.5, 9.5)
_LOOP_WIDTH_RATIO = 0.45
_DEFAULT_RIBBON_STYLE = "gloss"


@dataclass(frozen=True)
class RibbonStyleProfile:
    """Per-style ribbon rendering controls."""

    name: str
    use_gradients: bool
    ribbon_scale: float
    helix_mult: float
    sheet_mult: float
    transition_mult: float
    loop_mult: float
    rail_scale: float
    rail_floor_px: float
    helix_outline_dark: float
    helix_outline_light: float
    sheet_outline_dark: float
    sheet_outline_light: float
    loop_dark_blend: float
    loop_width_scale: float
    gradient_helix_contrast: float
    gradient_sheet_contrast: float


_RIBBON_STYLE_PROFILES: dict[str, RibbonStyleProfile] = {
    "gloss": RibbonStyleProfile(
        name="gloss",
        use_gradients=True,
        ribbon_scale=1.42,
        helix_mult=1.12,
        sheet_mult=1.08,
        transition_mult=1.02,
        loop_mult=1.00,
        rail_scale=0.72,
        rail_floor_px=0.55,
        helix_outline_dark=0.08,
        helix_outline_light=0.03,
        sheet_outline_dark=0.08,
        sheet_outline_light=0.04,
        loop_dark_blend=0.09,
        loop_width_scale=0.92,
        gradient_helix_contrast=0.08,
        gradient_sheet_contrast=0.04,
    ),
    "illustration": RibbonStyleProfile(
        name="illustration",
        use_gradients=True,
        ribbon_scale=0.86,
        helix_mult=1.02,
        sheet_mult=0.98,
        transition_mult=0.92,
        loop_mult=1.00,
        rail_scale=0.40,
        rail_floor_px=0.28,
        helix_outline_dark=0.14,
        helix_outline_light=0.01,
        sheet_outline_dark=0.14,
        sheet_outline_light=0.02,
        loop_dark_blend=0.15,
        loop_width_scale=0.80,
        gradient_helix_contrast=0.0,
        gradient_sheet_contrast=0.0,
    ),
}


def default_ribbon_style() -> str:
    """Canonical default style used for --protein and protein=True."""
    return _DEFAULT_RIBBON_STYLE


def normalize_ribbon_style(style: str) -> str:
    """Normalize and validate ribbon style names."""
    key = style.strip().lower()
    if key not in _RIBBON_STYLE_PROFILES:
        raise ValueError(f"unknown ribbon style {style!r}")
    return key


def ribbon_style_profile(style: str) -> RibbonStyleProfile:
    """Return the resolved style profile for a ribbon style name."""
    return _RIBBON_STYLE_PROFILES[normalize_ribbon_style(style)]


def ribbon_style_names(*, include_aliases: bool = True) -> tuple[str, ...]:
    """Return valid ribbon style names for API/CLI validation."""
    names = list(_RIBBON_STYLE_PROFILES.keys())
    if include_aliases:
        return tuple(sorted(names))
    return tuple(sorted(names))


def ribbon_style_uses_gradients(style: str) -> bool:
    """Whether a ribbon style requires gradient defs."""
    return ribbon_style_profile(style).use_gradients


# ---------------------------------------------------------------------------
# Chain colour assignment
# ---------------------------------------------------------------------------


def _muted_pastel_color(hex_color: str) -> str:
    """Desaturate/lighten chain colours for textbook illustration styling."""
    from xyzrender.colors import Color

    c = Color.from_str(hex_color)
    h, lightness, s = c.to_hls()
    lightness = min(1.0, lightness + 0.18 * (1.0 - lightness))
    s = max(0.20, s * 0.58)
    return Color.from_hls(h, lightness, s).hex


def assign_chain_colors(cfg: "RenderConfig", chain_ids: list[str], style: str | None = None) -> dict[str, str]:
    """Return ``chain_id → hex colour`` mapping.

    Uses ``cfg.chain_colors`` overrides where given, filling the rest from
    ``cfg.protein_palette``.
    """
    overrides = cfg.chain_colors or {}
    palette = cfg.protein_palette
    style_name = normalize_ribbon_style(style or cfg.protein_style)
    pastel_auto = style_name == "illustration"
    result: dict[str, str] = {}
    auto_idx = 0
    for cid in chain_ids:
        if cid in overrides:
            result[cid] = overrides[cid]
        else:
            raw = palette[auto_idx % len(palette)]
            result[cid] = _muted_pastel_color(raw) if pastel_auto else raw
            auto_idx += 1
    return result


# ---------------------------------------------------------------------------
# Gradient defs for ribbon shading
# ---------------------------------------------------------------------------


def ribbon_gradient_defs(chain_colors: dict[str, str], style: str = "gloss") -> list[str]:
    """Emit ``<linearGradient>`` defs for ribbon shading (one per unique chain colour).

    Gradient IDs are keyed by hex colour:
    - ``rg_h_{hex_without_hash}`` for helices (higher contrast)
    - ``rg_e_{hex_without_hash}`` for sheets  (softer contrast)
    """
    from xyzrender.colors import Color

    profile = ribbon_style_profile(style)
    lines: list[str] = []
    seen: set[str] = set()

    def _mk_stops(hex_color: str, contrast: float) -> tuple[str, str, str]:
        c = Color.from_str(hex_color)
        if profile.name == "illustration":
            dark = c.blend(Color(0, 0, 0), 0.055).hex
            light = c.blend(Color(255, 255, 255), 0.045).hex
            specular = c.blend(Color(255, 255, 255), 0.070).hex
            return dark, light, specular
        dark = c.blend(Color(0, 0, 0), 0.07 + 0.16 * contrast).hex
        light = c.blend(Color(255, 255, 255), 0.06 + 0.12 * contrast).hex
        specular = c.blend(Color(255, 255, 255), 0.10 + 0.18 * contrast).hex
        return dark, light, specular

    for _cid, hex_color in chain_colors.items():
        if hex_color in seen:
            continue
        seen.add(hex_color)
        h_gid = f"rg_h_{hex_color[1:]}"  # helix: stronger contrast
        e_gid = f"rg_e_{hex_color[1:]}"  # sheet: softer contrast
        h_dark, h_light, h_spec = _mk_stops(hex_color, contrast=profile.gradient_helix_contrast)
        e_dark, e_light, e_spec = _mk_stops(hex_color, contrast=profile.gradient_sheet_contrast)
        lines.append(
            f'    <linearGradient id="{h_gid}" x1="0" y1="0" x2="1" y2="0">'
            f'<stop offset="0%" stop-color="{h_dark}"/>'
            f'<stop offset="30%" stop-color="{h_light}"/>'
            f'<stop offset="50%" stop-color="{h_spec}"/>'
            f'<stop offset="70%" stop-color="{h_light}"/>'
            f'<stop offset="100%" stop-color="{h_dark}"/>'
            f"</linearGradient>"
        )
        lines.append(
            f'    <linearGradient id="{e_gid}" x1="0" y1="0" x2="1" y2="0">'
            f'<stop offset="0%" stop-color="{e_dark}"/>'
            f'<stop offset="30%" stop-color="{e_light}"/>'
            f'<stop offset="50%" stop-color="{e_spec}"/>'
            f'<stop offset="70%" stop-color="{e_light}"/>'
            f'<stop offset="100%" stop-color="{e_dark}"/>'
            f"</linearGradient>"
        )
    return lines


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _catmull_rom_3d(points: np.ndarray, steps: int = _SPLINE_STEPS) -> np.ndarray:
    """Smooth 3D Catmull-Rom spline through *points* with *steps* between each pair.

    Returns an array of shape ``(n_segments * steps + 1, 3)``.
    """
    n = len(points)
    if n < 2:
        return points.copy()

    # Phantom endpoints (reflected) so the curve starts/ends tangent to first/last segment
    ext = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])
    ts = np.linspace(0.0, 1.0, steps, endpoint=False)  # (steps,)
    t2 = ts * ts
    t3 = t2 * ts

    out: list[np.ndarray] = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        # Coefficients, shape (3,) each
        a = 2.0 * p1
        b = -p0 + p2
        c = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
        d = -p0 + 3.0 * p1 - 3.0 * p2 + p3
        # Broadcast to (steps, 3): ts has shape (steps,), coeffs have shape (3,)
        q = 0.5 * (a + b * ts[:, None] + c * t2[:, None] + d * t3[:, None])
        out.append(q)
    out.append(points[[-1]])
    return np.vstack(out)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _adaptive_spline_steps(ca_pos: np.ndarray, scale: float) -> int:
    """Choose spline density automatically from chain size and projected span."""
    n = len(ca_pos)
    if n <= 1:
        return _SPLINE_STEPS

    span_world = float(np.max(np.ptp(ca_pos[:, :2], axis=0)))
    span_px = span_world * max(scale, 1e-8)
    if n > 2200:
        steps = 4
    elif n > 1200:
        steps = 5
    elif n > 700:
        steps = 6
    else:
        steps = 8

    if span_px > 2200.0:
        steps = max(4, steps - 1)
    elif span_px < 900.0:
        steps = min(10, steps + 1)
    return steps


def _ss_spline_steps(base_steps: int, ss_type: str) -> int:
    """Secondary-structure-specific tessellation density."""
    if ss_type == "H":
        return max(6, min(12, base_steps + 1))
    if ss_type == "E":
        return max(3, min(5, round(base_steps * 0.35)))
    return base_steps


def _clamp_half_width(half_width: float, scale: float, px_limits: tuple[float, float]) -> float:
    """Clamp a half-width in world units via projected full-width pixel limits."""
    if scale <= _EPS:
        return half_width
    min_px, max_px = px_limits
    full_px = 2.0 * half_width * scale
    clamped_px = min(max(full_px, min_px), max_px)
    return clamped_px / (2.0 * scale)


def _outline_color(base: str, dark_blend: float, light_blend: float) -> str:
    """Derive an outline color from a chain color using dark/light blends."""
    from xyzrender.colors import Color

    c = Color.from_str(base)
    if dark_blend > 0.0:
        c = c.blend(Color(0, 0, 0), dark_blend)
    if light_blend > 0.0:
        c = c.blend(Color(255, 255, 255), light_blend)
    return c.hex


def _fallback_perp(tangent: np.ndarray) -> np.ndarray:
    """Return a stable arbitrary unit vector perpendicular to *tangent*."""
    arb = np.array([1.0, 0.0, 0.0]) if abs(tangent[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = np.cross(tangent, arb)
    if np.linalg.norm(perp) <= _EPS:
        perp = np.cross(tangent, np.array([0.0, 0.0, 1.0]))
    return _unit(perp)


def _transport_normal(prev_n: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Parallel-transport *prev_n* from tangent *t0* onto tangent *t1*."""
    axis = np.cross(t0, t1)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= _EPS:
        n = prev_n - np.dot(prev_n, t1) * t1
        nn = np.linalg.norm(n)
        return n / nn if nn > _EPS else _fallback_perp(t1)

    axis = axis / axis_norm
    cos_a = float(np.clip(np.dot(t0, t1), -1.0, 1.0))
    angle = float(np.arctan2(axis_norm, cos_a))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    # Rodrigues rotation formula
    n = prev_n * c + np.cross(axis, prev_n) * s + axis * np.dot(axis, prev_n) * (1.0 - c)
    n = n - np.dot(n, t1) * t1
    nn = np.linalg.norm(n)
    return n / nn if nn > _EPS else _fallback_perp(t1)


def _preferred_normal(
    idx: int,
    ca_pos: np.ndarray,
    o_pos: list[np.ndarray | None],
    tangents: np.ndarray,
    ss_types: list[str],
) -> np.ndarray | None:
    """Return an orientation preference normal (or None when unavailable)."""
    t = tangents[idx]
    ss = ss_types[idx] if idx < len(ss_types) else "C"
    if ss == "H":
        p_prev = ca_pos[max(0, idx - 1)]
        p_next = ca_pos[min(len(ca_pos) - 1, idx + 1)]
        raw = p_prev + p_next - 2 * ca_pos[idx]
    elif o_pos[idx] is not None:
        raw = o_pos[idx] - ca_pos[idx]
    else:
        t_prev = tangents[max(0, idx - 1)]
        t_next = tangents[min(len(ca_pos) - 1, idx + 1)]
        raw = np.cross(t_prev, t_next)
    raw = raw - np.dot(raw, t) * t
    rn = np.linalg.norm(raw)
    if rn <= _EPS:
        return None
    return raw / rn


def _compute_normals_mixed(
    ca_pos: np.ndarray,
    o_pos: list[np.ndarray | None],
    ss_types: list[str],
) -> np.ndarray:
    """Compute smoothly transported ribbon normals across mixed SS segments."""
    n = len(ca_pos)
    if n == 0:
        return np.zeros((0, 3), dtype=float)

    tangents = np.zeros((n, 3), dtype=float)
    for i in range(n):
        p_prev = ca_pos[max(0, i - 1)]
        p_next = ca_pos[min(n - 1, i + 1)]
        t = p_next - p_prev
        tn = np.linalg.norm(t)
        if tn > _EPS:
            tangents[i] = t / tn
        elif i > 0:
            tangents[i] = tangents[i - 1]
        else:
            tangents[i] = np.array([1.0, 0.0, 0.0], dtype=float)

    preferred: list[np.ndarray | None] = [
        _preferred_normal(i, ca_pos, o_pos, tangents, ss_types) for i in range(n)
    ]

    normals = np.zeros((n, 3), dtype=float)
    seed = next((p for p in preferred if p is not None), None)
    normals[0] = seed if seed is not None else _fallback_perp(tangents[0])

    for i in range(1, n):
        normals[i] = _transport_normal(normals[i - 1], tangents[i - 1], tangents[i])

    # Global sign alignment towards chemically meaningful normals (when available).
    score = 0.0
    for i, pref in enumerate(preferred):
        if pref is not None:
            score += float(np.dot(normals[i], pref))
    if score < 0.0:
        normals = -normals

    # Smooth with a wider window, while re-projecting perpendicular to the local tangent.
    if n >= 3:
        smoothed = normals.copy()
        weights = [1.0, 2.0, 3.0, 2.0, 1.0]
        offsets = [-2, -1, 0, 1, 2]
        for i in range(n):
            acc = np.zeros(3, dtype=float)
            wsum = 0.0
            for off, w in zip(offsets, weights, strict=False):
                j = i + off
                if 0 <= j < n:
                    acc += normals[j] * w
                    wsum += w
            if wsum <= _EPS:
                continue
            v = acc / wsum
            v = v - np.dot(v, tangents[i]) * tangents[i]
            vn = np.linalg.norm(v)
            if vn > _EPS:
                smoothed[i] = v / vn
        normals = smoothed

    # Enforce sign continuity to prevent ribbon flips.
    for i in range(1, n):
        if np.dot(normals[i], normals[i - 1]) < 0:
            normals[i] = -normals[i]

    return normals


def _proj_xy(
    p: np.ndarray,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: float,
    canvas_h: float,
) -> tuple[float, float]:
    """Orthographic 3D → 2D projection (matches renderer._proj)."""
    return canvas_w / 2 + scale * (p[0] - cx), canvas_h / 2 - scale * (p[1] - cy)


def _quad_polygon_svg(
    corners_3d: np.ndarray,
    color: str,
    outline: str | None,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: float,
    canvas_h: float,
    stroke_width: float = 0.6,
) -> tuple[float, list[str]]:
    """Project a 4-corner 3D quad to 2D and return (centroid_z, svg_lines)."""
    pts_2d = [_proj_xy(c, scale, cx, cy, canvas_w, canvas_h) for c in corners_3d]
    pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_2d)
    centroid_z = float(corners_3d[:, 2].mean())
    stroke_attr = "none" if not outline or stroke_width <= 0 else outline
    lines = [f'  <polygon points="{pts_str}" fill="{color}" stroke="{stroke_attr}" stroke-width="{stroke_width:.2f}"/>']
    return centroid_z, lines


def _triangle_polygon_svg(
    corners_3d: np.ndarray,
    color: str,
    outline: str | None,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: float,
    canvas_h: float,
    stroke_width: float = 0.6,
) -> tuple[float, list[str]]:
    """Project a 3-corner 3D triangle to 2D and return (centroid_z, svg_lines)."""
    pts_2d = [_proj_xy(c, scale, cx, cy, canvas_w, canvas_h) for c in corners_3d]
    pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_2d)
    centroid_z = float(corners_3d[:, 2].mean())
    stroke_attr = "none" if not outline or stroke_width <= 0 else outline
    lines = [f'  <polygon points="{pts_str}" fill="{color}" stroke="{stroke_attr}" stroke-width="{stroke_width:.2f}"/>']
    return centroid_z, lines


def _polyline_svg(
    points_3d: np.ndarray,
    color: str,
    stroke_width: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: float,
    canvas_h: float,
) -> tuple[float, list[str]] | None:
    """Project a 3D polyline and return a stroked SVG path item."""
    if len(points_3d) < 2 or stroke_width <= 0:
        return None
    pts_2d = [_proj_xy(p, scale, cx, cy, canvas_w, canvas_h) for p in points_3d]
    d = f"M {pts_2d[0][0]:.1f} {pts_2d[0][1]:.1f}" + "".join(f" L {x:.1f} {y:.1f}" for x, y in pts_2d[1:])
    z = float(points_3d[:, 2].mean())
    lines = [
        f'  <path d="{d}" fill="none" stroke="{color}" '
        f'stroke-width="{stroke_width:.2f}" stroke-linecap="round" stroke-linejoin="round"/>'
    ]
    return z, lines


def _strip_polygon_svg(
    left_3d: np.ndarray,
    right_3d: np.ndarray,
    fill: str,
    outline_color: str | None,
    stroke_width: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: float,
    canvas_h: float,
) -> tuple[float, list[str]] | None:
    """Project ribbon rails into one closed polygon strip item."""
    m = min(len(left_3d), len(right_3d))
    if m < 2:
        return None
    poly_3d = np.vstack([left_3d[:m], right_3d[:m][::-1]])
    pts_2d = [_proj_xy(p, scale, cx, cy, canvas_w, canvas_h) for p in poly_3d]
    pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_2d)
    z = float(np.vstack([left_3d[:m], right_3d[:m]])[:, 2].mean())
    stroke = "none" if not outline_color or stroke_width <= 0 else outline_color
    return z, [f'  <polygon points="{pts_str}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}"/>']


def _strip_with_rails_items(
    centerline_3d: np.ndarray,
    normals_3d: np.ndarray,
    half_width: float,
    steps: int,
    fill: str,
    outline: str | None,
    body_outline_width: float,
    rail_scale: float,
    rail_floor_px: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: float,
    canvas_h: float,
) -> list[tuple[float, list[str]]]:
    """Render one smooth ribbon strip with optional side rails."""
    if len(centerline_3d) < 2:
        return []
    left = _catmull_rom_3d(centerline_3d + normals_3d * half_width, steps=steps)
    right = _catmull_rom_3d(centerline_3d - normals_3d * half_width, steps=steps)
    items: list[tuple[float, list[str]]] = []
    strip = _strip_polygon_svg(left, right, fill, outline, body_outline_width, scale, cx, cy, canvas_w, canvas_h)
    if strip is not None:
        items.append(strip)
    projected_half = max(half_width * scale, _EPS)
    rail_w = max(projected_half * 0.16 * rail_scale, rail_floor_px)
    rail_w = min(rail_w, projected_half * 0.28)
    if outline:
        left_rail = _polyline_svg(left, outline, rail_w, scale, cx, cy, canvas_w, canvas_h)
        right_rail = _polyline_svg(right, outline, rail_w, scale, cx, cy, canvas_w, canvas_h)
        if left_rail is not None:
            items.append(left_rail)
        if right_rail is not None:
            items.append(right_rail)
    return items


def _loop_items(
    ca_pos: np.ndarray,
    color: str,
    stroke_w: float,
    style: RibbonStyleProfile,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: float,
    canvas_h: float,
) -> list[tuple[float, list[str]]]:
    """Generate depth-sorted path items for a loop/coil segment.

    Each segment is rendered as one darker stroked path to keep connectors
    visually slim and consistent with ribbon contours.
    """
    if len(ca_pos) < _MIN_LOOP_RES:
        return []

    from xyzrender.colors import Color

    steps = _adaptive_spline_steps(ca_pos, scale)
    # Smooth spline through CA positions
    spline = _catmull_rom_3d(ca_pos, steps=steps)
    if len(spline) < 2:
        return []

    # Project all spline points to 2D
    px = [_proj_xy(p, scale, cx, cy, canvas_w, canvas_h) for p in spline]

    # Tube colour: slightly darkened chain colour (no stacked highlight stroke)
    c = Color.from_str(color)
    stroke_color = c.blend(Color(0, 0, 0), style.loop_dark_blend).hex
    base_w = max(stroke_w * style.loop_width_scale, 0.9)

    # Build one SVG path per segment (one item per residue-span for z-ordering)
    items: list[tuple[float, list[str]]] = []
    for seg in range(len(ca_pos) - 1):
        start = seg * steps
        end = min((seg + 1) * steps + 1, len(px))
        seg_pts = px[start:end]
        seg_3d = spline[start:end]

        if len(seg_pts) < 2:
            continue

        # Build SVG path string for this segment
        d = f"M {seg_pts[0][0]:.1f} {seg_pts[0][1]:.1f}"
        for pt in seg_pts[1:]:
            d += f" L {pt[0]:.1f} {pt[1]:.1f}"

        centroid_z = float(seg_3d[:, 2].mean())
        svg_lines = [
            f'  <path d="{d}" fill="none" stroke="{stroke_color}" '
            f'stroke-width="{base_w:.1f}" stroke-linecap="round" stroke-linejoin="round"/>'
        ]
        items.append((centroid_z, svg_lines))

    return items


# ---------------------------------------------------------------------------
# Segment extraction helpers
# ---------------------------------------------------------------------------


def _extract_ca_o_ss(
    residues: list["ResidueData"],
    pos: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray | None], list[str]]:
    """Extract CA positions, O positions and normalized ss labels."""
    ca_list: list[np.ndarray] = []
    o_list: list[np.ndarray | None] = []
    ss_list: list[str] = []
    for res in residues:
        if res.ca_index is None:
            continue
        ca_list.append(pos[res.ca_index])
        o_list.append(pos[res.o_index] if res.o_index is not None else None)
        ss = res.ss_type if res.ss_type in {"H", "E", "C"} else "C"
        ss_list.append(ss)
    return np.array(ca_list), o_list, ss_list


def _split_backbone_segments(
    residues: list["ResidueData"],
    pos: np.ndarray,
) -> list[list["ResidueData"]]:
    """Split a chain into continuous CA segments using sequence + geometry.

    This prevents rendering across true backbone discontinuities (missing
    residues, unresolved loops) while keeping style transitions continuous
    inside each segment.
    """
    ca_res = [r for r in residues if r.ca_index is not None]
    if len(ca_res) < 2:
        return [ca_res] if ca_res else []

    segments: list[list["ResidueData"]] = [[ca_res[0]]]
    prev = ca_res[0]
    for cur in ca_res[1:]:
        seq_step = int(cur.res_seq) - int(prev.res_seq)
        gap = float(np.linalg.norm(pos[cur.ca_index] - pos[prev.ca_index]))
        split = (seq_step != _MAX_SEQ_STEP) or (gap > _MAX_CA_GAP)
        if split:
            segments.append([cur])
        else:
            segments[-1].append(cur)
        prev = cur
    return [seg for seg in segments if len(seg) >= 2]


def _runs_of_type(ss: list[str], target: str) -> list[tuple[int, int]]:
    """Return inclusive [start, end] index runs for *target* SS label."""
    out: list[tuple[int, int]] = []
    i = 0
    n = len(ss)
    while i < n:
        if ss[i] != target:
            i += 1
            continue
        j = i
        while j + 1 < n and ss[j + 1] == target:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def _continuous_segment_items(
    ca_pos: np.ndarray,
    o_pos: list[np.ndarray | None],
    ss: list[str],
    color: str,
    style: RibbonStyleProfile,
    helix_outline: str,
    sheet_outline: str,
    loop_outline: str,
    half_ribbon: float,
    loop_half_base: float,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: float,
    canvas_h: float,
) -> list[tuple[float, list[str]]]:
    """Render one continuous backbone segment with style transitions."""
    n = len(ca_pos)
    if n < 2:
        return []

    normals = _compute_normals_mixed(ca_pos, o_pos, ss)
    glossy = style.use_gradients
    helix_fill = f"url(#rg_h_{color[1:]})" if glossy else color
    sheet_fill = f"url(#rg_e_{color[1:]})" if glossy else color
    loop_fill = color

    style_half = half_ribbon * style.ribbon_scale
    helix_half = _clamp_half_width(style_half * style.helix_mult, scale, _HELIX_WIDTH_PX)
    sheet_half = _clamp_half_width(style_half * style.sheet_mult, scale, _SHEET_WIDTH_PX)
    transition_half = _clamp_half_width(style_half * style.transition_mult, scale, _TRANSITION_WIDTH_PX)
    loop_base = min(loop_half_base, transition_half * _LOOP_WIDTH_RATIO)
    loop_half = _clamp_half_width(loop_base * style.loop_mult, scale, _LOOP_WIDTH_PX)

    base_steps = _adaptive_spline_steps(ca_pos, scale)
    helix_steps = _ss_spline_steps(base_steps, "H")
    sheet_steps = _ss_spline_steps(base_steps, "E")
    rail_scale = style.rail_scale
    transition_sw = min(max(transition_half * scale * 0.20, 0.58), 1.25)
    items: list[tuple[float, list[str]]] = []
    helix_links = np.array([(ss[i] == "H" and ss[i + 1] == "H") for i in range(n - 1)], dtype=bool)
    sheet_links = np.array([(ss[i] == "E" and ss[i + 1] == "E") for i in range(n - 1)], dtype=bool)

    # Ribbon-like quads for all non-coil links, including transition links.
    for i in range(n - 1):
        if helix_links[i] or sheet_links[i]:
            continue
        t0, t1 = ss[i], ss[i + 1]
        if t0 == "C" and t1 == "C":
            continue
        p0, p1 = ca_pos[i], ca_pos[i + 1]
        b0, b1 = normals[i], normals[i + 1]
        if t0 == "H":
            w0 = helix_half
        elif t0 == "E":
            w0 = sheet_half
        else:
            w0 = transition_half
        if t1 == "H":
            w1 = helix_half
        elif t1 == "E":
            w1 = sheet_half
        else:
            w1 = transition_half
        corners = np.array([p0 + b0 * w0, p0 - b0 * w0, p1 - b1 * w1, p1 + b1 * w1])

        if "H" in {t0, t1}:
            fill = helix_fill
            contour = helix_outline
        elif "E" in {t0, t1}:
            fill = sheet_fill
            contour = sheet_outline
        else:
            fill = loop_fill
            contour = loop_outline
        items.append(
            _quad_polygon_svg(corners, fill, contour, scale, cx, cy, canvas_w, canvas_h, stroke_width=transition_sw)
        )

    # Helix runs use a continuous strip polygon (PyMOL-like solid ribbon body).
    for s, e in _runs_of_type(ss, "H"):
        if e - s + 1 < _MIN_HELIX_RES:
            continue
        items.extend(
            _strip_with_rails_items(
                ca_pos[s : e + 1],
                normals[s : e + 1],
                helix_half,
                helix_steps,
                helix_fill,
                helix_outline,
                min(max(helix_half * scale * 0.20, 0.62), 1.30),
                rail_scale,
                style.rail_floor_px,
                scale,
                cx,
                cy,
                canvas_w,
                canvas_h,
            )
        )

    # Sheet runs also use continuous strips, with arrowheads retained.
    for s, e in _runs_of_type(ss, "E"):
        if e - s + 1 < _MIN_SHEET_RES:
            continue
        items.extend(
            _strip_with_rails_items(
                ca_pos[s : e + 1],
                normals[s : e + 1],
                sheet_half,
                sheet_steps,
                sheet_fill,
                sheet_outline,
                min(max(sheet_half * scale * 0.20, 0.62), 1.30),
                rail_scale,
                style.rail_floor_px,
                scale,
                cx,
                cy,
                canvas_w,
                canvas_h,
            )
        )

    # Coil runs use round tube paths for a continuous, hand-drawn look.
    loop_stroke = max(2.0 * loop_half * scale, 1.0)
    for s, e in _runs_of_type(ss, "C"):
        if e - s + 1 < _MIN_LOOP_RES:
            continue
        items.extend(
            _loop_items(
                ca_pos[s : e + 1],
                loop_fill,
                loop_stroke,
                style,
                scale,
                cx,
                cy,
                canvas_w,
                canvas_h,
            )
        )

    # Keep arrowheads for sufficiently long beta runs.
    _sheet_sw = max(scale * 0.006 * rail_scale, style.rail_floor_px)
    for s, e in _runs_of_type(ss, "E"):
        if e - s + 1 < _MIN_SHEET_RES:
            continue
        p_base = ca_pos[e - 1]
        p_tip_ca = ca_pos[e]
        tangent = _unit(p_tip_ca - p_base)
        tip = p_tip_ca + tangent * (np.linalg.norm(p_tip_ca - p_base) * _ARROW_TIP_FRAC)
        b_base = normals[e - 1] * sheet_half * _ARROW_FLANGE
        arrow = np.array([p_base + b_base, p_base - b_base, tip])
        items.append(
            _triangle_polygon_svg(
                arrow,
                sheet_fill,
                sheet_outline,
                scale,
                cx,
                cy,
                canvas_w,
                canvas_h,
                stroke_width=_sheet_sw,
            )
        )

    return items


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def ribbon_svg_items(
    protein_data: "ProteinData | ProteinSemantics",
    cfg: "RenderConfig",
    pos: np.ndarray,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: float,
    canvas_h: float,
) -> list[tuple[float, list[str]]]:
    """Return ``(z_depth, svg_lines)`` tuples for all ribbon elements.

    Sorted ascending by z_depth so the caller can drain them into the
    main painter's-algorithm render loop (same pattern as NCI patches).
    """
    chain_ids = [cid for cid in protein_data.chains.keys() if cid not in cfg.exclude_chains]
    chain_colors = assign_chain_colors(cfg, chain_ids, style=cfg.protein_style)
    half_ribbon = cfg.ribbon_width / 2.0
    loop_half_base = cfg.loop_width / 2.0
    style = ribbon_style_profile(cfg.protein_style)

    items: list[tuple[float, list[str]]] = []

    for cid in chain_ids:
        chain = protein_data.chains[cid]
        color = chain_colors[cid]
        helix_outline = _outline_color(color, style.helix_outline_dark, style.helix_outline_light)
        sheet_outline = _outline_color(color, style.sheet_outline_dark, style.sheet_outline_light)
        loop_outline = _outline_color(color, style.loop_dark_blend, 0.0)
        for backbone_seg in _split_backbone_segments(chain.residues, pos):
            ca_pos, o_pos, ss = _extract_ca_o_ss(backbone_seg, pos)
            if len(ca_pos) < 2:
                continue
            items.extend(
                _continuous_segment_items(
                    ca_pos,
                    o_pos,
                    ss,
                    color,
                    style,
                    helix_outline,
                    sheet_outline,
                    loop_outline,
                    half_ribbon,
                    loop_half_base,
                    scale,
                    cx,
                    cy,
                    canvas_w,
                    canvas_h,
                )
            )

    items.sort(key=lambda x: x[0])
    return items


def trace_svg_items(
    protein_data: "ProteinData | ProteinSemantics",
    cfg: "RenderConfig",
    pos: np.ndarray,
    scale: float,
    cx: float,
    cy: float,
    canvas_w: float,
    canvas_h: float,
) -> list[tuple[float, list[str]]]:
    """Fallback backbone trace rendering for TRACE_ONLY confidence."""
    chain_ids = [cid for cid in protein_data.chains.keys() if cid not in cfg.exclude_chains]
    chain_colors = assign_chain_colors(cfg, chain_ids, style=cfg.protein_style)
    loop_stroke = max(cfg.loop_width * scale * 1.35, 1.2)
    style = ribbon_style_profile(cfg.protein_style)
    items: list[tuple[float, list[str]]] = []

    for cid in chain_ids:
        trace = protein_data.trace_chains.get(cid, [])
        if len(trace) < 2:
            # Fall back to CA indices from residues if explicit trace is missing.
            trace = [r.ca_index for r in protein_data.chains[cid].residues if r.ca_index is not None]
        if len(trace) < 2:
            continue
        ca_pos = np.array([pos[i] for i in trace], dtype=float)
        items.extend(_loop_items(ca_pos, chain_colors[cid], loop_stroke, style, scale, cx, cy, canvas_w, canvas_h))

    items.sort(key=lambda x: x[0])
    return items
