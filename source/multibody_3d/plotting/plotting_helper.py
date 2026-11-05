"""plotting_helper.py

Matplotlib-free polyline builders for the spring, damper, torsion and force-arrow
symbols drawn by ``plotting_main``.

Every builder takes arrays with leading axes ``(F, n)`` (frames, elements) and
returns line segments shaped ``(F, n * k, 2, 3)``, each segment being
``[start, end]``.  The symbols are drawn in the plane spanned by the element
axis and a perpendicular vector from :func:`perpendicular_frame`.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def unit_direction(a, b):
    """Unit vector from *a* to *b*, shape ``(F, n, 3)``; zero where they coincide."""
    ab = b - a
    return ab / np.maximum(np.linalg.norm(ab, axis=-1, keepdims=True), _EPS)


def reference_axes(unit):
    """World axis (0, 1, 2) per element that stays furthest from *unit* in every frame.

    *unit* is ``(F, n, 3)``; pass the unit vectors of all frames of the result so
    a snapshot orients its symbols like the animation does.  Non-finite frames
    are ignored.
    """
    unit = np.nan_to_num(unit, nan=0.0, posinf=0.0, neginf=0.0)
    return np.argmin(np.abs(unit).max(axis=0), axis=-1)


def perpendicular_frame(unit, ref):
    """Unit vectors ``u, v`` perpendicular to *unit*, with ``v = unit x u``.

    ``u`` is the projection of world axis *ref* (one index per element) onto the
    plane normal to *unit*, so it varies smoothly while *unit* moves.
    """
    e = np.eye(3)[ref]
    u = e - np.sum(e * unit, axis=-1, keepdims=True) * unit
    u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), _EPS)
    return u, np.cross(unit, u)


def _polyline_segments(points):
    """Polylines ``(F, n, M, 3)`` -> segments ``(F, n * (M - 1), 2, 3)``."""
    return np.stack([points[:, :, :-1], points[:, :, 1:]], axis=3).reshape(points.shape[0], -1, 2, 3)


def _stacked_segments(pairs, n_frames):
    """List of ``(start, end)`` arrays ``(F, n, 3)`` -> segments ``(F, n * len(pairs), 2, 3)``."""
    segs = np.stack([np.stack(pair, axis=-2) for pair in pairs], axis=2)
    return segs.reshape(n_frames, -1, 2, 3)


def coil_segments(a, b, u, v, radius, turns=7, steps=10, lead=0.12):
    """Helical spring from *a* to *b*, wound around the a-b axis in the plane of *u*, *v*."""
    s     = np.linspace(0.0, 1.0, turns * steps + 1)
    along = np.concatenate([[0.0], lead + (1.0 - 2.0 * lead) * s, [1.0]])
    angle = np.concatenate([[0.0], 2.0 * np.pi * turns * s, [0.0]])
    gain  = np.concatenate([[0.0], np.ones_like(s), [0.0]])          # 0 on the two lead wires
    wind  = gain[:, None] * (np.cos(angle)[:, None] * u[:, :, None, :]
                             + np.sin(angle)[:, None] * v[:, :, None, :])
    points = a[:, :, None, :] + along[:, None] * (b - a)[:, :, None, :] + radius * wind
    return _polyline_segments(points)


def dashpot_segments(a, b, u, size):
    """Dashpot between *a* and *b*: lead wire, cylinder open towards *b*, piston and rod.

    The cylinder is centred on the midpoint and spans *u* by ``+/- size``.
    """
    ab     = b - a
    length = np.linalg.norm(ab, axis=-1, keepdims=True)
    d      = ab / np.maximum(length, _EPS)
    box    = np.minimum(2.0 * size, 0.6 * length)           # cylinder length
    base   = a + 0.5 * (length - box) * d                   # closed end of the cylinder
    piston = base + 0.55 * box * d
    w      = size * u
    return _stacked_segments([
        (a, base),
        (base - w, base + w),
        (base - w, base - w + box * d),
        (base + w, base + w + box * d),
        (piston - 0.5 * w, piston + 0.5 * w),
        (piston, b),
    ], a.shape[0])


def ring_segments(centre, e1, e2, radius, steps=32):
    """Circle of *radius* around *centre* in the plane of *e1*, *e2*."""
    angle  = np.linspace(0.0, 2.0 * np.pi, steps + 1)
    points = centre[:, :, None, :] + radius * (np.cos(angle)[:, None] * e1[:, :, None, :]
                                               + np.sin(angle)[:, None] * e2[:, :, None, :])
    return _polyline_segments(points)


def tangent_dashpot_segments(centre, e1, e2, radius):
    """Small dashpot on a ring of *radius*, laid along the tangent *e2* with its plates across *e1*."""
    def on_ring(angle):
        return centre + radius * (np.cos(angle) * e1 + np.sin(angle) * e2)

    p3, p4 = on_ring(np.deg2rad(-10.0)), on_ring(np.deg2rad(10.0))
    big, small, cap = radius * e1, 0.5 * radius * e1, 0.4 * radius * e2
    return _stacked_segments([
        (p3 - big, p3 + big),
        (p4 - small, p4 + small),
        (p3 - big, p3 - big + cap),
        (p3 + big, p3 + big + cap),
    ], centre.shape[0])


def arrow_segments(origin, vec, u, v, head):
    """Arrows from *origin* along *vec*: a shaft plus four barbs at the tip.

    *u*, *v* span the plane across the arrow; *head* caps the barb length.
    """
    length = np.linalg.norm(vec, axis=-1, keepdims=True)
    d      = vec / np.maximum(length, _EPS)
    tip    = origin + vec
    size   = np.minimum(0.3 * length, head)
    base   = tip - size * d
    w      = 0.35 * size
    return _stacked_segments([
        (origin, tip),
        (tip, base + w * u),
        (tip, base - w * u),
        (tip, base + w * v),
        (tip, base - w * v),
    ], origin.shape[0])
