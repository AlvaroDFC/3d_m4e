import sympy as sym
from dataclasses import dataclass
from typing import List
from .joint_system_3d import JointType

'''
This file contains static methods and helper functions for velocity transformation computations, including:
- skew(): 3x3 skew-symmetric matrix of a 3x1 vector
- _axis_angle_rotation(): Rodrigues rotation matrix from axis-angle parameters
- _exp_rotation(): Rotation matrix from exponential coordinates'''

@dataclass(frozen=True, slots=True)
class RootToLeafPaths:
    """Root-to-leaf paths cache."""
    body_paths: List[List[int]]   # [[b1,b2,...], ...] excluding ground
    joint_paths: List[List[int]]  # [[j(b1),j(b2),...], ...] aligned with body_paths

def skew(v: sym.Matrix) -> sym.Matrix:
    """Return the 3x3 skew-symmetric matrix of a 3x1 vector."""
    if v.shape != (3, 1):
        raise ValueError(f"skew: expected (3,1), got {v.shape}.")
    x, y, z = v[0, 0], v[1, 0], v[2, 0]
    return sym.Matrix([
        [   0, -z,  y],
        [   z,  0, -x],
        [  -y,  x,  0],
    ])

def _type_code(jt: JointType) -> str:
    """Return the 1-letter joint code ('R','P','S','U','C','F')."""
    # Prefer enum value if it is the one-letter code
    v = getattr(jt, "value", jt)
    if isinstance(v, str) and len(v) == 1:
        return v
    # Fallback: enum name if it matches
    n = getattr(jt, "name", str(v))
    if isinstance(n, str) and len(n) == 1:
        return n
    return str(v)


def _axis_angle_rotation(u: sym.Matrix, theta: sym.Expr) -> sym.Matrix:
    """Rodrigues rotation matrix: rotation by *theta* about unit axis *u* (3x1).

    .. math::

        R = I + \\sin\\theta\\, K + (1 - \\cos\\theta)\\, K^2

    where :math:`K = \\mathrm{skew}(u)`.
    """
    K = skew(u)
    return sym.eye(3) + sym.sin(theta) * K + (1 - sym.cos(theta)) * K * K


def _A_from_quaternion_sym(e0, e) -> sym.Matrix:
    """Rotation matrix from unit quaternion ``[e0,e1,e2,e3] = [w,x,y,z]``.

    Returns a 3×3 sympy Matrix (polynomial in quaternion components).
    """
    e = sym.Matrix(e)
    A = (2*e0**2 - 1) * sym.eye(3) + 2 * (e * e.T + e0 * skew(e))

    return A