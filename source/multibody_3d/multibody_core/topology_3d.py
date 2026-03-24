# source/multibody/topology_3d.py
"""
topology_3d.py

Pure topology utilities for 3D joint-coordinate multibody systems.

Conventions
-----------
- Bodies: ground = 0, bodies = 1..NBodies (1-based).
- Edges: each joint defines a directed edge (parent -> child).
- The system is a TREE rooted at ground:
    * every body 1..NBodies appears exactly once as a child
    * parent is 0 or 1..NBodies
    * no cycles
    * all bodies reachable from ground

Root-to-leaf paths
------------------
Returned body paths are sequences of BODY ids excluding ground:
    [root_child, ..., leaf]

Returned joint paths are sequences of JOINT INDICES (0-based, into the joint list)
aligned with body paths:
    joint_path[i] is the joint that connects parent(body_path[i]) -> body_path[i]

Btrack
------
Btrack[body, j] is True if body is downstream of joint j, i.e., joint j lies on the
unique path from ground to 'body'. (Ancestors-of-body indicator per joint.)
Shape: (NBodies+1, NJoints). Row 0 is all False.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import sympy as sym


BodyId = int
JointIndex = int
Edge = Tuple[BodyId, BodyId]  # (parent, child)


def build_adjacency(edges: Sequence[Edge], NBodies: int, *, ground_id: int = 0) -> Dict[int, List[int]]:
    """
    Build adjacency dict: parent_body -> sorted list of child bodies.
    Ensures all nodes exist as keys (0..NBodies) even if leaf.
    """
    adj: Dict[int, List[int]] = {i: [] for i in range(ground_id, NBodies + 1)}
    for p, c in edges:
        adj.setdefault(p, []).append(c)
        adj.setdefault(c, [])  # ensure key exists
    for k in adj:
        adj[k] = sorted(adj[k])
    return adj


@dataclass(frozen=True)
class TreeIndex:
    """
    Cached parent/joint index arrays for a rooted tree.

    parent_body_of_body[b]  = parent body id (0 for root children)
    parent_joint_of_body[b] = joint index (0-based) that connects parent->b
    child_to_joint[b]       = same as parent_joint_of_body[b]
    """
    parent_body_of_body: List[int]      # len NBodies+1, index 0 valid (0)
    parent_joint_of_body: List[int]     # len NBodies+1, index 0 = -1
    child_to_joint: List[int]           # len NBodies+1, index 0 = -1


def validate_tree(edges: Sequence[Edge], NBodies: int, *, ground_id: int = 0) -> TreeIndex:
    """
    Validate edges define a single rooted tree at ground.

    Raises ValueError with descriptive messages on failure.
    Returns TreeIndex on success.
    """
    if NBodies < 1:
        raise ValueError(f"NBodies must be >= 1. Got NBodies={NBodies}.")

    if len(edges) != NBodies:
        raise ValueError(
            f"Invalid number of joints/edges: expected NBodies={NBodies} edges "
            f"(one parent joint per body), got {len(edges)}."
        )

    # child uniqueness + range checks
    child_to_edge_idxs: Dict[int, List[int]] = {}
    for j, (p, c) in enumerate(edges):
        if not (1 <= c <= NBodies):
            raise ValueError(f"Joint {j}: child={c} out of range [1..{NBodies}].")
        if not (p == ground_id or 1 <= p <= NBodies):
            raise ValueError(f"Joint {j}: parent={p} out of range {{0}}∪[1..{NBodies}].")
        if p == c:
            raise ValueError(f"Joint {j}: self-parenting detected (parent==child=={c}).")
        child_to_edge_idxs.setdefault(c, []).append(j)

    duplicates = {c: js for c, js in child_to_edge_idxs.items() if len(js) > 1}
    if duplicates:
        parts = ", ".join([f"body {c} in joints {js}" for c, js in sorted(duplicates.items())])
        raise ValueError(f"Duplicate child body detected (each body must have exactly one parent joint): {parts}.")

    missing = [b for b in range(1, NBodies + 1) if b not in child_to_edge_idxs]
    if missing:
        raise ValueError(f"Missing child body indices (each body 1..NBodies must appear once as a child): {missing}.")

    # build parent pointers
    parent_body_of_body = [ground_id] * (NBodies + 1)
    parent_joint_of_body = [-1] * (NBodies + 1)
    child_to_joint = [-1] * (NBodies + 1)

    for j, (p, c) in enumerate(edges):
        parent_body_of_body[c] = p
        parent_joint_of_body[c] = j
        child_to_joint[c] = j

    # cycle detection by ancestor-walk (guaranteed termination iff reaches ground)
    for b in range(1, NBodies + 1):
        seen = set()
        cur = b
        chain = []
        while cur != ground_id:
            if cur in seen:
                chain_str = " -> ".join(map(str, chain + [cur]))
                raise ValueError(f"Cycle detected while tracing ancestors of body {b}: {chain_str}.")
            seen.add(cur)
            chain.append(cur)
            cur = parent_body_of_body[cur]

    # reachability: DFS/BFS from ground
    adj = build_adjacency(edges, NBodies, ground_id=ground_id)
    visited = set([ground_id])
    stack = [ground_id]
    while stack:
        n = stack.pop()
        for ch in adj.get(n, []):
            if ch not in visited:
                visited.add(ch)
                stack.append(ch)

    unreachable = [b for b in range(1, NBodies + 1) if b not in visited]
    if unreachable:
        raise ValueError(
            f"Disconnected system: bodies not reachable from ground {ground_id}: {unreachable}. "
            f"Ensure the graph is a single tree rooted at ground."
        )

    return TreeIndex(
        parent_body_of_body=parent_body_of_body,
        parent_joint_of_body=parent_joint_of_body,
        child_to_joint=child_to_joint,
    )

def compute_root_to_leaf_joint_paths(
    adjacency: Dict[int, List[int]], child_to_joint: Sequence[int], *, root: int = 0
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Return (body_paths, joint_paths) for all root-to-leaf branches.

    body_paths:  [ [b1, b2, ...], ... ]
    joint_paths: [ [j(b1), j(b2), ...], ... ]  where j(bk) is the parent joint index of bk.
    """
    body_paths: List[List[int]] = []
    joint_paths: List[List[int]] = []

    def dfs(body: int, bp: List[int], jp: List[int]) -> None:
        bp2 = bp + [body]
        jp2 = jp + [child_to_joint[body]]
        children = adjacency.get(body, [])
        if not children:
            body_paths.append(bp2)
            joint_paths.append(jp2)
            return
        for ch in children:
            dfs(ch, bp2, jp2)

    for ch in adjacency.get(root, []):
        dfs(ch, [], [])
    return body_paths, joint_paths


def compute_Btrack(
    parent_joint_of_body: Sequence[int],
    parent_body_of_body: Sequence[int],
    NBodies: int,
    NJoints: int,
    *,
    ground_id: int = 0,
) -> np.ndarray:
    """
    Compute ancestor-joint indicator matrix.

    Btrack[b, j] = True if body b is downstream of joint j, i.e., joint j is on the
    path ground -> ... -> b.

    Shape: (NBodies+1, NJoints). Row 0 is all False.
    """
    Btrack = np.zeros((NBodies + 1, NJoints), dtype=bool)
    for b in range(1, NBodies + 1):
        cur = b
        while cur != ground_id:
            j = parent_joint_of_body[cur]
            if j < 0:
                break
            Btrack[b, j] = True
            cur = parent_body_of_body[cur]
    return Btrack

def to_display_value(x, nd=3):
    # SymPy scalar
    if isinstance(x, sym.Basic):
        if x.is_number:
            try:
                return np.round(float(sym.N(x)), nd)
            except Exception:
                return str(x)
        return str(x)

    # Python numeric
    if isinstance(x, (float, np.number)):
        return np.round(float(x), nd)

    # Vector/list-like
    if isinstance(x, (list, tuple)):
        return [to_display_value(v, nd) for v in x]

    return x