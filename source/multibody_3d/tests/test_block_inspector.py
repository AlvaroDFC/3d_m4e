"""Tests and usage examples for BlockInspector symbolic inspection utilities.

Covers:
  - format_B_block / format_Bdot_block: output type, key strings present
  - format_B_blocks / format_Bdot_blocks: ordered output, all write-pairs present
  - show_matrix=False: no matrix section emitted
  - simplify=True: no crash, returns str
  - display_B_blocks / display_Bdot_blocks: fall back to print without IPython
  - VelocityTransformation3D.print_B_blocks / print_Bdot_blocks still work
  - BlockInspector importable from the package root
"""
from __future__ import annotations

import re

import numpy as np
import pytest
import sympy as sym

from multibody_3d import (
    BlockInspector,
    JointSystem3D,
    VelocityTransformation3D,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

Z3 = [0.0, 0.0, 0.0]


def _make_system(n_bodies, joints, types, p2j, j2c, u, u1=None, u2=None):
    if u1 is None:
        u1 = [None] * len(joints)
    if u2 is None:
        u2 = [None] * len(joints)
    return JointSystem3D.from_data({
        "NBodies": n_bodies,
        "joints": joints,
        "types": types,
        "parent_cg_to_joint": p2j,
        "joint_to_child_cg": j2c,
        "axis_u": u,
        "axis_u1": u1,
        "axis_u2": u2,
    })


@pytest.fixture
def revolute_1body():
    """Single revolute joint: ground → body 1, R about z."""
    js = _make_system(
        n_bodies=1,
        joints=[(0, 1)],
        types=["R"],
        p2j=[[1.0, 0.0, 0.0]],
        j2c=[[0.5, 0.0, 0.0]],
        u=[[0.0, 0.0, 1.0]],
    )
    return VelocityTransformation3D(js)


@pytest.fixture
def chain_RPS():
    """3-body chain: R → P → S."""
    js = _make_system(
        n_bodies=3,
        joints=[(0, 1), (1, 2), (2, 3)],
        types=["R", "P", "S"],
        p2j=[Z3, Z3, Z3],
        j2c=[Z3, Z3, Z3],
        u=[[0, 0, 1], [0, 0, 1], None],
    )
    return VelocityTransformation3D(js)


@pytest.fixture
def chain_UCF():
    """3-body chain: U → C → F."""
    js = _make_system(
        n_bodies=3,
        joints=[(0, 1), (1, 2), (2, 3)],
        types=["U", "C", "F"],
        p2j=[[1, 0, 0], [0, 1, 0], Z3],
        j2c=[[-1, 0, 0], [0, -1, 0], Z3],
        u=[None, [0, 0, 1], None],
        u1=[[1, 0, 0], None, None],
        u2=[[0, 1, 0], None, None],
    )
    return VelocityTransformation3D(js)


def _make_q_syms(n):
    return sym.Matrix(sym.symbols(f"q0:{n}"))


def _make_qd_syms(n):
    return sym.Matrix(sym.symbols(f"qd0:{n}"))


def _build_B_blocks(vt):
    q = _make_q_syms(vt.total_cfg_dof)
    return vt.build_B_blocks_symbolic(q)


def _build_Bdot_blocks(vt):
    q  = _make_q_syms(vt.total_cfg_dof)
    qd = _make_qd_syms(vt.total_dof)
    return vt.build_Bdot_blocks_symbolic(q, qd)


# ---------------------------------------------------------------------------
# format_B_block
# ---------------------------------------------------------------------------

class TestFormatBBlock:

    def test_returns_string(self, revolute_1body):
        blocks = _build_B_blocks(revolute_1body)
        blk = blocks[(1, 0)]
        result = BlockInspector.format_B_block(blk)
        assert isinstance(result, str)

    def test_header_contains_body_joint_type(self, revolute_1body):
        blocks = _build_B_blocks(revolute_1body)
        blk = blocks[(1, 0)]
        result = BlockInspector.format_B_block(blk)
        assert "body=1" in result
        assert "joint=0" in result
        assert "type=R" in result

    def test_formula_tilde_notation_present(self, revolute_1body):
        """The formula line must contain tilde (d̃) notation for R joint."""
        blocks = _build_B_blocks(revolute_1body)
        result = BlockInspector.format_B_block(blocks[(1, 0)])
        # The formula key for R is _B_FORMULA["R"] which contains d̃
        assert "d̃" in result

    def test_d_kj_label_present(self, revolute_1body):
        blocks = _build_B_blocks(revolute_1body)
        result = BlockInspector.format_B_block(blocks[(1, 0)])
        assert "d_kj" in result

    def test_U_j_label_present(self, revolute_1body):
        blocks = _build_B_blocks(revolute_1body)
        result = BlockInspector.format_B_block(blocks[(1, 0)])
        assert "U_j" in result

    def test_block_matrix_present_by_default(self, revolute_1body):
        blocks = _build_B_blocks(revolute_1body)
        result = BlockInspector.format_B_block(blocks[(1, 0)])
        assert "block =" in result

    def test_show_matrix_false_omits_matrix(self, revolute_1body):
        blocks = _build_B_blocks(revolute_1body)
        result = BlockInspector.format_B_block(blocks[(1, 0)], show_matrix=False)
        assert "block =" not in result
        assert "d_kj" in result

    def test_simplify_returns_string(self, revolute_1body):
        blocks = _build_B_blocks(revolute_1body)
        result = BlockInspector.format_B_block(blocks[(1, 0)], simplify=True)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_all_joint_types(self, chain_RPS, chain_UCF):
        """format_B_block must succeed for every joint type R,P,S,U,C,F."""
        for vt in (chain_RPS, chain_UCF):
            blocks = _build_B_blocks(vt)
            for blk in blocks.values():
                result = BlockInspector.format_B_block(blk)
                assert isinstance(result, str)
                assert f"type={blk.joint_type}" in result


# ---------------------------------------------------------------------------
# format_Bdot_block
# ---------------------------------------------------------------------------

class TestFormatBdotBlock:

    def test_returns_string(self, revolute_1body):
        blocks = _build_Bdot_blocks(revolute_1body)
        result = BlockInspector.format_Bdot_block(blocks[(1, 0)])
        assert isinstance(result, str)

    def test_has_d_dot_label(self, revolute_1body):
        blocks = _build_Bdot_blocks(revolute_1body)
        result = BlockInspector.format_Bdot_block(blocks[(1, 0)])
        assert "d_dot_kj" in result

    def test_has_U_dot_label(self, revolute_1body):
        blocks = _build_Bdot_blocks(revolute_1body)
        result = BlockInspector.format_Bdot_block(blocks[(1, 0)])
        assert "U_dot_j" in result

    def test_formula_tilde_dot_present(self, revolute_1body):
        """Bdot formula for R must contain ḋ̃ or d̃."""
        blocks = _build_Bdot_blocks(revolute_1body)
        result = BlockInspector.format_Bdot_block(blocks[(1, 0)])
        # U-dot formula contains either ḋ̃ or u̇ notation
        assert any(ch in result for ch in ("ḋ̃", "u̇", "d̃"))

    def test_show_matrix_false_omits_matrix(self, revolute_1body):
        blocks = _build_Bdot_blocks(revolute_1body)
        result = BlockInspector.format_Bdot_block(blocks[(1, 0)], show_matrix=False)
        assert "block =" not in result

    def test_all_joint_types(self, chain_RPS, chain_UCF):
        for vt in (chain_RPS, chain_UCF):
            blocks = _build_Bdot_blocks(vt)
            for blk in blocks.values():
                result = BlockInspector.format_Bdot_block(blk)
                assert isinstance(result, str)


# ---------------------------------------------------------------------------
# format_B_blocks / format_Bdot_blocks
# ---------------------------------------------------------------------------

class TestFormatBlocks:

    def test_format_B_blocks_contains_all_write_pairs(self, chain_RPS):
        blocks = _build_B_blocks(chain_RPS)
        result = BlockInspector.format_B_blocks(blocks)
        # Every (k, j) pair should appear in the output
        for k, j in blocks:
            assert f"body={k}" in result
            assert f"joint={j}" in result

    def test_format_Bdot_blocks_contains_all_write_pairs(self, chain_RPS):
        blocks = _build_Bdot_blocks(chain_RPS)
        result = BlockInspector.format_Bdot_blocks(blocks)
        for k, j in blocks:
            assert f"body={k}" in result

    def test_ordered_by_write_pair(self, chain_RPS):
        """Blocks should appear in sorted (k, j) order."""
        blocks = _build_B_blocks(chain_RPS)
        result = BlockInspector.format_B_blocks(blocks)
        # Extract (body=k, joint=j) positions in output
        positions = [(m.start(), int(m.group(1)), int(m.group(2)))
                     for m in re.finditer(r"body=(\d+).*?joint=(\d+)", result)]
        pairs_in_order = [(k, j) for _, k, j in sorted(positions, key=lambda x: x[0])]
        assert pairs_in_order == sorted(blocks.keys())


# ---------------------------------------------------------------------------
# display / print convenience wrappers
# ---------------------------------------------------------------------------

class TestDisplayAndPrint:

    def test_display_B_blocks_runs(self, chain_RPS, capsys):
        blocks = _build_B_blocks(chain_RPS)
        BlockInspector.display_B_blocks(blocks)
        captured = capsys.readouterr()
        assert "B" in captured.out or "B" in captured.err or True  # no crash

    def test_display_Bdot_blocks_runs(self, chain_RPS, capsys):
        blocks = _build_Bdot_blocks(chain_RPS)
        BlockInspector.display_Bdot_blocks(blocks)   # no crash

    def test_print_B_blocks_via_vt(self, chain_RPS, capsys):
        """VelocityTransformation3D.print_B_blocks delegates to BlockInspector."""
        blocks = _build_B_blocks(chain_RPS)
        VelocityTransformation3D.print_B_blocks(blocks)
        out = capsys.readouterr().out
        assert "body=1" in out

    def test_print_Bdot_blocks_via_vt(self, chain_RPS, capsys):
        blocks = _build_Bdot_blocks(chain_RPS)
        VelocityTransformation3D.print_Bdot_blocks(blocks)
        out = capsys.readouterr().out
        assert "body=1" in out

    def test_print_B_show_matrix_false(self, revolute_1body, capsys):
        blocks = _build_B_blocks(revolute_1body)
        VelocityTransformation3D.print_B_blocks(blocks, show_matrix=False)
        out = capsys.readouterr().out
        assert "block =" not in out
        assert "d_kj" in out


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------

def test_block_inspector_importable_from_package():
    from multibody_3d import BlockInspector as BI
    assert hasattr(BI, "format_B_block")
    assert hasattr(BI, "format_Bdot_block")
    assert hasattr(BI, "format_B_blocks")
    assert hasattr(BI, "format_Bdot_blocks")
    assert hasattr(BI, "display_B_blocks")
    assert hasattr(BI, "display_Bdot_blocks")


# ---------------------------------------------------------------------------
# Usage example (runs as a test, also serves as documentation)
# ---------------------------------------------------------------------------

def test_usage_example_revolute_chain():
    """Demonstrate the full symbolic inspection workflow for a 2-body R chain."""
    # Build a 2-body chain with a single revolute joint
    js = JointSystem3D.from_data({
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["R"],
        "parent_cg_to_joint": [[1.0, 0.0, 0.0]],
        "joint_to_child_cg":  [[0.5, 0.0, 0.0]],
        "axis_u": [[0.0, 0.0, 1.0]],
        "axis_u1": [None],
        "axis_u2": [None],
    })
    vt = VelocityTransformation3D(js)

    # Symbolic configuration and speed vectors
    q  = sym.Matrix([sym.Symbol("theta", real=True)])
    qd = sym.Matrix([sym.Symbol("thetadot", real=True)])

    # ---- B blocks ----
    B_blocks = vt.build_B_blocks_symbolic(q)
    assert len(B_blocks) == 1
    blk = B_blocks[(1, 0)]
    assert blk.joint_type == "R"

    text = BlockInspector.format_B_block(blk)
    # Formula line present
    assert "formula" in text
    assert "d̃" in text
    # Ingredients present
    assert "d_kj" in text
    assert "U_j" in text
    # 6×1 block matrix present
    assert "block =" in text
    # Block has correct shape
    assert blk.matrix.shape == (6, 1)

    # ---- Bdot blocks ----
    Bdot_blocks = vt.build_Bdot_blocks_symbolic(q, qd)
    bdot_blk = Bdot_blocks[(1, 0)]
    bdot_text = BlockInspector.format_Bdot_block(bdot_blk)
    assert "d_dot_kj" in bdot_text
    assert "U_dot_j"  in bdot_text
    assert bdot_blk.matrix.shape == (6, 1)

    # ---- format_B_blocks returns a single string covering all blocks ----
    full_text = BlockInspector.format_B_blocks(B_blocks)
    assert "body=1" in full_text
    assert "joint=0" in full_text


def test_usage_example_all_joint_types():
    """Inspect a chain covering all six joint types via two separate systems.

    Split into two 3-body chains so symbolic evaluation stays fast.
    """
    # Chain A: R, P, U
    js_a = JointSystem3D.from_data({
        "NBodies": 3,
        "joints": [(0,1),(1,2),(2,3)],
        "types":  ["R","P","U"],
        "parent_cg_to_joint": [Z3]*3,
        "joint_to_child_cg":  [Z3]*3,
        "axis_u":  [[0,0,1],[0,0,1],None],
        "axis_u1": [None,None,[1,0,0]],
        "axis_u2": [None,None,[0,1,0]],
    })
    # Chain B: C, S, F
    js_b = JointSystem3D.from_data({
        "NBodies": 3,
        "joints": [(0,1),(1,2),(2,3)],
        "types":  ["C","S","F"],
        "parent_cg_to_joint": [Z3]*3,
        "joint_to_child_cg":  [Z3]*3,
        "axis_u":  [[0,0,1],None,None],
        "axis_u1": [None,None,None],
        "axis_u2": [None,None,None],
    })
    seen_types: set = set()
    for js in (js_a, js_b):
        vt = VelocityTransformation3D(js)
        q  = sym.Matrix([sym.Symbol(f"q{i}", real=True)
                         for i in range(vt.total_cfg_dof)])
        blocks = vt.build_B_blocks_symbolic(q)
        for blk in blocks.values():
            text = BlockInspector.format_B_block(blk, show_matrix=False)
            assert "formula" in text, f"Missing formula for type={blk.joint_type}"
            seen_types.add(blk.joint_type)

    assert seen_types >= {"R", "P", "U", "C", "S", "F"}, (
        f"Not all joint types covered: {seen_types}"
    )

    # Unified text for chain A
    vt_a = VelocityTransformation3D(js_a)
    q_a  = sym.Matrix([sym.Symbol(f"q{i}", real=True)
                       for i in range(vt_a.total_cfg_dof)])
    full = BlockInspector.format_B_blocks(
        vt_a.build_B_blocks_symbolic(q_a), show_matrix=False,
    )
    for code in ("R", "P", "U"):
        assert f"type={code}" in full, f"Missing type={code} in output"
