"""_velocity_transformation_inspector.py

Pure symbolic inspection and display utilities for B and Bdot block dicts.

Moved out of ``velocity_transformation_3d.py`` to keep display / debugging
concerns separate from the mathematical computation code.  Keeping it in a
dedicated file also prevents the JAX backend section of
``_velocity_transformation_helper.py`` from being polluted with IPython /
SymPy-printer logic.

The only public symbol is :class:`BlockInspector`.  Import it via the main
module (``from .velocity_transformation_3d import BlockInspector``) or
directly (``from ._velocity_transformation_inspector import BlockInspector``).
"""
from __future__ import annotations

import sympy as sym

from ._velocity_transformation_helper import (
    SymbolicBBlock,
    SymbolicBdotBlock,
    WritePair,
)


class BlockInspector:
    """Lightweight symbolic inspection utilities for B and Bdot block dicts.

    All methods are *pure* (return strings, never mutate state) so they can be
    composed freely or called from a Jupyter notebook cell.

    Quick-start
    -----------
    ::

        vt     = VelocityTransformation3D(js)
        q_syms = sym.Matrix(sym.symbols(f"q0:{vt.total_cfg_dof}"))
        blocks = vt.build_B_blocks_symbolic(q_syms)

        # 1. Pretty-print all blocks at once:
        BlockInspector.display_B_blocks(blocks)

        # 2. Inspect a single block:
        print(BlockInspector.format_B_block(blocks[(1, 0)]))

        # 3. Just show ingredients (d, U) without the full matrix:
        print(BlockInspector.format_B_block(blocks[(1, 0)], show_matrix=False))

        # 4. Optionally apply trig simplification to matrix entries:
        print(BlockInspector.format_B_block(blocks[(1, 0)], simplify=True))

    Compact formula notation (tilde = skew-symmetric matrix of a vector)
    ---------------------------------------------------------------------
    B blocks, by joint type::

        R  :  [ −d̃·u ]   [ u ]ᵀ          (1 speed DOF)
        P  :  [   u  ]   [ 0 ]ᵀ            (1 speed DOF)
        U  :  [ −d̃·U ]   [ U ]ᵀ          (2 speed DOFs, U=[u₁|u₂])
        C  :  [ −d̃·u  u ]  /  [ u  0 ]ᵀ  (2 speed DOFs)
        S  :  [  −d̃  ]   [  I ]ᵀ          (3 speed DOFs, U=Aₚ)
        F  :  [  I  −d̃ ]  /  [ 0   I ]ᵀ  (6 speed DOFs)

    Bdot blocks replace d→d, U→U with their time-derivatives ḋ, U̇.

    Limitations
    -----------
    * SymPy's ``pretty()`` printer is used for matrices.  Expressions
      involving unevaluated ``MatMul`` or ``MatrixSymbol`` objects (e.g., the
      opaque ``Arel_{j}`` rotation symbols in the position cache) will print
      as their symbolic names rather than numerical 3×3 blocks.
    * ``simplify=True`` calls ``sym.trigsimp`` entry-by-entry, which can be
      *slow* for large matrices or deeply nested expressions; use it
      interactively, not in hot loops.
    * The tilde notation (``d̃ = skew(d)``) is shown in the *formula header*
      only; entries of the block matrix are always the fully expanded SymPy
      expressions.  No attempt is made to factor them back to tilde form.
    """

    # ---- compact formula strings per joint type ----

    _B_FORMULA: dict[str, str] = {
        "R": "⎡ −d̃·u ⎤  (top 3 rows)   u: axis (3×1)",
        "P": "⎡   u  ⎤  (top 3 rows)   u: axis (3×1)",
        "U": "⎡ −d̃·U ⎤  (top 3 rows)   U=[u₁|u₂]: basis (3×2)",
        "C": "⎡ −d̃·u  u ⎤  (top)   u: axis;  DOFs=[θ,s]",
        "S": "⎡  −d̃  ⎤  (top 3 rows)   U=Aₚ: parent rotation (3×3)",
        "F": "⎡  I   −d̃ ⎤  (top)   DOFs=[t₁t₂t₃ ω₁ω₂ω₃]",
    }
    _B_FORMULA_BOT: dict[str, str] = {
        "R": "⎣   u  ⎦  (bot 3 rows)",
        "P": "⎣   0  ⎦  (bot 3 rows)",
        "U": "⎣   U  ⎦  (bot 3 rows)",
        "C": "⎣   u   0 ⎦  (bot)",
        "S": "⎣   I  ⎦  (bot 3 rows)",
        "F": "⎣  0    I  ⎦  (bot)",
    }
    _BDOT_FORMULA: dict[str, str] = {
        "R": "⎡ −ḋ̃·u − d̃·u̇ ⎤  (top 3 rows)",
        "P": "⎡      u̇      ⎤  (top 3 rows)",
        "U": "⎡ −ḋ̃·U − d̃·U̇ ⎤  (top 3 rows)",
        "C": "⎡ −ḋ̃·u − d̃·u̇  u̇ ⎤  (top)",
        "S": "⎡     −ḋ̃      ⎤  (top 3 rows)",
        "F": "⎡   0    −ḋ̃   ⎤  (top)",
    }
    _BDOT_FORMULA_BOT: dict[str, str] = {
        "R": "⎣      u̇      ⎦  (bot 3 rows)",
        "P": "⎣       0     ⎦  (bot 3 rows)",
        "U": "⎣      U̇      ⎦  (bot 3 rows)",
        "C": "⎣      u̇   0  ⎦  (bot)",
        "S": "⎣       0     ⎦  (bot 3 rows)",
        "F": "⎣   0     0   ⎦  (bot)",
    }

    # ---- internal helpers ---------------------------------------------------

    @staticmethod
    def _pretty_mat(M: sym.Matrix, simplify: bool = False) -> str:
        """Return a SymPy ``pretty()`` string for matrix *M*.

        When *simplify* is True each entry is passed through
        ``sym.trigsimp`` before printing — useful for angle/quaternion
        expressions but can be slow.
        """
        if simplify:
            M = M.applyfunc(sym.trigsimp)
        return sym.pretty(M, use_unicode=True)

    @staticmethod
    def _vec_row(label: str, v: sym.Matrix, simplify: bool) -> str:
        """One-line representation of a (3,1) or (3,m) matrix with a label."""
        if v.shape == (3, 1):
            displayed = v.T   # show as row for compactness
        else:
            displayed = v
        return f"  {label} = {BlockInspector._pretty_mat(displayed, simplify)}"

    # ---- public API ---------------------------------------------------------

    @staticmethod
    def format_B_block(
        blk: SymbolicBBlock,
        *,
        simplify: bool = False,
        show_matrix: bool = True,
    ) -> str:
        """Format a single symbolic B block as a human-readable string.

        Parameters
        ----------
        blk : SymbolicBBlock
        simplify : bool
            Apply ``sym.trigsimp`` to each matrix entry before printing.
            Useful for Rodrigues/quaternion expressions; can be slow.
        show_matrix : bool
            When *False*, print only the ingredients (``d_kj``, ``U_j``)
            without the assembled 6×m block matrix.
        """
        code = blk.joint_type
        sep  = "─" * 60
        lines = [
            sep,
            f"B  block  body={blk.body_id}  joint={blk.joint_index}  "
            f"type={code}  "
            f"rows {blk.row_slice.start}:{blk.row_slice.stop}  "
            f"cols {blk.col_slice.start}:{blk.col_slice.stop}",
            f"  formula (top) : {BlockInspector._B_FORMULA.get(code, '?')}",
            f"  formula (bot) : {BlockInspector._B_FORMULA_BOT.get(code, '?')}",
            BlockInspector._vec_row("d_kj", blk.d_kj, simplify),
            BlockInspector._vec_row("U_j ", blk.U_j,  simplify),
        ]
        if show_matrix:
            lines.append("  block =")
            lines.append(
                "\n".join("    " + ln for ln in
                          BlockInspector._pretty_mat(blk.matrix, simplify).splitlines())
            )
        return "\n".join(lines)

    @staticmethod
    def format_Bdot_block(
        blk: SymbolicBdotBlock,
        *,
        simplify: bool = False,
        show_matrix: bool = True,
    ) -> str:
        """Format a single symbolic Bdot block as a human-readable string."""
        code = blk.joint_type
        sep  = "─" * 60
        lines = [
            sep,
            f"Ḃ  block  body={blk.body_id}  joint={blk.joint_index}  "
            f"type={code}  "
            f"rows {blk.row_slice.start}:{blk.row_slice.stop}  "
            f"cols {blk.col_slice.start}:{blk.col_slice.stop}",
            f"  formula (top) : {BlockInspector._BDOT_FORMULA.get(code, '?')}",
            f"  formula (bot) : {BlockInspector._BDOT_FORMULA_BOT.get(code, '?')}",
            BlockInspector._vec_row("d_kj    ", blk.d_kj,     simplify),
            BlockInspector._vec_row("d_dot_kj", blk.d_dot_kj, simplify),
            BlockInspector._vec_row("U_j     ", blk.U_j,      simplify),
            BlockInspector._vec_row("U_dot_j ", blk.U_dot_j,  simplify),
        ]
        if show_matrix:
            lines.append("  block =")
            lines.append(
                "\n".join("    " + ln for ln in
                          BlockInspector._pretty_mat(blk.matrix, simplify).splitlines())
            )
        return "\n".join(lines)

    @staticmethod
    def format_B_blocks(
        blocks: "dict[WritePair, SymbolicBBlock]",
        *,
        simplify: bool = False,
        show_matrix: bool = True,
    ) -> str:
        """Format all B blocks in write-pair order, separated by blank lines."""
        return "\n\n".join(
            BlockInspector.format_B_block(blk, simplify=simplify, show_matrix=show_matrix)
            for blk in (blocks[k] for k in sorted(blocks))
        )

    @staticmethod
    def format_Bdot_blocks(
        blocks: "dict[WritePair, SymbolicBdotBlock]",
        *,
        simplify: bool = False,
        show_matrix: bool = True,
    ) -> str:
        """Format all Bdot blocks in write-pair order, separated by blank lines."""
        return "\n\n".join(
            BlockInspector.format_Bdot_block(blk, simplify=simplify, show_matrix=show_matrix)
            for blk in (blocks[k] for k in sorted(blocks))
        )

    @staticmethod
    def display_B_blocks(
        blocks: "dict[WritePair, SymbolicBBlock]",
        *,
        simplify: bool = False,
        show_matrix: bool = True,
    ) -> None:
        """Print all B blocks.  In Jupyter, attempts LaTeX rendering via IPython."""
        text = BlockInspector.format_B_blocks(
            blocks, simplify=simplify, show_matrix=show_matrix,
        )
        BlockInspector._output(text)

    @staticmethod
    def display_Bdot_blocks(
        blocks: "dict[WritePair, SymbolicBdotBlock]",
        *,
        simplify: bool = False,
        show_matrix: bool = True,
    ) -> None:
        """Print all Bdot blocks.  In Jupyter, attempts LaTeX rendering via IPython."""
        text = BlockInspector.format_Bdot_blocks(
            blocks, simplify=simplify, show_matrix=show_matrix,
        )
        BlockInspector._output(text)

    @staticmethod
    def _output(text: str) -> None:
        """Print *text*, using ``IPython.display`` when running inside Jupyter."""
        try:
            from IPython.display import display as _ipy_display  # type: ignore
            from IPython.display import Markdown as _Markdown      # type: ignore
            _ipy_display(_Markdown("```\n" + text + "\n```"))
        except Exception:
            print(text)
