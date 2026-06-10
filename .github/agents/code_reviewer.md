# Agent instructions

Before making changes:

1. Review instruction from backend-code-correctness-review.prompt.md
2. Inspect the relevant files and identify the smallest safe change.
3. Prefer a report before large refactors.
4. Run or propose appropriate validation:
   - `ruff check`
   - `black --check`
   - `pytest`
   - targeted example scripts when relevant
5. Do not modify symbolic equations, B/Bdot formulas, coordinate mappings, or JAX evaluator logic unless the requested task requires it.
6. For dead-code cleanup, classify each deletion as:
   - safe,
   - likely safe but needs confirmation,
   - unsafe without tests.
7. Preserve local assignment alignment within logical assignment blocks when editing files that already use that style.