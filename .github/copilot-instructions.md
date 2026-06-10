# Repository instructions for Copilot

This repository implements a symbolic and numerical multibody dynamics framework.

Follow these rules:

- Preserve public APIs unless explicitly asked to change them.
- Keep topology, joint definitions, symbolic kinematics, JAX runtime evaluation, points/forces, and inspection utilities separated.
- Do not mix display/debug logic into numerical or symbolic runtime modules.
- Avoid aggressive symbolic operations such as `expand`, `simplify`, or `trigsimp` in construction/runtime paths unless justified.
- Do not rebuild static topology, geometry, lambdified functions, or JAX JIT evaluators inside repeated numerical calls.
- Prefer compact symbolic/block-level expressions over fully expanded symbolic matrices.
- Treat 2D code as legacy/reference unless explicitly refactoring shared abstractions.
- Keep examples as examples; do not place reusable core logic in example files.
- Follow PEP8 where practical.
- For consecutive assignment statements in the same logical block, align `=` signs locally when it improves readability. Do not align across blank lines or unrelated sections.
- Before deleting code, confirm it is unused or ask for approval.