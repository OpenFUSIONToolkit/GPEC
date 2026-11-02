---
name: annotation_style
description: How to cite Fortran code in annotations — use papers and submodule names, not line numbers
type: feedback
---

Do NOT cite Fortran source line numbers in code annotations (e.g., "gpeq.f:100"). The Fortran codebase is actively developed and line numbers change. Instead, cite:
- Paper equations (e.g., "Park et al. 2007, Eq. 8")
- Fortran subroutine/function names (e.g., "matches Fortran `gpeq_contra`")

**Why:** Fortran GPEC is still under active development; line-number references rot quickly.

**How to apply:** When the fortran-physics-reviewer suggests line-number annotations, convert them to subroutine-name or paper-equation references before writing to code.
