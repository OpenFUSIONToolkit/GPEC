# Islands Module

The Islands module is a steady-state, multi-species drift-kinetic solver for the
resonant magnetic island/layer region in tokamaks — the nonlinear analog of
SLAYER, generalizing the Modified Rutherford Equation. It is under active
development. The design documents live under `islands/design/` in the
documentation source tree, and the module conventions in `src/Islands/CLAUDE.md`.

## Status — milestone M1 (skeleton)

M1 lands the numerical skeleton, not the physics numbers (module `CLAUDE.md`, the
`[VERIFY]` policy):

  - `Islands.PhaseSpace` — the `(x, ξ, λ→y, E, σ)` phase-space grids with
    layer-clustered mappings (design `04 §1`): Fourier spectral `∂ξ`, high-order
    finite-difference `∂x`/`∂y` on stretched grids, and Gauss energy quadrature.
    Pure numerics; no physics coefficients.
  - `Islands.Operators` — the `AbstractTerm` operator stack and residual assembly
    (design `03 §2`) as allocation-free, AD-compatible *structural stubs*. Every
    physics coefficient is a supplied data field, never a literal — literature
    numbers stay `[VERIFY]`/`[CHECKED]`-gated until human-cleared.
  - `Islands.Verify` — the manufactured-solution (MMS) and AD-vs-finite-difference
    JVP harness backing verification ladder **A1/A2** (design `05 §A`), exercised
    by `test/runtests_islands_{grids,operators}.jl`.

The physics operators (drift-frequency coefficients, collision kernels, the
Δ moments and York thresholds) land in later milestones and remain gated until
their `[VERIFY]` tags are cleared.

## API Reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Islands]
```
