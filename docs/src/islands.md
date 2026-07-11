# Islands Module

The Islands module is a steady-state, multi-species drift-kinetic solver for the
resonant magnetic island/layer region in tokamaks — the nonlinear analog of
SLAYER, generalizing the Modified Rutherford Equation. It is under active
development; the module conventions live in `src/Islands/CLAUDE.md`.

This page is the **API reference**. The narrative documentation lives in the
**Islands** section of this site: the [project overview](islands/index.md),
the equations-and-figures chapter of what is
[implemented and verified so far](islands/numerics.md), the
[Paper I figure contract](islands/papers/paper-1/OUTLINE.md), and the full
[design document set](islands/design/00-roadmap.md).

## Status — M1 skeleton + M2 Level-0 solve machinery (structure, gated physics)

M1 landed the numerical skeleton and M2 the Level-0 solve machinery — structure
only, no physics numbers (module `CLAUDE.md`, the `[VERIFY]` policy):

  - `Islands.PhaseSpace` — the `(x, ξ, λ→y, E, σ)` phase-space grids with
    layer-clustered mappings (design `04 §1`): Fourier spectral `∂ξ`, high-order
    finite-difference `∂x`/`∂y` on stretched grids, and Gauss energy quadrature.
    Pure numerics; no physics coefficients.
  - `Islands.Operators` — the `AbstractTerm` operator stack and residual assembly
    (design `03 §2`) as allocation-free, AD-compatible structure, including the
    mimetic (exactly conservative) pitch-angle diffusion and the neoclassical-
    matching far-field boundary conditions (`01 §3` — never bare Neumann). Every
    physics coefficient is a supplied data field, never a literal.
  - `Islands.SpeciesLists` — first-class species arrays (`02 §1`, Decision D3):
    backgrounds, `Bulk`/`Trace` roles, trace-criteria checks that warn and never
    silently degrade.
  - `Islands.Frames` — THE frequency/frame conversion module (`01 §5`): the
    conversion *forms* with every sign/normalization a NaN-gated
    `FrameConvention` field until human-cleared (QUESTIONS Q3).
  - `Islands.Fields` — the Level-0 quasineutrality closure structure: `Q(Ω)`,
    `h(Ω)` (supplied prefactor), the coefficient-free identity
    `⟨∂²h/∂x²⟩_Ω = 0` (ladder A7), and the NaN-gated `ElectronClosure` constants.
  - `Islands.Moments` — `J̄_∥` assembly and the `Δ_cos`/`Δ_sin` Ampère
    projections (`01 §4`) with **required, gated** prefactors; `Ω`-average and
    channel-split diagnostics.
  - `Islands.Solvers` — matrix-free Newton–Krylov (ForwardDiff JVP + GMRES,
    Eisenstat–Walker forcing), the TSVD-regularized physics-block preconditioner
    skeleton (`04 §3, §5`), tiny-grid dense debug Jacobian, and pseudo-arclength
    continuation with fold detection.
  - `Islands.Verify` — MMS/JVP harness (ladder A1/A2), solve-level MMS and
    zero-drive configurations (A5), and the `y_c`-block conditioning monitor
    (A8), exercised by `test/runtests_islands_{grids,operators,solve}.jl`.

Structural gates **A1–A5, A7 (coefficient-free part), A8** run in CI. The
physics numbers (drift-frequency coefficients, collision kernels, closure
constants, the Δ prefactors, threshold values) remain `[VERIFY]`-gated: the
B-ladder benchmarks in `benchmarks/islands/` ship skipped, each naming the
`QUESTIONS.md` entries (Q2–Q4) whose human clearance un-gates it. The
literature-facing physics gates are **tiered by reproducibility** (Decision D9,
docs/05): scalings, regime trends, and internal differentials are the primary
quantitative checks, and absolute literature numbers are audit-gated (reported
only with input manifests). The Paper-I figure contract is
`docs/src/islands/papers/paper-1/OUTLINE.md`.

## API Reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Islands]
```

### Phase-space grids (`Islands.PhaseSpace`)

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Islands.PhaseSpace]
```

### Species (`Islands.SpeciesLists`)

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Islands.SpeciesLists]
```

### Frames and parameters (`Islands.Frames`)

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Islands.Frames]
```

### Operator stack (`Islands.Operators`)

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Islands.Operators]
```

### Field-equation closure structure (`Islands.Fields`)

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Islands.Fields]
```

### Output moments (`Islands.Moments`)

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Islands.Moments]
```

### Cleared physics coefficients (`Islands.Coefficients`)

Human-cleared Level-0 coefficient builders (the M2b derivation-lane fill-ins;
see the Derivations section).

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Islands.Coefficients]
```

### Newton–Krylov solve (`Islands.Solvers`)

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Islands.Solvers]
```

### Verification harness (`Islands.Verify`)

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Islands.Verify]
```
