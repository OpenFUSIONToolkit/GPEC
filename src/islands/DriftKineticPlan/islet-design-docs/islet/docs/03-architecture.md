# 03 — Architecture (Julia)

Design principle: **the physics levels are configurations, not code paths.** One
discretization, one Newton solver, one state vector; orderings are swappable
operators and flags. If implementing a new level requires touching the solver
loop, the architecture has failed.

## 1. Package layout

```
ISLET.jl/
├── Project.toml
├── CLAUDE.md
├── README.md
├── docs/                    # these design documents (normative)
├── src/
│   ├── ISLET.jl
│   ├── geometry/            # AbstractEquilibrium: AnalyticCircular (L0),
│   │                        #   MillerAnalytic (L2 entry; the D23a geometry),
│   │                        #   NumericalEquilibrium (L2; DCON/gEQDSK ingest)
│   ├── phasespace/          # grids (x, ξ[, θ]; λ, E, σ), maps, quadrature,
│   │                        #   layer-clustered mappings (docs/04)
│   ├── species/             # Species, backgrounds, roles (docs/02)
│   ├── frames/              # THE frequency/frame conversion module (docs/01 §5)
│   ├── operators/           # the stack (see §2)
│   ├── fields/              # Φ̃ quasineutrality residual; A_∥ Ampère residual (L3)
│   ├── closures/            # torque balance, χ⊥ transport, radiation (L4)
│   ├── moments/             # Δ_cos, Δ_sin, profiles, channel decompositions
│   ├── solvers/             # Newton–Krylov, continuation, trace-species linear pass
│   ├── io/                  # config (TOML), results (HDF5/JLD2), provenance
│   └── verify/              # benchmark harness callable from tests AND scripts
├── test/                    # unit + symmetry + conservation tests (fast)
├── benchmarks/              # the docs/05 ladder (slow; CI-gated subsets)
└── scripts/                 # surface generation, paper figures
```

## 2. The operator stack

The state is `U = (g_1, …, g_Nbulk, Φ̃ [, A_∥ at L3] [, ω, E_r at L4])`. The
residual is assembled as a sum of operator applications:

```julia
abstract type AbstractTerm end
# each implements: apply!(R, term, U, cache), and is either matrix-free or
# provides a local stencil for preconditioning

struct ParallelStreaming   <: AbstractTerm end   # includes island B̃_r ∂x
struct MagneticDrift       <: AbstractTerm       # variant = :original (finite L̂_B⁻¹, I19)
                                                 #         | :improved (L̂_B⁻¹ → 0 proxy of
                                                 #           the cosθ ∂B/∂ψ structure, D21
                                                 #           Eq. A2) — the 8.73→1.46 ρ_bi
                                                 #           toggle, docs/01 §2.1
struct ExBDrift            <: AbstractTerm end
struct Collisions{M}       <: AbstractTerm       # M = PitchAngle (L0; energy-dependence
                                                 #     sub-toggle :chandrasekhar (I19) |
                                                 #     :vcubed (D21), docs/01 §2.3) |
                                                 #     FokkerPlanckMulti (L1)
struct GradientDrive       <: AbstractTerm end   # (v_E+v_D+v_ψ̃)·∇F₀ ; dispatches
                                                 # on background type (Maxwellian /
                                                 # SlowingDown)
struct PerpTransport       <: AbstractTerm end   # χ⊥, D⊥ (L4)
struct RadiationSink       <: AbstractTerm end   # (L4, energy closure)
```

Configuration = list of terms per species + field-equation set + closure set,
read from a TOML config. **Named configurations are pinned in `verify/`**:
`:imada2019` (B5a; note it targets the L23-amended physics, not I19 Eq. A.1 as
printed — docs/01 header), `:dudkovskaia2021` (B5b), `:leigh2023` (B5c),
`:sauter_limit`, `:slayer_limit`, … — the toggle-comparison studies the
project exists to do are then config diffs, and every published figure names
its configuration.

Rules:
- No term may inspect which other terms are active (no hidden coupling).
- Every term carries its own verification hook (an analytic limit or manufactured
  solution registered in `verify/`).
- Orderings that *remove* structure (e.g. orbit averaging O5) are implemented as
  alternative phase-space configurations, not operators: `phase = :orbit_averaged
  (4D)` vs `:full (5D)`. Terms are written against an abstract phase-space
  interface so both share implementations where possible.

## 3. Solver strategy

- **Steady-state Newton–Krylov** (D2): matrix-free JVPs via AD; GMRES; physics-
  block preconditioner (docs/04 §4).
- **AD policy**: JVPs and small-parameter sensitivities via ForwardDiff duals
  through the operator stack (write terms generically over `eltype`); evaluate
  Enzyme for reverse-mode ∂Δ/∂p over the full parameter vector when surface
  generation begins. AD-compatibility is a CI test per term (no
  `Float64`-hardcoded buffers — reuse the per-thread preallocation patterns from
  GPEC/QuadGK work, but typed generically).
- **Continuation**: pseudo-arclength in any component of p (and in ψ̃₀/w).
  Purposes: (i) Newton globalization by homotopy from analytic-limit solutions,
  (ii) Δ-surface generation as a byproduct, (iii) bifurcation tracking — the
  penetration bifurcation at Level 3 *is* a fold in the continuation curve, so the
  solver must detect/step around folds from the start.
- **Trace pass**: after bulk convergence, each Trace species is one linear solve
  with frozen bulk fields (docs/02 §1.2); reuses the same operator stack and
  Krylov machinery with a linear RHS.
- **ω closure (L4)**: ω, E_r appended to U; torque-balance residual appended to R.
  Structurally identical Newton system — this is why the closure is cheap *if*
  the architecture holds.

## 4. Data and provenance

- Config: TOML in, full resolved config (all defaults expanded, git SHA, term
  list, grid spec) stored inside every output file. A result that can't
  regenerate itself is a bug.
- Output: HDF5/JLD2 with (p, Δ_cos, Δ_sin, channel decompositions, convergence
  metadata, profiles on demand). Surface datasets are append-only stores keyed by
  p; the emulator (M12) trains from these.

## 5. External interfaces

> **Superseded where in-repo (docs/06 §1):** with ISLET living inside the GPEC
> repository, the Δ′ and SLAYER interfaces below become direct Julia calls
> against GPEC's implementations, exercised in CI. The file-based forms remain
> the spec for any standalone/external consumers.

- **Δ′(w) input**: file-based interface (resistive DCON/STRIDE output → a small
  table Δ′ vs. w at the rational surface, with the matching radius L_x recorded).
  Keep it dumb and versioned; no live coupling until the physics settles.
- **Equilibrium input (L2)**: gEQDSK + profiles, mapped through the same
  representations the GPEC stack uses; `AnalyticCircular` remains permanently
  available for regression.
- **SLAYER (L3 verification)**: comparison harness that maps ISLET's linear-limit
  (Δ_cos + iΔ_sin) onto SLAYER's Δ(Q) convention — the frames module owns the Q
  mapping [VERIFY: Park 2022 conventions — paper not yet in the reference
  library]. **Status:** the in-repo SLAYER arrives with the Tearing module PR
  (#238, `src/Tearing/InnerLayer/SLAYER/`), sequenced before ISLET starts
  (docs/06 §1); code against the Tearing-module layout, not `develop`'s old
  `src/InnerLayer` placeholder.
- **NEO/NCLASS (L1 verification)**: no-island neoclassical cross-check driver
  (export local parameters → run → compare bootstrap/flows).

## 6. Performance posture (sizing, not optimization)

L0 4D: N_x × N_ξ × N_λ × N_E × σ ~ 400×64×128×32×2 ≈ 2×10⁸ dof upper bound
per bulk species before layer-adapted grids reduce N_x, N_λ needs — matrix-free
mandatory. The prior-art cost baseline makes the case concrete: kokuchou's
dense-block shooting method needed ≈16.6 GB *per energy grid point* and
O(100 GB) total at n_ξ = 30, n_p = 145, and memory (not physics) set its
accessible parameter window (docs/04 §6). Single-node multithread first (Julia
threads; the GC-contention lessons from parallel QuadGK apply: preallocate
per-thread caches). MPI only if
L2 5D demands it; do not architect for MPI speculatively, but keep halo-friendly
array layouts (x outermost) so the option stays open. Every solve logs a cost
model entry (dof, iterations, wall time) — the emulator strategy depends on
knowing what a point costs.
