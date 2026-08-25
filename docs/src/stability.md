# Ideal MHD Stability (ForceFreeStates)

The `ForceFreeStates` module implements ideal MHD stability analysis for axisymmetric toroidal
plasmas following the direct Newcomb criterion described in [Glasser 2016].  It solves the
Euler-Lagrange (EL) system derived from the potential energy functional, identifies singular
(rational) surfaces where resonant coupling occurs, and returns eigenmode energies, the
tearing stability parameters Δ', and the full inter-surface Δ' matrix.

## Physical background

Ideal MHD stability is determined by the sign of the perturbed potential energy

```math
\delta W[\xi] = \int_0^{\psi_\mathrm{lim}} \mathcal{F}(\xi, \xi') \, d\psi,
```

where ``\xi(\psi)`` is the poloidal displacement vector.  The extremum of ``\delta W`` over all
admissible ``\xi`` satisfies the Euler-Lagrange system [Glasser 2016, Eq. 24]:

```math
\frac{d}{d\psi}
\begin{pmatrix} U_1 \\ U_2 \end{pmatrix}
=
\begin{pmatrix} A & B \\ C & D \end{pmatrix}
\begin{pmatrix} U_1 \\ U_2 \end{pmatrix},
\quad
A = -Q\bar{F}^{-1}\bar{K}, \;
B = Q\bar{F}^{-1}Q, \;
C = \bar{G} - \bar{K}^\dagger\bar{F}^{-1}\bar{K}, \;
D = \bar{K}^\dagger\bar{F}^{-1}Q,
```

where ``\bar{F}``, ``\bar{K}``, ``\bar{G}`` are the MHD metric matrices in Fourier-mode space
and ``Q = \mathrm{diag}(1/(m - nq))`` is the singular factor.  The Newcomb criterion states
that the plasma is stable if and only if this system admits a regular solution that remains
finite across every rational surface.

**Key references**

| Paper | Content |
|-------|---------|
| [Glasser 2016] Phys. Plasmas **23**, 112506 | Newcomb criterion, EL system, standard DCON integration |
| [Glasser 2018a] Phys. Plasmas **25**, 032507 | Riccati reformulation, reduced stiffness near singular surfaces |
| [Glasser 2018b] Phys. Plasmas **25**, 032501 | STRIDE code: parallel FM integration, inter-surface Δ' matrix |

## Integration methods

Three formalisms are available, selected by `integrator`. Forward and Riccati solve the same
EL system and differ in numerical strategy and in what they leave behind for the rest of the
pipeline: the forward driver returns dense displacement profiles, the Riccati driver returns
the inter-surface ``\Delta'`` matrix. Galerkin solves the outer region variationally instead,
and is documented in `docs/src/galerkin.md`.

### Forward integration

`forward_eulerlagrange_integration` is the baseline driver — our implementation of the
standard DCON radial integration [Glasser 2016].  It integrates the EL ODE directly
in ``(U_1, U_2)`` using the adaptive 9th-order `Vern9` solver.  Near each rational surface the
columns of ``U_2`` that correspond to resonant modes are zeroed via Gaussian reduction (GR),
keeping the solution bounded; `transform_u!` undoes the reduction at the end, so `u_store`
comes back dense in the axis (Euler-Lagrange) basis.  This is the reference path for
correctness comparisons, the only path whose solution the perturbed-equilibrium stage can
consume, and the only path that supports `kinetic_factor > 0`.

Enable with:
```toml
[ForceFreeStates]
integrator = "forward"
```

### Riccati integration

`riccati_eulerlagrange_integration` (the default) is our implementation of the STRIDE
approach [Glasser 2018b], built on the dual Riccati reformulation [Glasser 2018a].  It
decomposes the radial domain into
independent chunks, integrates each chunk's fundamental-matrix (FM) propagator in parallel
using `Threads.@threads`, then multiplies the propagators in order and applies each
singular-surface crossing serially.  It is the only driver that produces the inter-surface
``\Delta'`` matrix.  Because chunk endpoints are all it stores, `u_store` is sparse and stays
in the Riccati basis — dense ``\xi`` profiles and the
`ForceFreeStates/Solutions/ForwardIntegration/xi_*` datasets require the forward driver.

Crossings and the outer-plasma re-integration use the dual Riccati matrix
``S = U_1 \cdot U_2^{-1}`` [Glasser 2018a, Eq. 19]:

```math
\frac{dS}{d\psi} = w^\dagger \bar{F}^{-1} w - S\bar{G}S, \qquad
w = Q - \bar{K}S.
```

``S`` remains bounded near rational surfaces (where ``U_1, U_2`` grow exponentially).  Rather
than integrating the quadratic Riccati ODE directly (which blows up when ``|S|`` is large),
the code integrates the linear EL system with `sing_der!` as the RHS and recovers
``S = U_1 U_2^{-1}`` via periodic renormalization — an approach that is mathematically
equivalent to O(Δψ) but uses `Vern9`'s full 9th-order accuracy.  Renormalization is
triggered whenever ``\max(|U_1|)`` or ``\max(|U_2|)`` exceeds the threshold `ucrit`, and is
forced at the end of each chunk.  At singular surface crossings,
`riccati_cross_ideal_singular_surf!` applies the small-asymptotic matching directly in column
`ipert_res` — without Gaussian reduction — and renormalizes to ``(S, I)``.

Enable with:
```toml
[ForceFreeStates]
integrator = "riccati"
nchunks    = 0         # 0 = auto: derived from the singular-surface count alone
```

#### Chunking and thread independence

`balance_integration_chunks` splits the base chunks until the count reaches a target derived
from the number of singular surfaces, `max(2 m_s + 3, 8(m_s + 1) + m_s)`, where ``m_s`` is
`msing`.  Setting `nchunks` overrides that target; a value below the ``2 m_s + 3`` floor is
clamped up with a warning.  The target never consults `Threads.nthreads()`, and every chunk
integrates independently from identity initial conditions, so the results do not depend on how
many threads `julia -t` provides — threads change wall-clock only.

#### Bidirectional integration for large N

For large mode counts the FM propagator for a chunk ending near a rational surface is
ill-conditioned: the EL solutions grow exponentially toward the rational surface, so the
forward FM amplifies numerical errors.  GPEC follows the STRIDE approach [Glasser 2018b,
Sec. III.A]: the crossing chunk (the last sub-chunk before each rational surface) is
integrated *backward* — from the rational surface toward the interior — producing a
well-conditioned backward FM ``\Phi_L``.  The forward propagation is recovered as
``\Phi_L^{-1}`` via an LU solve in serial assembly, which is accurate precisely because
``\Phi_L`` is well-conditioned.

The implementation uses a `direction` field on `IntegrationChunk`:

- `direction = +1`: standard forward integration, `tspan = (ψ_start, ψ_end)`.
- `direction = -1`: backward integration, `tspan = (ψ_end, ψ_start)` (reversed).

`chunk_el_integration_bounds(...; bidirectional=true)` assigns `direction = -1` to every
crossing chunk.  `balance_integration_chunks` preserves this: the sub-chunk closest to the
rational surface inherits `direction`, while the earlier sub-chunk always gets `direction=+1`.

**Accuracy** (N=26, DIIID-like example): energy eigenvalue within 2% of the forward path.
The residual ~2% gap comes from the different crossing convention (Riccati-style direct
zeroing vs GR), not from ODE tolerance; it is present at every thread count.

### Galerkin

`integrator = "galerkin"` solves the same EL system variationally instead of integrating it:
the outer region is discretized on packed Hermite-cubic elements and solved as one global banded
system, giving the RDCON resistive ``\Delta'`` matrix and the PEST-3 matching blocks.  It computes its own vacuum response and
returns no free-boundary energies, no ODE trace, and no fixed-boundary `crit` scan.  With
`gal_match_flag` it also matches the inner layer, producing a driven ``\xi`` solution that
`PerturbedEquilibrium` consumes.  Kinetic runs are not supported.  See
`docs/src/galerkin.md` for the solver and its `gal_*` knobs.

Enable with:
```toml
[ForceFreeStates]
integrator = "galerkin"
```

## Local stability: Mercier and ballooning (s–α)

Setting `local_stability_flag = true` in `[ForceFreeStates]` runs a local high-``n``
stability scan over every flux surface, in addition to the global ideal analysis above.
For the derivation and implementation details behind these diagnostics, see
[Ballooning and Mercier Local Stability](ballooning.md).
Three diagnostics are produced and stored under the `LocalStability/` HDF5 group, each a profile
in normalized poloidal flux ``\psi``:

- **Mercier criterion ``D_I``** (`LocalStability/D_I`) — the ideal interchange criterion. A surface
  is Mercier-unstable where ``D_I > 0``. It is evaluated from the ``\det(\bar{d}_0)`` of the
  integrated local-mode matrix.
- **Resistive interchange ``D_R``** (`LocalStability/D_R`) — the Glasser–Greene–Johnson resistive
  interchange criterion ``D_R = D_I + (H - 1/2)^2``. The ``D_I`` term is the same
  ``\det(\bar{d}_0)`` value reported in `LocalStability/D_I`; ``H`` is computed from the legacy
  Mercier/GGJ flux-surface averages of the field and metric quantities. ``D_R > 0``
  indicates resistive interchange instability.
- **Ballooning ``\Delta'``** (`LocalStability/ballooning_Delta_prime`) — the high-``n`` ballooning
  stability index, obtained by integrating the ballooning equation along the field line and
  taking the jump in the logarithmic derivative of the solution between the two asymptotic
  ends.

!!! note "Two different Δ' quantities"
    `LocalStability/ballooning_Delta_prime` is the **local high-``n`` ballooning** index and is
    distinct from the **resistive tearing** ``\Delta'`` described in the next section, which
    is written under `SingularSurfaces/` and `PerturbedEquilibrium/SingularCoupling/Delta_prime`.
    They measure different instabilities; do not confuse them.

### s–α diagram

The local criteria can be mapped over a two-parameter ``(p', q')`` scan at a single flux
surface — the classic s–α (shear–pressure) stability diagram. For a chosen surface the
pressure gradient ``p'`` and shear ``q'`` are scaled away from their equilibrium values, and
``\Delta'`` and ``D_I`` are recomputed on the resulting grid. The ``\Delta' = 0`` and
``D_I = 0`` contours bound the ballooning- and Mercier-stable regions, with the equilibrium
operating point marked inside.

The example script `examples/DIIID-like_ideal_example/analyze_example.jl` demonstrates the
full workflow: it plots the ``s`` and ``\alpha`` profiles, the ``D_I`` and ballooning
``\Delta'`` profiles, and the 2-D s–α maps with their zero contours, using
`salpha_reference`, `compute_ballooning_stability!`, and `scan_delta_prime_map`.

## Δ' tearing stability parameter

### Per-surface Δ' (`delta_prime`)

At each rational surface the asymptotic matching condition gives the tearing stability
parameter [Glasser 2016]:

```math
\Delta'_s = \frac{c_{a,r}[i_s,i_s,2] - c_{a,l}[i_s,i_s,2]}{4\pi^2 \psi_0},
```

where ``c_{a,l}`` and ``c_{a,r}`` are the left and right asymptotic coefficients at surface
``s``, and ``i_s`` is the column index of the resonant mode.  Positive ``\Delta' > 0``
indicates a tearing-unstable surface.

The Riccati and parallel FM paths populate `intr.sing[s].delta_prime` (a length-``n_\mathrm{res}``
vector) inline during each crossing.  A companion vector `delta_prime_col` (length N) stores
the coupling of all poloidal modes to the resonant mode at surface ``s``:

```math
(\Delta'_\mathrm{col})_{j,i} = \frac{c_{a,r}[j,i_s,2] - c_{a,l}[j,i_s,2]}{4\pi^2 \psi_0}.
```

The diagonal element ``(\Delta'_\mathrm{col})_{i_s,i}`` equals `delta_prime[i]` exactly by
construction.

### Inter-surface Δ' matrix (`delta_prime_matrix`)

`compute_delta_prime_matrix!` assembles an ``m_\mathrm{sing} \times m_\mathrm{sing}``
inter-surface tearing matrix following the STRIDE global BVP [Glasser 2018b, Sec. III.B].
Internally, the solver builds a raw ``2 m_\mathrm{sing} \times 2 m_\mathrm{sing}`` matrix
whose rows/columns index the *left* and *right* inner-layer boundaries of every rational
surface; the stored PEST3-convention ``\Delta'`` is the four-term combination
``\text{dp\_raw}[2i, 2j] - \text{dp\_raw}[2i, 2j{-}1] - \text{dp\_raw}[2i{-}1, 2j] + \text{dp\_raw}[2i{-}1, 2j{-}1]``
that folds the raw block into a per-surface response.  The BVP unknowns are the plasma
state at the left and right inner-layer boundaries of every rational surface; the driving
terms are unit-amplitude asymptotic solutions at each boundary.  The resulting matrix
encodes the full plasma response between all pairs of surfaces and is required for
resistive stability analysis of multi-surface configurations.

The BVP is well-conditioned because it is formulated using the split ``(\Phi_R, \Phi_L)``
propagator blocks from bidirectional integration rather than the monolithic forward product
``\Phi_L^{-1} \Phi_R`` (which is ill-conditioned for large N):

```math
\Phi_R[j] \cdot x_R[j-1] - \Phi_L[j] \cdot x_L[j] = 0
\quad \text{(junction at } \psi_m[j]\text{)},
```

where ``\Phi_R[j]`` is the forward FM product from ``\psi_{R,j-1}`` to the junction, and
``\Phi_L[j]`` is the backward crossing FM from ``\psi_{L,j}`` to the junction.

The matrix is written to the HDF5 output under `SingularSurfaces/Delta_prime_matrix`.
The Galerkin integrator computes the same quantity in the same PEST-3 convention and
publishes it on the same path, so downstream consumers (SLAYER among them) never branch
on which formalism ran.

## Configuration reference

All `ForceFreeStates` options are set in the `[ForceFreeStates]` section of `gpec.toml`.

```toml
[ForceFreeStates]
# Integration driver
integrator   = "riccati"  # "forward" for dense xi profiles and kinetic runs; "galerkin" for the RDCON outer-region solve
nchunks      = 0          # Riccati chunk-count target (0 = auto, from msing alone)

# Mode space
nn_low       = 1       # lowest toroidal mode number
nn_high      = 1       # highest toroidal mode number
delta_mlow   = 0       # extra low poloidal modes (m < mlow)
delta_mhigh  = 0       # extra high poloidal modes (m > mhigh)

# ODE solver
numsteps_init     = 200    # initial step budget per chunk
numunorms_init    = 50     # renorm checkpoint budget
reltol            = 1e-6   # ODE relative tolerance

# Local stability
local_stability_flag = false  # scan Mercier D_I, resistive D_R, and ballooning Δ' over ψ

# Output
verbose              = true
write_outputs_to_HDF5 = true
```

The number of Julia threads is controlled at startup via `-t N` or the `JULIA_NUM_THREADS`
environment variable; it is not a runtime parameter.

## API Reference

The Galerkin Δ′ solver (`src/ForceFreeStates/Galerkin/`) is documented separately in
`docs/src/galerkin.md`.

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ForceFreeStates]
Pages = ["ForceFreeStates.jl", "CoreTypes.jl", "Surfaces/Types.jl", "Riccati/Types.jl", "Matching/DeltaPrime.jl", "Result.jl", "Surfaces/Resist.jl", "Surfaces/ResistEval.jl", "Matching/ResonantMatch.jl", "EulerLagrange.jl", "Surfaces/Finding.jl", "Surfaces/Asymptotics.jl", "Fourfit.jl", "Kinetic.jl", "FixedBoundaryStability.jl", "Utils.jl", "Free.jl", "Riccati/Propagators.jl", "Riccati/Crossings.jl", "Riccati/DeltaPrimeBVP.jl", "Riccati/Driver.jl"]
```

## Example usage

### Run stability analysis from a TOML configuration

```julia
using GeneralizedPerturbedEquilibrium, TOML

const FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates

ex     = "examples/Solovev_ideal_example"
inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))

ctrl  = FFS.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(
            GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex))

intr  = FFS.ForceFreeStatesInternal(; dir_path=ex)
intr.wall_settings = GeneralizedPerturbedEquilibrium.Vacuum.WallShapeSettings(;
    (Symbol(k) => v for (k, v) in inputs["Wall"])...)
intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
FFS.sing_lim!(intr, ctrl, equil)
FFS.sing_find!(intr, equil)
intr.mlow  = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
intr.mpert = intr.mhigh - intr.mlow + 1
intr.numpert_total = intr.mpert * intr.npert

metric = FFS.make_metric(equil, intr.mpert)
ffit   = FFS.make_matrix(equil, intr, metric)

# Choose integration driver.  The top-level `eulerlagrange_integration` dispatches
# on ctrl.integrator and always returns a 4-tuple
# (odet, propagators, chunks, S_at_surface_left).  The trailing three are `nothing`
# on the forward path.
odet, _, _, _ = FFS.eulerlagrange_integration(ctrl, equil, ffit, intr)

vac = FFS.free_run(odet, ctrl, equil, ffit, intr)
println("Energy eigenvalue et[1] = ", real(vac.et[1]))
```

### Inspect Δ' at singular surfaces

```julia
for s in 1:intr.msing
    sing = intr.sing[s]
    println("Surface $s: ψ = $(sing.psi_s), m/n = $(sing.m[1])/$(sing.n[1])")
    println("  Δ' = $(real(sing.delta_prime[1]))")
end
```

### Access inter-surface Δ' matrix (Riccati path)

```julia
# intr.delta_prime_matrix is msing × msing after riccati_eulerlagrange_integration.
# Internally the solver builds a 2·msing × 2·msing raw matrix; the stored Δ' is
# the PEST3 four-term combination that folds the raw block into a per-surface
# tearing parameter.
dpm = intr.delta_prime_matrix
println("Δ' matrix size: ", size(dpm))
println("Diagonal (self-response Δ'):")
for j in 1:intr.msing
    println("  Surface $j: ", real(dpm[j, j]))
end
```

## Notes

- The standard path does not populate `delta_prime`; the canonical Δ' is the STRIDE BVP
  `SingularSurfaces/Delta_prime_matrix` from the parallel FM path. `ca_l`/`ca_r` are filled
  only by ideal surface crossings (kinetic and galerkin-matched runs emit zero-extent
  `ca_left`/`ca_right` sentinels).
- The Riccati and parallel FM paths compute Δ' inline at each crossing, using the
  direct diagonal formula (no GR permutation).  The result in `delta_prime_col[ipert_res, i]`
  equals `delta_prime[i]` to machine precision.
- `delta_prime_matrix` contains raw BVP coefficients, not asymptotic-normalized values;
  its diagonal elements do **not** in general equal `delta_prime`.
- ODE step counts depend on the equilibrium profile and mode count; the `numsteps_init`
  parameter sets the initial allocation but the solver adapts automatically.

## See also

- `docs/src/galerkin.md` — RDCON outer-region Galerkin Δ′ solver (part of this module)
- `docs/src/equilibrium.md` — build the `PlasmaEquilibrium` object required by this module
- `docs/src/vacuum.md` — vacuum response computed from the EL solution in `free_run`
- `docs/src/perturbed_equilibrium.md` — downstream singular coupling analysis using Δ'
