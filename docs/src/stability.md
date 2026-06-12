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

Three integration drivers are available, all solving the same EL system but with different
numerical strategies.

### Standard integration

`eulerlagrange_integration` is the baseline driver.  It integrates the EL ODE directly in
``(U_1, U_2)`` using Tsit5 with adaptive step control.  Near each rational surface the
columns of ``U_2`` that correspond to resonant modes are zeroed via Gaussian reduction (GR),
keeping the solution bounded.  This is the reference path for correctness comparisons.

Enable with (default):
```toml
[ForceFreeStates]
use_riccati  = false
use_parallel = false
```

### Riccati integration

`riccati_eulerlagrange_integration` reformulates the problem in terms of the dual Riccati
matrix ``S = U_1 \cdot U_2^{-1}`` [Glasser 2018a, Eq. 19]:

```math
\frac{dS}{d\psi} = w^\dagger \bar{F}^{-1} w - S\bar{G}S, \qquad
w = Q - \bar{K}S.
```

``S`` remains bounded near rational surfaces (where ``U_1, U_2`` grow exponentially), so the
solver takes fewer steps.  Rather than integrating the quadratic Riccati ODE directly (which
blows up when ``|S|`` is large), the code integrates the linear EL system with
`sing_der!` as the RHS and recovers ``S = U_1 U_2^{-1}`` via periodic renormalization — an
approach that is mathematically equivalent to O(Δψ) but uses the ODE solver's full 5th-order
accuracy.

Renormalization is triggered whenever ``\max(|U_1|)`` or ``\max(|U_2|)`` exceeds the
threshold `ucrit` (default 1e6), and is forced at the end of each chunk.  At singular surface
crossings, `riccati_cross_ideal_singular_surf!` applies the small-asymptotic matching
directly in column `ipert_res` — without Gaussian reduction — and renormalizes to ``(S, I)``.

Enable with:
```toml
[ForceFreeStates]
use_riccati  = true
use_parallel = false
```

**Speedup** (benchmarked on reference examples):

| Example | N modes | Speedup vs standard |
|---------|---------|---------------------|
| Solovev | 8  | ~1.6× (1 thread), ~2.8× (4 threads) |
| DIIID   | 26 | ~2.0× (1 thread), ~1.3× (4 threads) |

### Parallel fundamental-matrix (FM) integration

`parallel_eulerlagrange_integration` decomposes the radial domain into independent chunks and
integrates each chunk in parallel using `Threads.@threads`.  Each chunk produces a
fundamental-matrix (FM) propagator.  Serial post-processing multiplies the propagators in
order and applies each singular-surface crossing, recovering the same EL trajectory as the
Riccati path.

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

Enable with:
```toml
[ForceFreeStates]
use_parallel = true
```

**Accuracy** (N=26, DIIID-like example): energy eigenvalue within 2% of standard path.
The residual ~2% gap comes from the different crossing convention (Riccati-style direct
zeroing vs GR), not from ODE tolerance; it is present in both 1-thread and 4-thread runs.

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

The matrix is only populated by the parallel FM path and is written to the HDF5 output
under `singular/delta_prime_matrix`.

## Configuration reference

All `ForceFreeStates` options are set in the `[ForceFreeStates]` section of `gpec.toml`.

```toml
[ForceFreeStates]
# Integration driver
use_riccati  = false   # true: Riccati path (faster, same accuracy)
use_parallel = false   # true: parallel FM path (multi-thread, large N)

# Mode space
nn_low       = 1       # lowest toroidal mode number
nn_high      = 1       # highest toroidal mode number
delta_mlow   = 0       # extra low poloidal modes (m < mlow)
delta_mhigh  = 0       # extra high poloidal modes (m > mhigh)

# ODE solver
numsteps_init     = 200    # initial step budget per chunk
numunorms_init    = 50     # renorm checkpoint budget
reltol            = 1e-6   # ODE relative tolerance

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
Pages = ["ForceFreeStates.jl", "ForceFreeStatesStructs.jl", "Mercier.jl", "Resist.jl", "Bal.jl", "EulerLagrange.jl", "Sing.jl", "Fourfit.jl", "Kinetic.jl", "FixedBoundaryStability.jl", "Utils.jl", "Free.jl", "Riccati.jl"]
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
FFS.sing_lim!(intr, ctrl, equil)
intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
FFS.sing_find!(intr, equil)
intr.mlow  = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
intr.mpert = intr.mhigh - intr.mlow + 1
intr.mband = intr.mpert - 1
intr.numpert_total = intr.mpert * intr.npert

metric = FFS.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)
ffit   = FFS.make_matrix(equil, intr, metric)

# Choose integration driver.  The top-level `eulerlagrange_integration` dispatches
# to the parallel or Riccati path based on ctrl.use_parallel / ctrl.use_riccati,
# and always returns a 4-tuple (odet, propagators, chunks, S_at_surface_left).
odet, _, _, _ = FFS.eulerlagrange_integration(ctrl, equil, ffit, intr)

vac = FFS.free_run!(odet, ctrl, equil, ffit, intr)
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

### Access inter-surface Δ' matrix (parallel FM path)

```julia
# intr.delta_prime_matrix is msing × msing after parallel_eulerlagrange_integration.
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

- The standard path does not populate `delta_prime`; use `PerturbedEquilibrium.SingularCoupling`
  for Δ' on the standard path (it reads `ca_l`/`ca_r` directly).
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
- `docs/src/vacuum.md` — vacuum response computed from the EL solution in `free_run!`
- `docs/src/perturbed_equilibrium.md` — downstream singular coupling analysis using Δ'
