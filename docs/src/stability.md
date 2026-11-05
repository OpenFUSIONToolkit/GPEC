# Stability

`ForceFreeStates` determines the ideal MHD stability of the equilibrium and produces the force-free states, the displacement fields that every later stage builds on.
It implements the direct Newcomb criterion of the Fortran DCON ([Glasser (2016)](citations.md#ForceFreeStates)): rather than solving an eigenvalue problem for the growth rate, it integrates the Euler–Lagrange equations of the perturbed potential energy across the plasma and reads stability from the solutions themselves.

## What it computes

For a toroidal mode number n, the perturbation is expanded in poloidal harmonics m over a window around the resonant range, and the displacement ``\xi(\psi)`` becomes a vector of harmonic amplitudes.
The ideal MHD potential energy ``\delta W[\xi]`` is a quadratic functional of ``\xi``, and its extrema satisfy a linear system of ODEs in ``\psi`` whose coefficients are singular at every rational surface q = m/n, where the resonant harmonic decouples from the field-line bending.
Integrating that system from the axis outward with the prescribed crossing of each rational surface gives the force-free states: the set of displacements that satisfy force balance everywhere inside the plasma for arbitrary boundary values.

Three results follow from them:

- **Fixed-boundary stability.** A zero of the determinant of the solution matrix between the axis and the edge means a displacement vanishing at the boundary with negative energy, so the plasma is unstable even with a perfectly conducting wall on its surface (the `crit` scan).
- **Free-boundary stability.** With `vac_flag = true`, the [Vacuum](vacuum.md) module supplies the vacuum energy of the `[Wall]` section (no wall when the section is absent), and the total energy ``W = W_\mathrm{plasma} + W_\mathrm{vacuum}`` on the control surface is a Hermitian matrix in the boundary harmonics. Its generalized eigenvalues with respect to the power norm are the free-boundary eigenmode energies; a negative least eigenvalue means an ideal external kink.
- **The tearing index Δ′.** Across each rational surface the solution has a large and a small asymptotic part. The Δ′ matrix of the outer region, assembled from the force-free states, gives the jump of the small solution driven by a unit tearing perturbation at each surface; it is the outer-region input of every resistive calculation ([Tearing](inner_layer.md), `[Match]`).

## Governing equations

Ideal stability is decided by the sign of the perturbed potential energy

```math
\delta W[\xi] = \int_0^{\psi_\mathrm{lim}} \mathcal{F}(\xi, \xi')\, d\psi ,
```

whose extrema over all admissible ``\xi`` satisfy a first-order system in the harmonic amplitudes ``U_1 = \xi`` and their conjugate momenta ``U_2`` (Glasser 2016, Eq. 24):

```math
\frac{d}{d\psi}\begin{pmatrix} U_1 \\ U_2 \end{pmatrix}
= \mathbf{L}(\psi)\begin{pmatrix} U_1 \\ U_2 \end{pmatrix},
```

where ``\mathbf{L}`` is built from the Fourier-space metric matrices ``\bar{F}``, ``\bar{K}`` and ``\bar{G}`` of the equilibrium and the resonance factor ``Q = \mathrm{diag}(m - nq)``.
The resonance factor vanishes at each rational surface, so the solutions there behave as ``|\psi - \psi_s|^{-1/2 \pm \sqrt{-D_I}}``, a large and a small solution governed by the Mercier index ``D_I``; GPEC assembles the right-hand side from reduced matrices that carry explicit ``(m - nq)`` factors rather than their inverses, the convention used on every page.
The Newcomb criterion states that the plasma is fixed-boundary stable if and only if the solution that is regular at the axis remains nonsingular across every rational surface and up to the edge.

The Riccati integrator works with the matrix ``S = U_1 U_2^{-1}`` ([Glasser, Kolemen and Glasser (2018)](citations.md#ForceFreeStates), Eq. 19), which obeys a quadratic matrix ODE and stays bounded where ``U_1`` and ``U_2`` grow exponentially, so that the rational surfaces can be crossed without the Gaussian reduction of the forward integration.
Its global boundary-value problem over all surfaces gives the Δ′ matrix of [Glasser and Kolemen (2018)](citations.md#Tearing), stored in the PEST-3 convention: the diagonal entry of each surface is its outer-region Δ′, positive when the outer region drives the tearing mode, and the off-diagonal entries couple the surfaces.

## Integrators

`[ForceFreeStates] integrator` selects the radial method; its numerical settings sit in a section of the same name.

- `"Riccati"` (the default; `[Riccati]`) splits the domain into independent chunks integrated in parallel, crosses each rational surface in the Riccati basis and assembles the Δ′ matrix. It stores the chunk endpoints only, so it yields no dense ``\xi`` profiles.
- `"Forward"` (`[Forward]`) integrates the system directly in ``(U_1, U_2)`` with Gaussian reduction at each rational surface. It returns dense displacement profiles, and is the only integrator whose result `plasma_response` accepts (the response operators are built from the dense ``\xi`` solution; a wall is refused too) and the only one that supports the kinetic solve.
- `"Galerkin"` (`[Galerkin]`) solves the outer region variationally on a finite-element basis, giving the resistive Δ′ matrix and the PEST-3 blocks but no free-boundary energies; see [Galerkin Solver](galerkin.md).

`kinetic = "KineticForces"` replaces the ideal energy matrices by their drift-kinetic form, which needs `[Equilibrium] kinetic_file`; see [Kinetic Forces](kinetic_forces.md).

## Running it

From `examples/DIIID-like_ideal_example/gpec.toml` (abridged):

```toml
[ForceFreeStates]
nn = 1                   # toroidal mode number, or consecutive numbers as a list ([1, 2, 3])
vac_flag = true          # free-boundary energies with the [Wall] vacuum response
qlow = 1.02              # inner edge of the integration domain
qhigh = 1e3              # outer bound of qlim (see Integration domain)
delta_mlow = 8           # extra poloidal modes below the resonant range
delta_mhigh = 8          # extra poloidal modes above it
integrator = "Forward"   # "Riccati" is the default; "Forward" (and no [Wall]) is required for [PerturbedEquilibrium]
truncate_at_rational_offset = true
rational_offset_fraction = 0.2

[Forward]
eulerlagrange_tolerance = 1e-10
save_interval = 1        # save every Nth ODE step; steps near rational surfaces are always saved
renorm_threshold = 1e3   # renormalize the solution basis above this column norm
```

`examples/DIIID-like_riccati_deltaprime_example` sets the same three keys under `[Riccati]`.

The same solve from a script takes the deck keys as keywords of the problem and the integrator:

```julia
using GeneralizedPerturbedEquilibrium

eq  = PlasmaEquilibrium("examples/DIIID-like_ideal_example/TkMkr_D3Dlike_Hmode.geqdsk"; jac_type="hamada")
ffs = solve(eq, Riccati(; eulerlagrange_tolerance=1e-10); nn=1, delta_mlow=8, delta_mhigh=8, vac_flag=true, qlow=1.02)

real(ffs.free_boundary.et[1])                 # least free-boundary eigenmode energy: negative is unstable
for (s, sing) in enumerate(ffs.surfaces)
    println("q = $(sing.m[1])/$(sing.n[1]) at ψ_N = $(sing.psifac): Δ′ = $(real(ffs.delta_prime.matrix[s, s]))")
end
```

`main(["examples/DIIID-like_riccati_deltaprime_example"]).ffs` returns the same result from a deck.
The number of Julia threads is set at startup (`-t N` or `JULIA_NUM_THREADS`); the chunking never depends on it, so threads change the wall-clock time only.

## Outputs

| Dataset | Meaning |
|---|---|
| `ForceFreeStates/FreeBoundaryStability/W_plasma`, `W_vacuum`, `W_freeboundary` | plasma, vacuum and total energy matrices on the control surface, power-normalized (per unit ⟨\|ξ\|²⟩) |
| `ForceFreeStates/FreeBoundaryStability/eigenmode_energies` | generalized eigenvalues of ``W``, most unstable first; `eigenmode_plasma_energies` and `eigenmode_vacuum_energies` split each one |
| `ForceFreeStates/FreeBoundaryStability/W_freeboundary_eigenmodes` | the eigenvectors, unit power norm |
| `ForceFreeStates/FreeBoundaryStability/vacuum_eigenvalue` | least eigenvalue of the vacuum energy matrix |
| `ForceFreeStates/EdgeScan/` | `total_energy`, `plasma_energy`, `vacuum_energy` and `vacuum_eigenvalue` against `psi` and `q` as the control surface moves outward from `dW_edge_scan_start` |
| `ForceFreeStates/Solutions/ForwardIntegration/` | the forward integrator's `psi`, `q`, `xi_psi`, `xi_s`, `dxi_psidpsi` profiles and the `crit` scan whose zero crossings mark fixed-boundary instability |
| `ForceFreeStates/EulerLagrangeMatrices/` | the ``\bar{F}``, ``\bar{K}``, ``\bar{G}`` matrices on `psi`, written when `write_el_matrices = true` |
| `ForceFreeStates/LayerOverlap/` | the resistive-layer overlap scan, when `psilim_from_layer_overlap` is set (table below) |
| `SingularSurfaces/rational_psi`, `rational_q`, `rational_m`, `rational_n`, `dqdpsi` | the rational surfaces of the domain, core to edge |
| `SingularSurfaces/D_I`, `E`, `F`, `G`, `H`, `K`, `M` | the Mercier index and the Glasser–Greene–Johnson layer coefficients at each surface |
| `SingularSurfaces/ca_left`, `ca_right` | the asymptotic (large, small) coefficients on either side of each surface, from ideal crossings |
| `SingularSurfaces/Delta_prime_matrix` | the Δ′ matrix in the PEST-3 convention, from `"Riccati"` or `"Galerkin"`; `Delta_prime_raw` is the side-major (left, right) matrix it is folded from, `Delta_coil` the edge coil-response columns |
| `SingularSurfaces/Kinetic/` | the surfaces of a kinetic solve |

The energy matrices are power-normalized: the eigenmode energies are the generalized eigenvalues of the pencil ``W v = \lambda N v``, with ``N`` the surface-norm matrix ``\xi^\dagger N \xi = \langle|\xi|^2\rangle`` built from the Jacobian Fourier coefficients, so each is an energy per unit surface-averaged squared displacement rather than joules, and the spectrum is invariant to the working coordinate because ``W`` and ``N`` transform by the same congruence.
`Info/psilim` and `Info/qlim` record the outer edge of the integration domain.

## Integration domain

The integration runs from the axis (`qlow` sets the inner edge) to `psilim`, the flux of the target safety factor `qlim`.
`qlim` starts at `min(qmax, qhigh)`, with `qmax` the value of q at the equilibrium's `psihigh`.
With `truncate_at_rational_offset = true` (the default, for diverted equilibria where q diverges at the separatrix), `qlim` moves to `(m_last + rational_offset_fraction)/n`, a fixed fraction of a rational spacing past the last surface, stepped back by `1/n` until it no longer exceeds the bound it started from; a bound that falls on a rational, or less than `rational_offset_fraction/n` above one, therefore excludes that surface.
With `truncate_at_rational_offset = false` (limited or analytic equilibria with finite edge q), `qhigh` and `psihigh` set the domain directly.

### Resistive-layer overlap cap

Near the separatrix the rational surfaces crowd faster than their resistive layers narrow, and past the point where adjacent layers touch the matched-asymptotic treatment is not defined (the diverted-plasma response paper of Fitzpatrick (2025), listed under [Tearing](citations.md#Tearing)).
`psilim_from_layer_overlap = true` locates that point: each rational surface of each n in the run, including surfaces beyond the equilibrium grid placed on the logarithmic divergence of q at the X-points, gets a layer width from the kinetic profiles with the closures of `[SLAYER]` (a `SLAYERModel` in a script), and the domain is cut at the last surface before two adjacent layers overlap.
The cap needs kinetic profiles, acts before the `rational_offset_fraction` step, and can only narrow the domain; `examples/DIIID-like_truncation_example` exercises it.
The scan is written to `ForceFreeStates/LayerOverlap/`:

| Dataset | Contents |
|---|---|
| `psi`, `m`, `n`, `r_s` | scanned surfaces and their minor radius √(2ψ_t/B₀) |
| `delta_s_abs`, `width_delta_s` | Riccati layer thickness, in metres and as a Δψ width |
| `width_visco`, `width_dr` | visco-resistive and diffusive-resistive Δψ widths; `width_dr` sets the cut |
| `extrapolated` | 1 where the surface lies beyond the equilibrium grid |
| `psilim_overlap`, `first_overlap_index` | the cut, and the first surface whose layer overlaps its neighbour |
| `applied` | 1 when the cap was passed to the truncation |

The perpendicular diffusivities that set the layer widths are described with the SLAYER inputs on the [Tearing](inner_layer.md) page.

## Numerical notes

- The Riccati integrator splits the domain into chunks whose count is derived from the number of singular surfaces (`[Riccati] nchunks` overrides the target, `0` is automatic); every chunk integrates from identity initial conditions, so the result is independent of the thread count.
- The chunk ending at each rational surface is integrated backward, from the surface inward, because the forward propagator is ill-conditioned there; the Δ′ boundary-value problem is assembled from the split forward and backward propagators for the same reason.
- Both integrators use the adaptive ninth-order `Vern9` solver at `eulerlagrange_tolerance`; the Riccati basis is recovered by renormalizing whenever the solution exceeds `renorm_threshold`.
- The two crossing conventions differ slightly, so Riccati and Forward free-boundary energies agree to within a few percent, not to machine precision.

Local stability (Mercier, resistive interchange and ballooning) is a separate stage on the same equilibrium; see [Local Stability](ballooning.md).

## API reference

The Galerkin solver's functions are documented on the [Galerkin Solver](galerkin.md) page.

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ForceFreeStates]
Pages = ["ForceFreeStates.jl", "CoreTypes.jl", "Surfaces/Types.jl", "Riccati/Types.jl", "Matching/DeltaPrime.jl", "Result.jl", "Surfaces/ResistEval.jl", "Matching/ResonantMatch.jl", "Matching/MatchProblem.jl", "EulerLagrange.jl", "Surfaces/Finding.jl", "Surfaces/Asymptotics.jl", "Fourfit.jl", "Kinetic.jl", "FixedKineticMatrices.jl", "FixedBoundaryStability.jl", "Utils.jl", "Free.jl", "Riccati/Propagators.jl", "Riccati/Crossings.jl", "Riccati/DeltaPrimeBVP.jl", "Riccati/Driver.jl"]
```
