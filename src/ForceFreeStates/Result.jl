"""
    DeltaPrimeData

Δ′ products of the Riccati/STRIDE boundary-value problem, moved off the solve-time
`ForceFreeStatesInternal` scratch when the result is assembled. Present only when the
Riccati integrator ran with a vacuum edge condition and at least one singular surface.

## Fields

  - `matrix::Matrix{ComplexF64}` - Inter-surface Δ′ of shape (msing × msing) in PEST3
    convention, the odd-parity tearing projection of `raw`.
  - `raw::Matrix{ComplexF64}` - Raw outer-region matching matrix D′ of shape
    (2msing × 2msing) in the side-major ordering `[L_s1, R_s1, L_s2, R_s2, …]`.
  - `coil::Matrix{ComplexF64}` - Edge coil-response matrix of shape
    (2msing × numpert_total); column k is the resonant small-solution response at each
    surface side to a unit source on edge poloidal mode k. Empty when the vacuum-edge BVP
    was not assembled.
"""
struct DeltaPrimeData
    matrix::Matrix{ComplexF64}
    raw::Matrix{ComplexF64}
    coil::Matrix{ComplexF64}
end

"""
    SolutionProfiles

The solve's ξ solution on its radial grid, in the exact shape PerturbedEquilibrium consumes.
Field names mirror the `OdeState` store subset so the consumers read the same names whichever
formalism produced the solution.

## Fields

  - `basis::Symbol` - Which formalism's basis the profiles are in: `:el_axis` (forward
    integrator, axis Euler-Lagrange basis) or `:gal_native` (inner-layer-matched Galerkin
    solution, identity-at-edge basis).
  - `step::Int` - Number of stored radial nodes.
  - `psi_store::Vector{Float64}` - ψ at each node.
  - `q_store::Vector{Float64}` - Safety factor at each node.
  - `u_store::Array{ComplexF64,4}` - `(N, N, 2, step)` solution state: Ξ_ψ in the first
    component and its conjugate momentum in the second.
  - `du_store::Array{ComplexF64,3}` - `(N, N, step)` dΞ_ψ/dψ.
  - `xi_s_store::Array{ComplexF64,3}` - `(N, N, step)` Clebsch displacement Ξ_s.
"""
struct SolutionProfiles
    basis::Symbol
    step::Int
    psi_store::Vector{Float64}
    q_store::Vector{Float64}
    u_store::Array{ComplexF64,4}
    du_store::Array{ComplexF64,3}
    xi_s_store::Array{ComplexF64,3}
end

"""
    ForceFreeStatesResult

The published product of a force-free-states solve: everything downstream stages
(PerturbedEquilibrium, KineticForces, SLAYER, the HDF5 writer) are allowed to read.
`ForceFreeStatesInternal` remains solve-time scratch and does not cross a module boundary
once this has been built.

Optional fields are `Union{Nothing,T}` and follow a presence-equals-capability rule: a
consumer that needs one gates on [`require`](@ref) (or [`require_solution`](@ref) for the
ξ profiles) and warns-and-skips rather than erroring, so a result from an integrator that
cannot supply a given product still flows through the pipeline.

A jump condition at the rational surfaces is always applied before a final result is built,
so bpen and closure are always present. 

## Fields

  - `integrator::Symbol` - `:forward`, `:riccati`, or `:galerkin`.
  - `control::ForceFreeStatesControl` - Provenance snapshot of the controls the solve ran with.
  - `equil::Equilibrium.PlasmaEquilibrium` - The equilibrium actually integrated against (the re-formed one on the two-pass path).
  - `mlow`, `mhigh`, `mpert`, `nlow`, `nhigh`, `npert`, `numpert_total` - Resolved (m, n) mode space; see `ModeSpace`.
  - `psilow`, `psilim`, `qlim`, `q1lim` - Integration bounds, and q with its ψ-derivative at `psilim`.
  - `dir_path::String` - Working directory of the run.
  - `wall_settings::Vacuum.WallShapeSettings` - Wall shape used by the vacuum calculation.
  - `metric::MetricData`, `ffit::FourFitVars` - Metric data and Euler-Lagrange matrix interpolants.
  - `surfaces::Vector{SingType}` - Ideal singular surfaces in the integration domain, with asymptotic bases and GGJ coefficients.
  - `kinetic::NamedTuple` - Kinetic singular-surface scan (`kmsing`, `kinsing`, `scan_psi`, `scan_cond`, `scan_threshold`); empty unless the finder ran.
  - `closure::Symbol` - How the basis is closed at the rationals: `:ideal` (the ideal jump
    condition was imposed) or `:matched` (an inner-layer solution was matched in).
  - `bpen::Matrix{ComplexF64}` - `(msing × numpert_total)` penetrated resonant field from the
    inner layer, per surface and driving mode; all zeros under `:ideal` closure.
  - `solution::Union{Nothing,SolutionProfiles}` - THE solve's ξ solution; `nothing` when the
    formalism produced none (Riccati, and Galerkin without a match).
  - `diagnostics::Union{Nothing,OdeState}` - The integrator's raw ODE state (ψ trace, `crit`,
    edge scan, asymptotic coefficients). Written to HDF5; not a consumable solution.
  - `wp::Union{Nothing,Matrix{ComplexF64}}` - Fixed-boundary plasma energy matrix
    `W_p = (U₂·U₁⁻¹)/ψ₀²` at `psilim`. Present for any Euler-Lagrange sweep (forward or
    Riccati) regardless of `vac_flag` — the fixed-boundary run's energy product — and
    identical to `free_boundary.wp` when the free-boundary calculation ran.
  - `free_boundary::Union{Nothing,FreeBoundaryResult}` - Free-boundary energies and eigenmodes; `nothing` when the vacuum step was skipped or the formalism computes none.
  - `delta_prime::Union{Nothing,DeltaPrimeData}` - STRIDE Δ′ products; `nothing` unless the Riccati BVP produced them.
  - `galerkin::Union{Nothing,GalerkinResult}` - RDCON Galerkin solve, including the RPEC inner-layer match when requested.
"""
struct ForceFreeStatesResult{E<:Equilibrium.PlasmaEquilibrium,F<:FourFitVars} <: ModeSpace
    integrator::Symbol
    control::ForceFreeStatesControl
    equil::E

    # Mode space and integration domain, copied out of the solve-time scratch.
    mlow::Int
    mhigh::Int
    mpert::Int
    nlow::Int
    nhigh::Int
    npert::Int
    numpert_total::Int
    psilow::Float64
    psilim::Float64
    qlim::Float64
    q1lim::Float64
    dir_path::String
    wall_settings::Vacuum.WallShapeSettings

    # Assembly products, always present.
    metric::MetricData
    ffit::F
    surfaces::Vector{SingType}
    kinetic::@NamedTuple{kmsing::Int, kinsing::Vector{SingType}, scan_psi::Vector{Float64}, scan_cond::Vector{Float64}, scan_threshold::Float64}

    # Closure of the basis at the rationals, always present.
    closure::Symbol
    bpen::Matrix{ComplexF64}

    # Per-formalism products; presence is the capability signal.
    solution::Union{Nothing,SolutionProfiles}
    diagnostics::Union{Nothing,OdeState}
    wp::Union{Nothing,Matrix{ComplexF64}}
    free_boundary::Union{Nothing,FreeBoundaryResult}
    delta_prime::Union{Nothing,DeltaPrimeData}
    galerkin::Union{Nothing,GalerkinResult}
end

"""
    require(result::ForceFreeStatesResult, field::Symbol, calc::AbstractString) -> Bool

Warn-and-skip gate: `true` iff the optional field `field` was populated by the integrator
that produced `result`. Warns naming the calculation being skipped otherwise.
"""
function require(result::ForceFreeStatesResult, field::Symbol, calc::AbstractString)
    getfield(result, field) === nothing || return true
    @warn "Skipping $calc: `$field` was not produced by the $(result.integrator) integrator"
    return false
end

"""
    require_solution(result::ForceFreeStatesResult, calc::AbstractString) -> Bool

Warn-and-skip gate for ξ-profile consumers: [`require`](@ref) specialized to `solution`,
with a message naming what a solution takes to produce.
"""
function require_solution(result::ForceFreeStatesResult, calc::AbstractString)
    result.solution === nothing || return true
    @warn "Skipping $calc: no ξ solution — dense profiles require a Forward (or matched Galerkin) run; " *
          "this result came from the $(result.integrator) integrator"
    return false
end

# Pack the RPEC-matched Galerkin solution into `SolutionProfiles`. Mirrors Fortran
# `idcon_build`'s gal branch (idcon.f) and `globalsol.bin` (match.f): the on-surface `issing`
# grid points are dropped as zero placeholders where the resonant series diverges, Ξ′ is the
# analytic Galerkin derivative rather than a differenced value spline, and
# Ξ_s = −A⁻¹(B·Ξ′ + C·Ξ) is the same outer ideal-MHD relation `sing_der!` uses. The grid runs
# inner→edge, so the last node is the control surface and carries the edge boundary condition.
function _matched_gal_profiles(gal_result::GalerkinResult, ffit::FourFitVars, intr::ModeSpace)
    sol = gal_result.solution
    m = gal_result.match
    npert = intr.numpert_total

    keep = .!sol.issing
    psi_f = sol.psi[keep]
    q_f = sol.q[keep]
    xi_f = m.xi[:, keep, :]          # (npert, ngrid_f, mcoil)
    dxi_f = m.xi_deriv[:, keep, :]
    ngrid_f = length(psi_f)

    # u_store[:, :, 2] stays zero: the conjugate momentum has no consumer on this path and no
    # Galerkin counterpart (matches Fortran's unused u2 in the gal branch).
    u_store = zeros(ComplexF64, npert, npert, 2, ngrid_f)
    du_store = zeros(ComplexF64, npert, npert, ngrid_f)
    xi_s_store = zeros(ComplexF64, npert, npert, ngrid_f)

    hint = Ref(1)
    for ip in 1:ngrid_f
        ξ = @view xi_f[:, ip, :]
        ξ′ = @view dxi_f[:, ip, :]
        @views u_store[:, :, 1, ip] .= ξ
        @views du_store[:, :, ip] .= ξ′
        @views compute_node_xi_s!(xi_s_store[:, :, ip], ξ′, ξ, ffit, psi_f[ip]; hint=hint)
    end

    return SolutionProfiles(:gal_native, ngrid_f, psi_f, q_f, u_store, du_store, xi_s_store)
end

"""
    build_result(integrator, ctrl, equil, intr, metric, ffit, odet, free_energies, gal_data)
        -> ForceFreeStatesResult

Assemble the published result once the solve is finished — the one place that decides what a
formalism's raw output means downstream. Beyond materializing the forward path's derivative
stores, packing the matched Galerkin solution, and forming the fixed-boundary `W_p` when the
free-boundary stage did not run, every field is copied or aliased from what the stages
already produced.

`odet` is the integrator's ODE state (`nothing` for Galerkin); `gal_data` the Galerkin solve
(`nothing` otherwise). The two are never both present: additive Galerkin does not exist.
"""
function build_result(
    integrator::Symbol,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    intr::ForceFreeStatesInternal,
    metric::MetricData,
    ffit::FourFitVars,
    odet::Union{Nothing,OdeState},
    free_energies::Union{Nothing,FreeBoundaryResult},
    gal_data::Union{Nothing,GalerkinResult}
)
    matched = gal_data !== nothing && gal_data.match !== nothing

    # The forward sweep is the only formalism whose stores need materializing; doing it here
    # keeps `SolutionProfiles.du_store`/`xi_s_store` populated by construction.
    solution = if integrator === :forward && odet !== nothing
        materialize_derivative_stores!(odet, equil, ffit, intr)
        SolutionProfiles(:el_axis, odet.step, odet.psi_store, odet.q_store,
            odet.u_store, odet.du_store, odet.xi_s_store)
    elseif matched
        _matched_gal_profiles(gal_data, ffit, intr)
    else
        nothing
    end

    delta_prime = isempty(intr.delta_prime_matrix) ? nothing :
                  DeltaPrimeData(intr.delta_prime_matrix, intr.delta_prime_raw, intr.delta_coil)
    kinetic = (kmsing=intr.kmsing, kinsing=intr.kinsing, scan_psi=intr.kinsing_scan_psi,
        scan_cond=intr.kinsing_scan_cond, scan_threshold=intr.kinsing_scan_threshold)

    # The ideal-flag match deliberately skips the inner-layer Δ, so its basis is ideal-closed
    # and carries no penetrated field.
    closure = (matched && !ctrl.gal_ideal_flag) ? :matched : :ideal
    bpen = closure === :matched ? gal_data.match.bpen :
           zeros(ComplexF64, intr.msing, intr.numpert_total)

    # Fixed-boundary plasma energy matrix at the edge; free of any vacuum dependence, so a
    # vac_flag=false run still publishes its energy product. Aliases free_run's when it ran.
    wp = if free_energies !== nothing
        free_energies.wp
    elseif odet !== nothing
        (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2
    else
        nothing
    end

    return ForceFreeStatesResult(
        integrator, ctrl, equil,
        intr.mlow, intr.mhigh, intr.mpert, intr.nlow, intr.nhigh, intr.npert, intr.numpert_total,
        intr.psilow, intr.psilim, intr.qlim, intr.q1lim, intr.dir_path, intr.wall_settings,
        metric, ffit, intr.sing, kinetic,
        closure, bpen,
        solution, odet, wp, free_energies, delta_prime, gal_data
    )
end
