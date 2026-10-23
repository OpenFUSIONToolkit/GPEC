# Problem.jl
#
# The LocalStability stage posed as solve(problem, algorithm): Mercier D_I, resistive D_R,
# ballooning Δ' and the first ballooning α boundary over the equilibrium ψ grid.

"""
    LocalStabilityProblem(equil)

Local high-n stability of an equilibrium: the Mercier and resistive interchange criteria and
the ballooning Δ' on every flux surface, and the first ballooning α boundary. Solve with
`solve(prob, Ballooning())` to get a [`LocalStabilityResult`](@ref). A deck poses the same
problem through its `[LocalStability]` section.

## Fields

  - `equil::Equilibrium.PlasmaEquilibrium` - The equilibrium the profiles are computed on.
"""
struct LocalStabilityProblem
    equil::Equilibrium.PlasmaEquilibrium
end

"""
    Ballooning(; theta_k=0.0, n_scan=24, alpha_boundary_scan=true, verbose=true)

The local-stability algorithm: the ballooning coefficient system per flux surface, plus the
BALOO-style α-boundary scan. Its keywords are the `[LocalStability]` deck keys.

## Fields

  - `theta_k::Float64` - Ballooning angle θ_k, used by both the profile scan and the α-boundary scan.
  - `n_scan::Int` - Resolution of the α scan for the first boundary (see [`ballooning_alpha_boundary`](@ref)).
  - `alpha_boundary_scan::Bool` - Run the α-boundary scan, a root find per ψ surface; `false` leaves the α fields empty.
  - `verbose::Bool` - Print progress and per-surface α-scan failures.
"""
@kwdef struct Ballooning
    theta_k::Float64 = 0.0
    n_scan::Int = 24
    alpha_boundary_scan::Bool = true
    verbose::Bool = true
end

"""
    LocalStabilityResult

The local-stability profiles returned by `solve(::LocalStabilityProblem, ::Ballooning)`.

## Fields

  - `psi::Vector{Float64}` - Normalized flux grid of the profiles (the equilibrium `profiles.xs`).
  - `D_I::Vector{Float64}` - Mercier interchange criterion `det(d0bar)`; stable where negative.
  - `D_R::Vector{Float64}` - Resistive interchange criterion `D_I + (H - 1/2)²` (see [`resistive_interchange_h`](@ref)).
  - `ballooning_Delta_prime::Vector{Float64}` - High-n ballooning Δ', zero for ψ > 1.
  - `ballooning_psi::Vector{Float64}` - Grid of the α-boundary scan; empty when the scan is off.
  - `alpha::Vector{Float64}` - Experimental α on `ballooning_psi`, NaN outside the scan window.
  - `alpha_critical::Vector{Float64}` - First-boundary critical α on `ballooning_psi`, NaN outside the scan window or with no boundary.
  - `di_psi` - Cubic interpolant of `D_I·ψ` over `psi`, for evaluating `D_I` off the grid (see [`mercier_D_I`](@ref)).
"""
struct LocalStabilityResult{S}
    psi::Vector{Float64}
    D_I::Vector{Float64}
    D_R::Vector{Float64}
    ballooning_Delta_prime::Vector{Float64}
    ballooning_psi::Vector{Float64}
    alpha::Vector{Float64}
    alpha_critical::Vector{Float64}
    di_psi::S
end

"""
    mercier_D_I(result::LocalStabilityResult, psi) -> Float64

Mercier `D_I` at `psi`, interpolated through the smooth `D_I·ψ` profile rather than `D_I` itself.
"""
mercier_D_I(result::LocalStabilityResult, psi::Real) = result.di_psi(psi)[1] / psi

function CommonSolve.solve(prob::LocalStabilityProblem, alg::Ballooning)
    equil = prob.equil
    xs = equil.profiles.xs
    locstab_fs = zeros(Float64, length(xs), 5)
    compute_ballooning_stability!(locstab_fs, equil; theta_k=alg.theta_k, verbose=alg.verbose)
    boundary = if alg.alpha_boundary_scan
        ballooning_alpha_boundary(equil; theta_k=alg.theta_k, n_scan=alg.n_scan, verbose=alg.verbose)
    else
        (psi=Float64[], alpha=Float64[], alpha_critical=Float64[])
    end
    # A Series spline, not a plain-vector one: the two differ in the last bits off the knots.
    di_psi = cubic_interp(xs, Series(locstab_fs[:, 1:1]); extrap=ExtendExtrap())
    return LocalStabilityResult(collect(xs), locstab_fs[:, 1] ./ xs, locstab_fs[:, 2] ./ xs, locstab_fs[:, 4],
        boundary.psi, boundary.alpha, boundary.alpha_critical, di_psi)
end
