"""
    Utils

Shared numerical helpers for KineticForces: the generic sign-change root scan and the
kinetic-resonance surface locator built on it.
"""

"""
    find_sign_change_roots(f, grid) → Vector{Float64}

Locate the zeros of a callable `f` by scanning consecutive `grid` nodes for strict sign
changes (`f(x[i])·f(x[i+1]) < 0`) and refining each bracket with `Roots.Brent`. Returns
the refined roots in grid order (empty if `f` never changes sign; a node value of exactly
zero is not treated as a crossing).

Single source of truth for the scan-then-Brent idiom — used for dB/dθ extrema in the
bounce averaging and for kinetic-resonance surfaces in the ψ quadrature paneling.
"""
function find_sign_change_roots(f, grid)
    roots = Float64[]
    fprev = f(grid[firstindex(grid)])
    for i in (firstindex(grid)+1):lastindex(grid)
        fcur = f(grid[i])
        if fprev * fcur < 0
            push!(roots, find_zero(f, (grid[i-1], grid[i]), Roots.Brent()))
        end
        fprev = fcur
    end
    return roots
end

"""
    kinetic_resonance_psi_nodes(kinetic_profiles) → Vector{Float64}

ψ_N locations of kinetic resonance surfaces, for use as ψ-quadrature panel boundaries
(and, per the kinetic-aware grid-packing plan, as mandatory equilibrium knots).

Currently locates the ℓ = 0 superbanana-plateau resonance ω_E(ψ) = 0, where the ExB
frequency in the resonance denominator Ω(x) = ℓ_eff·ω_b·√x + n·(ω_E + ω_d·x) vanishes at
low energy and the NTV torque density peaks. The bounce/precession terms (cylindrical
ω_b, ω_d estimates at thermal energy x = 1, Logan & Park 2013 §IV–V) are a planned
extension behind additional keyword arguments; both consumers call this function unchanged.
"""
function kinetic_resonance_psi_nodes(kinetic_profiles::Equilibrium.KineticProfileSplines)
    return find_sign_change_roots(kinetic_profiles.omegaE_spline, kinetic_profiles.xs)
end
