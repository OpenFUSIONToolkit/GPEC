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
    _resonance_nodes_from_frequencies(wbhat_f, welec_f, wdhat_f, grid; n, nl, xeval=2.5) → Vector{Float64}

Scan `grid` for the zeros of the resonance operator
`Ω_ℓ(x; ψ) = ℓ·ω_b(ψ)·√x + n·(ω_E(ψ) + ω_d(ψ)·x)` at energy `x = xeval`, for every bounce
harmonic ℓ ∈ −nl:nl, given per-ψ frequency callables. `xeval` defaults to 2.5 — the peak of
the Maxwellian-weighted drive `x^2.5·e^−x`, where the resonance overlaps the most particles,
so the located ψ surfaces sit on the NTV torque-density peaks (on the DIII-D case the x=2.5
nodes land ~3× closer to the measured `dT/dψ` spikes than the thermal-energy x=1 estimate).
Roots from all harmonics are concatenated (deduplication against coincident surfaces happens
in `psi_panel_points`).
"""
function _resonance_nodes_from_frequencies(wbhat_f, welec_f, wdhat_f, grid; n::Int, nl::Int, xeval::Float64=2.5)
    nodes = Float64[]
    sx = sqrt(xeval)
    for l in (-nl):nl
        append!(nodes, find_sign_change_roots(psi -> l * wbhat_f(psi) * sx + n * (welec_f(psi) + wdhat_f(psi) * xeval), grid))
    end
    return nodes
end

"""
    kinetic_resonance_psi_nodes(kinetic_profiles, equil; n, nl, zi=1, mi=2, electron=false, wdfac=1.0, xeval=2.5) → Vector{Float64}

ψ_N locations of kinetic resonance surfaces, for use as ψ-quadrature panel boundaries
(and, per the kinetic-aware grid-packing plan, as mandatory equilibrium knots).

Locates the zeros of the trapped-branch (leff = ℓ) resonance denominator
`Ω(x) = leff·ω_b·√x + n·(ω_E + ω_d·x)` at energy `x = xeval`, for every bounce harmonic
ℓ ∈ −nl:nl — this is where the energy-space resonance sweeps through the drive-weighted bulk
and the NTV torque density peaks (Logan & Park, Phys. Plasmas 20, 122507 (2013), §IV–V).
`xeval` defaults to 2.5, the peak of the Maxwellian-weighted drive `x^2.5·e^−x`; on the
DIII-D case those nodes sit ~3× closer to the measured `dT/dψ` spikes than the thermal x=1
estimate. The ℓ = 0 node is the ω_d-shifted ExB (superbanana-plateau) resonance.

The frequencies use the pitch-averaged large-aspect-ratio closed forms of the `rlar`
method (`tpsi!` in Torque.jl / Fortran pentrc torque.F90): `ω_b = (π/4)·√(ε/2)·wtran` and
`ω_d = q·T_s/(2·ε·R₀²·Z·e·B₀)·wdfac`, with ε = ⟨r⟩/⟨R⟩ clamped away from the axis where
the estimate degenerates. Panel placement only needs ~peak-width accuracy, so these cheap
estimates (single spline evaluations) are sufficient and no bounce averaging is performed.
"""
function kinetic_resonance_psi_nodes(kinetic_profiles::Equilibrium.KineticProfileSplines, equil;
    n::Int, nl::Int, zi::Int=1, mi::Int=2, electron::Bool=false, wdfac::Float64=1.0, xeval::Float64=2.5)
    chrg = electron ? -e : zi * e
    mass = electron ? me : mi * mp
    T_spline = electron ? kinetic_profiles.Te_spline : kinetic_profiles.Ti_spline
    ro = equil.ro
    bo = equil.params.b0
    q_spline = equil.profiles.q_spline
    avg_r = equil.geometry.avg_r_spline
    avg_R = equil.geometry.avg_R_spline

    # ε clamp (Fortran torque.F90 does not clamp): bounds wdhat ∝ 1/ε near the axis; paneling-only.
    epsr_f = psi -> max(avg_r(psi) / avg_R(psi), 1e-6)
    wbhat_f = psi -> (π / 4) * sqrt(epsr_f(psi) / 2) * sqrt(2 * T_spline(psi) / mass) / (q_spline(psi) * ro)
    wdhat_f = psi -> q_spline(psi) * T_spline(psi) / (2 * epsr_f(psi) * ro^2 * chrg * bo) * wdfac

    grid = filter(x -> x > 0, kinetic_profiles.xs)
    return _resonance_nodes_from_frequencies(wbhat_f, kinetic_profiles.omegaE_spline, wdhat_f, grid; n, nl, xeval)
end
