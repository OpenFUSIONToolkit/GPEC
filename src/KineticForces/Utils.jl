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
    for l in -nl:nl
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

"""
    kinetic_axis_validity_psi(kinetic_profiles, equil; zi=1, mi=2, electron=false) → Float64

ψ_N below which the zero-orbit-width drift-kinetic ordering fails: the outermost ψ on the
kinetic-profile grid where any of the three thermal orbit-width scales reaches the local minor
radius ⟨r⟩ — potato width `(q²ρ²R₀)^(1/3)`, banana width `q·ρ/√ε`, and poloidal gyroradius
`q·ρ/ε` (ε = ⟨r⟩/⟨R⟩, clamped as in `kinetic_resonance_psi_nodes`; thermal gyroradius
ρ = √(2·m·T)·/(Z·e·B₀) of the computed species). Inside this boundary trapped bananas become
potato orbits with width comparable to r itself, so the bounce-averaged kinetic response is
evaluated outside its validity domain (measured consequence: diverging kinetic increments and a
pathological EL step count). Returns 0.0 when no criterion is met anywhere. The Fortran precedent
(`ktanh_flag`, dcon/fourfit.F) suppressed the same region with four hand-tuned knobs; here the
boundary is derived from the profiles with no user parameters.
"""
function kinetic_axis_validity_psi(kinetic_profiles::Equilibrium.KineticProfileSplines, equil;
    zi::Int=1, mi::Int=2, electron::Bool=false)
    chrg = electron ? e : zi * e
    mass = electron ? me : mi * mp
    T_spline = electron ? kinetic_profiles.Te_spline : kinetic_profiles.Ti_spline
    q_spline = equil.profiles.q_spline
    avg_r = equil.geometry.avg_r_spline
    avg_R = equil.geometry.avg_R_spline
    ro = abs(equil.ro)
    bo = abs(equil.params.b0)
    psi_c = 0.0
    for psi in kinetic_profiles.xs
        psi <= 0 && continue
        r = avg_r(psi)
        r <= 0 && continue
        eps = max(r / avg_R(psi), 1e-6)
        rho = mass * sqrt(2 * T_spline(psi) / mass) / (abs(chrg) * bo)
        q = abs(q_spline(psi))
        w_orbit = max(cbrt(q^2 * rho^2 * ro), q * rho / sqrt(eps), q * rho / eps)
        w_orbit >= r && (psi_c = max(psi_c, psi))
    end
    return psi_c
end

"""
    kinetic_axis_validity_envelope(psi, psi_c) → Float64

C² quintic smoothstep for the near-axis kinetic suppression: 0 for ψ ≤ ψ_c (drift-kinetic model
invalid; kernel evaluation may be skipped), rising over `[ψ_c, 2ψ_c]`, 1 above. The transition
width is tied to ψ_c itself, so there is no independent width parameter. `psi_c ≤ 0` returns 1
(no suppression).
"""
function kinetic_axis_validity_envelope(psi::Float64, psi_c::Float64)
    psi_c <= 0 && return 1.0
    t = (psi - psi_c) / psi_c
    t <= 0 && return 0.0
    t >= 1 && return 1.0
    return t^3 * (10 + t * (6 * t - 15))
end
