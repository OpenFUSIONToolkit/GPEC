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
    orbit_widths(psi, T_spline, q_spline, avg_r, avg_R, mass, chrg, ro, bo)

Thermal orbit-width scales of one species at ψ: the minor radius ⟨r⟩ and inverse aspect ratio ε,
the gyroradius ρ = √(2mT)/(Z·e·B₀), and the three widths that bound the zero-orbit-width ordering —
potato (q²ρ²R₀)^(1/3), banana qρ/√ε, and poloidal gyroradius qρ/ε. Single source of truth for both
the validity boundary and the diagnostic profiles.
"""
@inline function orbit_widths(psi::Float64, T_spline, q_spline, avg_r, avg_R,
    mass::Float64, chrg::Float64, ro::Float64, bo::Float64)
    r = avg_r(psi)
    eps = max(r / avg_R(psi), 1e-6)
    rho = sqrt(2 * mass * T_spline(psi)) / (abs(chrg) * bo)
    q = abs(q_spline(psi))
    return (; r, eps, rho,
        w_potato=cbrt(q^2 * rho^2 * ro),
        rho_banana=q * rho / sqrt(eps),
        rho_theta=q * rho / eps)
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
        w = orbit_widths(psi, T_spline, q_spline, avg_r, avg_R, mass, chrg, ro, bo)
        w.r <= 0 && continue
        max(w.w_potato, w.rho_banana, w.rho_theta) >= w.r && (psi_c = max(psi_c, psi))
    end
    return psi_c
end

"""
    kinetic_axis_validity_psi(species, equil) → Float64

Near-axis validity boundary for a resolved multi-species set: the largest single-species ψ_c, so
suppression covers every ψ where *any* species violates the zero-orbit-width ordering. Heavier or
less-charged species carry wider orbits (ρ ∝ √(mT)/(Z e B₀)), so the maximum is normally set by the
main ion; impurities move it only if their √m/Z exceeds the main ion's.
"""
function kinetic_axis_validity_psi(species::AbstractVector{<:Equilibrium.ResolvedNTVSpecies}, equil)::Float64
    psi_c = 0.0
    for sp in species
        psi_c = max(psi_c, kinetic_axis_validity_psi(sp.profiles, equil; zi=sp.z, mi=sp.m, electron=sp.electron))
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

"""
    kinetic_validity_profiles(kinetic_profiles, equil; zi=1, mi=2, electron=false) → NamedTuple

Radial profiles of the drift-kinetic validity diagnostics, on the kinetic-profile ψ grid:
`psi`; the thermal orbit-width scales `rho_i` (gyroradius √(2mT)/(Z·e·B₀)), `rho_banana`
(q·ρ/√ε), `rho_theta` (poloidal gyroradius q·ρ/ε), `w_potato` ((q²ρ²R₀)^(1/3)); the local
geometry `r_minor` (⟨r⟩) and `d_separatrix` (⟨r⟩(1) − ⟨r⟩(ψ)); the profile gradient lengths
`L_p` and `L_q` (|X|·|dr/dψ|/|dX/dψ|, Inf where the profile is flat); the near-axis suppression
boundary `psi_c` (`kinetic_axis_validity_psi`) with its `envelope`; and `is_valid` — true where
every zero-orbit-width ordering holds at coefficient 1: max orbit width < r, ρ_banana < L_p and
L_q, and max(ρ_i, ρ_banana, ρ_θ) < d_separatrix. Validity is diagnostic only — nothing outside
the near-axis envelope is suppressed (the far edge and steep-gradient regions are flagged, not
zeroed, since they can dominate the physical NTV).
"""
function kinetic_validity_profiles(kinetic_profiles::Equilibrium.KineticProfileSplines, equil;
    zi::Int=1, mi::Int=2, electron::Bool=false)
    chrg = electron ? e : zi * e
    mass = electron ? me : mi * mp
    T_spline = electron ? kinetic_profiles.Te_spline : kinetic_profiles.Ti_spline
    q_spline, q_deriv = equil.profiles.q_spline, equil.profiles.q_deriv
    P_spline, P_deriv = equil.profiles.P_spline, equil.profiles.P_deriv
    avg_r = equil.geometry.avg_r_spline
    avg_R = equil.geometry.avg_R_spline
    r_deriv = deriv1(avg_r)
    ro = abs(equil.ro)
    bo = abs(equil.params.b0)
    r_sep = avg_r(1.0)
    psi_c = kinetic_axis_validity_psi(kinetic_profiles, equil; zi=zi, mi=mi, electron=electron)

    psi = [x for x in kinetic_profiles.xs if 0 < x <= 1]
    n = length(psi)
    rho_i = zeros(n)
    rho_banana = zeros(n)
    rho_theta = zeros(n)
    w_potato = zeros(n)
    r_minor = zeros(n)
    L_p = zeros(n)
    L_q = zeros(n)
    d_separatrix = zeros(n)
    envelope = zeros(n)
    is_valid = falses(n)
    for (i, x) in pairs(psi)
        r = avg_r(x)
        w = orbit_widths(x, T_spline, q_spline, avg_r, avg_R, mass, chrg, ro, bo)
        rho_i[i] = w.rho
        rho_banana[i] = w.rho_banana
        rho_theta[i] = w.rho_theta
        w_potato[i] = w.w_potato
        q = abs(q_spline(x))
        drdpsi = abs(r_deriv(x))
        r_minor[i] = r
        d_separatrix[i] = max(r_sep - r, 0.0)
        dP = abs(P_deriv(x))
        L_p[i] = dP > 0 ? abs(P_spline(x)) * drdpsi / dP : Inf
        dq = abs(q_deriv(x))
        L_q[i] = dq > 0 ? q * drdpsi / dq : Inf
        envelope[i] = kinetic_axis_validity_envelope(x, psi_c)
        w_orbit = max(w_potato[i], rho_banana[i], rho_theta[i])
        is_valid[i] = w_orbit < r && rho_banana[i] < L_p[i] && rho_banana[i] < L_q[i] &&
                      max(rho_i[i], rho_banana[i], rho_theta[i]) < d_separatrix[i]
    end
    return (psi=psi, rho_i=rho_i, rho_banana=rho_banana, rho_theta=rho_theta, w_potato=w_potato,
        r_minor=r_minor, L_p=L_p, L_q=L_q, d_separatrix=d_separatrix,
        psi_c=psi_c, envelope=envelope, is_valid=is_valid)
end

"""
    clear_rational_windows(psi_c, rationals) → Float64

Move the near-axis validity boundary outward until no rational surface lies inside the envelope's
transition band `[ψ_c, 2ψ_c]`. A rational sitting in the band gets its (near-singular) kinetic
increments multiplied by a rapidly varying, near-zero envelope, which the matrix splines cannot
represent — the resulting overshoot propagates NaNs into the stability solve. Where the orbit width
already reaches ⟨r⟩ the resonance is not trustworthy anyway, so the boundary is pushed past the
surface (suppressing it wholly) rather than cutting through it.
"""
function clear_rational_windows(psi_c::Float64, rationals::Vector{Float64})::Float64
    psi_c <= 0 && return psi_c
    for _ in 1:length(rationals)
        inside = filter(r -> psi_c - Equilibrium.RATIONAL_RES_RADIUS <= r <= 2 * psi_c, rationals)
        isempty(inside) && break
        psi_c = maximum(inside) + Equilibrium.RATIONAL_RES_RADIUS
    end
    return psi_c
end

"""
    axis_validity_boundary(kf_ctrl, species, kinetic_profiles, equil) → Float64

ψ_c for a run: the widest-orbit species' near-axis validity boundary, or `0.0` when suppression is
disabled or no kinetic profiles were loaded. Falls back to `kf_ctrl.zi`/`mi` when the resolved
species set is unavailable.
"""
function axis_validity_boundary(kf_ctrl::KineticForcesControl, species, kinetic_profiles, equil,
    rationals::Vector{Float64}=Float64[])::Float64
    (kf_ctrl.axis_validity_suppression && kinetic_profiles !== nothing) || return 0.0
    psi_c =
        species === nothing ?
        kinetic_axis_validity_psi(kinetic_profiles, equil; zi=kf_ctrl.zi, mi=kf_ctrl.mi, electron=kf_ctrl.electron) :
        kinetic_axis_validity_psi(species, equil)
    return clear_rational_windows(psi_c, rationals)
end

"""
    resonance_grid_nodes(ctrl, kf_ctrl, kinetic_profiles, species, equil, intr) → Vector{Float64}

ψ_N locations to pin into the two-pass equilibrium grid for a self-consistent kinetic run: the
located Ω_ℓ = 0 resonance surfaces for every toroidal mode in the run, outside the near-axis
validity region (nodes there are suppressed anyway). Empty for ideal runs — the ideal grid
criterion knows nothing about kinetic resonances, so this is the only thing that puts knots on
them.
"""
function resonance_grid_nodes(ctrl, kf_ctrl::KineticForcesControl, kinetic_profiles, species,
    equil, intr)::Vector{Float64}
    nodes = Float64[]
    (ctrl.kinetic_factor > 0 && ctrl.kinetic_source == "calculated" && kinetic_profiles !== nothing) || return nodes
    for n_res in intr.nlow:intr.nhigh
        n_res == 0 && continue
        append!(nodes, kinetic_resonance_psi_nodes(kinetic_profiles, equil;
            n=n_res, nl=kf_ctrl.nl, zi=kf_ctrl.zi, mi=kf_ctrl.mi,
            electron=kf_ctrl.electron, wdfac=kf_ctrl.wdfac))
    end
    psi_c = axis_validity_boundary(kf_ctrl, species, kinetic_profiles, equil,
        Float64[sng.psifac for sng in intr.sing])
    psi_c > 0 && filter!(p -> p > psi_c, nodes)
    return nodes
end

"""
    kinetic_validity_profiles(species, equil) → NamedTuple

Validity diagnostics for a resolved multi-species set, computed for the species that sets the
boundary (largest ψ_c). Reporting one species' profiles alongside another's ψ_c would make
`is_valid` contradict `envelope` in the same output group.
"""
function kinetic_validity_profiles(species::AbstractVector{<:Equilibrium.ResolvedNTVSpecies}, equil)
    isempty(species) && error("kinetic_validity_profiles: empty species set")
    widest = species[1]
    psi_c = -1.0
    for sp in species
        c = kinetic_axis_validity_psi(sp.profiles, equil; zi=sp.z, mi=sp.m, electron=sp.electron)
        c > psi_c && (psi_c = c; widest = sp)
    end
    return kinetic_validity_profiles(widest.profiles, equil; zi=widest.z, mi=widest.m, electron=widest.electron)
end
