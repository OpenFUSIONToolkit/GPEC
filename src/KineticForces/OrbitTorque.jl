"""
Orbit-averaged GAR torque (`forb`): the bounce frequency, toroidal precession and |δJ_ℓ|² of
[`calculate_gar`](@ref) taken along guiding-center orbits started on the surface instead of on the
surface itself. Off the zero-orbit-width limit these depend on energy, so each pitch carries them on
√x nodes and the energy integral splines them.
"""

# Largest tabulated √x; Ω and |δJ|² are held at the end nodes outside [u₁, u_K].
const ORBIT_U_MAX = 6.0
# Uniform time samples per orbit period for δJ_ℓ (the integrand is periodic, so the trapezoid is spectral).
const ORBIT_NSAMPLE = 256
# Half-width of the trapped–passing layer taken at zero width, in units of the widest class change seen at that energy.
const ORBIT_LAYER_FACTOR = 10.0

"""
    OrbitSurfaceTable

Guiding-center orbit quantities on one flux surface, per pitch λ, √x node and direction slot, in the
normalization of [`BounceData`](@ref) so that each reduces to the surface value at zero orbit width.
Slot 1 is σ = +1 (trapped) or the orbit paired with `+wb` (passing, θ decreasing); slot 2 is σ = −1
(trapped) or the orbit paired with `-wb` (passing, θ increasing).

## Fields

  - `lambda`, `trapped`: pitch grid and its surface (zero-width) class
  - `u`: √x nodes
  - `wb`: ω_b/√x in rad/s, size (nλ, nu, 2)
  - `wd`: the precession in the form of `BounceData.wd`, so Ω = ℓ_eff·(±wb)·√x + n·(ω_E + wd·x), size (nλ, nu, 2)
  - `dJdJ`: |δJ_ℓ|² as `BounceData.dJdJ`, size (nλ, nu, 2, 2lmax+1), indexed by the surface path's ℓ
  - `valid`: the node carries its orbit's values. Lost and axis orbits carry zero |δJ|². Unclosed orbits,
    forbidden starts and the trapped–passing layer carry the surface ω_b, ω_D and |δJ|²: the layer is the
    pitches within `ORBIT_LAYER_FACTOR` times the widest class change at that energy of the surface boundary,
    where the orbit-width expansion fails (the error there grows as width over distance to the separatrix)
  - `lmax`: bounce harmonics ℓ = -lmax:lmax
  - `nstatus`: orbit counts by class (`ORBIT_STATUS_CODES` order 0–4) and, last, closed orbits of the other class
"""
struct OrbitSurfaceTable
    lambda::Vector{Float64}
    trapped::BitVector
    u::Vector{Float64}
    wb::Array{Float64,3}
    wd::Array{Float64,3}
    dJdJ::Array{Float64,4}
    valid::BitArray{3}
    lmax::Int
    nstatus::Vector{Int}
end

"""
    orbit_energy_nodes(nenergy) → Vector{Float64}

√x nodes of the orbit table, clustered toward low energy: u_k = U·((k − ½)/K)^{3/2}, U = `ORBIT_U_MAX`.
"""
orbit_energy_nodes(nenergy::Int) = [ORBIT_U_MAX * ((k - 0.5) / nenergy)^1.5 for k in 1:nenergy]

"""
    orbit_surface_table(g, geo, intr, psi, theta0, q, welec, mass, chrg, T_s, n;
                        divxfac=1.0, lmax=1, nenergy=12, potential=false, nsample=ORBIT_NSAMPLE,
                        layer_factor=ORBIT_LAYER_FACTOR) → OrbitSurfaceTable

Integrate the orbits through (`psi`, `theta0`) on the pitch grid of the surface geometry `geo`, at the
√x nodes of [`orbit_energy_nodes`](@ref) and both signs of v∥, and tabulate ω_b, the precession and
|δJ_ℓ|². δJ_ℓ = R₀∮dτ [w²∇·ξ⊥ + (w² − ½λB/B₀)δB/B] e^{−in(2πζ − ⟨φ̇⟩t)} e^{−iℓω_b t}, with
w = v∥/v and ζ = (φ − ν)/2π the straight-field-line toroidal angle, so the resonance is Ω = ℓω_b − n⟨φ̇⟩.
A trapped label σ = ±1 is a different banana, each carried at half weight; a passing orbit moving
in +θ pairs with the surface path's `-wb` branch at harmonic −ℓ. With `potential`, `g` carries Φ and
ω_E is part of ⟨φ̇⟩, so it is taken out of `wd`.
"""
function orbit_surface_table(g::GuidingCenterField, geo::BounceGeometry, intr, psi::Float64, theta0::Float64,
    q::Float64, welec::Float64, mass::Float64, chrg::Float64, T_s::Float64, n::Int;
    divxfac::Float64=1.0, lmax::Int=1, nenergy::Int=12, potential::Bool=false, nsample::Int=ORBIT_NSAMPLE,
    layer_factor::Float64=ORBIT_LAYER_FACTOR)
    us = orbit_energy_nodes(nenergy)
    nl, nu, nh = geo.nlmda, length(us), 2 * lmax + 1
    trapped = BitVector(geo.sigma .== 0)
    wb = zeros(nl, nu, 2)
    wd = zeros(nl, nu, 2)
    dJdJ = zeros(nl, nu, 2, nh)
    valid = falses(nl, nu, 2)
    nstatus = zeros(Int, 6)
    buf = (dbob=zeros(ComplexF64, intr.mpert), divx=zeros(ComplexF64, intr.mpert), A=zeros(ComplexF64, nsample), u=zeros(6))
    surface_dJdJ = reduce(hcat, [bounce_harmonic(geo, l, n, q, intr.ro, intr.mpert).dJdJ for l in -lmax:lmax])
    mismatch = falses(nl, nu)
    for il in 1:nl, (k, u) in enumerate(us)
        wb[il, k, :] .= geo.wb[il]
        wd[il, k, :] .= geo.wd[il]
        geo.valid[il] || continue
        lam = geo.lambda[il]
        for (islot, sv) in enumerate((1, -1))
            o = guiding_center_orbit(g, psi, lam, u^2 * T_s, sv; mass, chrg, theta0)
            o === nothing || (nstatus[ORBIT_STATUS_CODES[o.status]+1] += 1)
            o !== nothing && o.status in (:lost, :axis) && continue
            closed = o !== nothing && o.status in (:trapped, :passing)
            if closed && (o.status == :trapped) != trapped[il]
                nstatus[6] += 1
                mismatch[il, k] = true
            end
            if !closed || (o.status == :trapped) != trapped[il]
                dJdJ[il, k, islot, :] .= view(surface_dJdJ, il, :)
                continue
            end
            slot = trapped[il] ? islot : (o.dir_theta < 0 ? 1 : 2)
            x = u^2
            wb[il, k, slot] = o.wb / u
            wd[il, k, slot] = (-o.wphi + (trapped[il] ? 0.0 : o.dir_theta * q * o.wb) - (potential ? welec : 0.0)) / x
            _orbit_action!(view(dJdJ, il, k, slot, :), o, g, intr, lam, n, lmax, divxfac, !trapped[il] && slot == 2, buf)
            valid[il, k, slot] = true
        end
    end
    _fill_separatrix_layer!(wb, wd, dJdJ, valid, mismatch, geo, trapped, surface_dJdJ, layer_factor)
    return OrbitSurfaceTable(geo.lambda[1:nl], trapped, us, wb, wd, dJdJ, valid, lmax, nstatus)
end

# Surface values on the trapped–passing layer (see `OrbitSurfaceTable`), per energy node.
function _fill_separatrix_layer!(wb, wd, dJdJ, valid, mismatch, geo, trapped, surface_dJdJ, layer_factor)
    (any(trapped) && !all(trapped)) || return nothing
    λtp = (maximum(geo.lambda[.!trapped]) + minimum(geo.lambda[trapped])) / 2
    dist = abs.(geo.lambda .- λtp)
    for k in axes(mismatch, 2)
        any(view(mismatch, :, k)) || continue
        width = layer_factor * maximum(dist[view(mismatch, :, k)])
        for il in eachindex(dist)
            dist[il] < width || continue
            wb[il, k, :] .= geo.wb[il]
            wd[il, k, :] .= geo.wd[il]
            for s in 1:2
                dJdJ[il, k, s, :] .= view(surface_dJdJ, il, :)
            end
            valid[il, k, :] .= false
        end
    end
    return nothing
end

# |δJ_ℓ|² of one orbit for ℓ = -lmax:lmax (the surface path's labels; `flip` maps them to −ℓ along the orbit).
function _orbit_action!(out, o::GuidingCenterOrbit, g::GuidingCenterField, intr, lam::Float64, n::Int, lmax::Int,
    divxfac::Float64, flip::Bool, buf)
    sol = o.sol
    τn = sol.t[end]
    N = length(buf.A)
    dτ = τn / N
    u0, uend = sol.u[1], sol.u[end]
    ζ0 = (u0[3] - g.nu((u0[1], u0[2]))) / twopi
    Δζ = (uend[3] - g.nu((uend[1], uend[2]))) / twopi - ζ0
    m1 = first(intr.mfac)
    for k in 1:N
        τ = (k - 1) * dτ
        u = sol(buf.u, τ)
        ψ, θ, w = u[1], u[2], u[4]
        intr.dbob_m(buf.dbob, ψ)
        intr.divx_m(buf.divx, ψ)
        e1 = cis(twopi * θ)
        em = cis(twopi * m1 * θ)
        dbob = divx = zero(ComplexF64)
        @inbounds for j in eachindex(buf.dbob)
            dbob += buf.dbob[j] * em
            divx += buf.divx[j] * em
            em *= e1
        end
        A = w^2 * divxfac * divx + (w^2 - 0.5 * lam * g.B((ψ, θ)) / g.bo) * dbob
        ζ = (u[3] - g.nu((ψ, θ))) / twopi - ζ0
        buf.A[k] = A * cis(-n * twopi * (ζ - Δζ * τ / τn))
    end
    wbbar = twopi / τn
    for (i, l) in enumerate(-lmax:lmax)
        lo = flip ? -l : l
        I = zero(ComplexF64)
        @inbounds for k in 1:N
            I += buf.A[k] * cis(-lo * twopi * (k - 1) / N)
        end
        out[i] = wbbar * abs2(I * dτ * g.ro) / 2 / g.ro^2
    end
    return nothing
end

"""
    calculate_orbit(tab, n, l, q, epsr, wdian, wdiat, welec, nuk, bo, bmax, n_s, T_s;
                    chi1, ro, energy_atol, energy_rtol, pitch_atol, pitch_rtol) → ComplexF64

Torque density of harmonic `l` from an [`OrbitSurfaceTable`](@ref): the pitch and energy integrals of
[`calculate_gar`](@ref) with its normalization, the frequencies and |δJ_ℓ|² splined in √x per pitch.
"""
function calculate_orbit(tab::OrbitSurfaceTable, n::Int, l::Int, q::Float64, epsr::Float64,
    wdian::Float64, wdiat::Float64, welec::Float64, nuk::Float64, bo::Float64, bmax::Float64,
    n_s::Float64, T_s::Float64; chi1::Float64, ro::Float64,
    energy_atol::Float64=1e-7, energy_rtol::Float64=1e-5, pitch_atol::Float64=1e-9, pitch_rtol::Float64=1e-6)::ComplexF64
    abs(l) <= tab.lmax || error("calculate_orbit: harmonic $l outside the table's ±$(tab.lmax)")
    nl, nu = length(tab.lambda), length(tab.u)
    nl == 0 && return ComplexF64(0.0)
    ih = l + tab.lmax + 1
    D = tab.dJdJ[:, :, :, ih]
    med = median(abs.(D[tab.valid]))
    dnorm = (isfinite(med) && med > 0) ? 1 / med : 1.0
    data = hcat(reshape(tab.wb, nl, :), reshape(tab.wd, nl, :), reshape(D .* dnorm, nl, :))
    fb = cubic_interp(tab.lambda, Series(data); bc=CubicFit())
    vals = zeros(3 * nu * 2)
    segbuf = QuadGK.alloc_segbuf(Float64, ComplexF64, Float64)
    bobmax = bo / bmax
    col(v, a, s) = view(v, (a-1)*2*nu+(s-1)*nu+1:(a-1)*2*nu+s*nu)
    function kernel(λ)
        fb(vals, λ)
        circ = λ <= bobmax
        leff = circ ? l + n * q : Float64(l)
        p = EnergyParams(wdian, wdiat, welec, 0.0, 0.0, circ ? nuk : nuk / (2 * epsr), leff, n, "harmonic", "maxwellian", 1.0, false)
        total = zero(ComplexF64)
        for s in 1:2
            Dk = col(vals, 3, s)
            all(iszero, Dk) && continue
            spl = cubic_interp(tab.u, Series(hcat(col(vals, 1, s), col(vals, 2, s), Dk)); bc=CubicFit())
            sgn = circ && s == 2 ? -1.0 : 1.0
            xint = _integrate_energy_orbit(p, leff, sgn, spl, tab.u, energy_atol, energy_rtol, segbuf)
            total += circ ? xint : 0.5 * xint
        end
        return total
    end
    lo, hi = first(tab.lambda), last(tab.lambda)
    split = clamp(bobmax, lo, hi)
    I, E = lo < split < hi ? quadgk(kernel, lo, split, hi; atol=pitch_atol, rtol=pitch_rtol) :
           quadgk(kernel, lo, hi; atol=pitch_atol, rtol=pitch_rtol)
    check_inner_quadrature(I, E, pitch_atol, pitch_rtol, "orbit pitch")
    tnorm = (-2 * n^2 / sqrt(π)) * (ro / bo) * n_s * T_s * (chi1 / twopi)
    return tnorm * I / dnorm
end

"""
    _integrate_energy_orbit(p, leff, sgn, spl, u, atol, rtol, segbuf) → ComplexF64

∫₀^{X_ENERGY_MAX} N(x)e^{−x}D(√x)/(iΩ − ν) dx with Ω(u) = leff·sgn·Wb(u)·u + n(ω_E + Wd(u)·u²), where `spl`
splines [Wb, Wd, D] on the √x nodes `u` and is held at the end nodes outside them. The poles are the sign changes of Ω on
a fine √x grid, removed by principal value and residue as in [`_integrate_energy_resonant`](@ref), to
which this reduces when Wb, Wd and D are constant.
"""
function _integrate_energy_orbit(p::EnergyParams, leff::Float64, sgn::Float64, spl, u::Vector{Float64},
    atol::Float64, rtol::Float64, segbuf)::ComplexF64
    n = p.n
    u1, uK = first(u), last(u)
    y, dy = zeros(3), zeros(3)
    cl(s) = clamp(s, u1, uK)
    function Ω(s)
        spl(y, cl(s))
        return leff * sgn * y[1] * s + n * (p.we + y[2] * s^2)
    end
    function dΩ(s)
        c = cl(s)
        spl(y, c)
        u1 < s < uK ? spl(dy, c; deriv=DerivOp(1)) : fill!(dy, 0.0)
        return leff * sgn * (y[1] + s * dy[1]) + n * (2s * y[2] + s^2 * dy[2])
    end
    function h(x)
        spl(y, cl(sqrt(x)))
        return _energy_numerator(x, p) * exp(-x) * y[3]
    end
    u_max = sqrt(X_ENERGY_MAX)

    x_poles = ComplexF64[]
    residues = ComplexF64[]
    pole_contribution = zero(ComplexF64)
    grid = range(0.0, u_max, 401)
    Ωa = Ω(grid[1])
    for j in 2:length(grid)
        Ωb = Ω(grid[j])
        if Ωa * Ωb < 0
            ur = find_zero(Ω, (grid[j-1], grid[j]), Bisection())
            xr = ur^2
            op = dΩ(ur) / (2ur)
            if abs(op) >= SINGULAR_EPS
                offset = _energy_collision_frequency(xr, p) / op
                if isfinite(offset)
                    x_pole = complex(xr, -offset)
                    R = h(xr) / (im * op)
                    pole_contribution += R * (log(X_ENERGY_MAX - x_pole) - log(-x_pole))
                    push!(x_poles, x_pole)
                    push!(residues, R)
                end
            end
        end
        Ωa = Ωb
    end

    function integrand_u(s)
        x = s * s
        val = h(x) / complex(-_energy_collision_frequency(x, p), Ω(s))
        @inbounds for k in eachindex(residues)
            val -= residues[k] / (x - x_poles[k])
        end
        return isfinite(val) ? 2s * val : zero(ComplexF64)
    end

    pts = [0.0, u_max]
    for xp in x_poles
        xr = real(xp)
        push!(pts, sqrt(xr))
        w = abs(imag(xp))
        if 0 < w < NARROW_POLE_FRACTION * xr && xr < NARROW_POLE_XMAX
            for d in (-10w, -2w, 2w, 10w)
                xb = xr + d
                0.0 < xb < X_ENERGY_MAX && push!(pts, sqrt(xb))
            end
        end
    end
    sort!(pts)
    unique!(pts)
    val, err = length(pts) <= MAX_ENERGY_BREAKS ? _quadgk_breaks(integrand_u, pts, atol, rtol, segbuf) :
               quadgk(integrand_u, pts...; atol, rtol, segbuf)
    check_inner_quadrature(val, err, atol, rtol, "orbit energy")
    return val + pole_contribution
end
