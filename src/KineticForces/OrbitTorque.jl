"""
Orbit-averaged GAR torque (`forb`): the bounce frequency, toroidal precession and |δJ_ℓ|² of
[`calculate_gar`](@ref) taken along guiding-center orbits started on the surface instead of on the
surface itself. At each pitch node and sign of v∥ the orbit class is followed in energy: class changes are
bisected, and each trapped or passing segment carries its own √x samples, refined until a spline reproduces them.
The energy integral runs over each segment separately and never uses surface values.
"""

# Default top of the energy range in √x: x^{5/2}e^{−x} is ~1e-14 of its peak at x = 42.
const ORBIT_U_MAX = 6.5
# Lowest sampled √x; a segment's values are held below its first node.
const ORBIT_U_MIN = 0.02
# Uniform time samples per orbit period for δJ_ℓ (the integrand is periodic, so the trapezoid is spectral).
const ORBIT_NSAMPLE = 256
# Class-scan points in √x per pitch and sign, the class-edge bisection width as a fraction of umax, and the node cap per segment.
const ORBIT_NSCAN = 24
const ORBIT_EDGE_TOL = 1e-3
const ORBIT_MAXNODES = 64
# Fixed t nodes per segment on which the pitch interpolation runs.
const ORBIT_NT = 64
# τ_max factor of the retry for an orbit not closed within the default τ_max.
const ORBIT_TAU_RETRY = 10.0
# Orbit rtol for the table: changes ω_b, ⟨φ̇⟩ and |δJ_ℓ|² by < 4e-5 from 1e-10 at 2.4× less cost.
const ORBIT_RTOL = 1e-8

"""
    OrbitSegment

One closed-class interval [`ua`, `ub`] in √x of a (pitch, sign of v∥) column, with the orbit's quantities on its
√x nodes `u`, in the normalization of [`BounceData`](@ref).

## Fields

  - `passing`: orbit class (false: trapped). It sets ℓ_eff, the collisionality and the weight in the energy integral
  - `dir`: sign of dθ/dt for a passing orbit (0 for trapped); dir > 0 pairs with the surface path's `-wb` branch at −ℓ
  - `ua`, `ub`: the bisected class edges (`ua` = 0 for the segment that starts at zero energy)
  - `u`: √x sample nodes inside [`ua`, `ub`]
  - `wb`: ω_b/√x in rad/s; `wd`: the precession in the form of `BounceData.wd`
  - `dJdJ`: |δJ_ℓ|², size (length(u), 2lmax+1), indexed by the surface path's ℓ
"""
struct OrbitSegment
    passing::Bool
    dir::Int
    ua::Float64
    ub::Float64
    u::Vector{Float64}
    wb::Vector{Float64}
    wd::Vector{Float64}
    dJdJ::Matrix{Float64}
end

"""
    OrbitSurfaceTable

Guiding-center orbit segments on one flux surface.

## Fields

  - `lambda`: pitch nodes (the surface path's grid)
  - `segments`: per pitch node, the closed-class segments of both signs of v∥
  - `lmax`: bounce harmonics ℓ = -lmax:lmax
  - `nstatus`: class-scan orbit counts in `ORBIT_STATUS_CODES` order (0–4), then sample nodes dropped because the orbit
    did not close or changed class
"""
struct OrbitSurfaceTable
    lambda::Vector{Float64}
    segments::Vector{Vector{OrbitSegment}}
    lmax::Int
    nstatus::Vector{Int}
end

"""
    orbit_energy_nodes(a, b, n) → Vector{Float64}

`n` Chebyshev nodes in [a, b], clustered toward both ends where ω_b falls off at a class edge.
"""
orbit_energy_nodes(a::Float64, b::Float64, n::Int) = [a + (b - a) * (1 - cos(π * (j - 0.5) / n)) / 2 for j in 1:n]

"""
    orbit_surface_table(g, geo, intr, psi, theta0, q, welec, mass, chrg, T_s, n;
                        divxfac=1.0, lmax=1, nenergy=12, potential=false, nsample=ORBIT_NSAMPLE,
                        umax=ORBIT_U_MAX, energy_tol=1e-2) → OrbitSurfaceTable

Follow the orbits through (`psi`, `theta0`) in energy on each pitch node of `geo` and both signs of v∥. On each
trapped or passing segment, ω_b, the precession and |δJ_ℓ|² start on `nenergy` nodes (at least 7) and gain midpoints until a spline
through every other node reproduces the rest to `energy_tol` of each quantity's largest value.
δJ_ℓ = R₀∮dτ [w²∇·ξ⊥ + (w² − ½λB/B₀)δB/B] e^{−in(2πζ − ⟨φ̇⟩t)} e^{−iℓω_b t}, with w = v∥/v and ζ = (φ − ν)/2π the
straight-field-line toroidal angle, so the resonance is Ω = ℓω_b − n⟨φ̇⟩. With `potential`, `g` carries Φ and ω_E is
part of ⟨φ̇⟩, so it is taken out of `wd`. Lost and axis segments carry nothing.
"""
function orbit_surface_table(g::GuidingCenterField, geo::BounceGeometry, intr, psi::Float64, theta0::Float64,
    q::Float64, welec::Float64, mass::Float64, chrg::Float64, T_s::Float64, n::Int;
    divxfac::Float64=1.0, lmax::Int=1, nenergy::Int=12, potential::Bool=false, nsample::Int=ORBIT_NSAMPLE,
    umax::Float64=ORBIT_U_MAX, energy_tol::Float64=1e-2)
    nl, nh = geo.nlmda, 2 * lmax + 1
    segments = [OrbitSegment[] for _ in 1:nl]
    counts = [zeros(Int, 6) for _ in 1:nl]
    for il in 1:nl
        geo.valid[il] || continue
        lam, nstatus = geo.lambda[il], counts[il]
        buf = (dbob=zeros(ComplexF64, intr.mpert), divx=zeros(ComplexF64, intr.mpert), A=zeros(ComplexF64, nsample), u=zeros(6))
        for sv in (1, -1)
            gco(u; kw...) = guiding_center_orbit(g, psi, lam, u^2 * T_s, sv; mass, chrg, theta0, rtol=ORBIT_RTOL, kw...)
            function orbit(u; dense=true)
                o = gco(u; dense)
                o !== nothing && o.status == :unclosed && (o = gco(u; dense, tau_max=2e4 * ORBIT_TAU_RETRY))
                return o
            end
            code(o) = o === nothing ? Int8(-1) : ORBIT_STATUS_CODES[o.status]
            # Class probes need only the status; bisection points are not retried, since near an edge the period diverges.
            for (ua, ub, c) in _class_segments(u -> code(orbit(u; dense=false)), u -> code(gco(u; dense=false)), umax, nstatus)
                c in (1, 2) || continue
                seg = _sample_segment(orbit, c == 2, ua, ub, lam, q, welec, potential, n, lmax, divxfac, g, intr, buf, nenergy,
                    energy_tol, nstatus)
                seg === nothing || push!(segments[il], seg)
            end
        end
    end
    nstatus = reduce(+, counts)
    return OrbitSurfaceTable(geo.lambda[1:nl], segments, lmax, nstatus)
end

# Class segments [(ua, ub, code)] of one column in √x ∈ [0, umax]: a scan with `cls`, then bisection of each class change with `edge`.
function _class_segments(cls, edge, umax::Float64, nstatus)
    scan = range(ORBIT_U_MIN, umax, ORBIT_NSCAN)
    c = map(cls, scan)
    foreach(ci -> ci >= 0 && (nstatus[ci+1] += 1), c)
    edges, codes = [0.0], [c[1]]
    for j in 2:length(scan)
        c[j] == c[j-1] && continue
        lo, hi = scan[j-1], scan[j]
        while hi - lo > ORBIT_EDGE_TOL * umax
            mid = (lo + hi) / 2
            edge(mid) == c[j-1] ? (lo = mid) : (hi = mid)
        end
        push!(edges, (lo + hi) / 2)
        push!(codes, c[j])
    end
    push!(edges, umax)
    return [(edges[k], edges[k+1], codes[k]) for k in eachindex(codes)]
end

# Orbit quantities on one closed segment, refined until a spline through every other node reproduces the rest.
function _sample_segment(orbit, passing::Bool, ua, ub, lam, q, welec, potential, n, lmax, divxfac, g, intr, buf, nenergy,
    energy_tol, nstatus)
    nh = 2 * lmax + 1
    samples = Dict{Float64,Tuple{Float64,Float64,Vector{Float64},Int}}()
    function sample!(u)
        haskey(samples, u) && return nothing
        o = orbit(u)
        if o === nothing || o.status != (passing ? :passing : :trapped)
            nstatus[6] += 1
            return nothing
        end
        x = u^2
        dj = zeros(nh)
        _orbit_action!(dj, o, g, intr, lam, n, lmax, divxfac, passing && o.dir_theta > 0, buf)
        wd = (-o.wphi + (passing ? o.dir_theta * q * o.wb : 0.0) - (potential ? welec : 0.0)) / x
        samples[u] = (o.wb / u, wd, dj, passing ? Int(o.dir_theta) : 0)
        return nothing
    end
    foreach(sample!, orbit_energy_nodes(max(ua, ORBIT_U_MIN), ub, max(nenergy, 7)))
    while true
        us = sort!(collect(keys(samples)))
        length(us) < 7 && break
        V = reduce(vcat, [hcat(samples[u][1], samples[u][2], samples[u][3]') for u in us])
        dmax = maximum(abs, view(V, :, 3:size(V, 2)))
        scale = [j < 3 ? maximum(abs, view(V, :, j)) : dmax for j in axes(V, 2)]
        coarse = 1:2:length(us)
        spl = cubic_interp(us[coarse], Series(V[coarse, :]); bc=CubicFit())
        bad = Int[]
        for j in 2:2:length(us)-1
            err = abs.(spl(us[j]) .- V[j, :])
            any(err .> energy_tol .* max.(scale, eps())) && push!(bad, j)
        end
        (isempty(bad) || length(us) >= ORBIT_MAXNODES) && break
        for j in bad
            sample!((us[j-1] + us[j]) / 2)
            sample!((us[j] + us[j+1]) / 2)
        end
    end
    us = sort!(collect(keys(samples)))
    length(us) < 4 && return nothing
    dir = samples[us[1]][4]
    return OrbitSegment(passing, dir, ua, ub, us, [samples[u][1] for u in us], [samples[u][2] for u in us],
        reduce(vcat, [samples[u][3]' for u in us]))
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
                    chi1, ro, energy_atol, energy_rtol, pitch_atol, pitch_rtol, classes=:all) → ComplexF64

Torque density of harmonic `l` from an [`OrbitSurfaceTable`](@ref) with the normalization of [`calculate_gar`](@ref).
Each segment is resampled on fixed nodes in t = (√x − u_a)/(u_b − u_a); adjacent pitch nodes with the same segment
sequence form a group, in which u_a, u_b and the resampled values are splined in λ. The pitch quadrature breaks between
groups and, at each λ, sums the energy integrals of the interpolated segments, each keyed by its orbit class (ℓ_eff,
collisionality, weight ½ per trapped sign), so the resonance is never interpolated. `classes` (`:all`, `:passing`,
`:trapped`) keeps only one orbit class.
"""
function calculate_orbit(tab::OrbitSurfaceTable, n::Int, l::Int, q::Float64, epsr::Float64,
    wdian::Float64, wdiat::Float64, welec::Float64, nuk::Float64, bo::Float64, bmax::Float64,
    n_s::Float64, T_s::Float64; chi1::Float64, ro::Float64,
    energy_atol::Float64=1e-7, energy_rtol::Float64=1e-5, pitch_atol::Float64=1e-9, pitch_rtol::Float64=1e-6,
    classes::Symbol=:all)::ComplexF64
    abs(l) <= tab.lmax || error("calculate_orbit: harmonic $l outside the table's ±$(tab.lmax)")
    nl = length(tab.lambda)
    nl < 4 && return ComplexF64(0.0)
    ih = l + tab.lmax + 1
    nz = [abs(d) for segs in tab.segments for s in segs for d in view(s.dJdJ, :, ih) if !iszero(d)]
    isempty(nz) && return ComplexF64(0.0)
    dnorm = 1 / median(nz)
    segbuf = QuadGK.alloc_segbuf(Float64, ComplexF64, Float64)
    t = orbit_energy_nodes(0.0, 1.0, ORBIT_NT)
    keep(seg) = !(classes == :passing && !seg.passing) && !(classes == :trapped && seg.passing)
    # Per pitch node: segment signature, and per segment [u_a; u_b; ω_b/√x, wd, |δJ_ℓ|² on the t nodes].
    sig = [[(seg.passing, seg.dir) for seg in tab.segments[il]] for il in 1:nl]
    rows = map(1:nl) do il
        map(tab.segments[il]) do seg
            spl = cubic_interp(seg.u, Series(hcat(seg.wb, seg.wd, view(seg.dJdJ, :, ih) .* dnorm)); bc=CubicFit())
            us = clamp.(seg.ua .+ t .* (seg.ub - seg.ua), first(seg.u), last(seg.u))
            vcat(seg.ua, seg.ub, vec(reduce(vcat, [spl(u)' for u in us])))
        end
    end
    lo, hi = first(tab.lambda), last(tab.lambda)
    bobmax = bo / bmax
    starts = [il for il in 1:nl if il == 1 || sig[il] != sig[il-1]]
    groups = map(eachindex(starts)) do k
        i1, i2 = starts[k], k < length(starts) ? starts[k+1] - 1 : nl
        λa = k == 1 ? lo : _group_break(tab.lambda[i1-1], tab.lambda[i1], bobmax)
        λb = k == length(starts) ? hi : _group_break(tab.lambda[i2], tab.lambda[i2+1], bobmax)
        segs = [_lambda_interp(tab.lambda[i1:i2], [rows[il][j] for il in i1:i2]) for j in eachindex(sig[i1])]
        (; λa, λb, sig=sig[i1], segs)
    end
    function kernel(λ)
        gr = groups[clamp(searchsortedlast([g.λa for g in groups], λ), 1, length(groups))]
        total = zero(ComplexF64)
        for (j, (passing, dir)) in enumerate(gr.sig)
            r = gr.segs[j](λ)
            ua, ub = max(r[1], 0.0), r[2]
            ub > ua || continue
            seg = (; passing)
            keep(seg) || continue
            u = ua .+ t .* (ub - ua)
            spl = cubic_interp(u, Series(reshape(r[3:end], ORBIT_NT, 3)); bc=CubicFit())
            leff = passing ? l + n * q : Float64(l)
            p = EnergyParams(wdian, wdiat, welec, 0.0, 0.0, passing ? nuk : nuk / (2 * epsr), leff, n, "harmonic", "maxwellian",
                1.0, false)
            sgn = passing && dir > 0 ? -1.0 : 1.0
            xint = _integrate_energy_orbit(p, leff, sgn, spl, u, ua, ub, energy_atol, energy_rtol, segbuf)
            total += passing ? xint : 0.5 * xint
        end
        return total
    end
    pts = unique!(sort!(vcat(lo, [g.λa for g in groups[2:end]], hi, lo < bobmax < hi ? bobmax : Float64[])))
    I, Err = quadgk(kernel, pts...; atol=pitch_atol, rtol=pitch_rtol)
    check_inner_quadrature(I, Err, pitch_atol, pitch_rtol, "orbit pitch")
    tnorm = (-2 * n^2 / sqrt(π)) * (ro / bo) * n_s * T_s * (chi1 / twopi)
    return tnorm * I / dnorm
end

# Pitch-quadrature break between adjacent nodes of different groups: bo/bmax when it lies between them, else the midpoint.
_group_break(λ1, λ2, bobmax) = λ1 < bobmax < λ2 ? bobmax : (λ1 + λ2) / 2

# Interpolant in λ of per-node vectors: cubic on four or more nodes, linear on two or three, constant on one.
function _lambda_interp(λs::Vector{Float64}, vs::Vector{Vector{Float64}})
    length(λs) == 1 && return λ -> vs[1]
    if length(λs) >= 4
        spl = cubic_interp(λs, Series(permutedims(reduce(hcat, vs))); bc=CubicFit(), extrap=ExtendExtrap())
        return λ -> spl(λ)
    end
    return function (λ)
        k = clamp(searchsortedlast(λs, λ), 1, length(λs) - 1)
        w = (λ - λs[k]) / (λs[k+1] - λs[k])
        return (1 - w) .* vs[k] .+ w .* vs[k+1]
    end
end

"""
    _cell_roots(f, a, b, deg) → Vector{Float64}

Real roots in [a, b) of `f`, which is a polynomial of degree ≤ `deg` on [a, b]: its Chebyshev-node interpolant gives the
coefficients exactly, the companion-matrix eigenvalues give the roots, and each real one is polished by Newton steps on `f`.
"""
function _cell_roots(f, a::Float64, b::Float64, deg::Int)
    mid, half = (a + b) / 2, (b - a) / 2
    t = [cos(π * (j - 0.5) / (deg + 1)) for j in 1:deg+1]
    c = [ti^k for ti in t, k in 0:deg] \ [f(mid + half * ti) for ti in t]
    scale = maximum(abs, c)
    roots = Float64[]
    scale > 0 || return roots
    d = findlast(ck -> abs(ck) > 1e-13 * scale, c) - 1
    d >= 1 || return roots
    C = zeros(d, d)
    for i in 2:d
        C[i, i-1] = 1.0
    end
    C[:, d] .= -c[1:d] ./ c[d+1]
    for z in eigvals(C)
        abs(imag(z)) < 1e-7 && -1 - 1e-9 <= real(z) < 1 + 1e-9 || continue
        s = clamp(mid + half * real(z), a, b)
        h = 1e-7 * max(half, abs(s))
        for _ in 1:3
            fs = f(s)
            dfs = (f(s + h) - f(s - h)) / 2h
            dfs == 0 && break
            snew = clamp(s - fs / dfs, a, b)
            abs(f(snew)) < abs(fs) || break
            s = snew
        end
        a <= s < b && push!(roots, s)
    end
    sort!(roots)
    return unique!(r -> round(r / (1e-9 * half); digits=0), roots)
end

"""
    _integrate_energy_orbit(p, leff, sgn, spl, u, ua, ub, atol, rtol, segbuf) → ComplexF64

∫ N(x)e^{−x}D(√x)/(iΩ − ν) dx over √x ∈ [ua, ub], with Ω(u) = leff·sgn·Wb(u)·u + n(ω_E + Wd(u)·u²), where `spl`
splines [Wb, Wd, D] on the √x nodes `u` and is held at the end nodes outside them. On each spline cell Ω is a
polynomial of degree ≤ 5 in √x, so its real roots are found exactly (see [`_cell_roots`](@ref)) and removed by principal
value and residue as in [`_integrate_energy_resonant`](@ref), to which this reduces when Wb, Wd and D are constant. The
stationary points of Ω are breakpoints, so a near-tangent resonance that does not cross zero is still resolved.
"""
function _integrate_energy_orbit(p::EnergyParams, leff::Float64, sgn::Float64, spl, u::Vector{Float64},
    ua::Float64, ub::Float64, atol::Float64, rtol::Float64, segbuf)::ComplexF64
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
    xa, xb = ua^2, ub^2

    x_poles = ComplexF64[]
    residues = ComplexF64[]
    pole_contribution = zero(ComplexF64)
    cells = unique!(vcat(ua, filter(s -> ua < s < ub, u), ub))
    extrema_u = Float64[]
    for j in 1:length(cells)-1
        a, b = cells[j], cells[j+1]
        append!(extrema_u, _cell_roots(dΩ, a, b, 4))
        for ur in _cell_roots(Ω, a, b, 5)
            ur > 0 || continue
            xr = ur^2
            op = dΩ(ur) / (2ur)
            abs(op) >= SINGULAR_EPS || continue
            offset = _energy_collision_frequency(xr, p) / op
            isfinite(offset) || continue
            x_pole = complex(xr, -offset)
            R = h(xr) / (im * op)
            pole_contribution += R * (log(xb - x_pole) - log(xa - x_pole))
            push!(x_poles, x_pole)
            push!(residues, R)
        end
    end

    function integrand_u(s)
        x = s * s
        val = h(x) / complex(-_energy_collision_frequency(x, p), Ω(s))
        @inbounds for k in eachindex(residues)
            val -= residues[k] / (x - x_poles[k])
        end
        return isfinite(val) ? 2s * val : zero(ComplexF64)
    end

    pts = vcat(ua, ub, extrema_u)
    for xp in x_poles
        xr = real(xp)
        push!(pts, sqrt(xr))
        w = abs(imag(xp))
        if 0 < w < NARROW_POLE_FRACTION * xr && xr < NARROW_POLE_XMAX
            for d in (-10w, -2w, 2w, 10w)
                xw = xr + d
                xa < xw < xb && push!(pts, sqrt(xw))
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
