# Shooting.jl
#
# Stable backward shooting solver for the GGJ inner-layer model. Direct
# Julia port of rmatch/deltar.f. Integrates the 4×4 origin-diagonalized
# resistive-layer ODE from `tmax` (large-x asymptotic regime) backward to
# a small `tmin` (origin Frobenius regime), then projects onto the local
# Frobenius basis at the origin and reads off the parity-projected
# matching data via `deltar_ratio`.
#
# The 4 origin Frobenius exponents are `al0 = (p1, 1/2, −p1, −1/2)` where
# `p1 = √(−D_I)` (Mercier-stable required). The two "small" (singular)
# modes are columns 3 and 4 (`x^{−p1}` and `x^{−1/2}`); the two "large"
# (smooth) modes are columns 1 and 2 (`x^{p1}` and `x^{1/2}`).
#
# Reference: A. H. Glasser, Phys. Plasmas **23**, 072505 (2016) and the
# Fortran rmatch/deltar.f.

using LinearAlgebra
using SpecialFunctions: gamma
using OrdinaryDiffEq

"""
    GGJShootingSystem

Precomputed origin- and infinity-side arrays for the deltar.f-style
backward shoot. Construct via [`_build_shooting_system`](@ref).
"""
struct GGJShootingSystem
    params::GGJParameters
    Q::ComplexF64
    p1::Float64
    pplus::Float64
    # origin side
    al0::SVector{4,Float64}                 # diagonal of origin exponent matrix
    al1::SMatrix{4,4,ComplexF64,16}         # first-order coupling (already in al0-diag basis)
    d0::SMatrix{4,4,ComplexF64,16}          # diagonalizing matrix at origin
    d0inv::SMatrix{4,4,ComplexF64,16}
    ups::Array{ComplexF64,3}                # (4, 4, nps) origin power-series coeffs
    nps::Int
    # infinity side
    d1::SMatrix{4,4,ComplexF64,16}          # diagonalizing matrix at infinity
    pexp::SVector{4,ComplexF64}             # log-exponents of asymptotic basis
    bl1::SVector{4,ComplexF64}              # ±λ exponents of t² in asymptotic basis
    tmax::Float64
    tmin::Float64
end

# -----------------------------------------------------------------------
# Origin arrays — port of deltar_origin (deltar.f lines 333–482).
# -----------------------------------------------------------------------

function _build_origin_arrays(p::GGJParameters, Q::ComplexF64; nps::Int=8, rtol::Float64=1e-6)
    p1v = p1(p)
    pplus = -0.5 + p1v

    e = ComplexF64(p.E); f = ComplexF64(p.F); h = ComplexF64(p.H)
    g = ComplexF64(p.G); k = ComplexF64(p.K); q = Q

    q3 = q^3
    kk1 = 1 + k * q3
    a1 = (g - k * e) * q * q / kk1
    a2 = sqrt(0.5 / (p1v * q))
    aplus  = h - 0.5 + p1v
    aminus = h - 0.5 - p1v

    # d0 — see deltar_origin lines 361–376.
    d0_m = zeros(ComplexF64, 4, 4)
    d0_m[1, 1] = q * a2
    d0_m[1, 2] = 1
    d0_m[1, 3] = -q * a2
    d0_m[1, 4] = 0
    d0_m[2, 2] = 1
    d0_m[3, 1] =  aplus * a2
    d0_m[3, 3] = -aminus * a2
    d0_m[4, 1] = -aplus * a2
    d0_m[4, 2] = -a1
    d0_m[4, 3] =  aminus * a2
    d0_m[4, 4] = 1

    # d0inv — same explicit antisymmetry pattern as deltar_origin lines 380–395.
    d0inv_m = zeros(ComplexF64, 4, 4)
    d0inv_m[1, 1] =  d0_m[3, 3]
    d0inv_m[2, 2] =  d0_m[4, 4]
    d0inv_m[1, 2] =  d0_m[4, 3]
    d0inv_m[2, 1] =  d0_m[3, 4]
    d0inv_m[3, 3] =  d0_m[1, 1]
    d0inv_m[4, 4] =  d0_m[2, 2]
    d0inv_m[3, 4] =  d0_m[2, 1]
    d0inv_m[4, 3] =  d0_m[1, 2]
    d0inv_m[1, 3] = -d0_m[1, 3]
    d0inv_m[2, 4] = -d0_m[2, 4]
    d0inv_m[1, 4] = -d0_m[2, 3]
    d0inv_m[2, 3] = -d0_m[1, 4]
    d0inv_m[3, 1] = -d0_m[3, 1]
    d0inv_m[4, 2] = -d0_m[4, 2]
    d0inv_m[3, 2] = -d0_m[4, 1]
    d0inv_m[4, 1] = -d0_m[3, 2]

    # al0 — origin exponents.
    al0 = SVector{4,Float64}(p1v, 0.5, -p1v, -0.5)

    # al1 in the original basis, then transform via d0inv * al1 * d0.
    al1_m = zeros(ComplexF64, 4, 4)
    al1_m[1, 3] = 1
    al1_m[2, 4] = -kk1
    al1_m[3, 1] = q
    al1_m[4, 2] = -q / kk1

    d0_s    = SMatrix{4,4,ComplexF64}(d0_m)
    d0inv_s = SMatrix{4,4,ComplexF64}(d0inv_m)
    al1_s   = d0inv_s * SMatrix{4,4,ComplexF64}(al1_m) * d0_s

    # ups[i, j, k] — origin power-series coefficients (deltar_origin lines 440–460).
    # ups(:,:,1) = I; higher orders from the al1 recurrence.
    ups = zeros(ComplexF64, 4, 4, nps)
    @inbounds for i in 1:4
        ups[i, i, 1] = 1
    end
    @inbounds for kk in 2:nps
        for j in 1:4, i in 1:4
            acc = ComplexF64(0)
            for l in 1:4
                acc += al1_s[i, l] * ups[l, j, kk - 1]
            end
            ups[i, j, kk] = acc / (al0[j] - al0[i] + 2 * (kk - 1))
        end
    end

    # tmin from the truncation-error bound on the origin series (line 470).
    unmax = 0.0
    @inbounds for j in 1:4, i in 1:4
        unmax = max(unmax, abs(ups[i, j, nps]))
    end
    fmin = 1.0
    tmin = fmin * 0.5 * (rtol / unmax)^(0.5 / nps)

    return (; al0, al1=al1_s, d0=d0_s, d0inv=d0inv_s, ups, nps, p1v, pplus, tmin)
end

# -----------------------------------------------------------------------
# Infinity arrays — port of deltar_infinity (deltar.f lines 490–666).
# With nxps = 1 the higher-order vps recurrence is dead code; vps[:,:,1] = I.
# -----------------------------------------------------------------------

function _build_infinity_arrays(p::GGJParameters, Q::ComplexF64,
                                d0inv::SMatrix{4,4,ComplexF64};
                                rtol::Float64=1e-6, fmax::Float64=1.0)
    e = ComplexF64(p.E); f = ComplexF64(p.F); h = ComplexF64(p.H)
    g = ComplexF64(p.G); k = ComplexF64(p.K); q = Q
    dr = mercier_dr(p)

    q3     = q^3
    kk1    = 1 + k * q3
    a1     = (g - k * e) * q * q / kk1
    lamda  = sqrt(q)
    lamdaq = lamda * q
    bb     = -0.25 * lamdaq * (1 + g + k * (dr - e))
    cc     =  0.25 * kk1 * (a1 * q * (1 - dr / q3) + dr - h * h)
    ddsq   = bb * bb - cc
    dd     = sqrt(ddsq)

    # m matrix (2×2) — deltar_infinity lines 521–524.
    m11 = -lamdaq + dr / lamdaq
    m12 =  h - dr / lamdaq
    m21 =  kk1 * (h + dr / lamdaq)
    m22 = -kk1 * (a1 / lamda + dr / lamdaq)

    pexp = SVector{4,ComplexF64}(bb - dd, bb + dd, -bb + dd, -bb - dd)

    # bmmat (2×2) and the first 2 columns of d1.
    d1_m = zeros(ComplexF64, 4, 4)
    bmmat = zeros(ComplexF64, 2, 2)
    @inbounds for j in 1:2
        bmmat[1, j] = m12
        bmmat[2, j] = 2 * pexp[j] - m11
        for i in 1:4
            d1_m[i, j] = d0inv[i, 1] * bmmat[1, j] + d0inv[i, 2] * bmmat[2, j] +
                         lamda * (-d0inv[i, 3] * bmmat[1, j] +
                                   d0inv[i, 4] * bmmat[2, j] / kk1)
        end
    end

    # bpmat (2×2) and columns 3,4 of d1.
    sigma = bmmat[1, 2] / bmmat[2, 2]
    tau   = bmmat[2, 1] / bmmat[1, 1]
    stfac = 0.5 / (1 - sigma * tau)
    bpmat = zeros(ComplexF64, 2, 2)
    bpmat[1, 1] = stfac / bmmat[1, 1]
    bpmat[2, 2] = stfac / bmmat[2, 2]
    bpmat[1, 2] = -tau   * bpmat[2, 2]
    bpmat[2, 1] = -sigma * bpmat[1, 1]
    @inbounds for j in 1:2
        for i in 1:4
            d1_m[i, j + 2] = (d0inv[i, 1] * bpmat[1, j] -
                              d0inv[i, 2] * bpmat[2, j] * kk1) / lamda +
                              d0inv[i, 3] * bpmat[1, j] +
                              d0inv[i, 4] * bpmat[2, j]
        end
    end

    d1 = SMatrix{4,4,ComplexF64}(d1_m)
    bl1 = SVector{4,ComplexF64}(-lamda, -lamda, lamda, lamda)

    # tmax (deltar_infinity lines 644–650, ntmax = 1 default).
    p1v = sqrt(-mercier_di(p))
    tmax_inner = sqrt((max(0.5, p1v)^2 - log(rtol)) / abs(lamda))
    tmax_outer = 6.0 / abs(q)
    tmax = fmax * min(tmax_inner, tmax_outer)

    return (; d1, pexp, bl1, tmax)
end

"""
    _build_shooting_system(p::GGJParameters, Q::ComplexF64; nps=8, rtol=1e-6, fmax=1.0)

Construct a [`GGJShootingSystem`](@ref) for the given parameters `p` and
complex frequency `Q`, precomputing the origin and infinity asymptotic arrays
used by the forward/backward shoots.
"""
function _build_shooting_system(p::GGJParameters, Q::ComplexF64;
                                nps::Int=8, rtol::Float64=1e-6, fmax::Float64=1.0)
    o = _build_origin_arrays(p, Q; nps=nps, rtol=rtol)
    inf = _build_infinity_arrays(p, Q, o.d0inv; rtol=rtol, fmax=fmax)
    return GGJShootingSystem(
        p, Q, o.p1v, o.pplus,
        o.al0, o.al1, o.d0, o.d0inv, o.ups, o.nps,
        inf.d1, inf.pexp, inf.bl1, inf.tmax, o.tmin,
    )
end

# -----------------------------------------------------------------------
# Origin Frobenius basis evaluator — port of deltar_upsfit (lines 233–267).
# Returns the 4×4 matrix `u(t)` whose columns are the 4 fundamental
# origin solutions evaluated at `t`. Solving `u * c0 = y` then projects a
# 4-component integrated state onto the origin basis.
# -----------------------------------------------------------------------

function _origin_basis(sys::GGJShootingSystem, t::Real)
    nps = sys.nps
    t2 = t * t
    u = zeros(ComplexF64, 4, 4)
    @inbounds for j in 1:4, i in 1:4
        # Horner in t² over k = nps..1
        acc = sys.ups[i, j, nps]
        for kk in (nps - 1):-1:1
            acc = acc * t2 + sys.ups[i, j, kk]
        end
        u[i, j] = acc
    end
    @inbounds for j in 1:4
        tj = t^sys.al0[j]
        for i in 1:4
            u[i, j] *= tj
        end
    end
    return SMatrix{4,4,ComplexF64}(u)
end

# -----------------------------------------------------------------------
# Initial condition at tmax — see deltar_run lines 121–126.
# With nxps = 1 in deltar.f, vps[:,:,1] = I, so the asymptotic-basis
# evaluator returns a diagonal v. The IC is then
#   y(tmax) = d1 * v(:, 1:2)
# which extracts the first two columns of d1 scaled by the algebraic-mode
# exponential factors. This selects the two "small" decaying modes of the
# large-t asymptotic basis.
# -----------------------------------------------------------------------

function _initial_condition(sys::GGJShootingSystem)
    t = sys.tmax
    fac1 = exp(0.5 * sys.bl1[1] * t * t + sys.pexp[1] * log(t))
    fac2 = exp(0.5 * sys.bl1[2] * t * t + sys.pexp[2] * log(t))
    y = zeros(ComplexF64, 4, 2)
    @inbounds for i in 1:4
        y[i, 1] = sys.d1[i, 1] * fac1
        y[i, 2] = sys.d1[i, 2] * fac2
    end
    return y
end

# -----------------------------------------------------------------------
# ODE right-hand side — port of deltar_der (lines 185–224).
#   dy/dt = al1 * y * t + diag(al0) * y / t
# -----------------------------------------------------------------------

function _ggj_der!(dy::AbstractMatrix{ComplexF64}, y::AbstractMatrix{ComplexF64},
                  sys::GGJShootingSystem, t::Real)
    mul!(dy, sys.al1, y)
    @inbounds for j in axes(y, 2), i in axes(y, 1)
        dy[i, j] = dy[i, j] * t + sys.al0[i] * y[i, j] / t
    end
    return nothing
end

# -----------------------------------------------------------------------
# Parity-projected matching ratio — port of deltar_ratio (lines 674–716).
# c0 is a 4×2 complex matrix whose rows are the four origin-mode
# coefficients of the two integrated solutions (modes 1: x^{p1},
# 2: x^{1/2}, 3: x^{−p1}, 4: x^{−1/2}).
# -----------------------------------------------------------------------

function _delta_from_c0(c0::AbstractMatrix{ComplexF64}, sys::GGJShootingSystem)
    pplus = sys.pplus
    dfac = -sys.d0[1, 3] / sys.d0[1, 1]
    phase = pplus * pi / 2
    gamfac = -pi / (sin(2 * phase) * gamma(1 + pplus)^2)

    delta1 = dfac * gamfac / tan(phase) *
             (c0[4, 1] * c0[3, 2] - c0[4, 2] * c0[3, 1]) /
             (c0[4, 1] * c0[1, 2] - c0[4, 2] * c0[1, 1])

    delta2 = dfac * gamfac * tan(phase) *
             (c0[2, 1] * c0[3, 2] - c0[2, 2] * c0[3, 1]) /
             (c0[2, 1] * c0[1, 2] - c0[2, 2] * c0[1, 1])

    return SVector{2,ComplexF64}(delta1, delta2)
end

# -----------------------------------------------------------------------
# Top-level shooting solver.
# -----------------------------------------------------------------------

"""
    solve_inner(::GGJModel{:shooting}, params::GGJParameters, γ::Number;
                reltol::Float64=1e-6, abstol::Float64=1e-6,
                rtol_origin::Float64=1e-6, nps::Int=8,
                fmax::Float64=1.0, solver=Rodas5P()) -> SVector{2,ComplexF64}

Solve the GGJ inner-layer matching problem by stable backward shooting in
the origin-diagonalized 4×4 basis. Direct port of the rmatch `deltar.f`
algorithm.

Returns the parity-projected matching data `(Δ₁, Δ₂)` (already rescaled
back to physical units via `rescale_delta`). Index ordering matches the
Fortran `deltar` output.

Tolerances `reltol`/`abstol` are the integrator tolerances; `rtol_origin`
controls the truncation error of the origin Frobenius series and the
choice of `tmin`.
"""
function solve_inner(::GGJModel{:shooting}, params::GGJParameters, γ::Number;
                     reltol::Float64=1e-6, abstol::Float64=1e-6,
                     rtol_origin::Float64=1e-6, nps::Int=8,
                     fmax::Float64=1.0, solver=Tsit5())
    Q = inner_Q(params, γ)
    sys = _build_shooting_system(params, Q; nps=nps, rtol=rtol_origin, fmax=fmax)

    y0 = _initial_condition(sys)

    prob = ODEProblem{true}(_ggj_der!, y0, (sys.tmax, sys.tmin), sys)
    sol = solve(prob, solver; reltol=reltol, abstol=abstol, save_everystep=false)

    y_end = sol.u[end]

    # Project onto origin Frobenius basis: solve u(tmin) * c0 = y_end.
    u = _origin_basis(sys, sys.tmin)
    c0 = Matrix(u) \ Matrix(y_end)

    Δ_raw = _delta_from_c0(c0, sys)
    return rescale_delta(Δ_raw, params)
end

solve_inner(::GGJModel{:shooting}, params::GGJParameters, γ::Real; kwargs...) =
    solve_inner(GGJModel{:shooting}(), params, ComplexF64(γ); kwargs...)
