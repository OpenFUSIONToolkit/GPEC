# RayAsymptotics.jl
#
# Complex-x extensions of the inps Wasow asymptotic kernel
# (InnerAsymptotics.jl) for the rotated-ray solver (Ray.jl).
#
# The inps construction (GW2020 Eqs. 3–52) depends only on Q — the contour
# enters only through the EVALUATION point. These methods therefore reuse the
# existing, validated `InnerAsymptoticsCache` and coefficient packing verbatim
# and add evaluation at complex x = e^{iθ}s (principal branch of x^r, valid
# for |arg x| < π), the plain-variable conversion used by the ray boundary
# data, the GW2020 Eq. (54) residual along the ray, and the series-radius
# selector `pick_smax` (the ray analog of `pick_xmax`).
#
# "Far-field boundary and the inward
# march"; the real-x methods of InnerAsymptotics.jl are untouched.

# Complex-argument Horner evaluator: same contract as _horner(x::Real, ...)
# in InnerAsymptotics.jl, with x^rvec on the principal branch.
function _horner(x::Complex, c::AbstractMatrix{ComplexF64};
    rvec::Union{Nothing,AbstractVector{<:Real}}=nothing,
    derivative::Bool=false)
    nrows, ncols = size(c)
    n = ncols - 1

    y = Vector{ComplexF64}(undef, nrows)
    @inbounds for i in 1:nrows
        y[i] = c[i, ncols]
    end
    @inbounds for k in (n-1):-1:0
        for i in 1:nrows
            y[i] = y[i] * x + c[i, k+1]
        end
    end

    if rvec !== nothing
        @inbounds for i in 1:nrows
            y[i] *= ComplexF64(x)^rvec[i]
        end
    end

    if !derivative
        return y, nothing
    end

    dy = similar(y)
    if rvec !== nothing
        @inbounds for i in 1:nrows
            dy[i] = c[i, ncols] * (rvec[i] + n)
        end
        @inbounds for k in (n-1):-1:0
            for i in 1:nrows
                dy[i] = dy[i] * x + c[i, k+1] * (rvec[i] + k)
            end
        end
        @inbounds for i in 1:nrows
            dy[i] = dy[i] * ComplexF64(x)^rvec[i] / x
        end
    else
        @inbounds for i in 1:nrows
            dy[i] = c[i, ncols] * n
        end
        @inbounds for k in (n-1):-1:0
            for i in 1:nrows
                dy[i] = dy[i] * x + c[i, k+1] * k
            end
        end
        @inbounds for i in 1:nrows
            dy[i] = dy[i] / x
        end
    end
    return y, dy
end

"""
    evaluate_asymptotics(cache, x::Complex; derivative=true, apply_T=true) -> (U, dU)

Evaluate the inps asymptotic basis at complex `x` (GW2020 Eq. 53; principal
branch, |arg x| < π). Returns the 6×2 matrix `U` whose columns are the two
power-like asymptotic solutions in the GW2020 Eq. (2) state convention
`u = (xΨ, Ξ, Υ, (xΨ)'/x, Ξ'/x, Υ'/x)`, and their x-derivatives `dU` if
requested. `apply_T=false` returns the pre-Jordan-basis representation used
by the residual measure.
"""
function evaluate_asymptotics(cache::InnerAsymptoticsCache, x::Complex;
    derivative::Bool=true, apply_T::Bool=true)
    xfac = 1 / (x * x)
    R = cache.R
    rvec = SVector(-R[1] / 2, -R[1] / 2, -R[2] / 2, -R[2] / 2)

    cc = _pack_y_coefs(cache)
    dd = _pack_qp_coefs(cache)

    yy, dyy = _horner(xfac, cc; rvec=rvec, derivative=derivative)
    zz, dzz = _horner(xfac, dd; derivative=derivative)

    y = SMatrix{2,2,ComplexF64}(yy[1], yy[2], yy[3], yy[4])
    q = SMatrix{2,2,ComplexF64}(zz[1], zz[2], zz[3], zz[4])
    p21 = SMatrix{4,2,ComplexF64}(zz[5], zz[6], zz[7], zz[8],
        zz[9], zz[10], zz[11], zz[12])

    pp_m = zeros(ComplexF64, 6, 2)
    pp_m[1, 1] = 1
    pp_m[2, 2] = 1
    @inbounds for i in 1:4, j in 1:2
        pp_m[i+2, j] = p21[i, j]
    end
    pp = SMatrix{6,2,ComplexF64}(pp_m)

    smat = @SMatrix ComplexF64[1 0; 0 xfac]

    qsy = q * smat * y
    U = pp * qsy

    if derivative
        chain = -2 * xfac / x
        dy_v = chain .* dyy
        dz_v = chain .* dzz

        dy = SMatrix{2,2,ComplexF64}(dy_v[1], dy_v[2], dy_v[3], dy_v[4])
        dq = SMatrix{2,2,ComplexF64}(dz_v[1], dz_v[2], dz_v[3], dz_v[4])
        dp21 = SMatrix{4,2,ComplexF64}(dz_v[5], dz_v[6], dz_v[7], dz_v[8],
            dz_v[9], dz_v[10], dz_v[11], dz_v[12])

        dpp_m = zeros(ComplexF64, 6, 2)
        @inbounds for i in 1:4, j in 1:2
            dpp_m[i+2, j] = dp21[i, j]
        end
        dpp = SMatrix{6,2,ComplexF64}(dpp_m)

        dsmat = @SMatrix ComplexF64[0 0; 0 (-2*xfac/x)]
        dqsy = q * smat * dy + q * dsmat * y + dq * smat * y
        dU = pp * dqsy + dpp * qsy

        if apply_T
            U = cache.T * U
            dU = cache.T * dU
        end
        return U, dU
    else
        if apply_T
            U = cache.T * U
        end
        return U, nothing
    end
end

"""
    physical_ua_dua(cache, x::Number) -> (ua, dua)

Convert the inps 6×2 basis at (possibly complex) `x` to physical `(Ψ, Ξ, Υ)`
values and x-derivatives (each 3×2; column 1 = large power solution,
column 2 = small). Same map as the deltac `inpso_get_ua/dua` convention and
the Galerkin backend's `_physical_ua_dua`, generalized to complex x.
"""
function physical_ua_dua(cache::InnerAsymptoticsCache, x::Number)
    xc = ComplexF64(x)
    U, dU = evaluate_asymptotics(cache, xc; derivative=true, apply_T=true)
    ua = zeros(ComplexF64, 3, 2)
    dua = zeros(ComplexF64, 3, 2)
    @inbounds for j in 1:2
        ua[1, j] = U[1, j] / xc
        ua[2, j] = U[2, j]
        ua[3, j] = U[3, j]
        dua[1, j] = U[4, j] - U[1, j] / (xc * xc)
        dua[2, j] = U[5, j] * xc
        dua[3, j] = U[6, j] * xc
    end
    return ua, dua
end

"""
    asymptotic_residual(cache, x::Complex) -> SVector{2,Float64}

GW2020 Eq. (54) convergence measure of the two power-like series columns at
complex `x`: ‖dU − x·J(x)·U‖∞ / max(‖dU‖∞, ‖x·J(x)·U‖∞), per column.
"""
function asymptotic_residual(cache::InnerAsymptoticsCache, x::Complex)
    U, dU = evaluate_asymptotics(cache, x; derivative=true, apply_T=false)

    xfac = 1 / (x * x)
    M = cache.J[1]
    if cache.kmax > 0
        M = M + xfac * cache.J[2]
    end
    if cache.kmax > 1
        M = M + xfac * xfac * cache.J[3]
    end

    matvec1 = dU
    matvec2 = -x * (M * U)
    matvec0 = matvec1 + matvec2

    delta = MVector{2,Float64}(0.0, 0.0)
    @inbounds for j in 1:2
        n0 = 0.0
        n1 = 0.0
        n2 = 0.0
        for i in 1:6
            n0 = max(n0, abs(matvec0[i, j]))
            n1 = max(n1, abs(matvec1[i, j]))
            n2 = max(n2, abs(matvec2[i, j]))
        end
        denom = max(n1, n2)
        delta[j] = denom == 0 ? 0.0 : n0 / denom
    end
    return SVector{2,Float64}(delta)
end

"""
    pick_smax(params, Q; θ=angle(Q)/4, eps=1e-9, kmax=12, cache=nothing,
              slogmin=-1.0, slogmax=6.5, dslog=0.01) -> (S, cache, achieved)

Ray analog of `pick_xmax`: sweep the ray parameter `s` log-uniformly and
return the smallest `s` at which the series residual along `x = e^{iθ}s`
drops below `eps`. If the target is never reached, returns the residual-
minimizing `s` with `achieved = false` (caller should warn). Also returns
the asymptotics cache for reuse.
"""
function pick_smax(params::GGJParameters, Q::ComplexF64;
    θ::Float64=angle(Q) / 4, eps::Float64=1e-9, kmax::Int=12,
    cache::Union{Nothing,InnerAsymptoticsCache}=nothing,
    slogmin::Float64=-1.0, slogmax::Float64=6.5, dslog::Float64=0.01)
    c = cache === nothing ? build_asymptotics(params, Q; kmax=kmax) : cache
    ph = cis(θ)
    best_s = 10.0^slogmin
    best_r = Inf
    slog = slogmin
    while slog <= slogmax
        s = 10.0^slog
        r = maximum(asymptotic_residual(c, ph * s))
        if r < eps
            return s, c, true
        end
        if r < best_r
            best_r = r
            best_s = s
        end
        slog += dslog
    end
    return best_s, c, false
end

pick_smax(params::GGJParameters, Q::Number; kwargs...) =
    pick_smax(params, ComplexF64(Q); kwargs...)
