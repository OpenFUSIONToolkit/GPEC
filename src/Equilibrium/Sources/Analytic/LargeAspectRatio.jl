"""
    LargeAspectRatioConfig(...)

A mutable struct holding parameters for the Large Aspect Ratio (LAR) plasma equilibrium model.

## Fields:

  - `r0`: The major radius of the plasma [m].
  - `a`: The minor radius of the plasma [m].
  - `beta0`: The beta value on axis (normalized pressure).
  - `q0`: The safety factor on axis.
  - `p_pres`: The exponent for the pressure profile, defined as `p00 * (1 - (r / a)^2)^p_pres`.
  - `p_sig`: The exponent that determines the shape of the current-related function profile.
  - `sigma_type`: The type of sigma profile, can be "default", "wesson" or "hollow". If "wesson", the sigma profile is defined as `sigma0 * (1 - (r / a)^2)^p_sig`. If "hollow", it is `sigma0 * (1 + hollow * (r / a)^2) * (1 - (r / a)^2)^p_sig`, an off-axis current peak that makes q fall from `q0` to a minimum before rising (reversed shear) when `hollow` is large enough.
  - `hollow`: The off-axis current enhancement of the "hollow" sigma profile (unused by the other profiles).
  - `mtheta`: The number of grid points in the poloidal direction.
  - `ma`: The number of grid points in the radial direction.
  - `zeroth`: If set to true, it neglects the Shafranov shift
"""
@kwdef mutable struct LargeAspectRatioConfig <: AnalyticEquilibriumConfig
    r0::Float64 = 10.0
    a::Float64 = 1.0
    beta0::Float64 = 1e-3
    q0::Float64 = 1.5
    qa::Float64 = 3.6        # Edge safety factor (legacy field; not consumed by current sigma_type options)
    B0::Float64 = 1.0        # On-axis toroidal field [T] (scales F and P)
    p_pres::Float64 = 2.0
    p_sig::Float64 = 1.0
    sigma_type::String = "default"
    hollow::Float64 = 0.0
    mtheta::Int = 128
    ma::Int = 128
    zeroth::Bool = false
end

"""
Build a `LargeAspectRatioConfig` from a parsed `[LAR_INPUT]` TOML table.
"""
function LargeAspectRatioConfig(input_dict::Dict{String,Any})
    return LargeAspectRatioConfig(; symbolize_keys(input_dict)...)
end

"""
    lar_init_conditions(rmin, sigma_type, params)

Initializes the starting radius and state vector for solving the LAR ODE system.
Also evaluates the initial derivative using the analytic model.

## Arguments:

  - `rmin`: Normalized starting radius (as a fraction of `lar_a`).
  - `lar_input`: A `LargeAspectRatioConfig` object containing equilibrium parameters.

## Returns:

  - `r`: Physical radius corresponding to `rmin * lar_a`.
  - `y`: Initial state vector of length 5.
"""
function lar_init_conditions(rmin::Float64, lar_input::LargeAspectRatioConfig)
    lar_a = lar_input.a
    lar_r0 = lar_input.r0
    q0 = lar_input.q0

    # Ensure rmin is not too small to avoid numerical issues in LAR equations
    # Use a physically meaningful minimum (0.001% of minor radius)
    if rmin < 1e-6
        @warn "Setting rmin to at least 1e-6 to avoid singularities in LAR equations."
        rmin_safe = 1e-6
    else
        rmin_safe = rmin
    end

    r = rmin_safe * lar_a
    y = zeros(5)

    y[1] = r^2 / (lar_r0 * q0)
    y[2] = 1.0
    y[3] = y[1] * lar_r0 / 2

    dy = zeros(5)
    lar_der(dy, r, y, lar_input)

    y[5] = dy[5] * r / 4

    q = r^2 * y[2] / (lar_r0 * y[1])
    y[4] = (q / r)^2 * y[5] / 2

    return r, y
end

# Current profile σ(x), x = r/a, with σ(0) = sigma0 for every `sigma_type`.
function _lar_sigma(x::Float64, sigma0::Float64, lar_input::LargeAspectRatioConfig)
    p_sig = lar_input.p_sig
    lar_input.sigma_type == "wesson" && return sigma0 * (1 - x^2)^p_sig
    lar_input.sigma_type == "hollow" && return sigma0 * (1 + lar_input.hollow * x^2) * (1 - x^2)^p_sig
    return sigma0 / (1 + x^(2 * p_sig))^(1 + 1 / p_sig)
end

"""
    lar_der(dy, r, y, lar_input)

Evaluates the spatial derivatives of the LAR (Large Aspect Ratio) equilibrium ODE system
at a given radius `r`, using the current state vector `y` and equilibrium parameters.

## Arguments:
- `dy`: A preallocated vector where the computed derivatives will be stored (in-place).
- `r`: The radial position at which the derivative is evaluated.
- `y`: The current state vector.
- `lar_input`: A `LargeAspectRatioConfig` object containing equilibrium shaping parameters.

## Returns:
- 0. The result is stored in-place in `dy`.
"""

function lar_der(dy::Vector{Float64}, r::Float64, y::Vector{Float64}, lar_input::LargeAspectRatioConfig)
    lar_a = lar_input.a
    lar_r0 = lar_input.r0

    q0 = lar_input.q0

    beta0 = lar_input.beta0
    p00 = beta0 / 2.0
    p_pres = lar_input.p_pres

    x = r / lar_a
    xfac = 1 - x^2
    pp = -2 * p_pres * p00 * x * xfac^(p_pres - 1)  # p'
    sigma0 = 2.0 / (q0 * lar_r0)

    sigma = _lar_sigma(x, sigma0, lar_input)

    # LAR equations have singularities at r=0; ensure we never evaluate there
    @assert r > 0.0 "LAR equations require r > 0 (called with r=$r)"

    bsq = (y[1] / r)^2 + y[2]^2     # B²
    q = r^2 * y[2] / (lar_r0 * y[1])

    dy[1] = -pp / bsq * y[1] + sigma * y[2] * r
    dy[2] = -pp / bsq * y[2] - sigma * y[1] / r
    dy[3] = y[1] * lar_r0 / r
    dy[4] = (q / r)^2 * y[5] / r
    dy[5] = y[2] * (r / q)^2 * (r / lar_r0) * (1 - 2 * (lar_r0 * q / y[2])^2 * pp)

    return 0
end

"""
    lar_run(lar_input)

Solves the LAR (Large Aspect Ratio) plasma equilibrium ODE system using analytic profiles
defined by `lar_input`, and returns the full solution table including derived quantities.

## Arguments:
- `lar_input`: A `LargeAspectRatioConfig` object containing profile and geometric parameters.

## Returns:
 - working on it not yet implemented
"""

function lar_run(equil_input::EquilibriumConfig, lar_input::LargeAspectRatioConfig)
    rmin = 1e-4
    lar_a = lar_input.a
    lar_r0 = lar_input.r0
    q0 = lar_input.q0
    beta0 = lar_input.beta0

    p00 = beta0 / 2.0
    lar_input.p_pres = max(lar_input.p_pres, 1.001)

    sigma0 = 2.0 / (q0 * lar_r0)

    ma = lar_input.ma
    mtau = lar_input.mtheta

    function dydr(du, u, p, r)
        lar_input = p
        return lar_der(du, r, u, lar_input)
    end

    r0, y0 = lar_init_conditions(rmin, lar_input)
    tspan = (r0, lar_a)
    p = lar_input

    prob = ODEProblem(dydr, y0, tspan, p)

    sol = solve(prob, Rosenbrock23(; autodiff=false); reltol=equil_input.flux_surface_rtol, abstol=equil_abstol(equil_input.flux_surface_rtol), dense=false)
    check_equil_solve(sol, "large-aspect-ratio radial profiles")

    r_arr = sol.t
    y_mat = reduce(hcat, sol.u)'
    steps = length(r_arr)

    temp = zeros(steps, 9)
    for i in 1:steps
        r = r_arr[i]
        @views y = y_mat[i, :]
        x = r / lar_a
        xfac = 1 - x^2
        pval = p00 * xfac^lar_input.p_pres
        sigma = _lar_sigma(x, sigma0, lar_input)
        q = r^2 * y[2] / (lar_r0 * y[1])
        temp[i, 1] = r
        temp[i, 2] = y[1]
        temp[i, 3] = y[2]
        temp[i, 4] = y[3]
        temp[i, 5] = y[4]
        temp[i, 6] = y[5]
        temp[i, 7] = pval
        temp[i, 8] = sigma
        temp[i, 9] = q
    end

    xs_r = temp[:, 1]
    fs_r = temp[:, 2:9]
    spl = cubic_interp(xs_r, Series(fs_r); extrap=ExtendExtrap())
    spl_deriv = deriv1(spl)

    dr = lar_a / (ma + 1)
    r = 0.0
    psio = temp[end, 4]  # ψ_edge

    sq_xs = zeros(ma + 1)
    sq_fs = zeros(ma + 1, 3)
    r_nodes = zeros(ma + 1)
    rzphi_y_nodes = range(0.0, 1.0; length=mtau + 1)
    rzphi_fs_nodes = zeros(ma + 1, mtau + 1, 2)

    hint = Ref(1)
    for ia in 1:(ma+1)
        r += dr
        r_nodes[ia] = r
        f = spl(r; hint=hint)
        f1 = spl_deriv(r; hint=hint)
        dψdr = f1[3]

        # Fill spline data for sq_in
        sq_xs[ia] = f[3] / psio  # ψ / psio
        sq_fs[ia, 1] = lar_r0 * f[2]  # F = R0 * Bphi
        sq_fs[ia, 2] = f[6]  # P
        sq_fs[ia, 3] = f[8]  # q

        # Compute Shafranov shift and fill rzphi grid
        r2 = -(f[4] * r / f[8]) / dψdr
        if lar_input.zeroth
            r2 = 0.0
        end
        for itau in 1:(mtau+1)
            θ = 2π * (itau - 1) / mtau
            cosθ, sinθ = cos(θ), sin(θ)
            rfac = r + r2 * cosθ
            rzphi_fs_nodes[ia, itau, 1] = lar_r0 + rfac * cosθ
            rzphi_fs_nodes[ia, itau, 2] = rfac * sinθ
        end
    end

    sq_in = cubic_interp(sq_xs, Series(sq_fs); extrap=ExtendExtrap())
    # rz_in_xs is ψ_N (see InverseRunInput struct docs).  Passing physical r
    # works only by accident when lar_a ≈ 1; otherwise the inverse solver
    # extrapolates the (R, Z) splines at outer surfaces.
    rz_in_xs = sq_xs
    rz_in_ys = collect(rzphi_y_nodes)

    @views rzphi_fs_nodes[:, end, :] .= rzphi_fs_nodes[:, 1, :]

    itp_2d_opts = (bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))
    rz_in_R = cubic_interp((rz_in_xs, rz_in_ys), rzphi_fs_nodes[:, :, 1]; itp_2d_opts...)
    rz_in_Z = cubic_interp((rz_in_xs, rz_in_ys), rzphi_fs_nodes[:, :, 2]; itp_2d_opts...)

    # LAR equilibrium has midplane symmetry, so magnetic axis is at Z = 0.
    # Analytic: ingest=nothing — replay regenerates from the [LAR_INPUT] TOML section.
    return InverseRunInput(equil_input, sq_in, rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z, lar_r0, 0.0, psio, nothing)
end
