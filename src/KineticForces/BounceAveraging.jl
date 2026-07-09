"""
    BounceAveraging

Bounce-averaging infrastructure for GAR NTV calculations.
Computes bounce-averaged frequencies (ωb, ωd) and perturbation action (|δJ|²)
as functions of pitch angle λ. For kinetic matrix methods, also computes
bounce-averaged W_μ, W_E vectors and their outer products.

Reference: [Logan et al., Phys. Plasmas 20, 122507 (2013)]
Ports Fortran torque.F90 lines 530-816.
"""

# ============================================================================
# BounceData struct
# ============================================================================

"""
    BounceData

Bounce-averaged quantities as functions of pitch angle λ.
Produced by `compute_bounce_data()`, consumed by pitch integration.
"""
struct BounceData
    nlmda::Int
    lambda::Vector{Float64}           # pitch angle grid points
    dlambda::Vector{Float64}          # weights (dx/dnorm from powspace)
    sigma::Vector{Int}                # 0=trapped, 1=passing
    wb::Vector{Float64}               # bounce frequency ωb(λ) [rad/s]
    wd::Vector{Float64}               # precession drift ωd(λ) [rad/s]
    dJdJ::Vector{Float64}             # ωb|δJ|²/ro² at each λ (real, for scalar torque)
    # For matrix path (nothing if scalar-only): packed layout (nlmda, nqty_matrix(mpert)).
    # Consumer fills into fbnce_data[:, 3:end] by direct copy. See `nqty_matrix` for
    # the 3-Hermitian-triangle + 3-full-block packing (Logan 2015 Eqs 7.30–7.35).
    wmats_vs_lambda::Union{Nothing, Matrix{ComplexF64}}
end


# ============================================================================
# Packed layout for kinetic matrix per-λ storage
# ============================================================================
# Of the six Logan-2015 matrices (Eqs 7.30–7.35), A = W_Z†W_Z, D = W_X†W_X,
# and H = W_Y†W_Y are Hermitian; B = W_Z†W_X, C = W_Z†W_Y, E = W_X†W_Y are not.
# We store only the upper triangle (i ≤ j) for the three Hermitian blocks and
# the full mpert² for the three non-Hermitian blocks. Block packing order:
# A-tri, D-tri, H-tri, B-full, C-full, E-full.

"""Number of packed complex entries per λ for the 6 kinetic matrices."""
@inline nqty_matrix(mpert::Int) = 3 * (mpert * (mpert + 1)) ÷ 2 + 3 * mpert^2

"""Upper-triangle index (1 ≤ i ≤ j ≤ mpert) within a triangular block."""
@inline _tri_idx(i::Int, j::Int) = (j * (j - 1)) ÷ 2 + i

"""Full-block index (column-major) within a non-Hermitian block."""
@inline _full_idx(i::Int, j::Int, mpert::Int) = (j - 1) * mpert + i


# ============================================================================
# Grid generation
# ============================================================================

"""
    powspace(xmin, xmax, pow, num, endpoints) → (points, weights)

Generate a grid with power-law concentration near endpoints.
Port of Fortran `powspace_sub` from equil/grid.f90.

# Arguments
- `xmin, xmax`: Grid bounds
- `pow::Int`: Power of grid concentration (higher = more refined near edges)
- `num::Int`: Number of grid points
- `endpoints::String`: Where to concentrate: "lower", "upper", or "both"

# Returns
- `points::Vector{Float64}`: Grid point locations
- `weights::Vector{Float64}`: Derivatives dx/dnorm (integration weights)
"""
function powspace(xmin::Float64, xmax::Float64, pow::Int, num::Int, endpoints::String)
    if xmax <= xmin
        error("powspace: xmax ($xmax) must be greater than xmin ($xmin)")
    end

    # Linear base grid in [-1,1], [0,1], or [-1,0]
    x = if endpoints == "lower"
        collect(range(-1.0, 0.0, length=num))
    elseif endpoints == "upper"
        collect(range(0.0, 1.0, length=num))
    elseif endpoints == "both"
        collect(range(-1.0, 1.0, length=num))
    else
        error("powspace: invalid endpoints '$endpoints' — use lower, upper, or both")
    end

    # Concentration weight: |(x-1)(x+1)|^pow
    weights = abs.((x .- 1.0) .* (x .+ 1.0)) .^ pow

    # Antiderivative of |(1-x²)|^pow — analytic for pow ≤ 9
    points = _powspace_antideriv(x, pow)

    # Stretch to desired range [xmin, xmax]
    deltay = points[end] - points[1]
    deltax = xmax - xmin
    scale = deltax / deltay
    points .= points .* scale
    weights .= weights .* scale
    points .= points .- points[1] .+ xmin

    # Weight includes the base grid span
    weights .= weights .* (x[end] - x[1])

    return points, weights
end

"""
    _powspace_antideriv(x, pow)

Analytic antiderivative of |(1-x²)|^pow for pow 1-9.
Matches Fortran powspace_sub cases exactly.
"""
function _powspace_antideriv(x::Vector{Float64}, pow::Int)
    if pow == 1
        return @. -x + x^3 / 3
    elseif pow == 2
        return @. x - (2x^3) / 3 + x^5 / 5
    elseif pow == 3
        return @. -x + x^3 - (3x^5) / 5 + x^7 / 7
    elseif pow == 4
        return @. x - (4x^3) / 3 + (6x^5) / 5 - (4x^7) / 7 + x^9 / 9
    elseif pow == 5
        return @. -x + (5x^3) / 3 - 2x^5 + (10x^7) / 7 - (5x^9) / 9 + x^11 / 11
    elseif pow == 6
        return @. x - 2x^3 + 3x^5 - (20x^7) / 7 + (5x^9) / 3 - (6x^11) / 11 + x^13 / 13
    elseif pow == 7
        return @. -x + (7x^3) / 3 - (21x^5) / 5 + 5x^7 - (35x^9) / 9 + (21x^11) / 11 - (7x^13) / 13 + x^15 / 15
    elseif pow == 8
        return @. x - (8x^3) / 3 + (28x^5) / 5 - 8x^7 + (70x^9) / 9 - (56x^11) / 11 + (28x^13) / 13 - (8x^15) / 15 + x^17 / 17
    elseif pow == 9
        return @. -x + 3x^3 - (36x^5) / 5 + 12x^7 - 14x^9 + (126x^11) / 11 - (84x^13) / 13 + (12x^15) / 5 - (9x^17) / 17 + x^19 / 19
    else
        error("powspace: pow=$pow not in analytic database (1-9)")
    end
end


# ============================================================================
# Core bounce averaging
# ============================================================================

"""
    compute_bounce_data(psi, n, l, q, bo, bmax, bmin, theta_bmax,
                        tspl, B_extrap, mfac, chi1, ro, dbob_m_f, divx_m_f,
                        divxfac, wdfac, mass, chrg, T_s, method;
                        nlmda=128, ntheta=128,
                        smat=nothing, tmat=nothing, xmat=nothing,
                        ymat=nothing, zmat=nothing) → BounceData

Compute bounce-averaged quantities as functions of pitch angle λ.
This is the core function that sets up all λ-dependent quantities
needed by the pitch-angle quadrature.

Ports Fortran torque.F90 lines 530-816 (GAR branch).

# Arguments
- `psi`: Normalized poloidal flux
- `n`: Toroidal mode number
- `l`: Bounce harmonic number
- `q`: Safety factor at this ψ
- `bo`: On-axis toroidal field [T]
- `bmax, bmin`: Max/min of B(θ) at this ψ
- `theta_bmax`: θ location of Bmax (nodal knot; the passing-transit start)
- `tspl`: Poloidal interpolant: tspl(θ) → [B, dB/dψ, dB/dθ, J, dJ/dψ]
- `mfac`: Poloidal mode numbers [mlow:mhigh]
- `chi1`: 2π·ψ₀ flux normalization
- `ro`: Major radius [m]
- `dbob_m_f`: δB/B Fourier modes at this ψ (ComplexF64 vector, length mpert)
- `divx_m_f`: ∇·ξ⊥ Fourier modes at this ψ (ComplexF64 vector, length mpert)
- `divxfac, wdfac`: Scaling factors
- `mass`: Particle mass [kg]
- `chrg`: Particle charge [C]
- `T_s`: Species temperature at this ψ [J]
- `method`: Method string (first char: f/t/p determines λ range)

# Keyword Arguments
- `nlmda`: Number of pitch angle grid points (default 128, matching Fortran pentrc nlmda)
- `ntheta`: Number of poloidal grid points per bounce (default 128)
- `smat, tmat, xmat, ymat, zmat`: Geometric matrices (mpert×mpert) for kinetic matrix path
"""
function compute_bounce_data(
    psi::Float64, n::Int, l::Int, q::Float64,
    bo::Float64, bmax::Float64, bmin::Float64,
    theta_bmax::Float64,
    tspl, B_extrap, mfac::Vector{Int}, chi1::Float64, ro::Float64,
    dbob_m_f::Vector{ComplexF64}, divx_m_f::Vector{ComplexF64},
    divxfac::Float64, wdfac::Float64,
    mass::Float64, chrg::Float64,
    T_s::Float64, method::String;
    nlmda::Int=128, ntheta::Int=128,
    smat::Union{Nothing,Matrix{ComplexF64}}=nothing,
    tmat::Union{Nothing,Matrix{ComplexF64}}=nothing,
    xmat::Union{Nothing,Matrix{ComplexF64}}=nothing,
    ymat::Union{Nothing,Matrix{ComplexF64}}=nothing,
    zmat::Union{Nothing,Matrix{ComplexF64}}=nothing
)
    mpert = length(mfac)
    do_matrices = !isnothing(smat)

    # Per-surface scratch, reused across all λ.
    scr = BounceScratch(ntheta, mpert)

    # Trapped-passing boundary and λ range
    lmdatpb = bo / bmax
    lmdamax = bo / bmin

    # Build λ grid based on method variant (Fortran lines 569-585)
    method_char = method[1]
    lambda, dlambda = _build_lambda_grid(method_char, lmdatpb, lmdamax, nlmda)

    # Pre-allocate output arrays
    wb_arr = zeros(Float64, nlmda)
    wd_arr = zeros(Float64, nlmda)
    dJdJ_arr = zeros(Float64, nlmda)
    sigma_arr = zeros(Int, nlmda)
    wmats_arr = do_matrices ? zeros(ComplexF64, nlmda, nqty_matrix(mpert)) : nothing

    # Thermal speed and drift normalization
    bhat = sqrt(2 * T_s / mass) / ro
    dhat = (T_s / chrg) / (bo * ro^2)

    for ilmda in 1:nlmda
        lmda = lambda[ilmda]

        # Determine trapped/passing (Fortran line 591-595)
        if lmda > (bo / bmax)
            sigma = 0  # trapped
        else
            sigma = 1  # passing
        end
        sigma_arr[ilmda] = sigma
        lnq = l + sigma * n * q  # effective resonance number

        # Find bounce points and build θ sub-grid
        _, _, tdt_pts, tdt_wts = _find_bounce_points_and_grid(
            lmda, bo, sigma, B_extrap, theta_bmax, psi, ntheta)

        # Bounce integrals over θ (Fortran lines 674-735)
        wbbar, wdbar, dJdJ_val, wmats_lmda = _bounce_integrate(
            tdt_pts, tdt_wts, lmda, lnq, sigma, n, q, bo,
            tspl, B_extrap, chi1, ro, mfac, dbob_m_f, divx_m_f, divxfac, wdfac,
            do_matrices, mpert, smat, tmat, xmat, ymat, zmat, scr)

        # Physical frequencies (Fortran lines 744-745)
        wb_arr[ilmda] = wbbar * bhat
        wd_arr[ilmda] = wdbar * dhat
        dJdJ_arr[ilmda] = dJdJ_val

        if do_matrices && !isnothing(wmats_lmda)
            @inbounds for iq in eachindex(wmats_lmda)
                wmats_arr[ilmda, iq] = wmats_lmda[iq]
            end
        end
    end

    return BounceData(nlmda, lambda, dlambda, sigma_arr,
                      wb_arr, wd_arr, dJdJ_arr, wmats_arr)
end


"""
    BounceScratch(ntheta, mpert)

Per-surface scratch buffers for the bounce-averaging inner loops. Allocated once in
`compute_bounce_data` and reused across every λ (each λ previously reallocated these
θ- and mode-sized arrays). Sizes are fixed for a flux surface (`ntheta` sub-grid points,
`mpert` Fourier modes), so a single set of buffers serves the whole λ sweep. Buffers that
the callers zero-initialize are `fill!`-reset per λ, preserving bit-for-bit results.

## Fields
- `g_wb::Vector{Float64}`: length `ntheta` — bounce-action integrand samples
- `g_wd::Vector{Float64}`: length `ntheta` — drift integrand samples
- `cum_wb_arr::Vector{Float64}`: length `ntheta` — cumulative bounce action
- `d_re::Vector{Float64}`: length `ntheta` — real spline-derivative workspace
- `cp::Vector{Float64}`: length `ntheta` — Thomas-sweep workspace
- `jvtheta::Vector{ComplexF64}`: length `ntheta` — action integrand
- `bj_samples::Vector{ComplexF64}`: length `ntheta` — action bounce-integral samples
- `d_c::Vector{ComplexF64}`: length `ntheta` — complex spline-derivative workspace
- `wsamp::Vector{ComplexF64}`: length `ntheta` — per-mode W quadrature samples
- `wmu_mt::Matrix{ComplexF64}`: `mpert × ntheta` — W_μ per θ
- `wen_mt::Matrix{ComplexF64}`: `mpert × ntheta` — W_E per θ
- `expm::Vector{ComplexF64}`: length `mpert` — Fourier basis at a θ
- `pl::Vector{ComplexF64}`: length `ntheta` — bounce phase factor
- `wmu_ba::Vector{ComplexF64}`: length `mpert` — bounce-averaged W_μ
- `wen_ba::Vector{ComplexF64}`: length `mpert` — bounce-averaged W_E
- `wmats_lmda::Vector{ComplexF64}`: length `nqty_matrix(mpert)` — packed W outer products
- `tspl_f::Vector{Float64}`: length 5 — in-place tspl(θ) evaluation
"""
struct BounceScratch{TR, TC}
    g_wb::Vector{Float64}
    g_wd::Vector{Float64}
    cum_wb_arr::Vector{Float64}
    jvtheta::Vector{ComplexF64}
    wmu_mt::Matrix{ComplexF64}
    wen_mt::Matrix{ComplexF64}
    expm::Vector{ComplexF64}
    pl::Vector{ComplexF64}
    wmu_ba::Vector{ComplexF64}
    wen_ba::Vector{ComplexF64}
    wmats_lmda::Vector{ComplexF64}
    tspl_f::Vector{Float64}
    itp_r::TR
    itp_c::TC
end

function BounceScratch(ntheta::Int, mpert::Int)
    # Reusable fit-and-integrate interpolants on the unit θ-quadrature grid;
    # per λ their y contents are overwritten and z refit in place.
    xs = range(0.0, 1.0, length=ntheta)
    itp_r = cubic_interp(xs, zeros(Float64, ntheta); bc=CubicFit())
    itp_c = cubic_interp(xs, zeros(ComplexF64, ntheta); bc=CubicFit())
    return BounceScratch(
        Vector{Float64}(undef, ntheta),
        Vector{Float64}(undef, ntheta),
        Vector{Float64}(undef, ntheta),
        Vector{ComplexF64}(undef, ntheta),
        Matrix{ComplexF64}(undef, mpert, ntheta),
        Matrix{ComplexF64}(undef, mpert, ntheta),
        Vector{ComplexF64}(undef, mpert),
        Vector{ComplexF64}(undef, ntheta),
        Vector{ComplexF64}(undef, mpert),
        Vector{ComplexF64}(undef, mpert),
        Vector{ComplexF64}(undef, nqty_matrix(mpert)),
        Vector{Float64}(undef, 5),
        itp_r,
        itp_c,
    )
end

"""Refit `itp` in place to its current `y` contents (reuses the cached factorization)."""
@inline function _refit!(itp)
    FastInterpolations._solve_system!(itp.z, itp.cache, itp.y, itp.bc)
    return itp
end


# ============================================================================
# Internal helpers
# ============================================================================

"""
Build λ grid based on method character (f/t/p).
Returns (lambda, dlambda) with endpoints excluded.
"""
function _build_lambda_grid(method_char::Char, lmdatpb::Float64, lmdamax::Float64, nlmda::Int)
    lmdamin = 0.0

    if method_char == 't'
        # Trapped only: λ ∈ (lmdatpb, lmdamax), exclude endpoints
        pts_inc, wts_inc = powspace(lmdatpb, lmdamax, 1, 2 + nlmda, "both")
        return pts_inc[2:end-1], wts_inc[2:end-1]

    elseif method_char == 'p'
        # Passing only: λ ∈ (lmdamin, lmdatpb), exclude endpoints
        pts_inc, wts_inc = powspace(lmdamin, lmdatpb, 1, 2 + nlmda, "both")
        return pts_inc[2:end-1], wts_inc[2:end-1]

    else  # 'f' = full
        if lmdatpb ≈ lmdamax
            @warn "bmax ≈ bmin at this flux surface" maxlog=1
        end
        # Passing half with refinement near upper boundary
        nhalf_p = nlmda ÷ 2
        nhalf_t = nlmda - nhalf_p
        pts_p, wts_p = powspace(lmdamin, lmdatpb, 2, 2 + nhalf_p, "upper")
        pts_t, wts_t = powspace(lmdatpb, lmdamax, 2, 2 + nhalf_t, "lower")
        # Exclude boundary points
        lambda = vcat(pts_p[2:end-1], pts_t[2:end-1])
        dlambda = vcat(wts_p[2:end-1], wts_t[2:end-1])
        return lambda, dlambda
    end
end


"""
    _vpar_from_extrap(B_extrap, lmda, bo, θ) → v_par

Parallel-velocity factor `v_par = 1 − (λ/bo)·B(θ)` evaluated from the **extrap**
cubic of B (`B_extrap`), matching Fortran's separate `vspl` (torque.F90:599-600,
`vpar = vspl%f(1)` at :677 — "more consistent w/ bnce pts than direct from tspl").
Because the "extrap" endpoint derivatives are linear in the nodal data, the extrap
cubic of `1−(λ/bo)B` equals `1−(λ/bo)·(extrap cubic of B)`, so we build `B_extrap`
once per surface (CubicFit ≡ Fortran extrap) and reuse it for the vpar factor, the
bounce-point roots, and the deepest-well test — NOT the periodic `tspl`.
"""
@inline _vpar_from_extrap(B_extrap, lmda::Float64, bo::Float64, θ::Float64) =
    1.0 - (lmda / bo) * B_extrap(mod(θ, 1.0))


"""
Find bounce points for trapped/passing particles and build θ sub-grid.
Returns (t1, t2, theta_points, theta_weights).
"""
function _find_bounce_points_and_grid(
    lmda::Float64, bo::Float64, sigma::Int,
    B_extrap, theta_bmax::Float64, psi::Float64,
    ntheta::Int
)
    if sigma == 0  # trapped
        # Bounce points = roots of v_par(θ) = 1 − (λ/bo)·B_extrap(θ) in (0,1).
        # Fortran fits vspl "extrap" and calls spline_roots (torque.F90:600-602), which
        # solves each equilibrium interval's cubic analytically. Roots.find_zeros
        # adaptively finds ALL roots of the SAME extrap cubic (B_extrap), so the root
        # values match spline_roots (robust to the near-boundary interior dip a fixed
        # sign-change scan could miss). Sorted DESCENDING to match spline_roots order
        # (spline.f:1782-1785), which the marginally-trapped and deepest-well wrap
        # logic below assume.
        vpar_fn = θ -> _vpar_from_extrap(B_extrap, lmda, bo, θ)
        bpts = sort!(Roots.find_zeros(vpar_fn, 0.0, 1.0); rev=true)

        nbpts = length(bpts)
        if nbpts < 1
            @warn "No bounce points found at psi=$psi, λ=$lmda — using full transit" maxlog=3
            t1 = theta_bmax
            t2 = theta_bmax + 1.0
        elseif nbpts < 2
            # Marginally trapped
            t1 = bpts[1]
            t2 = bpts[1] + 1.0
        else
            # Find deepest potential well (Fortran lines 616-639)
            t1, t2 = _find_deepest_well(bpts, B_extrap, lmda, bo)
        end

        # Power-law grid refined near bounce points
        tdt_pts, tdt_wts = powspace(t1, t2, 4, ntheta, "both")

    else  # passing — full transit
        t1 = theta_bmax
        t2 = theta_bmax + 1.0
        tdt_pts, tdt_wts = powspace(t1, t2, 2, ntheta, "both")
    end

    return t1, t2, tdt_pts, tdt_wts
end


"""
Find deepest potential well among bounce point pairs.
Ports Fortran lines 616-639.
"""
function _find_deepest_well(bpts::Vector{Float64}, B_extrap, lmda::Float64, bo::Float64)
    nbpts = length(bpts)
    best_vpar = 0.0
    best_t1 = 0.0
    best_t2 = 1.0

    for i in 1:nbpts
        j = (i % nbpts) + 1  # next bounce point, wrapping
        if bpts[i] > bpts[j]
            # Wrapping case: midpoint crosses θ=0/1 boundary
            θmid = mod(0.5 * (bpts[i] + bpts[j] + 1.0), 1.0)
        else
            θmid = 0.5 * (bpts[i] + bpts[j])
        end
        vpar_mid = _vpar_from_extrap(B_extrap, lmda, bo, θmid)
        if vpar_mid > best_vpar
            best_t1 = bpts[i]
            best_t2 = bpts[j]
            if best_t2 < best_t1
                best_t2 += 1.0
            end
            best_vpar = vpar_mid
        end
    end

    if best_vpar ≈ 0.0
        @warn "Could not find potential well with positive v_par" maxlog=1
    end

    return best_t1, best_t2
end


"""
    _bounce_integrate(...)

Perform bounce integrals over θ sub-grid.
Computes ωb_bar, ωd_bar, |δJ|², and optionally W matrix outer products.
Ports Fortran torque.F90 lines 674-793.
"""
function _bounce_integrate(
    tdt_pts::Vector{Float64}, tdt_wts::Vector{Float64},
    lmda::Float64, lnq::Float64, sigma::Int, n::Int, q::Float64, bo::Float64,
    tspl, B_extrap, chi1::Float64, ro::Float64,
    mfac::Vector{Int}, dbob_m_f::Vector{ComplexF64}, divx_m_f::Vector{ComplexF64},
    divxfac::Float64, wdfac::Float64,
    do_matrices::Bool, mpert::Int,
    smat, tmat, xmat, ymat, zmat, scr::BounceScratch
)
    ntheta = length(tdt_pts)
    theta0 = tdt_pts[1]

    # θ-sample integrands (zero-reset per λ: the loop populates 2:ntheta-1 with
    # continue/break paths that rely on unwritten entries staying 0)
    g_wb = scr.g_wb    # J·B/√v_par · dθ/dx
    g_wd = scr.g_wd    # drift integrand · dθ/dx
    fill!(g_wb, 0.0)
    fill!(g_wd, 0.0)

    # Action integrand
    jvtheta = scr.jvtheta
    fill!(jvtheta, ComplexF64(0.0))

    # W vectors for matrix path
    wmu_mt = scr.wmu_mt
    wen_mt = scr.wen_mt
    if do_matrices
        fill!(wmu_mt, ComplexF64(0.0))
        fill!(wen_mt, ComplexF64(0.0))
    end

    # Scratch for hot-loop tspl evaluation + Fourier-basis buffer (fully written per use).
    tspl_f = scr.tspl_f
    expm = scr.expm

    for i in 2:ntheta-1  # Edge weights are 0 from powspace
        θ = tdt_pts[i]
        dt = tdt_wts[i]
        θmod = mod(θ, 1.0)

        tspl(tspl_f, θmod)
        B_val = tspl_f[1]
        dBdpsi = tspl_f[2]
        # dBdtheta = tspl_f[3]  # not needed here
        jac = tspl_f[4]
        djdpsi = tspl_f[5]

        # vpar from the extrap cubic of B (Fortran vspl, torque.F90:677), NOT the
        # periodic tspl B_val — which stays as the numerator field below (Fortran
        # bspl uses tspl%f(1)/(4) for the numerator, vspl%f(1) only for 1/√vpar).
        vpar = 1.0 - (lmda / bo) * B_extrap(θmod)

        if vpar <= 0
            # Zero crossing near bounce points — Fortran torque.F90:678-697 fill
            # semantics (sample i ↔ Fortran node i-1):
            if i < ntheta ÷ 2
                # Before midpoint: restart — zero everything up to and including
                # this sample (Fortran bspl%fs(:i-1,:)=0; jvtheta(:i)=0).
                fill!(view(g_wb, 1:i), 0.0)
                fill!(view(g_wd, 1:i), 0.0)
                fill!(view(jvtheta, 1:i), ComplexF64(0.0))
                continue
            else
                # After midpoint: hold the previous sample for the rest of the
                # grid. Fortran fills BOTH quantities from fs(i-2,1) — the wb
                # integrand — including the wd slot (bspl%fs(i-1:,2)=bspl%fs(i-2,1));
                # reproduced verbatim for parity.
                fill!(view(g_wb, i:ntheta), g_wb[i-1])
                fill!(view(g_wd, i:ntheta), g_wb[i-1])
                fill!(view(jvtheta, i:ntheta), jvtheta[i-1])
                break
            end
        end

        sqrt_vpar = sqrt(vpar)

        # Bounce integrands (Fortran lines 698-701)
        g_wb[i] = dt * jac * B_val / sqrt_vpar
        g_wd[i] = dt * jac * dBdpsi * (1.0 - 1.5 * lmda * B_val / bo) / sqrt_vpar +
                  dt * djdpsi * B_val * sqrt_vpar

        # Fourier modes at this θ (Fortran lines 702-708) — write into pre-allocated
        # expm buffer using the ORIGINAL expression order to preserve bit-level parity.
        @inbounds for mi in 1:mpert
            expm[mi] = cis(twopi * mfac[mi] * θ)
        end
        # Replaces `sum(dbob_m_f .* expm)` / `sum(divx_m_f .* expm) * divxfac`
        # with direct accumulators; same evaluation order as the broadcast + sum.
        dbob = ComplexF64(0.0)
        divx = ComplexF64(0.0)
        @inbounds for mi in 1:mpert
            dbob += dbob_m_f[mi] * expm[mi]
            divx += divx_m_f[mi] * expm[mi]
        end
        divx *= divxfac

        # Action integrand (Fortran line 706-708)
        phase = cis(-twopi * n * q * (θ - theta0))
        jvtheta[i] = dt * jac * B_val *
            (divx * sqrt_vpar + dbob * (1.0 - 1.5 * lmda * B_val / bo) / sqrt_vpar) *
            phase

        # W vectors for matrix path (Fortran lines 722-727). Element-by-element
        # write preserving original broadcast evaluation order exactly to keep
        # bit-level parity (matters because downstream quadrature is tolerance-sensitive).
        if do_matrices
            wmu_pre = dt * (lmda / bo)
            wen_pre = dt
            @inbounds for mi in 1:mpert
                wmu_mt[mi, i] = wmu_pre * expm[mi] / sqrt_vpar * phase / (2 * chi1)
                wen_mt[mi, i] = wen_pre * expm[mi] / (B_val * sqrt_vpar) * phase / (2 * chi1)
            end
        end

        # Smooth backfill for points zeroed before a restart (Fortran lines 730-734,
        # index ranges preserved verbatim: g fills samples 3..i-1, jv fills 2..i-1).
        if i >= 3 && g_wb[i-1] == 0.0
            fill!(view(g_wb, 3:i-1), g_wb[i])
            fill!(view(g_wd, 3:i-1), g_wd[i])
            fill!(view(jvtheta, 2:i-1), jvtheta[i])
        end
    end

    # Total bounce integrals — Fortran fits bspl%xs = linspace(0,1,ntheta) with
    # spline_fit("extrap") and integrates the cubic exactly (spline_int). The
    # tdt(2,i) weights contain dθ/dx so the samples live on the unit x-grid.
    itp_r = scr.itp_r
    copyto!(itp_r.y, g_wb)
    _refit!(itp_r)
    fsi_wb = scr.cum_wb_arr  # per-surface scratch; fully overwritten
    FastInterpolations.cumulative_integrate!(fsi_wb, itp_r)
    total_wb = fsi_wb[ntheta]
    copyto!(itp_r.y, g_wd)
    _refit!(itp_r)
    total_wd = FastInterpolations.integrate(itp_r)

    if total_wb ≈ 0.0
        # Degenerate case — return zeros
        return 0.0, 0.0, 0.0, nothing
    end

    # Bounce-averaged frequencies (Fortran lines 740-741)
    wbbar = ro * twopi / ((2 - sigma) * total_wb)
    wdbar = ro^2 * bo * wdfac * wbbar * 2 * (2 - sigma) * total_wd

    # Phase factor (Fortran line 750): pl_i = exp(-2πi·lnq·h(θ_i)) with
    # h = fsi_wb/((2-σ)·total_wb) the cumulative spline integral — NOT a
    # running trapezoid — matching bspl%fsi exactly.
    pl_denom = (2 - sigma) * total_wb
    one_minus_sigma = 1 - sigma
    pl = scr.pl  # per-surface scratch; fully overwritten
    @inbounds for i in 1:ntheta
        pl[i] = cis(-twopi * lnq * fsi_wb[i] / pl_denom)
    end

    # Action bounce integral (Fortran bjspl: cspline_fit("extrap") + cspline_int).
    itp_c = scr.itp_c
    bj_samples = itp_c.y
    @inbounds for i in 1:ntheta
        bj_samples[i] = conj(jvtheta[i]) * (pl[i] + one_minus_sigma / pl[i])
    end
    _refit!(itp_c)
    bj_integral = FastInterpolations.integrate(itp_c)

    # |δJ|² (Fortran line 756) — division by 2 corrects quadratic form
    dJdJ_val = wbbar * abs(bj_integral)^2 / 2.0 / ro^2

    # Matrix path: bounce-average W vectors and form outer products (Fortran lines 759-793)
    wmats_lmda = nothing
    if do_matrices
        # Bounce-average W_μ and W_E vectors (Fortran lines 762-767): per mode,
        # cspline_fit("extrap") + cspline_int of conj(W_m(θ))·(pl + (1-σ)/pl),
        # matching the bjspl quadrature above. Results land in per-surface scratch,
        # fully overwritten per mode.
        wmu_ba = scr.wmu_ba
        wen_ba = scr.wen_ba
        wsamp = itp_c.y
        @inbounds for mi in 1:mpert
            for i in 1:ntheta
                wsamp[i] = conj(wmu_mt[mi, i]) * (pl[i] + one_minus_sigma / pl[i])
            end
            _refit!(itp_c)
            wmu_ba[mi] = FastInterpolations.integrate(itp_c)
            for i in 1:ntheta
                wsamp[i] = conj(wen_mt[mi, i]) * (pl[i] + one_minus_sigma / pl[i])
            end
            _refit!(itp_c)
            wen_ba[mi] = FastInterpolations.integrate(itp_c)
        end

        # Reshape as 1×mpert for matrix multiply (Fortran lines 771-772)
        wmmt = reshape(wmu_ba, 1, mpert)
        wemt = reshape(wen_ba, 1, mpert)

        # Build W_X, W_Y, W_Z via geometric matrices (Fortran lines 773-775)
        wxmt = wmmt * xmat
        wymt = wmmt * (3 * smat + ymat) - 2.0 * wemt * smat
        wzmt = wmmt * (3 * tmat + zmat) - 2.0 * wemt * tmat

        # Flatten the 1×mpert row vectors to mpert-vectors for outer-product loops.
        wx = vec(wxmt)
        wy = vec(wymt)
        wz = vec(wzmt)

        # Scale by wbbar/ro² (Fortran line 789)
        scale = wbbar / ro^2
        Mu = (mpert * (mpert + 1)) ÷ 2
        wmats_lmda = scr.wmats_lmda
        # Fill with NaN to catch uninitialized entries
        fill!(wmats_lmda, ComplexF64(NaN, NaN))

        # A (Hermitian): upper triangle of W_Z†W_Z, rank-1 → conj(wz[i])·wz[j].
        off = 0
        @inbounds for j in 1:mpert, i in 1:j
            wmats_lmda[off + _tri_idx(i, j)] = conj(wz[i]) * wz[j] * scale
        end
        off += Mu
        # D (Hermitian): upper triangle of W_X†W_X.
        @inbounds for j in 1:mpert, i in 1:j
            wmats_lmda[off + _tri_idx(i, j)] = conj(wx[i]) * wx[j] * scale
        end
        off += Mu
        # H (Hermitian): upper triangle of W_Y†W_Y.
        @inbounds for j in 1:mpert, i in 1:j
            wmats_lmda[off + _tri_idx(i, j)] = conj(wy[i]) * wy[j] * scale
        end
        off += Mu
        # B (full): W_Z†W_X.
        @inbounds for j in 1:mpert, i in 1:mpert
            wmats_lmda[off + _full_idx(i, j, mpert)] = conj(wz[i]) * wx[j] * scale
        end
        off += mpert^2
        # C (full): W_Z†W_Y.
        @inbounds for j in 1:mpert, i in 1:mpert
            wmats_lmda[off + _full_idx(i, j, mpert)] = conj(wz[i]) * wy[j] * scale
        end
        off += mpert^2
        # E (full): W_X†W_Y.
        @inbounds for j in 1:mpert, i in 1:mpert
            wmats_lmda[off + _full_idx(i, j, mpert)] = conj(wx[i]) * wy[j] * scale
        end
    end

    return wbbar, wdbar, dJdJ_val, wmats_lmda
end
