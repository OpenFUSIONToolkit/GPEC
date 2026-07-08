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
    compute_bounce_data(psi, n, l, q, bo, bmax, bmin, ibmax, theta_bmax,
                        tspl, mfac, chi1, ro, dbob_m_f, divx_m_f,
                        divxfac, wdfac, mass, chrg, T_s, method;
                        nlmda=64, ntheta=128,
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
- `ibmax`: Index of Bmax in poloidal grid
- `theta_bmax`: θ location of Bmax
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
- `nlmda`: Number of pitch angle grid points (default 64)
- `ntheta`: Number of poloidal grid points per bounce (default 128)
- `smat, tmat, xmat, ymat, zmat`: Geometric matrices (mpert×mpert) for kinetic matrix path
"""
function compute_bounce_data(
    psi::Float64, n::Int, l::Int, q::Float64,
    bo::Float64, bmax::Float64, bmin::Float64,
    ibmax::Int, theta_bmax::Float64,
    tspl, mfac::Vector{Int}, chi1::Float64, ro::Float64,
    dbob_m_f::Vector{ComplexF64}, divx_m_f::Vector{ComplexF64},
    divxfac::Float64, wdfac::Float64,
    mass::Float64, chrg::Float64,
    T_s::Float64, method::String;
    nlmda::Int=64, ntheta::Int=128,
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
            lmda, bo, sigma, tspl, ibmax, theta_bmax,
            lmdatpb, lmdamax, psi, ntheta, scr)

        # Bounce integrals over θ (Fortran lines 674-735)
        wbbar, wdbar, dJdJ_val, wmats_lmda = _bounce_integrate(
            tdt_pts, tdt_wts, lmda, lnq, sigma, n, q, bo,
            tspl, chi1, ro, mfac, dbob_m_f, divx_m_f, divxfac, wdfac,
            do_matrices, mpert, smat, tmat, xmat, ymat, zmat, scr)

        # Physical frequencies (Fortran lines 744-745)
        wb_arr[ilmda] = wbbar * bhat
        wd_arr[ilmda] = wdbar * dhat
        dJdJ_arr[ilmda] = dJdJ_val

        if do_matrices && !isnothing(wmats_lmda)
            @inbounds for q in 1:length(wmats_lmda)
                wmats_arr[ilmda, q] = wmats_lmda[q]
            end
        end
    end

    return BounceData(nlmda, lambda, dlambda, sigma_arr,
                      wb_arr, wd_arr, dJdJ_arr, wmats_arr)
end


"""
    BounceScratch(ntheta, mpert; nfine=256)

Per-surface scratch buffers for the bounce-averaging inner loops. Allocated once in
`compute_bounce_data` and reused across every λ (each λ previously reallocated these
θ- and mode-sized arrays). Sizes are fixed for a flux surface (`ntheta` sub-grid points,
`mpert` Fourier modes), so a single set of buffers serves the whole λ sweep. Buffers that
the callers zero-initialize are `fill!`-reset per λ, preserving bit-for-bit results.

## Fields
- `cum_wb_arr::Vector{Float64}`: length `ntheta` — cumulative bounce action
- `jvtheta::Vector{ComplexF64}`: length `ntheta` — action integrand
- `wmu_mt::Matrix{ComplexF64}`: `mpert × ntheta` — W_μ per θ
- `wen_mt::Matrix{ComplexF64}`: `mpert × ntheta` — W_E per θ
- `expm::Vector{ComplexF64}`: length `mpert` — Fourier basis at a θ
- `pl::Vector{ComplexF64}`: length `ntheta` — bounce phase factor
- `wmu_ba::Vector{ComplexF64}`: length `mpert` — bounce-averaged W_μ
- `wen_ba::Vector{ComplexF64}`: length `mpert` — bounce-averaged W_E
- `wmats_lmda::Vector{ComplexF64}`: length `nqty_matrix(mpert)` — packed W outer products
- `tspl_f::Vector{Float64}`: length 5 — in-place tspl(θ) evaluation
- `vpar_fine::Vector{Float64}`: length `nfine+1` — v_par(θ) for bounce-point search;
  `_find_bounce_points_and_grid` derives its fine-grid size from this buffer's length
"""
struct BounceScratch
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
    vpar_fine::Vector{Float64}
end

function BounceScratch(ntheta::Int, mpert::Int; nfine::Int=256)
    return BounceScratch(
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
        Vector{Float64}(undef, nfine + 1),
    )
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
Find bounce points for trapped/passing particles and build θ sub-grid.
Returns (t1, t2, theta_points, theta_weights).
"""
function _find_bounce_points_and_grid(
    lmda::Float64, bo::Float64, sigma::Int,
    tspl, ::Int, theta_bmax::Float64,
    ::Float64, ::Float64, psi::Float64,
    ntheta::Int, scr::BounceScratch
)
    if sigma == 0  # trapped
        # Build v_par(θ) = 1 - (λ/bo)*B(θ) and find roots on a dense θ grid.
        # Grid size derives from the scratch buffer so it stays in sync with BounceScratch's nfine.
        nfine = length(scr.vpar_fine) - 1
        theta_fine = range(0.0, 1.0, length=nfine+1)
        vpar_fine = scr.vpar_fine
        @inbounds for i in 1:(nfine+1)
            tspl(scr.tspl_f, mod(theta_fine[i], 1.0))
            vpar_fine[i] = 1.0 - (lmda / bo) * scr.tspl_f[1]
        end

        # Find zero crossings
        bpts = Float64[]
        for i in 1:nfine
            if vpar_fine[i] * vpar_fine[i+1] < 0
                # Bisect for better accuracy
                θ_root = _bisect_vpar(tspl, scr.tspl_f, lmda, bo, theta_fine[i], theta_fine[i+1])
                push!(bpts, θ_root)
            end
        end

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
            t1, t2 = _find_deepest_well(bpts, tspl, scr.tspl_f, lmda, bo)
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
    _bisect_vpar(tspl, lmda, bo, θa, θb; tol=1e-12, maxiter=50) → θ

Bisect on `θ ∈ [θa, θb]` to find a bounce point — the angle where the parallel
velocity vanishes, `v_par(θ) = 1 - (λ/bo) · B(θ) = 0`. The caller is responsible
for supplying a bracket where `v_par(θa)` and `v_par(θb)` have opposite signs;
this routine does not check, it just halves toward the sign change.

# Arguments
- `tspl`: 1D θ-interpolant returning at least `[B(θ), …]` at first index;
  `tspl(mod(θ, 1.0))[1]` extracts the local field magnitude.
- `lmda`: pitch-angle parameter λ = μ·bo / E.
- `bo`: on-axis toroidal field used to normalise λ.
- `θa, θb`: bracket endpoints (normalised θ ∈ [0,1)); typically straddling a B-peak.

# Keyword arguments
- `tol`: termination tolerance — exits when either `|v_par(θ_mid)| < tol` or the
  bracket width `θb − θa < tol`. Default `1e-12` is tight enough that the
  residual `v_par` is dominated by the spline-evaluation roundoff of `tspl`.
- `maxiter`: hard iteration cap. Default `50` halves the bracket by ~10⁻¹⁵, well
  past `tol` on a unit-scale bracket; the cap is a runaway guard rather than the
  expected exit. On exhaustion the midpoint of the final bracket is returned.

# Returns
- `θ::Float64`: the converged (or capped) bounce-point angle.
"""
function _bisect_vpar(tspl, tspl_f::Vector{Float64}, lmda::Float64, bo::Float64, θa::Float64, θb::Float64; tol=1e-12, maxiter=50)
    tspl(tspl_f, mod(θa, 1.0))
    va = 1.0 - (lmda / bo) * tspl_f[1]
    for _ in 1:maxiter
        θm = 0.5 * (θa + θb)
        tspl(tspl_f, mod(θm, 1.0))
        vm = 1.0 - (lmda / bo) * tspl_f[1]
        if abs(vm) < tol || (θb - θa) < tol
            return θm
        end
        if va * vm < 0
            θb = θm
        else
            θa = θm
            va = vm
        end
    end
    return 0.5 * (θa + θb)
end


"""
Find deepest potential well among bounce point pairs.
Ports Fortran lines 616-639.
"""
function _find_deepest_well(bpts::Vector{Float64}, tspl, tspl_f::Vector{Float64}, lmda::Float64, bo::Float64)
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
        tspl(tspl_f, mod(θmid, 1.0))
        vpar_mid = 1.0 - (lmda / bo) * tspl_f[1]
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
    tspl, chi1::Float64, ro::Float64,
    mfac::Vector{Int}, dbob_m_f::Vector{ComplexF64}, divx_m_f::Vector{ComplexF64},
    divxfac::Float64, wdfac::Float64,
    do_matrices::Bool, mpert::Int,
    smat, tmat, xmat, ymat, zmat, scr::BounceScratch
)
    ntheta = length(tdt_pts)
    theta0 = tdt_pts[1]

    # Cumulative bounce integrals
    cum_wb = 0.0
    cum_wd = 0.0

    # θ-scratch arrays reused across λ from the per-surface BounceScratch. Zero-reset
    # the ones the loop populates only partially (i in 2:ntheta-1, with continue/break),
    # so unwritten entries stay 0 exactly as the previous `zeros(...)` did.
    cum_wb_arr = scr.cum_wb_arr
    fill!(cum_wb_arr, 0.0)

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

        vpar = 1.0 - (lmda / bo) * B_val

        if vpar <= 0
            # Zero crossing near bounce points — handle like Fortran lines 678-697
            if i < ntheta ÷ 2
                # Before midpoint: zero out previous entries
                continue
            else
                # After midpoint: clamp remaining entries
                break
            end
        end

        sqrt_vpar = sqrt(vpar)

        # Bounce integrands (Fortran lines 698-701)
        wb_integrand = dt * jac * B_val / sqrt_vpar
        wd_integrand = dt * jac * dBdpsi * (1.0 - 1.5 * lmda * B_val / bo) / sqrt_vpar +
                       dt * djdpsi * B_val * sqrt_vpar

        cum_wb += wb_integrand
        cum_wd += wd_integrand
        # Trapezoidal cumulative (matches Fortran spline_int semantics on linear fn):
        # bspl%fsi(j)/Δx = g_1 + ... + g_{j-1} + g_j/2, so subtract half the current sample.
        cum_wb_arr[i] = cum_wb - wb_integrand / 2

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
    end

    # Total bounce integrals — Fortran splines over bspl%xs = linspace(0,1,ntheta)
    # and integrates via spline_int, which is ≈ (1/(ntheta-1)) × Σ f_i. The tdt(2,i)
    # weights contain dθ/dx so Σ tdt·f is raw Riemann; divide by (ntheta-1) to get
    # the integral over the unit linear space [0,1] that Fortran produces.
    nrm = 1.0 / (ntheta - 1)
    total_wb = cum_wb * nrm
    total_wd = cum_wd * nrm

    if total_wb ≈ 0.0
        # Degenerate case — return zeros
        return 0.0, 0.0, 0.0, nothing
    end

    # Bounce-averaged frequencies (Fortran lines 740-741)
    wbbar = ro * twopi / ((2 - sigma) * total_wb)
    wdbar = ro^2 * bo * wdfac * wbbar * 2 * (2 - sigma) * total_wd

    # Phase factor (Fortran line 750). Ratio cum_wb_arr[i]/total_wb is dimensionless —
    # (ntheta-1) cancels, no scaling. For do_matrices we keep `pl` as a Vector because
    # it is referenced below; otherwise we fuse pl + bjspl → bj_integral in a single
    # pass, avoiding two Vector{ComplexF64}(ntheta) allocations.
    # Trapezoidal quadrature: boundary samples weighted by 0.5. jvtheta is zero at
    # i=1 and i=ntheta (loop above runs 2:ntheta-1) so the boundary terms contribute
    # nothing in practice, but writing the weights explicitly keeps the integration
    # self-correct if the boundary handling ever changes.
    pl_denom = (2 - sigma) * total_wb
    one_minus_sigma = 1 - sigma
    bj_integral = ComplexF64(0.0)
    if do_matrices
        pl = scr.pl
        @inbounds for i in 1:ntheta
            pl[i] = cis(-twopi * lnq * cum_wb_arr[i] * nrm / pl_denom)
        end
        @inbounds for i in 1:ntheta
            w = (i == 1 || i == ntheta) ? 0.5 : 1.0
            bj_integral += w * conj(jvtheta[i]) * (pl[i] + one_minus_sigma / (pl[i] + SINGULAR_EPS))
        end
    else
        pl = nothing
        @inbounds for i in 1:ntheta
            pli = cis(-twopi * lnq * cum_wb_arr[i] * nrm / pl_denom)
            w = (i == 1 || i == ntheta) ? 0.5 : 1.0
            bj_integral += w * conj(jvtheta[i]) * (pli + one_minus_sigma / (pli + SINGULAR_EPS))
        end
    end
    bj_integral *= nrm

    # |δJ|² (Fortran line 756) — division by 2 corrects quadratic form
    dJdJ_val = wbbar * abs(bj_integral)^2 / 2.0 / ro^2

    # Matrix path: bounce-average W vectors and form outer products (Fortran lines 759-793)
    wmats_lmda = nothing
    if do_matrices
        # Bounce-average W_μ and W_E vectors (Fortran lines 762-767).
        # Trapezoidal quadrature: boundary samples weighted by 0.5 (wmu_mt and wen_mt
        # are zero at i=1 and i=ntheta from the 2:ntheta-1 population loop above).
        wmu_ba = scr.wmu_ba
        wen_ba = scr.wen_ba
        fill!(wmu_ba, ComplexF64(0.0))
        fill!(wen_ba, ComplexF64(0.0))
        @inbounds for i in 1:ntheta
            w = (i == 1 || i == ntheta) ? 0.5 : 1.0
            factor = w * (pl[i] + one_minus_sigma / (pl[i] + SINGULAR_EPS))
            for mi in 1:mpert
                wmu_ba[mi] += conj(wmu_mt[mi, i]) * factor
                wen_ba[mi] += conj(wen_mt[mi, i]) * factor
            end
        end
        wmu_ba .*= nrm
        wen_ba .*= nrm

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
