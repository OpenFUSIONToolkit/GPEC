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
    # For matrix path (nothing if scalar-only):
    wmats_vs_lambda::Union{Nothing, Array{ComplexF64,4}}  # (mpert, mpert, 6, nlmda)
end


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
needed by the pitch angle ODE integrator.

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
    wmats_arr = do_matrices ? zeros(ComplexF64, mpert, mpert, 6, nlmda) : nothing

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
            lmdatpb, lmdamax, psi, ntheta)

        # Bounce integrals over θ (Fortran lines 674-735)
        wbbar, wdbar, dJdJ_val, wmats_lmda = _bounce_integrate(
            tdt_pts, tdt_wts, lmda, lnq, sigma, n, q, bo,
            tspl, chi1, ro, mfac, dbob_m_f, divx_m_f, divxfac, wdfac,
            do_matrices, mpert, smat, tmat, xmat, ymat, zmat)

        # Physical frequencies (Fortran lines 744-745)
        wb_arr[ilmda] = wbbar * bhat
        wd_arr[ilmda] = wdbar * dhat
        dJdJ_arr[ilmda] = dJdJ_val

        if do_matrices && !isnothing(wmats_lmda)
            wmats_arr[:, :, :, ilmda] .= wmats_lmda
        end
    end

    return BounceData(nlmda, lambda, dlambda, sigma_arr,
                      wb_arr, wd_arr, dJdJ_arr, wmats_arr)
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
    ntheta::Int
)
    if sigma == 0  # trapped
        # Build v_par(θ) = 1 - (λ/bo)*B(θ) and find roots
        # Use a dense θ grid to find zero crossings
        nfine = 256
        theta_fine = range(0.0, 1.0, length=nfine+1)
        vpar_fine = [1.0 - (lmda / bo) * tspl(mod(θ, 1.0))[1] for θ in theta_fine]

        # Find zero crossings
        bpts = Float64[]
        for i in 1:nfine
            if vpar_fine[i] * vpar_fine[i+1] < 0
                # Bisect for better accuracy
                θ_root = _bisect_vpar(tspl, lmda, bo, theta_fine[i], theta_fine[i+1])
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
            t1, t2 = _find_deepest_well(bpts, tspl, lmda, bo)
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
Bisect to find θ where v_par = 1 - (λ/bo)*B(θ) = 0.
"""
function _bisect_vpar(tspl, lmda::Float64, bo::Float64, θa::Float64, θb::Float64; tol=1e-12, maxiter=50)
    va = 1.0 - (lmda / bo) * tspl(mod(θa, 1.0))[1]
    for _ in 1:maxiter
        θm = 0.5 * (θa + θb)
        vm = 1.0 - (lmda / bo) * tspl(mod(θm, 1.0))[1]
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
function _find_deepest_well(bpts::Vector{Float64}, tspl, lmda::Float64, bo::Float64)
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
        vpar_mid = 1.0 - (lmda / bo) * tspl(mod(θmid, 1.0))[1]
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
    smat, tmat, xmat, ymat, zmat
)
    ntheta = length(tdt_pts)
    theta0 = tdt_pts[1]

    # Cumulative bounce integrals
    cum_wb = 0.0
    cum_wd = 0.0

    # Arrays for cumulative integrals (for phase factor computation)
    cum_wb_arr = zeros(Float64, ntheta)
    cum_wd_arr = zeros(Float64, ntheta)

    # Action integrand
    jvtheta = zeros(ComplexF64, ntheta)

    # W vectors for matrix path
    wmu_mt = do_matrices ? zeros(ComplexF64, mpert, ntheta) : nothing
    wen_mt = do_matrices ? zeros(ComplexF64, mpert, ntheta) : nothing

    for i in 2:ntheta-1  # Edge weights are 0 from powspace
        θ = tdt_pts[i]
        dt = tdt_wts[i]
        θmod = mod(θ, 1.0)

        tspl_f = tspl(θmod)
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
        cum_wb_arr[i] = cum_wb
        cum_wd_arr[i] = cum_wd

        # Fourier modes at this θ (Fortran lines 702-708)
        expm = [exp(im * twopi * m * θ) for m in mfac]
        dbob = sum(dbob_m_f .* expm)
        divx = sum(divx_m_f .* expm) * divxfac

        # Action integrand (Fortran line 706-708)
        phase = exp(-twopi * im * n * q * (θ - theta0))
        jvtheta[i] = dt * jac * B_val *
            (divx * sqrt_vpar + dbob * (1.0 - 1.5 * lmda * B_val / bo) / sqrt_vpar) *
            phase

        # W vectors for matrix path (Fortran lines 722-727)
        if do_matrices
            wmu_mt[:, i] .= dt .* (lmda / bo) .* expm ./ sqrt_vpar .*
                phase ./ (2 * chi1)
            wen_mt[:, i] .= dt .* expm ./ (B_val * sqrt_vpar) .*
                phase ./ (2 * chi1)
        end
    end

    # Total bounce integrals
    total_wb = cum_wb
    total_wd = cum_wd

    if total_wb ≈ 0.0
        # Degenerate case — return zeros
        return 0.0, 0.0, 0.0, nothing
    end

    # Bounce-averaged frequencies (Fortran lines 740-741)
    wbbar = ro * twopi / ((2 - sigma) * total_wb)
    wdbar = ro^2 * bo * wdfac * wbbar * 2 * (2 - sigma) * total_wd

    # Phase factor (Fortran line 750)
    # Using wb-based phase (electric precession dominates)
    pl = [exp(-twopi * im * lnq * cum_wb_arr[i] / ((2 - sigma) * total_wb)) for i in 1:ntheta]

    # Bounce-averaged action (Fortran line 752)
    bjspl = [conj(jvtheta[i]) * (pl[i] + (1 - sigma) / (pl[i] + 1e-30)) for i in 1:ntheta]

    # Cumulative trapezoidal integration of bjspl
    bj_integral = ComplexF64(0.0)
    for i in 2:ntheta
        bj_integral += bjspl[i]  # weights already in jvtheta via dt
    end

    # |δJ|² (Fortran line 756) — division by 2 corrects quadratic form
    dJdJ_val = wbbar * abs(bj_integral)^2 / 2.0 / ro^2

    # Matrix path: bounce-average W vectors and form outer products (Fortran lines 759-793)
    wmats_lmda = nothing
    if do_matrices
        # Bounce-average W_μ and W_E vectors
        wmu_ba = zeros(ComplexF64, mpert)
        wen_ba = zeros(ComplexF64, mpert)
        for i in 2:ntheta-1
            factor = conj(pl[i]) + (1 - sigma) / (pl[i] + 1e-30)
            # Note: Fortran uses conj(wmu_mt) * (pl + (1-σ)/pl), then integrates
            # The conjugate on W is because we want W† later
            wmu_ba .+= conj.(wmu_mt[:, i]) .* factor
            wen_ba .+= conj.(wen_mt[:, i]) .* factor
        end

        # Reshape as 1×mpert for matrix multiply (Fortran lines 771-772)
        wmmt = reshape(wmu_ba, 1, mpert)
        wemt = reshape(wen_ba, 1, mpert)

        # Build W_X, W_Y, W_Z via geometric matrices (Fortran lines 773-775)
        wxmt = wmmt * xmat
        wymt = wmmt * (3 * smat + ymat) - 2.0 * wemt * smat
        wzmt = wmmt * (3 * tmat + zmat) - 2.0 * wemt * tmat

        # Conjugate transpose for outer products
        wxmc = conj.(transpose(wxmt))
        wymc = conj.(transpose(wymt))
        wzmc = conj.(transpose(wzmt))

        # 6 outer products (Fortran lines 779-784)
        wmats_lmda = zeros(ComplexF64, mpert, mpert, 6)
        op_A = wzmc * wzmt  # A: W_Z† W_Z
        op_B = wzmc * wxmt  # B: W_Z† W_X
        op_C = wzmc * wymt  # C: W_Z† W_Y
        op_D = wxmc * wxmt  # D: W_X† W_X
        op_E = wxmc * wymt  # E: W_X† W_Y
        op_H = wymc * wymt  # H: W_Y† W_Y

        # Scale by wbbar/ro² (Fortran line 789)
        scale = wbbar / ro^2
        wmats_lmda[:, :, 1] .= op_A .* scale
        wmats_lmda[:, :, 2] .= op_B .* scale
        wmats_lmda[:, :, 3] .= op_C .* scale
        wmats_lmda[:, :, 4] .= op_D .* scale
        wmats_lmda[:, :, 5] .= op_E .* scale
        wmats_lmda[:, :, 6] .= op_H .* scale
    end

    return wbbar, wdbar, dJdJ_val, wmats_lmda
end
