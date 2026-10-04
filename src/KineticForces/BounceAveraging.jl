"""
    BounceAveraging

Bounce-averaging infrastructure for GAR NTV calculations.
Computes bounce-averaged frequencies (ωb, ωd) and perturbation action (|δJ|²)
as functions of pitch angle λ. For kinetic matrix methods, also computes
bounce-averaged W_μ, W_E vectors and their outer products.

Reference: [Logan et al., Phys. Plasmas 20, 122507 (2013)]
Ports Fortran torque.F90 lines 530-816.
"""

# Guard against a non-terminating spline cell walk; θ grids are O(10²) cells.
const MAX_SPLINE_CELLS = 100_000
# Stationary points closer than this in θ are the same root seen from both sides
# of a cell boundary.
const EXTREMUM_MERGE_TOL = 1e-12
# Cells a hinted cell search may step before falling back to bisection.
const HINT_STEP_BUDGET = 8

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
                        tspl, B_vpar, mfac, chi1, ro, dbob_m_f, divx_m_f,
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
- `tspl`: Periodic poloidal interpolant: tspl(θ) → [B, dB/dψ, dB/dθ, J, dJ/dψ]
- `B_vpar`: Periodic cubic of B(θ) used for v_par and the bounce-point roots
  (the Fortran `vspl` equivalent)
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
    tspl, B_vpar, mfac::Vector{Int}, chi1::Float64, ro::Float64,
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

    # B(θ) is decomposed once per surface into its cells and stationary points; every
    # λ then reuses it instead of rescanning [0,1].
    bf = _surface_b_field(B_vpar)
    bpts_buf = Float64[]
    # One resumable cell hint per monotone interval; λ ascends, so each interval's
    # crossing advances steadily and the next λ starts where the last one finished.
    hints = ones(Int, length(bf.theta) + 1)

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
            lmda, bo, sigma, B_vpar, theta_bmax, psi, ntheta, bf, bpts_buf, hints)

        # Bounce integrals over θ (Fortran lines 674-735)
        wbbar, wdbar, dJdJ_val, wmats_lmda = _bounce_integrate(
            tdt_pts, tdt_wts, lmda, lnq, sigma, n, q, bo,
            tspl, B_vpar, chi1, ro, mfac, dbob_m_f, divx_m_f, divxfac, wdfac,
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

Per-surface scratch for the bounce-averaging inner loops, allocated once in
`compute_bounce_data` and reused across every λ. Sizes are fixed for a flux surface
(`ntheta` sub-grid points, `mpert` Fourier modes). Buffers the loops populate only
partially are `fill!`-reset per λ.

## Fields
- `g_wb::Vector{Float64}`: length `ntheta` — bounce-action integrand samples
- `g_wd::Vector{Float64}`: length `ntheta` — drift integrand samples
- `cum_wb_arr::Vector{Float64}`: length `ntheta` — cumulative bounce-action integral
- `jvtheta::Vector{ComplexF64}`: length `ntheta` — action integrand
- `bj_samples::Vector{ComplexF64}`: length `ntheta` — action bounce-integral samples
- `wsamp::Vector{ComplexF64}`: length `ntheta` — per-mode W bounce-integral samples
- `wmu_mt::Matrix{ComplexF64}`: `mpert × ntheta` — W_μ per θ
- `wen_mt::Matrix{ComplexF64}`: `mpert × ntheta` — W_E per θ
- `expm::Vector{ComplexF64}`: length `mpert` — Fourier basis at a θ
- `pl::Vector{ComplexF64}`: length `ntheta` — bounce phase factor
- `wmu_ba::Vector{ComplexF64}`: length `mpert` — bounce-averaged W_μ
- `wen_ba::Vector{ComplexF64}`: length `mpert` — bounce-averaged W_E
- `wmats_lmda::Vector{ComplexF64}`: length `nqty_matrix(mpert)` — packed W outer products
- `tspl_f::Vector{Float64}`: length 5 — in-place tspl(θ) evaluation
- `int_w::Vector{Float64}`, `cumint_W::Matrix{Float64}`: precomputed exact-cubic
  quadrature weights on the fixed unit θ-grid (`∫ = int_w·y`, `cumulative = cumint_W·y`);
  shared read-only across surfaces, see `_quadrature_weights`
"""
struct BounceScratch
    g_wb::Vector{Float64}
    g_wd::Vector{Float64}
    cum_wb_arr::Vector{Float64}
    jvtheta::Vector{ComplexF64}
    bj_samples::Vector{ComplexF64}
    wsamp::Vector{ComplexF64}
    wmu_mt::Matrix{ComplexF64}
    wen_mt::Matrix{ComplexF64}
    expm::Vector{ComplexF64}
    pl::Vector{ComplexF64}
    wmu_ba::Vector{ComplexF64}
    wen_ba::Vector{ComplexF64}
    wmats_lmda::Vector{ComplexF64}
    tspl_f::Vector{Float64}
    int_w::Vector{Float64}
    cumint_W::Matrix{Float64}
end

function BounceScratch(ntheta::Int, mpert::Int)
    int_w, cumint_W = _quadrature_weights(ntheta)
    return BounceScratch(
        Vector{Float64}(undef, ntheta),
        Vector{Float64}(undef, ntheta),
        Vector{Float64}(undef, ntheta),
        Vector{ComplexF64}(undef, ntheta),
        Vector{ComplexF64}(undef, ntheta),
        Vector{ComplexF64}(undef, ntheta),
        Matrix{ComplexF64}(undef, mpert, ntheta),
        Matrix{ComplexF64}(undef, mpert, ntheta),
        Vector{ComplexF64}(undef, mpert),
        Vector{ComplexF64}(undef, ntheta),
        Vector{ComplexF64}(undef, mpert),
        Vector{ComplexF64}(undef, mpert),
        Vector{ComplexF64}(undef, nqty_matrix(mpert)),
        Vector{Float64}(undef, 5),
        int_w,
        cumint_W,
    )
end

# Exact-cubic θ-quadrature weights on the fixed unit grid, cached by ntheta.
const _QUAD_WEIGHTS = Dict{Int,Tuple{Vector{Float64},Matrix{Float64}}}()
const _QUAD_WEIGHTS_LOCK = ReentrantLock()

"""
    _quadrature_weights(ntheta) → (int_w, cumint_W)

Exact integral of the `CubicFit`-endpoint spline on the fixed grid `range(0,1,ntheta)`
is a constant linear functional of the node samples, so `∫ = int_w·y` and the cumulative
integral is `cumint_W·y`. The weights are obtained once per `ntheta` by evaluating the
public `integrate` / `cumulative_integrate!` on the unit basis vectors — bit-faithful to
fitting and integrating each sample vector directly, but reducing the per-λ hot loop to a
`dot`/`mul!`. Cached (build guarded by a lock); the returned arrays are read-only.
"""
function _quadrature_weights(ntheta::Int)
    lock(_QUAD_WEIGHTS_LOCK) do
        get!(_QUAD_WEIGHTS, ntheta) do
            xs = collect(range(0.0, 1.0, length=ntheta))
            int_w = zeros(Float64, ntheta)
            cumint_W = zeros(Float64, ntheta, ntheta)
            ej = zeros(Float64, ntheta)
            cbuf = zeros(Float64, ntheta)
            for j in 1:ntheta
                ej[j] = 1.0
                itp = cubic_interp(xs, ej; bc=CubicFit())
                int_w[j] = FastInterpolations.integrate(itp)
                FastInterpolations.cumulative_integrate!(cbuf, itp)
                @views cumint_W[:, j] .= cbuf
                ej[j] = 0.0
            end
            (int_w, cumint_W)
        end
    end
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
Parallel-velocity factor `v_par = 1 − (λ/bo)·B(θ)` from the periodic cubic of B
(`B_vpar`, built where the surface interpolants are constructed), keeping v_par
consistent with the bounce-point roots as in Fortran's `vspl`.
"""
@inline _vpar_from_spline(B_vpar, lmda::Float64, bo::Float64, θ::Float64) =
    1.0 - (lmda / bo) * B_vpar(mod(θ, 1.0))


"""
    SurfaceBField

The periodic cubic B(θ) of one flux surface, decomposed once and reused for every λ.
Holds the per-cell polynomial coefficients, B at every knot, and the stationary points
of B. Consecutive stationary points bound intervals on which B is monotone, so each
holds at most one bounce point; within such an interval the cached knot values locate
the cell by bisection and the cell's cubic is then solved in closed form.

## Fields
- `knot::Vector{Float64}`: cell boundaries, ascending, `knot[1] = 0`, `knot[end] = 1`
- `bknot::Vector{Float64}`: B at each knot
- `poly::Vector{NTuple{4,Float64}}`: per-cell `(d, c, b, a)` of `S(u) = d + cu + bu² + au³`,
  in the cell-local coordinate `u = θ − knot[i]`
- `theta::Vector{Float64}`: stationary points of B, ascending, in [0,1)
- `bval::Vector{Float64}`: B at each stationary point
"""
struct SurfaceBField
    knot::Vector{Float64}
    bknot::Vector{Float64}
    poly::Vector{NTuple{4,Float64}}
    theta::Vector{Float64}
    bval::Vector{Float64}
end

"""
Decompose the cubic `B_vpar` by walking its cells once, recording each cell's
polynomial and endpoint value and solving the quadratic dS/dθ = 0 on each to get the
exact stationary points. Uses only the public `coeffs`/`CellPoly` interface, so it
holds for whatever θ grid the surface interpolant was built on.
"""
function _surface_b_field(B_vpar)
    knot = Float64[0.0]
    bknot = Float64[]
    poly = NTuple{4,Float64}[]
    theta = Float64[]

    x = 0.0
    while true
        cell = coeffs(B_vpar, x)
        h = cell.xR - cell.xL
        d, c, b, a = cell.p
        push!(poly, (d, c, b, a))
        push!(bknot, d)                      # S(0) = B at the cell's left knot
        push!(knot, cell.xR)

        # S'(u) = c + 2b·u + 3a·u², u ∈ [0, h)
        qa, qb, qc = 3a, 2b, c
        if abs(qa) <= eps(Float64) * max(abs(qb), abs(qc), 1.0)
            if qb != 0
                u = -qc / qb
                (0.0 <= u < h) && push!(theta, cell.xL + u)
            end
        else
            disc = qb^2 - 4 * qa * qc
            if disc >= 0
                sq = sqrt(disc)
                for u in ((-qb - sq) / (2qa), (-qb + sq) / (2qa))
                    (0.0 <= u < h) && push!(theta, cell.xL + u)
                end
            end
        end

        cell.xR >= 1.0 && break
        x = cell.xR
        length(poly) > MAX_SPLINE_CELLS && error("ERROR: _surface_b_field - cell walk did not reach θ=1")
    end
    push!(bknot, evalpoly(knot[end] - knot[end-1], poly[end]))   # B at θ = 1

    sort!(theta)
    # A stationary point on a cell boundary is reported by both neighbouring cells.
    if length(theta) > 1
        keep = 1
        for i in 2:length(theta)
            if theta[i] - theta[keep] > EXTREMUM_MERGE_TOL
                keep += 1
                theta[keep] = theta[i]
            end
        end
        resize!(theta, keep)
    end

    bval = [_b_at(knot, poly, t) for t in theta]
    return SurfaceBField(knot, bknot, poly, theta, bval)
end

"""Cell index holding θ ∈ [0,1], from the ascending knot vector."""
@inline _cell_index(knot::Vector{Float64}, θ::Float64) =
    clamp(searchsortedlast(knot, θ), 1, length(knot) - 1)

"""Evaluate B at θ from the cached cell polynomials."""
@inline function _b_at(knot::Vector{Float64}, poly::Vector{NTuple{4,Float64}}, θ::Float64)
    i = _cell_index(knot, θ)
    return evalpoly(θ - knot[i], poly[i])
end

"""
Real roots of `a·u³ + b·u² + c·u + d = 0`, returned as `(count, r1, r2, r3)` with
unused slots `NaN`. Degenerate leading coefficients fall through to the quadratic and
linear cases; three distinct real roots use the trigonometric form, which stays well
conditioned where Cardano's radicals cancel.
"""
function _real_cubic_roots(a::Float64, b::Float64, c::Float64, d::Float64)
    scale = max(abs(b), abs(c), abs(d), 1.0)
    if abs(a) <= eps(Float64) * scale
        if abs(b) <= eps(Float64) * max(abs(c), abs(d), 1.0)
            c == 0 && return (0, NaN, NaN, NaN)
            return (1, -d / c, NaN, NaN)
        end
        disc = c * c - 4 * b * d
        disc < 0 && return (0, NaN, NaN, NaN)
        sq = sqrt(disc)
        # Cancellation-free quadratic roots (Numerical Recipes §5.6).
        q = c == 0 ? -0.5 * sq : -0.5 * (c + copysign(sq, c))
        r1 = q / b
        r2 = q == 0 ? r1 : d / q
        return (2, r1, r2, NaN)
    end

    B, C, D = b / a, c / a, d / a
    shift = B / 3
    p = C - B * B / 3
    q = 2 * B^3 / 27 - B * C / 3 + D
    disc = (q / 2)^2 + (p / 3)^3

    if p == 0 && q == 0
        return (1, -shift, NaN, NaN)               # triple root
    elseif abs(disc) <= 8 * eps(Float64) * max((q / 2)^2, abs(p / 3)^3)
        # Repeated root. The discriminant cancels to ~0 here, so the branches below
        # would lose it: disc > 0 by a rounding step reports only the simple root.
        t2 = -3q / (2p)
        return (3, 3q / p - shift, t2 - shift, t2 - shift)
    elseif disc > 0
        s = sqrt(disc)
        t = cbrt(-q / 2 + s) + cbrt(-q / 2 - s)
        return (1, t - shift, NaN, NaN)
    else
        # Three real roots: t_k = 2r·cos(φ − 2πk/3), r = √(−p/3), φ = acos(−q/2r³)/3.
        r = sqrt(-p / 3)
        φ = acos(clamp(-q / (2 * r^3), -1.0, 1.0)) / 3
        return (3, 2r * cos(φ) - shift, 2r * cos(φ - 2π / 3) - shift, 2r * cos(φ - 4π / 3) - shift)
    end
end

"""
Solve `B(θ) = btarget` inside cell `ic`, restricted to `θ ∈ [θlo, θhi]`. The cubic is
solved in closed form and polished with one Newton step, which recovers the digits the
closed form loses when two of its roots are nearly coincident. Returns `NaN` if no root
lies in the restricted range.
"""
function _cell_level_root(bf::SurfaceBField, ic::Int, btarget::Float64, θlo::Float64, θhi::Float64)
    d, c, b, a = bf.poly[ic]
    x0 = bf.knot[ic]
    ulo, uhi = θlo - x0, θhi - x0
    # Admit roots a rounding step outside the cell: θlo/θhi are stationary points and
    # cell edges, and the root can sit exactly on one.
    pad = 8 * eps(Float64) * max(abs(ulo), abs(uhi), bf.knot[ic+1] - x0)

    n, r1, r2, r3 = _real_cubic_roots(a, b, c, d - btarget)
    best, bestres = NaN, Inf
    for k in 1:n
        u = k == 1 ? r1 : (k == 2 ? r2 : r3)
        (isfinite(u) && ulo - pad <= u <= uhi + pad) || continue
        u = clamp(u, ulo, uhi)
        # One Newton step on S(u) − btarget, skipped at a stationary point.
        deriv = c + u * (2b + u * 3a)
        if deriv != 0
            un = u - (evalpoly(u, (d - btarget, c, b, a))) / deriv
            (ulo - pad <= un <= uhi + pad) && (u = clamp(un, ulo, uhi))
        end
        res = abs(evalpoly(u, (d - btarget, c, b, a)))
        if res < bestres
            best, bestres = u, res
        end
    end
    return isnan(best) ? NaN : x0 + best
end

"""
Where `btarget` sits relative to cell `ic` of a monotone span: `-1` before it, `0`
inside, `+1` past it. `s` carries the span's direction so one comparison serves both.
The span's first and last cells are entered part-way, at the stationary points bounding
it, so their outer edge value comes from `ba`/`bb` rather than from a knot.
"""
@inline function _cell_position(
    bf::SurfaceBField, ic::Int, ia::Int, ib::Int,
    ba::Float64, bb::Float64, btarget::Float64, s::Float64
)
    left = ic == ia ? ba : bf.bknot[ic]
    right = ic == ib ? bb : bf.bknot[ic+1]
    s * btarget < s * left && return -1
    s * btarget > s * right && return 1
    return 0
end

"""
Cell holding the `B = btarget` crossing on a monotone span running from cell `ia` to
cell `ib`.

λ advances monotonically through `compute_bounce_data`, so `btarget = bo/λ` falls
monotonically and each interval's crossing walks steadily along the cells in one
direction. Resuming from the previous λ's cell therefore costs a step or two, the same
hint idiom FastInterpolations uses for its own searches. Bisection stays as the fallback
for the first λ of a surface and for the sweep near a stationary point, where B is flat
and the crossing can cross many cells between consecutive λ.
"""
function _locate_cell(
    bf::SurfaceBField, ia::Int, ib::Int, ba::Float64, bb::Float64,
    btarget::Float64, hint::Int
)
    s = bb >= ba ? 1.0 : -1.0
    ic = clamp(hint, ia, ib)
    pos = _cell_position(bf, ic, ia, ib, ba, bb, btarget, s)
    steps = 0
    while pos != 0 && steps < HINT_STEP_BUDGET
        ic += pos
        (ia <= ic <= ib) || return _bisect_cell(bf, ia, ib, btarget, s)
        pos = _cell_position(bf, ic, ia, ib, ba, bb, btarget, s)
        steps += 1
    end
    return pos == 0 ? ic : _bisect_cell(bf, ia, ib, btarget, s)
end

"""Bisection on the span's monotone knot values, for when the hint does not pay off."""
function _bisect_cell(bf::SurfaceBField, ia::Int, ib::Int, btarget::Float64, s::Float64)
    ia >= ib && return ia
    s * bf.bknot[ia+1] > s * btarget && return ia
    s * bf.bknot[ib] <= s * btarget && return ib
    lo, hi = ia + 1, ib
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        s * bf.bknot[mid] <= s * btarget ? (lo = mid) : (hi = mid)
    end
    return lo
end

"""
Solve the single `B = btarget` crossing on `[θa, θb]`, a span on which B is monotone and
which does not cross the θ = 0/1 seam. Returns `(θ, cell)`; the cell feeds back as the
next λ's hint. `θ` is `NaN` when no root lies in the span.
"""
function _monotone_segment_root(
    bf::SurfaceBField, θa::Float64, θb::Float64,
    btarget::Float64, hint::Int
)
    ia = _cell_index(bf.knot, θa)
    ib = max(_cell_index(bf.knot, θb), ia)
    ba = _b_at(bf.knot, bf.poly, θa)
    bb = _b_at(bf.knot, bf.poly, θb)

    ic = ia == ib ? ia : _locate_cell(bf, ia, ib, ba, bb, btarget, hint)
    root = _cell_level_root(bf, ic, btarget, max(θa, bf.knot[ic]), min(θb, bf.knot[ic+1]))
    # A target landing within rounding of a knot can select the neighbouring cell.
    isnan(root) && ic > ia && (root = _cell_level_root(bf, ic - 1, btarget, max(θa, bf.knot[ic-1]), min(θb, bf.knot[ic])))
    isnan(root) && ic < ib && (root = _cell_level_root(bf, ic + 1, btarget, max(θa, bf.knot[ic+1]), min(θb, bf.knot[ic+2])))
    return root, ic
end

"""
Bounce points of `v_par(θ) = 1 − (λ/bo)·B(θ)` for a trapped particle. `B = bo/λ` at a
bounce point, so a monotone interval between stationary points of B contains one iff
`bo/λ` lies strictly between its endpoint B values — a scalar test against cached
values. The crossing is then solved directly from the cell's cubic coefficients: no
iteration, no spline evaluation, and no blind search, since `hints` carries each
interval's cell over from the previous λ.

Returns roots sorted descending, the order the deepest-well and marginally-trapped logic
downstream assumes; fewer than two roots signals a degenerate λ and sends the caller to
the fallback. `hints` is sized `length(bf.theta) + 1`, the extra slot being the far half
of the interval that wraps through the seam.
"""
function _bounce_points_at_lambda!(
    bpts::Vector{Float64}, hints::Vector{Int},
    bf::SurfaceBField, lmda::Float64, bo::Float64
)
    empty!(bpts)
    k = length(bf.theta)
    k < 2 && return bpts

    btarget = bo / lmda
    for i in 1:k
        j = i == k ? 1 : i + 1
        (bf.bval[i] - btarget) * (bf.bval[j] - btarget) < 0 || continue
        θa, θb = bf.theta[i], bf.theta[j]

        root = NaN
        if i < k
            root, hints[i] = _monotone_segment_root(bf, θa, θb, btarget, hints[i])
        else
            # The last interval wraps through θ = 0/1. Split it at the seam, where B is
            # continuous for the periodic fit, and solve whichever half spans btarget.
            if (bf.bval[i] - btarget) * (bf.bknot[end] - btarget) <= 0
                root, hints[k] = _monotone_segment_root(bf, θa, 1.0, btarget, hints[k])
            else
                root, hints[k+1] = _monotone_segment_root(bf, 0.0, θb, btarget, hints[k+1])
            end
        end

        isnan(root) && return empty!(bpts)
        push!(bpts, mod(root, 1.0))
    end

    sort!(bpts; rev=true)
    return bpts
end


"""
Find bounce points for trapped/passing particles and build θ sub-grid.
Returns (t1, t2, theta_points, theta_weights).
"""
function _find_bounce_points_and_grid(
    lmda::Float64, bo::Float64, sigma::Int,
    B_vpar, theta_bmax::Float64, psi::Float64,
    ntheta::Int, bf::SurfaceBField, bpts_buf::Vector{Float64}, hints::Vector{Int}
)
    if sigma == 0  # trapped
        # Bounce points: all roots of v_par(θ) = 1 − (λ/bo)·B_vpar(θ) in (0,1),
        # sorted descending — the same order as Fortran spline_roots, which the
        # marginally-trapped and deepest-well wrap logic below assume.
        bpts = _bounce_points_at_lambda!(bpts_buf, hints, bf, lmda, bo)

        if length(bpts) < 2
            # Degenerate λ: v_par is tangent to zero at an extremum of B (so no
            # interval brackets strictly), or B has fewer than two stationary
            # points. Rare — the λ grid excludes both trapped-passing endpoints —
            # so fall back to the adaptive whole-interval scan and keep its behaviour.
            vpar_fn = θ -> _vpar_from_spline(B_vpar, lmda, bo, θ)
            bpts = sort!(Roots.find_zeros(vpar_fn, 0.0, 1.0); rev=true)
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
            t1, t2 = _find_deepest_well(bpts, B_vpar, lmda, bo)
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
Find the deepest potential well (largest midpoint v_par) among bounce-point pairs,
handling pairs that wrap through θ = 0/1.
"""
function _find_deepest_well(bpts::Vector{Float64}, B_vpar, lmda::Float64, bo::Float64)
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
        vpar_mid = _vpar_from_spline(B_vpar, lmda, bo, θmid)
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
    tspl, B_vpar, chi1::Float64, ro::Float64,
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

        # v_par from the periodic cubic (consistent with the bounce points);
        # the periodic tspl B_val remains the numerator field in the integrands.
        vpar = 1.0 - (lmda / bo) * B_vpar(θmod)

        if vpar <= 0
            # Negative v_par near a bounce point: same fill rules as the Fortran bounce loop.
            if i < ntheta ÷ 2
                # Before midpoint: restart — zero everything up to this sample.
                fill!(view(g_wb, 1:i), 0.0)
                fill!(view(g_wd, 1:i), 0.0)
                fill!(view(jvtheta, 1:i), ComplexF64(0.0))
                continue
            else
                # After midpoint: hold the previous sample to the end of the grid.
                # The wd slot is deliberately held from the wb integrand (g_wb),
                # reproducing the Fortran behavior for parity.
                fill!(view(g_wb, i:ntheta), g_wb[i-1])
                fill!(view(g_wd, i:ntheta), g_wb[i-1])
                fill!(view(jvtheta, i:ntheta), jvtheta[i-1])
                break
            end
        end

        sqrt_vpar = sqrt(vpar)

        # Bounce integrands
        g_wb[i] = dt * jac * B_val / sqrt_vpar
        g_wd[i] = dt * jac * dBdpsi * (1.0 - 1.5 * lmda * B_val / bo) / sqrt_vpar +
                  dt * djdpsi * B_val * sqrt_vpar

        # Fourier modes at this θ
        @inbounds for mi in 1:mpert
            expm[mi] = cis(twopi * mfac[mi] * θ)
        end
        dbob = ComplexF64(0.0)
        divx = ComplexF64(0.0)
        @inbounds for mi in 1:mpert
            dbob += dbob_m_f[mi] * expm[mi]
            divx += divx_m_f[mi] * expm[mi]
        end
        divx *= divxfac

        # Action integrand
        phase = cis(-twopi * n * q * (θ - theta0))
        jvtheta[i] = dt * jac * B_val *
            (divx * sqrt_vpar + dbob * (1.0 - 1.5 * lmda * B_val / bo) / sqrt_vpar) *
            phase

        # W vectors for matrix path
        if do_matrices
            wmu_pre = dt * (lmda / bo)
            wen_pre = dt
            @inbounds for mi in 1:mpert
                wmu_mt[mi, i] = wmu_pre * expm[mi] / sqrt_vpar * phase / (2 * chi1)
                wen_mt[mi, i] = wen_pre * expm[mi] / (B_val * sqrt_vpar) * phase / (2 * chi1)
            end
        end

        # Smooth backfill for points zeroed before a restart. Exact equality with
        # 0.0 is safe: the restart branch set these entries with fill!(…, 0.0).
        if i >= 3 && g_wb[i-1] == 0.0
            fill!(view(g_wb, 3:i-1), g_wb[i])
            fill!(view(g_wd, 3:i-1), g_wd[i])
            fill!(view(jvtheta, 2:i-1), jvtheta[i])
        end
    end

    # Total bounce integrals: the samples live on the fixed unit x-grid (the tdt weights
    # carry dθ/dx), so the exact integral of the endpoint-fit cubic is a fixed linear
    # combination of them (precomputed weights, see `_quadrature_weights`). With the
    # 1/√v_par endpoint singularities the quadrature scheme is a leading-order effect, so
    # this reproduces exact-cubic integration (not a trapezoid sum).
    fsi_wb = scr.cum_wb_arr  # per-surface scratch; fully overwritten
    mul!(fsi_wb, scr.cumint_W, g_wb)
    total_wb = fsi_wb[ntheta]
    total_wd = dot(scr.int_w, g_wd)

    if total_wb ≈ 0.0
        # Degenerate case — return zeros
        return 0.0, 0.0, 0.0, nothing
    end

    # Bounce-averaged frequencies. wbbar already carries one factor of ro that its own
    # normalization bhat = sqrt(2T/m)/ro cancels; reusing it inside wdbar imports that ro
    # a third time while dhat = (T/q)/(bo·ro²) removes only the two written explicitly, so
    # the drift prefactor takes ro, not ro². (Otherwise ω_D = wdbar·dhat carries a surplus
    # length: 4π·(I₂/I₁)·(T/q) is already V/Wb = 1/s, so the extra ro leaves m/s.)
    wbbar = ro * twopi / ((2 - sigma) * total_wb)
    wdbar = ro * bo * wdfac * wbbar * 2 * (2 - sigma) * total_wd

    # Phase factor pl_i = exp(-2πi·lnq·fsi_wb(θ_i)/((2-σ)·total_wb)), using the
    # cumulative spline integral of the bounce action.
    pl_denom = (2 - sigma) * total_wb
    one_minus_sigma = 1 - sigma
    pl = scr.pl  # per-surface scratch; fully overwritten
    @inbounds for i in 1:ntheta
        pl[i] = cis(-twopi * lnq * fsi_wb[i] / pl_denom)
    end

    # Action bounce integral (exact-cubic quadrature, as for the totals above).
    bj_samples = scr.bj_samples
    @inbounds for i in 1:ntheta
        bj_samples[i] = conj(jvtheta[i]) * (pl[i] + one_minus_sigma / pl[i])
    end
    bj_integral = dot(scr.int_w, bj_samples)

    # |δJ|² — division by 2 corrects the quadratic form
    dJdJ_val = wbbar * abs(bj_integral)^2 / 2.0 / ro^2

    # Matrix path: bounce-average W vectors and form outer products
    wmats_lmda = nothing
    if do_matrices
        # Per mode, exact-cubic quadrature of conj(W_m(θ))·(pl + (1-σ)/pl),
        # matching the action bounce integral above.
        wmu_ba = scr.wmu_ba
        wen_ba = scr.wen_ba
        wsamp = scr.wsamp
        @inbounds for mi in 1:mpert
            for i in 1:ntheta
                wsamp[i] = conj(wmu_mt[mi, i]) * (pl[i] + one_minus_sigma / pl[i])
            end
            wmu_ba[mi] = dot(scr.int_w, wsamp)
            for i in 1:ntheta
                wsamp[i] = conj(wen_mt[mi, i]) * (pl[i] + one_minus_sigma / pl[i])
            end
            wen_ba[mi] = dot(scr.int_w, wsamp)
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
