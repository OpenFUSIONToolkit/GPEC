"""
    IdealMatrices

Ideal-MHD stability matrix ψ-splines assembled by [`build_matrix_splines`](@ref). Each field flattens a
`numpert_total × numpert_total` matrix to `numpert_total^2` complex series (`J_spline` to `2·mpert−1`),
following the appendix of Glasser Phys. Plasmas 2016 112506.

Suffixes: `_prim` is a primitive (pre-Schur-complement) form, kept because the kinetic reduction
needs the unreduced matrix; `_lower` is a Cholesky factor rather than the matrix itself; `_gal` is
the Galerkin solver's variant of the same matrix.

## Fields

  - `A_spline`, `B_spline`, `C_spline` - the ideal A, B, C. In a kinetic run these are NOT what the
    ODE integrates; see [`FKGMatrices`](@ref).
  - `D_spline_prim`, `E_spline_prim` - pre-Schur-reduction geometric forms
    (D = χ₁·(g23 + q·g33·m/n); E = (-χ₁/n)·(q'·χ₁·g33 - 2π·i·χ₁·g31·singfac + jθ·I)). Downstream
    kinetic FKG Schur complements consume these primitive forms.
  - `H_spline` - primitive Ḡ before the Schur complement (Ḡ = H − C†A⁻¹C), also consumed by the
    kinetic reduction.
  - `F_spline_lower` - reduced F̄ as its lower-triangular Cholesky factor L, F̄ = L·Lᴴ. The ideal
    Euler-Lagrange kernel only ever solves against F̄, so the factor is stored instead of F̄ itself.
  - `F_spline_prim` - primitive F before the Schur complement (for kinetic).
  - `F_spline_gal` - independently interpolated reduced Hermitian F̄, retained for coefficient diagnostics.
  - `K_spline`, `G_spline` - reduced K̄ and Ḡ of the Euler-Lagrange system
    d/dψ(F̄·ξ′ + K̄·ξ) = K̄†·ξ′ + Ḡ·ξ (Eqs A5-A7).
  - `J_spline` - Jacobian Fourier band (2·mpert−1 conjugate-symmetric coefficients per surface), used
    to assemble the power-normalization matrix N in Free.jl.
"""
@kwdef struct IdealMatrices{S<:CubicSeriesInterpolant}
    A_spline::S
    B_spline::S
    C_spline::S
    D_spline_prim::S
    E_spline_prim::S
    H_spline::S
    F_spline_lower::S
    F_spline_prim::S
    F_spline_gal::S
    K_spline::S
    G_spline::S
    J_spline::S
end

"""
    FKGMatrices

Kinetic-MHD matrix ψ-splines assembled by [`build_kinetic_matrix_splines`](@ref). Present on a
[`MatrixSplines`](@ref) only for kinetic runs; its presence means the solution obeys the FKG ODE
rather than the ideal Euler-Lagrange relation.

The FKG system is non-Hermitian, so the adjoint side of each reduced block is a distinct matrix
rather than a transpose of the forward one. The `_adj` suffix marks those adjoint-side counterparts
(Fortran's `aat` arrays); they must be evaluated, never derived from their unsuffixed partner.

## Fields

  - `A_spline`, `B_spline`, `C_spline` - the kinetic-modified A, B, C consumed by `sing_der!`. Unlike their
    ideal counterparts, A here is non-Hermitian and requires an LU rather than a Cholesky.
  - `F0_spline` - reduced F̄ of the kinetic system, F₀ = F_prim − D†A_kin⁻¹D.
  - `P_spline`, `P_spline_adj` - P = (iD)†A_kin⁻¹B_k and its adjoint-side counterpart.
  - `Kk_spline`, `Kk_spline_adj` - kinetic K̄ = E − (iD)†A_kin⁻¹C_kin and its adjoint-side counterpart.
  - `R1_spline`, `R2_spline`, `R3_spline` - blocks built from the D- and E-perturbation kinetic
    components of the kinetic energy and torque matrices, which have no ideal counterpart. R3 is the
    adjoint-side partner of R2.
  - `G_spline_adj` - adjoint-side Ḡ, Ḡ_adj = H_kin − C_adj†A_kin⁻¹C_kin.

All nine reduced blocks are pre-computed per ψ by the FKG Schur complement
(Logan 2015 Appendix C, Eqs C.1-C.11) so `sing_der!` can assemble the ODE right-hand side with
explicit (m−nq) factors instead of 1/(m−nq).
"""
@kwdef struct FKGMatrices{S<:CubicSeriesInterpolant}
    A_spline::S
    B_spline::S
    C_spline::S
    F0_spline::S
    P_spline::S
    P_spline_adj::S
    Kk_spline::S
    Kk_spline_adj::S
    R1_spline::S
    R2_spline::S
    R3_spline::S
    G_spline_adj::S
end

"""
    MatrixSplines

Fourier-fitted stability matrices for one run. `kinetic === nothing` marks an ideal run; otherwise
the kinetic matrices are authoritative for the ODE and the ideal ones remain available for output
and for the Galerkin/vacuum paths that are defined only in the ideal basis.

## Fields

  - `ideal::IdealMatrices` - always present.
  - `kinetic::Union{Nothing,FKGMatrices}` - present iff the solve is kinetic (`ctrl.kinetic !== nothing`).
  - `_hint::Base.RefValue{Int}` - shared bracket-search hint for sequential (single-threaded)
    evaluation. Not thread-safe: parallel paths must pass their own hint.
"""
@kwdef struct MatrixSplines{S<:CubicSeriesInterpolant}
    ideal::IdealMatrices{S}
    kinetic::Union{Nothing,FKGMatrices{S}} = nothing
    _hint::Base.RefValue{Int} = Ref(1)
end

# Helper to create an empty complex series interpolant for default initialization
function _empty_series_interp_complex(n_series::Int)
    xs = collect(range(0.0, 1.0; length=5))
    Y = zeros(ComplexF64, 5, n_series)
    return cubic_interp(xs, Series(Y))
end

"""
    MetricData

Metric tensor components on the (ψ, θ) grid and their Fourier representation.

## Fields

  - `mpsi::Int`: Number of radial grid points minus one.
  - `mtheta::Int`: Number of poloidal grid points minus one.
  - `xs::Vector{Float64}`: Radial coordinates (normalized poloidal flux `ψ_norm`).
  - `ys::Vector{Float64}`: Poloidal angle coordinates `θ` in radians (0 to 2π).
  - `fs::Array{Float64, 3}`: The raw metric data on the grid, size `(mpsi, mtheta, 8)`.
    The 8 quantities are: `g¹¹`, `g²²`, `g³³`, `g²³`, `g³¹`, `g¹²`, `J`, `∂J/∂ψ`.
  - `fourier_coeffs::Utilities.FourierCoefficients`: The FFT coefficients (no spline interpolation needed).
"""
@kwdef mutable struct MetricData
    mpsi::Int
    mtheta::Int
    xs::Vector{Float64} = zeros(mpsi)
    ys::Vector{Float64} = zeros(mtheta)
    fs::Array{Float64,3} = zeros(mpsi, mtheta, 8)
    fourier_coeffs::Utilities.FourierCoefficients = Utilities.empty_FourierCoefficients()
end

MetricData(mpsi::Int, mtheta::Int) = MetricData(; mpsi, mtheta)

"""
    make_metric(equil::Equilibrium.PlasmaEquilibrium, mpert::Int) -> MetricData

Construct the metric tensor data on the (ψ, θ) grid of `equil`, Fourier-decomposed up to
harmonic `mpert - 1`. The coefficients stored in `metric.fs` are:

 1. g_ψψ / J
 2. g_θθ / J
 3. g_ζζ / J
 4. g_θζ / J
 5. g_ζψ / J
 6. g_ψθ / J
 7. J (Jacobian)
 8. ∂J/∂ψ
"""
function make_metric(equil::Equilibrium.PlasmaEquilibrium, mpert::Int)

    # --- Extract data from the PlasmaEquilibrium object ---
    mpsi = length(equil.rzphi_xs)
    mtheta = length(equil.rzphi_ys)

    # Set coordinate grids based on the input equilibrium
    # The equil.rzphi_ys is normalized (0 to 1), so scale to radians.
    metric = MetricData(mpsi, mtheta)
    metric.xs .= equil.rzphi_xs
    metric.ys .= equil.rzphi_ys .* 2π

    # Temporary array for the covariant basis vectors over J
    v = @MMatrix zeros(Float64, 3, 3)

    # --- Main computation loop over the (ψ, θ) grid ---
    # Access grid point values and derivatives directly from interpolant storage
    for ipsi in 1:mpsi
        for jtheta in 1:mtheta
            theta_norm = equil.rzphi_ys[jtheta] # θ is from 0 to 1

            # Grid point values: nodal_derivs.partials[1,:,:] = f, [2,:,:] = ∂f/∂ψ, [3,:,:] = ∂f/∂θ
            r_coord_sq = equil.rzphi_rsquared.nodal_derivs.partials[1, ipsi, jtheta]
            eta_offset = equil.rzphi_offset.nodal_derivs.partials[1, ipsi, jtheta]
            jac = equil.rzphi_jac.nodal_derivs.partials[1, ipsi, jtheta]
            jac1 = equil.rzphi_jac.nodal_derivs.partials[2, ipsi, jtheta] # ∂J/∂ψ

            rfac = sqrt(r_coord_sq)
            eta = 2π * (theta_norm + eta_offset)
            r_major = equil.ro + rfac * cos(eta) # This is the R coordinate

            # --- Covariant basis vectors e_ψ, e_θ, e_ζ divided by J ---
            fx1 = equil.rzphi_rsquared.nodal_derivs.partials[2, ipsi, jtheta]
            fx2 = equil.rzphi_offset.nodal_derivs.partials[2, ipsi, jtheta]
            fx3 = equil.rzphi_nu.nodal_derivs.partials[2, ipsi, jtheta]
            fy1 = equil.rzphi_rsquared.nodal_derivs.partials[3, ipsi, jtheta]
            fy2 = equil.rzphi_offset.nodal_derivs.partials[3, ipsi, jtheta]
            fy3 = equil.rzphi_nu.nodal_derivs.partials[3, ipsi, jtheta]

            v[1, 1] = fx1 / (2.0 * rfac * jac)
            v[1, 2] = fx2 * 2π * rfac / jac
            v[1, 3] = fx3 * r_major / jac
            v[2, 1] = fy1 / (2.0 * rfac * jac)
            v[2, 2] = (1.0 + fy2) * 2π * rfac / jac
            v[2, 3] = fy3 * r_major / jac
            v[3, 3] = 2π * r_major / jac

            # Store results
            v1 = @view v[1, :]
            v2 = @view v[2, :]
            metric.fs[ipsi, jtheta, 1] = dot(v1, v1) * jac
            metric.fs[ipsi, jtheta, 2] = dot(v2, v2) * jac
            metric.fs[ipsi, jtheta, 3] = v[3, 3] * v[3, 3] * jac
            metric.fs[ipsi, jtheta, 4] = v[2, 3] * v[3, 3] * jac
            metric.fs[ipsi, jtheta, 5] = v[3, 3] * v[1, 3] * jac
            metric.fs[ipsi, jtheta, 6] = dot(v1, v2) * jac
            metric.fs[ipsi, jtheta, 7] = jac
            metric.fs[ipsi, jtheta, 8] = jac1
        end
    end

    # FFT after dropping the duplicated θ=2π endpoint (matches Fortran fspline_fit_2)
    metric.fourier_coeffs = Utilities.FourierCoefficients(metric.xs, metric.ys, metric.fs, mpert - 1)
    return metric
end

"""
    build_kinetic_metric_matrices(equil, intr, metric) → NamedTuple

Compute the 5 kinetic geometric matrices (s, t, x, y, z) needed by
`KineticForces.BounceAveraging` for kinetic W vector outer products, for every toroidal mode n in
`intr.nlow:intr.nhigh`. Only Y (through m - nq) and Z (through n) depend on n; S, T and X are
repeated in each block so all five share one layout.

# Returns

NamedTuple `(smats, tmats, xmats, ymats, zmats)` of `CubicSeriesInterpolant`s, each mapping
ψ → a complex vector of `npert` stacked mpert×mpert blocks (column-major, length mpert²·npert);
block k holds n = nlow + k - 1. Use `kinetic_metric_block` to extract one block.

Reference: [Logan et al., Phys. Plasmas 20, 122507 (2013)]
"""
function build_kinetic_metric_matrices(equil::Equilibrium.PlasmaEquilibrium,
    intr::ModeSpace,
    metric::MetricData)
    (; mpert, nlow, nhigh, mlow, mhigh) = intr
    mpsi = metric.mpsi
    mtheta = metric.mtheta
    chi1 = 2π * equil.psio

    # Allocate fmodb grid: 8 quantities on (ψ,θ)
    fmodb_fs = zeros(Float64, mpsi, mtheta, 8)

    # Temporary for the covariant basis vectors (Fortran style, WITHOUT /jac)
    v = @MMatrix zeros(Float64, 3, 3)
    hint = Ref(1)

    for ipsi in 1:mpsi
        psi = metric.xs[ipsi]
        p1 = equil.profiles.P_deriv(psi; hint=hint)
        q = equil.profiles.q_spline.y[ipsi]

        for jtheta in 1:mtheta
            # Grid point values via nodal_derivs (same grid as make_metric)
            r_coord_sq = equil.rzphi_rsquared.nodal_derivs.partials[1, ipsi, jtheta]
            eta_offset = equil.rzphi_offset.nodal_derivs.partials[1, ipsi, jtheta]
            jac = equil.rzphi_jac.nodal_derivs.partials[1, ipsi, jtheta]
            jac1 = equil.rzphi_jac.nodal_derivs.partials[2, ipsi, jtheta]

            rfac = sqrt(r_coord_sq)
            theta_norm = equil.rzphi_ys[jtheta]
            eta = 2π * (theta_norm + eta_offset)
            rs = equil.ro + rfac * cos(eta)  # R coordinate

            # ∂/∂ψ and ∂/∂θ of (r², offset, nu)
            fx1 = equil.rzphi_rsquared.nodal_derivs.partials[2, ipsi, jtheta]
            fx2 = equil.rzphi_offset.nodal_derivs.partials[2, ipsi, jtheta]
            fx3 = equil.rzphi_nu.nodal_derivs.partials[2, ipsi, jtheta]
            fy1 = equil.rzphi_rsquared.nodal_derivs.partials[3, ipsi, jtheta]
            fy2 = equil.rzphi_offset.nodal_derivs.partials[3, ipsi, jtheta]
            fy3 = equil.rzphi_nu.nodal_derivs.partials[3, ipsi, jtheta]

            # Covariant basis vectors WITHOUT /jac (unlike make_metric)
            v[1, 1] = fx1 / (2.0 * rfac)
            v[1, 2] = fx2 * 2π * rfac
            v[1, 3] = fx3 * rs
            v[2, 1] = fy1 / (2.0 * rfac)
            v[2, 2] = (1.0 + fy2) * 2π * rfac
            v[2, 3] = fy3 * rs
            v[3, 3] = 2π * rs

            # Raw covariant metric g_ij
            g12 = v[1, 1] * v[2, 1] + v[1, 2] * v[2, 2] + v[1, 3] * v[2, 3]
            g13 = v[3, 3] * v[1, 3]
            g22 = v[2, 1]^2 + v[2, 2]^2 + v[2, 3]^2
            g23 = v[2, 3] * v[3, 3]
            g33 = v[3, 3]^2

            # B field and derivatives
            B = equil.eqfun_B.nodal_derivs.partials[1, ipsi, jtheta]
            dBdpsi = equil.eqfun_B.nodal_derivs.partials[2, ipsi, jtheta]
            dBdtheta = equil.eqfun_B.nodal_derivs.partials[3, ipsi, jtheta]
            b2h = B^2 / 2
            b2hp = B * dBdpsi
            b2ht = B * dBdtheta

            # θ-derivatives of eqfun metric quantities (Fortran eqfun%fy(2), eqfun%fy(3))
            eqfun_fy2 = equil.eqfun_metric1.nodal_derivs.partials[3, ipsi, jtheta]
            eqfun_fy3 = equil.eqfun_metric2.nodal_derivs.partials[3, ipsi, jtheta]

            # 8 fmodb quantities
            fmodb_fs[ipsi, jtheta, 1] = jac * (p1 + b2hp) -
                                        chi1^2 * b2ht * (g12 + q * g13) / (jac * b2h * 2)  # sband
            fmodb_fs[ipsi, jtheta, 2] =
                chi1^2 * b2ht * (g23 + q * g33) / (jac * b2h * 2)  # tband
            fmodb_fs[ipsi, jtheta, 3] = jac * b2h * 2               # xband
            fmodb_fs[ipsi, jtheta, 4] = jac1 * b2h * 2 -
                                        chi1^2 * b2h * 2 * eqfun_fy2                        # yband1
            fmodb_fs[ipsi, jtheta, 5] =
                -2π * chi1^2 / jac * (g12 + q * g13)                # yband2
            fmodb_fs[ipsi, jtheta, 6] =
                chi1^2 * b2h * 2 * eqfun_fy3                        # zband1
            fmodb_fs[ipsi, jtheta, 7] =
                2π * chi1^2 / jac * (g23 + q * g33)                 # zband2
            fmodb_fs[ipsi, jtheta, 8] =
                2π * chi1^2 / jac * (g22 + q * g23)                 # zband3
        end
    end

    # Fourier decompose (same approach as metric)
    fmodb_fc = Utilities.FourierCoefficients(metric.xs, metric.ys, fmodb_fs, mpert - 1)

    # Allocate flat storage for mpert×mpert matrices at each ψ
    npert = nhigh - nlow + 1
    smats_flat = zeros(ComplexF64, mpsi, mpert^2 * npert)
    tmats_flat = zeros(ComplexF64, mpsi, mpert^2 * npert)
    xmats_flat = zeros(ComplexF64, mpsi, mpert^2 * npert)
    ymats_flat = zeros(ComplexF64, mpsi, mpert^2 * npert)
    zmats_flat = zeros(ComplexF64, mpsi, mpert^2 * npert)

    # Fourier band storage
    sband = Vector{ComplexF64}(undef, 2 * mpert - 1)
    tband = Vector{ComplexF64}(undef, 2 * mpert - 1)
    xband = Vector{ComplexF64}(undef, 2 * mpert - 1)
    yband1 = Vector{ComplexF64}(undef, 2 * mpert - 1)
    yband2 = Vector{ComplexF64}(undef, 2 * mpert - 1)
    zband1 = Vector{ComplexF64}(undef, 2 * mpert - 1)
    zband2 = Vector{ComplexF64}(undef, 2 * mpert - 1)
    zband3 = Vector{ComplexF64}(undef, 2 * mpert - 1)

    for ipsi in 1:mpsi
        q = equil.profiles.q_spline.y[ipsi]

        # Fourier bands, conjugate-symmetric for ± modes
        for m in 0:(mpert-1)
            sband[mpert-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 1)
            tband[mpert-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 2)
            xband[mpert-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 3)
            yband1[mpert-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 4)
            yband2[mpert-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 5)
            zband1[mpert-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 6)
            zband2[mpert-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 7)
            zband3[mpert-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 8)
        end
        for k in 1:(mpert-1)
            sband[k+mpert] = conj(sband[mpert-k])
            tband[k+mpert] = conj(tband[mpert-k])
            xband[k+mpert] = conj(xband[mpert-k])
            yband1[k+mpert] = conj(yband1[mpert-k])
            yband2[k+mpert] = conj(yband2[mpert-k])
            zband1[k+mpert] = conj(zband1[mpert-k])
            zband2[k+mpert] = conj(zband2[mpert-k])
            zband3[k+mpert] = conj(zband3[mpert-k])
        end

        # Assemble the mpert×mpert block of each n
        for n in nlow:nhigh
            block_offset = (n - nlow) * mpert^2
            for m1 in mlow:mhigh
                ipert = m1 - mlow + 1
                nq = n * q
                for dm in (1-ipert):(mpert-ipert)
                    m2 = m1 + dm
                    singfac2 = m2 - nq
                    jpert = ipert + dm
                    dmidx = dm + mpert
                    iflat = block_offset + ipert + (jpert - 1) * mpert

                    smats_flat[ipsi, iflat] = sband[dmidx]
                    tmats_flat[ipsi, iflat] = tband[dmidx]
                    xmats_flat[ipsi, iflat] = xband[dmidx]
                    ymats_flat[ipsi, iflat] = yband1[dmidx] + im * singfac2 * yband2[dmidx]
                    zmats_flat[ipsi, iflat] = zband1[dmidx] + im * (m2 * zband2[dmidx] + n * zband3[dmidx])
                end
            end
        end
    end

    # Create CubicSeriesInterpolants over ψ
    itp_opts = (; extrap=ExtendExtrap())
    smats = cubic_interp(metric.xs, Series(smats_flat); itp_opts...)
    tmats = cubic_interp(metric.xs, Series(tmats_flat); itp_opts...)
    xmats = cubic_interp(metric.xs, Series(xmats_flat); itp_opts...)
    ymats = cubic_interp(metric.xs, Series(ymats_flat); itp_opts...)
    zmats = cubic_interp(metric.xs, Series(zmats_flat); itp_opts...)

    return (; smats, tmats, xmats, ymats, zmats)
end

"""
    kinetic_metric_block(itp, psi, mpert, k) → Matrix{ComplexF64}

The mpert×mpert block `k` (toroidal mode n = nlow + k - 1) at ψ of a geometric-matrix
interpolant from `build_kinetic_metric_matrices`.
"""
function kinetic_metric_block(itp, psi::Real, mpert::Int, k::Int)
    v = itp(psi)
    return reshape(v[((k-1)*mpert^2+1):(k*mpert^2)], mpert, mpert)
end


"""
    core_capped_knots(xs, rationals) -> Vector{Int}

Indices of `xs` the EL coefficient splines keep. Below the innermost rational (or ψ = 0.1, whichever
is smaller) knots are thinned to log-uniform spacing Δψ ≥ 0.05·ψ, which resolves the near-axis
Frobenius power laws at constant relative accuracy and stops dense core packing from amplifying node
error into spline third-derivative jumps that throttle the Euler-Lagrange step size. Knots inside a
rational's resolution window (`Equilibrium.RATIONAL_RES_RADIUS`) are always kept.
"""
function core_capped_knots(xs::Vector{Float64}, rationals::Vector{Float64})::Vector{Int}
    length(xs) < 3 && return collect(eachindex(xs))
    cap_edge = 0.1
    isempty(rationals) || (cap_edge = min(cap_edge, minimum(rationals) - Equilibrium.RATIONAL_RES_RADIUS))
    in_rational_window(x) = any(abs(x - r) <= Equilibrium.RATIONAL_RES_RADIUS for r in rationals)
    keep = Int[1]
    for i in 2:(length(xs)-1)
        x = xs[i]
        (x >= cap_edge || in_rational_window(x) || (x - xs[keep[end]]) >= 0.05 * x) && push!(keep, i)
    end
    push!(keep, length(xs))
    return keep
end

"""
    build_matrix_splines(equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal, metric::MetricData) -> MatrixSplines

Assemble the ideal-MHD stability matrices on every ψ surface of `metric` and fit them as ψ-splines
(Glasser Phys. Plasmas 2016 112506 Appendix A). F̄ is stored as its lower Cholesky factor
(F̄ = L·Lᴴ), since `sing_der!` only solves against it; see [`IdealMatrices`](@ref) for the fields.
"""
function build_matrix_splines(equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal, metric::MetricData)

    # --- Extract inputs ---
    profiles = equil.profiles
    mpsi = metric.mpsi

    # Flat storage for the spline fits; zeros (not undef) keeps the n≠n' blocks zero for multi-n
    amats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    bmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    cmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    dmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    emats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    hmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    fmats_lower_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    fmats_prim_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)  # primitive F before Schur complement (for kinetic)
    fmats_gal_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)   # reduced Hermitian F̄ (before Cholesky) for the Galerkin solver
    gmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    kmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    g11 = Vector{ComplexF64}(undef, 2 * intr.mpert - 1)
    g22 = Vector{ComplexF64}(undef, 2 * intr.mpert - 1)
    g33 = Vector{ComplexF64}(undef, 2 * intr.mpert - 1)
    g23 = Vector{ComplexF64}(undef, 2 * intr.mpert - 1)
    g31 = Vector{ComplexF64}(undef, 2 * intr.mpert - 1)
    g12 = Vector{ComplexF64}(undef, 2 * intr.mpert - 1)
    jmat = Vector{ComplexF64}(undef, 2 * intr.mpert - 1)
    jmat1 = Vector{ComplexF64}(undef, 2 * intr.mpert - 1)
    jmats_flat = zeros(ComplexF64, mpsi, 2 * intr.mpert - 1)
    a_inv_dmat_temp = Matrix{ComplexF64}(undef, intr.numpert_total, intr.numpert_total)
    a_inv_cmat_temp = Matrix{ComplexF64}(undef, intr.numpert_total, intr.numpert_total)

    imat = zeros(ComplexF64, 2 * intr.mpert - 1)
    imat[intr.mpert] = 1 + 0im

    hint = Ref(1)  # Linear search hint for sequential psi access
    for ipsi in 1:mpsi
        # --- Create views for this surface ---
        amats_flatview = @view amats_flat[ipsi, :]
        bmats_flatview = @view bmats_flat[ipsi, :]
        cmats_flatview = @view cmats_flat[ipsi, :]
        dmats_flatview = @view dmats_flat[ipsi, :]
        emats_flatview = @view emats_flat[ipsi, :]
        hmats_flatview = @view hmats_flat[ipsi, :]
        fmats_lower_flatview = @view fmats_lower_flat[ipsi, :]
        gmats_flatview = @view gmats_flat[ipsi, :]
        kmats_flatview = @view kmats_flat[ipsi, :]
        # --- Profiles ---
        psi = profiles.xs[ipsi]
        p1 = profiles.P_deriv(psi; hint=hint)
        q = profiles.q_spline.y[ipsi]
        q1 = profiles.q_deriv(psi; hint=hint)
        jtheta = -profiles.F_deriv(psi; hint=hint)
        chi1 = 2π * equil.psio

        # Fill modes 0, 1, ..., mpert-1 with conjugate symmetry for ± modes
        fc = metric.fourier_coeffs
        for m in 0:(intr.mpert-1)
            g11[intr.mpert-m] = Utilities.get_complex_coeff(fc, ipsi, m, 1)
            g22[intr.mpert-m] = Utilities.get_complex_coeff(fc, ipsi, m, 2)
            g33[intr.mpert-m] = Utilities.get_complex_coeff(fc, ipsi, m, 3)
            g23[intr.mpert-m] = Utilities.get_complex_coeff(fc, ipsi, m, 4)
            g31[intr.mpert-m] = Utilities.get_complex_coeff(fc, ipsi, m, 5)
            g12[intr.mpert-m] = Utilities.get_complex_coeff(fc, ipsi, m, 6)
            jmat[intr.mpert-m] = Utilities.get_complex_coeff(fc, ipsi, m, 7)
            jmat1[intr.mpert-m] = Utilities.get_complex_coeff(fc, ipsi, m, 8)
        end
        for k in 1:(intr.mpert-1)
            g11[k+intr.mpert] = conj(g11[intr.mpert-k])
            g22[k+intr.mpert] = conj(g22[intr.mpert-k])
            g33[k+intr.mpert] = conj(g33[intr.mpert-k])
            g23[k+intr.mpert] = conj(g23[intr.mpert-k])
            g31[k+intr.mpert] = conj(g31[intr.mpert-k])
            g12[k+intr.mpert] = conj(g12[intr.mpert-k])
            jmat[k+intr.mpert] = conj(jmat[intr.mpert-k])
            jmat1[k+intr.mpert] = conj(jmat1[intr.mpert-k])
        end
        # Save the Jacobian Fourier band at every surface for the power-normalization spline
        @views jmats_flat[ipsi, :] .= jmat

        for n in intr.nlow:intr.nhigh
            ipert_n = n - intr.nlow + 1
            nq = n * q
            # Construct primitive matrices via m1/dm loops
            for m1 in intr.mlow:intr.mhigh
                ipert_m = m1 - intr.mlow + 1
                singfac1 = m1 - nq
                for dm in (1-ipert_m):(intr.mpert-ipert_m)
                    m2 = m1 + dm
                    singfac2 = m2 - nq
                    jpert_m = ipert_m + dm
                    dmidx = dm + intr.mpert
                    # Flattened (m, n) × (m', n') index; n' = n in 2D
                    ipert = ipert_m + (ipert_n - 1) * intr.mpert
                    jpert = jpert_m + (ipert_n - 1) * intr.mpert
                    ipert_flat = ipert + (jpert - 1) * intr.numpert_total

                    # dmat/emat are the subterms of the composite matrices in Eqs. A7-A8, not the D, E of Eq. A6
                    amats_flatview[ipert_flat] = (2π)^2 * (n^2 * g22[dmidx] + n * (m1 + m2) * g23[dmidx] + m1 * m2 * g33[dmidx])
                    bmats_flatview[ipert_flat] = -2π * im * chi1 * (n * g22[dmidx] + (m1 + nq) * g23[dmidx] + m1 * q * g33[dmidx])
                    cmats_flatview[ipert_flat] =
                        2π * im * ((2π * im * chi1 * singfac2 * (n * g12[dmidx] + m1 * g31[dmidx])) -
                                   (q1 * chi1 * (n * g23[dmidx] + m1 * g33[dmidx]))) -
                        2π * im * (jtheta * singfac1 * imat[dmidx] + n * p1 / chi1 * jmat[dmidx])
                    dmats_flatview[ipert_flat] = 2π * chi1 * (g23[dmidx] + g33[dmidx] * m1 / n)
                    emats_flatview[ipert_flat] = -chi1 / n * (q1 * chi1 * g33[dmidx] - 2π * im * chi1 * g31[dmidx] * singfac2 + jtheta * imat[dmidx])
                    hmats_flatview[ipert_flat] =
                        (q1 * chi1)^2 * g33[dmidx] +
                        (2π * chi1)^2 * singfac1 * singfac2 * g11[dmidx] -
                        2π * im * chi1 * dm * q1 * chi1 * g31[dmidx] +
                        jtheta * q1 * chi1 * imat[dmidx] +
                        p1 * jmat1[dmidx]
                    fmats_lower_flatview[ipert_flat] = (chi1 / n)^2 * g33[dmidx]
                    kmats_flatview[ipert_flat] = 2π * im * chi1 * (g23[dmidx] + g33[dmidx] * m1 / n)
                end
            end
        end

        # Factorize and build composite matrices [Glasser Phys. Plasmas 2016 112506 Appendix A]
        # Nonsingular forms F̄ and K̄ with F = QF̄Qᴴ, K = QK̄ [Glasser Phys. Plasmas 2016 112506 eq. 29]
        # We multiply by Q (singfac = m - n*q) later when performing computations
        amat = reshape(amats_flatview, intr.numpert_total, intr.numpert_total)
        cmat = reshape(cmats_flatview, intr.numpert_total, intr.numpert_total)
        dmat = reshape(dmats_flatview, intr.numpert_total, intr.numpert_total)
        emat = reshape(emats_flatview, intr.numpert_total, intr.numpert_total)
        hmat = reshape(hmats_flatview, intr.numpert_total, intr.numpert_total)
        fmat = reshape(fmats_lower_flatview, intr.numpert_total, intr.numpert_total)
        kmat = reshape(kmats_flatview, intr.numpert_total, intr.numpert_total)
        gmat = reshape(gmats_flatview, intr.numpert_total, intr.numpert_total)

        # Save primitive F for kinetic matrix construction (before Schur complement)
        @views fmats_prim_flat[ipsi, :] .= fmats_lower_flatview

        # Schur complement reduction [Glasser Phys. Plasmas 2016 112506 eq. A5-A7]
        amat_fact = cholesky(Hermitian(amat, :L))
        ldiv!(a_inv_dmat_temp, amat_fact, dmat)         # A⁻¹D
        ldiv!(a_inv_cmat_temp, amat_fact, cmat)         # A⁻¹C
        fmat .-= adjoint(dmat) * a_inv_dmat_temp        # F̃ = F - D†A⁻¹D
        kmat .= emat .- (adjoint(kmat) * a_inv_cmat_temp)  # K̃ = E - K†A⁻¹C
        gmat .= hmat .- (adjoint(cmat) * a_inv_cmat_temp)  # G̃ = H - C†A⁻¹C

        # Keep the full reduced F̄ before it is overwritten by its Cholesky factor below
        @views fmats_gal_flat[ipsi, :] .= vec(fmat)

        # Store F̄ as its lower Cholesky factor; consumers only ever solve against F̄
        fmat .= cholesky(Hermitian(fmat)).L
    end

    # Complex series interpolants over the core-capped knot grid
    itp_opts = (; extrap=ExtendExtrap())
    keep = core_capped_knots(metric.xs, [s.psifac for s in intr.sing])
    mxs = metric.xs[keep]
    length(mxs) < length(metric.xs) &&
        @info "EL coefficient-spline grid: $(length(metric.xs)) -> $(length(mxs)) knots after core density cap"
    ideal = IdealMatrices(;
        A_spline=cubic_interp(mxs, Series(amats_flat[keep, :]); itp_opts...),
        B_spline=cubic_interp(mxs, Series(bmats_flat[keep, :]); itp_opts...),
        C_spline=cubic_interp(mxs, Series(cmats_flat[keep, :]); itp_opts...),
        D_spline_prim=cubic_interp(mxs, Series(dmats_flat[keep, :]); itp_opts...),
        E_spline_prim=cubic_interp(mxs, Series(emats_flat[keep, :]); itp_opts...),
        H_spline=cubic_interp(mxs, Series(hmats_flat[keep, :]); itp_opts...),
        F_spline_lower=cubic_interp(mxs, Series(fmats_lower_flat[keep, :]); itp_opts...),
        F_spline_prim=cubic_interp(mxs, Series(fmats_prim_flat[keep, :]); itp_opts...),
        F_spline_gal=cubic_interp(mxs, Series(fmats_gal_flat[keep, :]); itp_opts...),
        G_spline=cubic_interp(mxs, Series(gmats_flat[keep, :]); itp_opts...),
        K_spline=cubic_interp(mxs, Series(kmats_flat[keep, :]); itp_opts...),
        # Jacobian Fourier band ψ-spline, used for the power normalization in Free.jl
        J_spline=cubic_interp(mxs, Series(jmats_flat[keep, :]); itp_opts...))
    return MatrixSplines(; ideal)
end
