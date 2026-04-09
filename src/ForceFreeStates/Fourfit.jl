"""
    MetricData

A structure to hold the computed metric tensor components and their
Fourier representation. This is the Julia equivalent of the `fspline_type`
named `metric` in the Fortran `fourfit_make_metric` subroutine.

### Fields

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
    make_metric(equil::Equilibrium.PlasmaEquilibrium; mband::Int=10, fft_flag::Bool=true) -> MetricData

Constructs the metric tensor data on a (ψ, θ) grid from an input plasma equilibrium.
The metric coefficients stored in `metric.fs` include:

 1. g^ψψ · J
 2. g^θθ · J
 3. g^ζζ · J
 4. g^θζ · J
 5. g^ζψ · J
 6. g^ψθ · J
 7. J (Jacobian)
 8. ∂J/∂ψ

### Arguments

  - `mband::Int`: Number of Fourier modes to retain in the metric representation.
  - `fft_flag::Bool`: If `true`, enables use of Fourier fitting for storing metric coefficients.

### Returns

  - `metric::MetricData`:
    A structure containing the metric coefficients, coordinate grids, and Jacobians for the specified equilibrium.

### TODOs

Add kinetic metric tensor components for kin_flag = true
Remove mband if we decide to fully deprecate banded matrices
"""
function make_metric(equil::Equilibrium.PlasmaEquilibrium; mband::Int, fft_flag::Bool)

    # TODO: add kinetic metric tensor components

    # --- Extract data from the PlasmaEquilibrium object ---
    mpsi = length(equil.rzphi_xs)
    mtheta = length(equil.rzphi_ys)

    # Set coordinate grids based on the input equilibrium
    # The equil.rzphi_ys is normalized (0 to 1), so scale to radians.
    metric = MetricData(mpsi, mtheta)
    metric.xs .= equil.rzphi_xs
    metric.ys .= equil.rzphi_ys .* 2π

    # Temporary array for contravariant basis vectors
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

            # --- Compute contravariant basis vectors ∇ψ, ∇θ, ∇ζ ---
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

            # TODO: kinetic metric tensor here fmodb in Fortran
        end
    end

    # --- Compute Fourier coefficients (no spline overhead since we only access at grid points) ---
    metric.fourier_coeffs = Utilities.FourierCoefficients(metric.xs, metric.ys, metric.fs, mband)
    return metric
end

"""
    build_kinetic_metric_matrices(equil, intr, metric) → NamedTuple

Compute the 5 kinetic geometric matrices (s, t, x, y, z) needed by
`KineticForces.BounceAveraging` for kinetic W vector outer products.
These encode the coupling between magnetic perturbation modes and
the equilibrium geometry.

Ports Fortran `dcon_interface.f` lines 895-1013 (`idcon_action_matrices`).

# Steps
1. Compute 8 `fmodb` quantities on the (ψ,θ) grid from equilibrium geometry
2. Fourier decompose via `Utilities.FourierCoefficients`
3. Assemble mpert×mpert matrices from Fourier bands at each ψ
4. Create `CubicSeriesInterpolant`s over ψ

# Returns
NamedTuple `(smats, tmats, xmats, ymats, zmats)` of `CubicSeriesInterpolant`s,
each mapping ψ → flattened mpert² complex vector.

Reference: [Logan et al., Phys. Plasmas 20, 122507 (2013)]
"""
function build_kinetic_metric_matrices(equil::Equilibrium.PlasmaEquilibrium,
                                       intr::ForceFreeStatesInternal,
                                       metric::MetricData)
    mpsi = metric.mpsi
    mtheta = metric.mtheta
    chi1 = 2π * equil.psio

    # Allocate fmodb grid: 8 quantities on (ψ,θ)
    fmodb_fs = zeros(Float64, mpsi, mtheta, 8)

    # Temporary for contravariant basis vectors (Fortran style, WITHOUT /jac)
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

            # Contravariant basis vectors WITHOUT /jac (matches Fortran dcon_interface.f:925-931)
            v[1, 1] = fx1 / (2.0 * rfac)
            v[1, 2] = fx2 * 2π * rfac
            v[1, 3] = fx3 * rs
            v[2, 1] = fy1 / (2.0 * rfac)
            v[2, 2] = (1.0 + fy2) * 2π * rfac
            v[2, 3] = fy3 * rs
            v[3, 3] = 2π * rs

            # Raw g^ij (Fortran dcon_interface.f:933-937)
            g12 = v[1,1]*v[2,1] + v[1,2]*v[2,2] + v[1,3]*v[2,3]
            g13 = v[3,3] * v[1,3]
            g22 = v[2,1]^2 + v[2,2]^2 + v[2,3]^2
            g23 = v[2,3] * v[3,3]
            g33 = v[3,3]^2

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

            # 8 fmodb quantities (Fortran dcon_interface.f:939-948)
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
    fmodb_fc = Utilities.FourierCoefficients(metric.xs, metric.ys, fmodb_fs, intr.mband)

    # Allocate flat storage for mpert×mpert matrices at each ψ
    smats_flat = zeros(ComplexF64, mpsi, intr.mpert^2)
    tmats_flat = zeros(ComplexF64, mpsi, intr.mpert^2)
    xmats_flat = zeros(ComplexF64, mpsi, intr.mpert^2)
    ymats_flat = zeros(ComplexF64, mpsi, intr.mpert^2)
    zmats_flat = zeros(ComplexF64, mpsi, intr.mpert^2)

    # Fourier band storage
    mid = intr.mband + 1
    sband = Vector{ComplexF64}(undef, 2 * intr.mband + 1)
    tband = Vector{ComplexF64}(undef, 2 * intr.mband + 1)
    xband = Vector{ComplexF64}(undef, 2 * intr.mband + 1)
    yband1 = Vector{ComplexF64}(undef, 2 * intr.mband + 1)
    yband2 = Vector{ComplexF64}(undef, 2 * intr.mband + 1)
    zband1 = Vector{ComplexF64}(undef, 2 * intr.mband + 1)
    zband2 = Vector{ComplexF64}(undef, 2 * intr.mband + 1)
    zband3 = Vector{ComplexF64}(undef, 2 * intr.mband + 1)

    for ipsi in 1:mpsi
        q = equil.profiles.q_spline.y[ipsi]

        # Extract Fourier bands (Fortran dcon_interface.f:963-979)
        for m in 0:intr.mband
            sband[mid-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 1)
            tband[mid-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 2)
            xband[mid-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 3)
            yband1[mid-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 4)
            yband2[mid-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 5)
            zband1[mid-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 6)
            zband2[mid-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 7)
            zband3[mid-m] = Utilities.get_complex_coeff(fmodb_fc, ipsi, m, 8)
        end
        for k in 1:intr.mband
            sband[mid+k] = conj(sband[mid-k])
            tband[mid+k] = conj(tband[mid-k])
            xband[mid+k] = conj(xband[mid-k])
            yband1[mid+k] = conj(yband1[mid-k])
            yband2[mid+k] = conj(yband2[mid-k])
            zband1[mid+k] = conj(zband1[mid-k])
            zband2[mid+k] = conj(zband2[mid-k])
            zband3[mid+k] = conj(zband3[mid-k])
        end

        # Assemble mpert×mpert matrices (Fortran dcon_interface.f:981-997)
        # Single-n: use nlow as the toroidal mode number
        n = intr.nlow
        for m1 in intr.mlow:intr.mhigh
            ipert = m1 - intr.mlow + 1
            nq = n * q
            for dm in max(1-ipert, -intr.mband):min(intr.mpert-ipert, intr.mband)
                m2 = m1 + dm
                singfac2 = m2 - nq
                jpert = ipert + dm
                dmidx = dm + mid
                iflat = ipert + (jpert - 1) * intr.mpert

                smats_flat[ipsi, iflat] = sband[dmidx]
                tmats_flat[ipsi, iflat] = tband[dmidx]
                xmats_flat[ipsi, iflat] = xband[dmidx]
                ymats_flat[ipsi, iflat] = yband1[dmidx] + im * singfac2 * yband2[dmidx]
                zmats_flat[ipsi, iflat] = zband1[dmidx] +
                    im * (m2 * zband2[dmidx] + n * zband3[dmidx])
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
    make_matrix(metric::MetricData, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal) -> FourFitVars

Constructs main ForceFreeStates matrices for a given toroidal mode number and returns
them as a new `FourFitVars` object. See the appendix of the Glasser Phys. Plasmas 2016 112506
DCON paper for details on the matrix definitions. Performs the same function
as `fourfit_make_matrix` in the Fortran code, except F, G, and K are now
stored as dense matrices. The matrix F is stored in factorized form with
the lower triangle only, because F is Hermitian and can be written as
F = L · Lᴴ, which speeds up calculations later (i.e. `sing_der!``). Unlike
the Fortran, we also do not use OffsetArrays (indexed from -mband:mband),
but instead use standard Julia arrays and map the zero index to the middle.

Note that even when using dense matrices (delta_mband = 0), the
`mband` still appears here for backwards compatibility with the Fortran code,
where the Fourier splines expect it as input. So even though `mband` appears
a lot below, it is left to make implementing banded matrices easier in the future
and does not affect the actual matrix sizes, they are all dense.

### Arguments

  - `metric::MetricData`:
    Metric coefficients on the (ψ, θ) grid, including Fourier representations of g^ij and J.

### Returns

  - `ffit::FourFitVars`: A struct holding cubic spline fits of the assembled matrices

### TODOs

Add kinetic metric tensor components for kin_flag = true
Set powers if necessary
"""
function make_matrix(equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal, metric::MetricData)

    # --- Extract inputs ---
    profiles = equil.profiles
    mpsi = metric.mpsi

    # Allocations (use flat storage for all matrices to fill splines)
    # NOTE: Using zeros() instead of undef to ensure off-diagonal blocks (n≠n') are zero for block-diagonal multi-n
    amats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    bmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    cmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    dmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    emats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    hmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    fmats_lower_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    gmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    kmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    g11 = Vector{ComplexF64}(undef, 2 * intr.mband + 1)
    g22 = Vector{ComplexF64}(undef, 2 * intr.mband + 1)
    g33 = Vector{ComplexF64}(undef, 2 * intr.mband + 1)
    g23 = Vector{ComplexF64}(undef, 2 * intr.mband + 1)
    g31 = Vector{ComplexF64}(undef, 2 * intr.mband + 1)
    g12 = Vector{ComplexF64}(undef, 2 * intr.mband + 1)
    jmat = Vector{ComplexF64}(undef, 2 * intr.mband + 1)
    jmat1 = Vector{ComplexF64}(undef, 2 * intr.mband + 1)
    a_inv_dmat_temp = Matrix{ComplexF64}(undef, intr.numpert_total, intr.numpert_total)
    a_inv_cmat_temp = Matrix{ComplexF64}(undef, intr.numpert_total, intr.numpert_total)

    # Instead of using Offset Arrays like in Fortran (-mband:mband), we store everything in
    # a single 1:(2*mband+1) array and map the zero index to the middle
    mid = intr.mband + 1  # "zero" position in Julia arrays
    imat = zeros(ComplexF64, 2 * intr.mband + 1)
    imat[mid] = 1 + 0im

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

        # Fill lower half (modes 0, 1, ..., mband at indices mid, mid-1, ..., 1)
        # The 8 quantities are: g11, g22, g33, g23, g31, g12, jmat, jmat1
        fc = metric.fourier_coeffs
        for m in 0:intr.mband
            g11[mid-m] = Utilities.get_complex_coeff(fc, ipsi, m, 1)
            g22[mid-m] = Utilities.get_complex_coeff(fc, ipsi, m, 2)
            g33[mid-m] = Utilities.get_complex_coeff(fc, ipsi, m, 3)
            g23[mid-m] = Utilities.get_complex_coeff(fc, ipsi, m, 4)
            g31[mid-m] = Utilities.get_complex_coeff(fc, ipsi, m, 5)
            g12[mid-m] = Utilities.get_complex_coeff(fc, ipsi, m, 6)
            jmat[mid-m] = Utilities.get_complex_coeff(fc, ipsi, m, 7)
            jmat1[mid-m] = Utilities.get_complex_coeff(fc, ipsi, m, 8)
        end

        # Fill upper half (+1:mband) with conjugate symmetry
        for k in 1:intr.mband
            g11[mid+k] = conj(g11[mid-k])
            g22[mid+k] = conj(g22[mid-k])
            g33[mid+k] = conj(g33[mid-k])
            g23[mid+k] = conj(g23[mid-k])
            g31[mid+k] = conj(g31[mid-k])
            g12[mid+k] = conj(g12[mid-k])
            jmat[mid+k] = conj(jmat[mid-k])
            jmat1[mid+k] = conj(jmat1[mid-k])
        end

        # TODO: for 3D, would need an additional nlow:nhigh loop here for n/n' coupling
        for n in intr.nlow:intr.nhigh
            ipert_n = n - intr.nlow + 1
            nq = n * q
            # Construct primitive matrices via m1/dm loops
            for m1 in intr.mlow:intr.mhigh
                ipert_m = m1 - intr.mlow + 1
                singfac1 = m1 - nq
                for dm in max(1-ipert_m, -intr.mband):min(intr.mpert-ipert_m, intr.mband)
                    m2 = m1 + dm
                    singfac2 = m2 - nq
                    jpert_m = ipert_m + dm
                    dmidx = dm + mid
                    # Some complex indexing here... we flatten the 2D (mpert x npert) x (mpert x npert) matrix,
                    # so we need a "ipert" for m, m', n, and n' (in full 3D). In 2D, just m, m', and n
                    ipert = ipert_m + (ipert_n - 1) * intr.mpert
                    jpert = jpert_m + (ipert_n - 1) * intr.mpert # TODO: this will be jpert_n in 3D-ForceFreeStates
                    ipert_flat = ipert + (jpert - 1) * intr.numpert_total

                    # Compute matrix values at the current m, m', n (and eventually n')
                    # Note: dmat and emat here do not correspond to their representations in eq. A6 of the DCON paper,
                    # but rather to subterms that appear in the composite matrix construction in A7 and A8, respectively
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
        # TODO: Fortran threw an error if factorization fails for A/F due to small matrix bandwidth,
        # Add this check back in if we implement banded matrices

        # Schur complement reduction [Glasser Phys. Plasmas 2016 112506 eq. A5-A7]
        amat_fact = cholesky(Hermitian(amat, :L))
        ldiv!(a_inv_dmat_temp, amat_fact, dmat)         # A⁻¹D
        ldiv!(a_inv_cmat_temp, amat_fact, cmat)         # A⁻¹C
        fmat .-= adjoint(dmat) * a_inv_dmat_temp        # F̃ = F - D†A⁻¹D
        kmat .= emat .- (adjoint(kmat) * a_inv_cmat_temp)  # K̃ = E - K†A⁻¹C
        gmat .= hmat .- (adjoint(cmat) * a_inv_cmat_temp)  # G̃ = H - C†A⁻¹C

        # Store factorized F matrix (lower triangular only) since we always will need F⁻¹ later
        # and this make computation more efficient via combined forward and back substitution
        # TODO: does F stay Hermitian in the 3D case, allowing us to use the lower representation?
        fmat .= cholesky(Hermitian(fmat)).L

        # TODO: add kinetic matrices here
    end

    # --- Create Fourier coefficient splines (multi-quantity cubic interpolants) ---
    ffit = FourFitVars(; mpert=intr.mpert, mband=intr.mband, numpert_total=intr.numpert_total)

    # FastInterpolations now natively supports complex values - no need to split real/imag
    # Create complex series interpolants with per-column extrap BC
    ffit.amats = cubic_interp(metric.xs, Series(amats_flat); ffit.itp_opts...)
    ffit.bmats = cubic_interp(metric.xs, Series(bmats_flat); ffit.itp_opts...)
    ffit.cmats = cubic_interp(metric.xs, Series(cmats_flat); ffit.itp_opts...)
    ffit.dmats = cubic_interp(metric.xs, Series(dmats_flat); ffit.itp_opts...)
    ffit.emats = cubic_interp(metric.xs, Series(emats_flat); ffit.itp_opts...)
    ffit.hmats = cubic_interp(metric.xs, Series(hmats_flat); ffit.itp_opts...)
    ffit.fmats_lower = cubic_interp(metric.xs, Series(fmats_lower_flat); ffit.itp_opts...)
    ffit.gmats = cubic_interp(metric.xs, Series(gmats_flat); ffit.itp_opts...)
    ffit.kmats = cubic_interp(metric.xs, Series(kmats_flat); ffit.itp_opts...)

    # TODO: set powers
    # Do we need this yet? Only called if power_flag = true

    # This is used in free_run
    ffit.jmat = jmat

    return ffit
end
