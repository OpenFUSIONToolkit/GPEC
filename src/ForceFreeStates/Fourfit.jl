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
    make_matrix(metric::MetricData, equil::Equilibrium.PlasmaEquilibrium, intr::ForceFreeStatesInternal) -> FourFitVars

Constructs main ForceFreeStates matrices for a given toroidal mode number and returns
them as a new `FourFitVars` object. See the appendix of the 2016 Glasser
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

        # Factorize and build composites
        # Note: we store the nonsingular forms F̄ and K̄ with F = QF̄Qᴴ, K = QK̄ (eq. 29 in Glasser 2016)
        # We multiply by Q (singfac) later when performing computations later
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
        amat_fact = cholesky(Hermitian(amat, :L))
        ldiv!(a_inv_dmat_temp, amat_fact, dmat)
        ldiv!(a_inv_cmat_temp, amat_fact, cmat)
        fmat .-= adjoint(dmat) * a_inv_dmat_temp
        kmat .= emat .- (adjoint(kmat) * a_inv_cmat_temp)
        gmat .= hmat .- (adjoint(cmat) * a_inv_cmat_temp)

        # Store factorized F matrix (lower triangular only) since we always will need F⁻¹ later
        # and this make computation more efficient via combined forward and back substitution
        # TODO: does F stay Hermitian in the 3D case, allowing us to use the lower representation?
        fmat .= cholesky(Hermitian(fmat)).L

        # TODO: add kinetic matrices here
    end

    # --- Create Fourier coefficient splines (multi-quantity cubic interpolants) ---
    ffit = FourFitVars(; mpert=intr.mpert, mband=intr.mband, numpert_total=intr.numpert_total)

    # FastInterpolations now natively supports complex values - no need to split real/imag
    # Helper to create complex interpolant directly using CubicFit() for native endpoint handling
    @inline function make_complex_interp(xs, z_flat)
        return cubic_interp(xs, z_flat; bc=CubicFit(), extrap=:extension, search=LinearBinary())
    end

    # Create complex series interpolants with per-column extrap BC
    ffit.amats = make_complex_interp(metric.xs, amats_flat)
    ffit.bmats = make_complex_interp(metric.xs, bmats_flat)
    ffit.cmats = make_complex_interp(metric.xs, cmats_flat)
    ffit.dmats = make_complex_interp(metric.xs, dmats_flat)
    ffit.emats = make_complex_interp(metric.xs, emats_flat)
    ffit.hmats = make_complex_interp(metric.xs, hmats_flat)
    ffit.fmats_lower = make_complex_interp(metric.xs, fmats_lower_flat)
    ffit.gmats = make_complex_interp(metric.xs, gmats_flat)
    ffit.kmats = make_complex_interp(metric.xs, kmats_flat)

    # TODO: set powers
    # Do we need this yet? Only called if power_flag = true

    # This is used in free_run
    ffit.jmat = jmat

    return ffit
end
