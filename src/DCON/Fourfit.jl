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
  - `fourier_coeffs::Spl.FourierCoefficients`: The FFT coefficients (no spline interpolation needed).
"""
@kwdef mutable struct MetricData
    mpsi::Int
    mtheta::Int
    xs::Vector{Float64} = zeros(mpsi)
    ys::Vector{Float64} = zeros(mtheta)
    fs::Array{Float64,3} = zeros(mpsi, mtheta, 8)
    fourier_coeffs::Union{Spl.FourierCoefficients,Nothing} = nothing
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
    rzphi = equil.rzphi
    mpsi = length(rzphi.xs)
    mtheta = length(rzphi.ys)

    # Set coordinate grids based on the input equilibrium
    # The `rzphi.ys` from EquilibriumAPI is normalized (0 to 1), so scale to radians.
    metric = MetricData(mpsi, mtheta)
    metric.xs .= Vector(rzphi.xs)
    metric.ys .= Vector(rzphi.ys .* 2π)

    # Temporary array for contravariant basis vectors
    v = @MMatrix zeros(Float64, 3, 3)

    # --- Main computation loop over the (ψ, θ) grid ---
    # Use direct array access at grid points for performance
    for ipsi in 1:mpsi
        for jtheta in 1:mtheta
            theta_norm = rzphi.ys[jtheta] # θ is from 0 to 1

            # Direct array access for geometric quantities (see EquilibriumAPI.txt)
            r_coord_sq = rzphi.fs[ipsi, jtheta, 1]
            eta_offset = rzphi.fs[ipsi, jtheta, 2]
            jac = rzphi.fs[ipsi, jtheta, 4]
            jac1 = rzphi.fsx[ipsi, jtheta, 4] # ∂J/∂ψ

            rfac = sqrt(r_coord_sq)
            eta = 2π * (theta_norm + eta_offset)
            r_major = equil.ro + rfac * cos(eta) # This is the R coordinate

            # --- Compute contravariant basis vectors ∇ψ, ∇θ, ∇ζ ---
            fx1 = rzphi.fsx[ipsi, jtheta, 1]
            fx2 = rzphi.fsx[ipsi, jtheta, 2]
            fx3 = rzphi.fsx[ipsi, jtheta, 3]
            fy1 = rzphi.fsy[ipsi, jtheta, 1]
            fy2 = rzphi.fsy[ipsi, jtheta, 2]
            fy3 = rzphi.fsy[ipsi, jtheta, 3]

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
    metric.fourier_coeffs = Spl.FourierCoefficients(metric.xs, metric.ys, metric.fs, mband)
    return metric
end

"""
    make_matrix(metric::MetricData, equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal) -> FourFitVars

Constructs main DCON matrices for a given toroidal mode number and returns
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
function make_matrix(equil::Equilibrium.PlasmaEquilibrium, intr::DconInternal, metric::MetricData)

    # --- Extract inputs ---
    sq = equil.sq
    mpsi = metric.mpsi

    # Allocations (use flat storage for all matrices to fill splines)
    # TODO: This can be made more efficient for 2D equilibria by using block diagonals
    amats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    bmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    cmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    dmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    emats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    hmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    fmats_lower_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    gmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    kmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    g11 = zeros(ComplexF64, 2 * intr.mband + 1)
    g22 = zeros(ComplexF64, 2 * intr.mband + 1)
    g33 = zeros(ComplexF64, 2 * intr.mband + 1)
    g23 = zeros(ComplexF64, 2 * intr.mband + 1)
    g31 = zeros(ComplexF64, 2 * intr.mband + 1)
    g12 = zeros(ComplexF64, 2 * intr.mband + 1)
    jmat = zeros(ComplexF64, 2 * intr.mband + 1)
    jmat1 = zeros(ComplexF64, 2 * intr.mband + 1)

    # Instead of using Offset Arrays like in Fortran (-mband:mband), we store everything in
    # a single 1:(2*mband+1) array and map the zero index to the middle
    mid = intr.mband + 1  # "zero" position in Julia arrays
    imat = zeros(ComplexF64, 2 * intr.mband + 1)
    imat[mid] = 1 + 0im

    for ipsi in 1:mpsi
        # --- Create views for this surface ---
        amats_flatview = @view amats_flat[ipsi, :, :]
        bmats_flatview = @view bmats_flat[ipsi, :, :]
        cmats_flatview = @view cmats_flat[ipsi, :, :]
        dmats_flatview = @view dmats_flat[ipsi, :, :]
        emats_flatview = @view emats_flat[ipsi, :, :]
        hmats_flatview = @view hmats_flat[ipsi, :, :]
        fmats_lower_flatview = @view fmats_lower_flat[ipsi, :, :]
        gmats_flatview = @view gmats_flat[ipsi, :, :]
        kmats_flatview = @view kmats_flat[ipsi, :, :]
        # --- Profiles ---
        p1 = sq.fs1[ipsi, 2]
        q = sq.fs[ipsi, 4]
        q1 = sq.fs1[ipsi, 4]
        jtheta = -sq.fs1[ipsi, 1]
        chi1 = 2π * equil.psio

        # Fill lower half (modes 0, 1, ..., mband at indices mid, mid-1, ..., 1)
        # The 8 quantities are: g11, g22, g33, g23, g31, g12, jmat, jmat1
        fc = metric.fourier_coeffs
        for m in 0:intr.mband
            g11[mid-m] = Spl.get_complex_coeff(fc, ipsi, m, 1)
            g22[mid-m] = Spl.get_complex_coeff(fc, ipsi, m, 2)
            g33[mid-m] = Spl.get_complex_coeff(fc, ipsi, m, 3)
            g23[mid-m] = Spl.get_complex_coeff(fc, ipsi, m, 4)
            g31[mid-m] = Spl.get_complex_coeff(fc, ipsi, m, 5)
            g12[mid-m] = Spl.get_complex_coeff(fc, ipsi, m, 6)
            jmat[mid-m] = Spl.get_complex_coeff(fc, ipsi, m, 7)
            jmat1[mid-m] = Spl.get_complex_coeff(fc, ipsi, m, 8)
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
                    jpert = jpert_m + (ipert_n - 1) * intr.mpert # TODO: this will be jpert_n in 3D-DCON
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
        temp1 = amat_fact \ dmat
        temp2 = amat_fact \ cmat
        fmat .-= adjoint(dmat) * temp1
        kmat .= emat .- (adjoint(kmat) * temp2)
        gmat .= hmat .- (adjoint(cmat) * temp2)

        # Store factorized F matrix (lower triangular only) since we always will need F⁻¹ later
        # and this make computation more efficient via combined forward and back substitution
        # TODO: does F stay Hermitian in the 3D case, allowing us to use the lower representation?
        fmat .= cholesky(Hermitian(fmat)).L

        # TODO: add kinetic matrices here
    end

    # --- Fit splines using native FastInterpolations CubicSeriesInterpolant ---
    ffit = FourFitVars(; mpert=intr.mpert, mband=intr.mband, numpert_total=intr.numpert_total)

    # Pre-allocate shared buffers for real/imag extraction (reused for all 9 matrices)
    npsi, n_series = size(amats_flat)
    re_buf = Matrix{Float64}(undef, npsi, n_series)
    im_buf = Matrix{Float64}(undef, npsi, n_series)

    # Helper to extract real/imag into pre-allocated buffers (avoids broadcast allocation)
    @inline function extract_real_imag!(re::Matrix{Float64}, im::Matrix{Float64}, z::Matrix{ComplexF64})
        @inbounds for i in eachindex(z)
            re[i] = real(z[i])
            im[i] = imag(z[i])
        end
    end

    # Helper to create interpolant pair from complex matrix
    @inline function make_interp_pair(xs, z_flat, re_buf, im_buf)
        extract_real_imag!(re_buf, im_buf, z_flat)
        re_interp = cubic_interp(xs, re_buf; bc=Spl.extrap_bc_matrix(xs, re_buf), extrap=:extension, search=LinearBinary())
        im_interp = cubic_interp(xs, im_buf; bc=Spl.extrap_bc_matrix(xs, im_buf), extrap=:extension, search=LinearBinary())
        return (re_interp, im_interp)
    end

    # Create series interpolants with per-column extrap BC
    ffit.amats_real, ffit.amats_imag = make_interp_pair(metric.xs, amats_flat, re_buf, im_buf)
    ffit.bmats_real, ffit.bmats_imag = make_interp_pair(metric.xs, bmats_flat, re_buf, im_buf)
    ffit.cmats_real, ffit.cmats_imag = make_interp_pair(metric.xs, cmats_flat, re_buf, im_buf)
    ffit.dmats_real, ffit.dmats_imag = make_interp_pair(metric.xs, dmats_flat, re_buf, im_buf)
    ffit.emats_real, ffit.emats_imag = make_interp_pair(metric.xs, emats_flat, re_buf, im_buf)
    ffit.hmats_real, ffit.hmats_imag = make_interp_pair(metric.xs, hmats_flat, re_buf, im_buf)
    ffit.fmats_lower_real, ffit.fmats_lower_imag = make_interp_pair(metric.xs, fmats_lower_flat, re_buf, im_buf)
    ffit.gmats_real, ffit.gmats_imag = make_interp_pair(metric.xs, gmats_flat, re_buf, im_buf)
    ffit.kmats_real, ffit.kmats_imag = make_interp_pair(metric.xs, kmats_flat, re_buf, im_buf)

    # TODO: set powers
    # Do we need this yet? Only called if power_flag = true

    # This is used in free_run
    ffit.jmat = jmat

    return ffit
end
