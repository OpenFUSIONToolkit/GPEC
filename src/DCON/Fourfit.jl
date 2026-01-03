"""
    MetricData

A structure to hold the computed metric tensor components and their
Fourier-spline representation. This is the Julia equivalent of the `fspline_type`
named `metric` in the Fortran `fourfit_make_metric` subroutine.

### Fields

  - `mpsi::Int`: Number of radial grid points minus one.
  - `mtheta::Int`: Number of poloidal grid points minus one.
  - `xs::Vector{Float64}`: Radial coordinates (normalized poloidal flux `ψ_norm`).
  - `ys::Vector{Float64}`: Poloidal angle coordinates `θ` in radians (0 to 2π).
  - `fs::Array{Float64, 3}`: The raw metric data on the grid, size `(mpsi, mtheta, 8)`.
    The 8 quantities are: `g¹¹`, `g²²`, `g³³`, `g²³`, `g³¹`, `g¹²`, `J`, `∂J/∂ψ`.
  - `fspline::Spl.FourierSpline`: The fitted Fourier-cubic spline object.
"""
@kwdef mutable struct MetricData
    mpsi::Int
    mtheta::Int
    xs::Vector{Float64} = zeros(mpsi)
    ys::Vector{Float64} = zeros(mtheta)
    fs::Array{Float64,3} = zeros(mpsi, mtheta, 8)
    fspline::Union{Spl.FourierSpline,Nothing} = nothing
end

MetricData(mpsi::Int, mtheta::Int) = MetricData(; mpsi, mtheta)

"""
    make_metric(equil::Equilibrium.PlasmaEquilibrium; mpert::Int, fft_flag::Bool=true) -> MetricData

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

  - `mpert::Int`: Number of poloidal modes (determines Fourier modes as mpert-1).
  - `fft_flag::Bool`: If `true`, enables use of Fourier fitting for storing metric coefficients.

### Returns

  - `metric::MetricData`:
    A structure containing the metric coefficients, coordinate grids, and Jacobians for the specified equilibrium.

### TODOs

Add kinetic metric tensor components for kin_flag = true
"""
function make_metric(equil::Equilibrium.PlasmaEquilibrium; mpert::Int, fft_flag::Bool)

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
    for ipsi in 1:mpsi
        psi_norm = rzphi.xs[ipsi]
        for jtheta in 1:mtheta
            theta_norm = rzphi.ys[jtheta] # θ is from 0 to 1

            # Evaluate the geometry spline to get (R,Z) and their derivatives
            f, fx, fy = Spl.bicube_deriv1!(rzphi, psi_norm, theta_norm)

            # Extract geometric quantities from the spline data
            # See EquilibriumAPI.txt for `rzphi` quantities
            r_coord_sq = f[1]
            eta_offset = f[2]
            jac = f[4]
            jac1 = fx[4] # ∂J/∂ψ

            rfac = sqrt(r_coord_sq)
            eta = 2π * (theta_norm + eta_offset)
            r_major = equil.ro + rfac * cos(eta) # This is the R coordinate

            # --- Compute contravariant basis vectors ∇ψ, ∇θ, ∇ζ ---
            fx1, fx2, fx3 = fx[1], fx[2], fx[3]
            fy1, fy2, fy3 = fy[1], fy[2], fy[3]

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

    # --- Fit the grid data to a Fourier-cubic spline ---
    fit_method = fft_flag ? 2 : 1
    # In Fortran, `bctype` was set for the periodic `y` dimension. Here, the `FourierSpline`
    # `bctype` argument applies to the non-periodic `x` dimension. The Fortran
    # code used "extrap" for this.
    bctype_x = "not-a-knot"

    # The poloidal (y) dimension is handled implicitly as periodic by the Fourier transform.
    mband = mpert - 1  # Fourier bandwidth is mpert - 1
    metric.fspline = Spl.FourierSpline(
        metric.xs,
        metric.ys,
        metric.fs,
        mband;
        bctype=bctype_x,
        fit_method=fit_method
    )
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
    mband = intr.mpert - 1  # Fourier bandwidth is mpert - 1

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
    g11 = zeros(ComplexF64, 2 * mband + 1)
    g22 = zeros(ComplexF64, 2 * mband + 1)
    g33 = zeros(ComplexF64, 2 * mband + 1)
    g23 = zeros(ComplexF64, 2 * mband + 1)
    g31 = zeros(ComplexF64, 2 * mband + 1)
    g12 = zeros(ComplexF64, 2 * mband + 1)
    jmat = zeros(ComplexF64, 2 * mband + 1)
    jmat1 = zeros(ComplexF64, 2 * mband + 1)

    # Instead of using Offset Arrays like in Fortran (-mband:mband), we store everything in
    # a single 1:(2*mband+1) array and map the zero index to the middle
    mid = mband + 1  # "zero" position in Julia arrays
    imat = zeros(ComplexF64, 2 * mband + 1)
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

        # Fill lower half (0, -1, …, -mband)
        g11[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 1:mband+1]
        g22[mid:-1:1] .= metric.fspline.cs.fs[ipsi, mband+2:2*mband+2]
        g33[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 2*mband+3:3*mband+3]
        g23[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 3*mband+4:4*mband+4]
        g31[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 4*mband+5:5*mband+5]
        g12[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 5*mband+6:6*mband+6]
        jmat[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 6*mband+7:7*mband+7]
        jmat1[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 7*mband+8:8*mband+8]

        # Fill upper half (+1:mband) with conjugate symmetry
        for k in 1:mband
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
                for dm in max(1 - ipert_m, -mband):min(intr.mpert - ipert_m, mband)
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

    # --- Fit splines ---
    ffit = FourFitVars(; mpert=intr.mpert)
    ffit.amats = Spl.CubicSpline(metric.xs, amats_flat; bctype="extrap")
    ffit.bmats = Spl.CubicSpline(metric.xs, bmats_flat; bctype="extrap")
    ffit.cmats = Spl.CubicSpline(metric.xs, cmats_flat; bctype="extrap")
    ffit.dmats = Spl.CubicSpline(metric.xs, dmats_flat; bctype="extrap")
    ffit.emats = Spl.CubicSpline(metric.xs, emats_flat; bctype="extrap")
    ffit.hmats = Spl.CubicSpline(metric.xs, hmats_flat; bctype="extrap")
    ffit.fmats_lower = Spl.CubicSpline(metric.xs, fmats_lower_flat; bctype="extrap")
    ffit.gmats = Spl.CubicSpline(metric.xs, gmats_flat; bctype="extrap")
    ffit.kmats = Spl.CubicSpline(metric.xs, kmats_flat; bctype="extrap")

    # TODO: set powers
    # Do we need this yet? Only called if power_flag = true

    # This is used in free_run
    ffit.jmat = jmat

    return ffit
end