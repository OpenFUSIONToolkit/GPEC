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

    println("   Equilibrium grid: $mpsi (ψ) × $mtheta (θ)")
    println("   Fourier fit modes (mband): $mband")

    metric = MetricData(mpsi, mtheta)

    # Set coordinate grids based on the input equilibrium
    # The `rzphi.ys` from EquilibriumAPI is normalized (0 to 1), so scale to radians.
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
    make_matrix(metric::MetricData, equil::Equilibrium.PlasmaEquilibrium, ctrl::DconControl, intr::DconInternal) -> FourFitVars

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
Determine if set_psilim_via_dmlim logic is needed
"""
function make_matrix(equil::Equilibrium.PlasmaEquilibrium, ctrl::DconControl, intr::DconInternal, metric::MetricData)

    # --- Extract inputs ---
    sq = equil.sq
    mpsi = metric.mpsi

    if ctrl.verbose
        println("   Toroidal mode n=$(ctrl.nn), Poloidal modes m=$(intr.mlow):$(intr.mhigh) ($(intr.mpert) modes)")
        println("   Matrix bandwidth: $(intr.mband)")
    end

    # Allocations
    amats = zeros(ComplexF64, mpsi, intr.mpert, intr.mpert)
    bmats = zeros(ComplexF64, mpsi, intr.mpert, intr.mpert)
    cmats = zeros(ComplexF64, mpsi, intr.mpert, intr.mpert)
    dmats = zeros(ComplexF64, mpsi, intr.mpert, intr.mpert)
    emats = zeros(ComplexF64, mpsi, intr.mpert, intr.mpert)
    hmats = zeros(ComplexF64, mpsi, intr.mpert, intr.mpert)
    fmats_lower = zeros(ComplexF64, mpsi, intr.mpert, intr.mpert)
    gmats = zeros(ComplexF64, mpsi, intr.mpert, intr.mpert)
    kmats = zeros(ComplexF64, mpsi, intr.mpert, intr.mpert)
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
        amat = @view amats[ipsi, :, :]
        bmat = @view bmats[ipsi, :, :]
        cmat = @view cmats[ipsi, :, :]
        dmat = @view dmats[ipsi, :, :]
        emat = @view emats[ipsi, :, :]
        hmat = @view hmats[ipsi, :, :]
        fmat = @view fmats_lower[ipsi, :, :]
        gmat = @view gmats[ipsi, :, :]
        kmat = @view kmats[ipsi, :, :]

        # --- Profiles ---
        p1 = sq.fs1[ipsi, 2]
        q = sq.fs[ipsi, 4]
        q1 = sq.fs1[ipsi, 4]
        jtheta = -sq.fs1[ipsi, 1]
        chi1 = 2π * equil.psio
        nq = ctrl.nn * q

        # Fill lower half (0, -1, …, -mband)
        g11[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 1:intr.mband+1]
        g22[mid:-1:1] .= metric.fspline.cs.fs[ipsi, intr.mband+2:2*intr.mband+2]
        g33[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 2*intr.mband+3:3*intr.mband+3]
        g23[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 3*intr.mband+4:4*intr.mband+4]
        g31[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 4*intr.mband+5:5*intr.mband+5]
        g12[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 5*intr.mband+6:6*intr.mband+6]
        jmat[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 6*intr.mband+7:7*intr.mband+7]
        jmat1[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 7*intr.mband+8:8*intr.mband+8]

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

        # Construct primitive matrices via m1/dm loops
        ipert = 0
        for m1 in intr.mlow:intr.mhigh
            ipert += 1
            sing1 = m1 - nq
            for dm in max(1 - ipert, -intr.mband):min(intr.mpert - ipert, intr.mband)
                m2 = m1 + dm
                sing2 = m2 - nq
                jpert = ipert + dm
                dmidx = dm + mid

                amat[ipert, jpert] = (2π)^2 * (ctrl.nn^2 * g22[dmidx] + ctrl.nn * (m1 + m2) * g23[dmidx] + m1 * m2 * g33[dmidx])
                bmat[ipert, jpert] = -2π * im * chi1 * (ctrl.nn * g22[dmidx] + (m1 + nq) * g23[dmidx] + m1 * q * g33[dmidx])
                cmat[ipert, jpert] =
                    2π * im * ((2π * im * chi1 * sing2 * (ctrl.nn * g12[dmidx] + m1 * g31[dmidx])) -
                               (q1 * chi1 * (ctrl.nn * g23[dmidx] + m1 * g33[dmidx]))) -
                    2π * im * (jtheta * sing1 * imat[dmidx] + ctrl.nn * p1 / chi1 * jmat[dmidx])
                dmat[ipert, jpert] = 2π * chi1 * (g23[dmidx] + g33[dmidx] * m1 / ctrl.nn)
                emat[ipert, jpert] = -chi1 / ctrl.nn * (q1 * chi1 * g33[dmidx] - 2π * im * chi1 * g31[dmidx] * sing2 + jtheta * imat[dmidx])
                hmat[ipert, jpert] =
                    (q1 * chi1)^2 * g33[dmidx] +
                    (2π * chi1)^2 * sing1 * sing2 * g11[dmidx] -
                    2π * im * chi1 * dm * q1 * chi1 * g31[dmidx] +
                    jtheta * q1 * chi1 * imat[dmidx] +
                    p1 * jmat1[dmidx]
                fmat[ipert, jpert] = (chi1 / ctrl.nn)^2 * g33[dmidx]
                kmat[ipert, jpert] = 2π * im * chi1 * (g23[dmidx] + g33[dmidx] * m1 / ctrl.nn)
            end
        end

        # Factorize and build composites
        # TODO: Fortran threw an error if the factorization fails for a/fmat due to small matrix bandwidth,
        # (i.e. we cut off too many terms and matrix no longer positive definite). Should add this back in
        # if we implement banded matrices.
        amat_fact = cholesky(Hermitian(amat, :L))
        temp1 = amat_fact \ dmat
        temp2 = amat_fact \ cmat
        # Use * for matrix multiplication (instead of .* for element-wise)
        fmat .-= adjoint(dmat) * temp1
        kmat .= emat .- (adjoint(kmat) * temp2)
        gmat .= hmat .- (adjoint(cmat) * temp2)

        # Store factorized F matrix (lower triangular only) since we always will need F⁻¹ later
        # and this make computation more efficient via combined forward and back substitution
        fmat .= cholesky(Hermitian(fmat)).L

        # TODO: add kinetic matrices here
    end

    # --- Fit splines (reshape 3D to 2D: (mpsi) × (mpert^2)) ---
    ffit = FourFitVars(; mpert=intr.mpert, mband=intr.mband)
    ffit.amats = Spl.CubicSpline(metric.xs, reshape(amats, mpsi, :); bctype=3)
    ffit.bmats = Spl.CubicSpline(metric.xs, reshape(bmats, mpsi, :); bctype=3)
    ffit.cmats = Spl.CubicSpline(metric.xs, reshape(cmats, mpsi, :); bctype=3)
    ffit.dmats = Spl.CubicSpline(metric.xs, reshape(dmats, mpsi, :); bctype=3)
    ffit.emats = Spl.CubicSpline(metric.xs, reshape(emats, mpsi, :); bctype=3)
    ffit.hmats = Spl.CubicSpline(metric.xs, reshape(hmats, mpsi, :); bctype=3)
    ffit.fmats_lower = Spl.CubicSpline(metric.xs, reshape(fmats_lower, mpsi, :); bctype=3)
    ffit.gmats = Spl.CubicSpline(metric.xs, reshape(gmats, mpsi, :); bctype=3)
    ffit.kmats = Spl.CubicSpline(metric.xs, reshape(kmats, mpsi, :); bctype=3)

    # TODO: set powers
    # Do we need this yet? Only called if power_flag = true

    # This is used in free_run
    ffit.jmat = jmat

    return ffit
end