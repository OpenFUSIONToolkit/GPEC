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

Remove mband if we decide to fully deprecate banded matrices
"""
function make_metric(equil::Equilibrium.PlasmaEquilibrium; mband::Int, fft_flag::Bool)

    # TODO: ensure kinetic metric tensor components are working

    # --- Extract data from the PlasmaEquilibrium object ---
    rzphi = equil.rzphi
    eqfun = equil.eqfun
    mpsi = length(rzphi.xs)
    mtheta = length(rzphi.ys)
    chi1 = 2π * equil.psio

    # Set coordinate grids based on the input equilibrium
    # The `rzphi.ys` from EquilibriumAPI is normalized (0 to 1), so scale to radians.
    metric = MetricData(mpsi, mtheta)
    metric.xs .= Vector(rzphi.xs)
    metric.ys .= Vector(rzphi.ys .* 2π)

    # Set up kinetic Fourier spline components
    # The initial set up is identical to the ideal metric components
    fmodb = MetricData(mpsi, mtheta)
    fmodb.xs .= Vector(rzphi.xs)
    fmodb.ys .= Vector(rzphi.ys .* 2π)

    # Temporary array for contravariant basis vectors
    v = @MMatrix zeros(Float64, 3, 3)

    # --- Main computation loop over the (ψ, θ) grid ---
    for ipsi in 1:mpsi
        psi_norm = rzphi.xs[ipsi]
        p1 = equil.sq.fs1[ipsi, 2]
        q = equil.sq.fs[ipsi, 4]
        for jtheta in 1:mtheta
            theta_norm = rzphi.ys[jtheta] # θ is from 0 to 1

            # Evaluate the geometry spline to get (R,Z) and their derivatives
            f, fx, fy = Spl.bicube_deriv1!(rzphi, psi_norm, theta_norm)

            # Evaluate the geometry spline to get (R,Z) and their derivatives
            eqfunf, eqfunfx, eqfunfy = Spl.bicube_deriv1!(eqfun, psi_norm, theta_norm)

            # Extract geometric quantities from the spline data
            # See EquilibriumAPI.txt for `rzphi` quantities
            r_coord_sq = f[1]
            eta_offset = f[2]
            jac = f[4]
            jac1 = fx[4] # ∂J/∂ψ
            b2h = eqfunf[1]^2/2
            b2hp = eqfunf[1]*eqfunfx[1] # ∂(B²/2)/∂ψ
            b2ht = eqfunf[1]*eqfunfy[1] # ∂(B²/2)/∂θ

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
            g12 = sum(v[1, :] .* v[2, :])*jac^2
            g13 = v[3, 3]*v[1, 3]*jac^2
            g22 = sum(v[2, :] .^ 2)*jac^2
            g23 = v[2, 3]*v[3, 3]*jac^2
            g33 = v[3, 3]^2*jac^2

            # Store results (computer metric tensor components)
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

            # Compute kinetic metric tensor components
            fmodb.fs[ipsi, jtheta, 1] = jac*(p1+b2hp) - chi1^2*b2ht*(g12+q*g13)/(jac*b2h*2)
            fmodb.fs[ipsi, jtheta, 2] = chi1^2*b2ht*(g23 + q*g33)/(jac*b2h*2)
            fmodb.fs[ipsi, jtheta, 3] = jac*b2h*2
            fmodb.fs[ipsi, jtheta, 4] = jac1*b2h*2 - chi1^2*b2h*2*eqfunfy[2]
            fmodb.fs[ipsi, jtheta, 5] = -2π*chi1^2/jac*(g12+q*g13)
            fmodb.fs[ipsi, jtheta, 6] = chi1^2*b2h*2*eqfunfy[3]
            fmodb.fs[ipsi, jtheta, 7] = 2π*chi1^2/jac*(g23 + q*g33)
            fmodb.fs[ipsi, jtheta, 8] = 2π*chi1^2/jac*(g22 + q*g23)
        end
    end

    # --- Fit the grid data to a Fourier-cubic spline ---
    fit_method = fft_flag ? 2 : 1
    # The `bctype` argument applies to the non-periodic radial (x) dimension.
    bctype_x = "extrap"

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

        # Fill lower half (0, -1, …, -mband)
        g11[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 1:(intr.mband+1)]
        g22[mid:-1:1] .= metric.fspline.cs.fs[ipsi, (intr.mband+2):(2*intr.mband+2)]
        g33[mid:-1:1] .= metric.fspline.cs.fs[ipsi, (2*intr.mband+3):(3*intr.mband+3)]
        g23[mid:-1:1] .= metric.fspline.cs.fs[ipsi, (3*intr.mband+4):(4*intr.mband+4)]
        g31[mid:-1:1] .= metric.fspline.cs.fs[ipsi, (4*intr.mband+5):(5*intr.mband+5)]
        g12[mid:-1:1] .= metric.fspline.cs.fs[ipsi, (5*intr.mband+6):(6*intr.mband+6)]
        jmat[mid:-1:1] .= metric.fspline.cs.fs[ipsi, (6*intr.mband+7):(7*intr.mband+7)]
        jmat1[mid:-1:1] .= metric.fspline.cs.fs[ipsi, (7*intr.mband+8):(8*intr.mband+8)]

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

        # Kinetic matrix corrections (Fortran lines 357-373). In 2D we have a single n,
        # so reuse the lowest n to define nq for the kinetic pieces.
        n = intr.nlow
        nq = n * q
        ipert = 0
        for m1 in intr.mlow:intr.mhigh
            ipert += 1
            singfac1 = m1 - nq
            for dm in max(1-ipert, -intr.mband):min(intr.mpert-ipert, intr.mband)
                m2 = m1 + dm
                singfac2 = m2 - nq
                jpert = ipert + dm
                dmidx = dm + mid
                ipert_flat = ipert + (jpert - 1) * intr.numpert_total
                dmats_flatview[ipert_flat] = chi1^2 * (g22[dmidx] + q * g23[dmidx] + q * (g23[dmidx] + q * g33[dmidx]))
                emats_flatview[ipert_flat] = chi1^2 * (q1 * (g23[dmidx] + q * g33[dmidx]) - 2π * im * (g12[dmidx] + q * g31[dmidx]) * singfac2) +
                                             p1 * jmat[dmidx]
            end
        end

        #= Comment out cholesky and uncomment this and the comment block after the fit splines part to look at det(F) at each psi
        q_diag = ((intr.mlow:intr.mhigh) .- q*intr.nlow)
        fmat .= q_diag .* fmat .* q_diag' # Apply Q on both sides to get F = Q F̄ Qᴴ
        =#

    end

    # --- Fit splines ---
    ffit = FourFitVars(; mpert=intr.mpert, mband=intr.mband)
    ffit.amats = Spl.CubicSpline(metric.xs, amats_flat; bctype="extrap")
    ffit.bmats = Spl.CubicSpline(metric.xs, bmats_flat; bctype="extrap")
    ffit.cmats = Spl.CubicSpline(metric.xs, cmats_flat; bctype="extrap")
    ffit.dmats = Spl.CubicSpline(metric.xs, dmats_flat; bctype="extrap")
    ffit.emats = Spl.CubicSpline(metric.xs, emats_flat; bctype="extrap")
    ffit.hmats = Spl.CubicSpline(metric.xs, hmats_flat; bctype="extrap")
    ffit.fmats_lower = Spl.CubicSpline(metric.xs, fmats_lower_flat; bctype="extrap")
    ffit.gmats = Spl.CubicSpline(metric.xs, gmats_flat; bctype="extrap")
    ffit.kmats = Spl.CubicSpline(metric.xs, kmats_flat; bctype="extrap")

    #=
    psi = metric.xs
    q = equil.sq.fs[:,4]
    fmats_frob = [det(reshape(fmats_lower_flat[ipsi, :], intr.numpert_total, intr.numpert_total)) for ipsi in 1:mpsi]
    println(size(fmats_frob))
    @save "fmat_frobenius_at_psi.jld2" fmats_frob psi q
    error("Debug")
    =#

    # TODO: set powers
    # Do we need this yet? Only called if power_flag = true

    # This is used in free_run
    ffit.jmat = jmat

    return ffit
end

"""
    make_kinetic_matrix(equil, intr, ctrl, metric, ffit) -> FourFitVars

Computes kinetic damping matrices and extends FourFitVars with kinetic terms.
Implements Fortran fourfit_kinetic_matrix method 0 (lines 983-1275).

# Arguments

  - `equil::Equilibrium.PlasmaEquilibrium`: Plasma equilibrium data
  - `intr::DconInternal`: Internal parameters including mband, mlow, mhigh, mpert
  - `ctrl::DconControl`: Control parameters for kinetic calculations such as the grid type
  - `metric::MetricData`: Metric coefficients on the (ψ, θ) grid
  - `ffit::FourFitVars`: Structure to store the computed spline matricesn

# Algorithm

 1. Loop over radial grid (psi) in parallel
 2. For each psi, sum kinetic contributions over all ell values
 3. Call compute_tpsi_matrices() for ions/electrons
 4. Apply normalization factors (kinfac1, kinfac2) with optional tanh smoothing
 5. Evaluate ideal matrices (A,B,C,D,E,H,F) from existing splines
 6. Add kinetic terms to create modified matrices (non-Hermitian!)
 7. Factor modified A using LU decomposition
 8. Compute 11 composite matrices for ODE solver
 9. Fit all matrices to cubic splines in psi

# Modifications from Ideal MHD

  - A matrix becomes non-Hermitian → use LU instead of Cholesky
  - New composite matrices needed for kinetic ODE formulation

# Returns

Modified `ffit` with populated kinetic matrix splines
"""
function make_kinetic_matrix(
    equil::Equilibrium.PlasmaEquilibrium,
    intr::DconInternal,
    ctrl::DconControl,
    metric::MetricData,
    ffit::FourFitVars
)::FourFitVars

    #TODO: in the original Fortran code, there was some parallelization stuff here so we can add that later

    # Extract parameters
    mpsi = metric.mpsi
    chi1 = 2π * equil.psio
    nl = ctrl.kinetic.nl

    if ctrl.nn_low == 0 #Safe guard
        error("ctrl.nn_low must be nonzero for kinetic calculations")
    end

    if ctrl.kingridtype != 0 #TODO - implement methods 1-4 from DCON (also document what each of these methods is)
        error("Only kingridtype = 0 (default) is implemented currently")
    end

    # Determine particle type flag
    ft = if ctrl.passing_flag && ctrl.trapped_flag
        "f"  # full distribution
    elseif ctrl.trapped_flag
        "t"  # trapped only
    elseif ctrl.passing_flag
        "p"  # passing only
    else
        error("Kinetic calculations require passing_flag and/or trapped_flag")
    end

    # Allocate flat storage arrays (for spline fitting)
    #TODO: I am not sure if therse are the same size as in Fortran - they are kwmatls(mpert,mpert,6,0:mpsi,-nl:nl) and I'm not sure what nl is
    # Ah- I think Claude did lines 1068-1073 here instead
    kwmats_flat = [zeros(ComplexF64, mpsi, intr.numpert_total^2) for _ in 1:6]
    ktmats_flat = [zeros(ComplexF64, mpsi, intr.numpert_total^2) for _ in 1:6]

    #TODO: these may all need to be splines? see lines 1044-1054 in Fortran
    akmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    bkmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    ckmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    f0mats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    pmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    paats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    kkmats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    kkaats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    r1mats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    r2mats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    r3mats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)
    gaats_flat = zeros(ComplexF64, mpsi, intr.numpert_total^2)

    # Parallel loop over radial surfaces (matches Fortran OMP PARALLEL DO)
    # TODO: the above is Claude's claim- verify this is true
    Threads.@threads for ipsi in 1:mpsi
        psifac = metric.xs[ipsi]

        # Accumulate kinetic contributions over ell (sequential per thread)
        kwmat_sum = zeros(ComplexF64, intr.mpert, intr.mpert, 6)
        ktmat_sum = zeros(ComplexF64, intr.mpert, intr.mpert, 6)

        for ell in (-nl):nl
            # Ions
            if ctrl.ion_flag
                kwmat_l, _ = compute_tpsi_matrices(
                    psifac, ctrl.nn_low, ell, equil, ctrl, intr,
                    is_electron=false, particle_type=ft*"wmm"
                )
                _, ktmat_l = compute_tpsi_matrices(
                    psifac, ctrl.nn_low, ell, equil, ctrl, intr,
                    is_electron=false, particle_type=ft*"tmm"
                )
                kwmat_sum .+= kwmat_l
                ktmat_sum .+= ktmat_l
            end

            # Electrons
            if ctrl.electron_flag
                kwmat_l, _ = compute_tpsi_matrices(
                    psifac, ctrl.nn_low, ell, equil, ctrl, intr,
                    is_electron=true, particle_type=ft*"wmm"
                )
                _, ktmat_l = compute_tpsi_matrices(
                    psifac, ctrl.nn_low, ell, equil, ctrl, intr,
                    is_electron=true, particle_type=ft*"tmm"
                )
                kwmat_sum .+= kwmat_l
                ktmat_sum .+= ktmat_l
            end
        end

        # Apply normalization and optional tanh smoothing
        if ctrl.ktanh_flag
            factor = (1.0 + tanh((psifac - ctrl.ktc) * ctrl.ktw))
            kwmat_sum .*= ctrl.kinfac1 * factor
            ktmat_sum .*= ctrl.kinfac2 * factor
        else
            kwmat_sum .*= ctrl.kinfac1
            ktmat_sum .*= ctrl.kinfac2
        end

        # Store raw kinetic matrices (for diagnostics)
        for i in 1:6
            kwmats_flat[i][ipsi, :] = vec(kwmat_sum[:, :, i])
            ktmats_flat[i][ipsi, :] = vec(ktmat_sum[:, :, i])
        end

        # Evaluate ideal MHD matrices at this psi
        amat_ideal = reshape(Spl.spline_eval!(ffit.amats, psifac), intr.numpert_total, intr.numpert_total)
        bmat_ideal = reshape(Spl.spline_eval!(ffit.bmats, psifac), intr.numpert_total, intr.numpert_total)
        cmat_ideal = reshape(Spl.spline_eval!(ffit.cmats, psifac), intr.numpert_total, intr.numpert_total)
        dmat_ideal = reshape(Spl.spline_eval!(ffit.dmats, psifac), intr.numpert_total, intr.numpert_total)
        emat_ideal = reshape(Spl.spline_eval!(ffit.emats, psifac), intr.numpert_total, intr.numpert_total)
        hmat_ideal = reshape(Spl.spline_eval!(ffit.hmats, psifac), intr.numpert_total, intr.numpert_total)
        dbat = copy(dmat_ideal)  # Copy for later use
        ebat = copy(emat_ideal)
        fmat = reshape(Spl.spline_eval!(ffit.fmats_lower, psifac), intr.numpert_total, intr.numpert_total)

        # Add kinetic contributions to ideal matrices
        amat = amat_ideal .+ kwmat_sum[:, :, 1] .+ ktmat_sum[:, :, 1]
        bmat = bmat_ideal .+ kwmat_sum[:, :, 2] .+ ktmat_sum[:, :, 2]
        cmat = cmat_ideal .+ kwmat_sum[:, :, 3] .+ ktmat_sum[:, :, 3]
        dmat = dmat_ideal .+ kwmat_sum[:, :, 4] .+ ktmat_sum[:, :, 4]
        emat = emat_ideal .+ kwmat_sum[:, :, 5] .+ ktmat_sum[:, :, 5]
        hmat = hmat_ideal .+ kwmat_sum[:, :, 6] .+ ktmat_sum[:, :, 6]

        # Compute auxiliary matrices
        caat = cmat .- 2.0 .* ktmat_sum[:, :, 3]
        b1mat = im .* dbat

        # Factor non-Hermitian A matrix (CRITICAL: use LU, not Cholesky!)
        amat_lu = lu(amat)

        # Compute composite kinetic matrices (Fortran lines 1184-1265)

        # f0mat = F - D† * A⁻¹ * D
        temp1 = amat_lu \ dbat
        f0mat = fmat .- adjoint(dbat) * temp1

        # Compute U = I - (A⁻¹)†
        temp2 = copy(amat)
        temp2 = amat_lu \ temp2  # Should be ≈ I
        aamat = adjoint(temp2)
        umat = I - aamat

        # bkmat = K_w2 + K_t2 + i*χ₁/(2πn) * (K_w1 + K_t1)
        bkmat = kwmat_sum[:, :, 2] .+ ktmat_sum[:, :, 2] .+
                im * chi1 / (2π * ctrl.nn_low) .* (kwmat_sum[:, :, 1] .+ ktmat_sum[:, :, 1])

        # bkaat = K_w2 - K_t2 + i*χ₁/(2πn) * (K_w1 + K_t1)
        bkaat = kwmat_sum[:, :, 2] .- ktmat_sum[:, :, 2] .+
                im * chi1 / (2π * ctrl.nn_low) .* (kwmat_sum[:, :, 1] .+ ktmat_sum[:, :, 1])

        # pmat = B₁† * A⁻¹ * B_k
        temp2 = amat_lu \ bkmat
        pmat = adjoint(b1mat) * temp2

        # paat = B_kaa† * A⁻¹ * B₁ - i*χ₁/(2πn) * U * B₁
        temp2 = amat_lu \ b1mat
        paat = adjoint(bkaat) * temp2 .- im * chi1 / (2π * ctrl.nn_low) .* (umat * b1mat)
        paat = adjoint(paat)

        # r1mat (complex expression from Fortran lines 1213-1217)
        temp1 = kwmat_sum[:, :, 1] .+ ktmat_sum[:, :, 1]
        temp2 = amat_lu \ bkmat
        r1mat =
            kwmat_sum[:, :, 4] .+ ktmat_sum[:, :, 4] .-
            (chi1 / (2π * ctrl.nn_low))^2 .* adjoint(temp1) .+
            im * chi1 / (2π * ctrl.nn_low) .* adjoint(bkaat) .-
            im * chi1 / (2π * ctrl.nn_low) .* (aamat * bkmat) .-
            adjoint(bkaat) * temp2

        # kkmat = E_b - B₁† * A⁻¹ * C
        temp1 = amat_lu \ cmat
        kkmat = ebat .- adjoint(b1mat) * temp1

        # kkaat = E_b† - C_aa† * A⁻¹ * B₁
        temp1 = amat_lu \ b1mat
        kkaat = adjoint(ebat) .- adjoint(caat) * temp1

        # r2mat
        temp1 = kwmat_sum[:, :, 5] .+ ktmat_sum[:, :, 5] .-
                im * chi1 / (2π * ctrl.nn_low) .* (kwmat_sum[:, :, 3] .+ ktmat_sum[:, :, 3])
        temp2 = amat_lu \ cmat
        r2mat = temp1 .+ im * chi1 / (2π * ctrl.nn_low) .* (umat * cmat) .-
                adjoint(bkaat) * temp2

        # r3mat
        temp1 = kwmat_sum[:, :, 5] .- ktmat_sum[:, :, 5] .-
                im * chi1 / (2π * ctrl.nn_low) .* (kwmat_sum[:, :, 3] .- ktmat_sum[:, :, 3])
        temp2 = amat_lu \ bkmat
        r3mat = adjoint(temp1) .- adjoint(caat) * temp2

        # gaat = H - C_aa† * A⁻¹ * C
        temp2 = amat_lu \ cmat
        gaat = hmat .- adjoint(caat) * temp2

        # Store to flat arrays
        akmats_flat[ipsi, :] = vec(amat)
        bkmats_flat[ipsi, :] = vec(bmat)
        ckmats_flat[ipsi, :] = vec(cmat)
        f0mats_flat[ipsi, :] = vec(f0mat)
        pmats_flat[ipsi, :] = vec(pmat)
        paats_flat[ipsi, :] = vec(paat)
        kkmats_flat[ipsi, :] = vec(kkmat)
        kkaats_flat[ipsi, :] = vec(kkaat)
        r1mats_flat[ipsi, :] = vec(r1mat)
        r2mats_flat[ipsi, :] = vec(r2mat)
        r3mats_flat[ipsi, :] = vec(r3mat)
        gaats_flat[ipsi, :] = vec(gaat)
    end

    # Fit all matrices to cubic splines
    for i in 1:6
        ffit.kwmats[i] = Spl.CubicSpline(metric.xs, kwmats_flat[i]; bctype="extrap")
        ffit.ktmats[i] = Spl.CubicSpline(metric.xs, ktmats_flat[i]; bctype="extrap")
    end

    ffit.akmats = Spl.CubicSpline(metric.xs, akmats_flat; bctype="extrap")
    ffit.bkmats = Spl.CubicSpline(metric.xs, bkmats_flat; bctype="extrap")
    ffit.ckmats = Spl.CubicSpline(metric.xs, ckmats_flat; bctype="extrap")
    ffit.f0mats = Spl.CubicSpline(metric.xs, f0mats_flat; bctype="extrap")
    ffit.pmats = Spl.CubicSpline(metric.xs, pmats_flat; bctype="extrap")
    ffit.paats = Spl.CubicSpline(metric.xs, paats_flat; bctype="extrap")
    ffit.kkmats = Spl.CubicSpline(metric.xs, kkmats_flat; bctype="extrap")
    ffit.kkaats = Spl.CubicSpline(metric.xs, kkaats_flat; bctype="extrap")
    ffit.r1mats = Spl.CubicSpline(metric.xs, r1mats_flat; bctype="extrap")
    ffit.r2mats = Spl.CubicSpline(metric.xs, r2mats_flat; bctype="extrap")
    ffit.r3mats = Spl.CubicSpline(metric.xs, r3mats_flat; bctype="extrap")
    ffit.gaats = Spl.CubicSpline(metric.xs, gaats_flat; bctype="extrap")

    return ffit
end

"""
    action_matrices!(ffit::FourFitVars, intr::DconInternal,
                             equil::Equilibrium.PlasmaEquilibrium, ctrl::DconControl, metric::MetricData)

Compute equilibrium action matrices necessary to calculate perturbed modB.
This is a conversion of the fourfit_action_matrix function.

These matrices (S, T, X, Y, Z) represent the coupling between different poloidal
mode numbers in the perturbed equilibrium and are essential for kinetic energy
calculations in MHD stability analysis.

# Arguments

  - `ffit::FourFitVars`: Structure to store the computed spline matrices
  - `intr::DconInternal`: Internal parameters including mband, mlow, mhigh, mpert
  - `equil::Equilibrium.PlasmaEquilibrium`: Plasma equilibrium data
  - `ctrl::DconControl`: Control parameters for kinetic calculations like verbosity
  - `metric::MetricData`: Metric coefficients on the (ψ, θ) grid

# Physical Meaning

The matrices appear in the perturbed energy integral:
δW = ∫∫ξ†·W·ξ dψdθ
where ξ is the plasma displacement vector and W is the Euler-Lagrange operator.

# Notes

  - Uses the same mid-index convention as `make_matrix`: mode m is at index m + mband + 1
  - Exploits conjugate symmetry for real quantities in Fourier space: f(m) = conj(f(-m))
  - Matrices are stored as flat arrays and reshaped at each radial location before spline fitting
"""
function action_matrices!(ffit::FourFitVars, intr::DconInternal,
    equil::Equilibrium.PlasmaEquilibrium, ctrl::DconControl, metric::MetricData)

    if ctrl.verbose
        println("   Computing action matrices S, T, X, Y, Z")
    end

    nn = intr.nlow  # Single toroidal mode number for 2D DCON TODO: see what to do with more toroidal modes later
    sq = equil.sq # Safety factor profile (I think)
    ifac = 1im # Imaginary unit factor

    mpsi = metric.mpsi
    mpert = intr.mpert
    mband = intr.mband
    mlow = intr.mlow
    mhigh = intr.mhigh

    # Allocate flat storage for the 5 output matrices
    smats_flat = zeros(ComplexF64, mpsi, mpert^2)
    tmats_flat = zeros(ComplexF64, mpsi, mpert^2)
    xmats_flat = zeros(ComplexF64, mpsi, mpert^2)
    ymats_flat = zeros(ComplexF64, mpsi, mpert^2)
    zmats_flat = zeros(ComplexF64, mpsi, mpert^2)

    # Allocate band arrays for Fourier coefficients
    # Using mid-index convention: mode -mband is at index 1, mode 0 is at index mband+1, mode +mband is at index 2*mband+1
    mid = mband + 1
    sband = zeros(ComplexF64, 2 * mband + 1)
    tband = zeros(ComplexF64, 2 * mband + 1)
    xband = zeros(ComplexF64, 2 * mband + 1)
    yband1 = zeros(ComplexF64, 2 * mband + 1)
    yband2 = zeros(ComplexF64, 2 * mband + 1)
    zband1 = zeros(ComplexF64, 2 * mband + 1)
    zband2 = zeros(ComplexF64, 2 * mband + 1)
    zband3 = zeros(ComplexF64, 2 * mband + 1)

    # Allocate full matrices for assembly at each radius
    smat = zeros(ComplexF64, mpert, mpert)
    tmat = zeros(ComplexF64, mpert, mpert)
    xmat = zeros(ComplexF64, mpert, mpert)
    ymat = zeros(ComplexF64, mpert, mpert)
    zmat = zeros(ComplexF64, mpert, mpert)

    # Loop over radial locations
    for ipsi in 1:mpsi
        # Get safety factor at this radius
        q = sq.fs[ipsi, 4]

        # Extract Fourier bands from metric.fspline structure
        # The fspline stores Fourier bands for (g^ij, J, dJ/dψ)
        # Negative modes (0 down to -mband): stored at indices mid:-1:1
        sband[mid:-1:1] .= metric.fspline.cs.fs[ipsi, 1:(mband+1)]
        tband[mid:-1:1] .= metric.fspline.cs.fs[ipsi, (mband+2):(2*mband+2)]
        xband[mid:-1:1] .= metric.fspline.cs.fs[ipsi, (2*mband+3):(3*mband+3)]
        yband1[mid:-1:1] .= metric.fspline.cs.fs[ipsi, (3*mband+4):(4*mband+4)]
        yband2[mid:-1:1] .= metric.fspline.cs.fs[ipsi, (4*mband+5):(5*mband+5)]
        zband1[mid:-1:1] .= metric.fspline.cs.fs[ipsi, (5*mband+6):(6*mband+6)]
        zband2[mid:-1:1] .= metric.fspline.cs.fs[ipsi, (6*mband+7):(7*mband+7)]
        zband3[mid:-1:1] .= metric.fspline.cs.fs[ipsi, (7*mband+8):(8*mband+8)]

        # Exploit conjugate symmetry for positive modes: f(m) = conj(f(-m))
        for k in 1:mband
            sband[mid+k] = conj(sband[mid-k])
            tband[mid+k] = conj(tband[mid-k])
            xband[mid+k] = conj(xband[mid-k])
            yband1[mid+k] = conj(yband1[mid-k])
            yband2[mid+k] = conj(yband2[mid-k])
            zband1[mid+k] = conj(zband1[mid-k])
            zband2[mid+k] = conj(zband2[mid-k])
            zband3[mid+k] = conj(zband3[mid-k])
        end

        # Clear the matrices for this radius
        fill!(smat, 0)
        fill!(tmat, 0)
        fill!(xmat, 0)
        fill!(ymat, 0)
        fill!(zmat, 0)

        # Build coupling matrices
        # Loop over mode pairs (m1, m2) within the perturbation spectrum
        for m1_idx in 1:mpert
            m1 = mlow + m1_idx - 1

            # Only compute band-diagonal elements (coupling within ±mband)
            dm_min = max(1 - m1_idx, -mband)
            dm_max = min(mpert - m1_idx, mband)

            for dm in dm_min:dm_max
                m2 = m1 + dm
                m2_idx = m1_idx + dm

                # Resonance factor: m2 - n*q
                # This identifies resonant surfaces where mode rotation matches field line winding
                singfac2 = m2 - nn * q

                # Get band array index for mode difference dm
                dm_idx = dm + mid

                # Fill matrix elements
                # S, T, X are simple band elements
                smat[m1_idx, m2_idx] = sband[dm_idx]
                tmat[m1_idx, m2_idx] = tband[dm_idx]
                xmat[m1_idx, m2_idx] = xband[dm_idx]

                # Y has resonance-dependent correction
                ymat[m1_idx, m2_idx] = yband1[dm_idx] + ifac * singfac2 * yband2[dm_idx]

                # Z has multiple terms with mode number dependence
                zmat[m1_idx, m2_idx] = zband1[dm_idx] +
                                       ifac * (m2 * zband2[dm_idx] + nn * zband3[dm_idx])
            end
        end

        # Store flattened matrices at this radius
        # Julia is column-major, so reshape works naturally
        smats_flat[ipsi, :] = reshape(smat, (mpert^2,))
        tmats_flat[ipsi, :] = reshape(tmat, (mpert^2,))
        xmats_flat[ipsi, :] = reshape(xmat, (mpert^2,))
        ymats_flat[ipsi, :] = reshape(ymat, (mpert^2,))
        zmats_flat[ipsi, :] = reshape(zmat, (mpert^2,))
    end

    # Fit splines in radial direction for interpolation
    ffit.smats = Spl.CubicSpline(metric.xs, smats_flat; bctype="extrap")
    ffit.tmats = Spl.CubicSpline(metric.xs, tmats_flat; bctype="extrap")
    ffit.xmats = Spl.CubicSpline(metric.xs, xmats_flat; bctype="extrap")
    ffit.ymats = Spl.CubicSpline(metric.xs, ymats_flat; bctype="extrap")
    ffit.zmats = Spl.CubicSpline(metric.xs, zmats_flat; bctype="extrap")

    if ctrl.verbose
        println("   Action matrices computed and splined")
    end
end
