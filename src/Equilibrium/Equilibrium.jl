module Equilibrium

# --- Module-level Dependencies ---
import ..Spl

using Printf, OrdinaryDiffEq, DiffEqCallbacks, LinearAlgebra, HDF5
using TOML, Interpolations, ForwardDiff
import StaticArrays: @MMatrix

# --- Internal Module Structure ---
include("EquilibriumTypes.jl")
include("ReadEquilibrium.jl")
include("DirectEquilibrium.jl")
include("InverseEquilibrium.jl")
include("AnalyticEquilibrium.jl")

# --- Expose types and functions to the user ---
export setup_equilibrium, EquilibriumConfig, EquilibriumControl, EquilibriumOutput, PlasmaEquilibrium, EquilibriumParameters

# --- Constants ---
const mu0 = 4π * 1e-7

"""
    setup_equilibrium(equil_input::EquilInput)

The main public API for the `Equilibrium` module. It orchestrates the entire
process of reading an equilibrium file, running the appropriate solver, and
returning the final processed `PlasmaEquilibrium` object.

## Arguments:

  - `equil_input`: An `EquilInput` object containing all necessary setup parameters.

## Returns:

  - A `PlasmaEquilibrium` object containing the final result.
"""
function setup_equilibrium(path::String="equil.toml")
    return setup_equilibrium(EquilibriumConfig(path))
end
function setup_equilibrium(eq_config::EquilibriumConfig, additional_input=nothing)

    @printf "Equilibrium file: %s\n" eq_config.control.eq_filename

    eq_type = eq_config.control.eq_type
    # Parse file and prepare initial data structures and splines
    if eq_type == "efit"
        eq_input = read_efit(eq_config)
    elseif eq_type == "chease2"
        eq_input = read_chease2(eq_config)
    elseif eq_type == "chease"
        eq_input = read_chease(eq_config)
    elseif eq_type == "lar"

        if additional_input === nothing
            additional_input = LargeAspectRatioConfig(eq_config.control.eq_filename)
        end

        eq_input = lar_run(eq_config, additional_input)
    elseif eq_type == "sol"

        if additional_input === nothing
            additional_input = SolovevConfig(eq_config.control.eq_filename)
        end

        eq_input = sol_run(eq_config, additional_input)
    elseif eq_type == "inverse_testing"
        # Example 1D spline setup
        xs = collect(0.0:0.1:1.0)
        fs = sin.(2π .* xs)  # vector of Float64
        spline_ex = Spl.CubicSpline(xs, fs)
        #println(spline_ex)
        # Example 2D bicubic spline setup
        xs = 0.0:0.1:1.0
        ys = 0.0:0.2:1.0
        fs = [sin(2π * x) * cos(2π * y) for x in xs, y in ys, _ in 1:1]
        bicube_ex = Spl.BicubicSpline(collect(xs), collect(ys), fs)
        #println(bicube_ex)
        eq_input = InverseRunInput(
            eq_config,
            spline_ex, #sq_in
            bicube_ex, #rz_in
            0.0, #ro
            0.0, #zo
            1.0 #psio
        )
    else
        error("Equilibrium type $(equil_in.eq_type) is not implemented")
    end

    # Run the appropriate solver (direct or inverse) to get a PlasmaEquilibrium struct
    plasma_equilibrium = equilibrium_solver(eq_input)

    # add global parameters to the PlasmaEquilibrium struct
    equilibrium_global_parameters!(plasma_equilibrium)

    # Find q information
    equilibrium_qfind!(plasma_equilibrium)

    # Diagnoses grad-shafranov solution.
    equilibrium_gse!(plasma_equilibrium)

    return plasma_equilibrium
end

"""
    equilibrium_separatrix_find!(pe::PlasmaEquilibrium)

Finds the separatrix locations in the plasma equilibrium (rsep, zsep, rext, zext).
Performs the same function as equil_out_sep_find in the Fortran code.
"""
function equilibrium_separatrix_find!(pe::PlasmaEquilibrium)
    mpsi = length(pe.psi_grid) - 1
    mtheta = length(pe.theta_grid) - 1

    # Allocate vector to store eta offset
    vector = zeros(Float64, mtheta + 1)
    psi_edge = pe.psi_grid[mpsi+1]
    for it in 0:mtheta
        theta_val = pe.theta_grid[it+1]
        eta_offset = pe.eta_spline(psi_edge, theta_val)
        vector[it+1] = theta_val + eta_offset
    end

    psifac = psi_edge
    eta0 = 0.0
    idx = findmin(abs.(vector .- eta0))[2]
    theta = pe.theta_grid[idx]
    rsep = zeros(2)

    for iside in 1:2
        it = 0
        while true
            it += 1
            # Wrap theta to [0, 1] for periodic boundary before using
            theta_wrapped = mod(theta, 1.0)

            # Evaluate eta_spline and its derivative
            eta_val = pe.eta_spline(psifac, theta_wrapped)
            eta_dtheta = ForwardDiff.derivative(t -> pe.eta_spline(psifac, mod(t, 1.0)), theta_wrapped)

            eta = theta_wrapped + eta_val - eta0
            eta_theta = 1 + eta_dtheta
            if abs(eta_theta) < 1e-14
                # Avoid division by zero
                break
            end
            dtheta = -eta / eta_theta
            if isnan(dtheta) || isinf(dtheta)
                # Safety check for NaN/Inf
                break
            end
            theta += dtheta
            if abs(eta) <= 1e-10 || it > 100
                break
            end
        end
        theta = mod(theta, 1.0)  # Final wrap
        r2_val = pe.r2_spline(psifac, theta)
        eta_val = pe.eta_spline(psifac, theta)
        rsep[iside] = pe.ro + sqrt(r2_val) * cos(2π * (theta + eta_val))
        eta0 = 0.5
        idx = findmin(abs.(vector .- eta0))[2]
        theta = pe.theta_grid[idx]
    end

    # Top and bottom separatrix locations using Newton iteration
    zsep = zeros(2)
    rext = zeros(2)
    zext = zeros(2)

    for iside in 1:2
        eta0 = (iside == 1) ? 0.0 : 0.5
        idx = findmin(abs.(vector .- eta0))[2]
        theta = pe.theta_grid[idx]
        rfac = 0.0
        cosfac = 0.0
        z = 0.0
        it = 0
        while true
            it += 1
            # Wrap theta to [0, 1] for periodic boundary
            theta_wrapped = mod(theta, 1.0)

            # Compute values and derivatives using ForwardDiff
            r2 = pe.r2_spline(psifac, theta_wrapped)
            r2y = ForwardDiff.derivative(t -> pe.r2_spline(psifac, t), theta_wrapped)
            r2yy = ForwardDiff.derivative(t -> ForwardDiff.derivative(s -> pe.r2_spline(psifac, s), t), theta_wrapped)

            eta = pe.eta_spline(psifac, theta_wrapped)
            eta1 = ForwardDiff.derivative(t -> pe.eta_spline(psifac, t), theta_wrapped)
            eta2 = ForwardDiff.derivative(t -> ForwardDiff.derivative(s -> pe.eta_spline(psifac, s), t), theta_wrapped)

            rfac = sqrt(r2)
            rfac1 = r2y / (2 * rfac)
            rfac2 = (r2yy - r2y * rfac1 / rfac) / (2 * rfac)
            phase = 2π * (theta_wrapped + eta)
            phase1 = 2π * (1 + eta1)
            phase2 = 2π * eta2
            cosfac = cos(phase)
            sinfac = sin(phase)
            z = pe.zo + rfac * sinfac
            z1 = rfac * phase1 * cosfac + rfac1 * sinfac
            z2 = (2 * rfac1 * phase1 + rfac * phase2) * cosfac + (rfac2 - rfac * phase1^2) * sinfac
            if abs(z2) < 1e-14
                # Avoid division by zero
                break
            end
            dtheta = -z1 / z2
            if isnan(dtheta) || isinf(dtheta)
                # Safety check for NaN/Inf
                break
            end
            theta += dtheta
            tol = max(1e-12 * abs(theta), 1e-14)
            if abs(dtheta) < tol || it > 100
                break
            end
        end
        theta = mod(theta, 1.0)  # Final wrap
        theta_wrapped = theta
        r2 = pe.r2_spline(psifac, theta_wrapped)
        eta = pe.eta_spline(psifac, theta_wrapped)
        rfac = sqrt(r2)
        phase = 2π * (theta_wrapped + eta)
        cosfac = cos(phase)
        sinfac = sin(phase)
        rext[iside] = pe.ro + rfac * cosfac
        zsep[iside] = pe.zo + rfac * sinfac
        zext[iside] = zsep[iside]
    end

    pe.params.rsep = rsep
    pe.params.zsep = zsep
    pe.params.rext = rext
    pe.params.zext = zext
    return (rsep, zsep, rext, zext)
end

"""
    equilibrium_global_parameters!(pe::PlasmaEquilibrium)

Computes and populates global equilibrium parameters in the `PlasmaEquilibrium`
struct, such as rmean, amean, kappa, bt0, crnt, betat, betan, li1, etc. Performs
the same function as equil_out_global in the Fortran code.
"""
function equilibrium_global_parameters!(pe::PlasmaEquilibrium)
    mpsi = length(pe.psi_grid) - 1
    mtheta = length(pe.theta_grid) - 1

    # Use separatrix geometry
    rsep, zsep, rext, _ = equilibrium_separatrix_find!(pe)

    rmean = (rsep[2] + rsep[1]) / 2
    amean = (rsep[2] - rsep[1]) / 2
    aratio = rmean / amean
    kappa = (zsep[1] - zsep[2]) / (rsep[2] - rsep[1])
    delta1 = (rmean - rext[1]) / amean
    delta2 = (rmean - rext[2]) / amean
    dpsi = 1.0 - pe.psi_grid[mpsi+1]
    # Use spline to evaluate F and its derivative at boundary
    F_boundary = pe.F_spline(pe.psi_grid[mpsi+1])
    F_deriv_boundary = ForwardDiff.derivative(pe.F_spline, pe.psi_grid[mpsi+1])
    bt0 = (F_boundary + F_deriv_boundary * dpsi) / (2π * rmean)

    pe.params.rmean = rmean
    pe.params.amean = amean
    pe.params.aratio = aratio
    pe.params.kappa = kappa
    pe.params.delta1 = delta1
    pe.params.delta2 = delta2
    pe.params.bt0 = bt0

    psio = pe.psio
    gs1 = zeros(Float64, mtheta + 1)
    gs2 = zeros(Float64, mtheta + 1)

    psi_edge = pe.psi_grid[mpsi+1]
    for itheta in 0:mtheta
        theta_val = pe.theta_grid[itheta+1]

        # Evaluate splines and derivatives
        r2_val = pe.r2_spline(psi_edge, theta_val)
        eta_val = pe.eta_spline(psi_edge, theta_val)
        jac = pe.jac_spline(psi_edge, theta_val)

        r2_dtheta = ForwardDiff.derivative(t -> pe.r2_spline(psi_edge, t), theta_val)
        eta_dtheta = ForwardDiff.derivative(t -> pe.eta_spline(psi_edge, t), theta_val)

        chi1 = 2π * psio / jac
        jacfac = π / jac
        rfac = sqrt(r2_val)
        eta = 2π * (theta_val + eta_val)
        r = pe.ro + rfac * cos(eta)
        v21 = jacfac * r2_dtheta / (2π * rfac)
        v22 = jacfac * (1 + eta_dtheta) * (2 * rfac)
        v33 = jacfac * 2π * (r / π)
        dvsq = (v21^2 + v22^2) * (v33 * jac^2)^2
        gs1[itheta+1] = sqrt(dvsq) / (2π * r)
        gs2[itheta+1] = chi1 * dvsq / (2π * r)^2
    end

    int1 = sum(gs1) / (mtheta + 1)
    int2 = sum(gs2) / (mtheta + 1)
    crnt = int2 / (1e6 * mu0)
    bp0 = int2 / int1
    bwall = 1e6 * mu0 * crnt / (2π * amean)

    pe.params.crnt = crnt
    pe.params.bwall = bwall

    # Flux surface integrals using stored node values
    hs1 = pe.P_values .* pe.dVdpsi_values          # p * dV/dpsi
    hs2 = pe.dVdpsi_values                         # dV/dpsi
    hs3 = pe.P_values .^ 2 .* pe.dVdpsi_values     # p^2 * dV/dpsi

    dpsi_vec = diff(pe.psi_grid)
    fsi1 = sum((hs1[1:end-1] .+ hs1[2:end]) .* dpsi_vec) / 2
    fsi2 = sum((hs2[1:end-1] .+ hs2[2:end]) .* dpsi_vec) / 2
    fsi3 = sum((hs3[1:end-1] .+ hs3[2:end]) .* dpsi_vec) / 2

    volume = sum((pe.dVdpsi_values[1:end-1] .+ pe.dVdpsi_values[2:end]) .* dpsi_vec) / 2

    # Evaluate P and its derivative at axis for linear extrapolation
    P_axis = pe.P_spline(pe.psi_grid[1])
    P_deriv_axis = ForwardDiff.derivative(pe.P_spline, pe.psi_grid[1])
    p0 = P_axis - P_deriv_axis * pe.psi_grid[1]
    betat = 2 * (fsi1 / fsi2) / bt0^2
    betaj = 2 * sqrt(fsi3 / fsi2) / bwall^2
    betan = 100 * amean * bt0 * betat / crnt
    betap1 = 2 * (fsi1 / fsi2) / bp0^2
    betap2 = 4 * fsi1 / ((1e6 * mu0 * crnt)^2 * pe.ro)
    betap3 = 4 * fsi1 / ((1e6 * mu0 * crnt)^2 * rmean)
    li1 = fsi3 / fsi2 / bp0^2
    li2 = 2 * fsi3 / ((1e6 * mu0 * crnt)^2 * pe.ro)
    li3 = 2 * fsi3 / ((1e6 * mu0 * crnt)^2 * rmean)

    pe.params.psi0 = psio
    pe.params.psi_axis = pe.psio
    pe.params.psi_boundary = 1.0
    pe.params.psi_boundary_norm = 1.0
    pe.params.psi_axis_norm = 0.0
    pe.params.psi_norm = 0.0
    pe.params.psi_axis_offset = 0.0
    pe.params.psi_boundary_offset = 0.0
    pe.params.psi_axis_sign = 1
    pe.params.psi_boundary_sign = -1
    pe.params.psi_boundary_zero = false

    pe.params.q0 = pe.q_values[1]
    pe.params.b0 = bt0

    pe.params.volume = volume
    pe.params.betat = betat
    pe.params.betan = betan
    pe.params.betaj = betaj
    pe.params.betap1 = betap1
    pe.params.betap2 = betap2
    pe.params.betap3 = betap3
    pe.params.li1 = li1
    pe.params.li2 = li2
    pe.params.li3 = li3
end

"""
    equilibrium_qfind!(equil::PlasmaEquilibrium)

Finds the extrema of the safety factor profile q(ψ) in the plasma equilibrium
and computes derived q-values such as q0, qmin, qmax, qa, and q95. Performs
the same function as equil_out_qfind in the Fortran code.
"""
function equilibrium_qfind!(equil::PlasmaEquilibrium)

    psi_grid = equil.psi_grid
    q_values = equil.q_values
    mpsi = length(psi_grid) - 1
    psiexl = Float64[]
    qexl = Float64[]

    # Left endpoint
    push!(psiexl, psi_grid[1])
    push!(qexl, q_values[1])

    # Search for extrema in q(ψ) using derivatives
    for ipsi in 1:mpsi
        x0 = psi_grid[ipsi]
        x1 = psi_grid[ipsi+1]
        xmax = x1 - x0

        # Evaluate q and its derivatives using ForwardDiff
        a = equil.q_spline(x0)  # q value
        b = ForwardDiff.derivative(equil.q_spline, x0)  # first derivative
        c = ForwardDiff.derivative(x -> ForwardDiff.derivative(equil.q_spline, x), x0)  # second derivative
        d = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(equil.q_spline, y), x), x0)  # third derivative

        if d != 0.0
            xcrit = -c / d
            dx2 = xcrit^2 - 2b / d
            if dx2 ≥ 0
                dx = sqrt(dx2)
                for delta in (dx, -dx)
                    x = xcrit - delta
                    if 0 ≤ x < xmax
                        ψ = x0 + x
                        q_psi = equil.q_spline(ψ)
                        push!(psiexl, ψ)
                        push!(qexl, q_psi)
                    end
                end
            end
        end
    end

    # Right endpoint
    push!(psiexl, psi_grid[end])
    push!(qexl, q_values[end])

    equil.params.qextrema_psi = psiexl
    equil.params.qextrema_q = qexl
    equil.params.mextrema = length(psiexl)

    # Compute derived q-values using splines and node values
    q0_val = equil.q_spline(psi_grid[1])
    q0_deriv = ForwardDiff.derivative(equil.q_spline, psi_grid[1])
    q0 = q0_val - q0_deriv * psi_grid[1]
    qmax_edge = q_values[end]
    qmin = min(minimum(qexl), q0)
    qmax = max(maximum(qexl), qmax_edge)
    qa_val = equil.q_spline(psi_grid[end])
    qa_deriv = ForwardDiff.derivative(equil.q_spline, psi_grid[end])
    qa = qa_val + qa_deriv * (1.0 - psi_grid[end])

    q95 = equil.q_spline(0.95)

    # Store derived values
    equil.params.q0 = q0
    equil.params.qmin = qmin
    equil.params.qmax = qmax
    equil.params.qa = qa
    equil.params.q95 = q95
end

"""
    equilibrium_gse!(equil::PlasmaEquilibrium)

Diagnoses the Grad-Shafranov solution by computing the residual of the
Grad-Shafranov equation across the grid and writing diagnostic data to HDF5 files.
Performs the same function as equil_out_gse in the Fortran code.
"""
function equilibrium_gse!(equil::PlasmaEquilibrium)

    mpsi = length(equil.psi_grid) - 1
    mtheta = length(equil.theta_grid) - 1
    ro, zo = equil.ro, equil.zo
    psio = equil.psio
    verbose = equil.params.verbose
    diagnose_src = equil.params.diagnose_src
    diagnose_maxima = equil.params.diagnose_maxima

    if verbose
        println("Diagnosing Grad-Shafranov solution...")
    end

    rfac = zeros(Float64, mtheta + 1)
    angle = zeros(Float64, mtheta + 1)
    r = zeros(Float64, mpsi + 1, mtheta + 1)
    z = zeros(Float64, mpsi + 1, mtheta + 1)

    for ipsi in 0:mpsi
        psi_val = equil.psi_grid[ipsi+1]
        for itheta in 0:mtheta
            theta_val = equil.theta_grid[itheta+1]
            r2_val = equil.r2_spline(psi_val, theta_val)
            eta_val = equil.eta_spline(psi_val, theta_val)
            rfac[itheta+1] = sqrt(r2_val)
            angle[itheta+1] = 2π * (theta_val + eta_val)
        end
        r[ipsi+1, :] .= ro .+ rfac .* cos.(angle)
        z[ipsi+1, :] .= zo .+ rfac .* sin.(angle)
    end

    flux_fs = zeros(Float64, mpsi + 1, mtheta + 1, 2)
    for ipsi in 0:mpsi
        psi_val = equil.psi_grid[ipsi+1]
        for itheta in 0:mtheta
            theta_val = equil.theta_grid[itheta+1]

            # Evaluate spline values
            r2_val = equil.r2_spline(psi_val, theta_val)
            eta_val = equil.eta_spline(psi_val, theta_val)
            jac_val = equil.jac_spline(psi_val, theta_val)

            # Compute derivatives using ForwardDiff
            r2_dtheta = ForwardDiff.derivative(t -> equil.r2_spline(psi_val, t), theta_val)
            eta_dtheta = ForwardDiff.derivative(t -> equil.eta_spline(psi_val, t), theta_val)
            r2_dpsi = ForwardDiff.derivative(p -> equil.r2_spline(p, theta_val), psi_val)
            eta_dpsi = ForwardDiff.derivative(p -> equil.eta_spline(p, theta_val), psi_val)

            flux_fs[ipsi+1, itheta+1, 1] = r2_dtheta^2 / (4π^2 * r2_val) + (1 + eta_dtheta)^2 * 4 * r2_val
            flux_fs[ipsi+1, itheta+1, 2] = r2_dpsi * r2_dtheta / (4π^2 * r2_val) + eta_dpsi * (1 + eta_dtheta) * 4 * r2_val

            for iqty in 1:2
                flux_fs[ipsi+1, itheta+1, iqty] *= 2π * psio / jac_val
            end
        end
    end

    # Create 2D spline for flux using Interpolations.jl
    n_psi = length(equil.psi_grid)
    n_theta = length(equil.theta_grid)
    index_psi = range(1.0, Float64(n_psi), length=n_psi)
    theta_range = range(equil.theta_grid[1], equil.theta_grid[end], length=n_theta)

    flux_fs_1_itp = interpolate(flux_fs[:, :, 1], (BSpline(Cubic(Flat(OnGrid()))), BSpline(Cubic(Periodic(OnCell())))))
    flux_fs_1_scaled = scale(flux_fs_1_itp, index_psi, theta_range)
    psi_to_index_flux = interpolate((equil.psi_grid,), collect(index_psi), Gridded(Linear()))
    flux_1_spline = NonUniformSplineWrapper2D(flux_fs_1_scaled, psi_to_index_flux)

    flux_fs_2_itp = interpolate(flux_fs[:, :, 2], (BSpline(Cubic(Flat(OnGrid()))), BSpline(Cubic(Periodic(OnCell())))))
    flux_fs_2_scaled = scale(flux_fs_2_itp, index_psi, theta_range)
    flux_2_spline = NonUniformSplineWrapper2D(flux_fs_2_scaled, psi_to_index_flux)

    source = zeros(Float64, mpsi + 1, mtheta + 1)
    for ipsi in 0:mpsi
        psi_val = equil.psi_grid[ipsi+1]
        for itheta in 0:mtheta
            theta_val = equil.theta_grid[itheta+1]
            jac_val = equil.jac_spline(psi_val, theta_val)

            # Evaluate F and P derivatives at this psi
            s1 = equil.F_values[ipsi+1]  # F*2π value
            s1p = ForwardDiff.derivative(equil.F_spline, psi_val)  # F derivative
            s2p = ForwardDiff.derivative(equil.P_spline, psi_val)  # P derivative

            denom = (2π * r[ipsi+1, itheta+1])^2
            source[ipsi+1, itheta+1] = jac_val / (2π * psio * π^2) * (s1 * s1p / denom + s2p)
        end
    end

    # Compute flux derivatives on the grid
    flux_fsx = zeros(Float64, mpsi + 1, mtheta + 1)
    flux_fsy = zeros(Float64, mpsi + 1, mtheta + 1)
    for ipsi in 0:mpsi
        psi_val = equil.psi_grid[ipsi+1]
        for itheta in 0:mtheta
            theta_val = equil.theta_grid[itheta+1]
            flux_fsx[ipsi+1, itheta+1] = ForwardDiff.derivative(p -> flux_1_spline(p, theta_val), psi_val)
            flux_fsy[ipsi+1, itheta+1] = ForwardDiff.derivative(t -> flux_2_spline(psi_val, t), theta_val)
        end
    end

    total = flux_fsx .- flux_fsy .+ source
    error = abs.(total) ./ maximum([maximum(abs.(flux_fsx)), maximum(abs.(flux_fsy)), maximum(abs.(source))])
    errlog = ifelse.(error .> 0, log10.(error), 0.0)

    if diagnose_maxima
        fxmax = maximum(abs.(flux_fsx))
        fymax = maximum(abs.(flux_fsy))
        smax = maximum(abs.(source))
        emax = maximum(abs.(error))
        lmax = maximum(errlog)
        jmax = Tuple(argmax(errlog))
        println(" fxmax = $fxmax, fymax = $fymax, smax = $smax")
        println(" emax = $emax, lmax = $lmax, maxloc = ", (jmax[1]-1, jmax[2]-1))
    end

    # Integrated error criterion
    term = zeros(Float64, mpsi + 1, 2)
    for ipsi in 0:mpsi
        # Integrate from 0 to 1 (theta range) using simple average approximation
        # For periodic boundary with uniform grid, integral ≈ mean * (theta_max - theta_min)
        # Since theta_grid spans [0, 1], this simplifies to: mean
        term[ipsi+1, 1] = sum(flux_fsx[ipsi+1, :]) / (mtheta + 1)
        term[ipsi+1, 2] = sum(source[ipsi+1, :]) / (mtheta + 1)
    end

    totali = sum(term; dims=2)
    errori = abs.(totali)
    errlogi = @. ifelse(errori > 0, log10(errori), 0.0)

    if diagnose_src
        if verbose
            println("Writing diagnostics to HDF5 files...")
        end

        # Write contour data
        h5open(joinpath(dirname(equil.config.control.eq_filename), "gsec.h5"), "w") do file
            file["mpsi"] = mpsi
            file["mtheta"] = mtheta
            file["r"] = Float32.(r)
            file["z"] = Float32.(z)
            file["flux_fsx"] = Float32.(flux_fsx)
            file["flux_fsy"] = Float32.(flux_fsy)
            file["source"] = Float32.(source)
            file["total"] = Float32.(total)
            file["error"] = Float32.(error)
            file["errlog"] = Float32.(errlog)
        end

        # Write xy plot data
        h5open(joinpath(dirname(equil.config.control.eq_filename), "gse.h5"), "w") do file
            gse_data = Array{Float32,3}(undef, mpsi + 1, mtheta + 1, 7)
            for ipsi in 0:mpsi
                for itheta in 0:mtheta
                    gse_data[ipsi+1, itheta+1, 1] = Float32(equil.theta_grid[itheta+1])
                    gse_data[ipsi+1, itheta+1, 2] = Float32(equil.psi_grid[ipsi+1])
                    gse_data[ipsi+1, itheta+1, 3] = Float32(flux_fs[ipsi+1, itheta+1, 1])
                    gse_data[ipsi+1, itheta+1, 4] = Float32(flux_fs[ipsi+1, itheta+1, 2])
                    gse_data[ipsi+1, itheta+1, 5] = Float32(source[ipsi+1, itheta+1])
                    gse_data[ipsi+1, itheta+1, 6] = Float32(total[ipsi+1, itheta+1])
                    gse_data[ipsi+1, itheta+1, 7] = Float32(error[ipsi+1, itheta+1])
                end
            end
            file["gse_data"] = gse_data
        end

        # Write integrated error criterion
        h5open(joinpath(dirname(equil.config.control.eq_filename), "gsei.h5"), "w") do file
            file["xs"] = Float32.(equil.psi_grid)
            file["term"] = Float32.(term)
            file["totali"] = Float32.(totali)
            file["errori"] = Float32.(errori)
            file["errlogi"] = Float32.(errlogi)
        end
    end
end

end # module Equilibrium