module Equilibrium

# --- Module-level Dependencies ---
import ..Spl

using Printf, OrdinaryDiffEq, DiffEqCallbacks, LinearAlgebra, HDF5
using TOML, BSplineKit
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
    rzphi = pe.rzphi
    mpsi = size(rzphi.fs, 1) - 1
    mtheta = size(rzphi.fs, 2) - 1

    # Allocate vector to store eta offset from rzphi
    vector = zeros(Float64, mtheta + 1)
    for it in 0:mtheta
        f = Spl.bicube_eval!(rzphi, rzphi.xs[mpsi+1], rzphi.ys[it+1])
        vector[it+1] = rzphi.ys[it+1] + f[2]
    end

    psifac = rzphi.xs[mpsi+1]
    eta0 = 0.0
    idx = findmin(abs.(vector .- eta0))[2]
    theta = rzphi.ys[idx]
    rsep = zeros(2)

    for iside in 1:2
        it = 0
        while true
            it += 1
            f, _, fy = Spl.bicube_deriv1!(rzphi, psifac, theta)
            eta = theta + f[2] - eta0
            eta_theta = 1 + fy[2]
            dtheta = -eta / eta_theta
            theta += dtheta
            if abs(eta) <= 1e-10 || it > 100
                break
            end
        end
        f = Spl.bicube_eval!(rzphi, psifac, theta)
        rsep[iside] = pe.ro + sqrt(f[1]) * cos(2π * (theta + f[2]))
        eta0 = 0.5
        idx = findmin(abs.(vector .- eta0))[2]
        theta = rzphi.ys[idx]
    end

    # Top and bottom separatrix locations using Newton iteration
    zsep = zeros(2)
    rext = zeros(2)
    zext = zeros(2)

    for iside in 1:2
        eta0 = (iside == 1) ? 0.0 : 0.5
        idx = findmin(abs.(vector .- eta0))[2]
        theta = rzphi.ys[idx]
        rfac = 0.0
        cosfac = 0.0
        z = 0.0
        while true
            f, fx, fy, fxx, fxy, fyy = Spl.bicube_deriv2!(rzphi, psifac, theta)
            r2, r2y, r2yy = f[1], fy[1], fyy[1]
            eta, eta1, eta2 = f[2], fy[2], fyy[2]

            rfac = sqrt(r2)
            rfac1 = r2y / (2 * rfac)
            rfac2 = (r2yy - r2y * rfac1 / rfac) / (2 * rfac)
            phase = 2π * (theta + eta)
            phase1 = 2π * (1 + eta1)
            phase2 = 2π * eta2
            cosfac = cos(phase)
            sinfac = sin(phase)
            z = pe.zo + rfac * sinfac
            z1 = rfac * phase1 * cosfac + rfac1 * sinfac
            z2 = (2 * rfac1 * phase1 + rfac * phase2) * cosfac + (rfac2 - rfac * phase1^2) * sinfac
            dtheta = -z1 / z2
            theta += dtheta
            if abs(dtheta) < 1e-12 * abs(theta)
                break
            end
        end
        rext[iside] = pe.ro + rfac * cosfac
        zsep[iside] = z
        zext[iside] = z
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
    rzphi = pe.rzphi
    mpsi = size(rzphi.fs, 1) - 1
    mtheta = size(rzphi.fs, 2) - 1

    # Use separatrix geometry
    rsep, zsep, rext, _ = equilibrium_separatrix_find!(pe)

    rmean = (rsep[2] + rsep[1]) / 2
    amean = (rsep[2] - rsep[1]) / 2
    aratio = rmean / amean
    kappa = (zsep[1] - zsep[2]) / (rsep[2] - rsep[1])
    delta1 = (rmean - rext[1]) / amean
    delta2 = (rmean - rext[2]) / amean
    dpsi = 1.0 - rzphi.xs[mpsi+1]
    # Use spline to evaluate F and its derivative at boundary
    F_boundary = pe.F_spline(pe.psi_grid[mpsi+1])
    F_deriv_boundary = (BSplineKit.Derivative(1) * pe.F_spline)(pe.psi_grid[mpsi+1])
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

    for itheta in 0:mtheta
        f, _, fy = Spl.bicube_deriv1!(rzphi, rzphi.xs[mpsi+1], rzphi.ys[itheta+1])
        jac = f[4]
        chi1 = 2π * psio / jac
        jacfac = π / jac
        rfac = sqrt(f[1])
        eta = 2π * (rzphi.ys[itheta+1] + f[2])
        r = pe.ro + rfac * cos(eta)
        v21 = jacfac * fy[1] / (2π * rfac)
        v22 = jacfac * (1 + fy[2]) * (2 * rfac)
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
    P_deriv_axis = (BSplineKit.Derivative(1) * pe.P_spline)(pe.psi_grid[1])
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

        # Evaluate q and its derivatives using BSplineKit
        a = equil.q_spline(x0)  # q value
        b = (BSplineKit.Derivative(1) * equil.q_spline)(x0)  # first derivative
        c = (BSplineKit.Derivative(2) * equil.q_spline)(x0)  # second derivative
        d = (BSplineKit.Derivative(3) * equil.q_spline)(x0)  # third derivative

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
    q0_deriv = (BSplineKit.Derivative(1) * equil.q_spline)(psi_grid[1])
    q0 = q0_val - q0_deriv * psi_grid[1]
    qmax_edge = q_values[end]
    qmin = min(minimum(qexl), q0)
    qmax = max(maximum(qexl), qmax_edge)
    qa_val = equil.q_spline(psi_grid[end])
    qa_deriv = (BSplineKit.Derivative(1) * equil.q_spline)(psi_grid[end])
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

    rzphi = equil.rzphi
    mpsi, mtheta = rzphi.mx, rzphi.my
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
        for itheta in 0:mtheta
            rz_eval = Spl.bicube_eval!(rzphi, rzphi.xs[ipsi+1], rzphi.ys[itheta+1])
            rfac[itheta+1] = sqrt(rz_eval[1])
            angle[itheta+1] = 2π * (rzphi.ys[itheta+1] + rz_eval[2])
        end
        r[ipsi+1, :] .= ro .+ rfac .* cos.(angle)
        z[ipsi+1, :] .= zo .+ rfac .* sin.(angle)
    end

    flux_fs = zeros(Float64, mpsi + 1, mtheta + 1, 2)
    for ipsi in 0:mpsi
        for itheta in 0:mtheta
            f, fx, fy = Spl.bicube_deriv1!(rzphi, rzphi.xs[ipsi+1], rzphi.ys[itheta+1])
            f1, f2, f4 = f[1], f[2], f[4]

            fy1 = rzphi._fsy[ipsi+1, itheta+1, 1]
            fy2 = rzphi._fsy[ipsi+1, itheta+1, 2]
            fx1 = rzphi._fsx[ipsi+1, itheta+1, 1]
            fx2 = rzphi._fsx[ipsi+1, itheta+1, 2]

            flux_fs[ipsi+1, itheta+1, 1] = fy1^2 / (4π^2 * f1) + (1 + fy2)^2 * 4 * f1
            flux_fs[ipsi+1, itheta+1, 2] = fx1 * fy1 / (4π^2 * f1) + fx2 * (1 + fy2) * 4 * f1

            for iqty in 1:2
                flux_fs[ipsi+1, itheta+1, iqty] *= 2π * psio / f4
            end
        end
    end
    flux = Spl.BicubicSpline(collect(rzphi.xs), collect(rzphi.ys), flux_fs; bctypex="extrap", bctypey="periodic")

    source = zeros(Float64, mpsi + 1, mtheta + 1)
    for ipsi in 0:mpsi
        for itheta in 0:mtheta
            rz_eval = Spl.bicube_eval!(rzphi, rzphi.xs[ipsi+1], rzphi.ys[itheta+1])
            f4 = rz_eval[4]
            psi_val = equil.psi_grid[ipsi+1]
            # Evaluate F and P derivatives at this psi
            s1 = equil.F_values[ipsi+1]  # F*2π value
            s1p = (BSplineKit.Derivative(1) * equil.F_spline)(psi_val)  # F derivative
            s2p = (BSplineKit.Derivative(1) * equil.P_spline)(psi_val)  # P derivative

            denom = (2π * r[ipsi+1, itheta+1])^2
            source[ipsi+1, itheta+1] = f4 / (2π * psio * π^2) * (s1 * s1p / denom + s2p)
        end
    end

    total = flux.fsx[:, :, 1] .- flux.fsy[:, :, 2] .+ source
    error = abs.(total) ./ maximum([maximum(abs.(flux.fsx[:, :, 1])), maximum(abs.(flux.fsy[:, :, 2])), maximum(abs.(source))])
    errlog = ifelse.(error .> 0, log10.(error), 0.0)

    if diagnose_maxima
        fxmax = maximum(abs.(flux.fsx[:, :, 1]))
        fymax = maximum(abs.(flux.fsy[:, :, 2]))
        smax = maximum(abs.(source))
        emax = maximum(abs.(error))
        lmax = maximum(errlog)
        jmax = ind2sub(size(errlog), argmax(errlog))
        println(" fxmax = $fxmax, fymax = $fymax, smax = $smax")
        println(" emax = $emax, lmax = $lmax, maxloc = ", jmax .- 1)
    end

    # Integrated error criterion
    term = zeros(Float64, mpsi + 1, 2)
    for ipsi in 0:mpsi
        fs_matrix = zeros(Float64, mtheta + 1, 2)
        fs_matrix[:, 1] = flux.fsx[ipsi+1, :, 1]
        fs_matrix[:, 2] = source[ipsi+1, :]

        spline = Spl.CubicSpline(Vector(flux.ys), fs_matrix; bctype="periodic")
        Spl.spline_integrate!(spline)

        term[ipsi+1, :] .= spline.fsi[end, :]
        # spline will be automatically deallocated by finalizer
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
            file["flux_fsx"] = Float32.(flux.fsx[:, :, 1])
            file["flux_fsy"] = Float32.(flux.fsy[:, :, 2])
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
                    gse_data[ipsi+1, itheta+1, 1] = Float32(flux.ys[itheta+1])
                    gse_data[ipsi+1, itheta+1, 2] = Float32(flux.xs[ipsi+1])
                    gse_data[ipsi+1, itheta+1, 3] = Float32(flux.fs[ipsi+1, itheta+1, 1])
                    gse_data[ipsi+1, itheta+1, 4] = Float32(flux.fs[ipsi+1, itheta+1, 2])
                    gse_data[ipsi+1, itheta+1, 5] = Float32(source[ipsi+1, itheta+1])
                    gse_data[ipsi+1, itheta+1, 6] = Float32(total[ipsi+1, itheta+1])
                    gse_data[ipsi+1, itheta+1, 7] = Float32(error[ipsi+1, itheta+1])
                end
            end
            file["gse_data"] = gse_data
        end

        # Write integrated error criterion
        h5open(joinpath(dirname(equil.config.control.eq_filename), "gsei.h5"), "w") do file
            file["xs"] = Float32.(flux.xs)
            file["term"] = Float32.(term)
            file["totali"] = Float32.(totali)
            file["errori"] = Float32.(errori)
            file["errlogi"] = Float32.(errlogi)
        end
    end
end

end # module Equilibrium