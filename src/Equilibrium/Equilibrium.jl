module Equilibrium

# --- Module-level Dependencies ---

using Printf, OrdinaryDiffEq, DiffEqCallbacks, LinearAlgebra, HDF5
using TOML
import FastInterpolations
using FastInterpolations: cubic_interp, deriv1, deriv2, deriv3, LinearBinary, CubicFit, PeriodicBC, AbstractExtrap, ExtendExtrap, WrapExtrap, n_series
using AdaptiveArrayPools
import StaticArrays: @MMatrix, SVector

# --- Internal Module Structure ---
include("EquilibriumTypes.jl")
include("ReadEquilibrium.jl")
include("DirectEquilibrium.jl")
include("InverseEquilibrium.jl")
include("AnalyticEquilibrium.jl")

# --- Expose types and functions to the user ---
export setup_equilibrium, EquilibriumConfig, PlasmaEquilibrium, EquilibriumParameters, ProfileSplines

# --- Constants ---
const mu0 = 4π * 1e-7

"""
    setup_equilibrium(eq_config::EquilibriumConfig)

The main public API for the `Equilibrium` module. It orchestrates the entire
process of reading an equilibrium file, running the appropriate solver, and
returning the final processed `PlasmaEquilibrium` object.

## Arguments:

  - `eq_config`: An `EquilibriumConfig` object containing all necessary setup parameters.

## Returns:

  - A `PlasmaEquilibrium` object containing the final result.
"""
function setup_equilibrium(path::String="equil.toml")
    return setup_equilibrium(EquilibriumConfig(path))
end
function setup_equilibrium(eq_config::EquilibriumConfig, additional_input=nothing)

    @printf "Equilibrium file: %s\n" eq_config.eq_filename

    eq_type = eq_config.eq_type
    # Parse file and prepare initial data structures and splines
    if eq_type == "efit"
        eq_input = read_efit(eq_config)
    elseif eq_type in ["chease2", "chease_ascii"]
        eq_input = read_chease_ascii(eq_config)
    elseif eq_type in ["chease", "chease_binary"]
        eq_input = read_chease_binary(eq_config)
    elseif eq_type == "lar"

        if additional_input === nothing
            additional_input = LargeAspectRatioConfig(eq_config.eq_filename)
        end

        eq_input = lar_run(eq_config, additional_input)
    elseif eq_type == "sol"

        if additional_input === nothing
            additional_input = SolovevConfig(eq_config.eq_filename)
        end

        eq_input = sol_run(eq_config, additional_input)
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
    mpsi = length(pe.rzphi_xs) - 1
    mtheta = length(pe.rzphi_ys) - 1

    # Allocate vector to store eta offset from rzphi (direct array access at grid points)
    vector = pe.rzphi_ys .+ @view pe.rzphi_offset.nodal_derivs.partials[1, end, :]

    edge_idx = mpsi + 1  # Edge flux surface index
    psi_edge = pe.rzphi_xs[edge_idx]
    eta0 = 0.0
    idx = findmin(abs.(vector .- eta0))[2]
    theta = pe.rzphi_ys[idx]
    rsep = zeros(2)

    for iside in 1:2
        it = 0
        hint2d = (Ref(1), Ref(1))  # Shared 2D hint for Newton iteration
        while true
            it += 1
            # Evaluate offset and its derivative
            offset = pe.rzphi_offset((psi_edge, theta); hint=hint2d)
            offset_y = pe.rzphi_offset((psi_edge, theta); deriv=Val((0, 1)), hint=hint2d)

            eta = theta + offset - eta0
            eta_theta = 1 + offset_y
            dtheta = -eta / eta_theta
            theta += dtheta
            if abs(eta) <= 1e-10 || it > 100
                break
            end
        end
        r2 = pe.rzphi_rsquared((psi_edge, theta))
        offset = pe.rzphi_offset((psi_edge, theta))
        rsep[iside] = pe.ro + sqrt(r2) * cos(2π * (theta + offset))
        eta0 = 0.5
        idx = findmin(abs.(vector .- eta0))[2]
        theta = pe.rzphi_ys[idx]
    end

    # Top and bottom separatrix locations using Newton iteration
    zsep = zeros(2)
    rext = zeros(2)
    zext = zeros(2)

    for iside in 1:2
        eta0 = (iside == 1) ? 0.0 : 0.5
        idx = findmin(abs.(vector .- eta0))[2]
        theta = pe.rzphi_ys[idx]
        rfac = 0.0
        cosfac = 0.0
        z = 0.0
        max_iter = 1000
        iter = 0
        hint2d = (Ref(1), Ref(1))  # Shared 2D hint for Newton iteration
        while iter < max_iter
            iter += 1
            # Evaluate rcoord and offset with derivatives
            r2 = pe.rzphi_rsquared((psi_edge, theta); hint=hint2d)
            r2y = pe.rzphi_rsquared((psi_edge, theta); deriv=Val((0, 1)), hint=hint2d)
            r2yy = pe.rzphi_rsquared((psi_edge, theta); deriv=Val((0, 2)), hint=hint2d)
            eta = pe.rzphi_offset((psi_edge, theta); hint=hint2d)
            eta1 = pe.rzphi_offset((psi_edge, theta); deriv=Val((0, 1)), hint=hint2d)
            eta2 = pe.rzphi_offset((psi_edge, theta); deriv=Val((0, 2)), hint=hint2d)

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
            # Wrap theta back into valid periodic range [0, 2π)
            y_period = pe.rzphi_ys[end] - pe.rzphi_ys[1] + (pe.rzphi_ys[2] - pe.rzphi_ys[1])
            theta = mod(theta - pe.rzphi_ys[1], y_period) + pe.rzphi_ys[1]
            if abs(dtheta) < 1e-12 * y_period
                break
            end
        end
        if iter >= max_iter
            @warn "Newton iteration for separatrix extrema did not converge" iside eta0 theta
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
    profiles = pe.profiles
    mpsi = length(pe.rzphi_xs) - 1
    mtheta = length(pe.rzphi_ys) - 1

    # Use separatrix geometry
    rsep, zsep, rext, _ = equilibrium_separatrix_find!(pe)

    rmean = (rsep[2] + rsep[1]) / 2
    amean = (rsep[2] - rsep[1]) / 2
    aratio = rmean / amean
    kappa = (zsep[1] - zsep[2]) / (rsep[2] - rsep[1])
    delta1 = (rmean - rext[1]) / amean
    delta2 = (rmean - rext[2]) / amean
    dpsi = 1.0 - pe.rzphi_xs[mpsi+1]
    psi_edge = profiles.xs[end]
    bt0 = (profiles.F_spline.y[end] + profiles.F_deriv(psi_edge; hint=Ref(profiles.npts_minus_1)) * dpsi) / (2π * rmean)

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

    # Direct array access at edge flux surface grid points using nodal_derivs
    for itheta in 0:mtheta
        # Function values (partials[1,:,:])
        r2 = pe.rzphi_rsquared.nodal_derivs.partials[1, end, itheta+1]
        offset = pe.rzphi_offset.nodal_derivs.partials[1, end, itheta+1]
        jac = pe.rzphi_jac.nodal_derivs.partials[1, end, itheta+1]
        # Theta derivatives (partials[3,:,:] = ∂f/∂y)
        r2_y = pe.rzphi_rsquared.nodal_derivs.partials[3, end, itheta+1]
        offset_y = pe.rzphi_offset.nodal_derivs.partials[3, end, itheta+1]

        chi1 = 2π * psio / jac
        jacfac = π / jac
        rfac = sqrt(r2)
        eta = 2π * (pe.rzphi_ys[itheta+1] + offset)
        r = pe.ro + rfac * cos(eta)
        v21 = jacfac * r2_y / (2π * rfac)
        v22 = jacfac * (1 + offset_y) * (2 * rfac)
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

    # Flux surface integrals (using profiles)
    P_vals = profiles.P_spline.y
    dVdpsi_vals = profiles.dVdpsi_spline.y
    hs1 = P_vals .* dVdpsi_vals                   # p * dV/dpsi
    hs2 = dVdpsi_vals                             # dV/dpsi
    hs3 = P_vals .^ 2 .* dVdpsi_vals              # p^2 * dV/dpsi

    dpsi_vec = diff(profiles.xs)
    fsi1 = sum((hs1[1:(end-1)] .+ hs1[2:end]) .* dpsi_vec) / 2
    fsi2 = sum((hs2[1:(end-1)] .+ hs2[2:end]) .* dpsi_vec) / 2
    fsi3 = sum((hs3[1:(end-1)] .+ hs3[2:end]) .* dpsi_vec) / 2

    volume = sum((dVdpsi_vals[1:(end-1)] .+ dVdpsi_vals[2:end]) .* dpsi_vec) / 2

    p0 = P_vals[1] - profiles.P_deriv(profiles.xs[1]; hint=Ref(1)) * profiles.xs[1]  # linear extrapolation
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

    pe.params.q0 = profiles.q_spline.y[1]
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

    profiles = equil.profiles
    xs = profiles.xs
    mpsi = length(xs) - 1
    psiexl = Float64[]
    qexl = Float64[]

    # Create derivative views for q-spline
    q_spline = profiles.q_spline
    q_d1 = deriv1(q_spline)
    q_d2 = deriv2(q_spline)
    q_d3 = deriv3(q_spline)

    # Left endpoint
    push!(psiexl, xs[1])
    push!(qexl, q_spline.y[1])

    # Search for extrema in q(ψ)
    for ipsi in 1:mpsi
        x0 = xs[ipsi]
        x1 = xs[ipsi+1]
        xmax = x1 - x0

        a = q_spline(x0)
        b = q_d1(x0)
        c = q_d2(x0)
        d = q_d3(x0)

        if d != 0.0
            xcrit = -c / d
            dx2 = xcrit^2 - 2b / d
            if dx2 ≥ 0
                dx = sqrt(dx2)
                for delta in (dx, -dx)
                    x = xcrit - delta
                    if 0 ≤ x < xmax
                        ψ = x0 + x
                        push!(psiexl, ψ)
                        push!(qexl, q_spline(ψ))
                    end
                end
            end
        end
    end

    # Right endpoint
    push!(psiexl, xs[end])
    push!(qexl, q_spline.y[end])

    equil.params.qextrema_psi = psiexl
    equil.params.qextrema_q = qexl
    equil.params.mextrema = length(psiexl)
    # Compute derived q-values
    q0 = q_spline.y[1] - profiles.q_deriv(xs[1]; hint=Ref(1)) * xs[1]
    qmax_edge = q_spline.y[end]
    qmin = min(minimum(qexl), q0)
    qmax = max(maximum(qexl), qmax_edge)
    qa = q_spline.y[end] + profiles.q_deriv(xs[end]; hint=Ref(profiles.npts_minus_1)) * (1.0 - xs[end])

    q95 = q_spline(0.95)

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

    profiles = equil.profiles
    mpsi = length(equil.rzphi_xs) - 1
    mtheta = length(equil.rzphi_ys) - 1
    ro, zo = equil.ro, equil.zo
    psio = equil.psio
    verbose = equil.params.verbose
    diagnose_src = equil.params.diagnose_src
    diagnose_maxima = equil.params.diagnose_maxima

    if verbose
        println("Diagnosing Grad-Shafranov solution...")
    end

    # Compute R, Z coordinates using nodal_derivs access
    r = zeros(Float64, mpsi + 1, mtheta + 1)
    z = zeros(Float64, mpsi + 1, mtheta + 1)

    for ipsi in 1:(mpsi+1)
        rfac = @. sqrt(equil.rzphi_rsquared.nodal_derivs.partials[1, ipsi, :])
        angle = @. 2π * (equil.rzphi_ys + equil.rzphi_offset.nodal_derivs.partials[1, ipsi, :])
        r[ipsi, :] .= ro .+ rfac .* cos.(angle)
        z[ipsi, :] .= zo .+ rfac .* sin.(angle)
    end

    # Compute flux quantities using direct array access at grid points
    flux_fs = zeros(Float64, mpsi + 1, mtheta + 1, 2)
    flux_fsx = zeros(Float64, mpsi + 1, mtheta + 1, 2)  # x-derivatives
    flux_fsy = zeros(Float64, mpsi + 1, mtheta + 1, 2)  # y-derivatives
    for ipsi in 1:(mpsi+1)
        for itheta in 1:(mtheta+1)
            f1 = equil.rzphi_rsquared.nodal_derivs.partials[1, ipsi, itheta]
            f2 = equil.rzphi_offset.nodal_derivs.partials[1, ipsi, itheta]
            f4 = equil.rzphi_jac.nodal_derivs.partials[1, ipsi, itheta]
            fy1 = equil.rzphi_rsquared.nodal_derivs.partials[3, ipsi, itheta]
            fy2 = equil.rzphi_offset.nodal_derivs.partials[3, ipsi, itheta]
            fx1 = equil.rzphi_rsquared.nodal_derivs.partials[2, ipsi, itheta]
            fx2 = equil.rzphi_offset.nodal_derivs.partials[2, ipsi, itheta]

            flux_fs[ipsi, itheta, 1] = fy1^2 / (4π^2 * f1) + (1 + fy2)^2 * 4 * f1
            flux_fs[ipsi, itheta, 2] = fx1 * fy1 / (4π^2 * f1) + fx2 * (1 + fy2) * 4 * f1

            flux_fs[ipsi, itheta, 1] *= 2π * psio / f4
            flux_fs[ipsi, itheta, 2] *= 2π * psio / f4
        end
    end
    # Create flux interpolants for Grad-Shafranov diagnostics
    flux_opts = (search=LinearBinary(), bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))
    flux1 = cubic_interp((equil.rzphi_xs, equil.rzphi_ys), flux_fs[:, :, 1]; flux_opts...)
    flux2 = cubic_interp((equil.rzphi_xs, equil.rzphi_ys), flux_fs[:, :, 2]; flux_opts...)

    # Compute flux derivatives at all grid points for diagnostics
    hint2d = (Ref(1), Ref(1))  # Shared 2D hint for hot loop optimization
    for ipsi in 0:mpsi
        for itheta in 0:mtheta
            query_point = (equil.rzphi_xs[ipsi+1], equil.rzphi_ys[itheta+1])
            flux_fsx[ipsi+1, itheta+1, 1] = flux1(query_point; deriv=Val((1, 0)), hint=hint2d)
            flux_fsx[ipsi+1, itheta+1, 2] = flux2(query_point; deriv=Val((1, 0)), hint=hint2d)
            flux_fsy[ipsi+1, itheta+1, 1] = flux1(query_point; deriv=Val((0, 1)), hint=hint2d)
            flux_fsy[ipsi+1, itheta+1, 2] = flux2(query_point; deriv=Val((0, 1)), hint=hint2d)
        end
    end

    # Compute source term using direct array access
    source = zeros(Float64, mpsi + 1, mtheta + 1)
    hint = Ref(1)  # Linear search hint for sequential psi access
    for ipsi in 1:(mpsi+1)
        psi = profiles.xs[ipsi]
        s1 = profiles.F_spline.y[ipsi]
        s1p = profiles.F_deriv(psi; hint=hint)
        s2p = profiles.P_deriv(psi; hint=hint)
        for itheta in 1:(mtheta+1)
            f4 = equil.rzphi_jac.nodal_derivs.partials[1, ipsi, itheta]
            denom = (2π * r[ipsi, itheta])^2
            source[ipsi, itheta] = f4 / (2π * psio * π^2) * (s1 * s1p / denom + s2p)
        end
    end

    total = flux_fsx[:, :, 1] .- flux_fsy[:, :, 2] .+ source
    error = abs.(total) ./ maximum([maximum(abs.(flux_fsx[:, :, 1])), maximum(abs.(flux_fsy[:, :, 2])), maximum(abs.(source))])
    errlog = ifelse.(error .> 0, log10.(error), 0.0)

    if diagnose_maxima
        fxmax = maximum(abs.(flux_fsx[:, :, 1]))
        fymax = maximum(abs.(flux_fsy[:, :, 2]))
        smax = maximum(abs.(source))
        emax = maximum(abs.(error))
        lmax = maximum(errlog)
        jmax = ind2sub(size(errlog), argmax(errlog))
        println(" fxmax = $fxmax, fymax = $fymax, smax = $smax")
        println(" emax = $emax, lmax = $lmax, maxloc = ", jmax .- 1)
    end

    # Integrated error criterion
    term = zeros(Float64, mpsi + 1, 2)
    for ipsi in 1:(mpsi+1)
        fs_matrix = zeros(Float64, mtheta + 1, 2)
        fs_matrix[:, 1] = flux_fsx[ipsi, :, 1]
        fs_matrix[:, 2] = source[ipsi, :]

        # Compute total integral using FastInterpolations native integration
        itp = cubic_interp(equil.rzphi_ys, fs_matrix; bc=PeriodicBC())
        term[ipsi, :] .= FastInterpolations.integrate(itp)
    end

    totali = sum(term; dims=2)
    errori = abs.(totali)
    errlogi = @. ifelse(errori > 0, log10(errori), 0.0)

    if diagnose_src
        if verbose
            println("Writing diagnostics to HDF5 files...")
        end

        # Write contour data
        h5open(joinpath(dirname(equil.config.eq_filename), "gsec.h5"), "w") do file
            file["mpsi"] = mpsi
            file["mtheta"] = mtheta
            file["r"] = Float32.(r)
            file["z"] = Float32.(z)
            file["flux_fsx"] = Float32.(flux_fsx[:, :, 1])
            file["flux_fsy"] = Float32.(flux_fsy[:, :, 2])
            file["source"] = Float32.(source)
            file["total"] = Float32.(total)
            file["error"] = Float32.(error)
            file["errlog"] = Float32.(errlog)
        end

        # Write xy plot data
        h5open(joinpath(dirname(equil.config.eq_filename), "gse.h5"), "w") do file
            gse_data = Array{Float32,3}(undef, mpsi + 1, mtheta + 1, 7)
            for ipsi in 0:mpsi
                for itheta in 0:mtheta
                    gse_data[ipsi+1, itheta+1, 1] = Float32(flux.ys[itheta+1])
                    gse_data[ipsi+1, itheta+1, 2] = Float32(flux.xs[ipsi+1])
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
        h5open(joinpath(dirname(equil.config.eq_filename), "gsei.h5"), "w") do file
            file["xs"] = Float32.(flux.xs)
            file["term"] = Float32.(term)
            file["totali"] = Float32.(totali)
            file["errori"] = Float32.(errori)
            file["errlogi"] = Float32.(errlogi)
        end
    end
end

end # module Equilibrium
