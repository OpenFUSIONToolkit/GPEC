"""
    equilibrium_gse!(equil::PlasmaEquilibrium)

Diagnoses the Grad-Shafranov solution by computing the residual of the
Grad-Shafranov equation across the grid and writing diagnostic data to HDF5 files.
Performs the same function as equil_out_gse in the Fortran code.

Returns `(; xs, ys, flux_x, flux_y, source, total, error, errori)` on the `rzphi` (ψ, θ) nodes: the two
flux-divergence terms, the source, the residual `total = flux_x - flux_y + source`, `error = |total|` over
the largest term on the grid, and `errori`, the magnitude of the θ-integrated residual on each surface.
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
        @info "Diagnosing Grad-Shafranov solution"
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
    @views flux_fs[:, end, :] .= flux_fs[:, 1, :]
    flux_opts = (bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))
    flux1 = cubic_interp((equil.rzphi_xs, equil.rzphi_ys), flux_fs[:, :, 1]; flux_opts...)
    flux2 = cubic_interp((equil.rzphi_xs, equil.rzphi_ys), flux_fs[:, :, 2]; flux_opts...)

    # Compute flux derivatives at all grid points for diagnostics
    hint2d = (Ref(1), Ref(1))  # Shared 2D hint for hot loop optimization
    for ipsi in 0:mpsi
        for itheta in 0:mtheta
            query_point = (equil.rzphi_xs[ipsi+1], equil.rzphi_ys[itheta+1])
            flux_fsx[ipsi+1, itheta+1, 1] = flux1(query_point; deriv=DerivOp(1, 0), hint=hint2d)
            flux_fsx[ipsi+1, itheta+1, 2] = flux2(query_point; deriv=DerivOp(1, 0), hint=hint2d)
            flux_fsy[ipsi+1, itheta+1, 1] = flux1(query_point; deriv=DerivOp(0, 1), hint=hint2d)
            flux_fsy[ipsi+1, itheta+1, 2] = flux2(query_point; deriv=DerivOp(0, 1), hint=hint2d)
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
        @info "GS residuals: fxmax = $(@sprintf("%.3e", fxmax)), fymax = $(@sprintf("%.3e", fymax)), smax = $(@sprintf("%.3e", smax)), emax = $(@sprintf("%.3e", emax)), lmax = $(@sprintf("%.3f", lmax)), maxloc = $(jmax .- 1)"
    end

    # Integrated error criterion
    term = zeros(Float64, mpsi + 1, 2)
    for ipsi in 1:(mpsi+1)
        fs_matrix = zeros(Float64, mtheta + 1, 2)
        fs_matrix[:, 1] = flux_fsx[ipsi, :, 1]
        fs_matrix[:, 2] = source[ipsi, :]
        # Snap the repeated endpoint exactly equal to the start
        fs_matrix[end, :] .= fs_matrix[1, :]

        # Compute total integral using FastInterpolations native integration
        itp = cubic_interp(equil.rzphi_ys, Series(fs_matrix); bc=PeriodicBC())
        term[ipsi, :] .= FastInterpolations.integrate(itp)
    end

    totali = sum(term; dims=2)
    errori = abs.(totali)
    errlogi = @. ifelse(errori > 0, log10(errori), 0.0)

    if diagnose_src
        if verbose
            @info "Writing diagnostics to HDF5 files"
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
                    gse_data[ipsi+1, itheta+1, 1] = Float32(equil.rzphi_ys[itheta+1])
                    gse_data[ipsi+1, itheta+1, 2] = Float32(equil.rzphi_xs[ipsi+1])
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
            file["xs"] = Float32.(equil.rzphi_xs)
            file["term"] = Float32.(term)
            file["totali"] = Float32.(totali)
            file["errori"] = Float32.(errori)
            file["errlogi"] = Float32.(errlogi)
        end
    end
    return (; xs=equil.rzphi_xs, ys=equil.rzphi_ys, flux_x=flux_fsx[:, :, 1], flux_y=flux_fsy[:, :, 2], source, total, error, errori=vec(errori))
end
