using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
# The figures need a Makie backend from your global environment: CairoMakie saves files, GLMakie opens windows.
using GeneralizedPerturbedEquilibrium, CairoMakie
using GeneralizedPerturbedEquilibrium: Analysis
using GeneralizedPerturbedEquilibrium: Equilibrium, ForceFreeStates, LocalStability
using TOML

h5path = joinpath(@__DIR__, "gpec.h5")
saved = String[]
# Show a figure and save it beside this script.
function show_and_save(fig, name)
    display(fig)
    path = joinpath(@__DIR__, name * ".png")
    save(path, fig)
    push!(saved, path)
    return fig
end

# Summary plots, and the BALOO-style first ballooning stability boundary: experimental α against critical α
show_and_save(Analysis.Equilibrium.plot_equilibrium_summary(h5path), "equilibrium_summary")
show_and_save(Analysis.ForceFreeStates.plot_ffs_summary(h5path), "ffs_summary")
show_and_save(Analysis.PerturbedEquilibrium.plot_perturbed_equilibrium_summary(h5path), "pe_summary")
show_and_save(Analysis.ForceFreeStates.plot_ballooning_alpha_boundary(h5path), "ballooning_alpha_boundary")

# ----------------------------------------------------------------------
# Local stability: s-alpha profiles, D_I / ballooning Δ', and 2D s-alpha
# maps (ported from the former Bal_salpha_delta_di_summary notebook).
# These recompute directly from the equilibrium via the LocalStability helpers
# rather than reading gpec.h5, so the s-alpha scan can perturb (p', q').
# ----------------------------------------------------------------------

# Colour limits symmetric about zero over the finite values.
function symmetric_limits(z)
    m = max(maximum(abs, filter(isfinite, z); init=0.0), eps())
    return (-m, m)
end

# Two profiles side by side against normalized poloidal flux, NaNs dropped.
function profile_pair(psi, left, right; zero_line=false)
    fig = Figure(; size=(1100, 420))
    for (col, (y, label, title)) in enumerate((left, right))
        ax = Axis(fig[1, col]; xlabel="normalized poloidal flux ψ_N", ylabel=label, title)
        ok = isfinite.(y)
        lines!(ax, psi[ok], y[ok]; linewidth=2)
        zero_line && col == 2 && hlines!(ax, 0.0; color=:black, linestyle=:dash)
    end
    return fig
end

geqdsk_cases = [
    "TkMkr_D3Dlike_Hmode.geqdsk"
]

base_toml = TOML.parsefile(joinpath(@__DIR__, "gpec.toml"))

for geqdsk_file in geqdsk_cases
    case_label = splitext(basename(geqdsk_file))[1]
    println("\n=== Local stability: $case_label ===")

    eq_dict = deepcopy(base_toml["Equilibrium"])
    eq_dict["eq_filename"] = geqdsk_file
    eq_config = GeneralizedPerturbedEquilibrium.equilibrium_config(Dict{String,Any}("Equilibrium" => eq_dict), @__DIR__)
    equil = Equilibrium.PlasmaEquilibrium(eq_config)
    psi_norm = Vector(equil.profiles.xs)

    # s and alpha profiles vs normalized flux
    s_profile = fill(NaN, length(psi_norm))
    alpha_profile = fill(NaN, length(psi_norm))
    for i in eachindex(psi_norm)
        try
            ref = LocalStability.salpha_reference(i, equil)
            s_profile[i] = ref.s_ref
            alpha_profile[i] = ref.alpha_ref
        catch err
            @warn "s-alpha reference failed" i psi = psi_norm[i] exception = (err, catch_backtrace())
        end
    end
    show_and_save(profile_pair(psi_norm, (s_profile, "s", "Magnetic shear s"), (alpha_profile, "α", "Pressure gradient α")), "salpha_profiles_$(case_label)")

    # D_I (Mercier) and ballooning Δ' profiles
    locstab_fs = zeros(length(psi_norm), 5)
    LocalStability.compute_ballooning_stability!(locstab_fs, equil; compute_delta_prime=true)
    delta_prime = Vector(locstab_fs[:, 4])
    di_profile = fill(NaN, length(psi_norm))
    psi_mask = abs.(psi_norm) .> eps(Float64)
    di_profile[psi_mask] .= locstab_fs[psi_mask, 1] ./ psi_norm[psi_mask]
    show_and_save(profile_pair(psi_norm, (delta_prime, "Δ′", "Ballooning Δ′"), (di_profile, "D_I", "Mercier D_I"); zero_line=true), "delta_di_profiles_$(case_label)")

    # 2D s-alpha maps of Δ' and D_I at a near-edge surface, with zero contours
    psi_target = 0.974
    psi_idx_scan = argmin(abs.(psi_norm .- psi_target))
    s_scales = collect(range(-5.0, 5.0; length=30))
    alpha_scales = collect(range(-5.0, 5.0; length=30))
    scan = LocalStability.scan_delta_prime_map(psi_idx_scan, equil; theta_k=0.0, s_scales=s_scales, alpha_scales=alpha_scales)

    scan_qprime = scan.dqdpsi_ref .* scan.s_scales
    scan_pprime = scan.pprime_ref .* scan.alpha_scales
    qorder = sortperm(scan_qprime)
    porder = sortperm(scan_pprime)
    scan_qprime = scan_qprime[qorder]
    scan_pprime = scan_pprime[porder]
    # Rows of the scan follow q′ and columns p′; Makie takes z[x, y], so transpose to put p′ on x.
    scan_delta = permutedims(scan.delta_prime[qorder, porder])
    scan_di = permutedims(scan.di_values[qorder, porder])
    eq_point = ([scan.reference.pprime_ref], [scan.reference.qprime_ref])

    fig = Figure(; size=(1150, 460))
    for (col, (z, title)) in enumerate(((scan_delta, "Δ′"), (scan_di, "D_I")))
        ax = Axis(fig[1, 2col-1]; xlabel="p_ψ", ylabel="q_ψ", title)
        cf = contourf!(ax, scan_pprime, scan_qprime, z; levels=range(symmetric_limits(z)...; length=41), colormap=Reverse(:RdBu))
        contour!(ax, scan_pprime, scan_qprime, z; levels=[0.0], color=:black, linewidth=2)
        scatter!(ax, eq_point...; color=:green, marker=:star5, markersize=14)
        Colorbar(fig[1, 2col], cf)
    end
    show_and_save(fig, "salpha_maps_$(case_label)")

    # Overlay of Δ'=0 (ballooning) and D_I=0 (Mercier) stability boundaries
    fig = Figure(; size=(700, 520))
    ax = Axis(fig[1, 1]; xlabel="p_ψ", ylabel="q_ψ", title="Ballooning / Mercier stability boundaries")
    contour!(ax, scan_pprime, scan_qprime, scan_delta; levels=[0.0], color=:blue, linewidth=3)
    contour!(ax, scan_pprime, scan_qprime, scan_di; levels=[0.0], color=:red, linewidth=3, linestyle=:dash)
    scatter!(ax, eq_point...; color=:green, marker=:star5, markersize=16)
    axislegend(ax, [LineElement(; color=:blue, linewidth=3), LineElement(; color=:red, linewidth=3, linestyle=:dash),
            MarkerElement(; color=:green, marker=:star5, markersize=16)], ["Δ′ = 0", "D_I = 0", "equilibrium"])
    show_and_save(fig, "salpha_zero_contours_$(case_label)")

    # BALOO-style α vs ψ stability boundaries (1st and 2nd) over the Δ'(ψ, α) map
    bnd = LocalStability.ballooning_alpha_boundaries(equil)
    dpmap = LocalStability.ballooning_delta_prime_map(equil)
    show_and_save(Analysis.ForceFreeStates.plot_ballooning_alpha_boundaries(bnd, dpmap), "alpha_stability_boundary_$(case_label)")
end

println("Saved figures:")
foreach(p -> println("  ", p), saved)
