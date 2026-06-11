using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using GeneralizedPerturbedEquilibrium, Plots
using GeneralizedPerturbedEquilibrium: Analysis
using GeneralizedPerturbedEquilibrium: Equilibrium, ForceFreeStates
using LaTeXStrings, TOML
isinteractive() ? plotlyjs() : gr()

h5path = joinpath(@__DIR__, "gpec.h5")

# Summary plots
p_eq = Analysis.Equilibrium.plot_equilibrium_summary(h5path)
p_ffs = Analysis.ForceFreeStates.plot_ffs_summary(h5path)
p_pe = Analysis.PerturbedEquilibrium.plot_perturbed_equilibrium_summary(h5path)

display(p_eq);
Plots.savefig(p_eq, joinpath(@__DIR__, "equilibrium_summary.png"))
display(p_ffs);
Plots.savefig(p_ffs, joinpath(@__DIR__, "ffs_summary.png"))
display(p_pe);
Plots.savefig(p_pe, joinpath(@__DIR__, "pe_summary.png"))

# ----------------------------------------------------------------------
# Local stability: s-alpha profiles, D_I / ballooning Δ', and 2D s-alpha
# maps (ported from the former Bal_salpha_delta_di_summary notebook).
# These recompute directly from the equilibrium via the Bal.jl helpers
# rather than reading gpec.h5, so the s-alpha scan can perturb (p', q').
# ----------------------------------------------------------------------
eq_config = Equilibrium.EquilibriumConfig(TOML.parsefile(joinpath(@__DIR__, "gpec.toml"))["Equilibrium"], @__DIR__)
equil = Equilibrium.setup_equilibrium(eq_config)
psi_norm = Vector(equil.profiles.xs)

# s and alpha profiles vs normalized flux
s_profile = fill(NaN, length(psi_norm))
alpha_profile = fill(NaN, length(psi_norm))
for i in eachindex(psi_norm)
    try
        ref = ForceFreeStates.salpha_reference(i, equil)
        s_profile[i] = ref.s_ref
        alpha_profile[i] = ref.alpha_ref
    catch err
        @warn "s-alpha reference failed" i psi = psi_norm[i] exception = (err, catch_backtrace())
    end
end

s_mask = isfinite.(s_profile)
alpha_mask = isfinite.(alpha_profile)
p_s = plot(
    psi_norm[s_mask],
    s_profile[s_mask];
    xlabel="Norm. Poloidal Flux",
    ylabel="s",
    title="Magnetic shear s",
    label="s",
    lw=2,
    framestyle=:box,
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm
)
p_alpha = plot(
    psi_norm[alpha_mask],
    alpha_profile[alpha_mask];
    xlabel="Norm. Poloidal Flux",
    ylabel=L"\alpha",
    title="Pressure gradient α",
    label=L"\alpha",
    lw=2,
    framestyle=:box,
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm
)
p_salpha = plot(p_s, p_alpha; layout=(1, 2), size=(1100, 420))
display(p_salpha)
salpha_path = joinpath(@__DIR__, "salpha_profiles.png")
Plots.savefig(p_salpha, salpha_path)

# D_I (Mercier) and ballooning Δ' profiles
locstab_fs = zeros(length(psi_norm), 5)
ctrl = ForceFreeStates.ForceFreeStatesControl(; verbose=false)
ForceFreeStates.compute_ballooning_stability!(ctrl, locstab_fs, equil; compute_delta_prime=true)

delta_prime = Vector(locstab_fs[:, 4])
di_profile = fill(NaN, length(psi_norm))
psi_mask = abs.(psi_norm) .> eps(Float64)
di_profile[psi_mask] .= locstab_fs[psi_mask, 1] ./ psi_norm[psi_mask]

delta_mask = isfinite.(delta_prime)
di_mask = isfinite.(di_profile)
p_delta = plot(
    psi_norm[delta_mask],
    delta_prime[delta_mask];
    xlabel="Norm. Poloidal Flux",
    ylabel=L"\Delta'",
    title="Ballooning Δ'",
    label=L"\Delta'",
    lw=2,
    framestyle=:box,
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm
)
p_di = plot(
    psi_norm[di_mask],
    di_profile[di_mask];
    xlabel="Norm. Poloidal Flux",
    ylabel=L"D_I",
    title="Mercier D_I",
    label=L"D_I",
    lw=2,
    framestyle=:box,
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm
)
hline!(p_di, [0.0]; linestyle=:dash, color=:black, label=nothing)
p_localstab = plot(p_delta, p_di; layout=(1, 2), size=(1100, 420))
display(p_localstab)
localstab_path = joinpath(@__DIR__, "delta_di_profiles.png")
Plots.savefig(p_localstab, localstab_path)

# 2D s-alpha maps of Δ' and D_I at a near-edge surface, with zero contours
psi_target = 0.974
psi_idx_scan = argmin(abs.(psi_norm .- psi_target))
s_scales = collect(range(-5.0, 5.0; length=30))
alpha_scales = collect(range(-5.0, 5.0; length=30))
scan = ForceFreeStates.scan_delta_prime_map(psi_idx_scan, equil; theta_k=0.0, s_scales=s_scales, alpha_scales=alpha_scales)

scan_qprime = scan.dqdpsi_ref .* scan.s_scales
scan_pprime = scan.pprime_ref .* scan.alpha_scales
qorder = sortperm(scan_qprime)
porder = sortperm(scan_pprime)
scan_qprime = scan_qprime[qorder]
scan_pprime = scan_pprime[porder]
scan_delta = scan.delta_prime[qorder, porder]
scan_di = scan.di_values[qorder, porder]

function symmetric_clims(z)
    finite_vals = vec(z)[isfinite.(vec(z))]
    maxabs = isempty(finite_vals) ? 1.0 : maximum(abs.(finite_vals))
    maxabs = max(maxabs, eps(Float64))
    return (-maxabs, maxabs)
end

p_map_delta = contourf(
    scan_pprime,
    scan_qprime,
    scan_delta;
    c=cgrad([:blue, :white, :red]),
    clims=symmetric_clims(scan_delta),
    xlabel=L"p_{\psi}",
    ylabel=L"q_{\psi}",
    title=L"\Delta'",
    levels=41,
    framestyle=:box,
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm
)
contour!(p_map_delta, scan_pprime, scan_qprime, scan_delta; levels=[0.0], color=:black, linewidth=2)
scatter!(p_map_delta, [scan.reference.pprime_ref], [scan.reference.qprime_ref]; color=:green, marker=:star5, ms=7, label="equilibrium")
p_map_di = contourf(
    scan_pprime,
    scan_qprime,
    scan_di;
    c=cgrad([:blue, :white, :red]),
    clims=symmetric_clims(scan_di),
    xlabel=L"p_{\psi}",
    ylabel=L"q_{\psi}",
    title=L"D_I",
    levels=41,
    framestyle=:box,
    left_margin=10Plots.mm,
    bottom_margin=5Plots.mm
)
contour!(p_map_di, scan_pprime, scan_qprime, scan_di; levels=[0.0], color=:black, linewidth=2)
scatter!(p_map_di, [scan.reference.pprime_ref], [scan.reference.qprime_ref]; color=:green, marker=:star5, ms=7, label="equilibrium")
p_maps = plot(p_map_delta, p_map_di; layout=(1, 2), size=(1150, 460))
display(p_maps)
maps_path = joinpath(@__DIR__, "salpha_maps.png")
Plots.savefig(p_maps, maps_path)

# Overlay of Δ'=0 (ballooning) and D_I=0 (Mercier) stability boundaries
p_zero = plot(; xlabel=L"p_{\psi}", ylabel=L"q_{\psi}", title="Ballooning / Mercier stability boundaries", framestyle=:box, left_margin=10Plots.mm, bottom_margin=5Plots.mm)
contour!(p_zero, scan_pprime, scan_qprime, scan_delta; levels=[0.0], color=:blue, linewidth=3, colorbar=false)
contour!(p_zero, scan_pprime, scan_qprime, scan_di; levels=[0.0], color=:red, linewidth=3, linestyle=:dash, colorbar=false)
scatter!(p_zero, [scan.reference.pprime_ref], [scan.reference.qprime_ref]; color=:green, marker=:star5, ms=8, label="equilibrium")
display(p_zero)
zero_path = joinpath(@__DIR__, "salpha_zero_contours.png")
Plots.savefig(p_zero, zero_path)

# BALOO-style first ballooning stability boundary: experimental α vs critical α
p_baloo = Analysis.ForceFreeStates.plot_ballooning_alpha_boundary(h5path)
display(p_baloo)
baloo_path = joinpath(@__DIR__, "ballooning_alpha_boundary.png")
Plots.savefig(p_baloo, baloo_path)

println("Saved figures:")
for p in (
    joinpath(@__DIR__, "equilibrium_summary.png"),
    joinpath(@__DIR__, "ffs_summary.png"),
    joinpath(@__DIR__, "pe_summary.png"),
    salpha_path,
    localstab_path,
    maps_path,
    zero_path,
    baloo_path
)
    println("  ", p)
end
