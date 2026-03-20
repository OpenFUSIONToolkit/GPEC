"""
equil_spline_comparison.jl

Comparison of equilibrium descriptions produced by all three EFIT equilibrium
construction approaches, using "efit" as the reference.

Produces (saved to benchmarks/equil_spline_comparison_psihigh<value>/):
  profiles.png          — 4×3 panel: F, P, dV/dψ, q (full / core zoom / edge zoom)
  equilibrium_contours.png — 2×3 panel: flux surfaces + theta contours (full / core / x-point)
  rzphi_splines.png     — 4×3 panel: r², offset, ν, Jacobian at θ = 0, 0.25, 0.5, 0.75
  eqfun_splines.png     — 3×3 panel: |B|, metric₁, metric₂ at select θ values
  Printed numerical summaries of deep-core and far-edge differences for every spline.

Usage:
  julia --project=. benchmarks/equil_spline_comparison.jl [example_path] [psihigh]

psihigh values of interest:
  0.996  → poor et[1] agreement
  0.997  → ok   et[1] agreement
  0.999  → poor et[1] agreement again
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium
using TOML, Printf, Statistics, Plots

const EqPlot = Analysis.Equilibrium

# ─── CLI arguments ─────────────────────────────────────────────────────────────
example_path = length(ARGS) > 0 ? ARGS[1] :
    joinpath(@__DIR__, "../examples/DIIID-like_ideal_example")
psihigh_arg = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 0.997
config_path  = joinpath(example_path, "gpec.toml")

println("=" ^ 70)
println("Equilibrium Spline Comparison: efit / efit_arclength / efit_by_inversion")
println("Example  : $example_path")
println("psihigh  : $psihigh_arg")
println("=" ^ 70)

# ─── Output directory ──────────────────────────────────────────────────────────
psihigh_tag = replace(string(psihigh_arg), "." => "p")
outdir = joinpath(@__DIR__, "equil_spline_comparison_psihigh$(psihigh_tag)")
mkpath(outdir)
println("Output   : $outdir\n")

# ─── Config factory ────────────────────────────────────────────────────────────
function make_config(path::String, eq_type::String, psihigh::Float64)
    raw = TOML.parsefile(path)
    raw["Equilibrium"]["eq_type"]  = eq_type
    raw["Equilibrium"]["psihigh"] = psihigh
    return Equilibrium.EquilibriumConfig(raw["Equilibrium"], dirname(path))
end

# ─── Load all three equilibria ─────────────────────────────────────────────────
ref_method      = "efit"
compare_methods = ["efit_arclength", "efit_by_inversion"]
all_methods     = vcat(ref_method, compare_methods)

method_color = Dict(
    "efit"              => :blue,
    "efit_arclength"    => :darkorange,
    "efit_by_inversion" => :red,
)
method_style = Dict(
    "efit"              => :solid,
    "efit_arclength"    => :dash,
    "efit_by_inversion" => :dot,
)
method_label = Dict(
    "efit"              => "efit",
    "efit_arclength"    => "efit_arclength",
    "efit_by_inversion" => "efit_by_inv",
)

pes = Dict{String, Any}()
for m in all_methods
    println("--- Running: $m ---")
    cfg = make_config(config_path, m, psihigh_arg)
    try
        pes[m] = setup_equilibrium(cfg)
        println("  Done.\n")
    catch e
        println("  FAILED: $e\n")
    end
end

failed = setdiff(all_methods, keys(pes))
isempty(failed) || println("Skipping failed methods in comparisons: $(join(failed, ", "))")

haskey(pes, ref_method) || error("Reference method '$ref_method' failed — cannot continue.")
pe_ref                 = pes[ref_method]
all_methods            = filter(m -> haskey(pes, m), all_methods)
compare_methods_active = filter(m -> haskey(pes, m), compare_methods)

# ─── Reconstruct efit_by_inversion grid for contour-plot overlay ───────────────
r_global_grid = Float64[]
z_global_grid = Float64[]
r_zoom_grid   = Float64[]
z_zoom_grid   = Float64[]
if haskey(pes, "efit_by_inversion")
let
    global r_global_grid, z_global_grid, r_zoom_grid, z_zoom_grid
    cfg_inv   = make_config(config_path, "efit_by_inversion", psihigh_arg)
    raw_inv   = Equilibrium.read_efit(cfg_inv)
    ro_g, zo_g = pes["efit_by_inversion"].ro, pes["efit_by_inversion"].zo
    psio_g    = raw_inv.psio
    psilow_g  = cfg_inv.psilow
    topology_g = Equilibrium.classify_topology(raw_inv, psio_g)

    ψ_RR = abs(raw_inv.psi_in((ro_g, zo_g); deriv=Val((2, 0))))
    ψ_ZZ = abs(raw_inv.psi_in((ro_g, zo_g); deriv=Val((0, 2))))
    a_low_g = min(sqrt(2 * psilow_g * psio_g / ψ_RR), sqrt(2 * psilow_g * psio_g / ψ_ZZ))

    nw_g = length(raw_inv.psi_in_xs)
    nh_g = length(raw_inv.psi_in_ys)
    β_r_g, β_z_g = 2.0, 2.0
    refine_g = 5

    nr_g = max(4, round(Int, refine_g * nw_g))
    nz_g = max(4, round(Int, refine_g * nh_g))
    r_g  = Equilibrium.make_stretched_r_grid(raw_inv.rmin, raw_inv.rmax, ro_g, nr_g, β_r_g)
    z_g  = Equilibrium.make_stretched_z_grid(raw_inv.zmin, raw_inv.zmax, zo_g, nz_g, topology_g, β_z_g)

    iro_g = clamp(searchsortedfirst(r_g, ro_g), 2, length(r_g))
    izo_g = clamp(searchsortedfirst(z_g,  zo_g), 2, length(z_g))
    dR_ax = r_g[iro_g] - r_g[iro_g - 1]
    dZ_ax = z_g[izo_g] - z_g[izo_g - 1]
    sc_r  = max(1, ceil(Int, dR_ax / (0.2 * a_low_g)))
    sc_z  = max(1, ceil(Int, dZ_ax / (0.2 * a_low_g)))
    if sc_r > 1
        nr_g = (nr_g - 1) * sc_r + 1
        r_g  = Equilibrium.make_stretched_r_grid(raw_inv.rmin, raw_inv.rmax, ro_g, nr_g, β_r_g)
        iro_g = clamp(searchsortedfirst(r_g, ro_g), 2, length(r_g))
        dR_ax = r_g[iro_g] - r_g[iro_g - 1]
    end
    if sc_z > 1
        nz_g = (nz_g - 1) * sc_z + 1
        z_g  = Equilibrium.make_stretched_z_grid(raw_inv.zmin, raw_inv.zmax, zo_g, nz_g, topology_g, β_z_g)
        izo_g = clamp(searchsortedfirst(z_g, zo_g), 2, length(z_g))
        dZ_ax = z_g[izo_g] - z_g[izo_g - 1]
    end
    r_global_grid = r_g
    z_global_grid = z_g

    d_max_g      = max(dR_ax, dZ_ax)
    threshold_g  = (10 * d_max_g)^2 * max(ψ_RR, ψ_ZZ) / (2 * psio_g)
    psi_zoom_max = 9 * threshold_g
    a_zoom_r = sqrt(2 * psi_zoom_max * psio_g / ψ_RR) * 1.2
    a_zoom_z = sqrt(2 * psi_zoom_max * psio_g / ψ_ZZ) * 1.2
    sinh_β   = sinh(β_r_g)
    nr_z = max(200, ceil(Int, 8 * β_r_g * a_zoom_r * 10 / (sinh_β * a_low_g)))
    nz_z = max(200, ceil(Int, 8 * β_r_g * a_zoom_z * 10 / (sinh_β * a_low_g)))
    r_zoom_grid = Equilibrium.make_stretched_r_grid(
        max(raw_inv.rmin, ro_g - a_zoom_r), min(raw_inv.rmax, ro_g + a_zoom_r),
        ro_g, nr_z, β_r_g)
    z_zoom_grid = Equilibrium.make_stretched_r_grid(
        max(raw_inv.zmin, zo_g - a_zoom_z), min(raw_inv.zmax, zo_g + a_zoom_z),
        zo_g, nz_z, β_r_g)
    println("  Grid vis: global $(nr_g)×$(nz_g), zoom $(nr_z)×$(nz_z)")
end
end

# ─── Common psi grid for numerical summaries ───────────────────────────────────
psi_lo = pe_ref.rzphi_xs[1]
psi_hi = pe_ref.rzphi_xs[end]
mpsi_eval = 8 * (length(pe_ref.rzphi_xs) - 1)
psi_full  = psi_lo .+ (psi_hi - psi_lo) .* sin.(range(0.0, 1.0; length=mpsi_eval+1) .* (π/2)).^2
mask_core = psi_full .< 0.10
mask_edge = psi_full .> 0.98
theta_select = [0.0, 0.25, 0.5, 0.75]

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  1D PROFILE COMPARISONS
# ═══════════════════════════════════════════════════════════════════════════════
println("=" ^ 70)
println("1. 1D Profile Comparisons")
println("=" ^ 70)

p = EqPlot.plot_profile_splines(pe_ref;
        color=method_color[ref_method], label=method_label[ref_method])
for m in compare_methods_active
    EqPlot.plot_profile_splines!(p, pes[m];
        color=method_color[m], linestyle=method_style[m], label=method_label[m])
end
savefig(p, joinpath(outdir, "profiles.png"))

profile_getters = [pe -> pe.profiles.F_spline, pe -> pe.profiles.P_spline,
                   pe -> pe.profiles.dVdpsi_spline, pe -> pe.profiles.q_spline]
profile_names   = ["F  (2π·R·Bₜ)", "μ₀·P", "dV/dψ", "q"]
for (pname, getter) in zip(profile_names, profile_getters)
    y_ref = [getter(pe_ref)(ψ) for ψ in psi_full]
    for m in compare_methods_active
        y_m  = [getter(pes[m])(ψ) for ψ in psi_full]
        Δy   = y_ref .- y_m
        @printf("  %-14s  vs %-18s  max|Δ|=%.3e  rms|Δ|=%.3e  edge max|Δ|=%.3e\n",
            pname, method_label[m], maximum(abs.(Δy)), sqrt(mean(Δy .^ 2)),
            any(mask_edge) ? maximum(abs.(Δy[mask_edge])) : NaN)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  EQUILIBRIUM CONTOURS (flux surfaces + theta lines)
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("2. Equilibrium Contours")
println("=" ^ 70)

p = EqPlot.plot_equilibrium_contours(pe_ref;
        color=method_color[ref_method], linestyle=method_style[ref_method],
        label=method_label[ref_method])
for m in compare_methods_active
    EqPlot.plot_equilibrium_contours!(p, pes[m];
        color=method_color[m], linestyle=method_style[m], label=method_label[m])
end

# Scatter efit_by_inversion R,Z computation grid on flux-surface panels (1=full, 2=core, 3=xpt)
if !isempty(r_global_grid)
let
    step_rg, step_zg = 20, 12
    Rg = [r_global_grid[ir] for ir in 1:step_rg:length(r_global_grid)
                             for _  in 1:step_zg:length(z_global_grid)]
    Zg = [z_global_grid[iz] for _  in 1:step_rg:length(r_global_grid)
                             for iz in 1:step_zg:length(z_global_grid)]
    step_rz, step_zz = 3, 3
    Rz = [r_zoom_grid[ir] for ir in 1:step_rz:length(r_zoom_grid)
                           for _  in 1:step_zz:length(z_zoom_grid)]
    Zz = [z_zoom_grid[iz] for _  in 1:step_rz:length(r_zoom_grid)
                           for iz in 1:step_zz:length(z_zoom_grid)]
    scatter!(p, Rg, Zg; ms=0.8, color=:grey, alpha=0.25, markerstrokewidth=0,
             label="global grid", subplot=1)
    scatter!(p, Rz, Zz; ms=0.8, color=:darkgrey, alpha=0.4, markerstrokewidth=0,
             label="zoom grid", subplot=1)
    scatter!(p, Rg, Zg; ms=1.2, color=:grey, alpha=0.3, markerstrokewidth=0,
             label="global grid", subplot=2)
    scatter!(p, Rz, Zz; ms=1.2, color=:darkgrey, alpha=0.5, markerstrokewidth=0,
             label="zoom grid", subplot=2)
    scatter!(p, Rg, Zg; ms=1.2, color=:grey, alpha=0.3, markerstrokewidth=0,
             label="global grid", subplot=3)
end
end

savefig(p, joinpath(outdir, "equilibrium_contours.png"))
println("  Saved equilibrium_contours.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  2D RZPHI GEOMETRIC SPLINES AT SELECT THETA SLICES
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("3. 2D rzphi Geometric Spline Profiles at Select θ Values")
println("=" ^ 70)

# linestyle distinguishes method; theta_colors (default) distinguish theta
p = EqPlot.plot_rzphi_splines(pe_ref; linestyle=method_style[ref_method],
        label=method_label[ref_method])
for m in compare_methods_active
    EqPlot.plot_rzphi_splines!(p, pes[m]; linestyle=method_style[m], label=method_label[m])
end
savefig(p, joinpath(outdir, "rzphi_splines.png"))

rzphi_getters = [pe -> ((ψ, θ) -> pe.rzphi_rsquared((ψ, θ))),
                 pe -> ((ψ, θ) -> pe.rzphi_offset((ψ, θ))),
                 pe -> ((ψ, θ) -> pe.rzphi_nu((ψ, θ))),
                 pe -> ((ψ, θ) -> pe.rzphi_jac((ψ, θ)))]
rzphi_names   = ["r²=(R-R₀)²+(Z-Z₀)²", "angle offset η/2π−θ", "ν (toroidal shift)", "Jacobian"]
for (sname, getter) in zip(rzphi_names, rzphi_getters)
    println("  $sname:")
    for m in compare_methods_active
        fn_ref = getter(pe_ref); fn_m = getter(pes[m])
        for θ in theta_select
            y_ref = [fn_ref(ψ, θ) for ψ in psi_full]
            y_m   = [fn_m(ψ, θ)   for ψ in psi_full]
            Δ = y_ref .- y_m
            Δc = any(mask_core) ? Δ[mask_core] : [0.0]
            Δe = any(mask_edge) ? Δ[mask_edge] : [0.0]
            @printf("    vs %-18s  θ=%.2f  core max|Δ|=%.3e  edge max|Δ|=%.3e  edge rms=%.3e\n",
                method_label[m], θ, maximum(abs.(Δc)), maximum(abs.(Δe)), sqrt(mean(Δe .^ 2)))
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  EQFUN PHYSICS SPLINES AT SELECT THETA SLICES
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("4. Physics (eqfun) Spline Profiles at Select θ Values")
println("=" ^ 70)

p = EqPlot.plot_eqfun_splines(pe_ref; linestyle=method_style[ref_method],
        label=method_label[ref_method])
for m in compare_methods_active
    EqPlot.plot_eqfun_splines!(p, pes[m]; linestyle=method_style[m], label=method_label[m])
end
savefig(p, joinpath(outdir, "eqfun_splines.png"))

eqfun_getters = [pe -> ((ψ, θ) -> pe.eqfun_B((ψ, θ))),
                 pe -> ((ψ, θ) -> pe.eqfun_metric1((ψ, θ))),
                 pe -> ((ψ, θ) -> pe.eqfun_metric2((ψ, θ)))]
eqfun_names   = ["|B| (total field [T])", "metric1 (e₁·e₂+q·e₃·e₁)/JB²", "metric2 (e₂·e₃+q·e₃²)/JB²"]
for (ename, getter) in zip(eqfun_names, eqfun_getters)
    println("  $ename:")
    for m in compare_methods_active
        fn_ref = getter(pe_ref); fn_m = getter(pes[m])
        for θ in theta_select
            y_ref = [fn_ref(ψ, θ) for ψ in psi_full]
            y_m   = [fn_m(ψ, θ)   for ψ in psi_full]
            Δ = y_ref .- y_m
            Δc = any(mask_core) ? Δ[mask_core] : [0.0]
            Δe = any(mask_edge) ? Δ[mask_edge] : [0.0]
            @printf("    vs %-18s  θ=%.2f  core max|Δ|=%.3e  edge max|Δ|=%.3e  edge rms=%.3e\n",
                method_label[m], θ, maximum(abs.(Δc)), maximum(abs.(Δe)), sqrt(mean(Δe .^ 2)))
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("Summary: axis / edge parameter comparison")
println("=" ^ 70)

col_labels = [method_label[m] for m in all_methods]
@printf("  %-22s  %s\n", "Parameter", join([@sprintf("%18s", l) for l in col_labels]))

function row(label, vals)
    @printf("  %-22s  %s\n", label, join([@sprintf("%18.6f", v) for v in vals]))
end
function row_int(label, vals)
    @printf("  %-22s  %s\n", label, join([@sprintf("%18d", v) for v in vals]))
end

row("ro [m]",        [pes[m].ro    for m in all_methods])
row("zo [m]",        [pes[m].zo    for m in all_methods])
row("psio [Wb/rad]", [pes[m].psio  for m in all_methods])
row("psilow",        [pes[m].rzphi_xs[1]   for m in all_methods])
row("psihigh",       [pes[m].rzphi_xs[end] for m in all_methods])
row("q(psilow)",     [pes[m].profiles.q_spline(pes[m].rzphi_xs[1])   for m in all_methods])
row("q(psihigh)",    [pes[m].profiles.q_spline(pes[m].rzphi_xs[end]) for m in all_methods])

q_profiles = Dict(m => [pes[m].profiles.q_spline(ψ) for ψ in psi_full] for m in all_methods)
row_int("q non-mono (edge)",
    [count(diff(q_profiles[m])[mask_edge[2:end]] .< 0) for m in all_methods])

println("\nAll figures saved to: $outdir")
