"""
equil_spline_comparison.jl

Detailed comparison of equilibrium descriptions produced by all three EFIT equilibrium
construction approaches, using "efit" as the reference.

Produces (saved to benchmarks/equil_spline_comparison_psihigh<value>/):
  1. 1D profile comparisons: F, P, dV/dψ, q with full-domain, core zoom, edge zoom, and diff subplots
  2. Flux surface contour overlays: psi(R,Z) with full, core zoom, and x-point zoom panels
  3. 2D rzphi spline profiles at θ = 0, 0.25, 0.5, 0.75 for each of the 4 geometric splines
  4. 2D eqfun physics spline profiles at the same theta slices
  5. Printed numerical summaries of deep-core and far-edge differences for every spline

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
# efit is the reference; efit_arclength and efit_by_inversion are compared to it.
ref_method     = "efit"
compare_methods  = ["efit_arclength", "efit_by_inversion"]
all_methods      = vcat(ref_method, compare_methods)

method_color = Dict(
    "efit"             => :blue,
    "efit_arclength"   => :darkorange,
    "efit_by_inversion" => :red,
)
method_style = Dict(
    "efit"             => :solid,
    "efit_arclength"   => :dash,
    "efit_by_inversion" => :dot,
)
method_label = Dict(
    "efit"             => "efit",
    "efit_arclength"   => "efit_arclength",
    "efit_by_inversion" => "efit_by_inv",
)
diff_color = Dict(
    "efit_arclength"   => :darkorange,
    "efit_by_inversion" => :red,
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
        Base.show_backtrace(stderr, catch_backtrace())
        println(stderr)
    end
end

failed = setdiff(all_methods, keys(pes))
isempty(failed) || println("Skipping failed methods in comparisons: $(join(failed, ", "))")

haskey(pes, ref_method) || error("Reference method '$ref_method' failed — cannot continue.")
pe_ref = pes[ref_method]
all_methods            = filter(m -> haskey(pes, m), all_methods)
compare_methods_active = filter(m -> haskey(pes, m), compare_methods)

# ─── Helpers ───────────────────────────────────────────────────────────────────
"Convert (ψ, θ) straight-field-line coordinates → (R, Z) physical coordinates."
function psi_theta_to_RZ(pe, psi, theta)
    r2   = pe.rzphi_rsquared((psi, theta))
    off  = pe.rzphi_offset((psi, theta))
    rfac = sqrt(max(r2, 0.0))
    η    = 2π * (theta + off)
    R    = pe.ro + rfac * cos(η)
    Z    = pe.zo + rfac * sin(η)
    return R, Z
end

"Compute the full (R, Z) trace of each flux surface in psi_vals."
function flux_surface_RZ(pe, psi_vals, theta_range)
    R_all = [Float64[] for _ in 1:length(psi_vals)]
    Z_all = [Float64[] for _ in 1:length(psi_vals)]
    for (k, psi) in enumerate(psi_vals)
        for θ in theta_range
            R, Z = psi_theta_to_RZ(pe, psi, θ)
            push!(R_all[k], R)
            push!(Z_all[k], Z)
        end
        push!(R_all[k], R_all[k][1])
        push!(Z_all[k], Z_all[k][1])
    end
    return R_all, Z_all
end

# Common psi evaluation grid — ldp distribution at 8× oversampling so inter-knot
# ringing is visible in both overplots and diff plots.
psi_lo = pe_ref.rzphi_xs[1]
psi_hi = pe_ref.rzphi_xs[end]
mpsi_eval = 8 * (length(pe_ref.rzphi_xs) - 1)
psi_full = psi_lo .+ (psi_hi - psi_lo) .* sin.(range(0.0, 1.0; length=mpsi_eval+1) .* (π/2)).^2

mask_core = psi_full .< 0.10
mask_edge = psi_full .> 0.98

theta_select = [0.0, 0.25, 0.5, 0.75]
theta_colors = [:blue, :green, :darkorange, :purple]

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  1D PROFILE COMPARISONS
# ═══════════════════════════════════════════════════════════════════════════════
println("=" ^ 70)
println("1. 1D Profile Comparisons")
println("=" ^ 70)

profile_specs = [
    ("F  (2π·R·Bₜ)",  pe -> pe.profiles.F_spline),
    ("μ₀·P",          pe -> pe.profiles.P_spline),
    ("dV/dψ",         pe -> pe.profiles.dVdpsi_spline),
    ("q",             pe -> pe.profiles.q_spline),
]
profile_fnames = ["profile_F", "profile_P", "profile_dVdpsi", "profile_q"]

for (idx, (pname, spl_getter)) in enumerate(profile_specs)
    spl_ref = spl_getter(pe_ref)
    y_ref   = [spl_ref(ψ) for ψ in psi_full]

    p_full = plot(psi_full, y_ref; lw=2, color=method_color[ref_method],
        ls=method_style[ref_method], label=method_label[ref_method],
        xlabel="ψ (normalized)", ylabel=pname,
        title="$pname — full domain  (psihigh=$psihigh_arg)")
    p_core = plot(psi_full[mask_core], y_ref[mask_core]; lw=2,
        color=method_color[ref_method], ls=method_style[ref_method],
        label=method_label[ref_method], xlabel="ψ", ylabel=pname, title="Deep core  (ψ < 0.10)")
    p_edge = plot(psi_full[mask_edge], y_ref[mask_edge]; lw=2,
        color=method_color[ref_method], ls=method_style[ref_method],
        label=method_label[ref_method], xlabel="ψ", ylabel=pname, title="Far edge  (ψ > 0.98)")
    p_diff = plot(; xlabel="ψ", ylabel="Δ$pname", title="Difference (vs efit)")
    hline!(p_diff, [0.0]; color=:black, lw=1, ls=:dot, label="")

    for m in compare_methods_active
        spl_m = spl_getter(pes[m])
        y_m   = [spl_m(ψ) for ψ in psi_full]
        Δy    = y_ref .- y_m

        plot!(p_full, psi_full, y_m; lw=1.5, color=method_color[m], ls=method_style[m],
            label=method_label[m])
        plot!(p_core, psi_full[mask_core], y_m[mask_core]; lw=1.5,
            color=method_color[m], ls=method_style[m], label=method_label[m])
        plot!(p_edge, psi_full[mask_edge], y_m[mask_edge]; lw=1.5,
            color=method_color[m], ls=method_style[m], label=method_label[m])
        plot!(p_diff, psi_full, Δy; lw=1.5, color=diff_color[m], label="efit − $(method_label[m])")

        @printf("  %-14s  vs %-18s  max|Δ|=%.3e  rms|Δ|=%.3e  edge max|Δ|=%.3e\n",
            pname, method_label[m], maximum(abs.(Δy)), sqrt(mean(Δy .^ 2)),
            any(mask_edge) ? maximum(abs.(Δy[mask_edge])) : NaN)
    end

    p_combo = plot(p_full, p_core, p_edge, p_diff; layout=(2, 2), size=(1400, 900),
        left_margin=8Plots.mm, bottom_margin=6Plots.mm)
    savefig(p_combo, joinpath(outdir, "$(profile_fnames[idx]).png"))
end

# ─── Reconstruct efit_by_inversion grids for grid-overplot visualization ───────
# Mirrors the grid construction in equilibrium_solver_by_inversion so we can
# scatter the actual grid points onto the flux-surface plots.
r_global_grid = Float64[]
z_global_grid = Float64[]
r_zoom_grid_vis = Float64[]
z_zoom_grid_vis = Float64[]
if haskey(pes, "efit_by_inversion")
let
    global r_global_grid, z_global_grid, r_zoom_grid_vis, z_zoom_grid_vis
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
    r_zoom_grid_vis = Equilibrium.make_stretched_r_grid(
        max(raw_inv.rmin, ro_g - a_zoom_r), min(raw_inv.rmax, ro_g + a_zoom_r),
        ro_g, nr_z, β_r_g)
    z_zoom_grid_vis = Equilibrium.make_stretched_r_grid(
        max(raw_inv.zmin, zo_g - a_zoom_z), min(raw_inv.zmax, zo_g + a_zoom_z),
        zo_g, nz_z, β_r_g)
    println("  Grid vis: global $(nr_g)×$(nz_g), zoom $(nr_z)×$(nz_z)")
end
end  # haskey(pes, "efit_by_inversion")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FLUX SURFACE CONTOUR OVERLAYS
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("2. Flux Surface Contour Overlays")
println("=" ^ 70)

psi_contours  = collect(range(psi_lo, psi_hi; length=20))
theta_contour = range(0.0, 1.0; length=256)

R_fs = Dict(m => flux_surface_RZ(pes[m], psi_contours, theta_contour) for m in all_methods)

p_ctr_full = plot(; aspect_ratio=:equal, xlabel="R [m]", ylabel="Z [m]",
    title="Flux surfaces: all methods  (psihigh=$psihigh_arg)")
p_ctr_core = plot(; aspect_ratio=:equal, xlabel="R [m]", ylabel="Z [m]",
    title="Deep core zoom (innermost 4 surfaces)")
p_ctr_xpt  = plot(; aspect_ratio=:equal, xlabel="R [m]", ylabel="Z [m]",
    title="Far edge / x-point zoom (outermost 3 surfaces)")

for m in all_methods
    Rs, Zs = R_fs[m]
    for k in 1:length(psi_contours)
        lbl = k == 1 ? method_label[m] : ""
        plot!(p_ctr_full, Rs[k], Zs[k]; color=method_color[m],
            ls=method_style[m], lw=0.9, alpha=0.7, label=lbl)
    end
    n_core = 4
    for k in 1:min(n_core, length(psi_contours))
        lbl = k == 1 ? method_label[m] : ""
        plot!(p_ctr_core, Rs[k], Zs[k]; color=method_color[m],
            ls=method_style[m], lw=1.5, label=lbl)
    end
    n_outer = length(psi_contours)
    for k in (n_outer - 2):n_outer
        lbl = k == n_outer - 2 ? method_label[m] : ""
        plot!(p_ctr_xpt, Rs[k], Zs[k]; color=method_color[m],
            ls=method_style[m], lw=1.5, label=lbl)
    end
end

# Grid overplots: scatter subsampled grid points on each panel (only if efit_by_inversion ran).
# Global grid: every 20th R × 12th Z (grey).  Zoom grid: every 3rd R × 3rd Z (darkgrey).
if !isempty(r_global_grid)
let
    step_rg, step_zg = 20, 12
    Rg = [r_global_grid[ir] for ir in 1:step_rg:length(r_global_grid)
                             for _  in 1:step_zg:length(z_global_grid)]
    Zg = [z_global_grid[iz] for _  in 1:step_rg:length(r_global_grid)
                             for iz in 1:step_zg:length(z_global_grid)]
    scatter!(p_ctr_full, Rg, Zg; ms=0.8, color=:grey, alpha=0.25,
        markerstrokewidth=0, label="global grid")

    step_rz, step_zz = 3, 3
    Rz = [r_zoom_grid_vis[ir] for ir in 1:step_rz:length(r_zoom_grid_vis)
                               for _  in 1:step_zz:length(z_zoom_grid_vis)]
    Zz = [z_zoom_grid_vis[iz] for _  in 1:step_rz:length(r_zoom_grid_vis)
                               for iz in 1:step_zz:length(z_zoom_grid_vis)]
    scatter!(p_ctr_full, Rz, Zz; ms=0.8, color=:darkgrey, alpha=0.4,
        markerstrokewidth=0, label="zoom grid")

    # Core panel: show both grids at full density within the core zoom region
    scatter!(p_ctr_core, Rg, Zg; ms=1.2, color=:grey, alpha=0.3,
        markerstrokewidth=0, label="global grid")
    scatter!(p_ctr_core, Rz, Zz; ms=1.2, color=:darkgrey, alpha=0.5,
        markerstrokewidth=0, label="zoom grid")

    # X-point panel: global grid only (zoom doesn't extend there)
    scatter!(p_ctr_xpt, Rg, Zg; ms=1.2, color=:grey, alpha=0.3,
        markerstrokewidth=0, label="global grid")
end
end  # !isempty(r_global_grid)

# Set zoom limits for core and x-point panels using the reference method
Rc, Zc = R_fs[ref_method]
n_core = 4
all_R_c = vcat(Rc[1:n_core]...); all_Z_c = vcat(Zc[1:n_core]...)
pad = 0.03
xlims!(p_ctr_core, minimum(all_R_c) - pad, maximum(all_R_c) + pad)
ylims!(p_ctr_core, minimum(all_Z_c) - pad, maximum(all_Z_c) + pad)

n_outer = length(psi_contours)
all_R_e = vcat(Rc[(n_outer-2):n_outer]...); all_Z_e = vcat(Zc[(n_outer-2):n_outer]...)
xpt_idx = argmin(all_Z_e)
xlims!(p_ctr_xpt, all_R_e[xpt_idx] - 0.20, all_R_e[xpt_idx] + 0.20)
ylims!(p_ctr_xpt, all_Z_e[xpt_idx] - 0.05, all_Z_e[xpt_idx] + 0.40)

p_ctr_combo = plot(p_ctr_full, p_ctr_core, p_ctr_xpt; layout=(1, 3), size=(2100, 800),
    left_margin=8Plots.mm, bottom_margin=6Plots.mm)
savefig(p_ctr_combo, joinpath(outdir, "flux_contours.png"))
println("  Saved flux contour figures.")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  2D RZPHI GEOMETRIC SPLINES AT SELECT THETA SLICES
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("3. 2D rzphi Geometric Spline Profiles at Select θ Values")
println("=" ^ 70)

rzphi_specs = [
    ("r²=(R-R₀)²+(Z-Z₀)²",  pe -> ((ψ, θ) -> pe.rzphi_rsquared((ψ, θ)))),
    ("angle offset  η/2π−θ", pe -> ((ψ, θ) -> pe.rzphi_offset((ψ, θ)))),
    ("ν  (toroidal shift)",  pe -> ((ψ, θ) -> pe.rzphi_nu((ψ, θ)))),
    ("Jacobian",             pe -> ((ψ, θ) -> pe.rzphi_jac((ψ, θ)))),
]
rzphi_fnames = ["rzphi_rsquared", "rzphi_angle_offset", "rzphi_nu", "rzphi_jacobian"]

for (sidx, (sname, fn_getter)) in enumerate(rzphi_specs)
    fn_ref  = fn_getter(pe_ref)
    yd_all  = [[fn_ref(ψ, θ) for ψ in psi_full] for θ in theta_select]

    p_full = plot(; xlabel="ψ", ylabel=sname,
        title="$sname — full domain  (psihigh=$psihigh_arg)")
    p_core = plot(; xlabel="ψ", ylabel=sname, title="Deep core  (ψ < 0.10)")
    p_edge = plot(; xlabel="ψ", ylabel=sname, title="Far edge  (ψ > 0.98)")
    p_diff = plot(; xlabel="ψ", ylabel="Δ$sname", title="efit − other")
    hline!(p_diff, [0.0]; color=:black, lw=1, ls=:dot, label="")

    for (tidx, θ) in enumerate(theta_select)
        tc = theta_colors[tidx]
        yd = yd_all[tidx]
        plot!(p_full, psi_full, yd; color=tc, lw=2, label="efit θ=$(θ)")
        plot!(p_core, psi_full[mask_core], yd[mask_core]; color=tc, lw=2, label="efit θ=$(θ)")
        plot!(p_edge, psi_full[mask_edge], yd[mask_edge]; color=tc, lw=2, label="efit θ=$(θ)")
    end

    println("  $sname:")
    for m in compare_methods_active
        fn_m   = fn_getter(pes[m])
        yi_all = [[fn_m(ψ, θ) for ψ in psi_full] for θ in theta_select]
        for (tidx, θ) in enumerate(theta_select)
            tc = theta_colors[tidx]
            yi = yi_all[tidx]
            Δ  = yd_all[tidx] .- yi
            plot!(p_full, psi_full, yi; color=tc, lw=1.2, ls=method_style[m],
                label="$(method_label[m]) θ=$(θ)")
            plot!(p_core, psi_full[mask_core], yi[mask_core]; color=tc, lw=1.2,
                ls=method_style[m], label="$(method_label[m]) θ=$(θ)")
            plot!(p_edge, psi_full[mask_edge], yi[mask_edge]; color=tc, lw=1.2,
                ls=method_style[m], label="$(method_label[m]) θ=$(θ)")
            plot!(p_diff, psi_full, Δ; color=tc, lw=1.2, ls=method_style[m],
                label="(efit−$(method_label[m])) θ=$(θ)")

            Δc = any(mask_core) ? Δ[mask_core] : [0.0]
            Δe = any(mask_edge) ? Δ[mask_edge] : [0.0]
            @printf("    vs %-18s  θ=%.2f  core max|Δ|=%.3e  edge max|Δ|=%.3e  edge rms=%.3e\n",
                method_label[m], θ, maximum(abs.(Δc)), maximum(abs.(Δe)), sqrt(mean(Δe .^ 2)))
        end
    end

    p_combo = plot(p_full, p_core, p_edge, p_diff; layout=(2, 2), size=(1400, 900),
        left_margin=8Plots.mm, bottom_margin=6Plots.mm)
    savefig(p_combo, joinpath(outdir, "$(rzphi_fnames[sidx]).png"))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  EQFUN PHYSICS SPLINES AT SELECT THETA SLICES
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("4. Physics (eqfun) Spline Profiles at Select θ Values")
println("=" ^ 70)

eqfun_specs = [
    ("|B| (total field [T])",          pe -> ((ψ, θ) -> pe.eqfun_B((ψ, θ)))),
    ("metric1  (e₁·e₂+q·e₃·e₁)/JB²", pe -> ((ψ, θ) -> pe.eqfun_metric1((ψ, θ)))),
    ("metric2  (e₂·e₃+q·e₃²)/JB²",   pe -> ((ψ, θ) -> pe.eqfun_metric2((ψ, θ)))),
]
eqfun_fnames = ["eqfun_Bmag", "eqfun_metric1", "eqfun_metric2"]

for (eidx, (ename, fn_getter)) in enumerate(eqfun_specs)
    fn_ref  = fn_getter(pe_ref)
    yd_all  = [[fn_ref(ψ, θ) for ψ in psi_full] for θ in theta_select]

    p_full = plot(; xlabel="ψ", ylabel=ename,
        title="$ename — full domain  (psihigh=$psihigh_arg)")
    p_core = plot(; xlabel="ψ", ylabel=ename, title="Deep core  (ψ < 0.10)")
    p_edge = plot(; xlabel="ψ", ylabel=ename, title="Far edge  (ψ > 0.98)")
    p_diff = plot(; xlabel="ψ", ylabel="Δ$ename", title="efit − other")
    hline!(p_diff, [0.0]; color=:black, lw=1, ls=:dot, label="")

    for (tidx, θ) in enumerate(theta_select)
        tc = theta_colors[tidx]
        yd = yd_all[tidx]
        plot!(p_full, psi_full, yd; color=tc, lw=2, label="efit θ=$(θ)")
        plot!(p_core, psi_full[mask_core], yd[mask_core]; color=tc, lw=2, label="efit θ=$(θ)")
        plot!(p_edge, psi_full[mask_edge], yd[mask_edge]; color=tc, lw=2, label="efit θ=$(θ)")
    end

    println("  $ename:")
    for m in compare_methods_active
        fn_m   = fn_getter(pes[m])
        yi_all = [[fn_m(ψ, θ) for ψ in psi_full] for θ in theta_select]
        for (tidx, θ) in enumerate(theta_select)
            tc = theta_colors[tidx]
            yi = yi_all[tidx]
            Δ  = yd_all[tidx] .- yi
            plot!(p_full, psi_full, yi; color=tc, lw=1.2, ls=method_style[m],
                label="$(method_label[m]) θ=$(θ)")
            plot!(p_core, psi_full[mask_core], yi[mask_core]; color=tc, lw=1.2,
                ls=method_style[m], label="$(method_label[m]) θ=$(θ)")
            plot!(p_edge, psi_full[mask_edge], yi[mask_edge]; color=tc, lw=1.2,
                ls=method_style[m], label="$(method_label[m]) θ=$(θ)")
            plot!(p_diff, psi_full, Δ; color=tc, lw=1.2, ls=method_style[m],
                label="(efit−$(method_label[m])) θ=$(θ)")

            Δc = any(mask_core) ? Δ[mask_core] : [0.0]
            Δe = any(mask_edge) ? Δ[mask_edge] : [0.0]
            @printf("    vs %-18s  θ=%.2f  core max|Δ|=%.3e  edge max|Δ|=%.3e  edge rms=%.3e\n",
                method_label[m], θ, maximum(abs.(Δc)), maximum(abs.(Δe)), sqrt(mean(Δe .^ 2)))
        end
    end

    p_combo = plot(p_full, p_core, p_edge, p_diff; layout=(2, 2), size=(1400, 900),
        left_margin=8Plots.mm, bottom_margin=6Plots.mm)
    savefig(p_combo, joinpath(outdir, "$(eqfun_fnames[eidx]).png"))
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

row("ro [m]",       [pes[m].ro    for m in all_methods])
row("zo [m]",       [pes[m].zo    for m in all_methods])
row("psio [Wb/rad]",[pes[m].psio  for m in all_methods])
row("psilow",       [pes[m].rzphi_xs[1]   for m in all_methods])
row("psihigh",      [pes[m].rzphi_xs[end] for m in all_methods])
row("q(psilow)",    [pes[m].profiles.q_spline(pes[m].rzphi_xs[1])   for m in all_methods])
row("q(psihigh)",   [pes[m].profiles.q_spline(pes[m].rzphi_xs[end]) for m in all_methods])

q_profiles = Dict(m => [pes[m].profiles.q_spline(ψ) for ψ in psi_full] for m in all_methods)
row_int("q non-mono (edge)",
    [count(diff(q_profiles[m])[mask_edge[2:end]] .< 0) for m in all_methods])

println("\nAll figures saved to: $outdir")
