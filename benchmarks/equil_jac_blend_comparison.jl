"""
equil_jac_blend_comparison.jl

Comparison of equilibrium descriptions produced with different Jacobian blending
settings, using pure Hamada (no blend) as the reference.

The Jacobian blend transitions the theta-coordinate from a Hamada (J=1) grid in the
core to an equal-arc (J=Bp) grid near the separatrix via a cubic smoothstep.  The
primary diagnostic is the theta-contour plot (Section 2b): with Hamada coordinates,
theta lines cluster near the X-point at high ψ; with equal-arc blending they become
near-uniform in arc length.

Variants compared:
  "hamada"      — pure Hamada everywhere (baseline)
  "blend_wide"  — Hamada core → equal_arc edge, blend over ψ ∈ [0.80, 0.98]
  "blend_narrow"— Hamada core → equal_arc edge, blend over ψ ∈ [0.95, 0.99]
  "equal_arc"   — pure equal_arc everywhere

Produces (saved to benchmarks/equil_jac_blend_comparison_psihigh<value>/):
  1. 1D profile comparisons: F, P, dV/dψ, q
  2. Flux surface contour overlays (surface geometry is Jacobian-independent)
  2b. Theta contour lines: KEY DIAGNOSTIC — shows theta grid uniformity in (R,Z)
  3. 2D rzphi spline profiles at θ = 0, 0.25, 0.5, 0.75 (Jacobian and offset change)
  4. 2D eqfun physics spline profiles at the same theta slices
  5. Printed numerical summaries for every spline

Usage:
  julia --project=. benchmarks/equil_jac_blend_comparison.jl [example_path] [psihigh]

Recommended psihigh values: 0.997, 0.999
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium
using TOML, Printf, Statistics, Plots

# ─── CLI arguments ─────────────────────────────────────────────────────────────
example_path = length(ARGS) > 0 ? ARGS[1] :
    joinpath(@__DIR__, "../examples/DIIID-like_ideal_example")
psihigh_arg = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 0.999
config_path = joinpath(example_path, "gpec.toml")

println("=" ^ 70)
println("Equilibrium Jacobian Blend Comparison")
println("Example  : $example_path")
println("psihigh  : $psihigh_arg")
println("=" ^ 70)

# ─── Output directory ──────────────────────────────────────────────────────────
psihigh_tag = replace(string(psihigh_arg), "." => "p")
outdir = joinpath(@__DIR__, "equil_jac_blend_comparison_psihigh$(psihigh_tag)")
mkpath(outdir)
println("Output   : $outdir\n")

# ─── Config factory ────────────────────────────────────────────────────────────
function make_blend_config(path, psihigh, jac_type;
        jac_type_edge="hamada", psi_jac_blend_start=1.0, psi_jac_blend_end=1.0)
    raw = TOML.parsefile(path)
    eq = raw["Equilibrium"]
    eq["psihigh"]             = psihigh
    eq["jac_type"]            = jac_type
    eq["jac_type_edge"]       = jac_type_edge
    eq["psi_jac_blend_start"] = psi_jac_blend_start
    eq["psi_jac_blend_end"]   = psi_jac_blend_end
    return Equilibrium.EquilibriumConfig(eq, dirname(path))
end

# ─── Variant definitions ───────────────────────────────────────────────────────
# Each entry: (key, display label, config kwargs)
variants = [
    ("hamada",       "hamada (baseline)",
        (jac_type="hamada",)),
    ("blend_wide",   "hamada→eq_arc [0.80, 0.98]",
        (jac_type="hamada", jac_type_edge="equal_arc",
         psi_jac_blend_start=0.80, psi_jac_blend_end=0.98)),
    ("blend_narrow", "hamada→eq_arc [0.95, 0.99]",
        (jac_type="hamada", jac_type_edge="equal_arc",
         psi_jac_blend_start=0.95, psi_jac_blend_end=0.99)),
    ("equal_arc",    "equal_arc (full)",
        (jac_type="equal_arc",)),
]

ref_key     = "hamada"
variant_keys = [v[1] for v in variants]

variant_color = Dict(
    "hamada"        => :blue,
    "blend_wide"    => :darkorange,
    "blend_narrow"  => :green,
    "equal_arc"     => :red,
)
variant_style = Dict(
    "hamada"        => :solid,
    "blend_wide"    => :dash,
    "blend_narrow"  => :dashdot,
    "equal_arc"     => :dot,
)
variant_label = Dict(v[1] => v[2] for v in variants)

# ─── Run all variants ──────────────────────────────────────────────────────────
pes = Dict{String, Any}()
for (key, label, kwargs) in variants
    println("--- Running: $label ---")
    cfg = make_blend_config(config_path, psihigh_arg, get(kwargs, :jac_type, "hamada"); kwargs...)
    try
        pes[key] = setup_equilibrium(cfg)
        println("  Done.\n")
    catch e
        println("  FAILED: $e\n")
    end
end

failed = setdiff(variant_keys, keys(pes))
isempty(failed) || println("Skipping failed variants: $(join(failed, ", "))")

haskey(pes, ref_key) || error("Reference variant '$ref_key' failed — cannot continue.")
pe_ref          = pes[ref_key]
active_keys     = filter(k -> haskey(pes, k), variant_keys)
compare_keys    = filter(k -> k != ref_key && haskey(pes, k), variant_keys)

# ─── Helpers ───────────────────────────────────────────────────────────────────
"Convert (ψ, θ) straight-field-line coordinates → (R, Z) physical coordinates."
function psi_theta_to_RZ(pe, psi, theta)
    r2  = pe.rzphi_rsquared((psi, theta))
    off = pe.rzphi_offset((psi, theta))
    rfac = sqrt(max(r2, 0.0))
    η   = 2π * (theta + off)
    R   = pe.ro + rfac * cos(η)
    Z   = pe.zo + rfac * sin(η)
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

# Common psi evaluation grid
psi_lo = pe_ref.rzphi_xs[1]
psi_hi = pe_ref.rzphi_xs[end]
mpsi_eval = 8 * (length(pe_ref.rzphi_xs) - 1)
psi_full = psi_lo .+ (psi_hi - psi_lo) .* sin.(range(0.0, 1.0; length=mpsi_eval+1) .* (π/2)).^2

mask_core = psi_full .< 0.10
mask_edge = psi_full .> 0.98
t_edge_full = -log.(1.0 .- psi_full[mask_edge])

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

    p_full = plot(psi_full, y_ref; lw=2, color=variant_color[ref_key],
        ls=variant_style[ref_key], label=variant_label[ref_key],
        xlabel="ψ (normalized)", ylabel=pname, legend=:best, xlims=(0, 1),
        title="$pname — full domain  (psihigh=$psihigh_arg)")
    p_core = plot(psi_full[mask_core], y_ref[mask_core]; lw=2,
        color=variant_color[ref_key], ls=variant_style[ref_key],
        label=variant_label[ref_key], xlabel="ψ  (log scale)", ylabel=pname,
        title="Deep core  (ψ < 0.10)", xscale=:log10, legend=:none)
    p_edge = plot(t_edge_full, y_ref[mask_edge]; lw=2,
        color=variant_color[ref_key], ls=variant_style[ref_key],
        label=variant_label[ref_key], xlabel="−ln(1−ψ)", ylabel=pname,
        title="Far edge  (ψ > 0.98)", legend=:none)
    p_diff = plot(; xlabel="ψ", ylabel="Δ$pname", title="Difference (vs hamada baseline)",
        legend=:none, xlims=(0, 1))
    hline!(p_diff, [0.0]; color=:black, lw=1, ls=:dot, label="")

    ref_xs = pe_ref.rzphi_xs
    ref_ys = [spl_ref(ψ) for ψ in ref_xs]
    scatter!(p_full, ref_xs, ref_ys; ms=3, markerstrokewidth=0, color=variant_color[ref_key], label="")
    let mask = ref_xs .< 0.10
        scatter!(p_core, ref_xs[mask], ref_ys[mask]; ms=3, markerstrokewidth=0, color=variant_color[ref_key], label="")
    end
    let mask = ref_xs .> 0.98
        scatter!(p_edge, -log.(1.0 .- ref_xs[mask]), ref_ys[mask]; ms=3, markerstrokewidth=0, color=variant_color[ref_key], label="")
    end

    for k in compare_keys
        spl_k = spl_getter(pes[k])
        y_k   = [spl_k(ψ) for ψ in psi_full]
        Δy    = y_ref .- y_k

        plot!(p_full, psi_full, y_k; lw=1.5, color=variant_color[k], ls=variant_style[k],
            label=variant_label[k])
        plot!(p_core, psi_full[mask_core], y_k[mask_core]; lw=1.5,
            color=variant_color[k], ls=variant_style[k], label=variant_label[k])
        plot!(p_edge, t_edge_full, y_k[mask_edge]; lw=1.5,
            color=variant_color[k], ls=variant_style[k], label=variant_label[k])
        plot!(p_diff, psi_full, Δy; lw=1.5, color=variant_color[k],
            label="hamada − $(variant_label[k])")

        k_xs = pes[k].rzphi_xs
        k_ys = [spl_k(ψ) for ψ in k_xs]
        scatter!(p_full, k_xs, k_ys; ms=3, markerstrokewidth=0, color=variant_color[k], label="")
        let mask = k_xs .< 0.10
            scatter!(p_core, k_xs[mask], k_ys[mask]; ms=3, markerstrokewidth=0, color=variant_color[k], label="")
        end
        let mask = k_xs .> 0.98
            scatter!(p_edge, -log.(1.0 .- k_xs[mask]), k_ys[mask]; ms=3, markerstrokewidth=0, color=variant_color[k], label="")
        end

        @printf("  %-14s  vs %-30s  max|Δ|=%.3e  rms|Δ|=%.3e  edge max|Δ|=%.3e\n",
            pname, variant_label[k], maximum(abs.(Δy)), sqrt(mean(Δy .^ 2)),
            any(mask_edge) ? maximum(abs.(Δy[mask_edge])) : NaN)
    end

    p_combo = plot(p_full, p_core, p_edge, p_diff; layout=(2, 2), size=(1400, 900),
        left_margin=8Plots.mm, bottom_margin=6Plots.mm)
    savefig(p_combo, joinpath(outdir, "$(profile_fnames[idx]).png"))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FLUX SURFACE CONTOUR OVERLAYS
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("2. Flux Surface Contour Overlays")
println("=" ^ 70)

psi_contours  = collect(range(psi_lo, psi_hi; length=20))
theta_contour = range(0.0, 1.0; length=256)

R_fs = Dict(k => flux_surface_RZ(pes[k], psi_contours, theta_contour) for k in active_keys)

p_ctr_full = plot(; aspect_ratio=:equal, xlabel="R [m]", ylabel="Z [m]",
    title="Flux surfaces: all variants  (psihigh=$psihigh_arg)")
p_ctr_core = plot(; aspect_ratio=:equal, xlabel="R [m]", ylabel="Z [m]",
    title="Deep core zoom (innermost 4 surfaces)")
p_ctr_xpt  = plot(; aspect_ratio=:equal, xlabel="R [m]", ylabel="Z [m]",
    title="Far edge / x-point zoom (outermost 3 surfaces)")

for k in active_keys
    Rs, Zs = R_fs[k]
    for i in 1:length(psi_contours)
        lbl = i == 1 ? variant_label[k] : ""
        plot!(p_ctr_full, Rs[i], Zs[i]; color=variant_color[k],
            ls=variant_style[k], lw=0.9, alpha=0.7, label=lbl)
    end
    for i in 1:min(4, length(psi_contours))
        lbl = i == 1 ? variant_label[k] : ""
        plot!(p_ctr_core, Rs[i], Zs[i]; color=variant_color[k],
            ls=variant_style[k], lw=1.5, label=lbl)
    end
    n = length(psi_contours)
    for i in (n-2):n
        lbl = i == n - 2 ? variant_label[k] : ""
        plot!(p_ctr_xpt, Rs[i], Zs[i]; color=variant_color[k],
            ls=variant_style[k], lw=1.5, label=lbl)
    end
end

Rc, Zc = R_fs[ref_key]
all_R_c = vcat(Rc[1:4]...); all_Z_c = vcat(Zc[1:4]...)
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

# ─── Theta contour plot — KEY DIAGNOSTIC ─────────────────────────────────────
# With Hamada coordinates, θ=const spokes cluster near the X-point at high ψ.
# With equal-arc blending, they become near-uniform in arc length there.
psi_dense = range(psi_lo, psi_hi; length=300)

p_tht_full = plot(; aspect_ratio=:equal, xlabel="R [m]", ylabel="Z [m]",
    title="θ = const lines: all variants  (psihigh=$psihigh_arg)")
p_tht_core = plot(; aspect_ratio=:equal, xlabel="R [m]", ylabel="Z [m]",
    title="Deep core zoom")
p_tht_xpt  = plot(; aspect_ratio=:equal, xlabel="R [m]", ylabel="Z [m]",
    title="Far edge / x-point zoom  ← KEY DIAGNOSTIC")

for k in active_keys
    pe = pes[k]
    θ_all = pe.rzphi_ys
    step  = max(1, length(θ_all) ÷ 32)
    θ_sub = θ_all[1:step:end]
    for (kidx, θ) in enumerate(θ_sub)
        lbl = kidx == 1 ? variant_label[k] : ""
        Rs = [psi_theta_to_RZ(pe, ψ, θ)[1] for ψ in psi_dense]
        Zs = [psi_theta_to_RZ(pe, ψ, θ)[2] for ψ in psi_dense]
        plot!(p_tht_full, Rs, Zs; color=variant_color[k], ls=variant_style[k], lw=0.9, alpha=0.7, label=lbl)
        plot!(p_tht_core, Rs, Zs; color=variant_color[k], ls=variant_style[k], lw=1.5, label=lbl)
        plot!(p_tht_xpt,  Rs, Zs; color=variant_color[k], ls=variant_style[k], lw=1.5, label=lbl)
    end
    Rg = [psi_theta_to_RZ(pe, ψ, θ)[1] for ψ in pe.rzphi_xs for θ in θ_sub]
    Zg = [psi_theta_to_RZ(pe, ψ, θ)[2] for ψ in pe.rzphi_xs for θ in θ_sub]
    scatter!(p_tht_full, Rg, Zg; ms=1.5, color=variant_color[k], alpha=0.5, markerstrokewidth=0, label="")
    scatter!(p_tht_core, Rg, Zg; ms=2.0, color=variant_color[k], alpha=0.6, markerstrokewidth=0, label="")
    scatter!(p_tht_xpt,  Rg, Zg; ms=2.0, color=variant_color[k], alpha=0.6, markerstrokewidth=0, label="")
end

xlims!(p_tht_core, minimum(all_R_c) - pad, maximum(all_R_c) + pad)
ylims!(p_tht_core, minimum(all_Z_c) - pad, maximum(all_Z_c) + pad)
xlims!(p_tht_xpt, all_R_e[xpt_idx] - 0.20, all_R_e[xpt_idx] + 0.20)
ylims!(p_tht_xpt, all_Z_e[xpt_idx] - 0.05, all_Z_e[xpt_idx] + 0.40)

p_tht_combo = plot(p_tht_full, p_tht_core, p_tht_xpt; layout=(1, 3), size=(2100, 800),
    left_margin=8Plots.mm, bottom_margin=6Plots.mm)
savefig(p_tht_combo, joinpath(outdir, "theta_contours.png"))
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

    p_full = plot(; xlabel="ψ", ylabel=sname, legend=:best, xlims=(0, 1),
        title="$sname — full domain  (psihigh=$psihigh_arg)")
    p_core = plot(; xlabel="ψ  (log scale)", ylabel=sname,
        title="Deep core  (ψ < 0.10)", xscale=:log10, legend=:none)
    p_edge = plot(; xlabel="−ln(1−ψ)", ylabel=sname,
        title="Far edge  (ψ > 0.98)", legend=:none)
    p_diff = plot(; xlabel="ψ", ylabel="Δ$sname",
        title="hamada − other", legend=:none, xlims=(0, 1))
    hline!(p_diff, [0.0]; color=:black, lw=1, ls=:dot, label="")

    for (tidx, θ) in enumerate(theta_select)
        tc = theta_colors[tidx]
        yd = yd_all[tidx]
        plot!(p_full, psi_full, yd; color=tc, lw=2, label="hamada θ=$(θ)")
        plot!(p_core, psi_full[mask_core], yd[mask_core]; color=tc, lw=2, label="hamada θ=$(θ)")
        plot!(p_edge, t_edge_full, yd[mask_edge]; color=tc, lw=2, label="hamada θ=$(θ)")
        xs = pe_ref.rzphi_xs; ys = [fn_ref(ψ, θ) for ψ in xs]
        scatter!(p_full, xs, ys; ms=3, markerstrokewidth=0, color=tc, label="")
        let mask = xs .< 0.10; scatter!(p_core, xs[mask], ys[mask]; ms=3, markerstrokewidth=0, color=tc, label="") end
        let mask = xs .> 0.98; scatter!(p_edge, -log.(1.0 .- xs[mask]), ys[mask]; ms=3, markerstrokewidth=0, color=tc, label="") end
    end

    println("  $sname:")
    for k in compare_keys
        fn_k   = fn_getter(pes[k])
        yi_all = [[fn_k(ψ, θ) for ψ in psi_full] for θ in theta_select]
        for (tidx, θ) in enumerate(theta_select)
            tc = theta_colors[tidx]
            yi = yi_all[tidx]
            Δ  = yd_all[tidx] .- yi
            plot!(p_full, psi_full, yi; color=tc, lw=1.2, ls=variant_style[k],
                label="$(variant_label[k]) θ=$(θ)")
            plot!(p_core, psi_full[mask_core], yi[mask_core]; color=tc, lw=1.2,
                ls=variant_style[k], label="$(variant_label[k]) θ=$(θ)")
            plot!(p_edge, t_edge_full, yi[mask_edge]; color=tc, lw=1.2,
                ls=variant_style[k], label="$(variant_label[k]) θ=$(θ)")
            plot!(p_diff, psi_full, Δ; color=tc, lw=1.2, ls=variant_style[k],
                label="(hamada−$(variant_label[k])) θ=$(θ)")
            xs = pes[k].rzphi_xs; ys = [fn_k(ψ, θ) for ψ in xs]
            scatter!(p_full, xs, ys; ms=3, markerstrokewidth=0, color=tc, label="")
            let mask = xs .< 0.10; scatter!(p_core, xs[mask], ys[mask]; ms=3, markerstrokewidth=0, color=tc, label="") end
            let mask = xs .> 0.98; scatter!(p_edge, -log.(1.0 .- xs[mask]), ys[mask]; ms=3, markerstrokewidth=0, color=tc, label="") end

            Δc = any(mask_core) ? Δ[mask_core] : [0.0]
            Δe = any(mask_edge) ? Δ[mask_edge] : [0.0]
            @printf("    vs %-30s  θ=%.2f  core max|Δ|=%.3e  edge max|Δ|=%.3e  edge rms=%.3e\n",
                variant_label[k], θ, maximum(abs.(Δc)), maximum(abs.(Δe)), sqrt(mean(Δe .^ 2)))
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

    p_full = plot(; xlabel="ψ", ylabel=ename, legend=:best, xlims=(0, 1),
        title="$ename — full domain  (psihigh=$psihigh_arg)")
    p_core = plot(; xlabel="ψ  (log scale)", ylabel=ename,
        title="Deep core  (ψ < 0.10)", xscale=:log10, legend=:none)
    p_edge = plot(; xlabel="−ln(1−ψ)", ylabel=ename,
        title="Far edge  (ψ > 0.98)", legend=:none)
    p_diff = plot(; xlabel="ψ", ylabel="Δ$ename",
        title="hamada − other", legend=:none, xlims=(0, 1))
    hline!(p_diff, [0.0]; color=:black, lw=1, ls=:dot, label="")

    for (tidx, θ) in enumerate(theta_select)
        tc = theta_colors[tidx]
        yd = yd_all[tidx]
        plot!(p_full, psi_full, yd; color=tc, lw=2, label="hamada θ=$(θ)")
        plot!(p_core, psi_full[mask_core], yd[mask_core]; color=tc, lw=2, label="hamada θ=$(θ)")
        plot!(p_edge, t_edge_full, yd[mask_edge]; color=tc, lw=2, label="hamada θ=$(θ)")
        xs = pe_ref.rzphi_xs; ys = [fn_ref(ψ, θ) for ψ in xs]
        scatter!(p_full, xs, ys; ms=3, markerstrokewidth=0, color=tc, label="")
        let mask = xs .< 0.10; scatter!(p_core, xs[mask], ys[mask]; ms=3, markerstrokewidth=0, color=tc, label="") end
        let mask = xs .> 0.98; scatter!(p_edge, -log.(1.0 .- xs[mask]), ys[mask]; ms=3, markerstrokewidth=0, color=tc, label="") end
    end

    println("  $ename:")
    for k in compare_keys
        fn_k   = fn_getter(pes[k])
        yi_all = [[fn_k(ψ, θ) for ψ in psi_full] for θ in theta_select]
        for (tidx, θ) in enumerate(theta_select)
            tc = theta_colors[tidx]
            yi = yi_all[tidx]
            Δ  = yd_all[tidx] .- yi
            plot!(p_full, psi_full, yi; color=tc, lw=1.2, ls=variant_style[k],
                label="$(variant_label[k]) θ=$(θ)")
            plot!(p_core, psi_full[mask_core], yi[mask_core]; color=tc, lw=1.2,
                ls=variant_style[k], label="$(variant_label[k]) θ=$(θ)")
            plot!(p_edge, t_edge_full, yi[mask_edge]; color=tc, lw=1.2,
                ls=variant_style[k], label="$(variant_label[k]) θ=$(θ)")
            plot!(p_diff, psi_full, Δ; color=tc, lw=1.2, ls=variant_style[k],
                label="(hamada−$(variant_label[k])) θ=$(θ)")
            xs = pes[k].rzphi_xs; ys = [fn_k(ψ, θ) for ψ in xs]
            scatter!(p_full, xs, ys; ms=3, markerstrokewidth=0, color=tc, label="")
            let mask = xs .< 0.10; scatter!(p_core, xs[mask], ys[mask]; ms=3, markerstrokewidth=0, color=tc, label="") end
            let mask = xs .> 0.98; scatter!(p_edge, -log.(1.0 .- xs[mask]), ys[mask]; ms=3, markerstrokewidth=0, color=tc, label="") end

            Δc = any(mask_core) ? Δ[mask_core] : [0.0]
            Δe = any(mask_edge) ? Δ[mask_edge] : [0.0]
            @printf("    vs %-30s  θ=%.2f  core max|Δ|=%.3e  edge max|Δ|=%.3e  edge rms=%.3e\n",
                variant_label[k], θ, maximum(abs.(Δc)), maximum(abs.(Δe)), sqrt(mean(Δe .^ 2)))
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

col_labels = [variant_label[k] for k in active_keys]
@printf("  %-22s  %s\n", "Parameter", join([@sprintf("%30s", l) for l in col_labels]))

function row(label, vals)
    @printf("  %-22s  %s\n", label, join([@sprintf("%30.6f", v) for v in vals]))
end
function row_int(label, vals)
    @printf("  %-22s  %s\n", label, join([@sprintf("%30d", v) for v in vals]))
end

row("ro [m]",        [pes[k].ro    for k in active_keys])
row("zo [m]",        [pes[k].zo    for k in active_keys])
row("psio [Wb/rad]", [pes[k].psio  for k in active_keys])
row("psilow",        [pes[k].rzphi_xs[1]   for k in active_keys])
row("psihigh",       [pes[k].rzphi_xs[end] for k in active_keys])
row("q(psilow)",     [pes[k].profiles.q_spline(pes[k].rzphi_xs[1])   for k in active_keys])
row("q(psihigh)",    [pes[k].profiles.q_spline(pes[k].rzphi_xs[end]) for k in active_keys])

q_profiles = Dict(k => [pes[k].profiles.q_spline(ψ) for ψ in psi_full] for k in active_keys)
row_int("q non-mono (edge)",
    [count(diff(q_profiles[k])[mask_edge[2:end]] .< 0) for k in active_keys])

println("\nAll figures saved to: $outdir")
