"""
equil_jac_blend_comparison.jl

Comparison of equilibrium descriptions produced with different Jacobian blending
settings, using pure Hamada (no blend) as the reference.

The Jacobian blend transitions the theta-coordinate from a Hamada (J=1) grid in the
core to an equal-arc (J=Bp) grid near the separatrix via a cubic smoothstep.  The
primary diagnostic is the equilibrium_contours.png x-point panel: with Hamada
coordinates, theta lines cluster near the X-point at high ψ; with equal-arc blending
they become near-uniform in arc length.

Variants compared:
  "hamada"       — pure Hamada everywhere (baseline)
  "blend_wide"   — Hamada core → equal_arc edge, blend over ψ ∈ [0.80, 0.98]
  "blend_narrow" — Hamada core → equal_arc edge, blend over ψ ∈ [0.95, 0.99]
  "equal_arc"    — pure equal_arc everywhere

Produces (saved to benchmarks/equil_jac_blend_comparison_psihigh<value>/):
  profiles.png          — 4×3 panel: F, P, dV/dψ, q (full / core zoom / edge zoom)
  equilibrium_contours.png — 2×3 panel: flux surfaces + theta contours (full / core / x-point)
  rzphi_splines.png     — 4×3 panel: r², offset, ν, Jacobian at θ = 0, 0.25, 0.5, 0.75
  eqfun_splines.png     — 3×3 panel: |B|, metric₁, metric₂ at select θ values
  Printed numerical summaries, spline differences, and stability eigenvalue et[1] per variant.

Usage:
  julia --project=. benchmarks/equil_jac_blend_comparison.jl [example_path] [psihigh]

Recommended psihigh values: 0.997, 0.999
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium
using TOML, Printf, Statistics, Plots
import FastInterpolations: cubic_interp, CubicFit, ExtendExtrap

const EqPlot = Analysis.Equilibrium

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
function make_blend_config(path, psihigh;
        jac_type="hamada", jac_type_edge="hamada", psi_jac_blend_start=1.0, psi_jac_blend_end=1.0)
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
    ("hamada",        "hamada (baseline)",
        (jac_type="hamada",)),
    ("blend_wide",    "hamada→eq_arc [0.80, 0.98]",
        (jac_type="hamada", jac_type_edge="equal_arc",
         psi_jac_blend_start=0.80, psi_jac_blend_end=0.98)),
    ("blend_narrow",  "hamada→eq_arc [0.95, 0.99]",
        (jac_type="hamada", jac_type_edge="equal_arc",
         psi_jac_blend_start=0.95, psi_jac_blend_end=0.99)),
    ("equal_arc",     "equal_arc (full)",
        (jac_type="equal_arc",)),
]

ref_key      = "hamada"
variant_keys = [v[1] for v in variants]

variant_color = Dict(
    "hamada"       => :blue,
    "blend_wide"   => :darkorange,
    "blend_narrow" => :green,
    "equal_arc"    => :red,
)
variant_style = Dict(
    "hamada"       => :solid,
    "blend_wide"   => :dash,
    "blend_narrow" => :dashdot,
    "equal_arc"    => :dot,
)
variant_label = Dict(v[1] => v[2] for v in variants)

# ─── Run all variants ──────────────────────────────────────────────────────────
pes = Dict{String, Any}()
for (key, label, kwargs) in variants
    println("--- Running: $label ---")
    cfg = make_blend_config(config_path, psihigh_arg; kwargs...)
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
pe_ref       = pes[ref_key]
active_keys  = filter(k -> haskey(pes, k), variant_keys)
compare_keys = filter(k -> k != ref_key && haskey(pes, k), variant_keys)

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
        color=variant_color[ref_key], label=variant_label[ref_key])
for k in compare_keys
    EqPlot.plot_profile_splines!(p, pes[k];
        color=variant_color[k], linestyle=variant_style[k], label=variant_label[k])
end
savefig(p, joinpath(outdir, "profiles.png"))

profile_getters = [pe -> pe.profiles.F_spline, pe -> pe.profiles.P_spline,
                   pe -> pe.profiles.dVdpsi_spline, pe -> pe.profiles.q_spline]
profile_names   = ["F  (2π·R·Bₜ)", "μ₀·P", "dV/dψ", "q"]
for (pname, getter) in zip(profile_names, profile_getters)
    y_ref = [getter(pe_ref)(ψ) for ψ in psi_full]
    for k in compare_keys
        y_k = [getter(pes[k])(ψ) for ψ in psi_full]
        Δy  = y_ref .- y_k
        @printf("  %-14s  vs %-30s  max|Δ|=%.3e  rms|Δ|=%.3e  edge max|Δ|=%.3e\n",
            pname, variant_label[k], maximum(abs.(Δy)), sqrt(mean(Δy .^ 2)),
            any(mask_edge) ? maximum(abs.(Δy[mask_edge])) : NaN)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  EQUILIBRIUM CONTOURS (flux surfaces + theta lines)
# ═══════════════════════════════════════════════════════════════════════════════
# KEY DIAGNOSTIC: the x-point panel of theta contours shows how blending
# redistributes theta grid points from X-point clustering to arc-length uniformity.
println("\n" * "=" ^ 70)
println("2. Equilibrium Contours  (key: x-point theta panel)")
println("=" ^ 70)

p = EqPlot.plot_equilibrium_contours(pe_ref;
        color=variant_color[ref_key], linestyle=variant_style[ref_key],
        label=variant_label[ref_key])
for k in compare_keys
    EqPlot.plot_equilibrium_contours!(p, pes[k];
        color=variant_color[k], linestyle=variant_style[k], label=variant_label[k])
end
savefig(p, joinpath(outdir, "equilibrium_contours.png"))
println("  Saved equilibrium_contours.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  2D RZPHI GEOMETRIC SPLINES AT SELECT THETA SLICES
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("3. 2D rzphi Geometric Spline Profiles at Select θ Values")
println("=" ^ 70)

# linestyle distinguishes variant; theta_colors (default) distinguish theta
p = EqPlot.plot_rzphi_splines(pe_ref; linestyle=variant_style[ref_key],
        label=variant_label[ref_key])
for k in compare_keys
    EqPlot.plot_rzphi_splines!(p, pes[k]; linestyle=variant_style[k], label=variant_label[k])
end
savefig(p, joinpath(outdir, "rzphi_splines.png"))

rzphi_getters = [pe -> ((ψ, θ) -> pe.rzphi_rsquared((ψ, θ))),
                 pe -> ((ψ, θ) -> pe.rzphi_offset((ψ, θ))),
                 pe -> ((ψ, θ) -> pe.rzphi_nu((ψ, θ))),
                 pe -> ((ψ, θ) -> pe.rzphi_jac((ψ, θ)))]
rzphi_names   = ["r²=(R-R₀)²+(Z-Z₀)²", "angle offset η/2π−θ", "ν (toroidal shift)", "Jacobian"]
for (sname, getter) in zip(rzphi_names, rzphi_getters)
    println("  $sname:")
    for k in compare_keys
        fn_ref = getter(pe_ref); fn_k = getter(pes[k])
        for θ in theta_select
            y_ref = [fn_ref(ψ, θ) for ψ in psi_full]
            y_k   = [fn_k(ψ, θ)   for ψ in psi_full]
            Δ = y_ref .- y_k
            Δc = any(mask_core) ? Δ[mask_core] : [0.0]
            Δe = any(mask_edge) ? Δ[mask_edge] : [0.0]
            @printf("    vs %-30s  θ=%.2f  core max|Δ|=%.3e  edge max|Δ|=%.3e  edge rms=%.3e\n",
                variant_label[k], θ, maximum(abs.(Δc)), maximum(abs.(Δe)), sqrt(mean(Δe .^ 2)))
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  EQFUN PHYSICS SPLINES AT SELECT THETA SLICES
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("4. Physics (eqfun) Spline Profiles at Select θ Values")
println("=" ^ 70)

p = EqPlot.plot_eqfun_splines(pe_ref; linestyle=variant_style[ref_key],
        label=variant_label[ref_key])
for k in compare_keys
    EqPlot.plot_eqfun_splines!(p, pes[k]; linestyle=variant_style[k], label=variant_label[k])
end
savefig(p, joinpath(outdir, "eqfun_splines.png"))

eqfun_getters = [pe -> ((ψ, θ) -> pe.eqfun_B((ψ, θ))),
                 pe -> ((ψ, θ) -> pe.eqfun_metric1((ψ, θ))),
                 pe -> ((ψ, θ) -> pe.eqfun_metric2((ψ, θ)))]
eqfun_names   = ["|B| (total field [T])", "metric1 (e₁·e₂+q·e₃·e₁)/JB²", "metric2 (e₂·e₃+q·e₃²)/JB²"]
for (ename, getter) in zip(eqfun_names, eqfun_getters)
    println("  $ename:")
    for k in compare_keys
        fn_ref = getter(pe_ref); fn_k = getter(pes[k])
        for θ in theta_select
            y_ref = [fn_ref(ψ, θ) for ψ in psi_full]
            y_k   = [fn_k(ψ, θ)   for ψ in psi_full]
            Δ = y_ref .- y_k
            Δc = any(mask_core) ? Δ[mask_core] : [0.0]
            Δe = any(mask_edge) ? Δ[mask_edge] : [0.0]
            @printf("    vs %-30s  θ=%.2f  core max|Δ|=%.3e  edge max|Δ|=%.3e  edge rms=%.3e\n",
                variant_label[k], θ, maximum(abs.(Δc)), maximum(abs.(Δe)), sqrt(mean(Δe .^ 2)))
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  STABILITY (et[1])
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("5. Stability: eigenmode energy et[1] for each Jacobian variant")
println("=" ^ 70)

# Load FFS and Wall config from the example gpec.toml — same config for all variants.
# The equilibrium (and thus Jacobian) is the only thing that differs.
let
    inputs   = TOML.parsefile(config_path)
    ffs_cfg  = inputs["ForceFreeStates"]
    wall_cfg = get(inputs, "Wall", Dict())

    # Replicate the FFS pipeline from GeneralizedPerturbedEquilibrium.main() in-process,
    # using the already-computed PlasmaEquilibrium objects to avoid re-running the ODE solver.
    function run_stability(equil)
        intr = ForceFreeStates.ForceFreeStatesInternal(; dir_path=example_path)
        ctrl = ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in ffs_cfg)...)
        intr.wall_settings = Vacuum.WallShapeSettings(;
            (Symbol(k) => v for (k, v) in wall_cfg)...)

        ctrl.delta_mhigh *= 2  # match main() convention

        ForceFreeStates.sing_lim!(intr, ctrl, equil)
        if ctrl.set_psilim_via_dmlim && ctrl.psiedge < intr.psilim
            ctrl.psiedge = 1.0
        end

        profiles_xs = equil.profiles.xs
        locstab_fs  = zeros(Float64, length(profiles_xs), 5)
        ctrl.mer_flag && ForceFreeStates.mercier_scan!(locstab_fs, equil)
        intr.locstab = cubic_interp(profiles_xs, locstab_fs;
            bc=CubicFit(), extrap=ExtendExtrap())

        intr.nlow  = ctrl.nn_low  == 0 ? ctrl.nn_high : max(1, ctrl.nn_low)
        intr.nhigh = ctrl.nn_high == 0 ? intr.nlow    : ctrl.nn_high
        intr.npert = intr.nhigh - intr.nlow + 1

        ForceFreeStates.sing_find!(intr, equil)

        if ctrl.cyl_flag
            intr.mlow  = ctrl.delta_mlow
            intr.mhigh = ctrl.delta_mhigh
        else
            intr.mlow  = trunc(Int, min(intr.nlow * equil.params.qmin, 0)) - 4 - ctrl.delta_mlow
            intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
        end
        intr.mpert = intr.mhigh - intr.mlow + 1
        intr.mband = min(max(intr.mpert - 1 - ctrl.delta_mband, 0), intr.mpert - 1)
        intr.numpert_total = intr.mpert * intr.npert

        metric = ForceFreeStates.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)
        ffit   = ForceFreeStates.make_matrix(equil, intr, metric)
        odet   = ForceFreeStates.eulerlagrange_integration(ctrl, equil, ffit, intr)
        vac    = ForceFreeStates.free_run!(odet, ctrl, equil, ffit, intr)
        return real(vac.et[1])
    end

    et_vals = Dict{String, Float64}()
    for k in active_keys
        print("  Running FFS for variant: $(variant_label[k]) ... ")
        flush(stdout)
        t = @elapsed et_vals[k] = run_stability(pes[k])
        @printf("et[1] = %.6f  (%.1f s)\n", et_vals[k], t)
    end

    println()
    et_ref = et_vals[ref_key]
    @printf("  %-30s  %10s  %10s\n", "Variant", "et[1]", "Δet[1] %")
    for k in active_keys
        pct = 100.0 * (et_vals[k] - et_ref) / abs(et_ref)
        @printf("  %-30s  %10.6f  %+10.4f%%\n", variant_label[k], et_vals[k], pct)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("6. Summary: axis / edge parameter comparison")
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
