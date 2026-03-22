"""
equil_jac_blend_convergence.jl

Convergence study for the Jacobian-blend stability benchmark.

For each of three resolution parameters (mpsi, mtheta, delta_mhigh), sweeps
values while holding the other two fixed, and reports et[1] for each Jacobian
variant.  Since all blended variants and pure equal_arc share the same boundary
Jacobian, their et[1] values should converge to the same value as resolution
increases; remaining differences are purely numerical (quadrature, ODE steps,
spline accuracy), not physics.

Baseline (nominal) settings:
  mpsi        = 128  (ldp grid)
  mtheta      = 256
  delta_mhigh = 16   (config value 8, doubled inside run_stability per main() convention)

Usage:
  julia --project=. benchmarks/equil_jac_blend_convergence.jl [example_path]
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium
using TOML, Printf, Statistics
import FastInterpolations: cubic_interp, CubicFit, ExtendExtrap

example_path = length(ARGS) > 0 ? ARGS[1] :
    joinpath(@__DIR__, "../examples/DIIID-like_ideal_example")
config_path  = joinpath(example_path, "gpec.toml")

println("=" ^ 70)
println("Jacobian Blend Convergence Study")
println("Example : $example_path")
println("=" ^ 70)

# ─── Config factory (same as jac_blend_comparison, plus mtheta) ────────────────
function make_blend_config(path, psihigh;
        jac_type="hamada", jac_type_edge="hamada",
        psi_jac_blend_start=1.0, psi_jac_blend_end=1.0,
        mpsi=128, grid_type="ldp", mtheta=256)
    raw = TOML.parsefile(path)
    eq  = raw["Equilibrium"]
    eq["psihigh"]             = psihigh
    eq["jac_type"]            = jac_type
    eq["jac_type_edge"]       = jac_type_edge
    eq["psi_jac_blend_start"] = psi_jac_blend_start
    eq["psi_jac_blend_end"]   = psi_jac_blend_end
    eq["mpsi"]                = mpsi
    eq["grid_type"]           = grid_type
    eq["mtheta"]              = mtheta
    return Equilibrium.EquilibriumConfig(eq, dirname(path))
end

# ─── Jacobian variants ─────────────────────────────────────────────────────────
variants = [
    ("hamada",        "hamada",
        (jac_type="hamada",)),
    ("blend_wide",    "blend [0.80,0.98]",
        (jac_type="hamada", jac_type_edge="equal_arc",
         psi_jac_blend_start=0.80, psi_jac_blend_end=0.98)),
    ("blend_narrow",  "blend [0.95,0.99]",
        (jac_type="hamada", jac_type_edge="equal_arc",
         psi_jac_blend_start=0.95, psi_jac_blend_end=0.99)),
    ("equal_arc",     "equal_arc",
        (jac_type="equal_arc",)),
]
variant_keys = [v[1] for v in variants]

# Variants with equal_arc boundary Jacobian at any psihigh ≤ 0.993
eq_arc_keys = ["blend_wide", "blend_narrow", "equal_arc"]
eq_arc_ref  = "equal_arc"

# ─── FFS pipeline ──────────────────────────────────────────────────────────────
function run_stability(equil, ffs_cfg, wall_cfg; delta_mhigh_override=nothing)
    intr = ForceFreeStates.ForceFreeStatesInternal(; dir_path=example_path)
    ctrl = ForceFreeStates.ForceFreeStatesControl(;
        (Symbol(k) => v for (k, v) in ffs_cfg)...)
    intr.wall_settings = Vacuum.WallShapeSettings(;
        (Symbol(k) => v for (k, v) in wall_cfg)...)

    ctrl.delta_mhigh = something(delta_mhigh_override, ctrl.delta_mhigh * 2)

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

# ─── Print one convergence table ───────────────────────────────────────────────
# param_vals: vector of parameter values tested
# et_table:   Dict(key => Vector{Float64}) with et[1] per param value
# param_name: string for the column header
function print_convergence_table(param_name, param_vals, et_table, eq_arc_keys, eq_arc_ref)
    keys_shown = filter(k -> haskey(et_table, k), variant_keys)
    labels     = [variants[findfirst(v -> v[1] == k, variants)][2] for k in keys_shown]

    # Header
    w = 14
    @printf("\n  %-*s", w, param_name)
    for lbl in labels; @printf("  %*s", w, lbl); end
    println()
    @printf("  %s", "-" ^ w)
    for _ in labels; @printf("  %s", "-" ^ w); end
    println()

    for (i, pval) in enumerate(param_vals)
        @printf("  %-*s", w, string(pval))
        ref_et = get(et_table[eq_arc_ref], i, NaN)
        for k in keys_shown
            et = get(et_table[k], i, NaN)
            if k == eq_arc_ref || !(k in eq_arc_keys)
                @printf("  %*.6f", w, et)
            else
                pct = isnan(ref_et) ? NaN : 100.0 * (et - ref_et) / abs(ref_et)
                @printf("  %*.4f%%", w-1, pct)
            end
        end
        println()
    end

    println()
    println("  Columns: et[1] for hamada and equal_arc (absolute);")
    println("           Δ% vs equal_arc for blended variants (numerical error).")
end

# ─── Load config ───────────────────────────────────────────────────────────────
inputs   = TOML.parsefile(config_path)
ffs_cfg  = inputs["ForceFreeStates"]
wall_cfg = get(inputs, "Wall", Dict())
psihigh  = inputs["Equilibrium"]["psihigh"]

# Nominal settings
NOM_MPSI        = 128
NOM_MTHETA      = 256
NOM_DELTA_MHIGH = 16   # config value 8, doubled per main() convention

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONVERGENCE IN mpsi
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("1. mpsi convergence  (mtheta=$NOM_MTHETA, delta_mhigh=$NOM_DELTA_MHIGH)")
println("=" ^ 70)
println("   Rebuilds equilibrium for each mpsi; all other settings nominal.\n")

mpsi_vals = [32, 64, 128, 256]
et_mpsi   = Dict(k => Float64[] for k in variant_keys)

for mpsi in mpsi_vals
    println("  --- mpsi = $mpsi ---")
    pes = Dict{String,Any}()
    for (key, label, kwargs) in variants
        cfg = make_blend_config(config_path, psihigh;
            mpsi=mpsi, grid_type="ldp", mtheta=NOM_MTHETA, kwargs...)
        try
            pes[key] = setup_equilibrium(cfg)
        catch e
            @printf("    FAILED (%s): %s\n", label, e)
        end
    end
    for k in variant_keys
        if haskey(pes, k)
            t = @elapsed et = run_stability(pes[k], ffs_cfg, wall_cfg;
                delta_mhigh_override=NOM_DELTA_MHIGH)
            push!(et_mpsi[k], et)
            @printf("    %-16s  et[1] = %.6f  (%.1f s)\n", variants[findfirst(v->v[1]==k,variants)][2], et, t)
        else
            push!(et_mpsi[k], NaN)
        end
    end
end

print_convergence_table("mpsi", mpsi_vals, et_mpsi, eq_arc_keys, eq_arc_ref)

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CONVERGENCE IN mtheta
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("2. mtheta convergence  (mpsi=$NOM_MPSI, delta_mhigh=$NOM_DELTA_MHIGH)")
println("=" ^ 70)
println("   Rebuilds equilibrium for each mtheta; all other settings nominal.\n")

# Minimum mtheta: FourierCoefficients requires mtheta/2 ≥ mband = mpert-1.
# With qmax≈5.5, delta_mhigh=16, mpert=34, mband=33 → mtheta_min=66.
mtheta_vals = [128, 256, 512, 1024]
et_mtheta   = Dict(k => Float64[] for k in variant_keys)

for mtheta in mtheta_vals
    println("  --- mtheta = $mtheta ---")
    pes = Dict{String,Any}()
    for (key, label, kwargs) in variants
        cfg = make_blend_config(config_path, psihigh;
            mpsi=NOM_MPSI, grid_type="ldp", mtheta=mtheta, kwargs...)
        try
            pes[key] = setup_equilibrium(cfg)
        catch e
            @printf("    FAILED (%s): %s\n", label, e)
        end
    end
    for k in variant_keys
        if haskey(pes, k)
            t = @elapsed et = run_stability(pes[k], ffs_cfg, wall_cfg;
                delta_mhigh_override=NOM_DELTA_MHIGH)
            push!(et_mtheta[k], et)
            @printf("    %-16s  et[1] = %.6f  (%.1f s)\n", variants[findfirst(v->v[1]==k,variants)][2], et, t)
        else
            push!(et_mtheta[k], NaN)
        end
    end
end

print_convergence_table("mtheta", mtheta_vals, et_mtheta, eq_arc_keys, eq_arc_ref)

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CONVERGENCE IN delta_mhigh
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("3. delta_mhigh convergence  (mpsi=$NOM_MPSI, mtheta=$NOM_MTHETA)")
println("=" ^ 70)
println("   Reuses nominal equilibria; only delta_mhigh changes in FFS.\n")

dmhigh_vals = [4, 8, 16, 32]
et_dmhigh   = Dict(k => Float64[] for k in variant_keys)

# Build nominal equilibria once
println("  Building nominal equilibria (mpsi=$NOM_MPSI, mtheta=$NOM_MTHETA)...")
pes_nom = Dict{String,Any}()
for (key, label, kwargs) in variants
    cfg = make_blend_config(config_path, psihigh;
        mpsi=NOM_MPSI, grid_type="ldp", mtheta=NOM_MTHETA, kwargs...)
    try
        pes_nom[key] = setup_equilibrium(cfg)
        println("  Done: $label")
    catch e
        @printf("  FAILED (%s): %s\n", label, e)
    end
end

for dmh in dmhigh_vals
    println("\n  --- delta_mhigh = $dmh ---")
    for k in variant_keys
        if haskey(pes_nom, k)
            t = @elapsed et = run_stability(pes_nom[k], ffs_cfg, wall_cfg;
                delta_mhigh_override=dmh)
            push!(et_dmhigh[k], et)
            @printf("    %-16s  et[1] = %.6f  (%.1f s)\n", variants[findfirst(v->v[1]==k,variants)][2], et, t)
        else
            push!(et_dmhigh[k], NaN)
        end
    end
end

print_convergence_table("delta_mhigh", dmhigh_vals, et_dmhigh, eq_arc_keys, eq_arc_ref)

println("=" ^ 70)
println("Done.")
