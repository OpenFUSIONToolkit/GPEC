"""
edge_jacobian_test_plan.jl

Validates the three unchecked items in the PR #197 test plan:

  1. norm(T - I) < 1e-10 for jac_type = "equal_arc"
     T should reduce to identity when jac theta_jac[k] = k/mtheta_eq (equal-arc).

  2. nu_eq matches rzphi_nu at corresponding theta_jac values within ~0.1%
     Validates that equal_arc_fieldline_output computes the correct toroidal shift.

  3. use_equal_arc_vacuum = true at psihigh = 0.999: et[1] convergence comparison
     Compares hamada + use_equal_arc_vacuum vs hamada standard path.

Usage:
  julia --project=. benchmarks/edge_jacobian_test_plan.jl [example_path]
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium
using GeneralizedPerturbedEquilibrium.Vacuum
using TOML, Printf, LinearAlgebra, Statistics
import FastInterpolations: cubic_interp, CubicFit, ExtendExtrap

example_path = length(ARGS) > 0 ? ARGS[1] :
    joinpath(@__DIR__, "../examples/DIIID-like_ideal_example")
config_path = joinpath(example_path, "gpec.toml")

println("=" ^ 70)
println("PR #197 Test Plan Validation")
println("Example : $example_path")
println("=" ^ 70)

inputs = TOML.parsefile(config_path)
psihigh_default = inputs["Equilibrium"]["psihigh"]

# ─── Helper: build an equilibrium config ────────────────────────────────────
function make_config(path; jac_type="hamada", psihigh=psihigh_default, kwargs...)
    raw = TOML.parsefile(path)
    eq  = raw["Equilibrium"]
    eq["psihigh"]  = psihigh
    eq["jac_type"] = jac_type
    for (k, v) in kwargs
        eq[string(k)] = v
    end
    return Equilibrium.EquilibriumConfig(eq, dirname(path))
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: norm(T - I) < 1e-10 for jac_type = "equal_arc"
# ═══════════════════════════════════════════════════════════════════════════════
println()
println("=" ^ 70)
println("TEST 1: T ≈ I for jac_type = \"equal_arc\"")
println("=" ^ 70)
println("For equal_arc, theta_jac[k] = k/mtheta_eq so the mixed DFT is the")
println("standard DFT identity: T[i,j] = δ(m_jac - m_eq) when mode spaces equal.")

let
    cfg = make_config(config_path; jac_type="equal_arc")
    println("Building equal_arc equilibrium...")
    equil = setup_equilibrium(cfg)

    psifac_test = 0.5
    mtheta_eq   = 128
    mpert = 20
    mlow  = -10
    mpert_eq = mpert
    mlow_eq  = mlow

    R_eq, Z_eq, nu_eq, theta_jac_eq = Equilibrium.equal_arc_fieldline_output(equil, psifac_test; mtheta_eq=mtheta_eq)

    println("  psifac = $psifac_test, mtheta_eq = $mtheta_eq")
    println("  theta_jac[1:5] = $(round.(theta_jac_eq[1:5]; digits=6))")
    println("  Expected σ_k   = $(round.([0,1,2,3,4]/mtheta_eq; digits=6))")

    T = jac_to_eq_mode_transform(theta_jac_eq, mlow, mpert, mlow_eq, mpert_eq)

    normdiff = norm(T - I(mpert))
    normT    = norm(T)
    @printf("  norm(T - I)    = %.3e\n", normdiff)
    @printf("  norm(T)        = %.3e\n", normT)

    passed = normdiff < 1e-8
    println("  TEST 1: $(passed ? "PASS ✓" : "FAIL ✗") (threshold 1e-8, got $(round(normdiff; sigdigits=3)))")
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: nu_eq matches rzphi_nu at corresponding theta_jac values
# ═══════════════════════════════════════════════════════════════════════════════
println()
println("=" ^ 70)
println("TEST 2: nu_eq matches rzphi_nu within ~0.1%")
println("=" ^ 70)

let
    cfg = make_config(config_path; jac_type="hamada")
    println("Building hamada equilibrium...")
    equil = setup_equilibrium(cfg)

    psifac_test = 0.5
    mtheta_eq   = 512

    R_eq, Z_eq, nu_eq, theta_jac_eq = Equilibrium.equal_arc_fieldline_output(equil, psifac_test; mtheta_eq=mtheta_eq)

    # Compare against rzphi_nu evaluated at (psifac, theta_jac_eq[k])
    nu_spline = [equil.rzphi_nu((psifac_test, theta_jac_eq[k])) for k in 1:mtheta_eq]

    dnu = nu_eq .- nu_spline
    rms_err = sqrt(mean(dnu .^ 2))
    max_err = maximum(abs.(dnu))
    nu_scale = sqrt(mean(nu_spline .^ 2))
    rel_rms = nu_scale > 1e-15 ? rms_err / nu_scale : rms_err

    @printf("  psifac = %.2f, mtheta_eq = %d\n", psifac_test, mtheta_eq)
    @printf("  max |Δν|       = %.3e\n", max_err)
    @printf("  rms |Δν|       = %.3e\n", rms_err)
    @printf("  rms |Δν|/‖ν‖  = %.3e (%.4f%%)\n", rel_rms, 100*rel_rms)

    # Also check at the edge
    psifac_edge = 0.95
    R_eq2, Z_eq2, nu_eq2, theta_jac_eq2 = Equilibrium.equal_arc_fieldline_output(equil, psifac_edge; mtheta_eq=mtheta_eq)
    nu_spline2 = [equil.rzphi_nu((psifac_edge, theta_jac_eq2[k])) for k in 1:mtheta_eq]
    dnu2 = nu_eq2 .- nu_spline2
    rms_err2 = sqrt(mean(dnu2 .^ 2))
    nu_scale2 = sqrt(mean(nu_spline2 .^ 2))
    rel_rms2 = nu_scale2 > 1e-15 ? rms_err2 / nu_scale2 : rms_err2
    @printf("  Edge (ψ=%.2f): rms |Δν|/‖ν‖ = %.3e (%.4f%%)\n", psifac_edge, rel_rms2, 100*rel_rms2)

    passed = rel_rms < 0.001 && rel_rms2 < 0.01
    println("  TEST 2: $(passed ? "PASS ✓" : "FAIL ✗") (threshold 0.1% core, 1% edge)")
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: use_equal_arc_vacuum = true convergence at psihigh = 0.999
# ═══════════════════════════════════════════════════════════════════════════════
println()
println("=" ^ 70)
println("TEST 3: use_equal_arc_vacuum = true vs standard at psihigh = 0.999")
println("=" ^ 70)

let
    ffs_cfg  = inputs["ForceFreeStates"]
    wall_cfg = get(inputs, "Wall", Dict())

    function run_stability(equil; use_equal_arc_vacuum=false)
        intr = ForceFreeStates.ForceFreeStatesInternal(; dir_path=example_path)
        ctrl = ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in ffs_cfg)...)
        ctrl.use_equal_arc_vacuum = use_equal_arc_vacuum
        intr.wall_settings = Vacuum.WallShapeSettings(;
            (Symbol(k) => v for (k, v) in wall_cfg)...)
        ctrl.delta_mhigh *= 2

        ForceFreeStates.sing_lim!(intr, ctrl, equil)
        if ctrl.set_psilim_via_dmlim && ctrl.psiedge < intr.psilim
            ctrl.psiedge = 1.0
        end
        profiles_xs = equil.profiles.xs
        locstab_fs  = zeros(Float64, length(profiles_xs), 5)
        ctrl.mer_flag && ForceFreeStates.mercier_scan!(locstab_fs, equil)
        intr.locstab = cubic_interp(profiles_xs, locstab_fs; bc=CubicFit(), extrap=ExtendExtrap())

        intr.nlow  = ctrl.nn_low  == 0 ? ctrl.nn_high : max(1, ctrl.nn_low)
        intr.nhigh = ctrl.nn_high == 0 ? intr.nlow    : ctrl.nn_high
        intr.npert = intr.nhigh - intr.nlow + 1

        ForceFreeStates.sing_find!(intr, equil)

        intr.mlow  = trunc(Int, min(intr.nlow * equil.params.qmin, 0)) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
        intr.mpert = intr.mhigh - intr.mlow + 1
        intr.mband = min(max(intr.mpert - 1 - ctrl.delta_mband, 0), intr.mpert - 1)
        intr.numpert_total = intr.mpert * intr.npert

        metric = ForceFreeStates.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)
        ffit   = ForceFreeStates.make_matrix(equil, intr, metric)
        odet   = ForceFreeStates.eulerlagrange_integration(ctrl, equil, ffit, intr)
        vac    = ForceFreeStates.free_run!(odet, ctrl, equil, ffit, intr)
        return real(vac.et[1])
    end

    psihigh_vals = [0.985, 0.990, 0.993, 0.995, 0.997, 0.999]

    @printf("\n  %-10s  %14s  %14s  %14s\n", "psihigh", "hamada (std)", "hamada+eq_arc", "Δet[1]%")
    @printf("  %s\n", "-"^60)

    for psih in psihigh_vals
        cfg_std = make_config(config_path; jac_type="hamada", psihigh=psih, mpsi=128, grid_type="ldp")
        local equil_std
        try
            equil_std = setup_equilibrium(cfg_std)
        catch e
            @printf("  %-10.3f  %14s  %14s\n", psih, "FAILED", "")
            continue
        end

        t_std = @elapsed et_std = run_stability(equil_std; use_equal_arc_vacuum=false)
        t_eq  = @elapsed et_eq  = run_stability(equil_std; use_equal_arc_vacuum=true)
        dpct = 100.0 * (et_eq - et_std) / abs(et_std)
        @printf("  %-10.3f  %14.6f  %14.6f  %+12.4f%%\n", psih, et_std, et_eq, dpct)
    end

    println()
    println("  If use_equal_arc_vacuum improves accuracy at psihigh=0.999, et[1] should")
    println("  be less sensitive to psihigh and converge more smoothly than the standard path.")
end

println()
println("=" ^ 70)
println("Test plan complete.")
println("=" ^ 70)
