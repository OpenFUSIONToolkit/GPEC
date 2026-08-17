# Sanity check: compute_delta_prime_from_ca! vs inline Δ' from riccati_cross_ideal_singular_surf!
#
# riccati_cross_ideal_singular_surf! computes Δ' inline at each singular surface crossing
# using the diagonal formula (no Gaussian reduction permutation):
#   Δ'[s] = (ca_r[ipert_res, ipert_res, 2, s] - ca_l[ipert_res, ipert_res, 2, s]) / (4π²·ψ₀)
#
# compute_delta_prime_from_ca! applies the identical formula post-hoc from the stored
# ca_l/ca_r arrays. Since both operate on the same data with the same formula, results
# should match to floating-point precision (not just approximately — exactly).
#
# This verifies that compute_delta_prime_from_ca! is a correct standalone implementation
# of the Δ' formula that can be used for testing or alternative integration drivers.
#
# Usage (from JPEC_main root):
#   julia --project=. benchmarks/benchmark_delta_prime_methods.jl

using LinearAlgebra, Printf, TOML
using GeneralizedPerturbedEquilibrium

const FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates

function setup_and_run_solovev()
    ex = joinpath(@__DIR__, "..", "test", "test_data", "regression_solovev_ideal_example")
    inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
    inputs["ForceFreeStates"]["verbose"] = false
    inputs["ForceFreeStates"]["integrator"] = "riccati"
    intr = FFS.ForceFreeStatesInternal(; dir_path=ex)
    ctrl = FFS.ForceFreeStatesControl(;
        (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
    eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
    equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config)
    intr.wall_settings = GeneralizedPerturbedEquilibrium.Vacuum.WallShapeSettings(;
        (Symbol(k) => v for (k, v) in inputs["Wall"])...)
    FFS.sing_lim!(intr, ctrl, equil)
    intr.nlow = ctrl.nn_low
    intr.nhigh = ctrl.nn_high
    intr.npert = 1
    FFS.sing_find!(intr, equil)
    intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
    intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
    intr.mpert = intr.mhigh - intr.mlow + 1
    intr.numpert_total = intr.mpert * intr.npert
    metric = FFS.make_metric(equil, intr.mpert)
    ffit = FFS.make_matrix(equil, intr, metric)
    odet, _, _, _ = FFS.riccati_eulerlagrange_integration(ctrl, equil, ffit, intr)
    return ctrl, equil, ffit, intr, odet
end

println("\n=== compute_delta_prime_from_ca! consistency check ===")
println("Verifies the standalone Δ' formula matches the inline Riccati crossing computation.")
println("Expected error: exactly zero (same formula, same data).\n")

ctrl, equil, ffit, intr, odet = setup_and_run_solovev()
msing = intr.msing

# Capture Δ' values set inline by riccati_cross_ideal_singular_surf! during integration
delta_prime_inline = [copy(intr.sing[s].delta_prime) for s in 1:msing]

# Now call compute_delta_prime_from_ca! — it reads the same ca_l/ca_r arrays and
# overwrites intr.sing[s].delta_prime using the identical diagonal formula
FFS.compute_delta_prime_from_ca!(odet, intr, equil)

println("  N=$(intr.numpert_total) modes, $msing singular surfaces\n")
@printf("  %6s  %4s  %4s  %22s  %22s  %12s\n",
    "Surf", "m", "n", "Δ' (inline)", "Δ' (from_ca)", "abs diff")
println("  " * "-"^76)

max_absdiff = let max_absdiff = 0.0
    for s in 1:msing
        sing = intr.sing[s]
        dp_from_ca = intr.sing[s].delta_prime
        for i in eachindex(delta_prime_inline[s])
            dp_il = delta_prime_inline[s][i]
            dp_fc = dp_from_ca[i]
            absdiff = abs(dp_fc - dp_il)
            max_absdiff = max(max_absdiff, absdiff)
            @printf("  %6d  %4d  %4d  %22.6f%+.6fi  %22.6f%+.6fi  %12.4e\n",
                s, sing.m[i], sing.n[i],
                real(dp_il), imag(dp_il),
                real(dp_fc), imag(dp_fc),
                absdiff)
        end
    end
    max_absdiff
end

println()
if max_absdiff == 0.0
    println("PASSED — Δ' values are bit-for-bit identical (max abs diff = 0.0)")
elseif max_absdiff < 1e-14
    @printf("PASSED — max abs diff = %.2e (floating-point rounding only)\n", max_absdiff)
else
    @printf("FAILED — max abs diff = %.2e (expected exact agreement)\n", max_absdiff)
    exit(1)
end
println()
