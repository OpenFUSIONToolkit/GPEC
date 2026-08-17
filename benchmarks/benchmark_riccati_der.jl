# Sanity check: riccati_der! correctly evaluates the explicit Riccati ODE.
#
# riccati_der! implements [Glasser 2018 Phys. Plasmas 25, 032507, Eq. 19]:
#   dS/dψ = w†·F̄⁻¹·w - S·Ḡ·S,   w = Q - K̄·S
#
# where Q = diag(1/(m - n·q)), F̄ = L·L† (Cholesky), K̄ and Ḡ are the MHD
# metric matrices evaluated at ψ.
#
# NOTE: The identity between this Riccati ODE and the EL chain rule
#   dS/dψ = dU₁·U₂⁻¹ - S·dU₂·U₂⁻¹
# holds ONLY for Hermitian S (physical states evolved from the axis, where
# S†=S is preserved by the EL symmetry). For arbitrary non-Hermitian (U₁, U₂),
# the two expressions differ — so this script compares riccati_der! against the
# explicit formula rather than against sing_der!.
#
# Usage (from JPEC_main root):
#   julia --project=. benchmarks/benchmark_riccati_der.jl

using LinearAlgebra, Random, Printf, TOML
using GeneralizedPerturbedEquilibrium

const FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates

function setup_solovev()
    ex = joinpath(@__DIR__, "..", "test", "test_data", "regression_solovev_ideal_example")
    inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
    inputs["ForceFreeStates"]["verbose"] = false
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
    return ctrl, equil, ffit, intr
end

# Evaluate the Riccati RHS explicitly from splines: dS = w†·F̄⁻¹·w - S·Ḡ·S
function riccati_rhs_manual(S, psi, equil, ffit, intr)
    N = intr.numpert_total
    L = zeros(ComplexF64, N, N)
    Kmat = zeros(ComplexF64, N, N)
    Gmat = zeros(ComplexF64, N, N)
    ffit.fmats_lower(vec(L), psi; hint=ffit._hint)
    ffit.kmats(vec(Kmat), psi; hint=ffit._hint)
    ffit.gmats(vec(Gmat), psi; hint=ffit._hint)

    q = equil.profiles.q_spline(psi)
    singfac = vec(1.0 ./ ((intr.mlow:intr.mhigh) .- q .* (intr.nlow:intr.nhigh)'))

    # w = Q - K̄·S  (Q is diagonal; add only the diagonal entries)
    w = -Kmat * S
    for i in 1:N
        w[i, i] += singfac[i]
    end

    # v = F̄⁻¹·w  via stored Cholesky factor L (L·L† = F̄)
    v = copy(w)
    ldiv!(LowerTriangular(L), v)
    ldiv!(UpperTriangular(L'), v)

    return adjoint(w) * v - S * Gmat * S
end

println("\n=== riccati_der! formula verification ===")
println("Verifies riccati_der! output matches manual evaluation of Glasser 2018 Eq. 19.")
println("Test state: Hermitian S (physical constraint). Expected error: ~machine epsilon.\n")

ctrl, equil, ffit, intr = setup_solovev()
N = intr.numpert_total

odet = FFS.OdeState(N, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
FFS.initialize_el_at_axis!(odet, ctrl, equil.profiles, intr)
chunks = FFS.chunk_el_integration_bounds(odet, ctrl, intr)

# 30% into each chunk: well inside the interval, away from singularities at psi_end
test_psis = [c.psi_start + 0.3 * (c.psi_end - c.psi_start) for c in chunks]

println("  N=$N modes, $(length(test_psis)) test ψ points (30% into each chunk)\n")
@printf("  %8s  %14s  %14s  %12s\n", "ψ", "‖dS_manual‖", "‖dS_ric‖", "rel error")
println("  " * "-"^54)

rng = Random.MersenneTwister(42)
threshold = 1e-10

max_err = let max_err = 0.0
    for psi in test_psis
        # Hermitian S: physical Riccati matrix is Hermitian (preserved by EL symmetry)
        A = randn(rng, ComplexF64, N, N)
        S = (A + A') / 2   # Hermitian by construction

        # Manual RHS
        dS_manual = riccati_rhs_manual(S, psi, equil, ffit, intr)

        # riccati_der! RHS
        u_ric = zeros(ComplexF64, N, N, 2)
        du_ric = zeros(ComplexF64, N, N, 2)
        u_ric[:, :, 1] .= S
        u_ric[:, :, 2] .= Matrix{ComplexF64}(I, N, N)
        dummy_chunk = FFS.IntegrationChunk(psi, psi, false, 0, 1)
        params = (ctrl, equil, ffit, intr, odet, dummy_chunk)
        FFS.riccati_der!(du_ric, u_ric, params, psi)
        dS_ric = du_ric[:, :, 1]

        ref = max(norm(dS_manual), 1e-10)
        err = norm(dS_ric - dS_manual) / ref
        max_err = max(max_err, err)
        status = err < threshold ? "" : "  ← FAIL"
        @printf("  %8.4f  %14.4e  %14.4e  %12.4e%s\n", psi, norm(dS_manual), norm(dS_ric), err, status)
    end
    max_err
end

println()
if max_err < threshold
    @printf("PASSED — max rel error = %.2e (threshold %.0e)\n", max_err, threshold)
else
    @printf("FAILED — max rel error = %.2e exceeds threshold %.0e\n", max_err, threshold)
    exit(1)
end
println()
