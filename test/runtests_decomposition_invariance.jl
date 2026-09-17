using Test
using TOML

# Decomposition invariance of the Riccati Δ′ path.
#
# The chunked propagator driver reassociates the fundamental-matrix products whenever the chunk
# decomposition changes, and floating-point products do not reassociate exactly in general — so
# Δ′ reproducibility is a real end-to-end claim, beyond the structural chunk-count invariance
# pinned by the unit tests in runtests_parallel_integration.jl. This file asserts it: the Δ′
# matrix is bit-identical when the same integration is cut into a genuinely different set of
# chunks. Driven through the public solve API so it exercises the production path itself.

const GP_TI = GeneralizedPerturbedEquilibrium

"""
Solve the DIII-D-like deck with the Riccati integrator at a given chunk count (`nchunks = 0` is
the msing-derived auto target) and return the published Δ′ matrix alongside the leading energy
eigenvalue, which is used only as a witness that the two decompositions really differed.
"""
function _solve_at_nchunks(dir::String, nchunks::Int)
    inputs = TOML.parsefile(joinpath(dir, "gpec.toml"))
    ffs_in = inputs["ForceFreeStates"]
    eq_config = GP_TI.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], dir)
    equil = GP_TI.Equilibrium.setup_equilibrium(eq_config, nothing)
    if GP_TI.Equilibrium.wants_two_pass(eq_config)
        mand = GP_TI.ForceFreeStates.rational_psi_nodes(equil; nlow=ffs_in["nn_low"], nhigh=ffs_in["nn_high"])
        psi_nodes = GP_TI.Equilibrium.refined_psi_grid(equil; tau=eq_config.psi_accuracy, mandatory=mand)
        rerun_input = GP_TI.Equilibrium.build_direct_from_ingest(eq_config, equil.ingest)
        equil = GP_TI.Equilibrium.setup_equilibrium(eq_config, rerun_input; override_psi_nodes=psi_nodes)
    end

    # Every ForceFreeStatesControl key from the deck except the ones the problem or the
    # integrator owns: nn_low/nn_high come from `nn`, nchunks and the formalism from the alg.
    ctrl_kwargs = Dict(Symbol(k) => v for (k, v) in ffs_in
                       if !(k in ("nn_low", "nn_high", "nchunks", "integrator")))
    ctrl_kwargs[:verbose] = false
    ctrl_kwargs[:write_outputs_to_HDF5] = false

    wall = GP_TI.Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in inputs["Wall"])...)
    prob = GP_TI.EulerLagrangeProblem(equil; nn=ffs_in["nn_low"], wall=wall, dir_path=dir, ctrl_kwargs...)
    return GP_TI.solve(prob, GP_TI.ForceFreeStates.Riccati(; nchunks=nchunks))
end

@testset "Decomposition invariance of the Riccati Δ′ path" begin
    # The deck whose Δ′ the regression harness pins, and where the BVP Δ′ is well-conditioned
    # (Solovev sits near marginal stability and its BVP Δ′ is pathological there).
    dir = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
    auto = _solve_at_nchunks(dir, 0)
    @test auto.delta_prime !== nothing
    msing = size(auto.delta_prime.matrix, 1)

    # The nchunks=0 target, mirroring balance_integration_chunks' internal formula (as
    # runtests_parallel_integration.jl does). Invariance is asserted ABOVE this floor only:
    # fewer chunks than the msing-derived minimum is not a different decomposition but a
    # structurally deficient one (the floor gives the rational-surface crossings room).
    auto_target = max(2 * msing + 3, 8 * (msing + 1) + msing)
    finer = _solve_at_nchunks(dir, auto_target + 11)
    @test finer.delta_prime !== nothing

    # Premise check: the two runs must be genuinely different computations, or the equality
    # below is vacuous. et[1] is decomposition-SENSITIVE on the unified driver, so its differing
    # witnesses that the decompositions differed; a failure here means the chunk steering
    # stopped taking effect (or et[1] invariance was restored) and wants a look. Its value is
    # pinned by the regression harness, not here.
    @test auto.free_boundary !== nothing && finer.free_boundary !== nothing
    @test finer.free_boundary.et[1] != auto.free_boundary.et[1]

    @test size(finer.delta_prime.matrix) == size(auto.delta_prime.matrix)

    # Bit-identical, not approximate: a tolerance would hide exactly the reassociation drift
    # this test exists to catch. The element-wise `===` pairs with the whole-matrix `==`
    # because NaN === NaN — the `==` is what fails if the computation degrades to NaN.
    for j in 1:size(auto.delta_prime.matrix, 1)
        @test finer.delta_prime.matrix[j, j] === auto.delta_prime.matrix[j, j]
    end
    @test finer.delta_prime.matrix == auto.delta_prime.matrix
end
