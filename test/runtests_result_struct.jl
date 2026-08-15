using TOML

# ForceFreeStatesResult: what each formalism publishes, and how consumers gate on it.
# The Solovev fixture deck (mpsi=16, delta_m=0) is coarse but has two rational surfaces,
# which is all these interface assertions need.
@testset "ForceFreeStatesResult" begin
    FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
    template = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")

    # Copy a deck into `dir` and apply `[ForceFreeStates]` overrides / extra sections.
    function _stage_deck(dir, source; ffs_overrides=Dict{String,Any}(), extra_sections=Dict{String,Any}())
        for name in readdir(source)
            cp(joinpath(source, name), joinpath(dir, name))
        end
        toml_path = joinpath(dir, "gpec.toml")
        inputs = TOML.parsefile(toml_path)
        merge!(inputs["ForceFreeStates"], ffs_overrides)
        merge!(inputs, extra_sections)
        open(io -> TOML.print(io, inputs), toml_path, "w")
        return dir
    end

    @testset "forward run publishes a dense axis-basis solution" begin
        mktempdir() do dir
            _stage_deck(dir, template)
            ret = GeneralizedPerturbedEquilibrium.main([dir])

            @test keys(ret) == (:ffs, :pe, :slayer)
            ffs = ret.ffs
            @test ffs isa FFS.ForceFreeStatesResult
            @test ffs.integrator === :forward

            # The solve's ξ solution, with its derivative stores populated by construction.
            sol = ffs.solution
            @test sol isa FFS.SolutionProfiles
            @test sol.basis === :el_axis
            @test sol.step == size(sol.u_store, 4)
            @test size(sol.du_store) == (ffs.numpert_total, ffs.numpert_total, sol.step)
            @test size(sol.xi_s_store) == (ffs.numpert_total, ffs.numpert_total, sol.step)
            @test !iszero(sol.du_store)
            @test !iszero(sol.xi_s_store)
            @test length(sol.psi_store) == sol.step
            @test length(sol.q_store) == sol.step

            # The raw ODE state rides along for the writer, distinct from the solution.
            @test ffs.diagnostics !== nothing
            @test ffs.diagnostics.crit_store !== nothing

            # Ideal closure: no inner layer was matched in.
            @test ffs.closure === :ideal
            @test size(ffs.bpen) == (length(ffs.surfaces), ffs.numpert_total)
            @test iszero(ffs.bpen)

            # Forward integration produces no STRIDE propagators, hence no Δ′ matrix.
            @test ffs.delta_prime === nothing
            @test ffs.galerkin === nothing
            @test ffs.free_boundary !== nothing            # vac_flag = true in the fixture

            # Mode space and domain are copied verbatim out of the solve-time scratch.
            @test ffs.mpert == ffs.mhigh - ffs.mlow + 1
            @test ffs.npert == ffs.nhigh - ffs.nlow + 1
            @test ffs.numpert_total == ffs.mpert * ffs.npert
            @test 0 < ffs.psilim <= 1
            @test !isempty(ffs.surfaces)
            @test ffs.control.integrator == "forward"
            @test ffs.dir_path == dir
        end
    end

    @testset "riccati run publishes Δ′ and diagnostics but no ξ solution" begin
        mktempdir() do dir
            _stage_deck(dir, template; ffs_overrides=Dict{String,Any}("integrator" => "riccati"))
            ret = GeneralizedPerturbedEquilibrium.main([dir])

            ffs = ret.ffs
            @test ffs.integrator === :riccati
            # Chunk-endpoint Riccati states are not a ξ solution; only the raw state is carried.
            @test ffs.solution === nothing
            @test ffs.diagnostics !== nothing
            @test !ffs.diagnostics.u_store_el_basis
            @test ffs.closure === :ideal
            @test iszero(ffs.bpen)

            @test ffs.delta_prime !== nothing
            @test size(ffs.delta_prime.matrix) == (length(ffs.surfaces), length(ffs.surfaces))
            @test size(ffs.delta_prime.raw) == (2 * length(ffs.surfaces), 2 * length(ffs.surfaces))
            @test ffs.free_boundary !== nothing
        end
    end

    @testset "require / require_solution gates" begin
        mktempdir() do dir
            _stage_deck(dir, template; ffs_overrides=Dict{String,Any}("integrator" => "riccati"))
            ffs = GeneralizedPerturbedEquilibrium.main([dir]).ffs

            # `require` passes silently on a populated field and warns once on an absent one.
            @test @test_logs FFS.require(ffs, :free_boundary, "a calculation")
            @test !(@test_logs (:warn,) FFS.require(ffs, :galerkin, "a calculation"))

            # `require_solution` is the ξ-specific presence gate.
            @test !(@test_logs (:warn,) FFS.require_solution(ffs, "a calculation"))
        end
    end

    @testset "PerturbedEquilibrium warn-skips on a riccati result" begin
        mktempdir() do dir
            _stage_deck(dir, template;
                ffs_overrides=Dict{String,Any}("integrator" => "riccati"),
                extra_sections=Dict{String,Any}(
                    "ForcingTerms" => Dict{String,Any}(
                        "forcing_data_file" => "forcing.dat",
                        "forcing_data_format" => "ascii"),
                    "PerturbedEquilibrium" => Dict{String,Any}(
                        "compute_response" => true,
                        "compute_singular_coupling" => true,
                        "verbose" => false,
                        "write_outputs_to_HDF5" => false)))
            cp(joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example", "forcing.dat"),
                joinpath(dir, "forcing.dat"))

            local ret
            @test_logs (:warn,) match_mode = :any (ret = GeneralizedPerturbedEquilibrium.main([dir]))
            # The stage runs to completion; both sub-calculations are simply not populated.
            @test ret.pe !== nothing
            @test isempty(ret.pe.permeability)
            @test isempty(ret.pe.C_delta_prime)
        end
    end

    # Standalone Galerkin with the RPEC inner-layer match: the matched outer solution becomes
    # the result's ξ solution, and the closure records whether an inner layer was matched in.
    @testset "matched Galerkin publishes a gal-native solution" begin
        for (deck, ideal) in (("LAR_ideal_match_test", true), ("LAR_resistive_match_test", false))
            @testset "$deck" begin
                mktempdir() do dir
                    _stage_deck(dir, joinpath(@__DIR__, "..", "examples", deck))
                    ffs = GeneralizedPerturbedEquilibrium.main([dir]).ffs

                    @test ffs.integrator === :galerkin
                    @test ffs.galerkin !== nothing
                    @test ffs.galerkin.match !== nothing

                    sol = ffs.solution
                    @test sol isa FFS.SolutionProfiles
                    @test sol.basis === :gal_native
                    @test sol.step == length(sol.psi_store) == size(sol.u_store, 4)
                    @test !iszero(sol.du_store)      # analytic galerkin Ξ′
                    @test !iszero(sol.xi_s_store)

                    # Galerkin runs no Euler-Lagrange sweep and computes no free-boundary energies.
                    @test ffs.diagnostics === nothing
                    @test ffs.free_boundary === nothing
                    @test ffs.wp === nothing
                    @test ffs.delta_prime === nothing

                    # The ideal-flag match skips the inner layer, so its basis is ideal-closed.
                    if ideal
                        @test ffs.closure === :ideal
                        @test iszero(ffs.bpen)
                    else
                        @test ffs.closure === :matched
                        @test ffs.bpen == ffs.galerkin.match.bpen
                        @test !iszero(ffs.bpen)
                    end
                end
            end
        end
    end
    @testset "fixed-boundary run still publishes the plasma energy matrix" begin
        mktempdir() do dir
            _stage_deck(dir, template; ffs_overrides=Dict{String,Any}("vac_flag" => false))
            ffs = GeneralizedPerturbedEquilibrium.main([dir]).ffs

            # No vacuum stage, so no free-boundary product — but W_p needs only the edge state.
            @test ffs.free_boundary === nothing
            @test ffs.wp !== nothing
            @test size(ffs.wp) == (ffs.numpert_total, ffs.numpert_total)
            @test all(isfinite, ffs.wp)
        end
    end

    @testset "free-boundary run aliases free_run's W_p" begin
        mktempdir() do dir
            _stage_deck(dir, template)
            ffs = GeneralizedPerturbedEquilibrium.main([dir]).ffs
            @test ffs.free_boundary !== nothing
            @test ffs.wp === ffs.free_boundary.wp
        end
    end
end
