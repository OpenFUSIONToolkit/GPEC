using TOML
using HDF5
include("h5_metadata_check.jl")

# ForceFreeStatesResult: what each formalism publishes, and how consumers gate on it.
# The Solovev fixture deck (mpsi=16, delta_m=0) is coarse but has two rational surfaces,
# which is all these interface assertions need.
@testset "ForceFreeStatesResult" begin
    FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
    template = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")

    # Copy a deck into `dir` and apply `[ForceFreeStates]` overrides / extra sections. An `integrator`
    # override carries the old integrator's ODE numerics over to the new one's section.
    function _stage_deck(dir, source; ffs_overrides=Dict{String,Any}(), extra_sections=Dict{String,Any}())
        for name in readdir(source)
            cp(joinpath(source, name), joinpath(dir, name))
        end
        toml_path = joinpath(dir, "gpec.toml")
        inputs = TOML.parsefile(toml_path)
        old = get(inputs["ForceFreeStates"], "integrator", "Riccati")
        merge!(inputs["ForceFreeStates"], ffs_overrides)
        new = get(inputs["ForceFreeStates"], "integrator", "Riccati")
        old == new || haskey(inputs, old) && (new == "Galerkin" ? delete!(inputs, old) : (inputs[new] = pop!(inputs, old)))
        for (name, section) in extra_sections
            haskey(inputs, name) ? merge!(inputs[name], section) : (inputs[name] = section)
        end
        if haskey(inputs, "TJAnalytic")
            # Interface checks need two rational surfaces, not a converged production grid.
            merge!(inputs["Equilibrium"], Dict("mpsi" => 32, "mtheta" => 64))
            merge!(inputs["TJAnalytic"], Dict("ma" => 32, "mtheta" => 32))
            merge!(inputs["ForceFreeStates"], Dict("delta_mlow" => 2, "delta_mhigh" => 2, "mtheta_vacuum" => 64))
            haskey(inputs, "Galerkin") && (inputs["Galerkin"]["nx"] = 32)
        end
        open(io -> TOML.print(io, inputs), toml_path, "w")
        return dir
    end

    forward_result = Ref{FFS.ForceFreeStatesResult}()
    riccati_result = Ref{FFS.ForceFreeStatesResult}()

    @testset "forward run publishes a dense axis-basis solution" begin
        mktempdir() do dir
            _stage_deck(dir, template)
            ret = GeneralizedPerturbedEquilibrium.main([dir])

            # One field per stage, nothing for the sections the deck does not hold.
            @test keys(ret) == (:equil, :local_stability, :ffs, :match, :resp, :pe, :kinetic, :error_fields, :tearing)
            @test ret.equil === ret.ffs.equil && all(isnothing, (ret.match, ret.resp, ret.pe, ret.kinetic, ret.error_fields, ret.tearing))
            ffs = ret.ffs
            forward_result[] = ffs
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
            @test ffs.integrator === :forward
        end
    end

    @testset "riccati run publishes Δ′ and diagnostics but no ξ solution" begin
        mktempdir() do dir
            _stage_deck(dir, template; ffs_overrides=Dict{String,Any}("integrator" => "Riccati"))
            ret = GeneralizedPerturbedEquilibrium.main([dir])

            ffs = ret.ffs
            riccati_result[] = ffs
            @test ffs.integrator === :riccati
            # Chunk-endpoint Riccati states are not a ξ solution; only the raw state is carried.
            @test ffs.solution === nothing
            @test ffs.diagnostics !== nothing
            @test !ffs.diagnostics.u_store_el_basis
            @test ffs.closure === :ideal
            @test iszero(ffs.bpen)

            @test ffs.delta_prime !== nothing
            dp = ffs.delta_prime
            @test size(dp.raw_freeboundary) == (2 * length(ffs.surfaces), 2 * length(ffs.surfaces))
            @test size(dp.raw_fixedboundary) == size(dp.raw_freeboundary) && dp.raw_fixedboundary != dp.raw_freeboundary
            # The PEST-3 blocks of both boundaries are stored, whichever formalism ran.
            for (blocks, raw) in ((dp.freeboundary, dp.raw_freeboundary), (dp.fixedboundary, dp.raw_fixedboundary))
                ref = FFS.pest3_decompose(raw)
                @test blocks.A ≈ ref.A && blocks.B ≈ ref.B && blocks.Gamma ≈ ref.Gamma && blocks.Delta ≈ ref.Delta
            end
            @test size(dp.ef) == (2 * length(ffs.surfaces), ffs.numpert_total)
            @test size(dp.edge_response) == (ffs.numpert_total, 2 * length(ffs.surfaces)) && !iszero(dp.edge_response)
            @test ffs.free_boundary !== nothing
        end
    end

    @testset "require / require_solution gates" begin
        let ffs = riccati_result[]
            # `require` passes silently on a populated field and warns once on an absent one.
            @test @test_logs FFS.require(ffs, :free_boundary, "a calculation")
            @test !(@test_logs (:warn,) FFS.require(ffs, :galerkin, "a calculation"))

            # `require_solution` is the ξ-specific presence gate.
            @test !(@test_logs (:warn,) FFS.require_solution(ffs, "a calculation"))
        end
    end

    @testset "PerturbedEquilibrium warn-skips on a riccati result" begin
        mktempdir() do dir
            _stage_deck(dir, joinpath(@__DIR__, "test_data", "regression_tj_ideal_example");
                ffs_overrides=Dict{String,Any}("integrator" => "Riccati"),
                extra_sections=Dict{String,Any}(
                    "ForcingTerms" => Dict{String,Any}(
                        "forcing_data_file" => "forcing.dat",
                        "forcing_data_format" => "ascii"),
                    "PerturbedEquilibrium" => Dict{String,Any}(
                        "compute_singular_coupling" => true)))
            cp(joinpath(@__DIR__, "..", "examples", "TJ_ideal_example", "forcing.dat"),
                joinpath(dir, "forcing.dat"))

            local ret
            @test_logs (:warn,) match_mode = :any (ret = GeneralizedPerturbedEquilibrium.main([dir]))
            # The stage runs to completion; both sub-calculations are simply not populated.
            @test ret.pe !== nothing
            @test isempty(ret.pe.response.permeability)
            @test isempty(ret.pe.response.C_delta_prime)
        end
    end

    # Standalone Galerkin publishes its ideal-closed EF columns as the ξ solution; with the RPEC
    # inner-layer match the matched outer solution replaces them, and the closure records which.
    @testset "Galerkin publishes a gal-native solution" begin
        for (deck, ideal) in (("LAR_ideal_match_test", true), ("LAR_resistive_match_test", false))
            @testset "$deck" begin
                mktempdir() do dir
                    _stage_deck(dir, joinpath(@__DIR__, "..", "examples", deck))
                    ffs = GeneralizedPerturbedEquilibrium.main([dir]).ffs

                    @test ffs.integrator === :galerkin
                    @test ffs.galerkin !== nothing
                    @test (ffs.match === nothing) == ideal

                    sol = ffs.solution
                    @test sol isa FFS.SolutionProfiles
                    @test sol.basis === :gal_native
                    @test sol.step == length(sol.psi_store) == size(sol.u_store, 4)
                    @test !iszero(sol.du_store)      # analytic galerkin Ξ′
                    @test !iszero(sol.xi_s_store)

                    # Galerkin runs no Euler-Lagrange sweep; the edge response of its closed columns gives the
                    # free-boundary energies (vac_flag = true in both decks).
                    @test ffs.diagnostics === nothing
                    @test size(ffs.wp) == (ffs.numpert_total, ffs.numpert_total)
                    @test ffs.free_boundary !== nothing && ffs.free_boundary.wp == ffs.wp
                    @test ffs.free_boundary.wt0 ≈ ffs.wp + ffs.free_boundary.wv rtol = 1e-12

                    # Galerkin publishes its Δ′ through the same unified field the Riccati BVP
                    # uses, in the same PEST-3 convention and the same shared layout.
                    dp = ffs.delta_prime
                    @test dp !== nothing
                    msing = ffs.galerkin.msing
                    for (blocks, raw) in ((dp.freeboundary, dp.raw_freeboundary), (dp.fixedboundary, dp.raw_fixedboundary))
                        @test size(raw) == (2msing, 2msing)
                        ref = FFS.pest3_decompose(raw)
                        @test blocks.A ≈ ref.A && blocks.B ≈ ref.B && blocks.Gamma ≈ ref.Gamma && blocks.Delta ≈ ref.Delta
                    end

                    # The EF block is normalized to the Riccati orientation at pack time.
                    @test size(dp.ef) == (2msing, ffs.numpert_total)
                    @test !iszero(dp.ef)
                    @test dp.edge_response == ffs.galerkin.edge_response[:, 1:(2msing)]

                    # Without a match the EF columns are the ideal closure, with a Hermitian δW.
                    if ideal
                        @test ffs.closure === :ideal
                        @test iszero(ffs.bpen)
                        @test norm(ffs.wp - ffs.wp') <= 1e-8 * norm(ffs.wp)
                    else
                        @test ffs.closure === :matched
                        @test ffs.bpen == ffs.match.bpen
                        @test !iszero(ffs.bpen)
                    end

                    # The closed profiles land in the shared Solutions layout: same names and
                    # (mode, solution, psi) axis order as ForwardIntegration, on the gal grid.
                    gal = "ForceFreeStates/Solutions/GalerkinIntegration"
                    HDF5.h5open(GeneralizedPerturbedEquilibrium.run_settings(TOML.parsefile(joinpath(dir, "gpec.toml")), dir).output_file, "r") do f
                        @test ffs.galerkin.solver in ("cholesky", "LU")
                        @test read(f["$gal/solver"]) == ffs.galerkin.solver
                        @test haskey(attrs(f["$gal/solver"]), "long_name")
                        @test read(f["$gal/psi"]) == sol.psi_store
                        @test read(f["$gal/xi_psi"]) == sol.u_store[:, :, 1, :]
                        @test read(f["$gal/dxi_psidpsi"]) == sol.du_store
                        @test read(f["$gal/xi_s"]) == sol.xi_s_store
                        @test read(f["ForceFreeStates/FreeBoundaryStability/W_plasma"]) == ffs.wp
                        # Matching diagnostics sit with the other per-surface results, and the
                        # raw outer basis is debug-gated off by default.
                        @test haskey(f, "SingularSurfaces/Match/cout") != ideal
                        @test !haskey(f, "$gal/Match")
                        if !ideal
                            # Match/ is self-describing: rows follow SingularSurfaces/rational_psi, and it names the model that ran.
                            mg = f["SingularSurfaces/Match"]
                            @test size(read(mg["bpen"]), 1) == length(read(f["SingularSurfaces/rational_psi"]))
                            @test read(mg["reconnected_flux"]) == ffs.match.reconnected_flux
                            @test attrs(mg)["inner_model"] == GeneralizedPerturbedEquilibrium.InnerLayer.model_name(ffs.match.model)
                            @test isempty(_collect_metadata_violations(mg))
                        end
                        @test !haskey(f, "$gal/Basis")
                        @test !haskey(f, "$gal/Solution")
                    end
                end
            end
        end
    end
    @testset "Galerkin basis_output dumps the raw outer basis" begin
        mktempdir() do dir
            _stage_deck(dir, joinpath(@__DIR__, "..", "examples", "LAR_ideal_match_test");
                extra_sections=Dict{String,Any}("Galerkin" => Dict{String,Any}("basis_output" => true)))
            ffs = GeneralizedPerturbedEquilibrium.main([dir]).ffs
            gal = "ForceFreeStates/Solutions/GalerkinIntegration"
            HDF5.h5open(GeneralizedPerturbedEquilibrium.run_settings(TOML.parsefile(joinpath(dir, "gpec.toml")), dir).output_file, "r") do f
                @test haskey(f, "$gal/Basis/xi_psi")
                basis = read(f["$gal/Basis/xi_psi"])
                # (mode, solution, psi) on the FULL gal grid, on-surface nodes included; the
                # columns are the 2·msing resonant basis solutions plus the coil-drive columns.
                @test size(basis, 2) == 2 * ffs.galerkin.msing + ffs.numpert_total
                @test size(basis, 3) == length(read(f["$gal/Basis/psi"]))
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
        let ffs = forward_result[]
            @test ffs.free_boundary !== nothing
            @test ffs.wp === ffs.free_boundary.wp
        end
    end
end
