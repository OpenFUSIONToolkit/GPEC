using TOML

# The scripting API: `PlasmaEquilibrium` / `solve(eq, alg)` / `perturbed_equilibrium` must run the
# same pipeline the TOML driver does. Every assertion is anchored on the coarse Solovev fixture
# deck (mpsi=16, mthvac=64, delta_m=0), so an API run is compared against `main` on the same deck.
@testset "solve API" begin
    GPEC = GeneralizedPerturbedEquilibrium
    FFS = GPEC.ForceFreeStates
    template = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
    deck = TOML.parsefile(joinpath(template, "gpec.toml"))

    # The fixture equilibrium and wall, built exactly as `build_inputs_from_toml` builds them.
    equil = GPEC.Equilibrium.setup_equilibrium(
        GPEC.Equilibrium.EquilibriumConfig(deck["Equilibrium"], template),
        GPEC.Equilibrium.SolovevConfig(deck["SOL_INPUT"])
    )
    wall = GPEC.Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in deck["Wall"])...)

    # The deck's `[ForceFreeStates]` block as `solve` keywords: everything except the knobs the
    # integrator object and the `nn` keyword own.
    ffs_kwargs = Dict(Symbol(k) => v for (k, v) in deck["ForceFreeStates"]
                      if !(k in ("integrator", "nn_low", "nn_high")))

    # Copy the fixture deck into `dir` and apply `[ForceFreeStates]` overrides / extra sections.
    function _stage_deck(dir; ffs_overrides=Dict{String,Any}(), extra_sections=Dict{String,Any}())
        for name in readdir(template)
            cp(joinpath(template, name), joinpath(dir, name))
        end
        toml_path = joinpath(dir, "gpec.toml")
        inputs = TOML.parsefile(toml_path)
        merge!(inputs["ForceFreeStates"], ffs_overrides)
        merge!(inputs, extra_sections)
        open(io -> TOML.print(io, inputs), toml_path, "w")
        return dir
    end

    @testset "forward solve matches the TOML-driven run" begin
        mktempdir() do dir
            reference = GPEC.main([_stage_deck(dir)]).ffs
            api = solve(equil, Forward(); nn=1, wall=wall, dir_path=dir, ffs_kwargs...)

            @test api isa FFS.ForceFreeStatesResult
            @test api.integrator === :forward
            @test api.free_boundary !== nothing
            @test api.free_boundary.et[1] ≈ reference.free_boundary.et[1] rtol = 1e-12
            @test api.diagnostics.nzero == reference.diagnostics.nzero
            @test api.mlow == reference.mlow
            @test api.mhigh == reference.mhigh
            @test api.psilim ≈ reference.psilim rtol = 1e-12
            @test length(api.surfaces) == length(reference.surfaces)

            # The solution is the forward integrator's dense axis-basis profile set.
            @test api.solution isa FFS.SolutionProfiles
            @test api.solution.basis === :el_axis
            @test api.closure === :ideal
        end
    end

    @testset "riccati solve produces the unified delta_prime" begin
        mktempdir() do dir
            reference = GPEC.main([_stage_deck(dir; ffs_overrides=Dict{String,Any}("integrator" => "riccati"))]).ffs
            prob = EulerLagrangeProblem(equil; nn=1, wall=wall, dir_path=dir, ffs_kwargs...)
            api = solve(prob, Riccati(; nchunks=40))

            @test api.integrator === :riccati
            @test api.control.nchunks == 40
            @test api.solution === nothing
            @test api.delta_prime !== nothing
            msing = length(api.surfaces)
            @test size(api.delta_prime.matrix) == (msing, msing)
            @test api.delta_prime.matrix ≈ FFS.pest3_decompose(api.delta_prime.raw).Δ

            # Chunking is a decomposition of the same problem, so Δ′ tracks the TOML run.
            @test size(reference.delta_prime.matrix) == size(api.delta_prime.matrix)
            @test api.delta_prime.matrix ≈ reference.delta_prime.matrix rtol = 1e-6
        end
    end

    @testset "galerkin solve returns a gal result with a unified delta_prime" begin
        mktempdir() do dir
            api = solve(equil, Galerkin(; nx=32); nn=1, wall=wall, dir_path=dir, ffs_kwargs...)

            @test api.integrator === :galerkin
            @test api.control.gal_nx == 32
            @test api.galerkin !== nothing
            @test api.free_boundary === nothing        # galerkin computes no free-boundary energies
            dp = api.delta_prime
            @test dp !== nothing
            msing = api.galerkin.msing
            @test size(dp.matrix) == (msing, msing)
            @test dp.matrix ≈ FFS.pest3_decompose(dp.raw).Δ
            @test dp.A !== nothing                     # the parity blocks the galerkin solve persists
        end
    end

    @testset "perturbed_equilibrium round-trips a forward result" begin
        forcing = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example", "forcing.dat")
        pe_section = Dict{String,Any}(
            "compute_response" => true,
            "compute_singular_coupling" => true,
            "verbose" => false,
            "write_outputs_to_HDF5" => false)

        mktempdir() do dir
            _stage_deck(dir;
                extra_sections=Dict{String,Any}(
                    "ForcingTerms" => Dict{String,Any}(
                        "forcing_data_file" => "forcing.dat",
                        "forcing_data_format" => "ascii"),
                    "PerturbedEquilibrium" => pe_section))
            cp(forcing, joinpath(dir, "forcing.dat"))
            reference = GPEC.main([dir]).pe

            ffs = solve(equil, Forward(); nn=1, wall=wall, dir_path=dir, ffs_kwargs...)
            pe = perturbed_equilibrium(ffs, RMPField(forcing);
                (Symbol(k) => v for (k, v) in pe_section)...)

            # The response matrices depend on the equilibrium and the solve, not on the drive.
            @test !isempty(pe.permeability)
            @test pe.permeability ≈ reference.permeability rtol = 1e-10
            @test size(pe.C_delta_prime) == size(reference.C_delta_prime)
            @test !isempty(pe.resonant_area_weighted_field)

            # The amplitude-linear outputs are compared through the driver's own input path:
            # the driver hands the stage the modes it snapshotted before the solve, whereas a
            # fresh RMPField re-reads the file and re-runs the normalization conversion.
            snapshot = GPEC.ForcingTerms.ForcingMode[]
            GPEC.ForcingTerms.load_forcing_data!(snapshot, dir, "forcing.dat", "ascii", false)
            injected = perturbed_equilibrium(ffs, RMPField(forcing); forcing_modes=snapshot,
                (Symbol(k) => v for (k, v) in pe_section)...)
            @test injected.resonant_area_weighted_field ≈ reference.resonant_area_weighted_field rtol = 1e-10
            @test injected.forcing_b ≈ reference.forcing_b rtol = 1e-10

            # `scale` is a uniform multiplier on the materialized forcing, and the response is
            # linear in it.
            scaled = perturbed_equilibrium(ffs, RMPField(forcing; scale=2.0);
                (Symbol(k) => v for (k, v) in pe_section)...)
            @test scaled.forcing_b ≈ 2 .* pe.forcing_b rtol = 1e-10
            @test scaled.resonant_area_weighted_field ≈ 2 .* pe.resonant_area_weighted_field rtol = 1e-10

            # Lazy source algebra materializes to the combined field: 3A - A == 2A drives
            # the same perturbed equilibrium as scale=2 (exercises +, -, * and the merge).
            combo = perturbed_equilibrium(ffs, 3 * RMPField(forcing) - RMPField(forcing);
                (Symbol(k) => v for (k, v) in pe_section)...)
            @test combo.forcing_b ≈ scaled.forcing_b rtol = 1e-10
            @test combo.resonant_area_weighted_field ≈ scaled.resonant_area_weighted_field rtol = 1e-10
        end
    end

    @testset "RMPField algebra is lazy and flattens" begin
        a = RMPField("a.dat")
        b = RMPField("b.dat"; scale=0.5)
        c = RMPField("c.dat")
        s = a + b
        @test s isa GPEC.ForcingTerms.RMPFieldSum
        @test length(s.terms) == 2
        @test length((a + b + c).terms) == 3
        d = 2.0 * s
        @test d.terms[1].scale == 2.0 + 0.0im
        @test d.terms[2].scale == 1.0 + 0.0im
        @test (im * a).scale == im
        @test (a * 3).scale == 3.0 + 0.0im
        @test (a - b).terms[2].scale == -0.5 + 0.0im
        @test (-a).scale == -1.0 + 0.0im
    end

    @testset "RMPField infers its format and carries its scale" begin
        @test RMPField("forcing.dat").ctrl.forcing_data_format == "ascii"
        @test RMPField("forcing.h5").ctrl.forcing_data_format == "hdf5"
        @test RMPField("forcing.dat"; format="hdf5").ctrl.forcing_data_format == "hdf5"
        @test isabspath(RMPField("forcing.dat").ctrl.forcing_data_file)
        @test RMPField("forcing.dat"; scale=3.0).scale == 3.0
        coil_field = RMPField(Dict{String,Any}[Dict{String,Any}("name" => "iu")]; machine="d3d")
        @test coil_field.ctrl.forcing_data_format == "coil"
        @test coil_field.ctrl.machine == "d3d"
        @test length(coil_field.ctrl.coil_sets_raw) == 1
    end

    @testset "integrator objects translate onto the control keys" begin
        kwargs = Dict{Symbol,Any}()
        FFS._apply_alg!(kwargs, Galerkin(; nx=64, rpec_flag=true))
        @test kwargs[:integrator] == "galerkin"
        @test kwargs[:gal_nx] == 64
        @test kwargs[:gal_rpec_flag]

        # A match implies the coil-response columns and fills the inner-layer knobs.
        FFS._apply_match!(kwargs, ResistiveMatch(; eta=[1e-6], inner_solver="ray"), Galerkin())
        @test kwargs[:gal_match_flag]
        @test kwargs[:gal_rpec_flag]
        @test kwargs[:gal_eta] == [1e-6]
        @test kwargs[:gal_inner_solver] == "ray"
        @test !kwargs[:gal_ideal_flag]

        # Every key the objects own is a `ForceFreeStatesControl` field.
        @test all(in(fieldnames(FFS.ForceFreeStatesControl)), keys(kwargs))
    end

    @testset "rejected keyword combinations" begin
        @test_throws ErrorException solve(equil, Forward(); nn=1, dir_path=".", ffs_kwargs..., kinetic_factor=0.5)
        @test_throws ErrorException solve(equil, Riccati(); nn=1, dir_path=".", ffs_kwargs..., match=ResistiveMatch())
        @test_throws ErrorException solve(equil, Forward(); nn=1, dir_path=".", ffs_kwargs..., match=ResistiveMatch())
        @test_throws ErrorException solve(equil, Forward(); nn=1, dir_path=".", ffs_kwargs..., integrator="riccati")
        @test_throws ErrorException solve(equil, Riccati(); nn=1, dir_path=".", ffs_kwargs..., nchunks=8)
        @test_throws ErrorException solve(equil, Forward(); nn=1, dir_path=".", ffs_kwargs..., nn_low=2)
    end
end
