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

        # Every key the objects own is a `ForceFreeStatesControl` field.
        @test all(in(fieldnames(FFS.ForceFreeStatesControl)), keys(kwargs))
    end

    @testset "rejected keyword combinations" begin
        # A calculated-source kinetic solve gates on profiles attached to the equilibrium.
        @test_throws ErrorException solve(equil, Forward(); nn=1, dir_path=".", ffs_kwargs...,
            kinetic_factor=0.5, kinetic_source="calculated")
        @test_throws ErrorException solve(equil, Forward(); nn=1, dir_path=".", ffs_kwargs..., integrator="riccati")
        @test_throws ErrorException solve(equil, Riccati(); nn=1, dir_path=".", ffs_kwargs..., nchunks=8)
        @test_throws ErrorException solve(equil, Forward(); nn=1, dir_path=".", ffs_kwargs..., nn_low=2)
    end

    # Path of the first field where `a` and `b` differ (recursing into structs, arrays and
    # dicts), or `nothing` when they are equal everywhere. Underscore fields are private
    # lazily-built caches (e.g. a spline's transpose) and are skipped.
    function first_diff(a, b, path="")
        typeof(a) === typeof(b) || return "$path (type)"
        a isa Union{Number,AbstractString,Symbol,Nothing,Function,Type} && return isequal(a, b) ? nothing : path
        a isa AbstractArray{<:Number} && return isequal(a, b) ? nothing : path
        if a isa AbstractArray
            size(a) == size(b) || return "$path (size)"
            for i in eachindex(a)
                d = first_diff(a[i], b[i], "$path[$i]")
                d === nothing || return d
            end
            return nothing
        end
        if a isa AbstractDict
            keys(a) == keys(b) || return "$path (keys)"
            for k in keys(a)
                d = first_diff(a[k], b[k], "$path[$k]")
                d === nothing || return d
            end
            return nothing
        end
        for f in fieldnames(typeof(a))
            startswith(string(f), "_") && continue
            isdefined(a, f) == isdefined(b, f) || return "$path.$f (definedness)"
            isdefined(a, f) || continue
            d = first_diff(getfield(a, f), getfield(b, f), "$path.$f")
            d === nothing || return d
        end
        return nothing
    end

    @testset "solves leave their inputs untouched" begin
        snap = deepcopy(equil)
        mktempdir() do dir
            gal = solve(equil, Galerkin(; nx=32, rpec_flag=true, cut_solution=true); nn=1, dir_path=dir, ffs_kwargs...)
            @test first_diff(equil, snap) === nothing
            # Matching reuses the outer solve: repeated match solves must not alter it.
            gal_snap = deepcopy(gal)
            n = length(MatchProblem(gal; ideal=true).surfaces)
            layer = (eta=fill(1e-7, n), rho=fill(1e-7, n))
            slow = solve(MatchProblem(gal; layer..., rotation=fill(1.0, n)), GGJModel())
            fast = solve(MatchProblem(gal; layer..., rotation=fill(10.0, n)), GGJModel())
            @test slow.bpen != fast.bpen
            @test first_diff(gal, gal_snap) === nothing
        end
    end

    @testset "a resistive match keeps its coefficients and drops the ideal δW" begin
        mktempdir() do dir
            ric = solve(equil, Riccati(); nn=1, dir_path=dir, ffs_kwargs...)
            @test ric.free_boundary !== nothing && !isempty(ric.delta_prime.coil)
            n = length(MatchProblem(ric; ideal=true).surfaces)
            matched = solve(MatchProblem(ric; eta=fill(1e-7, n), rho=fill(1e-7, n), rotation=fill(1.0, n)), GGJModel())
            @test matched.closure === :matched
            @test matched.match !== nothing && matched.bpen == matched.match.bpen
            @test matched.wp === nothing && matched.free_boundary === nothing
            @test matched.solution === nothing    # no Galerkin basis to build a matched ξ from
            # A matched result is a new solution, not an outer solve to match again.
            @test_throws ErrorException MatchProblem(matched; ideal=true)
        end
    end

    @testset "MatchProblem gates on its inputs" begin
        mktempdir() do dir
            # A Forward result carries no Δ′ payload, so the problem is unconstructible.
            fwd = solve(equil, Forward(); nn=1, dir_path=dir, ffs_kwargs...)
            @test_throws ErrorException MatchProblem(fwd; ideal=true)
            # A slab model can never close a matched solution.
            gal = solve(equil, Galerkin(; nx=32, rpec_flag=true); nn=1, dir_path=dir, ffs_kwargs...)
            prob = MatchProblem(gal; ideal=true)
            @test_throws ErrorException solve(prob, SLAYERModel())
            # The tearing model is the solve argument, never a TearingProblem keyword.
            @test_throws ErrorException TearingProblem(gal; inner_model=:ggj_ray)
            # The ideal reference match keeps the ideal closure and replaces the solution with
            # the bare coil columns in the identity-at-edge basis.
            matched = solve(prob, GGJModel())
            @test matched.closure === :ideal
            @test matched.match !== nothing
            # The ideal reference keeps the ideal δW (only a :matched closure drops it).
            @test matched.wp === gal.wp && matched.free_boundary === gal.free_boundary
            @test matched.solution !== nothing && matched.solution.basis === :gal_native
        end
    end

    @testset "kinetic profiles live on the equilibrium" begin
        kin_file = joinpath(@__DIR__, "..", "examples", "Solovev_kinetic_NTV_example", "kinetic.dat")
        eq = attach_kinetic_profiles!(deepcopy(equil), kin_file; zi=1)
        @test eq.kinetic isa GPEC.Equilibrium.KineticProfileSplines
        @test eq.kinetic.ni_spline(0.5) > 0
        # Kinetic solves stay TOML-driven even with profiles attached.
        @test_throws ErrorException solve(eq, Forward(); nn=1, dir_path=".", ffs_kwargs..., kinetic_factor=0.5)
    end
end
