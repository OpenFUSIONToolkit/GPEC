using TOML
using HDF5

# The scripting API: `PlasmaEquilibrium` / `solve(eq, alg)` / `perturbed_equilibrium` must run the
# same pipeline the TOML driver does. Every assertion is anchored on the coarse Solovev fixture
# deck (mpsi=16, mtheta_vacuum=64, delta_m=0), so an API run is compared against `main` on the same deck.
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

    # Copy the fixture deck into `dir` and apply `[Equilibrium]` / `[ForceFreeStates]` overrides and extra sections.
    function _stage_deck(dir; eq_overrides=Dict{String,Any}(), ffs_overrides=Dict{String,Any}(), extra_sections=Dict{String,Any}())
        for name in readdir(template)
            cp(joinpath(template, name), joinpath(dir, name))
        end
        toml_path = joinpath(dir, "gpec.toml")
        inputs = TOML.parsefile(toml_path)
        merge!(inputs["Equilibrium"], eq_overrides)
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

            # With the deck's own (auto) chunking the two runs are the same computation.
            same = solve(prob, Riccati())
            @test same.delta_prime.matrix ≈ reference.delta_prime.matrix rtol = 1e-12
            @test same.delta_prime.raw ≈ reference.delta_prime.raw rtol = 1e-12
            @test same.delta_prime.coil ≈ reference.delta_prime.coil rtol = 1e-12
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

            reference = GPEC.main([_stage_deck(dir; ffs_overrides=Dict{String,Any}("integrator" => "galerkin", "gal_nx" => 32))]).ffs
            @test reference.galerkin.msing == msing
            @test dp.matrix ≈ reference.delta_prime.matrix rtol = 1e-12
            @test dp.raw ≈ reference.delta_prime.raw rtol = 1e-12
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

            # Known drift: the deck injects the raw file modes, skipping the normalization conversion the API applies.
            @test_broken pe.forcing_b ≈ reference.forcing_b rtol = 1e-10
            @test_broken pe.resonant_area_weighted_field ≈ reference.resonant_area_weighted_field rtol = 1e-10

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
        @test_throws ErrorException solve(equil, Forward(); nn=1, dir_path=".", ffs_kwargs..., kinetic_factor=0.5)
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
            n = length(gal.surfaces)
            slow = solve(MatchProblem(gal; eta=1e-7, rho=1e-7, rotation=1.0), GGJModel())
            fast = solve(MatchProblem(gal; eta=fill(1e-7, n), rho=fill(1e-7, n), rotation=fill(10.0, n)), GGJModel())
            @test slow.bpen != fast.bpen
            # The result records which model it was matched with.
            @test slow.match.model === GGJModel()
            @test first_diff(gal, gal_snap) === nothing
        end
    end

    @testset "matched inner profiles: opt-in, and parity columns agree across inner models" begin
        mktempdir() do dir
            gal = solve(equil, Galerkin(; nx=32, rpec_flag=true, cut_solution=true); nn=1, dir_path=dir, ffs_kwargs...)
            prob = MatchProblem(gal; eta=1e-7, rho=1e-7, rotation=1.0)
            center = solve(prob, GGJModel()).match
            ray = solve(prob, GGJModel(; real_profiles=true)).match
            galm = solve(prob, GGJModel(; solver=:galerkin)).match
            # Δ and the penetrated field come from the rotated solve whether or not profiles are requested.
            @test ray.deltar == center.deltar && ray.bpen == center.bpen
            @test isempty(center.inner_psi) && isempty(center.inner_xi) && isempty(center.inner_b)
            @test !isempty(ray.inner_xi) && !isempty(galm.inner_xi)
            # At ψ_s the composite ξ is the even (Δ₊) column and b the odd (Δ₋) one, so a parity swap in the
            # profile assembly makes the two backends disagree here.
            for (i, sing) in enumerate(gal.surfaces)
                ir = findfirst(==(sing.psifac), ray.inner_psi[i])
                ig = findfirst(==(sing.psifac), galm.inner_psi[i])
                @test ir !== nothing && ig !== nothing
                @test isapprox(ray.inner_xi[i][ir, :], galm.inner_xi[i][ig, :]; rtol=1e-4)
                @test isapprox(ray.inner_b[i][ir, :], galm.inner_b[i][ig, :]; rtol=1e-4)
            end
        end
    end

    @testset "a resistive match keeps its coefficients and drops the ideal δW" begin
        mktempdir() do dir
            ric = solve(equil, Riccati(); nn=1, dir_path=dir, ffs_kwargs...)
            @test ric.free_boundary !== nothing && !isempty(ric.delta_prime.coil)
            matched = solve(MatchProblem(ric; eta=1e-7, rho=1e-7, rotation=1.0), GGJModel())
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
            # A resistive match needs every layer input.
            @test_throws ErrorException MatchProblem(gal; eta=1e-7)
            @test_throws ErrorException MatchProblem(gal; eta=fill(1e-7, length(gal.surfaces) + 1), rho=1e-7, rotation=1.0)
            # The tearing model is the solve argument, never a TearingProblem keyword.
            @test_throws ErrorException TearingProblem(gal; inner_model=:ggj_ray)
            # The deck's tearing stage propagates errors rather than finishing without a Tearing/ group.
            @test_throws ArgumentError GPEC.run_slayer_stage(gal, Dict{String,Any}("SLAYER" => Dict{String,Any}("enabled" => true, "inner_model" => "ggj_ray")), nothing)
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

    # Each workflow is run from a deck through `main` and from the same inputs through the API.
    # A known drift is `@test_broken`, so the change that removes it surfaces as an unexpected pass.
    @testset "deck and API parity" begin
        @testset "hdf5 file forcing" begin
            pe_kwargs = (compute_response=true, compute_singular_coupling=true, verbose=false, write_outputs_to_HDF5=false)
            mktempdir() do dir
                modes = GPEC.ForcingTerms.ForcingMode[]
                GPEC.ForcingTerms.load_forcing_data!(modes, joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example"), "forcing.dat", "ascii", false)
                h5open(f -> GPEC.ForcingTerms.save_forcing_to_h5(modes, f), joinpath(dir, "forcing.h5"), "w")
                _stage_deck(dir;
                    extra_sections=Dict{String,Any}(
                        "ForcingTerms" => Dict{String,Any}("forcing_data_file" => "forcing.h5", "forcing_data_format" => "hdf5"),
                        "PerturbedEquilibrium" => Dict{String,Any}(String(k) => v for (k, v) in pairs(pe_kwargs))))
                reference = GPEC.main([dir])
                pe = perturbed_equilibrium(reference.ffs, RMPField(joinpath(dir, "forcing.h5")); pe_kwargs...)

                @test pe.permeability ≈ reference.pe.permeability rtol = 1e-10
                # Known drift: the deck injects the raw file modes, skipping the normalization conversion the API applies.
                @test_broken pe.forcing_b ≈ reference.pe.forcing_b rtol = 1e-10
            end
        end

        @testset "analytic equilibrium on the two-pass auto grid" begin
            # A coarse psi_accuracy and no local-stability scan keep the 1024-interval pass-1 grid cheap.
            eq_overrides = Dict{String,Any}("grid_type" => "auto", "mpsi" => 0, "psi_accuracy" => 3e-2)
            ffs_overrides = Dict{String,Any}("local_stability_flag" => false, "vac_flag" => false)
            mktempdir() do dir
                reference = GPEC.main([_stage_deck(dir; eq_overrides=eq_overrides, ffs_overrides=ffs_overrides)]).ffs
                pass1 = GPEC.Equilibrium.setup_equilibrium(
                    GPEC.Equilibrium.EquilibriumConfig(merge(deck["Equilibrium"], eq_overrides), template),
                    GPEC.Equilibrium.SolovevConfig(deck["SOL_INPUT"])
                )
                @test reference.equil.rzphi_xs != pass1.rzphi_xs    # the deck re-formed on a refined grid

                kwargs = merge(ffs_kwargs, Dict(Symbol(k) => v for (k, v) in ffs_overrides))
                api = solve(pass1, Forward(); nn=1, wall=wall, dir_path=dir, kwargs...)
                # Known drift: the equilibrium does not store its analytic config, so `solve` cannot re-form it.
                @test_broken api.equil.rzphi_xs == reference.equil.rzphi_xs
            end
        end

        @testset "deck matching under gal_match_flag" begin
            layer = (eta=1e-7, rho=1e-7, rotation=1.0)
            # The galerkin inner model returns real-ψ profiles, so the composite inner profiles are compared too.
            mktempdir() do dir
                gal = solve(equil, Galerkin(; nx=32, rpec_flag=true, cut_solution=true); nn=1, wall=wall, dir_path=dir, ffs_kwargs...)
                api = solve(MatchProblem(gal; layer...), GGJModel(; solver=:galerkin))
                n = length(gal.surfaces)
                reference = GPEC.main([_stage_deck(dir;
                    ffs_overrides=Dict{String,Any}(
                        "integrator" => "galerkin", "gal_nx" => 32, "gal_rpec_flag" => true, "gal_match_flag" => true,
                        "gal_inner_solver" => "galerkin", "gal_eta" => fill(layer.eta, n), "gal_rho" => fill(layer.rho, n),
                        "gal_rotation" => fill(layer.rotation, n)))]).ffs

                @test reference.closure === api.closure === :matched
                @test api.bpen ≈ reference.bpen rtol = 1e-12
                @test api.match.xi ≈ reference.match.xi rtol = 1e-12
                @test length(api.match.inner_xi) == length(reference.match.inner_xi) == n
                @test all(isapprox.(api.match.inner_xi, reference.match.inner_xi; rtol=1e-12))

                uncut = solve(equil, Galerkin(; nx=32, rpec_flag=true); nn=1, wall=wall, dir_path=dir, ffs_kwargs...)
                matched = solve(MatchProblem(uncut; layer...), GGJModel(; solver=:galerkin))
                # Known drift: gal_match_flag implies the cut solution on the deck; the API keeps it only when asked.
                @test_broken length(matched.match.inner_xi) == length(reference.match.inner_xi)
            end
        end

        @testset "fixed kinetic source" begin
            mktempdir() do dir
                reference = GPEC.main([_stage_deck(dir; ffs_overrides=Dict{String,Any}("kinetic_factor" => 1e-9))]).ffs
                @test reference.free_boundary !== nothing

                kwargs = merge(ffs_kwargs, Dict(:kinetic_factor => 1e-9))
                # Known drift: `solve` rejects every kinetic run, although the fixed source needs no profiles.
                @test_broken solve(equil, Forward(); nn=1, wall=wall, dir_path=dir, kwargs...).free_boundary.et[1] ≈ reference.free_boundary.et[1] rtol = 1e-12
            end
        end
    end
end
