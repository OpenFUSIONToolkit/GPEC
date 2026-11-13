using TOML
using HDF5
include("display_check.jl")

# The scripting API: `PlasmaEquilibrium` / `solve(eq, alg)` / `perturbed_equilibrium` must run the
# same pipeline the TOML driver does. Every assertion is anchored on the coarse TJ fixture
# deck (mpsi=16, mtheta_vacuum=64, delta_m=0), so an API run is compared against `main` on the same deck.
@testset "solve API" begin
    GPEC = GeneralizedPerturbedEquilibrium
    FFS = GPEC.ForceFreeStates
    template = joinpath(@__DIR__, "test_data", "regression_tj_ideal_example")
    deck = TOML.parsefile(joinpath(template, "gpec.toml"))

    # The fixture equilibrium and wall, built exactly as `build_inputs_from_toml` builds them.
    equil = GPEC.Equilibrium.PlasmaEquilibrium(
        GPEC.equilibrium_config(deck, template),
        GPEC.analytic_config(deck)
    )
    wall = GPEC.Vacuum.parse_wall(get(deck, "Wall", nothing))

    # The deck's `[ForceFreeStates]` block as `solve` keywords, less `nn` and the wall; its `[Forward]` numerics
    # serve both ODE integrators.
    ffs_kwargs = filter(p -> !(p.first in (:nn, :wall)), GPEC.force_balance_keywords(deck, template))
    ode_kwargs = GPEC.section_keywords(deck["Forward"], "Forward", GPEC._struct_fields(Forward))
    forward = Forward(; ode_kwargs...)
    riccati = Riccati(; ode_kwargs...)

    # Copy the fixture deck into `dir` and apply `[Equilibrium]` / `[ForceFreeStates]` overrides and extra sections.
    # An `integrator` override moves the `[Forward]` numerics to `[Riccati]`, or drops them for Galerkin.
    function _stage_deck(dir; eq_overrides=Dict{String,Any}(), ffs_overrides=Dict{String,Any}(), extra_sections=Dict{String,Any}())
        for name in readdir(template)
            cp(joinpath(template, name), joinpath(dir, name))
        end
        toml_path = joinpath(dir, "gpec.toml")
        inputs = TOML.parsefile(toml_path)
        merge!(inputs["Equilibrium"], eq_overrides)
        merge!(inputs["ForceFreeStates"], ffs_overrides)
        ode = pop!(inputs, "Forward")
        inputs["ForceFreeStates"]["integrator"] in ("Forward", "Riccati") && (inputs[inputs["ForceFreeStates"]["integrator"]] = ode)
        foreach(section -> delete!(inputs, section), ("LocalStability", "Ballooning"))
        merge!(inputs, extra_sections)
        open(io -> TOML.print(io, inputs), toml_path, "w")
        return dir
    end

    @testset "forward solve matches the TOML-driven run" begin
        mktempdir() do dir
            reference = GPEC.main([_stage_deck(dir)]).ffs
            api = solve(equil, forward; nn=1, wall=wall, ffs_kwargs...)

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
            reference = GPEC.main([_stage_deck(dir; ffs_overrides=Dict{String,Any}("integrator" => "Riccati"))]).ffs
            prob = ForceBalanceProblem(equil; nn=1, wall=wall, ffs_kwargs...)
            api = solve(prob, Riccati(; ode_kwargs..., nchunks=40))

            @test api.integrator === :riccati
            @test api.control.algorithm.nchunks == 40
            @test api.solution === nothing
            @test api.delta_prime !== nothing
            msing = length(api.surfaces)
            @test size(api.delta_prime.matrix) == (msing, msing)
            @test api.delta_prime.matrix ≈ FFS.pest3_decompose(api.delta_prime.raw).Δ

            # Chunking is a decomposition of the same problem, so Δ′ tracks the TOML run.
            @test size(reference.delta_prime.matrix) == size(api.delta_prime.matrix)
            @test api.delta_prime.matrix ≈ reference.delta_prime.matrix rtol = 1e-6

            # With the deck's own (auto) chunking the two runs are the same computation.
            same = solve(prob, riccati)
            @test same.delta_prime.matrix ≈ reference.delta_prime.matrix rtol = 1e-12
            @test same.delta_prime.raw ≈ reference.delta_prime.raw rtol = 1e-12
            @test same.delta_prime.coil ≈ reference.delta_prime.coil rtol = 1e-12
        end
    end

    @testset "test kinetic matrices: kinetic_factor scales the source" begin
        # Large enough that the test matrices visibly shift the free-boundary energies.
        kf = 1e-3
        scaled = solve(equil, forward; nn=1, wall=wall, ffs_kwargs..., kinetic=TestKineticMatrices(; scale=kf))
        factored = solve(equil, forward; nn=1, wall=wall, ffs_kwargs..., kinetic=TestKineticMatrices(; scale=1.0), kinetic_factor=kf)
        ideal = solve(equil, forward; nn=1, wall=wall, ffs_kwargs...)

        @test scaled.mats.kinetic !== nothing && ideal.mats.kinetic === nothing
        @test scaled.closure === :regular && ideal.closure === :ideal && iszero(scaled.bpen)
        @test !isapprox(scaled.free_boundary.et[1], ideal.free_boundary.et[1]; rtol=1e-6)
        @test factored.free_boundary.et[1] ≈ scaled.free_boundary.et[1] rtol = 1e-10
        @test factored.kinetic.kmsing == scaled.kinetic.kmsing
        @test [s.psifac for s in factored.kinetic.kinsing] ≈ [s.psifac for s in scaled.kinetic.kinsing]
        # Matrices evaluated once rescale in a later solve exactly as the source does.
        km = kinetic_matrices(ForceBalanceProblem(equil; nn=1, wall=wall, ffs_kwargs..., kinetic=TestKineticMatrices(; scale=1.0)), forward)
        @test solve(equil, forward; nn=1, wall=wall, ffs_kwargs..., kinetic=km, kinetic_factor=kf).free_boundary.et == factored.free_boundary.et
        @test_throws "no kinetic source" kinetic_matrices(ForceBalanceProblem(equil; nn=1, ffs_kwargs...), forward)

        # Too large a scale for this coarse deck: the integration aborts near ψ = 0.4 and says so.
        @test_throws "Euler-Lagrange integration aborted" solve(equil, forward; nn=1, wall=wall, ffs_kwargs..., kinetic=TestKineticMatrices(; scale=0.1))
    end

    @testset "calculated kinetic solve matches the TOML-driven run" begin
        # The equilibrium carries the deck's kinetic profiles, so the API solves the same kinetic problem.
        kin_template = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_calculated")
        kdeck = TOML.parsefile(joinpath(kin_template, "gpec.toml"))
        kequil = GPEC.Equilibrium.PlasmaEquilibrium(
            GPEC.equilibrium_config(kdeck, kin_template),
            GPEC.analytic_config(kdeck)
        )
        @test kequil.kinetic !== nothing
        kwall = GPEC.Vacuum.parse_wall(get(kdeck, "Wall", nothing))
        kkwargs = filter(p -> !(p.first in (:nn, :wall)), GPEC.force_balance_keywords(kdeck, kin_template))
        mktempdir() do dir
            for name in readdir(kin_template)
                cp(joinpath(kin_template, name), joinpath(dir, name))
            end
            reference = GPEC.main([dir]).ffs
            api = solve(kequil, GPEC.force_balance_algorithm(kdeck); nn=1, wall=kwall, kkwargs...)
            @test api.mats.kinetic !== nothing
            @test api.free_boundary.et[1] ≈ reference.free_boundary.et[1] rtol = 1e-10
            @test api.diagnostics.nzero == reference.diagnostics.nzero
            @test api.kinetic.kmsing == reference.kinetic.kmsing
            # This fixture solves with a wall, and the free-space surface inductance cannot pair with its Λ.
            @test_throws "without a wall" plasma_response(api)

            # The KineticForces matrices evaluated once solve identically when reused, and only on the problem they were made for.
            alg = GPEC.force_balance_algorithm(kdeck)
            km = kinetic_matrices(ForceBalanceProblem(kequil; nn=1, wall=kwall, kkwargs...), alg)
            @test km.source === kkwargs[:kinetic] && size(km.kw, 1) == length(km.psi) >= length(km.problem_psi)
            reused = solve(kequil, alg; nn=1, wall=kwall, merge(kkwargs, Dict(:kinetic => km))...)
            @test reused.free_boundary.et == api.free_boundary.et && reused.free_boundary.wt == api.free_boundary.wt
            @test reused.kinetic.kmsing == api.kinetic.kmsing
            shifted = merge(kkwargs, Dict(:kinetic => km, :delta_mlow => get(kkwargs, :delta_mlow, FFS.ForceFreeStatesControl().delta_mlow) + 1))
            @test_throws "kinetic matrices were made for" solve(kequil, alg; nn=1, wall=kwall, shifted...)
        end
    end

    @testset "attached kinetic profiles refine the two-pass grid" begin
        # main and solve both re-form through maybe_reform_equilibrium, which reads equil.kinetic, so
        # the attached profiles shape the grid the same way on either path. The DIII-D-like profiles
        # add knots over the ideal ones.
        sl_dir = joinpath(@__DIR__, "..", "examples", "DIIID-like_SLAYER_example")
        sl_deck = TOML.parsefile(joinpath(sl_dir, "gpec.toml"))
        sl_config = GPEC.equilibrium_config(sl_deck, sl_dir)
        pass1 = GPEC.Equilibrium.PlasmaEquilibrium(sl_config)
        @test pass1.kinetic !== nothing
        sl_ctrl = GPEC.force_balance_control(sl_deck, sl_dir)
        sl_intr = FFS.ForceFreeStatesInternal()
        FFS.resolve_mode_space!(sl_intr, sl_ctrl)
        refined = GPEC.maybe_reform_equilibrium(pass1, sl_intr, sl_ctrl)
        @test refined.kinetic === pass1.kinetic        # carried through the re-form, not re-read
        kinetic = pass1.kinetic
        pass1.kinetic = nothing
        ideal = GPEC.maybe_reform_equilibrium(pass1, sl_intr, sl_ctrl)
        pass1.kinetic = kinetic
        @test refined.rzphi_xs != ideal.rzphi_xs
        # Profiles without omega_E (read for SLAYER, say) cannot form the NTV view: the grid follows the
        # equilibrium alone, unless the run computes calculated kinetic matrices, which need it.
        no_omega = GPEC.Equilibrium.KineticProfileData(; psi=kinetic.psi, n_i=kinetic.n_i, n_e=kinetic.n_e,
            T_i=kinetic.T_i, T_e=kinetic.T_e, chi_e=kinetic.chi_e, chi_i=kinetic.chi_i, chi_phi=kinetic.chi_phi)
        @test GPEC.Equilibrium.has_ntv_fields(kinetic)
        @test !GPEC.Equilibrium.has_ntv_fields(no_omega)
        pass1.kinetic = no_omega
        no_omega_grid = GPEC.maybe_reform_equilibrium(pass1, sl_intr, sl_ctrl)
        @test no_omega_grid.kinetic === no_omega
        @test no_omega_grid.rzphi_xs == ideal.rzphi_xs
        calc_deck = merge(sl_deck, Dict("ForceFreeStates" => merge(sl_deck["ForceFreeStates"], Dict("kinetic" => "KineticForces"))))
        calc_ctrl = GPEC.force_balance_control(calc_deck, sl_dir)
        @test_throws "missing required 'omega_E'" GPEC.maybe_reform_equilibrium(pass1, sl_intr, calc_ctrl)
        # The KineticForces source stops up front, before any solve.
        @test_throws "omega_E" solve(pass1, FFS.Forward(); nn=1, kinetic=GPEC.KineticForcesControl())
        pass1.kinetic = kinetic
    end

    @testset "galerkin solve returns a gal result with a unified delta_prime" begin
        mktempdir() do dir
            api = solve(equil, Galerkin(; nx=32); nn=1, wall=wall, ffs_kwargs...)

            @test api.integrator === :galerkin
            @test api.control.algorithm.nx == 32
            @test api.galerkin !== nothing
            @test api.free_boundary === nothing        # galerkin computes no free-boundary energies
            dp = api.delta_prime
            @test dp !== nothing
            msing = api.galerkin.msing
            @test size(dp.matrix) == (msing, msing)
            @test dp.matrix ≈ FFS.pest3_decompose(dp.raw).Δ
            @test dp.A !== nothing                     # the parity blocks the galerkin solve persists

            reference =
                GPEC.main([
                    _stage_deck(dir; ffs_overrides=Dict{String,Any}("integrator" => "Galerkin"),
                        extra_sections=Dict{String,Any}("Galerkin" => Dict{String,Any}("nx" => 32)))
                ]).ffs
            @test reference.galerkin.msing == msing
            @test dp.matrix ≈ reference.delta_prime.matrix rtol = 1e-12
            @test dp.raw ≈ reference.delta_prime.raw rtol = 1e-12
            # With no free-boundary δW, no plasma response can be built from it.
            @test_throws ErrorException plasma_response(api)
        end
    end

    @testset "perturbed_equilibrium round-trips a forward result" begin
        forcing = joinpath(@__DIR__, "..", "examples", "TJ_ideal_example", "forcing.dat")
        pe_section = Dict{String,Any}(
            "compute_singular_coupling" => true)
        pe_kw = (; verbose=false)

        mktempdir() do dir
            _stage_deck(dir;
                extra_sections=Dict{String,Any}(
                    "ForcingTerms" => Dict{String,Any}(
                        "forcing_data_file" => "forcing.dat",
                        "forcing_data_format" => "ascii"),
                    "PerturbedEquilibrium" => pe_section))
            cp(forcing, joinpath(dir, "forcing.dat"))
            reference = GPEC.main([dir]).pe

            ffs = solve(equil, forward; nn=1, wall=wall, ffs_kwargs...)
            resp = plasma_response(ffs; verbose=false)
            pe = perturbed_equilibrium(resp, RMPField(forcing); pe_kw...)

            # The response matrices depend on the equilibrium and the solve, not on the drive.
            @test resp isa PlasmaResponse && pe.response === resp
            @test !isempty(resp.permeability)
            @test resp.permeability ≈ reference.response.permeability rtol = 1e-10
            @test size(resp.C_delta_prime) == size(reference.response.C_delta_prime)
            @test !isempty(pe.resonant_area_weighted_field)

            # The deck reads and normalizes its forcing file exactly as a fresh RMPField does.
            @test pe.resonant_area_weighted_field ≈ reference.resonant_area_weighted_field rtol = 1e-10
            @test pe.forcing_b ≈ reference.forcing_b rtol = 1e-10

            # One response serves any number of fields: forcing it again reproduces the result.
            again = perturbed_equilibrium(resp, RMPField(forcing); pe_kw...)
            @test again.forcing_b == pe.forcing_b
            @test again.resonant_area_weighted_field == pe.resonant_area_weighted_field

            # `scale` is a uniform multiplier on the materialized forcing, and the response is
            # linear in it.
            scaled = perturbed_equilibrium(resp, RMPField(forcing; scale=2.0); pe_kw...)
            @test scaled.forcing_b ≈ 2 .* pe.forcing_b rtol = 1e-10
            @test scaled.resonant_area_weighted_field ≈ 2 .* pe.resonant_area_weighted_field rtol = 1e-10

            # Lazy source algebra materializes to the combined field: 3A - A == 2A drives
            # the same perturbed equilibrium as scale=2 (exercises +, -, * and the merge).
            combo = perturbed_equilibrium(resp, 3 * RMPField(forcing) - RMPField(forcing); pe_kw...)
            @test combo.forcing_b ≈ scaled.forcing_b rtol = 1e-10
            @test combo.resonant_area_weighted_field ≈ scaled.resonant_area_weighted_field rtol = 1e-10

            # A spectrum-literal leaf of the materialized modes drives the same perturbed equilibrium,
            # takes a weight, and sums with a file leaf.
            literal = perturbed_equilibrium(resp, RMPField(pe.forcing_modes); pe_kw...)
            @test literal.forcing_b == pe.forcing_b
            @test literal.resonant_area_weighted_field == pe.resonant_area_weighted_field
            @test perturbed_equilibrium(resp, 2 * RMPField(pe.forcing_modes); pe_kw...).forcing_b ≈ scaled.forcing_b rtol = 1e-10
            @test perturbed_equilibrium(resp, RMPField(pe.forcing_modes) + RMPField(forcing); pe_kw...).forcing_b ≈ scaled.forcing_b rtol = 1e-10

            # No regularization by default; an explicit width changes the regularized profiles, not the control-surface quantities.
            @test pe.regularization_width == 0.0
            reg = perturbed_equilibrium(resp, RMPField(forcing); regularization_width=0.05, pe_kw...)
            @test reg.regularization_width == 0.05
            @test reg.forcing_b == pe.forcing_b
            @test !(reg.xi_modes.clebsch_alpha ≈ pe.xi_modes.clebsch_alpha)
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
        @test (a-b).terms[2].scale == -0.5 + 0.0im
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

    @testset "rejected keyword combinations" begin
        @test_throws "needs kinetic profiles on the equilibrium" solve(equil, forward; nn=1, ffs_kwargs...,
            kinetic=GPEC.KineticForces.KineticForcesControl())
        @test_throws "does not support kinetic runs" solve(equil, riccati; nn=1, ffs_kwargs..., kinetic=TestKineticMatrices(; scale=1e-9))
        @test_throws "does not support kinetic runs" solve(equil, Galerkin(); nn=1, ffs_kwargs..., kinetic=TestKineticMatrices(; scale=1e-9))
        @test_throws ArgumentError solve(equil, forward; nn=1, ffs_kwargs..., integrator="riccati")
        @test_throws ArgumentError solve(equil, riccati; nn=1, ffs_kwargs..., nchunks=8)
        @test_throws ErrorException solve(equil, forward; nn=1, ffs_kwargs..., nn_low=2)
        for (key, value) in ((:frobenius_psi_max, 0.1), (:truncate_at_dW_peak, true), (:dW_edge_scan_start, 0.9))
            @test_throws ArgumentError solve(equil, Galerkin(); nn=1, ffs_kwargs..., (key => value,)...)
        end
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
            gal = solve(equil, Galerkin(; nx=32, rpec_flag=true, cut_solution=true); nn=1, ffs_kwargs...)
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

    @testset "MatchProblem defaults its layer inputs from the kinetic data" begin
        gal = solve(equil, Galerkin(; nx=32, rpec_flag=true, cut_solution=true); nn=1, ffs_kwargs...)
        # Without kinetic data the error names exactly the inputs still to give.
        @test_throws "MatchProblem: give eta, rho and rotation, or load kinetic profiles" MatchProblem(gal)
        @test_throws "MatchProblem: give rho, or load kinetic profiles" MatchProblem(gal; eta=1e-7, rotation=1.0)
        n = length(gal.surfaces)
        # A flat deuterium plasma rotating at 1 kHz per unit n.
        psi = collect(range(0.0, 1.0; length=64))
        flat(v) = fill(v, length(psi))
        kgal = deepcopy(gal)
        attach_kinetic_profiles!(kgal.equil, KineticProfileData(; psi, n_e=flat(5e19), n_i=flat(5e19), T_e=flat(1e3), T_i=flat(1e3),
            omega_E=flat(2π * 1e3)))
        # The defaults are the tearing problem's: η by the resistivity closure, ρ = m_i·n_i, and f = Ω_E/2π.
        prob = MatchProblem(kgal)
        @test prob.eta == GPEC.InnerLayer.layer_eta(kgal.equil, kgal.surfaces) && all(>(0), prob.eta)
        @test prob.rho ≈ fill(2 * GPEC.Utilities.M_P * 5e19, n)
        @test prob.rotation ≈ fill(1e3, n)
        # A supplied Coulomb logarithm feeds only the NTV collision model; η keeps its own closure.
        keq = deepcopy(kgal.equil)
        attach_kinetic_profiles!(keq, KineticProfileData(; psi, n_e=flat(5e19), n_i=flat(5e19), T_e=flat(1e3), T_i=flat(1e3),
            omega_E=flat(2π * 1e3), lnLambda=flat(30.0)))
        @test GPEC.InnerLayer.layer_eta(keq, kgal.surfaces) == prob.eta
        # Explicit values override each default on its own.
        given = MatchProblem(kgal; eta=1e-7, rotation=fill(3.0, n))
        @test given.eta == fill(1e-7, n) && given.rho == prob.rho && given.rotation == fill(3.0, n)
        # The derived inputs solve the same as those values given explicitly.
        explicit = MatchProblem(kgal; eta=prob.eta, rho=prob.rho, rotation=prob.rotation)
        @test solve(prob, GGJModel()).bpen == solve(explicit, GGJModel()).bpen
        # Data without an E×B rotation need an explicit rotation.
        attach_kinetic_profiles!(kgal.equil, KineticProfileData(; psi, n_e=flat(5e19), n_i=flat(5e19), T_e=flat(1e3), T_i=flat(1e3)))
        @test_throws "give rotation" MatchProblem(kgal)
        @test MatchProblem(kgal; rotation=1.0).rotation == fill(1.0, n)
    end

    @testset "CCC matching" begin
        ric = solve(equil, riccati; nn=1, wall=wall, ffs_kwargs...)
        n = length(ric.surfaces)
        layer = (eta=1e-7, rho=1e-7, rotation=1.0)
        ggj = solve(MatchProblem(ric; layer...), GGJModel())
        # Q* = 0, M⊥/M = 1 is the GGJ layer, through CCC's own Wasow basis and splitting solve.
        @test solve(MatchProblem(ric; layer...), CCCModel(; Qstar=0.0)).bpen ≈ ggj.bpen rtol = 1e-6
        drift = solve(MatchProblem(ric; layer...), CCCModel(; Qstar=0.05, mperp=1.2))
        @test drift.closure === :matched && all(isfinite, drift.bpen) && drift.bpen != ggj.bpen
        @test drift.match.model === CCCModel(; Qstar=0.05, mperp=1.2) && all(p -> p.Qstar == 0.05 && p.mperp == 1.2, drift.match.inner_params)
        # Without kinetic data CCC needs Q* given.
        @test_throws "give Qstar, or load kinetic profiles" solve(MatchProblem(ric; layer...), CCCModel())
        # From kinetic data, ω*ₑ = (2π/χ₁)(T_e n_e′/n_e + T_e′) per unit n, so a flat density and T_e = 1 keV·(1 − ψ/2) give
        # (2π/χ₁)·(−500 eV); Q* = n·ω*ₑ/Q₀ at each surface.
        psi = collect(range(0.0, 1.0; length=64))
        keq = deepcopy(ric.equil)
        attach_kinetic_profiles!(
            keq,
            KineticProfileData(; psi, n_e=fill(5e19, 64), n_i=fill(5e19, 64), T_e=1e3 .* (1 .- psi ./ 2),
                T_i=1e3 .* (1 .- psi ./ 2), omega_E=fill(2π * 1e3, 64))
        )
        ωse = GPEC.InnerLayer.layer_omega_star_e(keq, ric.surfaces)
        @test ωse ≈ fill(-500 / keq.psio, n) rtol = 1e-6
        ggj1 = FFS.ggj_parameters(ric.surfaces[1], keq; eta=1e-7, rho=1e-7, ising=1)
        p = GPEC.InnerLayer.layer_parameters(CCCModel(), ggj1, keq, ric.surfaces[1])
        @test p.ggj === ggj1 && p.mperp == 1.0
        @test p.Qstar ≈ ric.nlow * ωse[1] / GPEC.InnerLayer.growth_rate_scale(ggj1) rtol = 1e-12
    end

    @testset "CCC and SLAYER put the electron-frame resonance at the same lab frequency" begin
        # With s = γ − iω, SLAYER's layer co-rotates with the electrons where ĝ − i·Q_e = 0, and CCC's where Q + i·Q* = 0 (Kim
        # Eq. 1a). Both drift terms come from the same kinetic data, so at γ = 0 both zeros sit at ω_lab = n·Ω_E + ω*ₑ.
        ric = solve(equil, riccati; nn=1, wall=wall, ffs_kwargs...)
        psi = collect(range(0.0, 1.0; length=64))
        keq = deepcopy(ric.equil)
        attach_kinetic_profiles!(
            keq,
            KineticProfileData(; psi, n_e=fill(5e19, 64), n_i=fill(5e19, 64), T_e=1e3 .* (1 .- psi ./ 2),
                T_i=1e3 .* (1 .- psi ./ 2), omega_E=fill(2π * 1e3, 64))
        )
        TR, DS, IL = GPEC.Tearing, GPEC.Tearing.Dispersion, GPEC.InnerLayer
        sing = ric.surfaces[1]
        nn = only(sing.n)
        slayer_model = SLAYERModel(; P_perp_model=:chi_perp_e)
        sp = only(TR._tearing_params(slayer_model, keq, [sing], TR._slayer_profiles(keq.kinetic, slayer_model)))
        cp = IL.layer_parameters(CCCModel(), FFS.ggj_parameters(sing, keq; eta=1e-7, rho=1e-7, ising=1), keq, sing)
        ω_star_e = -sp.Q_e / sp.tauk                       # SLAYER's ω*ₑ (Q_e = −τ_k·ω*ₑ), the full mode's
        @test ω_star_e ≈ nn * IL.layer_omega_star_e(keq, [sing])[1] rtol = 1e-10
        # A linear residual's zero in the real lab frequency ω.
        zero_of(f) = (f0 = f(0.0); -real(f0 / (f(1.0) - f0)))
        for Ω_E in (0.0, 2π * 3e3)
            # SLAYER scans Q_lab = τ_k·ω_lab (γ = 0); its layer sees ĝ = −i·(Q_lab + q_shift).
            slayer = DS.surface_coupling(slayer_model, sp, 0.0; q_shift=TR._q_shift(sp, Ω_E))
            ω_slayer = zero_of(ω -> -im * DS.layer_argument(slayer, sp.tauk * ω) - im * sp.Q_e)
            # CCC's determinant takes the lab-frame s = −i·ω (γ = 0); its layer sees Q = inner_Q(s + i·n·Ω_E).
            ccc = DS.surface_coupling(CCCModel(), cp, 0.0; omega_E=Ω_E, n=nn)
            ω_ccc = zero_of(ω -> IL.inner_Q(cp, DS.layer_argument(ccc, -im * ω)) + im * cp.Qstar)
            @test ω_slayer ≈ nn * Ω_E + ω_star_e rtol = 1e-10
            @test ω_ccc ≈ nn * Ω_E + ω_star_e rtol = 1e-10
        end
    end

    @testset "matched inner profiles: opt-in, and parity columns agree across inner models" begin
        mktempdir() do dir
            gal = solve(equil, Galerkin(; nx=32, rpec_flag=true, cut_solution=true); nn=1, ffs_kwargs...)
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
            ric = solve(equil, riccati; nn=1, ffs_kwargs...)
            @test ric.free_boundary !== nothing && !isempty(ric.delta_prime.coil)
            matched = solve(MatchProblem(ric; eta=1e-7, rho=1e-7, rotation=1.0), GGJModel())
            @test matched.closure === :matched
            @test matched.match !== nothing && matched.bpen == matched.match.bpen
            @test matched.wp === nothing && matched.free_boundary === nothing
            @test matched.solution === nothing    # no Galerkin basis to build a matched ξ from
            # It carries no ideal δW, so no plasma response can be built from it.
            @test_throws ErrorException plasma_response(matched)
            # A matched result is a new solution, not an outer solve to match again.
            @test_throws ErrorException MatchProblem(matched; ideal=true)
        end
    end

    @testset "MatchProblem gates on its inputs" begin
        mktempdir() do dir
            # A Forward result carries no Δ′ payload, so the problem is unconstructible.
            fwd = solve(equil, forward; nn=1, ffs_kwargs...)
            @test_throws ErrorException MatchProblem(fwd; ideal=true)
            # A slab model can never close a matched solution.
            gal = solve(equil, Galerkin(; nx=32, rpec_flag=true); nn=1, ffs_kwargs...)
            prob = MatchProblem(gal; ideal=true)
            @test_throws ErrorException solve(prob, SLAYERModel())
            # A resistive match needs every layer input.
            @test_throws "give rho and rotation" MatchProblem(gal; eta=1e-7)
            @test_throws ErrorException MatchProblem(gal; eta=fill(1e-7, length(gal.surfaces) + 1), rho=1e-7, rotation=1.0)
            # The tearing model is the solve argument, never a TearingProblem keyword.
            @test_throws MethodError TearingProblem(gal; inner_model=:ggj_ray)
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
            pe_kwargs = (; compute_singular_coupling=true)
            mktempdir() do dir
                modes = GPEC.ForcingTerms.ForcingMode[]
                GPEC.ForcingTerms.load_forcing_data!(modes, joinpath(@__DIR__, "..", "examples", "TJ_ideal_example", "forcing.dat"), "ascii", false)
                h5open(f -> GPEC.ForcingTerms.save_forcing_to_h5(modes, f), joinpath(dir, "forcing.h5"), "w")
                _stage_deck(dir;
                    extra_sections=Dict{String,Any}(
                        "ForcingTerms" => Dict{String,Any}("forcing_data_file" => "forcing.h5", "forcing_data_format" => "hdf5"),
                        "PerturbedEquilibrium" => Dict{String,Any}(String(k) => v for (k, v) in pairs(pe_kwargs))))
                reference = GPEC.main([dir])
                resp = plasma_response(reference.ffs; compute_singular_coupling=true, verbose=false)
                pe = perturbed_equilibrium(resp, RMPField(joinpath(dir, "forcing.h5")); verbose=false)

                @test pe.response.permeability ≈ reference.pe.response.permeability rtol = 1e-10
                # The deck reads and normalizes its forcing file exactly as a fresh RMPField does.
                @test pe.forcing_b ≈ reference.pe.forcing_b rtol = 1e-10
            end
        end

        @testset "analytic equilibrium on the two-pass auto grid" begin
            # A coarse psi_accuracy keeps the 1024-interval pass-1 grid cheap; the deck runs no local-stability scan.
            eq_overrides = Dict{String,Any}("grid_type" => "auto", "mpsi" => 0, "psi_accuracy" => 3e-2)
            ffs_overrides = Dict{String,Any}("vac_flag" => false)
            mktempdir() do dir
                reference = GPEC.main([_stage_deck(dir; eq_overrides=eq_overrides, ffs_overrides=ffs_overrides)]).ffs
                pass1 = GPEC.Equilibrium.PlasmaEquilibrium(
                    GPEC.equilibrium_config(merge(deck, Dict{String,Any}("Equilibrium" => merge(deck["Equilibrium"], eq_overrides))), template),
                    GPEC.analytic_config(deck)
                )
                @test reference.equil.rzphi_xs != pass1.rzphi_xs    # the deck re-formed on a refined grid

                kwargs = merge(ffs_kwargs, Dict(Symbol(k) => v for (k, v) in ffs_overrides))
                api = solve(pass1, forward; nn=1, wall=wall, kwargs...)
                # The equilibrium stores its analytic config, so `solve` re-forms it on the deck's grid.
                @test api.equil.rzphi_xs == reference.equil.rzphi_xs
            end
        end

        @testset "deck tearing under [Tearing] with the CCC layer" begin
            t = GPEC.tearing_from_deck(
                Dict{String,Any}("Tearing" => Dict{String,Any}("inner_model" => "CCC"),
                    "CCC" => Dict{String,Any}("cfl" => 0.1, "Qstar" => [0.05, 0.1], "mperp" => 1.2))
            )
            @test t.model isa CCCModel{:ray} && t.model.options == (; cfl=0.1) && t.model.Qstar == [0.05, 0.1] && t.model.mperp == 1.2
            # The model's deck section round-trips.
            @test GPEC.toroidal_layer_model(GPEC.deck_sections(t.model), "CCC").Qstar == [0.05, 0.1]
            # A [CCC] section no stage uses is refused.
            mktempdir() do dir
                deck_dir = _stage_deck(dir; extra_sections=Dict{String,Any}("CCC" => Dict{String,Any}("cfl" => 0.1)))
                @test_throws "[CCC] is set but neither" GPEC.parse_deck(TOML.parsefile(joinpath(deck_dir, "gpec.toml")), deck_dir)
            end
        end

        @testset "deck matching under [Match]" begin
            layer = (eta=1e-7, rho=1e-7, rotation=1.0)
            # The deck's [Match] section for `n` surfaces and the [GGJ] solver.
            match_sections(n, solver) = Dict{String,Any}("GGJ" => Dict{String,Any}("solver" => solver),
                "Match" => Dict{String,Any}("eta" => fill(layer.eta, n), "rho" => fill(layer.rho, n), "rotation" => fill(layer.rotation, n)))

            mktempdir() do dir
                gal = solve(equil, Galerkin(; nx=32, rpec_flag=true); nn=1, wall=wall, ffs_kwargs...)
                # The galerkin inner model returns real-ψ profiles, so the inner profiles are compared too.
                api = solve(MatchProblem(gal; layer...), GGJModel(; solver=:galerkin))
                n = length(gal.surfaces)
                sections = merge(Dict{String,Any}("Galerkin" => Dict{String,Any}("nx" => 32, "rpec_flag" => true)), match_sections(n, "galerkin"))
                reference = GPEC.main([_stage_deck(dir; ffs_overrides=Dict{String,Any}("integrator" => "Galerkin"), extra_sections=sections)]).ffs

                @test reference.closure === api.closure === :matched
                @test h5open(f -> haskey(f, "Info/Runtimes/match"), joinpath(dir, "gpec.h5"))
                @test api.bpen ≈ reference.bpen rtol = 1e-12
                @test api.match.xi ≈ reference.match.xi rtol = 1e-12
                # The composite inner profiles need the cut solution, which Galerkin keeps only on request.
                @test isempty(reference.match.inner_xi) && isempty(api.match.inner_xi)
                cut = solve(equil, Galerkin(; nx=32, rpec_flag=true, cut_solution=true); nn=1, wall=wall, ffs_kwargs...)
                @test length(solve(MatchProblem(cut; layer...), GGJModel(; solver=:galerkin)).match.inner_xi) == n
            end

            mktempdir() do dir
                ric = solve(equil, riccati; nn=1, wall=wall, ffs_kwargs...)
                api = solve(MatchProblem(ric; layer...), GGJModel(; solver=:ray))
                reference = GPEC.main([_stage_deck(dir; ffs_overrides=Dict{String,Any}("integrator" => "Riccati"),
                    extra_sections=match_sections(length(ric.surfaces), "ray"))]).ffs

                @test reference.closure === :matched
                @test api.bpen ≈ reference.bpen rtol = 1e-12
                @test api.match.cout ≈ reference.match.cout rtol = 1e-12
            end

            # The CCC layer under [Match], named by inner_model with its options in [CCC].
            mktempdir() do dir
                ric = solve(equil, riccati; nn=1, wall=wall, ffs_kwargs...)
                n = length(ric.surfaces)
                api = solve(MatchProblem(ric; layer...), CCCModel(; Qstar=fill(0.05, n), mperp=fill(1.2, n)))
                sections = Dict{String,Any}("CCC" => Dict{String,Any}("solver" => "ray", "Qstar" => fill(0.05, n), "mperp" => fill(1.2, n)),
                    "Match" => Dict{String,Any}("inner_model" => "CCC", "eta" => fill(layer.eta, n), "rho" => fill(layer.rho, n),
                        "rotation" => fill(layer.rotation, n)))
                reference = GPEC.main([_stage_deck(dir; ffs_overrides=Dict{String,Any}("integrator" => "Riccati"), extra_sections=sections)]).ffs

                @test reference.closure === :matched && reference.match.model isa CCCModel{:ray} && reference.match.model.Qstar == fill(0.05, n)
                @test api.bpen ≈ reference.bpen rtol = 1e-12
                @test api.match.cout ≈ reference.match.cout rtol = 1e-12
            end

            # A forward solve has no Δ′ to match against, so the deck stops at parse time instead of after the solve.
            mktempdir() do dir
                deck_dir = _stage_deck(dir; extra_sections=Dict{String,Any}("Match" => Dict{String,Any}("ideal" => true)))
                @test_throws "[Match] needs Δ′" GPEC.parse_deck(TOML.parsefile(joinpath(deck_dir, "gpec.toml")), deck_dir)
            end

            # A perturbed equilibrium with a wall stops at parse time.
            mktempdir() do dir
                walled = Dict{String,Any}("Wall" => Dict{String,Any}("shape" => "conformal", "standoff_fraction" => 0.2),
                    "PerturbedEquilibrium" => Dict{String,Any}("compute_singular_coupling" => true))
                deck_dir = _stage_deck(dir; extra_sections=walled)
                @test_throws "without a wall" GPEC.parse_deck(TOML.parsefile(joinpath(deck_dir, "gpec.toml")), deck_dir)
            end
        end

        @testset "kinetic_format" begin
            # A rates-v1 kinetic file read through the deck's [Equilibrium] keys and through the API keyword.
            mktempdir() do dir
                psi = range(0.0, 1.0; length=11)
                open(joinpath(dir, "rates.kin"), "w") do io
                    for x in psi
                        println(io, join((x, 5e19, 5e19, 1e3, 1e3, 1e3, 16.0 + x, 1e3 * exp(x), 1e5 * exp(-x)), " "))
                    end
                end
                eq_overrides = Dict{String,Any}("kinetic_file" => "rates.kin", "kinetic_format" => "rates-v1")
                deck_dir = _stage_deck(dir; eq_overrides=eq_overrides)
                from_deck = GPEC.Equilibrium.configured_kinetic_profiles(GPEC.equilibrium_config(TOML.parsefile(joinpath(deck_dir, "gpec.toml")), deck_dir))
                from_api = attach_kinetic_profiles!(deepcopy(equil), joinpath(dir, "rates.kin"); format="rates-v1").kinetic
                @test from_deck.nu_i == from_api.nu_i && from_deck.lnLambda == from_api.lnLambda && from_deck.nu_e == from_api.nu_e
            end
        end

        @testset "deck kinetic source names" begin
            # A deck selects only the KineticForces model; test matrices and the retired key stop the parse.
            parsed(overrides) =
                mktempdir() do dir
                    GPEC.parse_deck(TOML.parsefile(joinpath(_stage_deck(dir; ffs_overrides=overrides), "gpec.toml")), dir)
                end
            @test_throws "script-only" parsed(Dict{String,Any}("kinetic" => "TestKineticMatrices"))
            @test_throws "not a kinetic source" parsed(Dict{String,Any}("kinetic" => "fixed"))
            @test_throws "no longer a key" parsed(Dict{String,Any}("kinetic_source" => "fixed"))
            # The removed large-aspect-ratio torque methods stop the parse whatever their value.
            for key in ("rlar_flag", "clar_flag"), value in (true, false)
                mktempdir() do dir
                    deck_dir = _stage_deck(dir; extra_sections=Dict{String,Any}("KineticForces" => Dict{String,Any}(key => value)))
                    @test_throws "large-aspect-ratio methods are not ported" GPEC.parse_deck(TOML.parsefile(joinpath(deck_dir, "gpec.toml")), deck_dir)
                end
            end
        end

        @testset "kinetic torque" begin
            # The deck's [KineticForces] stage is kinetic_torque on its perturbed equilibrium, saved by its save method.
            ntv_dir = joinpath(@__DIR__, "..", "examples", "TJ_kinetic_NTV_example")
            mktempdir() do dir
                for name in readdir(ntv_dir)
                    cp(joinpath(ntv_dir, name), joinpath(dir, name))
                end
                ntv_deck = TOML.parsefile(joinpath(dir, "gpec.toml"))
                delete!(ntv_deck, "LocalStability")
                delete!(ntv_deck, "Ballooning")
                ntv_deck["Run"]["verbose"] = false
                open(io -> TOML.print(io, ntv_deck), joinpath(dir, "gpec.toml"), "w")
                reference = GPEC.main([dir])
                @test reference.kinetic isa KineticForcesResult && reference.kinetic.pe === reference.pe
                _check_display(reference.kinetic)

                kf = kinetic_torque(reference.pe, GPEC.kinetic_forces_control(ntv_deck))
                @test sort!(collect(keys(kf.total.method_results))) == sort!(collect(keys(reference.kinetic.total.method_results)))
                for (method, r) in kf.total.method_results
                    @test r.total_torque == reference.kinetic.total.method_results[method].total_torque
                end

                # The collision keys reach the torque path: a zero collisionality factor is the collisionless operator.
                kf_with(kw...) =
                    kinetic_torque(reference.pe, GPEC.kinetic_forces_control(merge(ntv_deck, Dict("KineticForces" => merge(ntv_deck["KineticForces"], Dict{String,Any}(kw...))))))
                t_zero = kf_with("nutype" => "zero").total.method_results["fgar"].total_torque
                @test t_zero != reference.kinetic.total.method_results["fgar"].total_torque
                @test t_zero ≈ kf_with("collisionality_factor" => 0.0).total.method_results["fgar"].total_torque rtol = 1e-6
                @test kf.axis_psi_c == reference.kinetic.axis_psi_c

                api_file = joinpath(dir, "api.h5")
                save_to_hdf5(api_file, kf)
                h5open(joinpath(dir, "gpec.h5")) do deck_h5
                    h5open(api_file) do api_h5
                        for method in keys(kf.total.method_results)
                            @test read(api_h5["KineticForces/$method/total_torque"]) == read(deck_h5["KineticForces/$method/total_torque"])
                        end
                        @test read(api_h5["KineticForces/Validity/psi_c"]) == read(deck_h5["KineticForces/Validity/psi_c"])
                    end
                end

                # With qlow above q0 the response starts off axis, and the torque quadrature starts with it.
                ntv_deck["ForceFreeStates"]["qlow"] = 1.9
                open(io -> TOML.print(io, ntv_deck), joinpath(dir, "gpec.toml"), "w")
                raised = GPEC.main([dir])
                @test raised.pe.psi_grid[1] > ntv_deck["Equilibrium"]["psilow"]
                @test minimum(raised.kinetic.total.method_results["fgar"].psi_grid) >= raised.pe.psi_grid[1]
            end
        end
    end
end
