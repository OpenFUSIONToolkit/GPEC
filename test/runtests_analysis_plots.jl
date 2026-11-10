using CairoMakie
using HDF5
using TOML

# Every registered plot against real writer output, from the result in memory and from the gpec.h5 it was saved to:
# both sources must supply the same data, draw the same plots, and refuse the same ones with the registry's reason.
@testset "Analysis plots from a result or gpec.h5" begin
    GPEC = GeneralizedPerturbedEquilibrium
    A = GPEC.Analysis

    # Copy a coarse fixture deck into `dir` with HDF5 output on and the given extra sections; return main's results and its gpec.h5.
    function _run_fixture(dir, fixture; sections=Dict{String,Any}(), ffs=Dict{String,Any}(), edit=identity)
        template = joinpath(@__DIR__, "test_data", fixture)
        for name in readdir(template)
            cp(joinpath(template, name), joinpath(dir, name))
        end
        toml_path = joinpath(dir, "gpec.toml")
        inputs = TOML.parsefile(toml_path)
        merge!(inputs, sections)
        merge!(inputs["ForceFreeStates"], ffs)
        edit(inputs)
        open(io -> TOML.print(io, inputs), toml_path, "w")
        return GPEC.main([dir]), joinpath(dir, "gpec.h5")
    end

    # The data extractors each dual-source plot reads, by the capability that guards them.
    extractors = (
        (A.ForceFreeStates._mode_displacement_data, :solution_profiles), (A.ForceFreeStates._criterion_data, :fixed_boundary_criterion),
        (A.ForceFreeStates._energy_eigenvector_data, :free_boundary), (A.ForceFreeStates._delta_prime_data, :delta_prime),
        (A.ForceFreeStates._cond_fbar_data, :kinetic_scan), (A.Equilibrium._global_data, :equilibrium),
        (A.PerturbedEquilibrium._coupling_data, :singular_coupling))
    _same(a::NamedTuple, b::NamedTuple) = keys(a) == keys(b) && all(_same(a[k], b[k]) for k in keys(a))
    _same(a::AbstractArray{<:Number}, b::AbstractArray{<:Number}) = size(a) == size(b) && isapprox(a, b; rtol=1e-10, nans=true)
    _same(a::Number, b::Number) = isapprox(a, b; rtol=1e-12)
    _same(a, b) = isequal(a, b)

    # Every plot registered for `x` has a method for each source it lists, draws from `x` and from `h5` when the
    # capabilities allow, and otherwise throws the registry's reason from both.
    function _check_sources(x, h5)
        caps = A.capabilities(x)
        # The file holds every capability a plot can read from a path (coil geometry is in memory only).
        path_needs = union((e.needs for e in A.PLOTS if A._accepts(e, h5))...)
        @test intersect(caps, path_needs) ⊆ A.capabilities(h5)
        for (f, cap) in extractors
            cap in caps && hasmethod(f, Tuple{typeof(x)}) && @test _same(f(x), f(h5))
        end
        for e in A.PLOTS
            A._accepts(e, x) || continue
            @test all(T -> hasmethod(e.fn, Tuple{T}), e.sources)
            sources = A._accepts(e, h5) ? (x, h5) : (x,)
            if A._can(e.fn, caps)
                for src in sources
                    @test e.fn(src) isa Figure
                end
            else
                reason = A._why(A._missing(e, caps))
                for src in sources
                    err = try
                        e.fn(src)
                        nothing
                    catch caught
                        caught
                    end
                    @test err isa ArgumentError && occursin(reason, err.msg)
                end
            end
        end
    end

    @testset "ideal Forward run with coil forcing, the response and local stability" begin
        mktempdir() do dir
            ret, h5 = _run_fixture(
                dir,
                "regression_tj_ideal_example";
                ffs=Dict{String,Any}("dW_edge_scan_start" => 0.95),
                sections=Dict{String,Any}(
                    "ForcingTerms" => Dict{String,Any}("forcing_data_format" => "coil", "mtheta_coil" => 240, "nzeta_coil" => 32,
                        "coil_set" => [Dict{String,Any}("name" => "hoop", "source" => "pf_hoop", "radius" => 12.0, "height" => 1.2,
                            "currents" => [2.0e3], "tiltx" => [3.0])]),
                    "PerturbedEquilibrium" => Dict{String,Any}("compute_singular_coupling" => true))
            )
            for x in (ret.equil, ret.ffs, ret.resp, ret.pe)
                _check_sources(x, h5)
            end
            ret.local_stability === nothing || _check_sources(ret.local_stability, h5)
            for component in (:xi_psi, :b_psi, :Jb_theta, :Jb_zeta)
                @test _same(A.PerturbedEquilibrium._response_data(ret.pe, component), A.PerturbedEquilibrium._response_data(h5, component))
                @test A.PerturbedEquilibrium.plot_mode_spectrogram(ret.pe; component) isa Figure
            end
            @test_throws ArgumentError A.PerturbedEquilibrium.plot_mode_spectrogram(h5; component=:b_theta)
            @test A.PerturbedEquilibrium.plot_mode_spectrogram(h5; style=:lines) isa Figure
            @test_throws ArgumentError A.PerturbedEquilibrium.plot_mode_spectrogram(ret.pe; style=:contour)

            # Field views that take more than the result: any control-surface spectrum, the displaced boundary, the 3D surface.
            CF = A.CoilForcing
            @test CF.plot_control_surface(ret.pe, ret.pe.forcing_b_rootarea; label="the applied forcing", coils=true) isa Figure
            @test CF.plot_control_surface(ret.pe, ret.pe.forcing_b_rootarea ./ maximum(abs, ret.pe.forcing_b_rootarea); arb=true) isa Figure
            @test_throws ArgumentError CF.plot_control_surface(ret.pe, ret.pe.forcing_b; weighting=:flux)
            @test A.PerturbedEquilibrium.plot_perturbed_field_rz(ret.pe; component=:xi_n, deform=true) isa Figure
            @test CF.plot_surface_3d(ret.equil, ret.pe.forcing_modes, 1; coil_sets=ret.pe.coil_sets, mtheta=60, nphi=60) isa Figure
            @test CF.plot_bn_contour(randn(32, 16), 32, 16; n=1) isa Figure

            # The response alone draws its dominant modes, with no forcing; a perturbed equilibrium inherits them.
            @test A.capabilities(ret.resp) ⊆ A.capabilities(ret.pe)
            @test plot_dominant_mode_surface(ret.resp; mode=length(ret.resp.dominant_singular_values)) isa Figure
            @test_throws "mode must be 1 to" plot_dominant_mode_surface(ret.resp; mode=length(ret.resp.dominant_singular_values) + 1)
            @test_throws "coils = true needs" CF.plot_control_surface(ret.resp, ret.resp.dominant_right_singular_vectors[:, 1]; coils=true)
            # A bare equilibrium marks the q = m/n crossings of n = 1, which include every surface the solve found.
            bare, solved = A.Equilibrium._surfaces(ret.equil), A.Equilibrium._surfaces(ret.ffs)
            @test all(q -> any(≈(q), bare.sing_q), solved.sing_q) && !isempty(bare.sing_q)
            @test plot_equilibrium_summary(ret.equil; n=2) isa Figure
            # An RMPField draws its coil sources in 3D before any plasma; a mode table has no geometry to draw.
            hoop = RMPField((name="hoop", source="pf_hoop", radius=12.0, height=1.2, currents=[2.0e3], tiltx=[3.0]))
            @test plot_sources(hoop) isa Figure && plot_sources(2hoop - 0.5im * hoop; color=:set) isa Figure
            @test_throws A.REASONS[:coil_source] plot_sources(RMPField([GPEC.ForcingTerms.ForcingMode(; n=1, m=2, amplitude=1e-4)]))
            # A summary none of whose panels this run supplies is listed as unavailable, and says so when called.
            @test occursin("✗ plot_error_field_summary", repr(MIME"text/plain"(), available_plots(h5)))
            @test_throws A.REASONS[:ef_sensitivities] plot_error_field_summary(h5)

            # The catalog lists a Forward run's absent products with what to change, for a result, a path and main's tuple.
            listing = repr(MIME"text/plain"(), available_plots(ret.ffs))
            @test occursin(r"\nForceFreeStates\n", listing) && occursin("✓ plot_mode_displacement", listing)
            @test occursin("✗ plot_delta_prime ", listing) && occursin(A.REASONS[:delta_prime], listing)
            @test length(available_plots(ret).sections) == count(x -> x !== nothing && any(e -> A._accepts(e, x), A.PLOTS), unique(values(ret)))
            @test A.plot_counts(ret.ffs) == (11, 15)

            # Optional outputs this run does not produce throw instead of drawing a placeholder.
            @test_throws ArgumentError A.ForceFreeStates.plot_cond_fbar(h5)

            # The Grad-Shafranov residual plot is an unregistered stub that draws the residual computed in memory.
            @test A.Equilibrium.plot_gse(GPEC.Equilibrium.equilibrium_gse!(ret.equil)) isa Figure

            png = joinpath(dir, "qprofile.png")
            A.Equilibrium.plot_qprofile(ret.equil; save_path=png)
            @test isfile(png)

            theta_data, theta_grid, n_vals = A.PerturbedEquilibriumModes.modes_to_theta(h5, "PerturbedEquilibrium/Response/xi_psi")
            npsi = h5open(f -> length(read(f, "PerturbedEquilibrium/Response/psi")), h5)
            @test size(theta_data) == (npsi, length(theta_grid), length(n_vals))
        end
    end

    @testset "Riccati run offers Δ′ but no solution profiles" begin
        mktempdir() do dir
            ret, h5 = _run_fixture(dir, "regression_solovev_ideal_example";
                edit=d -> (d["ForceFreeStates"]["integrator"] = "Riccati"; d["Riccati"] = pop!(d, "Forward")))
            _check_sources(ret.ffs, h5)
            @test :delta_prime in A.capabilities(ret.ffs) && :solution_profiles ∉ A.capabilities(ret.ffs)
            @test_throws A.REASONS[:solution_profiles] A.ForceFreeStates.plot_mode_displacement(ret.ffs)
            @test occursin("✗ plot_mode_displacement", repr(MIME"text/plain"(), available_plots(ret.ffs)))
            # The summary leaves out the ξ panel rather than failing, keeping the other four.
            caps = A.capabilities(ret.ffs)
            @test count(((f, _),) -> A._can(f, caps), A.ForceFreeStates._ffs_panels(ret.ffs)) == 4
            @test A.ForceFreeStates.plot_ffs_summary(ret.ffs) isa Figure
            # The Δ′ family draws any part of its complex entries and refuses others.
            @test A.ForceFreeStates.plot_delta_prime(ret.ffs; part=:abs) isa Figure
            @test A.ForceFreeStates.plot_delta_ef(h5; part=:imag) isa Figure
            @test_throws ArgumentError A.ForceFreeStates.plot_delta_prime(ret.ffs; part=:phase)
        end
    end

    @testset "kinetic run draws the F̄ condition-number scan" begin
        # Synthetic test kinetic matrices are script-only, so this run goes through the API and saves its result.
        mktempdir() do dir
            inputs, config, analytic = GPEC.build_inputs_from_toml(joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example"))
            fixture = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
            kwargs = merge(GPEC.force_balance_keywords(inputs, fixture), Dict(:kinetic => TestKineticMatrices(; scale=1e-9)))
            prob = ForceBalanceProblem(PlasmaEquilibrium(config, analytic); kwargs...)
            ffs = solve(prob, GPEC.force_balance_algorithm(inputs))
            h5 = joinpath(dir, "gpec.h5")
            save_to_hdf5(h5, ffs)
            _check_sources(ffs, h5)
            @test A.ForceFreeStates.plot_cond_fbar(ffs) isa Figure
        end
    end
end
