using HDF5
using Plots
using TOML

# Smoke tests of the gpec.h5 plots against real writer output: each plot must read the datasets the
# writers emit, so a renamed path fails here instead of silently drawing a "No ... data" placeholder.
@testset "Analysis plots from gpec.h5" begin
    ENV["GKSwstype"] = "100"
    GPEC = GeneralizedPerturbedEquilibrium
    A = GPEC.Analysis

    # Copy a coarse fixture deck into `dir` with HDF5 output on and the given extra sections; return its gpec.h5.
    function _run_fixture(dir, fixture; sections=Dict{String,Any}())
        template = joinpath(@__DIR__, "test_data", fixture)
        for name in readdir(template)
            cp(joinpath(template, name), joinpath(dir, name))
        end
        toml_path = joinpath(dir, "gpec.toml")
        inputs = TOML.parsefile(toml_path)
        inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = true
        merge!(inputs, sections)
        open(io -> TOML.print(io, inputs), toml_path, "w")
        GPEC.main([dir])
        return joinpath(dir, "gpec.h5")
    end
    _placeholder(p) = any(sp -> startswith(string(get(sp.attr, :title, "")), "No "), p.subplots)
    _drew(p) = p isa Plots.Plot && !_placeholder(p)

    @testset "ideal run with coil forcing and the perturbed-equilibrium response" begin
        mktempdir() do dir
            h5 = _run_fixture(dir, "regression_solovev_ideal_example"; sections=Dict{String,Any}(
                "ForcingTerms" => Dict{String,Any}("forcing_data_format" => "coil", "mtheta_coil" => 240, "nzeta_coil" => 32,
                    "coil_set" => [Dict{String,Any}("name" => "hoop", "source" => "pf_hoop", "radius" => 1.5, "height" => 0.4,
                        "currents" => [2.0e3], "tiltx" => [3.0])]),
                "PerturbedEquilibrium" => Dict{String,Any}("compute_singular_coupling" => true,
                    "verbose" => false, "write_outputs_to_HDF5" => true)))

            for f in (A.Equilibrium.plot_qprofile, A.Equilibrium.plot_pressure_profile, A.Equilibrium.plot_f_profile,
                A.Equilibrium.plot_flux_surfaces, A.Equilibrium.plot_equilibrium_summary)
                @test _drew(f(h5))
            end
            for f in (A.ForceFreeStates.plot_mode_displacement, A.ForceFreeStates.plot_fixed_boundary_stability_criterion,
                A.ForceFreeStates.plot_energy_eigenvectors, A.ForceFreeStates.plot_edge_stability_scan,
                A.ForceFreeStates.plot_eigenvalues, A.ForceFreeStates.plot_delta_prime,
                A.ForceFreeStates.plot_ballooning_alpha_boundary, A.ForceFreeStates.plot_ffs_summary)
                @test _drew(f(h5))
            end
            for f in (A.PerturbedEquilibrium.plot_resonant_area_weighted_field_amplitude, A.PerturbedEquilibrium.plot_island_widths,
                A.PerturbedEquilibrium.plot_chirikov_parameter, A.PerturbedEquilibrium.plot_driven_delta_prime,
                A.PerturbedEquilibrium.plot_resonant_area_weighted_field, A.PerturbedEquilibrium.plot_perturbed_equilibrium_summary)
                @test _drew(f(h5))
            end
            for component in (:xi_psi, :b_psi, :Jb_theta, :Jb_zeta)
                @test _drew(A.PerturbedEquilibrium.plot_mode_spectrogram(h5; component=component))
            end
            @test_throws ErrorException A.PerturbedEquilibrium.plot_mode_spectrogram(h5; component=:b_theta)

            # Optional outputs this run does not produce: no kinetic scan, no GSE sidecar files.
            @test (@test_logs (:warn,) A.ForceFreeStates.plot_cond_fbar(h5)) === nothing
            @test A.Equilibrium.plot_gse_by_theta(h5) === nothing
            @test A.Equilibrium.plot_gse_integrated(h5) === nothing

            png = joinpath(dir, "qprofile.png")
            A.Equilibrium.plot_qprofile(h5; save_path=png)
            @test isfile(png)

            theta_data, theta_grid, n_vals = A.PerturbedEquilibriumModes.modes_to_theta(h5, "PerturbedEquilibrium/Response/xi_psi")
            npsi = h5open(f -> length(read(f, "PerturbedEquilibrium/Response/psi")), h5)
            @test size(theta_data) == (npsi, length(theta_grid), length(n_vals))
        end
    end

    @testset "kinetic run draws the F̄ condition-number scan" begin
        mktempdir() do dir
            h5 = _run_fixture(dir, "regression_solovev_kinetic_example")
            @test _drew(A.ForceFreeStates.plot_cond_fbar(h5))
        end
    end
end
