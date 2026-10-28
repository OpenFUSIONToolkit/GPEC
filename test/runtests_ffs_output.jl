using TOML, HDF5

@testset "FFS explicit output" begin
    GPEC = GeneralizedPerturbedEquilibrium
    mktempdir() do dir
        source = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
        for file in readdir(source)
            cp(joinpath(source, file), joinpath(dir, file))
        end
        deck = TOML.parsefile(joinpath(dir, "gpec.toml"))
        for section in ("LocalStability", "PerturbedEquilibrium", "ForcingTerms", "KineticForces", "SLAYER")
            pop!(deck, section, nothing)
        end
        deck["ForceFreeStates"]["write_outputs_to_HDF5"] = true
        open(io -> TOML.print(io, deck), joinpath(dir, "gpec.toml"), "w")
        reference = GPEC.main([dir]).ffs
        inputs, config, analytic = GPEC.build_inputs_from_toml(dir)
        eq = PlasmaEquilibrium(config, analytic)
        ctrl = GPEC.ForceFreeStates.control_from_deck(inputs["ForceFreeStates"])
        kwargs = GPEC.ForceFreeStates.problem_keywords(ctrl)
        mktempdir() do api_dir
            prob = ForceBalanceProblem(eq; nn=1, wall=GPEC.Vacuum.parse_wall(get(inputs, "Wall", nothing)), dir_path=api_dir, kwargs...)
            alg = ctrl.algorithm
            result = solve(prob, alg)
            @test result.problem === prob
            @test result.algorithm === alg
            @test !isfile(joinpath(api_dir, result.control.HDF5_filename))
            path = joinpath(api_dir, "saved.h5")
            save_to_hdf5(path, result)
            h5open(path, "r") do saved
                h5open(joinpath(dir, reference.control.HDF5_filename), "r") do expected
                    @test read(saved["ForceFreeStates/FreeBoundaryStability/eigenmode_energies"]) == read(expected["ForceFreeStates/FreeBoundaryStability/eigenmode_energies"])
                    @test TOML.parse(read(saved["Input/gpec_toml_raw"])) == TOML.parse(read(expected["Input/gpec_toml_raw"]))
                end
            end
            h5open(joinpath(api_dir, "group.h5"), "w") do file
                file["unrelated"] = 42
                group = create_group(file, "run")
                save_to_hdf5(group, result)
                @test isopen(file)
                @test read(file["unrelated"]) == 42
                @test haskey(group, "Input/gpec_toml_raw")
                close(group)
            end
        end
    end
end
