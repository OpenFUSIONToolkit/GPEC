using TOML, HDF5

@testset "FFS explicit output" begin
    GPEC = GeneralizedPerturbedEquilibrium
    mktempdir() do dir
        source = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
        for file in readdir(source)
            cp(joinpath(source, file), joinpath(dir, file))
        end
        deck = TOML.parsefile(joinpath(dir, "gpec.toml"))
        for section in ("LocalStability", "Ballooning", "PerturbedEquilibrium", "ForcingTerms", "KineticForces", "SLAYER")
            pop!(deck, section, nothing)
        end
        open(io -> TOML.print(io, deck), joinpath(dir, "gpec.toml"), "w")
        reference = GPEC.main([dir]).ffs
        inputs, config, analytic = GPEC.build_inputs_from_toml(dir)
        eq = PlasmaEquilibrium(config, analytic)
        kwargs = GPEC.force_balance_keywords(inputs, dir)
        mktempdir() do api_dir
            prob = ForceBalanceProblem(eq; kwargs...)
            alg = GPEC.force_balance_algorithm(inputs)
            result = solve(prob, alg)
            @test result.problem === prob
            @test result.algorithm === alg
            @test !isfile(joinpath(api_dir, "gpec.h5"))
            path = joinpath(api_dir, "saved.h5")
            save_to_hdf5(path, result)
            h5open(path, "r") do saved
                h5open(joinpath(dir, "gpec.h5"), "r") do expected
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
