using HDF5

# Exercise the gpec.h5 snapshot + replay pipeline (issue #171). The source run is
# a regular `main(dir)` invocation; the replay runs `main_from_h5` from a temp
# directory that contains nothing but the captured gpec.h5, proving that the
# snapshot is self-contained (no auxiliary TOML, no g-file, no forcing data).
@testset "Rerun from gpec.h5" begin
    template_dir = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")

    function read_scalars(path)
        return h5open(path, "r") do h5
            (
                et=read(h5, "vacuum/et"),
                ep=read(h5, "vacuum/ep"),
                ev=read(h5, "vacuum/ev"),
                mpert=read(h5, "info/mpert"),
                msing=read(h5, "singular/msing"),
                sing_psi=read(h5, "singular/psi"),
                sing_q=read(h5, "singular/q"),
            )
        end
    end

    # Copy the regression example into a temp directory and enable HDF5 output
    # so the rest of the test can work against a self-contained source snapshot.
    # We avoid editing the in-tree fixture (and its shared gpec.toml) directly.
    mktempdir() do snapshot_dir
        for name in readdir(template_dir)
            cp(joinpath(template_dir, name), joinpath(snapshot_dir, name))
        end
        toml_path = joinpath(snapshot_dir, "gpec.toml")
        toml_text = read(toml_path, String)
        toml_text = replace(toml_text, "write_outputs_to_HDF5 = false" => "write_outputs_to_HDF5 = true")
        write(toml_path, toml_text)

        @info "Generating source gpec.h5 for rerun tests"
        GeneralizedPerturbedEquilibrium.main([snapshot_dir])
        source_h5 = joinpath(snapshot_dir, "gpec.h5")
        @test isfile(source_h5)
        source_scalars = read_scalars(source_h5)

        @testset "self-contained replay matches source bit-for-bit" begin
            mktempdir() do replay_dir
                # No gpec.toml, no sol.toml, no g-file in replay_dir — this is
                # the hard constraint from issue #171.
                GeneralizedPerturbedEquilibrium.main([source_h5, "--output-dir", replay_dir])

                replay_h5 = joinpath(replay_dir, "gpec_rerun.h5")
                @test isfile(replay_h5)
                replay_scalars = read_scalars(replay_h5)

                @test replay_scalars.et == source_scalars.et
                @test replay_scalars.ep == source_scalars.ep
                @test replay_scalars.ev == source_scalars.ev
                @test replay_scalars.mpert == source_scalars.mpert
                @test replay_scalars.msing == source_scalars.msing
                @test replay_scalars.sing_psi == source_scalars.sing_psi
                @test replay_scalars.sing_q == source_scalars.sing_q
            end
        end

        @testset "override changes tolerance without breaking singular structure" begin
            mktempdir() do replay_dir
                GeneralizedPerturbedEquilibrium.main([
                    source_h5,
                    "--output-dir", replay_dir,
                    "--override", "ForceFreeStates.eulerlagrange_tolerance=1e-6",
                ])

                replay_h5 = joinpath(replay_dir, "gpec_rerun.h5")
                @test isfile(replay_h5)
                replay_scalars = read_scalars(replay_h5)

                # Loosening the tolerance should leave the rational-surface
                # structure intact and the real eigenvalue unchanged to high
                # precision, while the imaginary part (set by integrator
                # residuals) is allowed to drift.
                @test replay_scalars.mpert == source_scalars.mpert
                @test replay_scalars.msing == source_scalars.msing
                @test replay_scalars.sing_psi == source_scalars.sing_psi
                @test replay_scalars.sing_q == source_scalars.sing_q
                @test real(replay_scalars.et[1]) ≈ real(source_scalars.et[1]) rtol = 1e-4
            end
        end

        @testset "rerun refuses to overwrite its own source" begin
            # Default output name is `<basename>_rerun.h5`, so clobbering only
            # happens when the user tries to route the rerun back on top of the
            # source file. Exercise the guard by aiming the output at the
            # snapshot dir with the source filename.
            @test_throws ErrorException GeneralizedPerturbedEquilibrium.main([
                source_h5,
                "--output-dir", snapshot_dir,
                "--output-name", "gpec.h5",
            ])
        end
    end
end
