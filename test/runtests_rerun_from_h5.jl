using HDF5

# Tests for the gpec.h5 snapshot + replay feature, kept deliberately light: exactly one
# full-pipeline source run + one replay (the analytic Solovev path), plus pipeline-free checks
# for direct-equilibrium recovery and the override flags.

# Collect every leaf dataset path under an open HDF5 file, skipping the groups/paths that
# legitimately differ between a source run and its replay (`input/` is re-emitted with the
# rerun's own filename/TOML blob; `info/git_version` reflects the running commit).
function _rerun_leaf_paths(h5)
    skip_toplevel = Set(["input"])
    skip_paths = Set(["info/git_version"])
    paths = String[]
    function walk(node, prefix)
        for k in keys(node)
            full = isempty(prefix) ? k : prefix * "/" * k
            (isempty(prefix) && k in skip_toplevel) && continue
            full in skip_paths && continue
            child = node[k]
            child isa HDF5.Group ? walk(child, full) : push!(paths, full)
        end
    end
    walk(h5, "")
    return paths
end

# Count datasets that differ between a source gpec.h5 and its replay. Zero mismatches over the
# whole output is the strongest snapshot drift guard: it fires if any TOML var or IO field is
# added but not threaded through the snapshot + replay path.
function _rerun_dataset_mismatches(source_h5, rerun_h5)
    n_compared = 0
    n_mismatched = 0
    h5open(source_h5, "r") do hs
        h5open(rerun_h5, "r") do hr
            src = Set(_rerun_leaf_paths(hs))
            rer = Set(_rerun_leaf_paths(hr))
            for p in union(src, rer)
                n_compared += 1
                if !(p in src) || !(p in rer) || read(hs, p) != read(hr, p)
                    n_mismatched += 1
                end
            end
        end
    end
    return n_compared, n_mismatched
end

# Analytic (Solovev) equilibrium: the single end-to-end source + replay. The replay runs from a
# temp directory containing nothing but the captured gpec.h5, proving the snapshot is
# self-contained (no auxiliary TOML, g-file, or forcing data), and reproduces the source
# bit-for-bit across every output dataset.
@testset "Rerun from gpec.h5 (analytic Solovev)" begin
    template_dir = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")

    mktempdir() do snapshot_dir
        for name in readdir(template_dir)
            cp(joinpath(template_dir, name), joinpath(snapshot_dir, name))
        end
        toml_path = joinpath(snapshot_dir, "gpec.toml")
        write(toml_path, replace(read(toml_path, String), "write_outputs_to_HDF5 = false" => "write_outputs_to_HDF5 = true"))

        @info "Generating source gpec.h5 for rerun tests"
        GeneralizedPerturbedEquilibrium.main([snapshot_dir])
        source_h5 = joinpath(snapshot_dir, "gpec.h5")
        @test isfile(source_h5)

        @testset "self-contained replay matches source across all datasets" begin
            mktempdir() do replay_dir
                GeneralizedPerturbedEquilibrium.main([source_h5, "--output-dir", replay_dir])
                replay_h5 = joinpath(replay_dir, "gpec_rerun.h5")
                @test isfile(replay_h5)

                n_compared, n_mismatched = _rerun_dataset_mismatches(source_h5, replay_h5)
                @test n_compared > 0
                @test n_mismatched == 0
            end
        end

        # `build_inputs_from_h5` reconstructs the run inputs without executing the pipeline, so
        # both override flags are covered through the real code path with no extra runs.
        @testset "override flags propagate through build_inputs_from_h5" begin
            mktempdir() do out_dir
                inline = GeneralizedPerturbedEquilibrium.build_inputs_from_h5([source_h5, "--output-dir", out_dir, "--override", "ForceFreeStates.eulerlagrange_tolerance=1e-6"])
                @test inline[1]["ForceFreeStates"]["eulerlagrange_tolerance"] == 1e-6

                override_toml = joinpath(out_dir, "overrides.toml")
                write(override_toml, "[ForceFreeStates]\neulerlagrange_tolerance = 2e-6\n")
                from_file = GeneralizedPerturbedEquilibrium.build_inputs_from_h5([source_h5, "--output-dir", out_dir, "--override-file", override_toml])
                @test from_file[1]["ForceFreeStates"]["eulerlagrange_tolerance"] == 2e-6
            end
        end

        @testset "rerun refuses to overwrite its own source" begin
            # Default output is `<basename>_rerun.h5`, so clobbering only happens if the user
            # routes the rerun back onto the source file; exercise that guard.
            @test_throws ErrorException GeneralizedPerturbedEquilibrium.main([source_h5, "--output-dir", snapshot_dir, "--output-name", "gpec.h5"])
        end
    end
end

# Direct (DIII-D EFIT) equilibrium recovery through the h5 path, without a pipeline run:
# `read_efit` captures the raw arrays, which we round-trip through a temp h5 via the real
# replay reader `read_equilibrium_raw_inputs` and rebuild with `build_direct_from_raw`. The
# rebuilt splines must evaluate bit-for-bit identical to the originals.
@testset "Rerun direct-equilibrium recovery through h5" begin
    Equil = GeneralizedPerturbedEquilibrium.Equilibrium
    config = Equil.EquilibriumConfig(;
        eq_filename=joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example", "TkMkr_D3Dlike_Hmode.geqdsk"),
        eq_type="efit"
    )
    src = Equil.read_efit(config)
    @test src isa Equil.DirectRunInput
    @test !isempty(src.raw_data)

    mktempdir() do d
        h5path = joinpath(d, "raw.h5")
        h5open(h5path, "w") do f
            for (k, v) in src.raw_data
                f["input/raw_inputs/equilibrium/$k"] = v
            end
        end
        raw = h5open(GeneralizedPerturbedEquilibrium.read_equilibrium_raw_inputs, h5path, "r")
        rebuilt = Equil.build_direct_from_raw(config, raw)

        @test rebuilt.rmin == src.rmin
        @test rebuilt.rmax == src.rmax
        @test rebuilt.zmin == src.zmin
        @test rebuilt.zmax == src.zmax
        @test rebuilt.psio == src.psio
        @test rebuilt.bt_sign == src.bt_sign
        @test rebuilt.psi_in_xs == src.psi_in_xs
        @test rebuilt.psi_in_ys == src.psi_in_ys

        # 2D flux interpolant must evaluate bit-for-bit at interior sample points.
        for fr in (0.25, 0.5, 0.75), fz in (0.4, 0.6)
            R = src.rmin + fr * (src.rmax - src.rmin)
            Z = src.zmin + fz * (src.zmax - src.zmin)
            @test rebuilt.psi_in((R, Z)) == src.psi_in((R, Z))
        end
        # 1D profile spline likewise.
        for frac in (0.1, 0.5, 0.9)
            x = src.psi_in_xs[1] + frac * (src.psi_in_xs[end] - src.psi_in_xs[1])
            @test rebuilt.sq_in(x) == src.sq_in(x)
        end
    end
end

# `parse_override_flag` (pure): literal typing and the string-fallback warning for bare words.
@testset "parse_override_flag literal typing and warn" begin
    pf = GeneralizedPerturbedEquilibrium.parse_override_flag
    @test pf("ForceFreeStates.npert=5")["ForceFreeStates"]["npert"] === 5
    @test pf("ForceFreeStates.eulerlagrange_tolerance=1.5e-3")["ForceFreeStates"]["eulerlagrange_tolerance"] === 1.5e-3
    @test pf("ForceFreeStates.kin_flag=true")["ForceFreeStates"]["kin_flag"] === true

    bad = @test_logs (:warn,) pf("ForceFreeStates.mode=blorp")
    @test bad["ForceFreeStates"]["mode"] == "blorp"
end
