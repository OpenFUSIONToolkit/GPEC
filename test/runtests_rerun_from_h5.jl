using HDF5
using TOML

# Tests for the gpec.h5 snapshot + replay feature, kept deliberately light: one full-pipeline
# source run + replay (the analytic Solovev path), plus pipeline-free ingest round-trip checks
# for the direct (EFIT) and inverse (CHEASE) equilibrium kinds and the override flags.

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
                # `isequal` (not `!=`) so a faithfully-reproduced NaN counts as a match — some
                # datasets (e.g. ballooning `locstab/alpha_critical` where no boundary exists) are
                # legitimately all-NaN, and `NaN != NaN` would otherwise flag them as drift.
                if !(p in src) || !(p in rer) || !isequal(read(hs, p), read(hr, p))
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

        @testset "--coil-source coils errors when source has no coil snapshot" begin
            # This Solovev source run used ASCII forcing, so there is no
            # input/raw_inputs/coils group to replay from.
            mktempdir() do replay_dir
                @test_throws ErrorException GeneralizedPerturbedEquilibrium.main([
                    source_h5, "--output-dir", replay_dir, "--coil-source", "coils"
                ])
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Coil-geometry snapshot: a coil run stores its geometry in gpec.h5 so it can be
# replayed without the original .dat. The decisive test deletes the .dat between
# the source run and the replay: the default (recompute-from-TOML) path then
# fails, while `--coil-source coils` succeeds from the stored geometry.
@testset "Rerun from gpec.h5: coil geometry snapshot" begin
    template_dir = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example")
    src_dat = joinpath(@__DIR__, "..", "src", "ForcingTerms", "coil_geometries", "d3d_il.dat")

    read_resflux(path) =
        h5open(path, "r") do h5
            key = "perturbed_equilibrium/singular_coupling/resonant_flux"
            haskey(h5, key) ? read(h5, key) : ComplexF64[]
        end

    mktempdir() do snapshot_dir
        for name in readdir(template_dir)
            src = joinpath(template_dir, name)
            isfile(src) && cp(src, joinpath(snapshot_dir, name))
        end

        # Point the coil set at a throwaway .dat we can delete before replay.
        coil_dat = joinpath(snapshot_dir, "my_coils.dat")
        cp(src_dat, coil_dat)

        toml_path = joinpath(snapshot_dir, "gpec.toml")
        inputs = TOML.parsefile(toml_path)
        inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = true
        inputs["ForcingTerms"] = Dict{String,Any}(
            "forcing_data_format" => "coil",
            "coil_set" => [Dict{String,Any}(
                "dat_file" => coil_dat,
                "currents" => fill(1.0e3, 6)
            )]
        )
        open(toml_path, "w") do io
            TOML.print(io, inputs)
        end

        @info "Generating source gpec.h5 for coil rerun test"
        GeneralizedPerturbedEquilibrium.main([snapshot_dir])
        source_h5 = joinpath(snapshot_dir, "gpec.h5")
        @test isfile(source_h5)

        # The coil geometry actually used must be captured in the snapshot.
        h5open(source_h5, "r") do h5
            @test haskey(h5, "input/raw_inputs/coils")
            @test haskey(h5, "input/raw_inputs/coils/my_coils")
        end
        src_flux = read_resflux(source_h5)
        @test !isempty(src_flux)
        @test maximum(abs.(src_flux)) > 0

        # Delete the .dat: the snapshot must now be the only source of geometry.
        rm(coil_dat)

        @testset "default replay needs the .dat and fails once it is gone" begin
            mktempdir() do replay_dir
                @test_throws Exception GeneralizedPerturbedEquilibrium.main([
                    source_h5, "--output-dir", replay_dir
                ])
            end
        end

        @testset "--coil-source coils replays from the stored geometry" begin
            mktempdir() do replay_dir
                GeneralizedPerturbedEquilibrium.main([
                    source_h5, "--output-dir", replay_dir, "--coil-source", "coils"
                ])
                replay_h5 = joinpath(replay_dir, "gpec_rerun.h5")
                @test isfile(replay_h5)
                @test read_resflux(replay_h5) ≈ src_flux rtol = 1e-6
            end
        end
    end
end

# Round-trip the typed equilibrium ingest through a temp h5 exactly as the snapshot writer and
# replay reader do (field-by-field write + `read_equilibrium_ingest`), then rebuild the solver
# input. The rebuilt splines must evaluate bit-for-bit identical to the originals. This is the
# pipeline-free guard on the capture → store → restore path, run once per equilibrium kind.
function _roundtrip_ingest(ingest, kind)
    mktempdir() do d
        h5path = joinpath(d, "raw.h5")
        h5open(h5path, "w") do f
            f["input/raw_inputs/equilibrium/ingest_kind"] = kind
            for nm in fieldnames(typeof(ingest))
                f["input/raw_inputs/equilibrium/$nm"] = getfield(ingest, nm)
            end
        end
        restored = h5open(GeneralizedPerturbedEquilibrium.read_equilibrium_ingest, h5path, "r")
        @test restored isa typeof(ingest)
        return restored
    end
end

# Direct (DIII-D EFIT) equilibrium recovery through the h5 path, without a pipeline run.
@testset "Rerun direct-equilibrium recovery through h5" begin
    Equil = GeneralizedPerturbedEquilibrium.Equilibrium
    config = Equil.EquilibriumConfig(;
        eq_filename=joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example", "TkMkr_D3Dlike_Hmode.geqdsk"),
        eq_type="efit"
    )
    src = Equil.read_efit(config)
    @test src isa Equil.DirectRunInput
    @test src.ingest isa Equil.DirectIngest

    restored = _roundtrip_ingest(src.ingest, "direct")
    rebuilt = Equil.build_direct_from_ingest(config, restored)

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

# Inverse (CHEASE) equilibrium recovery through the same h5 path — exercises the InverseIngest
# branch and `build_inverse_from_ingest`, the kind that was not rerunnable before this change.
@testset "Rerun inverse-equilibrium recovery through h5" begin
    Equil = GeneralizedPerturbedEquilibrium.Equilibrium
    config = Equil.EquilibriumConfig(;
        eq_filename=joinpath(@__DIR__, "test_data", "CHEASE_test_data", "INP1_ascii"),
        eq_type="chease_ascii", jac_type="boozer", grid_type="ldp",
        psilow=0.01, psihigh=0.994, r0exp=6.8, b0exp=7.4
    )
    src = Equil.read_chease_ascii(config)
    @test src isa Equil.InverseRunInput
    @test src.ingest isa Equil.InverseIngest

    restored = _roundtrip_ingest(src.ingest, "inverse")
    rebuilt = Equil.build_inverse_from_ingest(config, restored)

    @test rebuilt.ro == src.ro
    @test rebuilt.zo == src.zo
    @test rebuilt.psio == src.psio
    @test rebuilt.rz_in_xs == src.rz_in_xs
    @test rebuilt.rz_in_ys == src.rz_in_ys

    # R, Z interpolants must evaluate bit-for-bit at interior (ψ, θ) sample points.
    for fpsi in (0.25, 0.5, 0.75), fth in (0.2, 0.6)
        ψ = src.rz_in_xs[1] + fpsi * (src.rz_in_xs[end] - src.rz_in_xs[1])
        θ = src.rz_in_ys[1] + fth * (src.rz_in_ys[end] - src.rz_in_ys[1])
        @test rebuilt.rz_in_R((ψ, θ)) == src.rz_in_R((ψ, θ))
        @test rebuilt.rz_in_Z((ψ, θ)) == src.rz_in_Z((ψ, θ))
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
