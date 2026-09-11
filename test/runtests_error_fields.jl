using HDF5
using TOML
using LinearAlgebra

# The ErrorFields coil linearization: a closed-form check on the cancelling offset, then one
# coil-forced Solovev run with an axisymmetric PF hoop (whose rigid-motion response is known
# analytically) and a tilted one, checked in memory, through the HDF5 writer/reader, and
# through the post-hoc gpec.h5 entry point that rebuilds the equilibrium from the file.
include("h5_metadata_check.jl")

@testset "error fields" begin
    GPEC = GeneralizedPerturbedEquilibrium
    PE = GPEC.PerturbedEquilibrium
    EF = GPEC.ErrorFields
    FT = GPEC.ForcingTerms

    @testset "cancelling offset" begin
        # General complex pair: the offset cancels the nominal overlap to round-off.
        δ, Sx, Sy = 0.3 - 0.7im, 1.0 + 2.0im, -0.5 + 0.25im
        Δx, Δy = EF.cancelling_offset(δ, Sx, Sy)
        @test abs(δ + Sx * Δx + Sy * Δy) < 1e-14
        # Axisymmetric pair S_y = i·S_x: the offset is the complex number −δ/S_x.
        Δx, Δy = EF.cancelling_offset(δ, Sx, im * Sx)
        @test Δx + im * Δy ≈ -δ / Sx
        # Degenerate pair (real multiples): minimum-norm least squares, finite.
        Δx, Δy = EF.cancelling_offset(1.0 + 0.0im, 1.0 + 0.0im, 2.0 + 0.0im)
        @test isfinite(Δx) && isfinite(Δy)
        @test 1.0 + Δx + 2Δy ≈ 0 atol = 1e-14
    end

    @testset "coil-forced Solovev run: identities, output, and post-hoc entry point" begin
        template = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")

        mktempdir() do dir
            for name in readdir(template)
                cp(joinpath(template, name), joinpath(dir, name))
            end
            toml_path = joinpath(dir, "gpec.toml")
            inputs = TOML.parsefile(toml_path)
            inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = true
            # Two PF hoops outside the plasma (Solovev R ≈ 0.67–1.33 m): one tilted so the run has
            # a nominal n=1 forcing, one axisymmetric so its rigid-motion response is known exactly.
            inputs["ForcingTerms"] = Dict{String,Any}(
                "forcing_data_format" => "coil", "mtheta_coil" => 240, "nzeta_coil" => 32,
                "coil_set" => [
                    Dict{String,Any}("name" => "hoop_tilted", "source" => "pf_hoop", "radius" => 1.5, "height" => 0.4, "currents" => [2.0e3], "tiltx" => [3.0]),
                    Dict{String,Any}("name" => "hoop_axi", "source" => "pf_hoop", "radius" => 1.5, "height" => -0.4, "currents" => [2.0e3])
                ])
            inputs["PerturbedEquilibrium"] = Dict{String,Any}(
                "compute_response" => true, "compute_singular_coupling" => true,
                "verbose" => false, "write_outputs_to_HDF5" => true)
            inputs["ErrorFields"] = Dict{String,Any}("verbose" => false)
            open(io -> TOML.print(io, inputs), toml_path, "w")

            res = GPEC.main([dir])
            sens, pe, ffs = res.coil_sensitivities, res.pe, res.ffs
            h5path = joinpath(dir, "gpec.h5")

            @test sens isa EF.CoilSensitivities
            @test sens.coil_names == ["hoop_tilted", "hoop_axi"]
            N = ffs.numpert_total
            @test size(sens.nominal_field) == (N, 2)
            @test size(sens.shift_sensitivity) == (N, 3, 2)
            @test size(sens.tilt_sensitivity) == (N, 3, 2)
            @test sens.b_t0 == ffs.equil.params.bt0
            @test sens.peak_current == [2.0e3, 2.0e3]
            @test all(<(1e-2), sens.shift_linearity_residual)
            @test all(<(1e-2), sens.tilt_linearity_residual)

            # These coils are the run's forcing: the nominal spectra sum to the run's own b̃.
            @test vec(sum(sens.nominal_field; dims=2)) ≈ pe.forcing_b_rootarea rtol = 1e-10

            # Axisymmetric hoop at n = 1: no nominal drive; a vertical shift or a rotation about
            # the machine axis changes nothing; a shift along y is the x shift rotated a quarter
            # turn toroidally, which multiplies the n = 1 mode by −i under the SFL toroidal angle
            # φ = −helicity·(2πζ + ν) (this fixture has helicity +1). The y tilt of
            # `apply_transforms` rotates x toward z, the left-handed sense about +y, so the tilt
            # pair carries the opposite sign.
            Sx, Sy, Sz = (sens.shift_sensitivity[:, a, 2] for a in 1:3)
            Tx, Ty, Tz = (sens.tilt_sensitivity[:, a, 2] for a in 1:3)
            @test norm(sens.nominal_field[:, 2]) < 1e-10 * norm(Sx)
            @test norm(Sz) < 1e-8 * norm(Sx)
            @test norm(Tz) < 1e-8 * norm(Tx)
            @test Sy ≈ -im .* Sx rtol = 1e-8
            @test Ty ≈ im .* Tx rtol = 1e-8

            # Projection onto the full-window dominant mode: the sets' overlaps sum to the run's.
            rc = PE.ResonantCoupling(pe, ffs)
            dom = PE.dominant_coupling(rc)
            table = EF.sensitivity_table(sens, dom)
            @test table.mode == 1
            @test sum(table.delta_nominal) ≈ pe.dominant_forcing_overlap[1] / sens.b_t0 rtol = 1e-10
            @test table.shift_rms[2] ≈ abs(table.shift[1, 2])  # axisymmetric: |S_y| = |S_x|
            for j in 1:2
                @test abs(table.delta_nominal[j] + table.shift[1, j] * table.cancelling_shift[1, j] + table.shift[2, j] * table.cancelling_shift[2, j]) < 1e-12
            end
            @test_throws ArgumentError EF.sensitivity_table(sens, dom; mode=length(dom.singular_values) + 1)
            windowed = EF.sensitivity_table(sens, PE.dominant_coupling(rc; psi_low=rc.rational_psi[end]))
            @test length(windowed.delta_nominal) == 2

            # HDF5: self-describing, and the reader and file-based table reproduce memory exactly.
            h5open(h5path, "r") do f
                @test haskey(f, "ErrorFields/CoilSensitivities/DominantMode/delta_nominal")
                @test isempty(_collect_metadata_violations(f))
            end
            from_file = EF.CoilSensitivities(h5path)
            @test from_file.coil_names == sens.coil_names
            @test from_file.m_modes == sens.m_modes && from_file.n_modes == sens.n_modes
            @test from_file.nominal_field == sens.nominal_field
            @test from_file.shift_sensitivity == sens.shift_sensitivity
            @test from_file.tilt_sensitivity == sens.tilt_sensitivity
            @test from_file.b_t0 == sens.b_t0
            table_file = EF.sensitivity_table(h5path)
            @test table_file.delta_nominal == table.delta_nominal
            @test table_file.shift == table.shift && table_file.tilt == table.tilt

            # Post-hoc entry point: equilibrium and coupling rebuilt from the file, same coils.
            cfg = FT.CoilConfig(GPEC.forcing_terms_control(inputs))
            sets = FT.load_coil_sets(cfg, 1; equil=ffs.equil)
            rebuilt = GPEC.equilibrium_from_h5(h5path)
            @test rebuilt.psilim == ffs.psilim
            @test rebuilt.equil.params.bt0 ≈ ffs.equil.params.bt0
            post_hoc = EF.compute_coil_sensitivities(h5path, sets)
            @test post_hoc.nominal_field ≈ sens.nominal_field rtol = 1e-10
            @test post_hoc.shift_sensitivity ≈ sens.shift_sensitivity rtol = 1e-10
            @test post_hoc.tilt_sensitivity ≈ sens.tilt_sensitivity rtol = 1e-10

            # Central differences: doubling the step moves the derivatives at O(h²).
            coarse = EF.compute_coil_sensitivities(sets, rc, ffs.equil, cfg,
                EF.ErrorFieldsControl(; fd_step_shift_m=2e-3, fd_step_tilt_deg=0.2); psi=ffs.psilim, b_t0=sens.b_t0)
            @test coarse.shift_sensitivity ≈ sens.shift_sensitivity rtol = 1e-4
            @test coarse.tilt_sensitivity ≈ sens.tilt_sensitivity rtol = 1e-3

            # Single-conductor sets: the set pivot is the conductor pivot.
            set_pivot = EF.compute_coil_sensitivities(sets, rc, ffs.equil, cfg,
                EF.ErrorFieldsControl(; rotation_center="set"); psi=ffs.psilim, b_t0=sens.b_t0)
            @test set_pivot.tilt_sensitivity ≈ sens.tilt_sensitivity rtol = 1e-10

            # Guards: a current-free set, a bad pivot name, a non-positive step.
            dead = FT.CoilSet(sets[1].name, sets[1].ncoil, sets[1].s, sets[1].nw, sets[1].nsec, sets[1].x, sets[1].y, sets[1].z, zeros(sets[1].ncoil))
            @test_throws ArgumentError EF.compute_coil_sensitivities([dead], rc, ffs.equil, cfg; psi=ffs.psilim, b_t0=sens.b_t0)
            @test_throws ArgumentError EF.compute_coil_sensitivities(sets, rc, ffs.equil, cfg,
                EF.ErrorFieldsControl(; rotation_center="pack"); psi=ffs.psilim, b_t0=sens.b_t0)
            @test_throws ArgumentError EF.compute_coil_sensitivities(sets, rc, ffs.equil, cfg,
                EF.ErrorFieldsControl(; fd_step_shift_m=0.0); psi=ffs.psilim, b_t0=sens.b_t0)
        end
    end
end
