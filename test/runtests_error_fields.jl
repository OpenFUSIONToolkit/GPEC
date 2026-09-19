using HDF5
using TOML
using LinearAlgebra
using Plots

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
            cp(joinpath(@__DIR__, "test_data", "ErrorFields", "tolerances_two_hoops.toml"), joinpath(dir, "tolerances.toml"))
            # Kinetic profiles consistent with this Solovev pressure, for the NTV torque couplings.
            cp(joinpath(@__DIR__, "..", "examples", "Solovev_kinetic_NTV_example", "kinetic.dat"), joinpath(dir, "kinetic.dat"))
            inputs["KineticForces"] = Dict{String,Any}("kinetic_file" => "kinetic.dat", "verbose" => false, "write_outputs_to_HDF5" => true)
            inputs["ErrorFields"] = Dict{String,Any}("verbose" => false, "tolerance_file" => "tolerances.toml",
                "MonteCarlo" => Dict{String,Any}("nsample" => 20_000, "nbatch" => 2, "seed" => 5, "nbins" => 100),
                # Density chosen so the ITPA threshold sits near the fixture's nominal overlap: risk neither 0 nor saturated.
                "scenario" => Dict{String,Any}("n_e" => 12.0),
                "Risk" => Dict{String,Any}("nsample_threshold" => 20_000, "seed" => 3, "scan_scales" => [0.5, 1.0, 2.0]),
                "NTV" => Dict{String,Any}("efc_coils" => ["hoop_tilted"]))
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

            # Axisymmetric hoop at n = 1: no nominal drive, nothing from a vertical shift or a rotation
            # about the machine axis, and S_y = −i·S_x, with the tilt pair in the opposite sense.
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
            @test_throws DimensionMismatch EF.sensitivity_table(sens, PE.dominant_coupling(rc.C[:, 1:(end-1)], rc.rational_psi))
            windowed = EF.sensitivity_table(sens, PE.dominant_coupling(rc; psi_low=rc.rational_psi[end]))
            @test length(windowed.delta_nominal) == 2

            # HDF5: self-describing, and the reader and file-based table reproduce memory exactly.
            h5open(h5path, "r") do f
                @test haskey(f, "ErrorFields/CoilSensitivities/DominantMode/delta_nominal")
                @test isempty(_collect_metadata_violations(f))
                @test haskey(f, "Input/RawInputs/ErrorFields/tolerance_toml_raw")
            end
            # The tolerance file was validated against these coil sets and echoed verbatim.
            snapshot = EF.read_tolerance_snapshot(h5path)
            @test snapshot isa EF.ToleranceSet
            @test snapshot.raw == read(joinpath(dir, "tolerances.toml"), String)
            @test [c.name for c in snapshot.coils] == ["hoop_tilted", "hoop_axi"]

            # The run's Monte Carlo (full window, dominant mode) is written and re-runnable from the file.
            mc = res.monte_carlo
            @test mc isa EF.MonteCarloResult
            @test mc.nsample == 20_000 && mc.nbatch == 2 && mc.seed == 5
            @test mc.delta_nominal ≈ abs(sum(table.delta_nominal))
            @test sum(mc.pdf .* diff(mc.bin_edges)) ≈ 1 atol = 1e-6
            @test mc.mean_abs_delta_efc < mc.mean_abs_delta
            mc_file = EF.MonteCarloResult(h5path)
            @test mc_file.pdf == mc.pdf && mc_file.bin_edges == mc.bin_edges && mc_file.nsample == 20_000
            rerun = EF.run_monte_carlo(h5path; nsample=20_000, nbatch=2, seed=5, nbins=100)
            @test rerun.pdf == mc.pdf
            windowed_mc = EF.run_monte_carlo(h5path; psi_low=rc.rational_psi[end], nsample=5_000, nbatch=1, seed=5, nbins=50)
            @test windowed_mc.delta_nominal ≈ abs(sum(windowed.delta_nominal))

            # Locking risk and tolerance scan: written, bounded, and reproducible from the file.
            risk = res.locking_risk
            @test risk isa EF.RiskResult
            @test risk.scaling.n == 1 && risk.scaling.dataset == "O,L" && risk.scaling.fit == "WLS"
            @test risk.threshold_nominal == EF.nominal_threshold(risk.scaling, EF.ScenarioParameters(ffs.equil; n_e=12.0))
            @test 0 <= risk.plock_efc <= risk.plock <= 100
            @test length(risk.plock_batches) == 2
            h5open(h5path, "r") do f
                @test haskey(f, "ErrorFields/Risk/plock_percent") && haskey(f, "ErrorFields/Risk/ToleranceScan/scale")
                @test read(f["ErrorFields/Risk/plock_percent"]) == risk.plock
                @test isempty(_collect_metadata_violations(f))
            end
            scan = EF.ToleranceScan(h5path)
            @test scan.scale == [0.5, 1.0, 2.0]
            @test all(diff(scan.plock) .>= -0.5)                          # risk grows with tolerance (to Monte Carlo noise)
            @test 0 < risk.plock < 100                                     # neither empty nor saturated at this density
            @test scan.plock[2] ≈ risk.plock rtol = 1e-12                # the scale-1 point is the run's own Monte Carlo
            again = EF.locking_risk(h5path; n_e=12.0, nsample=20_000, nbatch=2, seed=5, nbins=100,
                risk_ctrl=EF.RiskControl(; nsample_threshold=20_000, seed=3))
            @test again.plock == risk.plock
            # The scan's own file entry point at unit scale is that same re-run.
            rescan = EF.tolerance_scan(h5path; scales=[1.0], n_e=12.0, nsample=20_000, nbatch=2, seed=5, nbins=100,
                risk_ctrl=EF.RiskControl(; nsample_threshold=20_000, seed=3))
            @test rescan.scale == [1.0]
            @test rescan.plock[1] == again.plock
            @test rescan.plock_efc[1] == again.plock_efc

            # Analysis plots: every ErrorFields plot renders from the file and saves; the phasing map
            # of the two hoops is the closed form on their stored spectra.
            AEF = GPEC.Analysis.ErrorFields
            for (name, fn) in (("sens", AEF.plot_coil_sensitivities), ("pdf", AEF.plot_tolerance_pdf), ("risk", AEF.plot_locking_risk),
                ("thr", AEF.plot_threshold_scaling), ("mode", AEF.plot_dominant_mode_spectrum))
                png = joinpath(dir, "plot_$name.png")
                p = fn(h5path; save_path=png)
                @test p isa Plots.Plot
                @test isfile(png)
                name == "mode" && @test all(ser -> ser[:seriestype] === :steppre, p.series_list)
            end
            @test AEF.plot_coil_sensitivities(["run" => h5path, "again" => h5path]; quantity=:tilt)[1][:yaxis][:guide] == "|∂δ/∂θ| per 0.1°"
            @test_throws ArgumentError AEF.plot_coil_sensitivities(h5path; quantity=:bogus)
            # The target risk adds its own line and the allowable-tolerance markers.
            @test length(AEF.plot_locking_risk(h5path; target_percent=1.0).series_list) > length(AEF.plot_locking_risk(h5path).series_list)
            @test length(AEF.plot_error_field_summary(h5path; save_path=joinpath(dir, "summary.png")).subplots) == 4
            pmap = EF.phasing_map(h5path, ["hoop_tilted", "hoop_axi"]; nphase=36)
            @test length(pmap.phase_deg) == 1 && size(pmap.delta_per_kat) == (36,)
            kat = sens.winding_multiplier .* sens.peak_current ./ 1e3
            δ_each = [dot(dom.right_singular_vectors[:, 1], sens.nominal_field[:, j]) / kat[j] / sens.b_t0 for j in 1:2]
            @test pmap.delta_per_kat ≈ abs.(δ_each[1] .+ δ_each[2] .* cis.(deg2rad.(pmap.phase_deg[1])))
            @test AEF.plot_phasing_map(pmap; save_path=joinpath(dir, "phasing.png")) isa Plots.Plot
            @test AEF.plot_phasing_map(h5path, ["hoop_tilted", "hoop_axi"]; nphase=12) isa Plots.Plot
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

            # Correction-coil couplings: the tilted hoop as the correction array. Its overlap per kAt is
            # the table's nominal overlap over its ampere-turns; the residual field carries no dominant
            # mode; the torques are finite and written with the metadata contract.
            couplings = res.efc_couplings
            @test couplings isa Vector{EF.EFCCoupling} && length(couplings) == 1
            c = couplings[1]
            kat = sets[1].nw * 2.0e3 / 1e3
            @test c.coil_name == "hoop_tilted"
            @test c.delta_per_kat ≈ abs(table.delta_nominal[1]) / kat rtol = 1e-6
            @test 0 < c.overlap_percent <= 100
            @test isfinite(c.torque_full_per_kat2) && isfinite(c.torque_residual_per_kat2)
            @test abs(dot(dom.right_singular_vectors[:, 1], EF.residual_spectrum(dom, sens.nominal_field[:, 1]))) < 1e-12
            @test EF.read_efc_couplings(h5path)[1] == c
            h5open(h5path, "r") do f
                @test haskey(f, "ErrorFields/NTV/torque_residual_per_kat2")
                @test isempty(_collect_metadata_violations(f))
            end
            curve = EF.efc_current_curve(c; delta_threshold=risk.threshold_nominal, torque_budget=1.0)
            @test length(curve.delta_ef) == 500 && all(curve.current_linear .>= 0)
            ntv_plot = GPEC.Analysis.ErrorFields.plot_efc_ntv_limits(h5path; torque_budget=1.0, save_path=joinpath(dir, "ntv.png"))
            @test length(ntv_plot.series_list) >= 2                        # the single-mode and NTV-limited currents
            @test_throws ErrorException GPEC.efc_couplings(ffs, sets, rc, dom, cfg, GPEC.KineticForces.KineticForcesControl(), nothing)

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
            @test_throws ArgumentError EF.compute_coil_sensitivities(sets, rc, ffs.equil, cfg; psi=ffs.psilim, b_t0=0.0)
            @test_throws ArgumentError EF.compute_coil_sensitivities(sets, rc, ffs.equil, cfg,
                EF.ErrorFieldsControl(; rotation_center="pack"); psi=ffs.psilim, b_t0=sens.b_t0)
            @test_throws ArgumentError EF.compute_coil_sensitivities(sets, rc, ffs.equil, cfg,
                EF.ErrorFieldsControl(; fd_step_shift_m=0.0); psi=ffs.psilim, b_t0=sens.b_t0)
        end
    end
end
