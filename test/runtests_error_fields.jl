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

    @testset "locking risk convolution" begin
        # A flat |δ| distribution on [0, 2δ₀] against thresholds spread across it. Both halves are
        # constructed, so the answer is arithmetic rather than whatever a fixture happens to produce.
        edges = collect(range(0, 2.0e-4; length=201))
        pdf = fill(1 / (edges[end] - edges[1]), length(edges) - 1)
        mc = EF.MonteCarloResult(edges, pdf, pdf, reshape(pdf, :, 1), reshape(pdf, :, 1),
            1.0e-4, 2.0e-4, 1.0e-4, 1.0e-4, 0.0, 1000, 1, 1)
        sc = EF.threshold_scaling(; n=1)
        scen = EF.ScenarioParameters(; n_e=5.0, b_t0=2.0, r_0=1.7, beta_n=1.0, l_i=1.0)

        # Every threshold above the distribution: nothing ever locks.
        @test EF.locking_risk(mc, fill(1.0, 100), sc, scen).plock == 0
        # Every threshold below it: everything locks, to the width of the bin the zero edge pins.
        @test EF.locking_risk(mc, fill(1.0e-12, 100), sc, scen).plock ≈ 100 atol = 0.5
        # Thresholds spread uniformly across the distribution: a flat δ against a flat threshold
        # gives half, since P(δ > threshold) = 1/2 for two independent uniforms on the same range.
        straddling = EF.locking_risk(mc, collect(range(0, 2.0e-4; length=20_001)), sc, scen)
        @test 0 < straddling.plock < 100
        @test straddling.plock ≈ 50 atol = 1.0
        # Shifting the thresholds down can only raise the risk.
        lower = EF.locking_risk(mc, collect(range(0, 1.0e-4; length=20_001)), sc, scen)
        @test lower.plock > straddling.plock
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
            inputs["ErrorFields"] = Dict{String,Any}("verbose" => false, "tolerance_file" => "tolerances.toml",
                "MonteCarlo" => Dict{String,Any}("nsample" => 20_000, "nbatch" => 2, "seed" => 5, "nbins" => 100),
                "scenario" => Dict{String,Any}("n_e" => 12.0),
                "Risk" => Dict{String,Any}("nsample_threshold" => 20_000, "seed" => 3, "scan_scales" => [0.5, 1.0, 2.0]))
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
            @test table.delta_per_mm_shift[2] ≈ 1e-3 * abs(table.shift[1, 2])  # axisymmetric: |S_y| = |S_x|
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
            # Two 2 kA hoops on a toy equilibrium drive an overlap around 1e-5, two orders below the
            # ITPA threshold at any plausible density, so zero risk is the right answer here and the
            # scan is flat. The convolution itself is pinned by its own testset above, on a
            # distribution built to straddle a threshold.
            @test risk.plock == 0
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

            # The one-call overlap path. Built in memory from the same coupling the table used, so
            # the comparison is exact rather than up to an SVD phase.
            ctx = EF.ResonantDriveContext(ffs.equil, rc, cfg, ffs.psilim, sens.b_t0)
            @test ctx.psilim == ffs.psilim && ctx.b_t0 == sens.b_t0
            @test length(ctx.grids) == length(unique(rc.n_modes))
            ovs = EF.coil_overlaps(ctx, sets)
            @test [o.coil_name for o in ovs] == sens.coil_names

            m_low, m_high = extrema(rc.m_modes)
            for (j, o) in enumerate(ovs)
                # The chain assembled by hand: a change on either side now has to be deliberate.
                hand_modes = FT.ForcingMode[]
                for (n, g) in ctx.grids
                    append!(hand_modes, FT.coil_forcing_modes(sets[j], g, n, m_low, m_high))
                end
                hand = PE.rootarea_field(rc, hand_modes)
                @test o.spectrum ≈ hand rtol = 1e-12
                @test o.raw ≈ PE.coupling_overlap(dom, hand)[1] rtol = 1e-12
                @test o.delta ≈ o.raw / o.b_t0
                @test o.spectrum_norm ≈ norm(hand)
                @test o.fraction_percent ≈ 100 * abs(o.raw) / norm(hand)
                # And it is the same number the sensitivity sweep gets for its nominal tap.
                @test o.delta ≈ table.delta_nominal[j] rtol = 1e-10
            end

            # Combining these sets at unit weight is the run's own forcing, which the run wrote out.
            whole = EF.combine_overlaps(ovs, (o.coil_name => 1.0 for o in ovs)...; name="assembly")
            @test whole.coil_name == "assembly"
            @test whole.spectrum ≈ pe.forcing_b_rootarea rtol = 1e-10
            @test whole.delta ≈ pe.dominant_forcing_overlap[1] / sens.b_t0 rtol = 1e-10
            @test whole.fraction_percent ≈ 100 * abs(whole.raw) / whole.spectrum_norm

            # Weights scale the spectrum, and an unmatched name is an error, not a silent zero.
            doubled = EF.combine_overlaps(ovs, ovs[1].coil_name => 2.0)
            @test doubled.raw ≈ 2 * ovs[1].raw rtol = 1e-12
            @test doubled.spectrum_norm ≈ 2 * ovs[1].spectrum_norm rtol = 1e-12
            @test_throws ArgumentError EF.combine_overlaps(ovs, "no_such_coil" => 1.0)
            @test_throws ArgumentError EF.combine_overlaps(ovs)
            @test_throws ArgumentError EF.coil_overlaps(ctx, sets; mode=length(ctx.dom.singular_values) + 1)

            # Rebuilt from the file instead: same magnitudes, up to the singular vectors' free phase.
            from_h5 = EF.coil_overlaps(h5path, sets)
            @test abs.(getfield.(from_h5, :delta)) ≈ abs.(getfield.(ovs, :delta)) rtol = 1e-8
            @test getfield.(from_h5, :fraction_percent) ≈ getfield.(ovs, :fraction_percent) rtol = 1e-8

            # The grid override, and the warning when a deck is coarser than the converged default.
            @test EF.MIN_NZETA_PER_PERIOD == FT.NZETA_POINTS_PER_PERIOD
            coarse_cfg = EF.regrid(cfg; nzeta_coil=8)
            @test coarse_cfg.nzeta_coil == 8 && coarse_cfg.mtheta_coil == cfg.mtheta_coil
            @test EF.regrid(cfg).nzeta_coil == cfg.nzeta_coil
            @test_logs (:warn, r"points per period") match_mode = :any EF.ResonantDriveContext(ffs.equil, rc, coarse_cfg, ffs.psilim, sens.b_t0)

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
