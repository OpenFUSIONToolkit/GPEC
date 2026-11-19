using CairoMakie
using HDF5
using TOML

include("error_fields_fixture.jl")

# Every ErrorFields figure against the coil-forced TJ run, from the assessment in memory, from its gpec.h5 and from the
# post-hoc results the plots also take.
@testset "ErrorFields plots" begin
    GPEC = GeneralizedPerturbedEquilibrium
    PE = GPEC.PerturbedEquilibrium
    EF = GPEC.ErrorFields
    FT = GPEC.ForcingTerms
    AEF = GPEC.Analysis.ErrorFields
    _axes(fig) = [b for b in fig.content if b isa Axis]
    _plots(ax, T) = [p for p in ax.scene.plots if p isa T]

    mktempdir() do dir
        res, h5path, _ = error_fields_fixture(dir)
        ef = res.error_fields
        sens, mc, risk = ef.sensitivities, ef.monte_carlo, ef.risk
        dom = PE.dominant_coupling(PE.ResonantCoupling(res.pe.response))
        table = EF.sensitivity_table(sens, dom)
        snapshot = EF.read_tolerance_snapshot(h5path)
        ctx = EF.ResonantDriveContext(h5path)
        sets = FT.load_coil_sets(ctx.cfg, 1; equil=ctx.equil)
        ovs = EF.coil_overlaps(ctx, sets)

        # Every registered plot draws from the assessment and from its file.
        for e in GPEC.Analysis.PLOTS
            GPEC.Analysis._accepts(e, ef) || continue
            @test e.fn(ef) isa Figure
            @test e.fn(h5path) isa Figure
        end
        png = joinpath(dir, "summary.png")
        @test AEF.plot_error_field_summary(h5path; save_path=png) isa Figure
        @test isfile(png)

        # The spectrum plot draws the stored dominant mode, padded to zero at both ends by the step convention.
        y = last.(_plots(only(_axes(AEF.plot_dominant_mode_spectrum(h5path))), Stairs)[1][1][])
        @test maximum(y) ≈ maximum(abs.(dom.right_singular_vectors[:, 1])) rtol = 1e-6
        @test y[1] == 0 && y[end] == 0

        # Coil sensitivities: the quantity names the axis, several runs share it, and an in-memory table draws what
        # the file does.
        @test only(_axes(AEF.plot_coil_sensitivities(["run" => h5path, "again" => h5path]; quantity=:tilt))).xlabel[] == "|δ| per degree of tilt"
        @test only(_axes(AEF.plot_coil_sensitivities(h5path; quantity=:rim))).xlabel[] == "|δ| per mm of rim displacement"
        dots(fig) = _plots(only(_axes(fig)), Scatter)[1][1][]
        @test dots(AEF.plot_coil_sensitivities(table)) == dots(AEF.plot_coil_sensitivities(h5path))
        @test AEF.plot_coil_sensitivities(["file" => h5path, "memory" => table]) isa Figure
        @test AEF.plot_coil_sensitivities(h5path; quantity=:fraction, yscale=:log10) isa Figure
        @test only(_axes(AEF.plot_coil_sensitivities(h5path; yscale=:log10))).xscale[] === log10
        @test_throws ArgumentError AEF.plot_coil_sensitivities(h5path; quantity=:bogus)
        @test_throws ArgumentError AEF.plot_coil_sensitivities(table; quantity=:fraction)
        @test_throws ArgumentError AEF.plot_coil_sensitivities(h5path; coils=["hoop_tilted", "no_such_coil"])

        # The threshold needs the overlap distribution and the threshold together: a pair draws what the file does,
        # a lone RiskResult cannot.
        lines_of(fig) = [p[1][] for p in _plots(_axes(fig)[1], Lines)]
        @test lines_of(AEF.plot_threshold_scaling(["pair" => (mc, risk)])) == lines_of(AEF.plot_threshold_scaling(["f" => h5path]))
        @test_throws ArgumentError AEF.plot_threshold_scaling(["risk only" => risk])

        # Distributions, risk, budget and linearity, with their options.
        @test length(_plots(only(_axes(AEF.plot_tolerance_pdf(h5path; show_batches=true))), Lines)) ==
              length(_plots(only(_axes(AEF.plot_tolerance_pdf(h5path))), Lines)) + mc.nbatch
        @test length(_plots(only(_axes(AEF.plot_locking_risk(h5path; target_percent=1.0))), HLines)) == 1
        @test AEF.plot_tolerance_budget(h5path) isa Figure
        terms = EF.worst_case_terms(EF.without_coils(table, ["hoop_efc"]), snapshot, sets)
        @test AEF.plot_tolerance_budget(table, snapshot, sets; monte_carlo=mc, sort=:name) isa Figure
        @test AEF.plot_tolerance_budget(terms; top=1, sort=:none) isa Figure
        @test_throws ArgumentError AEF.plot_tolerance_budget(terms; sort=:bogus)
        @test AEF.plot_linearity_residuals(h5path; at=:step, yscale=:log10) isa Figure
        @test AEF.plot_linearity_residuals(sens; tolerances=snapshot, fd_step_shift_m=1e-3, fd_step_tilt_deg=0.1) isa Figure
        @test_throws ArgumentError AEF.plot_linearity_residuals(sens)
        @test_throws ArgumentError AEF.plot_linearity_residuals(h5path; at=:bogus)
        @test_throws ArgumentError AEF.plot_linearity_residuals(h5path; yscale=:bogus)

        # Phasors: one panel per source; the axisymmetric hoop alone has no n ≠ 0 field to draw.
        @test length(_axes(AEF.plot_overlap_phasors(["file" => h5path, "memory" => table]))) == 2
        @test_throws ArgumentError AEF.plot_overlap_phasors(h5path; coils=["hoop_axi"])
        @test_throws ArgumentError AEF.plot_overlap_phasors(h5path; coils=["no_such_coil"])

        # The coil diagnostics from a context rebuilt from the file.
        @test AEF.plot_applied_spectra(ctx, ovs) isa Figure
        @test AEF.plot_applied_spectra(h5path, sets) isa Figure
        @test AEF.plot_overlap_contributions(ctx, ovs) isa Figure
        png = joinpath(dir, "surface.png")
        @test AEF.plot_surface_overlay(ctx, ovs; ntheta=32, nzeta=24, save_path=png) isa Figure
        @test isfile(png)
        @test_throws ArgumentError AEF.plot_coil_sensitivities(ovs; quantity=:shift)
        @test AEF.plot_linearity_check(EF.linearity_check(ctx, sets, snapshot; scales=(1.0, 2.0))) isa Figure

        # The correction requirement: the surfaces, then one polar panel per array.
        src = only(o for o in ovs if o.coil_name == "hoop_tilted")
        cfg_y = FT.CoilConfig(; machine=ctx.cfg.machine, dat_dir=ctx.cfg.dat_dir, mtheta_coil=ctx.cfg.mtheta_coil, nzeta_coil=ctx.cfg.nzeta_coil,
            coil_sets=[FT.CoilSetConfig(; name="hoop_y", source="pf_hoop", radius=1.6, height=-0.3, currents=[2.0e3], tilty=[3.0])])
        arr_y = only(EF.coil_overlaps(ctx, FT.load_coil_sets(cfg_y, 1; equil=ctx.equil)))
        req = AEF.plot_correction_requirement(EF.correction_requirement(ctx, src, [src, arr_y]))
        @test length(_axes(req)) == 1 && count(b -> b isa PolarAxis, req.content) == 2

        # Phasing, the needed current and the NTV limits.
        @test AEF.plot_phasing_map(EF.phasing_map(h5path, ["hoop_tilted", "hoop_axi"]; nphase=36)) isa Figure
        @test AEF.plot_phasing_map(h5path, ["hoop_tilted", "hoop_axi"]; nphase=12) isa Figure
        c = only(ef.couplings)
        src_efc = only(o for o in ovs if o.coil_name == "hoop_efc")
        @test AEF.plot_needed_current(mc, src) isa Figure
        @test AEF.plot_needed_current(mc, src_efc; coupling=c, delta_threshold=risk.threshold_fit, torque_budget=1.0) isa Figure
        @test_throws ArgumentError AEF.plot_needed_current(mc, src_efc; coupling=c)
        # The rotation scan adds the torque-against-rotation panel, and profiles the deposition panel.
        @test length(_axes(AEF.plot_efc_ntv_limits(h5path; torque_budget=1.0))) == 2
        deposition = _axes(AEF.plot_efc_ntv_limits(h5path; torque_budget=1.0, profiles=true))
        @test length(deposition) == 3
        @test last.(_plots(deposition[3], Lines)[1][1][]) ≈ c.torque_full_profile[:, 2] rtol = 1e-6
    end
end
