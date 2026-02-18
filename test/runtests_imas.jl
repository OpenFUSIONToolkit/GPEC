# Tests that the IMAS equilibrium loading path (read_imas, Task 1) produces
# the same result as reading the same equilibrium directly from a gEQDSK file.
#
# Strategy (as recommended in the task description):
#   1. Read the existing test gEQDSK with the standard EFIT path → reference equil
#   2. Convert that same gEQDSK to an IMAS dd using EFIT.jl's geqdsk2imas!
#   3. Load via the new IMAS path → test equil
#   4. Compare key quantities — they should match to within interpolation tolerance

import EFIT
import EFIT.IMASdd

@testset "IMAS equilibrium: read_imas matches read_efit" begin

    data_dir    = joinpath(@__DIR__, "test_data", "regression_equilibrium_example")
    geqdsk_path = joinpath(data_dir, "EQDSK_COCOS_02")

    # ── 1. Reference: load via the standard EFIT g-file path ─────────────────
    efit_control = JPEC.Equilibrium.EquilibriumControl(;
        eq_type     = "efit",
        eq_filename = geqdsk_path,
        jac_type    = "boozer",
        psilow      = 0.01,
        psihigh     = 0.994,
    )
    efit_config = JPEC.Equilibrium.EquilibriumConfig(efit_control, JPEC.Equilibrium.EquilibriumOutput())
    equil_efit  = JPEC.Equilibrium.setup_equilibrium(efit_config)

    # ── 2. Convert the same gEQDSK → IMAS dd using EFIT.jl ───────────────────
    # set_time=0.0 is required because EQDSK_COCOS_02 has a non-standard header
    # that EFIT.jl cannot automatically parse a time value from.
    g  = EFIT.readg(geqdsk_path; set_time=0.0)
    dd = IMASdd.dd()

    # Manually set up the time structure so time_slice[] resolves correctly
    # in read_imas (which uses dd.global_time to select the active slice).
    dd.global_time              = g.time
    dd.equilibrium.time         = [g.time]
    resize!(dd.equilibrium.time_slice, 1)
    dd.equilibrium.time_slice[1].time = g.time

    # Fill all equilibrium profile data from the gEQDSK.
    # wall=nothing: wall data is not needed for the equilibrium comparison.
    # COCOS is auto-detected (geqdsk_cocos=0); dd is written in COCOS 11.
    EFIT.geqdsk2imas!(g, dd.equilibrium.time_slice[1]; wall=nothing)

    # ── 3. Load via the new IMAS path with the same grid settings ─────────────
    # eq_filename is unused for IMAS data loading; its directory is used for
    # any optional diagnostic output files (disabled by default).
    imas_control = JPEC.Equilibrium.EquilibriumControl(;
        eq_type     = "imas",
        eq_filename = joinpath(data_dir, "imas_placeholder"),
        jac_type    = "boozer",
        psilow      = 0.01,
        psihigh     = 0.994,
    )
    imas_config = JPEC.Equilibrium.EquilibriumConfig(imas_control, JPEC.Equilibrium.EquilibriumOutput())
    equil_imas  = JPEC.Equilibrium.setup_equilibrium(imas_config, dd)

    @test equil_imas isa JPEC.Equilibrium.PlasmaEquilibrium

    # ── 4. One-to-one comparison ──────────────────────────────────────────────
    # Both results come from the same physical data, so differences are only
    # from COCOS conversion (EFIT.jl) and our IMAS reader interpolation.
    # Tolerances are set at ~0.1% for scalar quantities and ~1% for profiles.

    # Magnetic axis R position
    @test isapprox(equil_efit.ro, equil_imas.ro, rtol=1e-3)

    # Magnetic axis Z position (may be near zero; use absolute tolerance)
    @test isapprox(equil_efit.zo, equil_imas.zo, atol=1e-3)

    # Normalising flux |psi_boundary - psi_axis|
    @test isapprox(equil_efit.psio, equil_imas.psio, rtol=1e-3)

    # Safety-factor profile q(psi): compare grid-point values
    @test isapprox(equil_efit.profiles.q_spline.y,
                   equil_imas.profiles.q_spline.y, rtol=1e-2)

    # Pressure profile p(psi)
    @test isapprox(equil_efit.profiles.P_spline.y,
                   equil_imas.profiles.P_spline.y, rtol=1e-2)

end

@testset "IMAS write: write_imas populates dd.mhd_linear" begin

    # Build a minimal synthetic DCON result to test write_imas without running
    # the full solver.  We just verify that the IMAS data dictionary is filled
    # correctly from the result NamedTuple.

    # ── 1. Create a dd with an equilibrium time slice already set ─────────────
    dd_out = IMASdd.dd()
    dd_out.equilibrium.time     = [0.0]
    resize!(dd_out.equilibrium.time_slice, 1)
    dd_out.equilibrium.time_slice[1].time = 0.0

    # ── 2. Build minimal control / internal structs ───────────────────────────
    # n = 1 run, 7 poloidal modes (mlow = -3 … mhigh = 3)
    n_modes = 7
    ctrl = JPEC.DCON.DconControl(;
        nn_low   = 1,
        nn_high  = 1,
        vac_flag = true,
        verbose  = false,
    )
    intr = JPEC.DCON.DconInternal(;
        nlow          = 1,
        nhigh         = 1,
        npert         = 1,
        mlow          = -3,
        mhigh         = 3,
        mpert         = n_modes,
        numpert_total = n_modes,
    )

    # ── 3. Synthetic VacuumData — eigenvalues in ascending real order ─────────
    # (negative real part → unstable, positive → stable)
    vac_data      = JPEC.DCON.VacuumData(10, n_modes, n_modes)
    vac_data.et  .= ComplexF64[-0.3, 0.1, 0.2, 0.4, 0.6, 0.8, 1.2]  # et[1] < 0: unstable

    fake_result = (ctrl=ctrl, equil=nothing, intr=intr,
                   ffit=nothing, odet=nothing, vac_data=vac_data)

    # ── 4. Run write_imas and check the dd ────────────────────────────────────
    JPEC.DCON.write_imas(dd_out, fake_result)

    mhd = dd_out.mhd_linear

    # Top-level metadata
    @test mhd.ideal_flag                        == 1
    @test mhd.ids_properties.homogeneous_time   == 1
    @test mhd.code.name                         == "JPEC"

    # One time slice with the right time
    @test length(mhd.time_slice)        == 1
    @test mhd.time_slice[1].time        == 0.0

    # One toroidal_mode entry per eigenmode
    @test length(mhd.time_slice[1].toroidal_mode) == n_modes

    # All entries should have n_tor == 1 (single n = 1 run)
    for tm in mhd.time_slice[1].toroidal_mode
        @test tm.n_tor == 1
    end

    # First eigenmode: most unstable, energy_perturbed < 0
    @test mhd.time_slice[1].toroidal_mode[1].energy_perturbed < 0

    # Energy values should match the real parts of et
    for i in 1:n_modes
        @test isapprox(mhd.time_slice[1].toroidal_mode[i].energy_perturbed,
                       real(vac_data.et[i]); atol=1e-10)
    end

end
