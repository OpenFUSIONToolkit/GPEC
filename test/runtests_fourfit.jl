using Test
using JPEC
using JPEC.Equilibrium
using JPEC.DCON
using JPEC.Spl

"""
    build_dummy_equilibrium(; mpsi=5, mtheta=6)

Create a lightweight PlasmaEquilibrium suitable for matrix/metric tests.
The spline data are simple constants to keep metrics well-defined and avoid
singularities during derivative evaluations.
"""
function build_dummy_equilibrium(; mpsi=5, mtheta=6)
    cfg = Equilibrium.EquilibriumConfig()
    params = Equilibrium.EquilibriumParameters()

    xs = collect(range(0.0, 1.0; length=mpsi))
    ys = collect(range(0.0, 1.0; length=mtheta))

    # 1D profiles: keep q slightly above zero, pressure small
    fs = hcat(zeros(mpsi), zeros(mpsi), ones(mpsi), 1 .+ 0.1 .* xs)
    sq_spline = Spl.CubicSpline(xs, fs)

    # 2D geometry spline: constant, nonsingular values
    rzphi_fs = ones(mpsi, mtheta, 4)
    rzphi_fs[:, :, 2] .= 0.0  # eta offset
    rzphi_fs[:, :, 3] .= 0.0  # nu
    rzphi_fs[:, :, 4] .= 1.0  # jacobian
    rzphi_spline = Spl.BicubicSpline(xs, ys, rzphi_fs)

    # 2D physics spline: constant B and zeros elsewhere
    eqfun_fs = ones(mpsi, mtheta, 3)
    eqfun_fs[:, :, 2] .= 0.0
    eqfun_fs[:, :, 3] .= 0.0
    eqfun_spline = Spl.BicubicSpline(xs, ys, eqfun_fs)

    ro = 1.0
    zo = 0.0
    psio = 0.1

    return Equilibrium.PlasmaEquilibrium(cfg, params, sq_spline, rzphi_spline, eqfun_spline, ro, zo, psio)
end

function build_dummy_intr(; mlow=-1, mhigh=1, mband=1, n=1)
    mpert = mhigh - mlow + 1
    npert = 1
    numpert_total = mpert * npert
    return DCON.DconInternal(; mlow, mhigh, mpert, mband, nlow=n, nhigh=n, npert, numpert_total)
end

function build_dummy_ctrl(; verbose=false, kin_flag=false, fft_flag=false, nn_low=1, nn_high=1)
    return DCON.DconControl(; verbose, kin_flag, fft_flag, nn_low, nn_high)
end

@testset "Fourfit.make_matrix" begin

    # Setup: Create a minimal test equilibrium
    @testset "Initialization" begin
        equil = build_dummy_equilibrium()

        # Set up internal DCON parameters
        intr = build_dummy_intr(; mlow=-2, mhigh=2, mband=2, n=1)

        # Create metric data
        metric = DCON.make_metric(equil; mband=intr.mband, fft_flag=false)

        @test !isnothing(metric)
        @test metric.mpsi > 0
        @test metric.mtheta > 0
    end

    @testset "Action Matrices" begin
        equil = build_dummy_equilibrium()

        intr = build_dummy_intr(; mlow=-1, mhigh=1, mband=1, n=1)
        ctrl = build_dummy_ctrl(; verbose=false, fft_flag=false, kin_flag=false, nn_low=intr.nlow, nn_high=intr.nhigh)

        metric = DCON.make_metric(equil; mband=intr.mband, fft_flag=false)
        ffit = DCON.make_matrix(equil, intr, metric)

        # Should run without throwing and populate action matrix splines
        DCON.action_matrices!(ffit, intr, equil, ctrl, metric)

        @test !isnothing(ffit.smats)
        @test !isnothing(ffit.tmats)
        @test !isnothing(ffit.xmats)
        @test !isnothing(ffit.ymats)
        @test !isnothing(ffit.zmats)
    end

    @testset "Matrix Construction" begin
        # Create a test equilibrium
        equil = build_dummy_equilibrium()

        intr = build_dummy_intr(; mlow=-1, mhigh=1, mband=1, n=1)

        metric = DCON.make_metric(equil; mband=intr.mband, fft_flag=false)

        # Call make_matrix
        ffit = DCON.make_matrix(equil, intr, metric)

        @test !isnothing(ffit)
        @test ffit.mpert == intr.mpert
        @test ffit.mband == intr.mband
    end

    @testset "Spline Creation" begin
        equil = build_dummy_equilibrium()

        intr = build_dummy_intr(; mlow=-1, mhigh=1, mband=1, n=1)

        metric = DCON.make_metric(equil; mband=intr.mband, fft_flag=false)
        ffit = DCON.make_matrix(equil, intr, metric)

        # Check that splines are created
        @test !isnothing(ffit.amats)
        @test !isnothing(ffit.bmats)
        @test !isnothing(ffit.cmats)
        @test !isnothing(ffit.dmats)
        @test !isnothing(ffit.emats)
        @test !isnothing(ffit.hmats)
        @test !isnothing(ffit.fmats_lower)
        @test !isnothing(ffit.gmats)
        @test !isnothing(ffit.kmats)
    end

    @testset "Matrix Properties" begin
        equil = build_dummy_equilibrium()

        intr = build_dummy_intr(; mlow=-1, mhigh=1, mband=1, n=1)

        metric = DCON.make_metric(equil; mband=intr.mband, fft_flag=false)
        ffit = DCON.make_matrix(equil, intr, metric)

        # Evaluate a spline at an intermediate psi
        psi_test = metric.xs[Int(metric.mpsi ÷ 2)]
        amat_eval = Spl.spline_eval!(ffit.amats, psi_test)

        # Check dimensions
        @test length(amat_eval) == intr.numpert_total^2

        # Check that matrices are not all zeros
        @test any(!iszero, real.(amat_eval))
    end

    @testset "Kinetic Corrections" begin
        equil = build_dummy_equilibrium()

        intr = build_dummy_intr(; mlow=-1, mhigh=1, mband=1, n=1)

        metric = DCON.make_metric(equil; mband=intr.mband, fft_flag=false)

        # This should run without error even with kinetic corrections
        ffit = DCON.make_matrix(equil, intr, metric)

        @test !isnothing(ffit)
        @test ffit.jmat !== nothing
    end

    @testset "Kinetic Matrix" begin
        equil = build_dummy_equilibrium()

        intr = build_dummy_intr(; mlow=-1, mhigh=1, mband=1, n=1)

        ctrl = DCON.DconControl(
            kin_flag=true,
            nn_low=1,
            nn_high=1,
            passing_flag=true,
            trapped_flag=true,
            ion_flag=true,
            electron_flag=false,
            kinfac1=1.0,
            kinfac2=1.0,
            kingridtype=0,
            kinetic=DCON.KineticParams(nl=1),
            verbose=false
        )

        metric = DCON.make_metric(equil; mband=intr.mband, fft_flag=false)
        ffit = DCON.make_matrix(equil, intr, metric)

        ffit = DCON.make_kinetic_matrix(equil, intr, ctrl, metric, ffit)

        @test !isnothing(ffit.akmats)
        @test !isnothing(ffit.kwmats[1])
        @test !isnothing(ffit.ktmats[1])
    end

end

println("All Fourfit tests passed!")

@testset "Fourfit.additional_safety" begin
    # FFT constraints: expect success when (mtheta-1) is power of two, error otherwise
    @testset "Metric FFT constraints" begin
        equil_ok = build_dummy_equilibrium(; mtheta=9) # 9-1=8 is power of two
        intr_ok = build_dummy_intr(; mlow=-1, mhigh=1, mband=1, n=1)
        metric_ok = DCON.make_metric(equil_ok; mband=intr_ok.mband, fft_flag=true)
        @test metric_ok.fspline !== nothing

        equil_bad = build_dummy_equilibrium(; mtheta=6) # 6-1=5 not power of two
        intr_bad = build_dummy_intr(; mlow=-1, mhigh=1, mband=1, n=1)
        @test_throws Exception DCON.make_metric(equil_bad; mband=intr_bad.mband, fft_flag=true)
    end

    # Action matrices should be band-limited: entries with |i-j|>mband ~ 0
    @testset "Action band-limitedness" begin
        equil = build_dummy_equilibrium()
        intr = build_dummy_intr(; mlow=-1, mhigh=1, mband=1, n=1)
        ctrl = build_dummy_ctrl(; verbose=false, fft_flag=false, kin_flag=false, nn_low=intr.nlow, nn_high=intr.nhigh)
        metric = DCON.make_metric(equil; mband=intr.mband, fft_flag=false)
        ffit = DCON.make_matrix(equil, intr, metric)
        DCON.action_matrices!(ffit, intr, equil, ctrl, metric)
        psi = metric.xs[Int(cld(metric.mpsi, 2))]
        mats = (
            reshape(Spl.spline_eval!(ffit.smats, psi), intr.mpert, intr.mpert),
            reshape(Spl.spline_eval!(ffit.tmats, psi), intr.mpert, intr.mpert),
            reshape(Spl.spline_eval!(ffit.xmats, psi), intr.mpert, intr.mpert),
            reshape(Spl.spline_eval!(ffit.ymats, psi), intr.mpert, intr.mpert),
            reshape(Spl.spline_eval!(ffit.zmats, psi), intr.mpert, intr.mpert)
        )
        for M in mats
            # Zero out entries within band and check remaining are ~0
            keep = [abs(i-j) <= intr.mband for i in 1:intr.mpert, j in 1:intr.mpert]
            offband = copy(M)
            offband[keep] .= 0
            @test sum(abs, offband) ≤ 1e-8
        end
    end

    # F positivity: reconstruct F and check Cholesky succeeds
    @testset "F positivity and Hermitian" begin
        equil = build_dummy_equilibrium()
        intr = build_dummy_intr(; mlow=-1, mhigh=1, mband=1, n=1)
        metric = DCON.make_metric(equil; mband=intr.mband, fft_flag=false)
        ffit = DCON.make_matrix(equil, intr, metric)
        psi = metric.xs[end]
        Lflat = Spl.spline_eval!(ffit.fmats_lower, psi)
        L = reshape(Lflat, intr.numpert_total, intr.numpert_total)
        F = L * adjoint(L)
        @test ishermitian(F)
        @test cholesky(Hermitian(F)).info == 0
    end

    # mband invariance when higher bands are zero
    @testset "mband overlap consistency (|dm|<=1)" begin
        equil = build_dummy_equilibrium()
        intr1 = build_dummy_intr(; mlow=-1, mhigh=1, mband=1, n=1)
        intr2 = build_dummy_intr(; mlow=-1, mhigh=1, mband=2, n=1)
        metric1 = DCON.make_metric(equil; mband=intr1.mband, fft_flag=false)
        metric2 = DCON.make_metric(equil; mband=intr2.mband, fft_flag=false)
        ffit1 = DCON.make_matrix(equil, intr1, metric1)
        ffit2 = DCON.make_matrix(equil, intr2, metric2)
        psi = metric1.xs[Int(cld(metric1.mpsi, 2))]
        A1 = reshape(Spl.spline_eval!(ffit1.amats, psi), intr1.numpert_total, intr1.numpert_total)
        A2 = reshape(Spl.spline_eval!(ffit2.amats, psi), intr2.numpert_total, intr2.numpert_total)
        # Compare only the band |i-j|<=1 where both should agree closely
        mask = [abs(i-j) <= 1 for i in 1:intr1.numpert_total, j in 1:intr1.numpert_total]
        @test maximum(abs.(A1[mask] .- A2[mask])) ≤ 1e-6
    end

    # Kinetic guards
    @testset "Kinetic guardrails" begin
        equil = build_dummy_equilibrium()
        intr = build_dummy_intr(; mlow=-1, mhigh=1, mband=1, n=1)
        metric = DCON.make_metric(equil; mband=intr.mband, fft_flag=false)
        ffit = DCON.make_matrix(equil, intr, metric)
        # kingridtype != 0 should throw
        ctrl_bad_grid = DCON.DconControl(; kin_flag=true, nn_low=intr.nlow, nn_high=intr.nhigh, kingridtype=1, trapped_flag=true, ion_flag=true)
        @test_throws ErrorException DCON.make_kinetic_matrix(equil, intr, ctrl_bad_grid, metric, ffit)
        # nn_low == 0 should throw
        ctrl_bad_n = DCON.DconControl(; kin_flag=true, nn_low=0, nn_high=0, trapped_flag=true, ion_flag=true)
        @test_throws ErrorException DCON.make_kinetic_matrix(equil, intr, ctrl_bad_n, metric, ffit)
    end

    # Spline edge evaluation should be finite
    @testset "Spline edge finiteness" begin
        equil = build_dummy_equilibrium()
        intr = build_dummy_intr(; mlow=-1, mhigh=1, mband=1, n=1)
        metric = DCON.make_metric(equil; mband=intr.mband, fft_flag=false)
        ffit = DCON.make_matrix(equil, intr, metric)
        for psi in (first(metric.xs), last(metric.xs))
            for sp in (ffit.amats, ffit.bmats, ffit.cmats, ffit.dmats, ffit.emats, ffit.hmats, ffit.gmats, ffit.kmats)
                v = Spl.spline_eval!(sp, psi)
                @test all(isfinite, real.(v))
            end
        end
    end
end
