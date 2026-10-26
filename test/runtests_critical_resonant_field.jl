using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.InnerLayer
using GeneralizedPerturbedEquilibrium.InnerLayer: InnerLayerModel

# Layer stub returning a prescribed torque balance, to test the maximum selection.
struct _CRFStubLayer <: InnerLayerModel end
const _CRF_Q0_STUB = 10.0
_crf_target(Q) = 0.1 + exp(-(Q - 1)^2 / 0.01) + 2exp(-(Q - 3)^2 / 0.01)
function InnerLayer.solve_inner(::_CRFStubLayer, p, Q::Number)
    Qc = -real(Q)   # undo the axis mirror cole_delta applies
    jxb = 2 * (_CRF_Q0_STUB - Qc) / _crf_target(Qc)
    return (; tearing=conj(im / jxb - 1e-2))
end

@testset "CriticalResonantField: torque balance" begin
    using GeneralizedPerturbedEquilibrium
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.InnerLayer: InnerLayerModel, SLAYERModel, SLAYERParameters, GGJModel
    using GeneralizedPerturbedEquilibrium.Tearing.CriticalResonantField
    using GeneralizedPerturbedEquilibrium.Runner
    using HDF5

    include("h5_metadata_check.jl")

    # DIII-D-like 2/1 layer (Fortran-normalized inputs of the shipped SLAYER example).
    _mk(; n=1, P_tor=1.0, Q_e=1.38, Q_i=-2.15, ising=1) = SLAYERParameters(;
        tau=1.2, lu=6.44e7, c_beta=0.112, D_norm=3.0, P_perp=1.0, P_tor=P_tor,
        Q_e=Q_e, Q_i=Q_i, iota_e=Q_e / (Q_e - Q_i), tauk=1.31e-4, tau_r=21.1,
        delta_n=885.0, rs=0.453, R0=1.74, bt=2.0, sval_r=1.09, eta=1.22e-8,
        d_beta=3.6e-3, m=2, n=n, ising=ising)
    model = SLAYERModel{:fitzpatrick}()

    @testset "Cole axis: branch lies between Q_e and Q0" begin
        p = _mk()
        @test cole_delta(model, p, 0.7) ≈ conj(InnerLayer.solve_inner(model, p, -0.7).tearing)
        Q0 = 5.45
        tb = TorqueBalance(model, p, Q0, 1.0, p.lu, 2 / p.sval_r^2, 1e-2)
        Qs, bal, Qpeak, br, idx, _ = torque_balance_scan(tb; n=401)
        # Cole's picture: with Q0 above the electron pole, balance holds only for Q_e < Q < Q0.
        @test p.Q_e < Qpeak < Q0
        @test bal[idx] > 0 && isfinite(br) && br > 0
        @test br ≈ sqrt(bal[idx] / (p.lu * 2 / p.sval_r^2))
        # Refining the grid moves b_crit by well under a percent.
        _, _, _, br_fine, _, _ = torque_balance_scan(tb; n=801)
        @test isapprox(br_fine, br; rtol=5e-3)
    end

    @testset "scan window" begin
        @test torque_balance_window(5.0, 1.0, -2.0) == (1.05, 10.0)
        @test torque_balance_window(0.5, 1.0, -2.0) == (0.8 * -2.0, 0.95)
        @test torque_balance_window(-0.5, 1.0, -2.0) == (1.5 * -2.0, 0.95)
    end

    @testset "local-maximum selection and pole rejection" begin
        p = _mk(; Q_e=50.0, Q_i=-50.0)
        tb = TorqueBalance(_CRFStubLayer(), p, _CRF_Q0_STUB, 1.0, p.lu, 1.0, 1e-2)
        _, bal, Qpeak, br, idx, _ = torque_balance_scan(tb; Qmin=0.0, Qmax=4.0, n=401)
        @test Qpeak ≈ 3.0 atol = 0.011
        @test bal[idx] ≈ _crf_target(Qpeak) rtol = 1e-8
        # A maximum on the electron pole is skipped for the next one, and a narrower exclusion keeps it.
        p_pole = _mk(; Q_e=3.0, Q_i=-50.0)
        tb_pole = TorqueBalance(_CRFStubLayer(), p_pole, _CRF_Q0_STUB, 1.0, p.lu, 1.0, 1e-2)
        _, _, Qpeak_pole, _, _, _ = torque_balance_scan(tb_pole; Qmin=0.0, Qmax=4.0, n=401)
        @test Qpeak_pole ≈ 1.0 atol = 0.011
        # A maximum 0.015 from the pole is excluded by the default width and kept with pole_width = 0.
        tb_near = TorqueBalance(_CRFStubLayer(), _mk(; Q_e=3.015, Q_i=-50.0), _CRF_Q0_STUB, 1.0, p.lu, 1.0, 1e-2)
        @test torque_balance_scan(tb_near; Qmin=0.0, Qmax=4.0, n=401)[3] ≈ 1.0 atol = 0.011
        @test torque_balance_scan(tb_near; Qmin=0.0, Qmax=4.0, n=401, pole_width=0.0)[3] ≈ 3.0 atol = 0.011
        # No interior maximum (monotone balance) returns NaN.
        _, _, Qnone, brnone, inone, _ = @test_logs (:warn,) torque_balance_scan(tb; Qmin=2.7, Qmax=2.95, n=51)
        @test isnan(Qnone) && isnan(brnone) && inone == 0
    end

    @testset "surfaces the window cannot handle are skipped" begin
        # Unknown rotation: no Q0, so no threshold.
        tb_nan = TorqueBalance(model, _mk(), NaN, 1.0, 6.44e7, 1.0, 1e-2)
        _, _, Qn, brn, in_, _ = @test_logs (:warn, r"not finite") torque_balance_scan(tb_nan; n=51)
        @test isnan(Qn) && isnan(brn) && in_ == 0
        # Reversed diamagnetic ordering (Q_e < 0 < Q_i) is not guessed at.
        tb_rev = TorqueBalance(model, _mk(; Q_e=-1.38, Q_i=2.15), -0.5, 1.0, 6.44e7, 1.0, 1e-2)
        _, _, Qr, brr, ir, _ = @test_logs (:warn, r"Q_i < 0 < Q_e") torque_balance_scan(tb_rev; n=51)
        @test isnan(Qr) && isnan(brr) && ir == 0
    end

    @testset "runner: P from P_tor, Q0 = τ_k·n·Ω_E" begin
        ω_E = 4.0e4
        params = [_mk(; n=1, P_tor=3.0, ising=1), _mk(; n=2, P_tor=5.0, ising=2)]
        ctrl = CriticalResonantFieldControl(; enabled=true, n=201, store_scan=true)
        r = run_critical_resonant_field(model, params, [ω_E, ω_E], [-2.0, -3.0], ctrl)
        @test r.enabled
        @test r.Q0 ≈ [p.tauk * p.n * ω_E for p in params]
        @test r.rational_index == [1, 2]
        # α is each surface's own S^(-1/3)·(Δ_crit − r_s Δ'); a tearing-unstable surface has no threshold.
        r_unst = @test_logs (:warn, r"2/2 surface skipped: it is tearing-unstable") match_mode = :any run_critical_resonant_field(
            model, params, [ω_E, ω_E], [-2.0, 0.5], ctrl)
        @test isequal(r_unst.br_crit[1], r.br_crit[1]) && isnan(r_unst.br_crit[2])
        # Zero E×B rotation is a valid input (Cole Sec. IV E: purely diamagnetic rotation) and gives a finite threshold.
        r_zero = run_critical_resonant_field(model, params[1:1], [0.0], [-2.0], ctrl)
        @test r_zero.Q0 == [0.0] && isfinite(r_zero.br_crit[1]) && r_zero.br_crit[1] > 0
        @test length(r.scan) == 2 && length(r.scan[1].Q) == 201
        @test !run_critical_resonant_field(model, params, [ω_E, ω_E], [-2.0, -3.0], CriticalResonantFieldControl()).enabled
        # Only the SLAYER layer defines the Cole-axis mirror the scan relies on.
        @test_throws "needs the SLAYER" run_critical_resonant_field(GGJModel(), params, [ω_E, ω_E], [-2.0, -3.0], ctrl)

        # The analysis runs inside run_slayer_from_inputs, so script callers get it too, and its
        # HDF5 output honours the metadata contract.
        c = SLAYERControl(; enabled=true, scan_mode=:brute_force, nre=4, nim=4, critical_resonant_field=ctrl)
        dp = ComplexF64[-2.0 0.0; 0.0 -3.0]
        res = run_slayer_from_inputs(model, params, dp, c; rational_psi=[0.5, 0.7], rational_q=[2.0, 1.5], omega_E=[ω_E, ω_E])
        @test res.critical_resonant_field.enabled
        @test isequal(res.critical_resonant_field.br_crit, r.br_crit)
        mktemp() do path, io
            close(io)
            h5open(path, "w") do f
                write_slayer_hdf5!(f, res)
            end
            h5open(path, "r") do f
                @test isempty(_collect_metadata_violations(f))
                g = f["Tearing/CriticalResonantField"]
                @test read(g["Q0"]) ≈ r.Q0
                @test read(g["br_crit"]) ≈ r.br_crit nans = true
                @test !haskey(g, "p_phi")
                @test haskey(g, "Scan/Surface_2/Delta")
            end
        end

        # Without any rotation there is no threshold to compute, so the run stops rather than assume Ω_E = 0.
        @test_throws "needs the E×B rotation" run_slayer_from_inputs(model, params, dp, c; rational_psi=[0.5, 0.7], rational_q=[2.0, 1.5])
        # A surface the override leaves without a rotation is skipped; the others are still computed.
        c_part = SLAYERControl(; enabled=true, scan_mode=:brute_force, nre=4, nim=4, critical_resonant_field=ctrl,
            omega_E_kHz=Dict("2/1" => ω_E / (2π * 1e3)))
        r_part = (@test_logs (:warn, r"2/2 surface skipped") match_mode = :any run_slayer_from_inputs(model, params, dp, c_part;
            rational_psi=[0.5, 0.7], rational_q=[2.0, 1.5])).critical_resonant_field
        @test r_part.Q0[1] ≈ r.Q0[1] && isnan(r_part.Q0[2])
        @test isequal(r_part.br_crit[1], r.br_crit[1]) && isnan(r_part.br_crit[2])
    end

    @testset "control: TOML parsing and validation" begin
        c = critical_resonant_field_control_from_toml(Dict("enabled" => true, "n" => 500))
        @test c.enabled && c.n == 500 && c.Qmin === nothing && !c.store_scan && c.pole_width == 0.02
        @test critical_resonant_field_control_from_toml(Dict("Qmin" => -2.0, "pole_width" => 0.05)).pole_width == 0.05
        @test_throws ArgumentError critical_resonant_field_control_from_toml(Dict("viscous_input" => 1.0))
        s = slayer_control_from_toml(Dict("enabled" => true, "CriticalResonantField" => Dict("enabled" => true)))
        @test s.critical_resonant_field.enabled
        @test_throws "n must be ≥ 3" Runner.validate(SLAYERControl(; critical_resonant_field=CriticalResonantFieldControl(; n=2)))
        @test_throws "pole_width" Runner.validate(SLAYERControl(; critical_resonant_field=CriticalResonantFieldControl(; pole_width=-0.1)))
    end
end
