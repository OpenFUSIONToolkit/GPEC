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
    using GeneralizedPerturbedEquilibrium.InnerLayer: InnerLayerModel, SLAYERModel, SLAYERParameters
    using GeneralizedPerturbedEquilibrium.Tearing.CriticalResonantField
    using GeneralizedPerturbedEquilibrium.Runner
    using GeneralizedPerturbedEquilibrium.Utilities: KineticProfiles
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
        tb = TorqueBalance(model, p, Q0, 1.0, p.lu, 2 / p.sval_r^2)
        Qs, bal, Qpeak, br, idx, _ = torque_balance_scan(tb; n=401)
        # Cole's picture: with Q0 above the electron pole, balance holds only for Q_e < Q < Q0.
        @test p.Q_e < Qpeak < Q0
        @test bal[idx] > 0 && isfinite(br) && br > 0
        @test br ≈ sqrt(bal[idx] / (p.lu * 2 / p.sval_r^2))
        # Refining the grid moves b_crit by well under a percent.
        _, _, _, br_fine, _, _ = torque_balance_scan(tb; n=801)
        @test isapprox(br_fine, br; rtol=5e-3)
    end

    @testset "Fortran scan window" begin
        @test torque_balance_window(5.0, 1.0, -2.0) == (1.05, 10.0)
        @test torque_balance_window(0.5, 1.0, -2.0) == (0.8 * -2.0, 0.95)
        @test torque_balance_window(-0.5, 1.0, -2.0) == (1.5 * -2.0, 0.95)
    end

    @testset "local-maximum selection and pole rejection" begin
        p = _mk(; Q_e=50.0, Q_i=-50.0)
        tb = TorqueBalance(_CRFStubLayer(), p, _CRF_Q0_STUB, 1.0, p.lu, 1.0)
        _, bal, Qpeak, br, idx, _ = torque_balance_scan(tb; Qmin=0.0, Qmax=4.0, n=401)
        @test Qpeak ≈ 3.0 atol = 0.011
        @test bal[idx] ≈ _crf_target(Qpeak) rtol = 1e-8
        # A maximum on the electron pole is skipped for the next one.
        p_pole = _mk(; Q_e=3.0, Q_i=-50.0)
        tb_pole = TorqueBalance(_CRFStubLayer(), p_pole, _CRF_Q0_STUB, 1.0, p.lu, 1.0)
        _, _, Qpeak_pole, _, _, _ = torque_balance_scan(tb_pole; Qmin=0.0, Qmax=4.0, n=401)
        @test Qpeak_pole ≈ 1.0 atol = 0.011
        # No interior maximum (monotone balance) returns NaN.
        _, _, Qnone, brnone, inone, _ = @test_logs (:warn,) torque_balance_scan(tb; Qmin=2.7, Qmax=2.95, n=51)
        @test isnan(Qnone) && isnan(brnone) && inone == 0
    end

    @testset "runner: P from P_tor, Q0 = τ_k·n·ω_E" begin
        psi = collect(range(0.0, 1.0; length=11))
        ω_E = 4.0e4
        prof = KineticProfiles(; psi=psi, n_e=fill(4e19, 11), T_e=fill(1.7e3, 11), T_i=fill(2e3, 11),
            omega=fill(ω_E, 11), omega_e=fill(-1e4, 11), omega_i=fill(1.6e4, 11))
        params = [_mk(; n=1, P_tor=3.0, ising=1), _mk(; n=2, P_tor=5.0, ising=2)]
        ctrl = CriticalResonantFieldControl(; enabled=true, n=201, store_scan=true)
        r = run_critical_resonant_field(params, [0.5, 0.7], prof, ctrl)
        @test r.enabled
        @test r.p_phi == [3.0, 5.0]
        @test r.q0 ≈ [p.tauk * p.n * ω_E for p in params]
        @test r.rational_index == [1, 2]
        @test length(r.scan) == 2 && length(r.scan[1].Q) == 201
        @test !run_critical_resonant_field(params, [0.5, 0.7], prof, CriticalResonantFieldControl()).enabled

        # HDF5 output honours the metadata contract.
        c = SLAYERControl(; enabled=true, scan_mode=:brute_force, nre=4, nim=4, critical_resonant_field=ctrl)
        base = run_slayer_from_inputs(params, ComplexF64[1.0 0.0; 0.0 1.0], c; rational_psi=[0.5, 0.7], rational_q=[2.0, 1.5])
        res = SLAYERResult((f === :critical_resonant_field ? r : getfield(base, f) for f in fieldnames(SLAYERResult))...)
        mktemp() do path, io
            close(io)
            h5open(path, "w") do f
                write_slayer_hdf5!(f, res)
            end
            h5open(path, "r") do f
                @test isempty(_collect_metadata_violations(f))
                g = f["Tearing/CriticalResonantField"]
                @test read(g["p_phi"]) == [3.0, 5.0]
                @test read(g["br_crit"]) ≈ r.br_crit nans = true
                @test haskey(g, "Scan/Surface_2/Delta")
            end
        end
    end

    @testset "control: TOML parsing and validation" begin
        c = critical_resonant_field_control_from_toml(Dict("enabled" => true, "n" => 500))
        @test c.enabled && c.n == 500 && c.Qmin === nothing && !c.store_scan
        @test critical_resonant_field_control_from_toml(Dict("Qmin" => -2.0)).Qmin == -2.0
        @test_throws ArgumentError critical_resonant_field_control_from_toml(Dict("viscous_input" => 1.0))
        s = slayer_control_from_toml(Dict("enabled" => true, "CriticalResonantField" => Dict("enabled" => true)))
        @test s.critical_resonant_field.enabled
        @test_throws ArgumentError Runner.validate(SLAYERControl(; inner_model=:ggj_shooting,
            critical_resonant_field=CriticalResonantFieldControl(; enabled=true)))
    end
end
