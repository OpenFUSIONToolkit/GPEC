@testset "Dispersion per-surface rotation shift" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.InnerLayer: InnerLayerModel, solve_inner
    using GeneralizedPerturbedEquilibrium.Dispersion
    using GeneralizedPerturbedEquilibrium.Tearing.Runner: SLAYERControl,
        slayer_control_from_toml, validate, _q_shifts
    using LinearAlgebra

    # Linear inner layer Δ(Q) = a + b·Q makes the applied Q offset readable
    # straight off the residual.
    struct RotTestModel <: InnerLayerModel
        a::ComplexF64
        b::ComplexF64
    end
    GeneralizedPerturbedEquilibrium.InnerLayer.solve_inner(
        m::RotTestModel, params, Q::Number) =
        InnerLayerResponse(m.a + m.b * ComplexF64(Q), zero(ComplexF64))

    model = RotTestModel(0.0im, 1.0 + 0im)

    @testset "q_shift defaults to zero and is inert" begin
        sc = surface_coupling(model, nothing, 1.0 + 0im; scale=1.0, tauk=1.0)
        @test sc.q_shift == 0.0
        # Unshifted residual is dp_diag - Δ(Q) = 1 - Q
        @test sc(2.0 + 0im) ≈ (1.0 - 2.0) + 0im
    end

    @testset "q_shift offsets the layer Q argument on the scalar residual" begin
        shift = 0.75
        sc0 = surface_coupling(model, nothing, 1.0 + 0im; scale=1.0, tauk=1.0)
        scs = surface_coupling(model, nothing, 1.0 + 0im; scale=1.0, tauk=1.0,
            q_shift=shift)
        @test scs.q_shift == shift
        for Q in (0.0 + 0im, 2.0 - 1.0im, -3.5 + 0.25im)
            @test scs(Q) ≈ sc0(Q + shift)
        end
    end

    @testset "q_shift is real: it moves Re(Q), not Im(Q)" begin
        # The shift models rotation, so it must not leak into the growth-rate
        # axis. Δ(Q) = Q here, so the residual difference IS the applied offset.
        sc0 = surface_coupling(model, nothing, 0.0 + 0im; scale=1.0, tauk=1.0)
        scs = surface_coupling(model, nothing, 0.0 + 0im; scale=1.0, tauk=1.0,
            q_shift=2.0)
        applied = sc0(1.0 + 1.0im) - scs(1.0 + 1.0im)   # = Δ(Q+s) - Δ(Q) = s
        @test real(applied) ≈ 2.0
        @test imag(applied) ≈ 0.0 atol = 1e-14
    end

    @testset "Coupled determinant applies each surface's own shift" begin
        # Diagonal Δ', so det = Π_k (dp_kk - Δ_k(Q·tauk_ref/tauk_k + shift_k)).
        s1, s2 = 0.5, -1.25
        sc1 = surface_coupling(model, nothing, 1.0 + 0im; scale=1.0, tauk=1.0,
            q_shift=s1)
        sc2 = surface_coupling(model, nothing, 2.0 + 0im; scale=1.0, tauk=2.0,
            q_shift=s2)
        dp = ComplexF64[1.0 0.0; 0.0 2.0]
        mc = multi_surface_coupling([sc1, sc2], dp)

        Q = 1.5 + 0.5im
        expected = (1.0 - (Q * (1.0 / 1.0) + s1)) * (2.0 - (Q * (1.0 / 2.0) + s2))
        @test mc(Q) ≈ expected

        # Zero shifts reproduce the pre-rotation determinant exactly.
        sc1z = surface_coupling(model, nothing, 1.0 + 0im; scale=1.0, tauk=1.0)
        sc2z = surface_coupling(model, nothing, 2.0 + 0im; scale=1.0, tauk=2.0)
        mcz = multi_surface_coupling([sc1z, sc2z], dp)
        @test mcz(Q) ≈ (1.0 - Q) * (2.0 - Q / 2)
    end

    @testset "_q_shifts converts kHz to Q units via tauk" begin
        p1 = slayer_parameters(n_e=5.0e19, t_e=1000.0, t_i=1000.0,
            omega=0.0, omega_e=1.0e4, omega_i=5.0e3,
            qval=2.0, sval_r=1.0, bt=2.0, rs=0.5, R0=1.7,
            mu_i=2.0, zeff=1.0, chi_perp=1.0, chi_tor=1.0, m=2, n=1)
        p2 = slayer_parameters(n_e=5.0e19, t_e=1000.0, t_i=1000.0,
            omega=0.0, omega_e=1.0e4, omega_i=5.0e3,
            qval=3.0, sval_r=1.5, bt=2.0, rs=0.6, R0=1.7,
            mu_i=2.0, zeff=1.0, chi_perp=1.0, chi_tor=1.0, m=3, n=1)
        params = [p1, p2]

        # Empty (default) means no shift anywhere.
        @test _q_shifts(SLAYERControl(), params, 2) == [0.0, 0.0]

        ctrl = SLAYERControl(; omega_shift_kHz=[1.0, -2.0])
        got = _q_shifts(ctrl, params, 2)
        @test got[1] ≈ p1.tauk * 2π * 1e3 * 1.0
        @test got[2] ≈ p2.tauk * 2π * 1e3 * -2.0

        # Length must match the number of surfaces analysed.
        @test_throws ArgumentError _q_shifts(SLAYERControl(; omega_shift_kHz=[1.0]),
            params, 2)
    end

    @testset "omega_shift_kHz parses and validates from TOML" begin
        ctrl = slayer_control_from_toml(Dict("omega_shift_kHz" => [0, 3.0, -1.5]))
        @test ctrl.omega_shift_kHz == [0.0, 3.0, -1.5]
        @test eltype(ctrl.omega_shift_kHz) === Float64
        # Default stays empty so existing decks are untouched.
        @test isempty(slayer_control_from_toml(Dict{String,Any}()).omega_shift_kHz)
        # A non-finite shift would silently poison every Q evaluation; the
        # validator rejects it (TOML cannot express NaN, so go through validate).
        @test_throws ArgumentError validate(SLAYERControl(; omega_shift_kHz=[NaN]))
    end
end
