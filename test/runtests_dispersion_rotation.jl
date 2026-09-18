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
        # Default :direct rescale: surface k sees Q·tauk_k/tauk_ref + q_shift_k.
        expected = (1.0 - (Q * (1.0 / 1.0) + s1)) * (2.0 - (Q * (2.0 / 1.0) + s2))
        @test mc(Q) ≈ expected

        # Zero shifts reproduce the un-rotated determinant exactly.
        sc1z = surface_coupling(model, nothing, 1.0 + 0im; scale=1.0, tauk=1.0)
        sc2z = surface_coupling(model, nothing, 2.0 + 0im; scale=1.0, tauk=2.0)
        mcz = multi_surface_coupling([sc1z, sc2z], dp)
        @test mcz(Q) ≈ (1.0 - Q) * (2.0 - 2Q)
    end

    @testset "_q_shifts: kinetic-file Ω_E by default, omega_E_kHz override, n-scaled" begin
        mk(; qval, rs, m, n) = slayer_parameters(n_e=5.0e19, t_e=1000.0, t_i=1000.0,
            omega_e=1.0e4, omega_i=5.0e3,
            qval=qval, sval_r=1.0, bt=2.0, rs=rs, R0=1.7,
            mu_i=2.0, zeff=1.0, chi_perp=1.0, chi_tor=1.0, m=m, n=n)
        params = [mk(; qval=1.5, rs=0.5, m=3, n=2), mk(; qval=2.0, rs=0.6, m=4, n=2)]
        Ω_file = [3.0e4, -1.0e4]   # rad/s per unit n, as carried by the kinetic file

        # No file rotation and no override: no shift anywhere.
        @test _q_shifts(SLAYERControl(), params, 2) == [0.0, 0.0]

        # File rotation by default. The mode sees n·Ω_E, and ĝ = i(Q_E − ω·τ_k) gives
        # Q_k = τ_k·(ω − n·Ω_E), so the offset is −τ_k·n·Ω_E.
        got = _q_shifts(SLAYERControl(), params, 2; omega_E=Ω_file)
        @test got ≈ [-params[1].tauk * 2 * Ω_file[1], -params[2].tauk * 2 * Ω_file[2]]

        # A non-empty omega_E_kHz replaces the file values, in the same per-unit-n convention.
        ovr = _q_shifts(SLAYERControl(; omega_E_kHz=[1.0, -2.0]), params, 2; omega_E=Ω_file)
        @test ovr ≈ [-params[1].tauk * 2 * 2π * 1e3 * 1.0, -params[2].tauk * 2 * 2π * 1e3 * -2.0]

        # Either source must supply exactly one value per analysed surface.
        @test_throws ArgumentError _q_shifts(SLAYERControl(; omega_E_kHz=[1.0]), params, 2)
        @test_throws ArgumentError _q_shifts(SLAYERControl(), params, 2; omega_E=[1.0])
    end

    @testset "Coupled roots do not depend on the reference surface" begin
        # Diagonal Δ′ decouples the surfaces, so each coupled root must be that surface's own
        # root in physical units whichever surface normalizes Q.
        d = ComplexF64[1.0+1.0im, 3.0+2.0im]
        tauk = [2.0e-4, 5.0e-4]
        scs = [surface_coupling(model, nothing, d[k]; scale=1.0, tauk=tauk[k]) for k in 1:2]
        # Δ(Q) = Q makes surface k's own root Q_k = d_k, i.e. ω + iγ = d_k/τ_k.
        expected = sort(d ./ tauk; by=real)
        for ref in 1:2
            mc = multi_surface_coupling(scs, diagm(d); ref_idx=ref, msing_max=2)
            # det is quadratic in Q: recover it from three samples and solve.
            xs = ComplexF64[0, 1, 2]
            c = [x^j for x in xs, j in 0:2] \ [mc(x) for x in xs]
            disc = sqrt(c[2]^2 - 4c[3] * c[1])
            roots = [(-c[2] + disc) / (2c[3]), (-c[2] - disc) / (2c[3])]
            @test sort(roots ./ tauk[ref]; by=real) ≈ expected rtol = 1e-10
        end
    end

    @testset "Rigid E×B rotation shifts ω by n·Ω_E and leaves γ unchanged" begin
        d = ComplexF64[1.0+1.0im, 3.0+2.0im]
        tauk = [2.0e-4, 5.0e-4]
        nΩ = 700.0
        static = [surface_coupling(model, nothing, d[k]; scale=1.0, tauk=tauk[k]) for k in 1:2]
        rigid = [surface_coupling(model, nothing, d[k]; scale=1.0, tauk=tauk[k], q_shift=-tauk[k] * nΩ)
                 for k in 1:2]
        function physical_roots(scs)
            mc = multi_surface_coupling(scs, diagm(d); ref_idx=1, msing_max=2)
            xs = ComplexF64[0, 1, 2]
            c = [x^j for x in xs, j in 0:2] \ [mc(x) for x in xs]
            disc = sqrt(c[2]^2 - 4c[3] * c[1])
            return sort([(-c[2] + disc) / (2c[3]), (-c[2] - disc) / (2c[3])] ./ tauk[1]; by=real)
        end
        r0, r1 = physical_roots(static), physical_roots(rigid)
        @test real.(r1) ≈ real.(r0) .+ nΩ rtol = 1e-10
        @test imag.(r1) ≈ imag.(r0) rtol = 1e-10
    end

    @testset "omega_E_kHz and tauk_rescale parse and validate from TOML" begin
        ctrl = slayer_control_from_toml(Dict("omega_E_kHz" => [0, 3.0, -1.5]))
        @test ctrl.omega_E_kHz == [0.0, 3.0, -1.5]
        @test eltype(ctrl.omega_E_kHz) === Float64
        # Default stays empty so existing decks are untouched.
        @test isempty(slayer_control_from_toml(Dict{String,Any}()).omega_E_kHz)
        # A non-finite shift would silently poison every Q evaluation; the
        # validator rejects it (TOML cannot express NaN, so go through validate).
        @test_throws ArgumentError validate(SLAYERControl(; omega_E_kHz=[NaN]))

        # tauk_rescale round-trips as a Symbol and rejects unknown values.
        @test slayer_control_from_toml(Dict("tauk_rescale" => "legacy")).tauk_rescale === :legacy
        @test SLAYERControl().tauk_rescale === :direct
        @test_throws ArgumentError validate(SLAYERControl(; tauk_rescale=:sideways))
    end

    @testset "tauk_rescale flips the inter-surface Q normalization" begin
        sc1 = surface_coupling(model, nothing, 1.0 + 0im; scale=1.0, tauk=1.0)
        sc2 = surface_coupling(model, nothing, 2.0 + 0im; scale=1.0, tauk=2.0)
        dp = ComplexF64[1.0 0.0; 0.0 2.0]
        Q = 1.5 + 0.5im

        leg = multi_surface_coupling([sc1, sc2], dp; tauk_rescale=:legacy)
        dir = multi_surface_coupling([sc1, sc2], dp; tauk_rescale=:direct)
        @test leg.tauk_rescale === :legacy
        # :legacy divides by tauk_k, :direct multiplies by it.
        @test leg(Q) ≈ (1.0 - Q) * (2.0 - Q / 2)
        @test dir(Q) ≈ (1.0 - Q) * (2.0 - 2Q)
        @test_throws ArgumentError multi_surface_coupling([sc1, sc2], dp;
            tauk_rescale=:sideways)
    end
end
