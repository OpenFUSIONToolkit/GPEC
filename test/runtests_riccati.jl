using LinearAlgebra
using TOML

@testset "Riccati Integration Tests" begin

    @testset "renormalize_riccati_inplace!" begin
        N = 4
        # Build a random (U₁, U₂) pair and verify renorm gives S = U₁·U₂⁻¹ with U₂_new = I
        rng = [1.0+0.5im  0.2im    0.1      0.3im;
               0.0        1.2+0.1im 0.0im   0.2;
               0.1+0.1im  0.0      0.9+0.3im 0.1im;
               0.0im      0.2      0.0      1.1+0.2im]
        U1 = rng .+ 0.5*I(N)
        U2 = 0.5*rng .+ I(N)  # near-identity to ensure invertibility

        u = zeros(ComplexF64, N, N, 2)
        u[:, :, 1] .= U1
        u[:, :, 2] .= U2

        S_expected = U1 / U2  # = U₁ · U₂⁻¹

        JPEC.ForceFreeStates.renormalize_riccati_inplace!(u, N)

        @test u[:, :, 2] ≈ I(N)
        @test u[:, :, 1] ≈ S_expected  rtol=1e-12
    end

    @testset "renormalize_riccati_inplace! idempotent" begin
        N = 3
        # If U₂ = I already, renorm should leave u unchanged
        S = [1.0+0.5im  0.2im    0.1;
             0.0im      1.2+0.1im 0.0;
             0.1+0.1im  0.0      0.9+0.3im]
        u = zeros(ComplexF64, N, N, 2)
        u[:, :, 1] .= S
        u[:, :, 2] .= I(N)

        JPEC.ForceFreeStates.renormalize_riccati_inplace!(u, N)

        @test u[:, :, 2] ≈ I(N)
        @test u[:, :, 1] ≈ S  rtol=1e-12
    end

    @testset "renormalize_riccati! (OdeState)" begin
        N = 3
        rng = [1.0+0.5im  0.2im    0.1;
               0.0im      1.2+0.1im 0.0;
               0.1+0.1im  0.0      0.9+0.3im]
        U1 = rng .+ 0.5*I(N)
        U2 = 0.2*rng .+ I(N)

        odet = JPEC.ForceFreeStates.OdeState(N, 10, 5, 1)
        odet.u[:, :, 1] .= U1
        odet.u[:, :, 2] .= U2

        S_expected = U1 / U2
        intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; mpert=N, numpert_total=N)

        JPEC.ForceFreeStates.renormalize_riccati!(odet, intr)

        @test odet.u[:, :, 2] ≈ I(N)
        @test odet.u[:, :, 1] ≈ S_expected  rtol=1e-12
    end

    @testset "Riccati integration matches standard ODE — Solovev example" begin
        # Run both standard and Riccati integrations on the Solovev regression test.
        # The energy eigenvalue et[1] should match to within 1%.
        ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")

        function run_solovev(use_riccati)
            inputs = TOML.parsefile(joinpath(ex, "jpec.toml"))
            inputs["ForceFreeStates"]["verbose"] = false
            inputs["ForceFreeStates"]["use_riccati"] = use_riccati
            intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
            ctrl = JPEC.ForceFreeStates.ForceFreeStatesControl(;
                (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
            eq_config = JPEC.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
            equil = JPEC.Equilibrium.setup_equilibrium(eq_config)
            intr.wall_settings = JPEC.Vacuum.WallShapeSettings(;
                (Symbol(k) => v for (k, v) in inputs["Wall"])...)
            JPEC.ForceFreeStates.sing_lim!(intr, ctrl, equil)
            intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
            JPEC.ForceFreeStates.sing_find!(intr, equil)
            intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
            intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
            intr.mpert = intr.mhigh - intr.mlow + 1
            intr.mband = intr.mpert - 1
            intr.numpert_total = intr.mpert * intr.npert
            metric = JPEC.ForceFreeStates.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)
            ffit = JPEC.ForceFreeStates.make_matrix(equil, intr, metric)
            if use_riccati
                odet = JPEC.ForceFreeStates.riccati_eulerlagrange_integration(ctrl, equil, ffit, intr)
            else
                odet = JPEC.ForceFreeStates.eulerlagrange_integration(ctrl, equil, ffit, intr)
            end
            vac = JPEC.ForceFreeStates.free_run!(odet, ctrl, equil, ffit, intr)
            return real(vac.et[1]), odet.step
        end

        et_std, steps_std = run_solovev(false)
        et_ric, steps_ric = run_solovev(true)

        # Energy eigenvalue matches to 1%
        @test isapprox(et_ric, et_std; rtol=0.01)

        # Riccati uses no more than 2x as many steps as standard
        @test steps_ric <= 2 * steps_std
    end

    @testset "Riccati end state has U₂ ≈ I" begin
        # After riccati_eulerlagrange_integration, odet.u[:,:,2] should be identity
        # (canonical Riccati convention after final renorm)
        ex = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
        inputs = TOML.parsefile(joinpath(ex, "jpec.toml"))
        inputs["ForceFreeStates"]["verbose"] = false
        inputs["ForceFreeStates"]["use_riccati"] = true
        intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; dir_path=ex)
        ctrl = JPEC.ForceFreeStates.ForceFreeStatesControl(;
            (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
        eq_config = JPEC.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], ex)
        equil = JPEC.Equilibrium.setup_equilibrium(eq_config)
        intr.wall_settings = JPEC.Vacuum.WallShapeSettings(;
            (Symbol(k) => v for (k, v) in inputs["Wall"])...)
        JPEC.ForceFreeStates.sing_lim!(intr, ctrl, equil)
        intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
        JPEC.ForceFreeStates.sing_find!(intr, equil)
        intr.mlow = min(intr.nlow * equil.params.qmin, 0) - 4 - ctrl.delta_mlow
        intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
        intr.mpert = intr.mhigh - intr.mlow + 1
        intr.mband = intr.mpert - 1
        intr.numpert_total = intr.mpert * intr.npert
        metric = JPEC.ForceFreeStates.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)
        ffit = JPEC.ForceFreeStates.make_matrix(equil, intr, metric)

        odet = JPEC.ForceFreeStates.riccati_eulerlagrange_integration(ctrl, equil, ffit, intr)

        N = intr.numpert_total
        @test odet.u[:, :, 2] ≈ I(N)  rtol=1e-10
    end
end
