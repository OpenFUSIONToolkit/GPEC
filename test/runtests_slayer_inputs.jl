@testset "SLAYER LayerInputs (build from equilibrium + profiles)" begin
    using GeneralizedPerturbedEquilibrium
    using GeneralizedPerturbedEquilibrium.Equilibrium
    using GeneralizedPerturbedEquilibrium.Utilities
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.ForceFreeStates: SingType
    using TOML

    # Load the Solovev analytic equilibrium shipped with the examples.
    # This exercise gets run once for all LayerInputs tests.
    dir_path = joinpath(dirname(@__DIR__), "examples", "Solovev_ideal_example")
    inputs   = TOML.parsefile(joinpath(dir_path, "gpec.toml"))
    eq_cfg   = Equilibrium.EquilibriumConfig(inputs["Equilibrium"], dir_path)
    equil    = Equilibrium.setup_equilibrium(eq_cfg)

    # Synthetic profiles (simple linear-in-ψ temperature decrease)
    psi_pts  = collect(0.0:0.1:1.0)
    profiles = KineticProfiles(; psi=psi_pts,
                                 n_e=fill(5.0e19, length(psi_pts)),
                                 T_e=1000.0 .* (1.0 .- 0.7 .* psi_pts),
                                 T_i=1000.0 .* (1.0 .- 0.6 .* psi_pts),
                                 omega=fill(0.0, length(psi_pts)),
                                 omega_e=fill(1.0e4, length(psi_pts)),
                                 omega_i=fill(5.0e3, length(psi_pts)))

    # Helper to build a minimal SingType without touching unused fields
    _mk_sing(; psi, q, q1, m, n, delta_prime=-10.0+0im) = SingType(
        psifac=psi, rho=sqrt(psi), m=[m], n=[n], q=q, q1=q1,
        grri=zeros(Float64, 0, 0), grre=zeros(Float64, 0, 0),
        delta_prime=ComplexF64[delta_prime],
        delta_prime_col=zeros(ComplexF64, 0, 0),
        ua_left=zeros(ComplexF64, 0, 0, 0),
        ua_right=zeros(ComplexF64, 0, 0, 0),
        psi_ua_left=0.0, psi_ua_right=0.0)

    @testset "surface_minor_radius: continuity + outboard > 0" begin
        # Minor radius grows monotonically with ψ (outboard midplane).
        r1 = surface_minor_radius(equil, 0.1)
        r2 = surface_minor_radius(equil, 0.5)
        r3 = surface_minor_radius(equil, 0.9)
        @test r1 < r2 < r3
        @test r1 > 0
    end

    @testset "surface_da_dpsi: FD agrees with numerical derivative" begin
        # Reference via a tighter FD
        for psi in (0.1, 0.4, 0.7)
            h_ref = 1e-4
            r_p = surface_minor_radius(equil, psi + h_ref)
            r_m = surface_minor_radius(equil, psi - h_ref)
            ref = (r_p - r_m) / (2 * h_ref)
            @test surface_da_dpsi(equil, psi) ≈ ref rtol = 1e-3
        end
    end

    @testset "surface_da_dpsi: one-sided near boundaries" begin
        # Near ψ=0 and ψ=1, the function falls back to one-sided FD and
        # should still produce a finite positive number (minor radius is
        # still increasing).
        d_near_axis  = surface_da_dpsi(equil, 1e-6)
        d_near_edge  = surface_da_dpsi(equil, 1.0 - 1e-6)
        @test isfinite(d_near_axis) && d_near_axis > 0
        @test isfinite(d_near_edge) && d_near_edge > 0
    end

    @testset "build_slayer_inputs: returns correct per-surface data" begin
        sings = [_mk_sing(psi=0.3, q=2.0, q1=1.5, m=2, n=1),
                 _mk_sing(psi=0.6, q=3.0, q1=2.5, m=3, n=1)]
        sl = build_slayer_inputs(equil, sings, profiles; bt=2.0)

        @test length(sl) == 2
        @test sl[1] isa SLAYERParameters
        @test sl[2] isa SLAYERParameters

        # ising traceability
        @test sl[1].ising == 1
        @test sl[2].ising == 2

        # Mode numbers flow through
        @test sl[1].m == 2 && sl[1].n == 1
        @test sl[2].m == 3 && sl[2].n == 1

        # Global geometry
        @test sl[1].R0 ≈ equil.ro
        @test sl[1].bt == 2.0

        # Minor radius and r-based shear recovered from the equilibrium
        rs1 = surface_minor_radius(equil, 0.3)
        da1 = surface_da_dpsi(equil, 0.3)
        @test sl[1].rs ≈ rs1
        @test sl[1].sval_r ≈ rs1 * 1.5 / (2.0 * da1)

        # Lundquist number and Q_e scale with surface parameters
        @test sl[1].lu != sl[2].lu
        @test sl[1].tauk != sl[2].tauk

        # Q_e, Q_i follow the layerinputs.f sign convention
        @test sl[1].Q_e == -sl[1].tauk * profiles.omega_e(0.3)
        @test sl[1].Q_i == -sl[1].tauk * profiles.omega_i(0.3)
    end

    @testset "build_slayer_inputs: chi_perp/chi_tor as scalars and callables" begin
        sings = [_mk_sing(psi=0.5, q=2.4, q1=1.2, m=2, n=1)]

        # Scalar
        sl_s = build_slayer_inputs(equil, sings, profiles;
                                    bt=2.0, chi_perp=2.0, chi_tor=1.5)
        # Callable with matching value
        chi_p(psi) = 2.0 + 0.0*psi
        chi_t(psi) = 1.5 + 0.0*psi
        sl_c = build_slayer_inputs(equil, sings, profiles;
                                    bt=2.0, chi_perp=chi_p, chi_tor=chi_t)
        @test sl_s[1].P_perp ≈ sl_c[1].P_perp
        @test sl_s[1].P_tor  ≈ sl_c[1].P_tor

        # Callable with ψ-dependence changes the result
        chi_p_var(psi) = 1.0 + 10.0 * psi                     # χ⊥(0.5) = 6.0 > 2.0
        sl_var = build_slayer_inputs(equil, sings, profiles;
                                      bt=2.0, chi_perp=chi_p_var, chi_tor=1.5)
        # P_perp = τ_r · χ⊥ / r² grows with χ⊥, so the varying-χ case at
        # ψ=0.5 (χ⊥=6) gives a *larger* P_perp than the scalar χ⊥=2.
        @test sl_var[1].P_perp > sl_s[1].P_perp
        @test sl_var[1].P_perp ≈ sl_s[1].P_perp * 6.0 / 2.0 rtol = 1e-10
    end

    @testset "build_slayer_inputs: dc_type propagates and dr_val activates offset" begin
        sings = [_mk_sing(psi=0.5, q=2.4, q1=1.2, m=2, n=1)]

        # dc_type=:none and dr_val=0.0 → dc_tmp = 0 regardless of dr_val
        sl_none = build_slayer_inputs(equil, sings, profiles;
                                       bt=2.0, dc_type=:none)
        @test sl_none[1].dc_tmp == 0.0

        # dc_type=:rfitzp with dr_val = 0 still gives zero
        sl_rf0 = build_slayer_inputs(equil, sings, profiles;
                                      bt=2.0, dc_type=:rfitzp, dr_val=0.0)
        @test sl_rf0[1].dc_tmp == 0.0

        # dc_type=:rfitzp with dr_val > 0 → nonzero negative offset
        sl_rf = build_slayer_inputs(equil, sings, profiles;
                                     bt=2.0, dc_type=:rfitzp, dr_val=0.01)
        @test sl_rf[1].dc_tmp < 0
        @test isfinite(sl_rf[1].dc_tmp)
    end

    @testset "build_slayer_inputs: empty sings returns empty vector" begin
        sl = build_slayer_inputs(equil, SingType[], profiles; bt=2.0)
        @test sl isa Vector{SLAYERParameters}
        @test isempty(sl)
    end
end
