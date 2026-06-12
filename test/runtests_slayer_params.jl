@testset "SLAYER LayerParameters" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.Utilities: MU_0, M_E, M_P, E_CHG, EPS_0

    # Reference inputs: a simple deuterium plasma case suitable for
    # hand-checking the SLAYER params formulas.
    function _ref_kwargs(; dr_val=0.0, dc_type=:none)
        return (
            n_e = 5.0e19, t_e = 1000.0, t_i = 1000.0,
            omega = 0.0, omega_e = 1.0e4, omega_i = 5.0e3,
            qval = 2.0, sval_r = 1.0, bt = 2.0,
            rs = 0.5, R0 = 1.7, mu_i = 2.0, zeff = 1.0,
            chi_perp = 1.0, chi_tor = 1.0,
            m = 2, n = 1,
            dr_val = dr_val, dgeo_val = 0.5, dc_type = dc_type,
            ising = 3,
        )
    end

    @testset "Test 1: round-trip from dimensional inputs" begin
        @info "Building SLAYERParameters from a reference deuterium case"
        p = slayer_parameters(; _ref_kwargs()...)

        # Identity / passthrough
        @test p.ising == 3
        @test p.m == 2
        @test p.n == 1
        @test p.rs == 0.5
        @test p.R0 == 1.7
        @test p.bt == 2.0
        @test p.sval_r == 1.0
        @test p.dc_tmp == 0.0   # dr_val == 0 ⇒ no offset
        @test p.dc_type === :none

        # Trivially exact ratios
        @test p.tau ≈ 1.0
        # Q_e = −tauk·1e4 = negative; Q_i = −tauk·5e3 = negative
        # Q_e − Q_i = −tauk·5e3 = Q_i (since Q_e = 2·Q_i) ⇒ iota_e = Q_e/Q_i = 2
        @test p.iota_e ≈ 2.0

        # Sign convention check (SLAYER layerinputs)
        @test p.Q_e == -p.tauk * 1.0e4
        @test p.Q_i == -p.tauk * 5.0e3    # SLAYER params convention: Q_i = −tauk·ω*i

        # Spitzer resistivity follows η = 1.65e-9·lnΛ/(T_e/1keV)^1.5
        # with lnΛ = 24 + 3 ln 10 − 0.5 ln n_e + ln T_e.
        lnLamb_expected = 24.0 + 3.0 * log(10.0) - 0.5 * log(5.0e19) + log(1000.0)
        eta_expected    = 1.65e-9 * lnLamb_expected / (1000.0 / 1e3)^1.5
        @test p.eta ≈ eta_expected rtol = 1e-12

        # Mass density and Alfvén time (independent of conductivity).
        rho_expected   = 2.0 * M_P * 5.0e19
        tau_h_expected = 1.7 * sqrt(MU_0 * rho_expected) / (1 * 1.0 * 2.0)
        # tauk = S^(1/3) · τ_H = (τ_R/τ_H)^(1/3)·τ_H = τ_R^(1/3)·τ_H^(2/3)
        @test p.tauk ≈ p.lu^(1/3) * tau_h_expected rtol = 1e-12
        @test p.tauk^3 / tau_h_expected^2 ≈ p.tau_r rtol = 1e-12

        # Lundquist number is large positive
        @test p.lu > 1e6
        @test p.lu < 1e9

        # Compressibility is in (0,1) for finite β
        @test 0.0 < p.c_beta < 1.0

        # Prandtl-like ratios are positive and equal here (chi_perp=chi_tor=1)
        @test p.P_perp ≈ p.P_tor
        @test p.P_perp > 0

        # D_norm = (d_β/r_s)·S^(1/3)·√(τ/(1+τ))
        D_norm_expected = (p.d_beta / p.rs) * p.lu^(1 / 3) * sqrt(p.tau / (1 + p.tau))
        @test p.D_norm ≈ D_norm_expected rtol = 1e-12

        # delta_n = S^(1/3)/r_s
        @test p.delta_n ≈ p.lu^(1 / 3) / p.rs rtol = 1e-12
    end

    @testset "Test 1b: dc_tmp formulas activate when dr_val ≠ 0" begin
        # All four dc_type branches must produce finite, non-NaN values
        # and respect the signs/structure of the formulas in
        # the SLAYER params dc_tmp formulas.
        p_none = slayer_parameters(; _ref_kwargs(dr_val=0.01, dc_type=:none)...)
        @test p_none.dc_tmp == 0.0   # :none ignores dr_val

        p_lar  = slayer_parameters(; _ref_kwargs(dr_val=0.01, dc_type=:lar)...)
        p_rf   = slayer_parameters(; _ref_kwargs(dr_val=0.01, dc_type=:rfitzp)...)
        p_tor  = slayer_parameters(; _ref_kwargs(dr_val=0.01, dc_type=:toroidal)...)

        @test isfinite(p_lar.dc_tmp)
        @test isfinite(p_rf.dc_tmp)
        @test isfinite(p_tor.dc_tmp)
        # dr_val > 0 with the (-dr_val) prefactor ⇒ negative dc_tmp for
        # :lar, :rfitzp, :toroidal branches.
        @test p_lar.dc_tmp < 0
        @test p_rf.dc_tmp  < 0
        @test p_tor.dc_tmp < 0

        # Sign flips with sign of dr_val
        p_lar_neg = slayer_parameters(;
            _ref_kwargs(dr_val=-0.01, dc_type=:lar)...)
        @test sign(p_lar_neg.dc_tmp) == -sign(p_lar.dc_tmp)

        # Reject unknown dc_type
        @test_throws ArgumentError slayer_parameters(;
            _ref_kwargs(dr_val=0.01, dc_type=:bogus)...)
    end

    @testset "Test 1c: SLAYERParameters direct kwarg construction" begin
        # The @kwdef constructor must accept all required fields and
        # default the optional ones.
        p = SLAYERParameters(;
            tau=1.0, lu=1e7, c_beta=0.1, D_norm=2.0,
            P_perp=10.0, P_tor=10.0,
            Q_e=-1.0, Q_i=0.5, iota_e=2.0/3.0,
            tauk=1e-4, tau_r=10.0, delta_n=400.0,
            rs=0.5, R0=1.7, bt=2.0, sval_r=1.0,
            eta=2.5e-8, d_beta=4e-3,
        )
        @test p.tau == 1.0
        @test p.dc_tmp == 0.0
        @test p.dc_type === :none
        @test p.dr_val == 0.0
        @test p.ising == 0
    end

    @testset "Test 2: r-based shear conversion" begin
        # Direct application of r_s · (dq/dψ) / (q · da/dψ).
        @test r_based_shear(0.5, 2.0, 4.0, 0.5) ≈ 2.0
        @test r_based_shear(1.0, 1.0, 1.0, 1.0) ≈ 1.0

        # Synthetic Solovev-like flux surface: a(ψ) = a₀·√ψ and q(ψ) =
        # q₀·(1 + α·ψ). Then dq/dψ = q₀·α, da/dψ = a₀/(2√ψ),
        # and the analytic r-based shear is
        #   s_r(ψ) = a(ψ)·(dq/dr)/q(ψ)
        #          = a₀√ψ · (dq/dψ)·(dψ/dr) / q(ψ)
        #          = a₀√ψ · q₀α · (2√ψ/a₀) / (q₀(1+α ψ))
        #          = 2αψ / (1+αψ).
        a0, q0, alpha = 0.6, 1.2, 1.5
        for psi in (0.1, 0.4, 0.7, 0.95)
            a       = a0 * sqrt(psi)
            q       = q0 * (1 + alpha * psi)
            dq_dpsi = q0 * alpha
            da_dpsi = a0 / (2 * sqrt(psi))
            expected = 2 * alpha * psi / (1 + alpha * psi)
            @test r_based_shear(a, q, dq_dpsi, da_dpsi) ≈ expected rtol = 1e-12
        end

        # Argument validation
        @test_throws ArgumentError r_based_shear(0.5, 2.0, 1.0, 0.0)
        @test_throws ArgumentError r_based_shear(0.5, 0.0, 1.0, 0.5)
    end
end
