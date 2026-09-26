@testset "SLAYER LayerParameters" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.Utilities: MU_0, M_E, M_P, E_CHG, EPS_0
    using GeneralizedPerturbedEquilibrium.Utilities: SpitzerModel, SpitzerHarmModel,
        SauterNeoModel

    # Reference inputs: a simple deuterium plasma case suitable for
    # hand-checking the SLAYER params formulas.
    function _ref_kwargs(; dr_val=0.0, dc_type=:none)
        return (
            n_e=5.0e19, t_e=1000.0, t_i=1000.0,
            omega=0.0, omega_e=-1.0e4, omega_i=5.0e3,
            qval=2.0, sval_r=1.0, bt=2.0,
            rs=0.5, R0=1.7, mu_i=2.0, zeff=1.0,
            chi_perp=1.0, chi_tor=1.0,
            m=2, n=1,
            dr_val=dr_val, dgeo_val=0.5, dc_type=dc_type,
            ising=3
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
        # Physical opposite-drift inputs: ω_*e < 0 < ω_*i, so Q = −tauk·ω gives
        # Q_e > 0 > Q_i and iota_e = Q_e/(Q_e − Q_i) = 1e4/(1e4 + 5e3) = 2/3
        @test p.iota_e ≈ 2 / 3

        # Sign convention check (SLAYER layerinputs): Q = −tauk·ω
        @test p.Q_e == p.tauk * 1.0e4
        @test p.Q_i == -p.tauk * 5.0e3

        # Default resistivity closure is neoclassical (Sauter F_33). The
        # trapped-particle correction raises η above plain Spitzer at the
        # same lnΛ: η_neo = η_Sp / F_33 with F_33 ∈ (0,1).
        p_spitzer = slayer_parameters(; _ref_kwargs()...,
            resistivity_model=SpitzerModel(), lnLambda_form=:nrl)
        @test p.eta > p_spitzer.eta
        @test 1.1 < p.eta / p_spitzer.eta < 4.0    # banana-regime F_33 ≈ 0.3–0.9

        # Legacy Fitzpatrick/TJ path: the Spitzer-Härm σ_∥ closure agrees with
        # the old Wesson 1.65e-9 form to ~2.3% (the two are independent Spitzer
        # formulas), so this is a genuine cross-check, not a tautology.
        lnLamb_wesson = 24.0 + 3.0 * log(10.0) - 0.5 * log(5.0e19) + log(1000.0)
        eta_wesson = 1.65e-9 * lnLamb_wesson / (1000.0 / 1e3)^1.5
        p_legacy = slayer_parameters(; _ref_kwargs()...,
            resistivity_model=SpitzerHarmModel(), lnLambda_form=:wesson)
        @test p_legacy.eta ≈ eta_wesson rtol = 3e-2

        # τ_R = μ₀ r_s² / η holds for whichever closure was selected.
        @test p.tau_r ≈ MU_0 * p.rs^2 / p.eta rtol = 1e-12
        @test p_legacy.tau_r ≈ MU_0 * p_legacy.rs^2 / p_legacy.eta rtol = 1e-12

        # Mass density and Alfvén time (independent of conductivity).
        rho_expected = 2.0 * M_P * 5.0e19
        tau_h_expected = 1.7 * sqrt(MU_0 * rho_expected) / (1 * 1.0 * 2.0)
        # tauk = S^(1/3) · τ_H = (τ_R/τ_H)^(1/3)·τ_H = τ_R^(1/3)·τ_H^(2/3)
        @test p.tauk ≈ p.lu^(1 / 3) * tau_h_expected rtol = 1e-12
        @test p.tauk^3 / tau_h_expected^2 ≈ p.tau_r rtol = 1e-12

        # Lundquist number is large positive
        @test p.lu > 1e6
        @test p.lu < 1e9

        # Compressibility is in (0,1) for finite β
        @test 0.0 < p.c_beta < 1.0

        # Prandtl-like ratios are positive and equal here (chi_perp=chi_tor=1)
        @test p.P_perp ≈ p.P_tor
        @test p.P_perp > 0

        # D_norm = (d_β/r_s)·S^(1/3)·√ι_e — the electron share of the total
        # diamagnetic frequency, not the temperature ratio.
        D_norm_expected = (p.d_beta / p.rs) * p.lu^(1 / 3) * sqrt(p.iota_e)
        @test p.D_norm ≈ D_norm_expected rtol = 1e-12

        # delta_n = S^(1/3)/r_s
        @test p.delta_n ≈ p.lu^(1 / 3) / p.rs rtol = 1e-12
    end

    @testset "Test 1b: dc_tmp formulas activate when dr_val ≠ 0" begin
        # All four dc_type branches must produce finite, non-NaN values
        # and respect the signs/structure of the formulas in
        # the SLAYER params dc_tmp formulas.
        p_none = slayer_parameters(; _ref_kwargs(; dr_val=0.01, dc_type=:none)...)
        @test p_none.dc_tmp == 0.0   # :none ignores dr_val

        p_lar = slayer_parameters(; _ref_kwargs(; dr_val=0.01, dc_type=:lar)...)
        p_rf = slayer_parameters(; _ref_kwargs(; dr_val=0.01, dc_type=:rfitzp)...)
        p_tor = slayer_parameters(; _ref_kwargs(; dr_val=0.01, dc_type=:toroidal)...)

        @test isfinite(p_lar.dc_tmp)
        @test isfinite(p_rf.dc_tmp)
        @test isfinite(p_tor.dc_tmp)
        # dr_val > 0 with the (-dr_val) prefactor ⇒ negative dc_tmp for
        # :lar, :rfitzp, :toroidal branches.
        @test p_lar.dc_tmp < 0
        @test p_rf.dc_tmp < 0
        @test p_tor.dc_tmp < 0

        # Sign flips with sign of dr_val
        p_lar_neg = slayer_parameters(;
            _ref_kwargs(; dr_val=-0.01, dc_type=:lar)...)
        @test sign(p_lar_neg.dc_tmp) == -sign(p_lar.dc_tmp)

        # Reject unknown dc_type
        @test_throws ArgumentError slayer_parameters(;
            _ref_kwargs(; dr_val=0.01, dc_type=:bogus)...)
    end

    @testset "Test 1c: SLAYERParameters direct kwarg construction" begin
        # The @kwdef constructor must accept all required fields and
        # default the optional ones.
        p = SLAYERParameters(;
            tau=1.0, lu=1e7, c_beta=0.1, D_norm=2.0,
            P_perp=10.0, P_tor=10.0,
            Q_e=-1.0, Q_i=0.5, iota_e=2.0 / 3.0,
            tauk=1e-4, tau_r=10.0, delta_n=400.0,
            rs=0.5, R0=1.7, bt=2.0, sval_r=1.0,
            eta=2.5e-8, d_beta=4e-3
        )
        @test p.tau == 1.0
        @test p.dc_tmp == 0.0
        @test p.dc_type === :none
        @test p.dr_val == 0.0
        @test p.ising == 0
    end

    @testset "Test 1d: reject diamagnetic frequencies that leave D undefined" begin
        # iota_e = ω_*e/(ω_*e − ω_*i) must be finite and positive for D ∝ √iota_e.
        _drifts(ωe, ωi) = merge(_ref_kwargs(), (omega_e=ωe, omega_i=ωi))
        @test_throws ArgumentError slayer_parameters(; _drifts(-1.0e4, -1.0e4)...)   # degenerate: iota_e singular
        @test_throws ArgumentError slayer_parameters(; _drifts(-1.0e4, -2.0e4)...)   # same sign, |ω_*i| > |ω_*e|: iota_e = -1
        @test_throws ArgumentError slayer_parameters(; _drifts(0.0, 5.0e3)...)       # no electron drift: iota_e = 0
    end

    @testset "Test 1e: toroidal χ∥ closure algebra" begin
        # The :toroidal W_d balance takes its field-line geometry from (g_w, K∥) = (dgeo_val, kpar_val):
        # W_d = √8·(χ⊥/χ∥)^¼/g_w and the free-streaming χ∥ = 2v_te/(√π·K∥·W_d).
        using GeneralizedPerturbedEquilibrium.InnerLayer.SLAYER: _solve_dc_tmp
        common = (; dr_val=0.01, chi_perp=1.0, t_e=1000.0, zeff=1.0, rs=0.5, R0=1.7, sval_r=1.0, n_tor=1)
        vte = sqrt(2.0 * common.t_e * E_CHG / M_E)
        g_cyl = sqrt(common.rs / common.R0 * common.sval_r * common.n_tor)
        k_cyl = common.n_tor * common.sval_r / common.R0
        dc(dc_type, g, k; tau_ee=1e-6) = _solve_dc_tmp(; common..., dc_type=dc_type, dgeo_val=g, kpar_val=k, tau_ee=tau_ee)

        # Cylindrical geometry reproduces :lar (which is per metre, hence the r_s) and :rfitzp.
        @test dc(:toroidal, g_cyl, k_cyl) ≈ common.rs * dc(:lar, 0.0, nothing) rtol = 1e-12
        @test dc(:toroidal, g_cyl, k_cyl) ≈ dc(:rfitzp, 0.0, nothing) rtol = 1e-8
        # kpar_val = nothing falls back to the cylindrical K∥.
        @test dc(:toroidal, g_cyl, nothing) == dc(:toroidal, g_cyl, k_cyl)

        # Collisional limit: χ∥ is the Spitzer-Härm value, so K∥ drops out.
        tau_c = 1e-20
        chi_smfp = 1.581 * tau_c * vte^2 / (1.0 + 0.2535 * common.zeff)
        for (g, k) in ((0.3, 0.4), (0.3, 4.0), (0.6, 0.4))
            @test dc(:toroidal, g, k; tau_ee=tau_c) ≈ 0.5 * π^1.5 * (-common.dr_val) * (chi_smfp / common.chi_perp)^0.25 * g rtol = 1e-8
        end

        # Free-streaming limit: the fixed point solves in closed form,
        # χ∥^(3/4) = 2·v_te·g_w / (√(8π)·K∥·χ⊥^(1/4)), so Δ_crit ∝ g_w^(4/3)·K∥^(-1/3).
        tau_f = 1e20
        for (g, k) in ((0.3, 0.4), (0.3, 4.0), (0.6, 0.4))
            chi_fs = (2.0 * vte * g / (sqrt(8π) * k * common.chi_perp^0.25))^(4 / 3)
            @test dc(:toroidal, g, k; tau_ee=tau_f) ≈ 0.5 * π^1.5 * (-common.dr_val) * (chi_fs / common.chi_perp)^0.25 * g rtol = 1e-8
        end

        # dgeo_val = 0 keeps its meaning of no toroidal offset.
        @test dc(:toroidal, 0.0, k_cyl) == 0.0
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
            a = a0 * sqrt(psi)
            q = q0 * (1 + alpha * psi)
            dq_dpsi = q0 * alpha
            da_dpsi = a0 / (2 * sqrt(psi))
            expected = 2 * alpha * psi / (1 + alpha * psi)
            @test r_based_shear(a, q, dq_dpsi, da_dpsi) ≈ expected rtol = 1e-12
        end

        # Argument validation
        @test_throws ArgumentError r_based_shear(0.5, 2.0, 1.0, 0.0)
        @test_throws ArgumentError r_based_shear(0.5, 0.0, 1.0, 0.5)
    end

    @testset "Test 3: reverse-shear invariance" begin
        # The layer timescales and widths depend on |dq/dr|, not its sign: a negative-shear
        # surface must reduce to its positive-shear mirror, with only the recorded sval_r
        # diagnostic keeping the sign. dc_type=:lar with nonzero dr_val exercises the Wd
        # iteration and the critical-Δ square roots as well as tau_h.
        base = _ref_kwargs(; dr_val=-0.1, dc_type=:lar)
        pos = slayer_parameters(; base...)
        neg = slayer_parameters(; merge(base, (; sval_r=-1.0))...)

        # The sign survives where it is a diagnostic, not a magnitude.
        @test pos.sval_r == 1.0
        @test neg.sval_r == -1.0

        # Every normalized layer quantity is bit-identical: abs(-1.0) === 1.0,
        # so the whole downstream chain reproduces exactly.
        for f in (:tau, :lu, :c_beta, :D_norm, :P_perp, :P_tor, :Q_e, :Q_i,
            :iota_e, :tauk, :tau_r, :delta_n, :eta, :d_beta, :dc_tmp)
            @test getfield(neg, f) == getfield(pos, f)
        end

        # tau_h > 0 keeps the Lundquist number positive, so S^(1/3) is defined on reverse shear.
        @test neg.lu > 0
        @test neg.tauk > 0
    end
end
