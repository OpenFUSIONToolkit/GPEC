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
    inputs = TOML.parsefile(joinpath(dir_path, "gpec.toml"))
    eq_cfg = Equilibrium.EquilibriumConfig(inputs["Equilibrium"], dir_path)
    sol_cfg = Equilibrium.SolovevConfig(inputs["SOL_INPUT"])
    equil = Equilibrium.setup_equilibrium(eq_cfg, sol_cfg)

    # Synthetic profiles (simple linear-in-ψ temperature decrease)
    psi_pts = collect(0.0:0.1:1.0)
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
        d_near_axis = surface_da_dpsi(equil, 1e-6)
        d_near_edge = surface_da_dpsi(equil, 1.0 - 1e-6)
        @test isfinite(d_near_axis) && d_near_axis > 0
        @test isfinite(d_near_edge) && d_near_edge > 0
    end

    @testset "build_slayer_inputs: returns correct per-surface data" begin
        sings = [_mk_sing(psi=0.3, q=2.0, q1=1.5, m=2, n=1),
            _mk_sing(psi=0.6, q=3.0, q1=2.5, m=3, n=1)]
        # dr_val=0.0 bypasses the build_slayer_inputs requirement that sing.restype be
        # pre-populated by ForceFreeStates.resist_eval_all! — the test sings here are
        # minimal stubs without restype, so we supply dr_val explicitly.
        # compute_omega_star=false makes Q_e/Q_i pass through directly from profiles.omega_e/i
        # rather than being recomputed from n_e/T_e/T_i gradients — required for the Q_e ==
        # -tauk·omega_e(ψ) identity check below.
        sl = build_slayer_inputs(equil, sings, profiles; bt=2.0, dr_val=0.0,
            compute_omega_star=false)

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

        # Q_e, Q_i follow the SLAYER layerinputs sign convention
        @test sl[1].Q_e == -sl[1].tauk * profiles.omega_e(0.3)
        @test sl[1].Q_i == -sl[1].tauk * profiles.omega_i(0.3)
    end

    @testset "build_slayer_inputs: bt defaults to the physical B_T, not a normalization" begin
        # `b0exp` is a normalization, not a field; passing it as `bt` ran the layer physics at the
        # wrong toroidal field. Asserted behaviourally rather than by grepping the source, so a
        # refactor cannot silently break the check and a reintroduction anywhere in the chain still
        # fails: tau_h = R0*sqrt(mu0*rho)/(n*sval_r*bt), so lu is linear in the field actually used
        # and the resolved bt is recoverable from the returned parameters.
        #
        # Note this fixture cannot exhibit the original bug: Solovev is normalized so that
        # b0exp == 1.0 and F(psi)/(2*pi*R0) == 1.0 to roundoff, which is exactly why the defect
        # survived. It shows up on a deck whose normalization differs from its field -- the
        # DIII-D-like EFIT deck has b0exp = 1.0 against a physical ~1.95 T. What is pinned here is
        # therefore the resolution rule itself, which is deck-independent.
        sings = [_mk_sing(psi=0.3, q=2.0, q1=1.5, m=2, n=1)]
        bt_phys = Float64(equil.profiles.F_spline(0.3)) / (2π * equil.ro)

        sl_default = build_slayer_inputs(equil, sings, profiles; dr_val=0.0, compute_omega_star=false)
        sl_at_phys = build_slayer_inputs(equil, sings, profiles; bt=bt_phys, dr_val=0.0, compute_omega_star=false)
        sl_double = build_slayer_inputs(equil, sings, profiles; bt=2 * bt_phys, dr_val=0.0, compute_omega_star=false)

        # Leaving bt unset must be identical to passing the physical field explicitly.
        @test sl_default[1].lu ≈ sl_at_phys[1].lu rtol = 1e-12

        # An explicit bt overrides, and lu tracks it linearly -- so the field that was actually
        # used is what the parameters encode, not merely some field.
        @test sl_double[1].lu / sl_default[1].lu ≈ 2.0 rtol = 1e-10
        @test sl_double[1].lu != sl_default[1].lu
    end

    @testset "build_slayer_inputs: chi_perp/chi_tor as scalars and callables" begin
        sings = [_mk_sing(psi=0.5, q=2.4, q1=1.2, m=2, n=1)]

        # Scalar (dr_val=0.0 bypasses the sing.restype requirement; see comment above)
        sl_s = build_slayer_inputs(equil, sings, profiles;
            bt=2.0, chi_perp=2.0, chi_tor=1.5, dr_val=0.0)
        # Callable with matching value
        chi_p(psi) = 2.0 + 0.0*psi
        chi_t(psi) = 1.5 + 0.0*psi
        sl_c = build_slayer_inputs(equil, sings, profiles;
            bt=2.0, chi_perp=chi_p, chi_tor=chi_t, dr_val=0.0)
        @test sl_s[1].P_perp ≈ sl_c[1].P_perp
        @test sl_s[1].P_tor ≈ sl_c[1].P_tor

        # Callable with ψ-dependence changes the result
        chi_p_var(psi) = 1.0 + 10.0 * psi                     # χ⊥(0.5) = 6.0 > 2.0
        sl_var = build_slayer_inputs(equil, sings, profiles;
            bt=2.0, chi_perp=chi_p_var, chi_tor=1.5, dr_val=0.0)
        # P_perp = τ_r · χ⊥ / r² grows with χ⊥, so the varying-χ case at
        # ψ=0.5 (χ⊥=6) gives a *larger* P_perp than the scalar χ⊥=2.
        @test sl_var[1].P_perp > sl_s[1].P_perp
        @test sl_var[1].P_perp ≈ sl_s[1].P_perp * 6.0 / 2.0 rtol = 1e-10
    end

    @testset "build_slayer_inputs: dc_type propagates and dr_val activates offset" begin
        sings = [_mk_sing(psi=0.5, q=2.4, q1=1.2, m=2, n=1)]

        # dc_type=:none and dr_val=0.0 → dc_tmp = 0 regardless of dr_val
        sl_none = build_slayer_inputs(equil, sings, profiles;
            bt=2.0, dc_type=:none, dr_val=0.0)
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

    @testset "build_slayer_inputs: omega_star carries the toroidal mode number" begin
        # ω_* = k_θ·v_* scales with the poloidal mode number; written in flux
        # coordinates the q is absorbed by dψ/dr = r·B_φ/q, leaving
        # ω_* = n·(dp/dψ)/(e·n_e). Resolve the same surface at n = 1 and n = 2:
        # Q/tauk = -ω_* must double, while iota_e = Q_e/(Q_e - Q_i) is a ratio
        # and must not move. Asserted through the returned parameters so the
        # check survives a refactor of where the factor is applied.
        s1 = [_mk_sing(psi=0.3, q=2.0, q1=1.5, m=2, n=1)]
        s2 = [_mk_sing(psi=0.3, q=2.0, q1=1.5, m=4, n=2)]
        sl1 = build_slayer_inputs(equil, s1, profiles; bt=2.0, dr_val=0.0)
        sl2 = build_slayer_inputs(equil, s2, profiles; bt=2.0, dr_val=0.0)

        @test sl2[1].Q_e / sl2[1].tauk ≈ 2 * (sl1[1].Q_e / sl1[1].tauk) rtol = 1e-12
        @test sl2[1].Q_i / sl2[1].tauk ≈ 2 * (sl1[1].Q_i / sl1[1].tauk) rtol = 1e-12
        @test sl2[1].iota_e ≈ sl1[1].iota_e rtol = 1e-12
    end

end
