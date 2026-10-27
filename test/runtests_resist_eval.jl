@testset "ResistEval: GGJ geometric coefficients + GGJ builder" begin
    using GeneralizedPerturbedEquilibrium
    using GeneralizedPerturbedEquilibrium.Equilibrium
    using GeneralizedPerturbedEquilibrium.ForceFreeStates
    using GeneralizedPerturbedEquilibrium.ForceFreeStates: SingType, ResistGeometry
    using GeneralizedPerturbedEquilibrium.LocalStability
    using GeneralizedPerturbedEquilibrium.Utilities
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.ForceFreeStates: ggj_parameters
    using FastInterpolations
    using TOML

    # Load the bundled Solovev example equilibrium once for all tests.
    dir_path = joinpath(dirname(@__DIR__), "examples", "Solovev_ideal_example")
    inputs = TOML.parsefile(joinpath(dir_path, "gpec.toml"))
    eq_cfg = Equilibrium.EquilibriumConfig(inputs["Equilibrium"], dir_path)
    sol_cfg = Equilibrium.SolovevConfig(inputs["SOL_INPUT"])
    equil = Equilibrium.PlasmaEquilibrium(eq_cfg, sol_cfg)

    # Spitzer-like η at ~1 keV and deuterium ρ at n_e = 5e19 m⁻³.
    eta, rho = 3e-8, 2 * Utilities.PhysicalConstants.M_P * 5e19

    @testset "resist_geometry: returns finite values with expected signs" begin
        # Pick a few interior surfaces; compute q1 from the equilibrium
        dq = deriv_view(equil.profiles.q_spline, 1)
        for psi in (0.2, 0.5, 0.8)
            q1 = dq(psi)
            rg = ForceFreeStates.resist_geometry(equil, psi, q1)

            @test rg isa ResistGeometry
            for f in (rg.E, rg.F, rg.G, rg.H, rg.K, rg.M)
                @test isfinite(f)
            end
            # Geometric averages are positive
            @test rg.avg_bsq_over_dpsisq > 0
            @test rg.avg_bsq > 0
            # Mass factor M > 0 (denominator in G and K)
            @test rg.M > 0
            # Pressure is positive on this Solovev equilibrium
            @test rg.p_local > 0
            @test rg.v1_local > 0
        end
    end

    @testset "resist_geometry vs ballooning D_I: D_I = E + F + H − ¼" begin
        # Independent D_I reference from the ballooning coefficient system:
        # det(d0bar) is the Mercier interchange D_I that the local-stability
        # scan stores in locstab[:,1]. Build it on the radial grid, interpolate
        # to a few surface ψ values, and check against the GGJ reconstruction.
        xs = equil.profiles.xs
        di_ref = Float64[LocalStability.prepare_ballooning_coefficients(i, equil).di for i in eachindex(xs)]
        di_spline = cubic_interp(xs, di_ref)

        dq = deriv_view(equil.profiles.q_spline, 1)
        for psi in (0.3, 0.5, 0.7)
            q1 = dq(psi)
            rg = ForceFreeStates.resist_geometry(equil, psi, q1)
            di_from_ggj = rg.E + rg.F + rg.H - 0.25

            # det(d0bar) is D_I directly (no ψ factor at the coefficient level).
            di_from_ballooning = di_spline(psi)

            # The ballooning det(d0bar) route is an INDEPENDENT D_I evaluation
            # (not the surface-average θ-integrals that build E,F,H), so the two
            # agree only to the few-percent level set by each route's separate
            # discretization — looser than the old same-integral cross-check.
            @test abs(di_from_ggj - di_from_ballooning) < 8e-2 * abs(di_from_ballooning)
        end
    end

    @testset "resist_eval_all!: populates restype on every surface" begin
        # Build a couple of synthetic SingTypes, run the populator, verify
        # restype goes from nothing to ResistGeometry on each.
        dq = deriv_view(equil.profiles.q_spline, 1)
        s1 = SingType(; psifac=0.3, rho=sqrt(0.3), m=[2], n=[1],
            q=2.0, q1=dq(0.3),
            delta_prime=ComplexF64[],
            delta_prime_col=zeros(ComplexF64, 0, 0),
            ua_left=zeros(ComplexF64, 0, 0, 0),
            ua_right=zeros(ComplexF64, 0, 0, 0),
            psi_ua_left=0.0, psi_ua_right=0.0)
        s2 = SingType(; psifac=0.7, rho=sqrt(0.7), m=[3], n=[1],
            q=3.0, q1=dq(0.7),
            delta_prime=ComplexF64[],
            delta_prime_col=zeros(ComplexF64, 0, 0),
            ua_left=zeros(ComplexF64, 0, 0, 0),
            ua_right=zeros(ComplexF64, 0, 0, 0),
            psi_ua_left=0.0, psi_ua_right=0.0)

        @test s1.restype === nothing
        @test s2.restype === nothing

        intr = ForceFreeStates.ForceFreeStatesInternal(; sing=[s1, s2], msing=2)
        ForceFreeStates.resist_eval_all!(intr, equil)

        @test intr.sing[1].restype isa ResistGeometry
        @test intr.sing[2].restype isa ResistGeometry
        # Idempotent — second call shouldn't recompute (already non-nothing)
        rg_first = intr.sing[1].restype
        ForceFreeStates.resist_eval_all!(intr, equil)
        @test intr.sing[1].restype === rg_first
    end

    @testset "ggj_parameters: builds GGJParameters from a surface + η, ρ" begin
        dq = deriv_view(equil.profiles.q_spline, 1)
        s1 = SingType(; psifac=0.3, rho=sqrt(0.3), m=[2], n=[1],
            q=2.0, q1=dq(0.3),
            delta_prime=ComplexF64[],
            delta_prime_col=zeros(ComplexF64, 0, 0),
            ua_left=zeros(ComplexF64, 0, 0, 0),
            ua_right=zeros(ComplexF64, 0, 0, 0),
            psi_ua_left=0.0, psi_ua_right=0.0)
        intr = ForceFreeStates.ForceFreeStatesInternal(; sing=[s1], msing=1)
        ForceFreeStates.resist_eval_all!(intr, equil)

        gs = [ggj_parameters(intr.sing[1], equil; eta=eta, rho=rho, ising=1)]
        @test gs[1] isa GGJParameters

        # Geometric coefficients flow through unchanged from restype
        rg = intr.sing[1].restype
        @test gs[1].E ≈ rg.E
        @test gs[1].F ≈ rg.F
        @test gs[1].G ≈ rg.G
        @test gs[1].H ≈ rg.H
        @test gs[1].K ≈ rg.K
        @test gs[1].M ≈ rg.M

        # Timescales are positive and physical
        @test gs[1].taua > 0
        @test gs[1].taur > 0
        @test gs[1].taur > gs[1].taua    # resistive ≫ Alfvén for any tokamak
        @test gs[1].taur / gs[1].taua > 1e3   # Lundquist S well into resistive regime

        # ising traceability
        @test gs[1].ising == 1
    end

    @testset "ggj_parameters: errors when restype not populated" begin
        s_unpop = SingType(; psifac=0.5, rho=sqrt(0.5), m=[2], n=[1],
            q=2.0, q1=1.0,
            delta_prime=ComplexF64[],
            delta_prime_col=zeros(ComplexF64, 0, 0),
            ua_left=zeros(ComplexF64, 0, 0, 0),
            ua_right=zeros(ComplexF64, 0, 0, 0),
            psi_ua_left=0.0, psi_ua_right=0.0)
        @test s_unpop.restype === nothing
        @test_throws ArgumentError ggj_parameters(s_unpop, equil; eta=1e-7, rho=1e-7)
    end

    @testset "GGJ solve_inner runs on built parameters" begin
        dq = deriv_view(equil.profiles.q_spline, 1)
        s1 = SingType(; psifac=0.3, rho=sqrt(0.3), m=[2], n=[1],
            q=2.0, q1=dq(0.3),
            delta_prime=ComplexF64[],
            delta_prime_col=zeros(ComplexF64, 0, 0),
            ua_left=zeros(ComplexF64, 0, 0, 0),
            ua_right=zeros(ComplexF64, 0, 0, 0),
            psi_ua_left=0.0, psi_ua_right=0.0)
        intr = ForceFreeStates.ForceFreeStatesInternal(; sing=[s1], msing=1)
        ForceFreeStates.resist_eval_all!(intr, equil)
        gs = [ggj_parameters(intr.sing[1], equil; eta=eta, rho=rho, ising=1)]

        # Verify D_I < 0 so the GGJ shooting solver doesn't bail
        @test mercier_di(gs[1]) < 0

        Δ = solve_inner(GGJModel(; solver=:shooting), gs[1], 0.01 + 0.0im)
        @test Δ isa InnerLayerResponse
        @test isfinite(Δ.tearing)
        @test isfinite(Δ.interchange)
    end
end
