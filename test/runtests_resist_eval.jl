@testset "ResistEval: GGJ geometric coefficients + GGJ builder" begin
    using GeneralizedPerturbedEquilibrium
    using GeneralizedPerturbedEquilibrium.Equilibrium
    using GeneralizedPerturbedEquilibrium.ForceFreeStates
    using GeneralizedPerturbedEquilibrium.ForceFreeStates: SingType, ResistGeometry
    using GeneralizedPerturbedEquilibrium.Utilities
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using FastInterpolations
    using TOML

    # Load the bundled Solovev example equilibrium once for all tests.
    dir_path = joinpath(dirname(@__DIR__), "examples", "Solovev_ideal_example")
    inputs = TOML.parsefile(joinpath(dir_path, "gpec.toml"))
    eq_cfg = Equilibrium.EquilibriumConfig(inputs["Equilibrium"], dir_path)
    equil = Equilibrium.setup_equilibrium(eq_cfg)

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

    @testset "resist_geometry vs Mercier: D_I = E + F + H − ¼" begin
        # Run mercier_scan! to get the independent D_I·ψ on the radial grid,
        # interpolate to a few surface ψ values, and check against the
        # GGJ-coefficient reconstruction.
        npts = equil.profiles.npts
        locstab = zeros(Float64, npts, 3)
        ForceFreeStates.mercier_scan!(locstab, equil)
        di_psi_spline = cubic_interp(equil.profiles.xs, locstab[:, 1])

        dq = deriv_view(equil.profiles.q_spline, 1)
        for psi in (0.3, 0.5, 0.7)
            q1 = dq(psi)
            rg = ForceFreeStates.resist_geometry(equil, psi, q1)
            di_from_ggj = rg.E + rg.F + rg.H - 0.25

            # Mercier writes D_I·ψ to locstab[:,1]
            di_from_mercier = di_psi_spline(psi) / psi

            # Both methods compute D_I via different combinations of the
            # same theta integrals; agreement is set by the spline /
            # numerical-integration noise floor (relative tolerance 1e-3).
            @test abs(di_from_ggj - di_from_mercier) < 1e-3 * abs(di_from_mercier)
        end
    end

    @testset "resist_eval_all!: populates restype on every surface" begin
        # Build a couple of synthetic SingTypes, run the populator, verify
        # restype goes from nothing to ResistGeometry on each.
        dq = deriv_view(equil.profiles.q_spline, 1)
        s1 = SingType(; psifac=0.3, rho=sqrt(0.3), m=[2], n=[1],
            q=2.0, q1=dq(0.3),
            grri=zeros(Float64, 0, 0), grre=zeros(Float64, 0, 0),
            delta_prime=ComplexF64[],
            delta_prime_col=zeros(ComplexF64, 0, 0),
            ua_left=zeros(ComplexF64, 0, 0, 0),
            ua_right=zeros(ComplexF64, 0, 0, 0),
            psi_ua_left=0.0, psi_ua_right=0.0)
        s2 = SingType(; psifac=0.7, rho=sqrt(0.7), m=[3], n=[1],
            q=3.0, q1=dq(0.7),
            grri=zeros(Float64, 0, 0), grre=zeros(Float64, 0, 0),
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

    @testset "build_ggj_inputs: builds GGJParameters from sings + profiles" begin
        # Synthetic profiles
        psi_pts = collect(0.0:0.1:1.0)
        profiles = KineticProfiles(; psi=psi_pts,
            n_e=fill(5.0e19, length(psi_pts)),
            T_e=1000.0 .* (1.0 .- 0.7 .* psi_pts),
            T_i=1000.0 .* (1.0 .- 0.6 .* psi_pts),
            omega=fill(0.0, length(psi_pts)),
            omega_e=fill(1.0e4, length(psi_pts)),
            omega_i=fill(5.0e3, length(psi_pts)))

        dq = deriv_view(equil.profiles.q_spline, 1)
        s1 = SingType(; psifac=0.3, rho=sqrt(0.3), m=[2], n=[1],
            q=2.0, q1=dq(0.3),
            grri=zeros(Float64, 0, 0), grre=zeros(Float64, 0, 0),
            delta_prime=ComplexF64[],
            delta_prime_col=zeros(ComplexF64, 0, 0),
            ua_left=zeros(ComplexF64, 0, 0, 0),
            ua_right=zeros(ComplexF64, 0, 0, 0),
            psi_ua_left=0.0, psi_ua_right=0.0)
        intr = ForceFreeStates.ForceFreeStatesInternal(; sing=[s1], msing=1)
        ForceFreeStates.resist_eval_all!(intr, equil)

        gs = build_ggj_inputs(equil, intr.sing, profiles; mu_i=2.0, zeff=1.0)
        @test length(gs) == 1
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

    @testset "build_ggj_inputs: errors when restype not populated" begin
        # Need ≥4 points for the cubic spline
        psi_pts = collect(0.0:0.25:1.0)
        n = length(psi_pts)
        profiles = KineticProfiles(; psi=psi_pts,
            n_e=fill(5.0e19, n), T_e=fill(1000.0, n), T_i=fill(1000.0, n),
            omega=fill(0.0, n), omega_e=fill(1.0e4, n), omega_i=fill(5.0e3, n))

        s_unpop = SingType(; psifac=0.5, rho=sqrt(0.5), m=[2], n=[1],
            q=2.0, q1=1.0,
            grri=zeros(Float64, 0, 0), grre=zeros(Float64, 0, 0),
            delta_prime=ComplexF64[],
            delta_prime_col=zeros(ComplexF64, 0, 0),
            ua_left=zeros(ComplexF64, 0, 0, 0),
            ua_right=zeros(ComplexF64, 0, 0, 0),
            psi_ua_left=0.0, psi_ua_right=0.0)
        @test s_unpop.restype === nothing
        @test_throws ArgumentError build_ggj_inputs(equil, [s_unpop], profiles)
    end

    @testset "GGJ solve_inner runs on built parameters" begin
        psi_pts = collect(0.0:0.1:1.0)
        profiles = KineticProfiles(; psi=psi_pts,
            n_e=fill(5.0e19, length(psi_pts)),
            T_e=1000.0 .* (1.0 .- 0.7 .* psi_pts),
            T_i=fill(1000.0, length(psi_pts)),
            omega=fill(0.0, length(psi_pts)),
            omega_e=fill(0.0, length(psi_pts)),
            omega_i=fill(0.0, length(psi_pts)))

        dq = deriv_view(equil.profiles.q_spline, 1)
        s1 = SingType(; psifac=0.3, rho=sqrt(0.3), m=[2], n=[1],
            q=2.0, q1=dq(0.3),
            grri=zeros(Float64, 0, 0), grre=zeros(Float64, 0, 0),
            delta_prime=ComplexF64[],
            delta_prime_col=zeros(ComplexF64, 0, 0),
            ua_left=zeros(ComplexF64, 0, 0, 0),
            ua_right=zeros(ComplexF64, 0, 0, 0),
            psi_ua_left=0.0, psi_ua_right=0.0)
        intr = ForceFreeStates.ForceFreeStatesInternal(; sing=[s1], msing=1)
        ForceFreeStates.resist_eval_all!(intr, equil)
        gs = build_ggj_inputs(equil, intr.sing, profiles; mu_i=2.0)

        # Verify D_I < 0 so the GGJ shooting solver doesn't bail
        @test mercier_di(gs[1]) < 0

        Δ = solve_inner(GGJModel(; solver=:shooting), gs[1], 0.01 + 0.0im)
        @test Δ isa InnerLayerResponse
        @test isfinite(Δ.tearing)
        @test isfinite(Δ.interchange)
    end
end
