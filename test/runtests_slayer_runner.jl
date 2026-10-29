@testset "Tearing: TearingProblem + TearingConfig + HDF5 output" begin
    using GeneralizedPerturbedEquilibrium
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium: GGJModel, SLAYERModel, TearingProblem, TearingConfig, solve
    using GeneralizedPerturbedEquilibrium.Tearing
    using GeneralizedPerturbedEquilibrium.Tearing.Dispersion
    using HDF5
    using TOML

    include("h5_metadata_check.jl")
    # ------- Helper: build a synthetic SLAYERParameters with full control
    function _mk_params(; rs=0.5, lu=1e7, tauk=1e-4,
        Q_e=-1.0, Q_i=0.5, m=2, n=1, ising=1,
        c_beta=0.1, D_norm=2.0)
        return SLAYERParameters(;
            tau=1.0, lu=lu, c_beta=c_beta, D_norm=D_norm,
            P_perp=20.0, P_tor=10.0,
            Q_e=Q_e, Q_i=Q_i,
            iota_e=Q_e == Q_i ? 0.0 : Q_e / (Q_e - Q_i),
            tauk=tauk, tau_r=1.0, delta_n=lu^(1 / 3) / rs,
            rs=rs, R0=1.7, bt=2.0, sval_r=1.0,
            eta=2.5e-8, d_beta=4e-3,
            m=m, n=n, ising=ising
        )
    end


    # run_slayer_from_inputs with the problem keywords (coupling and rotation) split from the scan settings.
    _PROBLEM_KEYS = (:coupling_mode, :msing_max, :omega_E_kHz, :critical_resonant_field)
    function _run(params, dp; model=SLAYERModel(), surfaces=(;), kwargs...)
        problem = TearingProblem(nothing; (k => v for (k, v) in kwargs if k in _PROBLEM_KEYS)...)
        config = TearingConfig(; (k => v for (k, v) in kwargs if !(k in _PROBLEM_KEYS))...)
        return run_slayer_from_inputs(model, params, dp, problem, config; surfaces...)
    end

    @testset "defaults + validation" begin
        p, c = TearingProblem(nothing), TearingConfig()
        @test p.coupling_mode === :uncoupled
        @test p.msing_max == 3
        @test c.scan_mode === :amr
        @test c.root_classification === :auto
        @test Tearing.validate(p, c) === nothing
        @test_throws ArgumentError Tearing.validate(TearingProblem(nothing; coupling_mode=:bogus), c)
        @test_throws ArgumentError Tearing.validate(TearingProblem(nothing; msing_max=0), c)
        @test_throws ArgumentError Tearing.validate(p, TearingConfig(; scan_mode=:bogus))
        @test_throws ArgumentError Tearing.validate(p, TearingConfig(; nre=1))
        @test_throws ArgumentError Tearing.validate(p, TearingConfig(; root_classification=:bogus))
        # Layer closures are checked when the model is built.
        @test_throws ArgumentError SLAYERModel(; D_c_type=:bogus)
        @test_throws ArgumentError SLAYERModel(; P_perp_model=:bogus)
    end

    @testset "deck: [Tearing], [Tearing.search] and [SLAYER] become the solve's arguments" begin
        deck = Dict{String,Any}(
            "Tearing" => Dict{String,Any}("inner_model" => "SLAYER", "coupling_mode" => "coupled", "msing_max" => 2,
                "search" => Dict{String,Any}("scan_mode" => "brute_force", "Q_re_range" => [-5.0, 5.0], "Q_im_range" => [-1.0, 3.0],
                    "nre" => 50, "nim" => 40, "amr_passes" => 3, "amr_max_cells" => 50_000, "pole_threshold" => 1e5,
                    "filter_above_poles" => false, "root_classification" => "winding")),
            "SLAYER" => Dict{String,Any}("D_c_type" => "fitzpatrick", "D_R" => 0.01, "P_perp_model" => "D_perp"))
        t = GeneralizedPerturbedEquilibrium.tearing_from_deck(deck)
        @test t.problem[:coupling_mode] === :coupled && t.problem[:msing_max] == 2
        c = t.config
        @test c.scan_mode === :brute_force && c.root_classification === :winding
        @test c.Q_re_range == (-5.0, 5.0) && c.Q_im_range == (-1.0, 3.0)
        @test c.nre == 50 && c.nim == 40 && c.amr_passes == 3 && c.amr_max_cells == 50_000
        @test c.pole_threshold == 1e5 && c.filter_above_poles == false
        @test t.model == SLAYERModel(; D_c_type=:fitzpatrick, D_R=0.01)
        # Unknown keys and the retired [SLAYER] keys raise.
        bad(section, key, value) = (d = deepcopy(deck); d[section][key] = value; d)
        @test_throws ArgumentError GeneralizedPerturbedEquilibrium.tearing_from_deck(bad("Tearing", "mistyped_key", 42))
        @test_throws ArgumentError GeneralizedPerturbedEquilibrium.slayer_model(bad("SLAYER", "mistyped_key", 42))
        @test_throws "moved to [Tearing.search]" GeneralizedPerturbedEquilibrium._stop_on_deprecated_deck_keys(bad("SLAYER", "nre", 4))
        @test_throws "chi_e_default" GeneralizedPerturbedEquilibrium._stop_on_deprecated_deck_keys(bad("SLAYER", "chi_perp_e", 1.0))
        @test_throws "inner_model" GeneralizedPerturbedEquilibrium.tearing_from_deck(Dict{String,Any}("Tearing" => Dict{String,Any}()))
    end

    @testset "layer diffusivities: file profiles, else the kinetic data's defaults" begin
        psi = collect(range(0.0, 1.0; length=5))
        data(; kw...) = GeneralizedPerturbedEquilibrium.Equilibrium.KineticProfileData(; psi, n_e=fill(5e19, 5), T_e=fill(1e3, 5), T_i=fill(8e2, 5), kw...)
        no_chi = (; chi_perp_e=nothing, chi_perp_i=nothing, chi_tor=nothing)
        # D_perp is the default, and it refuses to run without an ion diffusivity from the file or the data.
        @test SLAYERModel().P_perp_model === :D_perp
        @test_throws "needs the ion diffusivity" Tearing._slayer_chi_inputs(no_chi, SLAYERModel(), data(); warn=false)
        @test Tearing._slayer_chi_inputs(no_chi, SLAYERModel(), data(; chi_i_default=0.7); warn=false).chi_perp_i == 0.7
        @test Tearing._slayer_chi_inputs(no_chi, SLAYERModel(; P_perp_model=:chi_perp_e), data(); warn=false).chi_perp_i === nothing
        @test_throws "needs the confinement time" Tearing._slayer_chi_inputs(no_chi, SLAYERModel(; P_perp_model=:tau_E), data(); warn=false)
        @test_throws ArgumentError data(; chi_i_default=-0.5)
        @test_throws ArgumentError data(; tau_E=Inf)
    end

    @testset "TearingProblem solve forwards surfaces and Δ'" begin
        # A result with no singular surfaces short-circuits before any equilibrium access,
        # so a stand-in result is enough to pin the forwarding of surfaces / delta_prime.
        no_surfaces = (equil=nothing, surfaces=GeneralizedPerturbedEquilibrium.ForceFreeStates.SingType[], delta_prime=nothing)
        r = solve(TearingProblem(no_surfaces), SLAYERModel())
        @test isempty(r.params)
        @test r.enabled == false

        # With surfaces to analyse, the profiles must come from the equilibrium.
        bare = (equil=(; kinetic=nothing), surfaces=[nothing], delta_prime=nothing)
        @test_throws "carries no kinetic profiles" solve(TearingProblem(bare), SLAYERModel())
    end

    @testset "SLAYER view of the kinetic profiles" begin
        kin_path = joinpath(dirname(@__DIR__), "examples", "DIIID-like_ideal_example", "TkMkr_D3Dlike_Hmode_kinetic.h5")
        data = GeneralizedPerturbedEquilibrium.Equilibrium.read_kinetic_file(kin_path)
        view = Tearing._slayer_profiles(data, SLAYERModel())
        j = length(data.psi) ÷ 2
        @test view.profiles.n_e(data.psi[j]) ≈ data.n_e[j]
        @test view.profiles.omega_E(data.psi[j]) ≈ data.omega_E[j]
        @test view.chi_perp_e(data.psi[j]) ≈ data.chi_e[j]
        @test view.chi_perp_i(data.psi[j]) ≈ data.chi_i[j]
        @test view.has_omega_E
        # Without an E×B rotation profile the view records its absence and carries zero rotation.
        no_rotation = GeneralizedPerturbedEquilibrium.Equilibrium.KineticProfileData(; psi=data.psi, n_e=data.n_e, T_e=data.T_e, T_i=data.T_i)
        bare_view = Tearing._slayer_profiles(no_rotation, SLAYERModel())
        @test !bare_view.has_omega_E
        @test bare_view.profiles.omega_E(data.psi[j]) == 0.0
        @test_throws "missing required dataset 'T_i'" Tearing._slayer_profiles(
            GeneralizedPerturbedEquilibrium.Equilibrium.KineticProfileData(; psi=data.psi, n_e=data.n_e, T_e=data.T_e), SLAYERModel())
    end

    # Δ′ is unified across formalisms, so a Galerkin run feeds SLAYER exactly as a Riccati one
    # does: `result.delta_prime.matrix` is populated and already sized to the surface list, which
    # is the predicate the tearing solve uses to accept it over the per-surface diagonal stub.
    @testset "Galerkin-fed SLAYER: gal Δ' drives the coupled solve" begin
        mktempdir() do dir
            deck = joinpath(@__DIR__, "..", "examples", "LAR_ideal_match_test")
            for name in readdir(deck)
                cp(joinpath(deck, name), joinpath(dir, name))
            end
            # Preserve the two-surface producer/consumer path on a small Galerkin grid.
            deck_path = joinpath(dir, "gpec.toml")
            inputs = TOML.parsefile(deck_path)
            merge!(inputs["Equilibrium"], Dict("mpsi" => 32, "mtheta" => 64))
            merge!(inputs["TJAnalytic"], Dict("ma" => 32, "mtheta" => 32))
            merge!(inputs["ForceFreeStates"], Dict("delta_mlow" => 2, "delta_mhigh" => 2, "mtheta_vacuum" => 64))
            inputs["Galerkin"]["nx"] = 32
            open(io -> TOML.print(io, inputs), deck_path, "w")
            ffs = GeneralizedPerturbedEquilibrium.main([dir]).ffs
            @test ffs.integrator === :galerkin

            dpm = ffs.delta_prime.matrix
            @test size(dpm) == (length(ffs.surfaces), length(ffs.surfaces))
            @test size(dpm, 1) == 2

            # The gal Δ′ goes through SLAYER's coupled dispersion solve unmodified; the surface
            # parameters are synthetic because the LAR deck carries no kinetic profiles.
            params = [_mk_params(; rs=0.5, lu=1.0e7, tauk=1.0e-4, m=2, ising=1),
                _mk_params(; rs=0.6, lu=2.0e7, tauk=1.2e-4, m=3, ising=2)]
            r = _run(params, dpm; coupling_mode=:coupled, scan_mode=:brute_force,
                Q_re_range=(-1.0, 1.0), Q_im_range=(-0.5, 0.8), nre=20, nim=20, pole_threshold=1e5)
            @test r.enabled
            @test r.coupled_extraction isa GrowthRateResult
        end
    end

    @testset "run_slayer_from_inputs: E×B rotation enters only the coupled determinant" begin
        params = [_mk_params(; rs=0.5, lu=1.0e7, tauk=1.0e-4, m=2, n=1, ising=1),
            _mk_params(; rs=0.6, lu=2.0e7, tauk=1.2e-4, m=3, n=1, ising=2)]
        dpm = ComplexF64[-2.0 0.0; 0.0 -3.0]
        Ω_E = [2.0e3, -1.0e3]
        grid = (; scan_mode=:brute_force, Q_re_range=(-1.0, 1.0), Q_im_range=(-0.5, 0.8),
            nre=8, nim=8, pole_threshold=1e5)

        # Uncoupled layers are solved in their own plasma frame: no shift, even when supplied.
        r_unc = _run(params, dpm; coupling_mode=:uncoupled, grid..., surfaces=(; omega_E=Ω_E))
        @test r_unc.q_shift == [0.0, 0.0]
        # Their roots are reported in the lab frame: Q moves by τ_k·n·Ω_E, γ and the layer-frame extraction do not.
        r_unc0 = _run(params, dpm; coupling_mode=:uncoupled, grid...)
        @test isequal(r_unc.Q_root, r_unc0.Q_root .+ [1.0e-4 * Ω_E[1], 1.2e-4 * Ω_E[2]])
        @test r_unc.gamma == r_unc0.gamma
        @test isequal([gr.Q_root for gr in r_unc.per_surface_extraction], r_unc0.Q_root)
        for k in 1:2
            has_root = !isnan(r_unc0.Q_root[k])
            @test r_unc.omega[k] ≈ r_unc0.omega[k] + (has_root ? Ω_E[k] : 0.0)
        end
        # The resolved rotation is still reported in both modes.
        @test r_unc.omega_E == Ω_E

        # Coupled: the kinetic-file Ω_E Dopplers each surface by −τ_k·n·Ω_E.
        r_cpl = _run(params, dpm; coupling_mode=:coupled, grid..., surfaces=(; omega_E=Ω_E))
        @test r_cpl.q_shift ≈ [-1.0e-4 * 1 * Ω_E[1], -1.2e-4 * 1 * Ω_E[2]]
        @test r_cpl.omega_E == Ω_E
        # Without a rotation profile nothing is shifted, and the unknown Ω_E is reported as NaN.
        r_none = _run(params, dpm; coupling_mode=:coupled, grid...)
        @test r_none.q_shift == [0.0, 0.0]
        @test all(isnan, r_none.omega_E)
        # The mode sees n·Ω_E: the same rotation shifts an n = 2 surface twice as far.
        r_n2 = _run([_mk_params(; tauk=1.0e-4, m=3, n=2)], ComplexF64[-2.0;;]; coupling_mode=:coupled, grid..., surfaces=(; omega_E=[2.0e3]))
        @test r_n2.q_shift ≈ [-1.0e-4 * 2 * 2.0e3]
        @test_throws "rational surfaces were analysed" _run(params, dpm; coupling_mode=:coupled, grid..., surfaces=(; omega_E=[1.0]))

        # omega_E_kHz takes precedence over the file on the surfaces it lists; the rest keep the file value.
        r_ovr = _run(params, dpm; coupling_mode=:coupled, omega_E_kHz=Dict("2/1" => 1.0), grid..., surfaces=(; omega_E=Ω_E))
        @test r_ovr.omega_E ≈ [2π * 1e3, Ω_E[2]]
        @test r_ovr.q_shift ≈ [-1.0e-4 * 2π * 1e3, -1.2e-4 * Ω_E[2]]
        # Without a file profile, surfaces the override does not list stay unknown: NaN and no shift.
        r_part = _run(params, dpm; coupling_mode=:coupled, omega_E_kHz=Dict("2/1" => 1.0), grid...)
        @test r_part.omega_E[1] ≈ 2π * 1e3
        @test isnan(r_part.omega_E[2])
        @test r_part.q_shift ≈ [-1.0e-4 * 2π * 1e3, 0.0]
        # Uncoupled, the override moves only its surface's lab-frame root; an unknown Ω_E moves nothing.
        r_upart = _run(params, dpm; coupling_mode=:uncoupled, omega_E_kHz=Dict("2/1" => 1.0), grid...)
        @test isequal(r_upart.Q_root, r_unc0.Q_root .+ [1.0e-4 * 2π * 1e3, 0.0])
        @test r_upart.gamma == r_unc0.gamma
        @test_throws "matching no analysed surface" _run(params, dpm; coupling_mode=:coupled, omega_E_kHz=Dict("5/1" => 1.0), grid..., surfaces=(; omega_E=Ω_E))

        # A Doppler offset beyond the scan box is flagged rather than silently losing the root:
        # τ_ref·n·Ω_E = 1e-4 · 3e4 = 3 lies outside Re(Q) ∈ [-1, 1].
        @test_logs (:warn, r"outside Q_re_range") match_mode = :any _run(params, dpm; coupling_mode=:coupled, grid..., surfaces=(; omega_E=[3.0e4, 0.0]))

        # Rotation without a toroidal mode number would silently vanish, so it is an error.
        params_n0 = [_mk_params(; rs=0.5, lu=1.0e7, tauk=1.0e-4, m=2, n=0, ising=1)]
        @test_throws "needs the toroidal mode number" _run(params_n0, ComplexF64[-2.0;;]; coupling_mode=:coupled, grid..., surfaces=(; omega_E=[2.0e3]))
    end

    @testset "run_slayer_from_inputs: no surfaces is an empty result" begin
        r = _run(SLAYERParameters[], zeros(ComplexF64, 0, 0))
        @test r.enabled == false
        @test isempty(r.Q_root)
        @test isempty(r.params)
    end

    @testset "run_slayer_from_inputs: validation catches size mismatch" begin
        params = [_mk_params()]
        bad_dp = ComplexF64[0.0 0.0; 0.0 0.0]
        @test_throws ArgumentError _run(params, bad_dp)
    end

    @testset "delta_prime_to_rs_reference: ψ_N → r_s conversion" begin
        # Two surfaces with distinct K and α; the transform is
        # Δ̂_ij = K_i^(1/2+α_i) · Δ'_ij · K_j^(α_j−1/2).
        K1, alpha1 = 0.9, 0.54
        K2, alpha2 = 1.2, 0.60
        p1 = SLAYERParameters(; tau=1.0, lu=1e7, c_beta=0.1, D_norm=2.0,
            P_perp=20.0, P_tor=10.0, Q_e=-1.0, Q_i=0.5, iota_e=2 / 3,
            tauk=1e-4, tau_r=1.0, delta_n=1.0, rs=0.4, R0=1.7, bt=2.0,
            sval_r=1.0, eta=2.5e-8, d_beta=4e-3, m=2, n=1, ising=1,
            k_ref=K1, alpha_mercier=alpha1)
        p2 = SLAYERParameters(; tau=1.0, lu=1e7, c_beta=0.1, D_norm=2.0,
            P_perp=20.0, P_tor=10.0, Q_e=-1.0, Q_i=0.5, iota_e=2 / 3,
            tauk=1e-4, tau_r=1.0, delta_n=1.0, rs=0.5, R0=1.7, bt=2.0,
            sval_r=1.0, eta=2.5e-8, d_beta=4e-3, m=3, n=1, ising=2,
            k_ref=K2, alpha_mercier=alpha2)
        dp = ComplexF64[10.0+1im 2.0-0.5im; 3.0+0im 1.5+2im]
        out = Tearing.delta_prime_to_rs_reference(dp, [p1, p2])
        # Diagonal carries the scalar K^(2α)
        @test out[1, 1] ≈ K1^(2alpha1) * dp[1, 1]
        @test out[2, 2] ≈ K2^(2alpha2) * dp[2, 2]
        # Off-diagonals carry the split row/column factors
        @test out[1, 2] ≈ K1^(0.5 + alpha1) * K2^(alpha2 - 0.5) * dp[1, 2]
        @test out[2, 1] ≈ K2^(0.5 + alpha2) * K1^(alpha1 - 0.5) * dp[2, 1]
        # At the slab point α = 1/2 the diagonal factor is exactly K
        pslab = SLAYERParameters(; tau=1.0, lu=1e7, c_beta=0.1, D_norm=2.0,
            P_perp=20.0, P_tor=10.0, Q_e=-1.0, Q_i=0.5, iota_e=2 / 3,
            tauk=1e-4, tau_r=1.0, delta_n=1.0, rs=0.4, R0=1.7, bt=2.0,
            sval_r=1.0, eta=2.5e-8, d_beta=4e-3, m=2, n=1, ising=1,
            k_ref=0.8, alpha_mercier=0.5)
        dp1 = ComplexF64[5.0 + 0im;;]
        @test Tearing.delta_prime_to_rs_reference(dp1, [pslab])[1, 1] ≈ 0.8 * 5.0
        # Default parameters (k_ref = 1) give the identity regardless of α
        @test Tearing.delta_prime_to_rs_reference(dp, [_mk_params(), _mk_params()]) ≈ dp
    end

    coupled_result = Ref{TearingResult}()
    @testset "run_slayer_from_inputs: coupled mode finds known root" begin
        # Build a 2-surface problem with a known coupled root by construction.
        p1 = _mk_params(; rs=0.5, lu=1.0e7, tauk=1.0e-4, Q_e=-1.0, Q_i=0.5,
            m=2, ising=1)
        p2 = _mk_params(; rs=0.6, lu=2.0e7, tauk=1.2e-4, Q_e=-0.8, Q_i=0.4,
            m=3, ising=2)
        params = [p1, p2]

        model = SLAYERModel()
        # Pick a target Q and pin the diagonal Δ'_kk so det(M(Q_target)) = 0
        Q_target = 0.2 + 0.3im
        # Compute what each surface sees at Q_target (with per-surface
        # rescaling: surface 2 sees Q_target * tauk_1/tauk_2).
        Q_1 = Q_target * (p1.tauk / p1.tauk)         # = Q_target
        Q_2 = Q_target * (p1.tauk / p2.tauk)
        Δ1 = InnerLayer.solve_inner(model, p1, Q_1).tearing * p1.lu^(1 / 3)
        Δ2 = InnerLayer.solve_inner(model, p2, Q_2).tearing * p2.lu^(1 / 3)
        # Setting dp[k,k] = Δ_k at Q_target makes both diagonals of M vanish,
        # which makes det(M) = 0 at Q_target.
        dp = ComplexF64[Δ1 0.0; 0.0 Δ2]

        r = _run(params, dp; coupling_mode=:coupled, scan_mode=:brute_force, Q_re_range=(-1.0, 1.0), Q_im_range=(-0.5, 0.8),
            nre=80, nim=80, pole_threshold=1e5, store_scan=true, surfaces=(; rational_psi=[0.45, 0.72], rational_q=[2.0, 3.0]))
        coupled_result[] = r
        @test r.enabled
        @test length(r.Q_root) == 1          # single coupled eigenvalue
        @test abs(r.Q_root[1] - Q_target) < 2e-2       # grid-resolution limited
        @test r.coupled_extraction isa GrowthRateResult
        @test isempty(r.per_surface_extraction)
    end

    @testset "save_to_hdf5: round-trip structure" begin
        # Serialize the known-root integration result without repeating the scan.
        r = coupled_result[]

        @test r.model === SLAYERModel(; variant=:fitzpatrick)

        mktemp() do path, io
            close(io)
            h5open(path, "w") do f
                save_to_hdf5(f, r)
            end

            # Metadata contract must hold for Tearing/ too — the full-run schema test
            # only exercises an ideal deck, which writes no Tearing/ group.
            h5open(path, "r") do f
                viol = _collect_metadata_violations(f)
                isempty(viol) || @error "metadata contract violations in Tearing/" viol
                @test isempty(viol)
                @test attrs(f["Tearing"])["layer_model"] == "slayer"
                @test attrs(f["Tearing"])["inner_model"] == "slayer_fitzpatrick"
            end

            h5open(path, "r") do f
                g = f["Tearing"]
                @test haskey(g, "enabled") && read(g["enabled"]) == 1
                # Settings are not echoed — inputs live only under Input/ (the merged TOML).
                @test !haskey(g, "Settings")
                @test haskey(g, "PerSurface")
                # Surface identity: present when the caller supplied it, so Tearing
                # results plot against psi/q even when SLAYER analyzed a surface subset.
                @test read(g["PerSurface/rational_psi"]) == [0.45, 0.72]
                @test read(g["PerSurface/rational_q"]) == [2.0, 3.0]
                @test haskey(g, "Roots")
                @test haskey(g, "Diagnostics")
                @test haskey(g, "Scan")

                # Per-surface arrays have the right length
                @test length(read(g["PerSurface/rational_index"])) == 2
                @test read(g["PerSurface/rational_index"]) == [1, 2]
                @test read(g["PerSurface/lu"])[1] ≈ 1.0e7
                @test read(g["PerSurface/lu"])[2] ≈ 2.0e7

                # Roots arrays
                @test length(read(g["Roots/Q_root"])) == 1    # coupled
                @test length(read(g["Roots/omega"])) == 1

                # Layer-thickness diagnostic: one entry per surface, with
                # the physical thickness [m] and the drift scale.
                @test length(read(g["LayerWidths/delta_s_abs"])) == 2
                @test all(read(g["LayerWidths/delta_s_abs"]) .>= 0)
                @test haskey(g["LayerWidths"], "delta_s_over_d_beta")
                @test haskey(g["LayerWidths"], "d_beta")

                # Ragged diagnostics use flat+offsets encoding
                @test haskey(g["Diagnostics/ValidRoots"], "flat")
                @test haskey(g["Diagnostics/ValidRoots"], "offsets")
                @test !haskey(g["Diagnostics"], "ValidRootsWinding")    # threshold mode

                # Scan group present (store_scan=true)
                @test haskey(g, "Scan/Surface_1")
                @test read(g["Scan/Surface_1/kind"]) == "brute_force"
            end
        end
    end

    @testset "root_classification = :winding: uncoupled + coupled, HDF5" begin
        p1 = _mk_params(; rs=0.5, lu=1.0e7, tauk=1.0e-4, m=2, ising=1)
        p2 = _mk_params(; rs=0.6, lu=2.0e7, tauk=1.2e-4, m=3, ising=2)
        model = SLAYERModel(; variant=:fitzpatrick)
        # Diagonal Δ' placing surface 1's zero at Q1 and surface 2's at Q2 (each
        # in its own normalization), so the coupled determinant has two
        # distinct zeros. Both points are off the scan nodes and away from Q = 0.
        Q1, Q2 = -0.1213 + 0.0537im, 0.1517 + 0.1029im
        Δ1 = InnerLayer.solve_inner(model, p1, Q1).tearing * p1.lu^(1 / 3)
        Δ2 = InnerLayer.solve_inner(model, p2, Q2).tearing * p2.lu^(1 / 3)
        dp = ComplexF64[Δ1 0.0; 0.0 Δ2]
        for mode in (:uncoupled, :coupled)
            run(dp, cls; surfaces=(;)) = _run([p1, p2], dp; model, surfaces, scan_mode=:brute_force, coupling_mode=mode,
                Q_re_range=(-0.5, 0.5), Q_im_range=(-0.3, 0.3), nre=40, nim=40,
                root_classification=cls, pole_threshold_adaptive=true)   # adaptive: threshold-mode reference
            r = run(dp, :winding; surfaces=(; rational_psi=[0.45, 0.72], rational_q=[2.0, 3.0]))
            ref = run(dp, :threshold)
            exts = mode === :coupled ? [r.coupled_extraction] : r.per_surface_extraction
            @test all(gr -> gr.classification === :winding, exts)
            @test all(gr -> all(w -> w ≈ -1, gr.valid_roots_winding), exts)
            if mode === :uncoupled
                # Polished. The SLAYER residual is only smooth to |r| ~ 1e-2 within
                # 1e-4 of these zeros, which bounds any polish; the grid step is 0.026.
                @test abs(r.Q_root[1] - Q1) < 1e-3
                @test abs(r.Q_root[2] - Q2) < 1e-3
                @test maximum(abs.(r.Q_root .- ref.Q_root)) < 1e-3
            else
                # zeros at Q1 and Q2·τ_k1/τ_k2 in the reference normalization
                # (surface k sees Q·τ_kk/τ_k1); the least stable is reported,
                # unpolished since the coupled path does not polish
                @test length(r.coupled_extraction.valid_roots) == 2
                @test abs(r.Q_root[1] - Q2 * p1.tauk / p2.tauk) < 2e-2    # grid-resolution limited
                @test r.Q_root == ref.Q_root
            end
            if mode === :coupled
                # A Hermitian off-diagonal coupling moves the zeros off the diagonal entries' own; both classifications must
                # still agree on the reported root, and every winding root counts −1.
                dpc = ComplexF64[Δ1 0.3*sqrt(abs(Δ1 * Δ2)); 0.3*sqrt(abs(Δ1 * Δ2)) Δ2]
                rc = run(dpc, :winding; surfaces=(; rational_psi=[0.45, 0.72], rational_q=[2.0, 3.0]))
                refc = run(dpc, :threshold)
                @test !isempty(rc.coupled_extraction.valid_roots)
                @test all(w -> w ≈ -1, rc.coupled_extraction.valid_roots_winding)
                @test rc.Q_root == refc.Q_root
            end
            mktemp() do path, io
                close(io)
                h5open(path, "w") do f
                    save_to_hdf5(f, r)
                end
                h5open(path, "r") do f
                    viol = _collect_metadata_violations(f)
                    isempty(viol) || @error "metadata contract violations in Tearing/" viol
                    @test isempty(viol)
                    g = f["Tearing/Diagnostics"]
                    for k in ("ValidRootsWinding", "PolesWinding", "FilteredRootsWinding")
                        @test haskey(g, k)
                    end
                    @test read(g["ValidRootsWinding/offsets"]) == read(g["ValidRoots/offsets"])
                    @test all(w -> w ≈ -1, read(g["ValidRootsWinding/flat"]))
                end
            end
        end
    end

    @testset "GGJ per-surface writer: metadata contract" begin
        # The GGJ branch of _write_per_surface! is unreachable from the SLAYER
        # round-trip above; enforce its annotation-table rows on a synthetic write.
        params = [GGJParameters(; E=0.1, F=0.2, G=0.3, H=0.4, K=0.5, taua=1e-6, taur=1.0, ising=i) for i in 1:2]
        dp = ComplexF64[1.0 0.0; 0.0 2.0]
        mktemp() do path, io
            close(io)
            h5open(path, "w") do f
                g = HDF5.create_group(f, "Tearing")
                Tearing._write_per_surface!(g, params, dp)
                Tearing._annotate_tearing!(g)
            end
            h5open(path, "r") do f
                viol = _collect_metadata_violations(f)
                isempty(viol) || @error "metadata contract violations in GGJ PerSurface/" viol
                @test isempty(viol)
            end
        end
    end

    @testset "save_to_hdf5: an empty result still emits enabled=0" begin
        r = Tearing.empty_tearing_result(TearingProblem(nothing), TearingConfig(), SLAYERModel())
        mktemp() do path, io
            close(io)
            h5open(path, "w") do f
                save_to_hdf5(f, r)
            end
            h5open(path, "r") do f
                g = f["Tearing"]
                @test read(g["enabled"]) == 0
                @test !haskey(g, "PerSurface")    # no further groups
                @test !haskey(g, "Roots")
            end
        end
    end

end
