@testset "Runner: Control + run_slayer + HDF5 output" begin
    using GeneralizedPerturbedEquilibrium
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.Dispersion
    using GeneralizedPerturbedEquilibrium.Runner
    using HDF5

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

    @testset "SLAYERControl defaults + validation" begin
        c = SLAYERControl()
        @test c.enabled == false
        @test c.inner_model === :slayer_fitzpatrick
        @test c.scan_mode === :amr
        @test c.coupling_mode === :uncoupled
        @test c.msing_max == 3

        # Validation catches bad symbols
        @test_throws ArgumentError Runner.validate(
            SLAYERControl(; inner_model=:bogus))
        @test_throws ArgumentError Runner.validate(
            SLAYERControl(; scan_mode=:bogus))
        @test_throws ArgumentError Runner.validate(
            SLAYERControl(; coupling_mode=:bogus))
        @test_throws ArgumentError Runner.validate(
            SLAYERControl(; dc_type=:bogus))
        @test_throws ArgumentError Runner.validate(
            SLAYERControl(; msing_max=0))
        @test_throws ArgumentError Runner.validate(
            SLAYERControl(; nre=1))
    end

    @testset "slayer_control_from_toml: nested sections flatten" begin
        section = Dict{String,Any}(
            "enabled" => true,
            "inner_model" => "slayer_fitzpatrick",
            "scan_mode" => "brute_force",
            "coupling_mode" => "coupled",
            "dc_type" => "rfitzp",
            "msing_max" => 2,
            "bt" => 1.8,
            "mu_i" => 2.0,
            "dr_val" => 0.01,
            "scan_grid" => Dict{String,Any}(
                "Q_re_range" => [-5.0, 5.0],
                "Q_im_range" => [-1.0, 3.0],
                "nre" => 50,
                "nim" => 40),
            "amr" => Dict{String,Any}(
                "passes" => 3,
                "max_cells" => 50_000),
            "growth_rate_filter" => Dict{String,Any}(
                "pole_threshold" => 1e5,
                "filter_above_poles" => false),
            "profile_file" => "profiles.h5"
        )
        c = slayer_control_from_toml(section)
        @test c.enabled
        @test c.inner_model === :slayer_fitzpatrick
        @test c.scan_mode === :brute_force
        @test c.coupling_mode === :coupled
        @test c.dc_type === :rfitzp
        @test c.msing_max == 2
        @test c.bt === 1.8
        @test c.dr_val == 0.01
        @test c.Q_re_range == (-5.0, 5.0)
        @test c.Q_im_range == (-1.0, 3.0)
        @test c.nre == 50
        @test c.nim == 40
        @test c.amr_passes == 3
        @test c.amr_max_cells == 50_000
        @test c.pole_threshold == 1e5
        @test c.filter_above_poles == false
        @test c.profile_file == "profiles.h5"

        # Unknown keys should raise
        bad = merge(section, Dict{String,Any}("mistyped_key" => 42))
        @test_throws ArgumentError slayer_control_from_toml(bad)
    end

    @testset "run_slayer: result-facing form forwards surfaces and Δ'" begin
        # A result with no singular surfaces short-circuits before any equilibrium access,
        # so a stand-in result is enough to pin the forwarding of surfaces / delta_prime.
        c = SLAYERControl(; enabled=true, profile_file="unused.h5")
        no_surfaces = (equil=nothing, surfaces=GeneralizedPerturbedEquilibrium.ForceFreeStates.SingType[],
            delta_prime=nothing)
        r = run_slayer(no_surfaces, c)
        @test isempty(r.params)

        # A disabled control never looks at the result at all.
        r_off = run_slayer(no_surfaces, SLAYERControl(; enabled=false, profile_file="unused.h5"))
        @test r_off.enabled == false
    end

    # Δ′ is unified across formalisms, so a Galerkin run feeds SLAYER exactly as a Riccati one
    # does: `result.delta_prime.matrix` is populated and already sized to the surface list, which
    # is the predicate `run_slayer` uses to accept it over the per-surface diagonal stub.
    @testset "Galerkin-fed SLAYER: gal Δ' drives the coupled solve" begin
        mktempdir() do dir
            deck = joinpath(@__DIR__, "..", "examples", "LAR_ideal_match_test")
            for name in readdir(deck)
                cp(joinpath(deck, name), joinpath(dir, name))
            end
            ffs = GeneralizedPerturbedEquilibrium.main([dir]).ffs
            @test ffs.integrator === :galerkin

            dpm = ffs.delta_prime.matrix
            @test size(dpm) == (length(ffs.surfaces), length(ffs.surfaces))
            @test size(dpm, 1) == 2

            # The gal Δ′ goes through SLAYER's coupled dispersion solve unmodified; the surface
            # parameters are synthetic because the LAR deck carries no kinetic profiles.
            params = [_mk_params(; rs=0.5, lu=1.0e7, tauk=1.0e-4, m=2, ising=1),
                _mk_params(; rs=0.6, lu=2.0e7, tauk=1.2e-4, m=3, ising=2)]
            c = SLAYERControl(; enabled=true, coupling_mode=:coupled, scan_mode=:brute_force,
                Q_re_range=(-1.0, 1.0), Q_im_range=(-0.5, 0.8), nre=20, nim=20, pole_threshold=1e5)
            r = run_slayer_from_inputs(params, dpm, c)
            @test r.enabled
            @test r.coupled_extraction isa GrowthRateResult
        end
    end

    @testset "run_slayer_from_inputs: disabled path is a no-op" begin
        c = SLAYERControl(; enabled=false)
        params = [_mk_params()]
        dp = ComplexF64[0.0 + 0im;;]                      # 1×1 matrix
        r = run_slayer_from_inputs(params, dp, c)
        @test r.enabled == false
        @test isempty(r.Q_root)
        @test isempty(r.params)
    end

    @testset "run_slayer_from_inputs: validation catches size mismatch" begin
        c = SLAYERControl(; enabled=true)
        params = [_mk_params()]
        bad_dp = ComplexF64[0.0 0.0; 0.0 0.0]
        @test_throws ArgumentError run_slayer_from_inputs(params, bad_dp, c)
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
        out = Runner.delta_prime_to_rs_reference(dp, [p1, p2])
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
        dp1 = ComplexF64[5.0+0im;;]
        @test Runner.delta_prime_to_rs_reference(dp1, [pslab])[1, 1] ≈ 0.8 * 5.0
        # Default parameters (k_ref = 1) give the identity regardless of α
        @test Runner.delta_prime_to_rs_reference(dp, [_mk_params(), _mk_params()]) ≈ dp
    end

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

        c = SLAYERControl(; enabled=true,
            inner_model=:slayer_fitzpatrick,
            scan_mode=:brute_force,
            coupling_mode=:coupled,
            Q_re_range=(-1.0, 1.0),
            Q_im_range=(-0.5, 0.8),
            nre=80, nim=80,
            pole_threshold=1e5)      # tuned for lu^(1/3) scale
        r = run_slayer_from_inputs(params, dp, c)
        @test r.enabled
        @test length(r.Q_root) == 1          # single coupled eigenvalue
        @test abs(r.Q_root[1] - Q_target) < 2e-2       # grid-resolution limited
        @test r.coupled_extraction isa GrowthRateResult
        @test isempty(r.per_surface_extraction)
    end

    @testset "write_slayer_hdf5!: round-trip structure" begin
        p1 = _mk_params(; rs=0.5, lu=1.0e7, tauk=1.0e-4, m=2, ising=1)
        p2 = _mk_params(; rs=0.6, lu=2.0e7, tauk=1.2e-4, m=3, ising=2)
        params = [p1, p2]

        # Diagonal dp, zero coupling → trivial root structure at Q_target=0
        Q_target = 0.0 + 0.0im
        model = SLAYERModel()
        Δ1 = InnerLayer.solve_inner(model, p1, Q_target).tearing * p1.lu^(1 / 3)
        Δ2 = InnerLayer.solve_inner(model, p2, Q_target).tearing * p2.lu^(1 / 3)
        dp = ComplexF64[Δ1 0.0; 0.0 Δ2]

        c = SLAYERControl(; enabled=true,
            scan_mode=:brute_force,
            coupling_mode=:coupled,
            Q_re_range=(-0.5, 0.5),
            Q_im_range=(-0.3, 0.3),
            nre=40, nim=40,
            pole_threshold=1e5,
            store_scan=true)
        r = run_slayer_from_inputs(params, dp, c;
            rational_psi=[0.45, 0.72], rational_q=[2.0, 3.0])

        mktemp() do path, io
            close(io)
            h5open(path, "w") do f
                write_slayer_hdf5!(f, r)
            end

            # Metadata contract must hold for Tearing/ too — the full-run schema test
            # only exercises an ideal deck, which writes no Tearing/ group.
            h5open(path, "r") do f
                viol = _collect_metadata_violations(f)
                isempty(viol) || @error "metadata contract violations in Tearing/" viol
                @test isempty(viol)
                @test attrs(f["Tearing"])["layer_model"] == "slayer"
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

                # Scan group present (store_scan=true)
                @test haskey(g, "Scan/Surface_1")
                @test read(g["Scan/Surface_1/kind"]) == "brute_force"
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
                Runner._write_per_surface!(g, params, dp)
                Runner._annotate_tearing!(g)
            end
            h5open(path, "r") do f
                viol = _collect_metadata_violations(f)
                isempty(viol) || @error "metadata contract violations in GGJ PerSurface/" viol
                @test isempty(viol)
            end
        end
    end

    @testset "write_slayer_hdf5!: disabled result still emits enabled=0" begin
        c = SLAYERControl(; enabled=false)
        r = empty_slayer_result(c)
        mktemp() do path, io
            close(io)
            h5open(path, "w") do f
                write_slayer_hdf5!(f, r)
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
