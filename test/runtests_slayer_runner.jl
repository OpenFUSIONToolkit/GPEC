@testset "Runner: Control + run_slayer + HDF5 output" begin
    using GeneralizedPerturbedEquilibrium
    using GeneralizedPerturbedEquilibrium.InnerLayer
    using GeneralizedPerturbedEquilibrium.Dispersion
    using GeneralizedPerturbedEquilibrium.Runner
    using HDF5

    # ------- Helper: build a synthetic SLAYERParameters with full control
    function _mk_params(; rs=0.5, lu=1e7, tauk=1e-4,
                         Q_e=-1.0, Q_i=0.5, m=2, n=1, ising=1,
                         c_beta=0.1, D_norm=2.0)
        return SLAYERParameters(
            tau=1.0, lu=lu, c_beta=c_beta, D_norm=D_norm,
            P_perp=20.0, P_tor=10.0,
            Q_e=Q_e, Q_i=Q_i,
            iota_e = Q_e == Q_i ? 0.0 : Q_e/(Q_e - Q_i),
            tauk=tauk, tau_r=1.0, delta_n=lu^(1/3)/rs,
            rs=rs, R0=1.7, bt=2.0, sval_r=1.0,
            eta=2.5e-8, d_beta=4e-3,
            m=m, n=n, ising=ising,
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
            "enabled"       => true,
            "inner_model"   => "slayer_fitzpatrick",
            "scan_mode"     => "brute_force",
            "coupling_mode" => "coupled",
            "dc_type"       => "rfitzp",
            "msing_max"     => 2,
            "bt"            => 1.8,
            "mu_i"          => 2.0,
            "dr_val"        => 0.01,
            "scan_grid" => Dict{String,Any}(
                "Q_re_range" => [-5.0, 5.0],
                "Q_im_range" => [-1.0, 3.0],
                "nre"        => 50,
                "nim"        => 40),
            "amr" => Dict{String,Any}(
                "passes"     => 3,
                "max_cells"  => 50_000),
            "growth_rate_filter" => Dict{String,Any}(
                "pole_threshold"     => 1e5,
                "filter_above_poles" => false),
            "profile_source" => "inline",
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

        # Unknown keys should raise
        bad = merge(section, Dict{String,Any}("mistyped_key" => 42))
        @test_throws ArgumentError slayer_control_from_toml(bad)
    end

    @testset "run_slayer_from_inputs: disabled path is a no-op" begin
        c = SLAYERControl(; enabled=false)
        params = [_mk_params()]
        dp = ComplexF64[0.0+0im;;]                      # 1×1 matrix
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

    @testset "run_slayer_from_inputs: coupled mode finds known root" begin
        # Build a 2-surface problem with a known coupled root by construction.
        p1 = _mk_params(rs=0.5, lu=1.0e7, tauk=1.0e-4, Q_e=-1.0, Q_i=0.5,
                         m=2, ising=1)
        p2 = _mk_params(rs=0.6, lu=2.0e7, tauk=1.2e-4, Q_e=-0.8, Q_i=0.4,
                         m=3, ising=2)
        params = [p1, p2]

        model = SLAYERModel()
        # Pick a target Q and pin the diagonal Δ'_kk so det(M(Q_target)) = 0
        Q_target = 0.2 + 0.3im
        # Compute what each surface sees at Q_target (with per-surface
        # rescaling: surface 2 sees Q_target * tauk_1/tauk_2).
        Q_1 = Q_target * (p1.tauk / p1.tauk)         # = Q_target
        Q_2 = Q_target * (p1.tauk / p2.tauk)
        Δ1 = InnerLayer.solve_inner(model, p1, Q_1).tearing * p1.lu^(1/3)
        Δ2 = InnerLayer.solve_inner(model, p2, Q_2).tearing * p2.lu^(1/3)
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
        p1 = _mk_params(rs=0.5, lu=1.0e7, tauk=1.0e-4, m=2, ising=1)
        p2 = _mk_params(rs=0.6, lu=2.0e7, tauk=1.2e-4, m=3, ising=2)
        params = [p1, p2]

        # Diagonal dp, zero coupling → trivial root structure at Q_target=0
        Q_target = 0.0 + 0.0im
        model = SLAYERModel()
        Δ1 = InnerLayer.solve_inner(model, p1, Q_target).tearing * p1.lu^(1/3)
        Δ2 = InnerLayer.solve_inner(model, p2, Q_target).tearing * p2.lu^(1/3)
        dp = ComplexF64[Δ1 0.0; 0.0 Δ2]

        c = SLAYERControl(; enabled=true,
                            scan_mode=:brute_force,
                            coupling_mode=:coupled,
                            Q_re_range=(-0.5, 0.5),
                            Q_im_range=(-0.3, 0.3),
                            nre=40, nim=40,
                            pole_threshold=1e5,
                            store_scan=true)
        r = run_slayer_from_inputs(params, dp, c)

        mktemp() do path, io
            close(io)
            h5open(path, "w") do f
                write_slayer_hdf5!(f, r)
            end
            h5open(path, "r") do f
                g = f["slayer"]
                @test haskey(g, "enabled") && read(g["enabled"]) == 1
                @test haskey(g, "settings")
                @test haskey(g, "per_surface")
                @test haskey(g, "roots")
                @test haskey(g, "diagnostics")
                @test haskey(g, "scan")

                # Settings round-trip
                @test read(g["settings/inner_model"])   == "slayer_fitzpatrick"
                @test read(g["settings/scan_mode"])     == "brute_force"
                @test read(g["settings/coupling_mode"]) == "coupled"
                @test read(g["settings/nre"]) == 40

                # Per-surface arrays have the right length
                @test length(read(g["per_surface/ising"])) == 2
                @test read(g["per_surface/ising"]) == [1, 2]
                @test read(g["per_surface/lu"])[1] ≈ 1.0e7
                @test read(g["per_surface/lu"])[2] ≈ 2.0e7

                # Roots arrays
                @test length(read(g["roots/Q_root_real"])) == 1    # coupled
                @test length(read(g["roots/omega_Hz"]))    == 1

                # Ragged diagnostics use flat+offsets encoding
                @test haskey(g["diagnostics/valid_roots"], "flat_real")
                @test haskey(g["diagnostics/valid_roots"], "flat_imag")
                @test haskey(g["diagnostics/valid_roots"], "offsets")

                # Scan group present (store_scan=true)
                @test haskey(g, "scan/surface_1")
                @test read(g["scan/surface_1/kind"]) == "brute_force"
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
                g = f["slayer"]
                @test read(g["enabled"]) == 0
                @test !haskey(g, "settings")      # no further groups
                @test !haskey(g, "per_surface")
            end
        end
    end
end
