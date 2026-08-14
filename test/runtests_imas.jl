# Unit tests for IMAS integration (read_imas and write_imas).
# These use mock data only — no actual GPEC ODE runs.
# Each test verifies one specific piece of the data pipeline.
using Test
using IMASdd
using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium

@testset "IMAS integration" begin

    # Helpers: build a minimal mock IMASdd.dd with a simple circular equilibrium
    function make_mock_dd(; cocos=11)
        dd = IMASdd.dd()
        dd.global_time = 1.0

        # One time-slice
        resize!(dd.equilibrium.time_slice, 1.0; wipe=true)
        eqt = dd.equilibrium.time_slice[1]
        eqt.time = 1.0

        # COCOS 11: ψ_IMAS = 2π × ψ_internal
        psi_axis_int = 0.0
        psi_bnd_int  = 1.5   # Wb/rad
        cf = cocos == 11 ? 2π : 1.0

        eqt.global_quantities.psi_axis     = psi_axis_int * cf
        eqt.global_quantities.psi_boundary = psi_bnd_int  * cf

        nw = 64
        psi_1d = collect(LinRange(psi_axis_int, psi_bnd_int, nw))
        eqt.profiles_1d.psi      = psi_1d .* cf
        eqt.profiles_1d.f        = fill(5.0, nw)
        eqt.profiles_1d.pressure = collect(LinRange(1e4, 0.0, nw))
        eqt.profiles_1d.q        = collect(LinRange(1.0, 3.0, nw))

        # Minimal 2D ψ(R,Z) — rough circular flux surfaces
        nR, nZ = 32, 40
        R_grid = collect(LinRange(1.0, 2.5, nR))
        Z_grid = collect(LinRange(-0.8, 0.8, nZ))
        Rmag   = 1.75
        psi_rz = [min((R - Rmag)^2 / 0.6 + Z^2 / 0.7, psi_bnd_int * 1.4) for R in R_grid, Z in Z_grid]

        resize!(eqt.profiles_2d, 1; wipe=true)
        prof2d              = eqt.profiles_2d[1]
        prof2d.grid.dim1    = R_grid
        prof2d.grid.dim2    = Z_grid
        prof2d.psi          = psi_rz .* cf

        return dd, psi_bnd_int
    end

    # Test 1: read_imas — COCOS 11 conversion produces correct psio
    @testset "read_imas: COCOS 11 conversion" begin
        dd, psi_bnd_int = make_mock_dd(cocos=11)

        config = EquilibriumConfig(; eq_type="imas", eq_filename="mock", imas_cocos=11)
        result = Equilibrium.read_imas(config, dd)

        # psio should equal |psi_boundary - psi_axis| in internal (COCOS 2) units
        @test isapprox(result.psio, psi_bnd_int; rtol=1e-6)
        @test result.rmin ≈ 1.0  atol=1e-9
        @test result.rmax ≈ 2.5  atol=1e-9
    end

    # Test 2: read_imas — COCOS 2 input requires no conversion
    @testset "read_imas: COCOS 2 no-op conversion" begin
        dd, psi_bnd_int = make_mock_dd(cocos=2)

        config = EquilibriumConfig(; eq_type="imas", eq_filename="mock", imas_cocos=2)
        result = Equilibrium.read_imas(config, dd)

        @test isapprox(result.psio, psi_bnd_int; rtol=1e-6)
    end

    # Test 3: read_imas — splines are callable after construction
    @testset "read_imas: splines callable" begin
        dd, _ = make_mock_dd()
        config = EquilibriumConfig(; eq_type="imas", eq_filename="mock", imas_cocos=11)
        result = Equilibrium.read_imas(config, dd)

        # sq_in is a CubicSeriesInterpolant: in-place call sq_in(output_buf, x)
        # Returns 4 values: F, μ₀P, q, √ψ_norm
        buf = zeros(4)
        result.sq_in(buf, 0.5)
        @test all(isfinite, buf)
        @test buf[1] > 0   # F = R·|Bt| must be positive

        # psi_in is a 2D CubicInterpolantND: called with a tuple (R, Z)
        psi_center = result.psi_in((1.75, 0.0))
        @test isfinite(psi_center)
        @test psi_center >= 0.0   # ψ should be non-negative at magnetic axis region
    end

    # An imas run started via `main(; dd)` must thread the dd through to `additional_input`;
    # guards against the dispatch silently dropping it.
    @testset "main dd dispatch: dd reaches additional_input" begin
        dd, _ = make_mock_dd()
        mktempdir() do dir
            write(joinpath(dir, "gpec.toml"), "[Equilibrium]\neq_type = \"imas\"\nimas_cocos = 11\n")
            inputs, eq_config, additional_input =
                GeneralizedPerturbedEquilibrium.build_inputs_from_toml(dir; dd=dd)
            @test eq_config.eq_type == "imas"
            @test additional_input === dd
            # Without a dd, an imas config leaves additional_input === nothing.
            _, _, no_dd = GeneralizedPerturbedEquilibrium.build_inputs_from_toml(dir)
            @test no_dd === nothing
        end
    end

    # Test 4: write_imas — single n, correct metadata and energy_perturbed
    @testset "write_imas: single n_tor" begin
        dd = IMASdd.dd()

        # Mock result: npert=1 (n=1 only), 3 eigenvalues sorted ascending
        mock_et      = [0.4+0im, 0.7+0im, 1.1+0im]
        mock_n_idx   = [0, 0, 0]   # all belong to n=1 (j=0)
        mock_result  = (
            free_energies = (et=mock_et, n_tor_idx=mock_n_idx),
            intr     = (numpert_total=3, npert=1, nlow=1),
        )

        GeneralizedPerturbedEquilibrium.write_imas(dd, mock_result)

        @test dd.mhd_linear.code.name  == "GPEC"
        @test dd.mhd_linear.ideal_flag == 1
        @test length(dd.mhd_linear.time_slice) == 1

        ts = dd.mhd_linear.time_slice[1]
        @test length(ts.toroidal_mode) == 1   # one entry per n_tor

        mode = ts.toroidal_mode[1]
        @test mode.n_tor == 1
        # Least stable for n=1 is the minimum real part = real(et[1]) = 0.4
        @test mode.energy_perturbed ≈ 0.4
    end

    # Test 5: write_imas — multiple n, each gets its own least-stable δW
    @testset "write_imas: multi-n least-stable per n_tor" begin
        dd = IMASdd.dd()

        # Mock result: npert=2 (n=1,2), 4 eigenvalues interleaved across n-blocks.
        # et sorted ascending (least stable globally first).
        # n=1 (j=0) has eigenvalues: real=0.3 (idx 1) and real=0.6 (idx 3)
        # n=2 (j=1) has eigenvalues: real=0.5 (idx 2) and real=0.7 (idx 4)
        mock_et    = [0.3+0im, 0.5+0im, 0.6+0im, 0.7+0im]
        mock_n_idx = [0, 1, 0, 1]
        mock_result = (
            free_energies = (et=mock_et, n_tor_idx=mock_n_idx),
            intr     = (numpert_total=4, npert=2, nlow=1),
        )

        GeneralizedPerturbedEquilibrium.write_imas(dd, mock_result)

        ts = dd.mhd_linear.time_slice[1]
        @test length(ts.toroidal_mode) == 2   # one entry per n_tor

        # Mode ordering: j=0 → index 1, j=1 → index 2
        @test ts.toroidal_mode[1].n_tor == 1
        @test ts.toroidal_mode[2].n_tor == 2

        # Each n gets the minimum real(et) for that n-block
        @test ts.toroidal_mode[1].energy_perturbed ≈ 0.3   # min over {0.3, 0.6}
        @test ts.toroidal_mode[2].energy_perturbed ≈ 0.5   # min over {0.5, 0.7}
    end

    # Test 6: write_imas — consistency between n=1 alone and combined n=1,2 run
    # Proves the n=1 result from a multi-n run matches a single-n run.
    @testset "write_imas: multi-n consistent with individual runs" begin
        # Single n=1 run
        dd_single = IMASdd.dd()
        result_single = (
            free_energies = (et=[0.3+0im, 0.6+0im], n_tor_idx=[0, 0]),
            intr     = (numpert_total=2, npert=1, nlow=1),
        )
        GeneralizedPerturbedEquilibrium.write_imas(dd_single, result_single)

        # Single n=2 run
        dd_single2 = IMASdd.dd()
        result_single2 = (
            free_energies = (et=[0.5+0im, 0.7+0im], n_tor_idx=[0, 0]),
            intr     = (numpert_total=2, npert=1, nlow=2),
        )
        GeneralizedPerturbedEquilibrium.write_imas(dd_single2, result_single2)

        # Combined n=1,2 run (same eigenvalues, now interleaved)
        dd_multi = IMASdd.dd()
        result_multi = (
            free_energies = (et=[0.3+0im, 0.5+0im, 0.6+0im, 0.7+0im], n_tor_idx=[0, 1, 0, 1]),
            intr     = (numpert_total=4, npert=2, nlow=1),
        )
        GeneralizedPerturbedEquilibrium.write_imas(dd_multi, result_multi)

        ts_single  = dd_single.mhd_linear.time_slice[1]
        ts_single2 = dd_single2.mhd_linear.time_slice[1]
        ts_multi   = dd_multi.mhd_linear.time_slice[1]

        # n=1 energy from combined run must match the standalone n=1 run
        @test ts_multi.toroidal_mode[1].energy_perturbed == ts_single.toroidal_mode[1].energy_perturbed
        # n=2 energy from combined run must match the standalone n=2 run
        @test ts_multi.toroidal_mode[2].energy_perturbed == ts_single2.toroidal_mode[1].energy_perturbed
    end

end
