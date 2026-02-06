@testset "Test Spline Module" begin
    # TODO: add Fourier spline unit test, test against a theoretical error limit for splines?
    # testing for all boundary conditions
    try
        # Test the spline setup and evaluation functions
        @info "Testing spline setup and evaluation"

        # Make x^3 spline (a polynomial spline should be highly accurate)
        xs = collect(range(1; stop=2, length=21))
        fx3 = xs .^ 3
        fs_matrix = hcat(fx3)
        spline = JPEC.Spl.CubicSpline(xs, fs_matrix; bctype="extrap")

        # Interpolate + extrapolate the spline on a finer grid
        xs_fine = collect(range(0.9; stop=2.1, length=100))
        fs_fine, fsx_fine, fsxx_fine, fsxxx_fine = JPEC.SplinesMod.spline_eval(spline, xs_fine, 3)
        # Check accuracy of spline for x^3 and its derivatives
        @test maximum(abs.(fs_fine .- xs_fine .^ 3)) < 1e-8
        @test maximum(abs.(fsx_fine .- 3 .* xs_fine .^ 2)) < 1e-7
        @test maximum(abs.(fsxx_fine .- 6 .* xs_fine)) < 1e-7
        @test maximum(abs.(fsxxx_fine .- 6.0)) < 1e-6

        # Make sine spline
        xs = collect(range(π / 8; stop=3π / 8, length=50))
        fs = sin.(xs)
        spline = JPEC.Spl.CubicSpline(xs, fs; bctype="extrap")

        # Interpolate the spline on a finer grid
        xs_fine = collect(range(π / 6; stop=π / 3, length=100))
        fs_fine, fsx_fine, fsxx_fine, fsxxx_fine = JPEC.SplinesMod.spline_eval(spline, xs_fine, 3)
        # Check accuracy of spline (higher tolerances for higher derivatives)
        @test maximum(abs.(fs_fine .- sin.(xs_fine))) < 1e-6
        @test maximum(abs.(fsx_fine .- cos.(xs_fine))) < 1e-4
        @test maximum(abs.(fsxx_fine .+ sin.(xs_fine))) < 1e-3
        @test maximum(abs.(fsxxx_fine .+ cos.(xs_fine))) < 1e-2

        # Make e^-ix and e^ix complex valued spline
        xs = collect(range(0; stop=6, length=100))
        fm = exp.(-im .* xs)
        fp = exp.(im .* xs)
        fs_matrix = hcat(fm, fp)
        spline = JPEC.Spl.CubicSpline(xs, fs_matrix; bctype="extrap")

        xs_fine = collect(range(2; stop=2.5, length=100))
        fs_fine, fsx_fine, fsxx_fine, fsxxx_fine = JPEC.SplinesMod.spline_eval(spline, xs_fine, 3)

        # Check accuracy for e^-ix and e^ix and their derivatives
        # Note: third derivative begins to lose accuracy due to oscillatory nature
        @test maximum(abs.(fs_fine[:, 1] .- exp.(-im .* xs_fine))) < 1e-7
        @test maximum(abs.(fsx_fine[:, 1] .+ im .* exp.(-im .* xs_fine))) < 1e-5
        @test maximum(abs.(fsxx_fine[:, 1] .- (-1) .* exp.(-im .* xs_fine))) < 1e-3
        @test maximum(abs.(fsxxx_fine[:, 1] .- im .* exp.(-im .* xs_fine))) < 5e-2
        @test maximum(abs.(fs_fine[:, 2] .- exp.(im .* xs_fine))) < 1e-7
        @test maximum(abs.(fsx_fine[:, 2] .- im .* exp.(im .* xs_fine))) < 1e-5
        @test maximum(abs.(fsxx_fine[:, 2] .- (-1) .* exp.(im .* xs_fine))) < 1e-3
        @test maximum(abs.(fsxxx_fine[:, 2] .+ im .* exp.(im .* xs_fine))) < 5e-2

        # Test bicubic spline setup and evaluation for a 2D function
        xs = collect(range(0; stop=2π, length=100))
        ys = collect(range(0; stop=2π, length=100))

        f1(x, y) = sin(x) * cos(y) + 1
        f2(x, y) = cos(x) * sin(y) + 1
        fvals = Array{Float64}(undef, length(xs), length(ys), 2)
        for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
            fvals[ix, iy, 1] = f1(x, y)
            fvals[ix, iy, 2] = f2(x, y)
        end

        bcspline = JPEC.Spl.BicubicSpline(xs, ys, fvals)

        xs_fine = collect(range(π / 2; stop=π, length=100))
        ys_fine = collect(range(0; stop=π / 2, length=100))
        fs_fine, fsx_fine, fsy_fine = JPEC.Spl.bicube_eval(bcspline, xs_fine, ys_fine, 1)

        X, Y = [x for x in xs_fine, y in ys_fine], [y for x in xs_fine, y in ys_fine]
        f1_true = sin.(X) .* cos.(Y) .+ 1
        f1x_true = cos.(X) .* cos.(Y)
        f1y_true = -sin.(X) .* sin.(Y)
        f2_true = cos.(X) .* sin.(Y) .+ 1
        f2x_true = -sin.(X) .* sin.(Y)
        f2y_true = cos.(X) .* cos.(Y)

        # Check accuracy
        @test maximum(abs.(fs_fine[:, :, 1] .- f1_true)) < 1e-6
        @test maximum(abs.(fsx_fine[:, :, 1] .- f1x_true)) < 1e-5
        @test maximum(abs.(fsy_fine[:, :, 1] .- f1y_true)) < 1e-4
        @test maximum(abs.(fs_fine[:, :, 2] .- f2_true)) < 1e-6
        @test maximum(abs.(fsx_fine[:, :, 2] .- f2x_true)) < 1e-5
        @test maximum(abs.(fsy_fine[:, :, 2] .- f2y_true)) < 1e-4
    catch e
        @test false
        @error "Spline tests failed: $e"
    end
end

@testset "Empty Spline Constructors" begin
    @info "Testing empty spline constructors for type stability"

    # Test Float64 empty spline
    empty_cs_f64 = JPEC.Spl.empty_CubicSpline(Float64)
    @test empty_cs_f64.handle == C_NULL
    @test empty_cs_f64.mx == 0
    @test empty_cs_f64.nqty == 0
    @test typeof(empty_cs_f64) == JPEC.Spl.CubicSpline{Float64}
    @test length(empty_cs_f64._xs) == 0
    @test size(empty_cs_f64._fs) == (0, 0)
    @test length(empty_cs_f64._f) == 0
    @test length(empty_cs_f64._f1) == 0
    @test length(empty_cs_f64._f2) == 0
    @test length(empty_cs_f64._f3) == 0

    # Test ComplexF64 empty spline
    empty_cs_c64 = JPEC.Spl.empty_CubicSpline(ComplexF64)
    @test empty_cs_c64.handle == C_NULL
    @test empty_cs_c64.mx == 0
    @test empty_cs_c64.nqty == 0
    @test typeof(empty_cs_c64) == JPEC.Spl.CubicSpline{ComplexF64}
    @test length(empty_cs_c64._xs) == 0
    @test size(empty_cs_c64._fs) == (0, 0)
    @test length(empty_cs_c64._f) == 0
    @test length(empty_cs_c64._f1) == 0
    @test length(empty_cs_c64._f2) == 0
    @test length(empty_cs_c64._f3) == 0

    # Test default type (ComplexF64)
    empty_cs_default = JPEC.Spl.empty_CubicSpline()
    @test typeof(empty_cs_default) == JPEC.Spl.CubicSpline{ComplexF64}
    @test empty_cs_default.handle == C_NULL

    # Test empty BicubicSpline
    empty_bs = JPEC.Spl.empty_BicubicSpline()
    @test empty_bs.handle == C_NULL
    @test empty_bs.mx == 0
    @test empty_bs.my == 0
    @test empty_bs.nqty == 0
    @test typeof(empty_bs) == JPEC.Spl.BicubicSpline
    @test length(empty_bs._xs) == 0
    @test length(empty_bs._ys) == 0
    @test size(empty_bs._fs) == (0, 0, 0)
    @test length(empty_bs._f) == 0
    @test length(empty_bs._fx) == 0
    @test length(empty_bs._fy) == 0
    @test length(empty_bs._fxx) == 0
    @test length(empty_bs._fxy) == 0
    @test length(empty_bs._fyy) == 0

    # Test empty FourierSpline
    empty_fs = JPEC.Spl.empty_FourierSpline()
    @test empty_fs.handle == C_NULL
    @test empty_fs.mx == 0
    @test empty_fs.my == 0
    @test empty_fs.nqty == 0
    @test typeof(empty_fs) == JPEC.Spl.FourierSpline
    @test length(empty_fs._xs) == 0
    @test length(empty_fs._ys) == 0
    @test size(empty_fs._fs) == (0, 0, 0)
    # Test nested CubicSpline
    @test empty_fs.cs.handle == C_NULL
    @test typeof(empty_fs.cs) == JPEC.Spl.CubicSpline{ComplexF64}
end

@testset "Empty Spline Assertions" begin
    @info "Testing that empty splines throw assertion errors when evaluated"

    empty_cs = JPEC.Spl.empty_CubicSpline(Float64)

    # Test spline_eval! assertion
    @test_throws AssertionError JPEC.Spl.spline_eval!(empty_cs, 0.5)

    # Test spline_deriv1! assertion
    @test_throws AssertionError JPEC.Spl.spline_deriv1!(empty_cs, 0.5)

    # Test spline_deriv2! assertion
    @test_throws AssertionError JPEC.Spl.spline_deriv2!(empty_cs, 0.5)

    # Test spline_deriv3! assertion
    @test_throws AssertionError JPEC.Spl.spline_deriv3!(empty_cs, 0.5)

    # Test spline_eval with vector assertion
    @test_throws AssertionError JPEC.Spl.spline_eval(empty_cs, [0.5, 1.0])

    # Test BicubicSpline assertions
    empty_bs = JPEC.Spl.empty_BicubicSpline()
    @test_throws AssertionError JPEC.Spl.bicube_eval!(empty_bs, 0.5, 0.5)
    @test_throws AssertionError JPEC.Spl.bicube_deriv1!(empty_bs, 0.5, 0.5)
    @test_throws AssertionError JPEC.Spl.bicube_deriv2!(empty_bs, 0.5, 0.5)
    @test_throws AssertionError JPEC.Spl.bicube_eval(empty_bs, [0.5], [0.5])

    # Test FourierSpline assertions
    empty_fs = JPEC.Spl.empty_FourierSpline()
    @test_throws AssertionError JPEC.Spl.fspline_eval(empty_fs, 0.5, 0.5)
    @test_throws AssertionError JPEC.Spl.fspline_eval(empty_fs, [0.5], [0.5])
end

@testset "Spline Replacement" begin
    @info "Testing that empty splines can be replaced with real splines"

    # Create an empty spline
    empty_cs = JPEC.Spl.empty_CubicSpline(ComplexF64)
    @test empty_cs.handle == C_NULL

    # Create a real spline
    xs = collect(range(0.0; stop=1.0, length=10))
    fs = ones(ComplexF64, 10, 1)
    real_spline = JPEC.Spl.CubicSpline(xs, fs; bctype="extrap")
    @test real_spline.handle != C_NULL

    # Verify the real spline can be evaluated
    result = JPEC.Spl.spline_eval!(real_spline, 0.5)
    @test length(result) == 1
    @test isapprox(result[1], 1.0 + 0.0im, atol=1e-10)
end
