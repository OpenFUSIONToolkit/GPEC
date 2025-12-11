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