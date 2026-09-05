# Regression test for the axis initial condition of the Euler-Lagrange integration.
#
# Free-boundary energies from the Frobenius free-axis start (`fixed_axis = false`, the
# default before this test existed) were wrong on flat-core diverted equilibria: one
# eigenvalue of the plasma response matrix W_p sat ~10x off its Fortran DCON value and,
# where it crossed zero, `et[1]` reported a spurious -1e2 ... -1e5 "instability".  The
# fixture is a 129x129 TokaMaker geqdsk of a synthetic DIII-D-like ramp-up slice
# (q0 = 3.42, q95 = 5.9, diverted, no wall, n = 1) on which Fortran DCON (v1.5.5, same
# mlow/mhigh/psilim/dmlim, mpsi 256) gives a stiff W_p eigenvalue of +1.609e4, no
# negative W_p eigenvalue, and et[1] = +1.42; the old default gave et[1] = -8.4e4.
#
# The `@test_broken` lines pin the DEFECT of the Frobenius start: if `compute_axis_init`
# is ever repaired to deliver its documented psilow^(|m|/2) limit, they report an
# "unexpected pass" and should then be promoted to plain `@test`s.
using LinearAlgebra
using GeneralizedPerturbedEquilibrium.ForceFreeStates: ForceFreeStatesControl

const FIXED_AXIS_FIXTURE = joinpath(@__DIR__, "test_data", "regression_rampup_fixed_axis")
# Fortran DCON, GPEC v1.5.5-323, on the 257x257 parent of the fixture at mpsi 256 / mtheta 512.
const WP_STIFF_DCON = 1.6092e4

function run_fixed_axis_case(fixed_axis::Bool)
    dir = mktempdir()
    for f in readdir(FIXED_AXIS_FIXTURE)
        cp(joinpath(FIXED_AXIS_FIXTURE, f), joinpath(dir, f))
    end
    toml = joinpath(dir, "gpec.toml")
    s = read(toml, String)
    s = replace(s, r"^fixed_axis.*\n"m => "")
    s = replace(s, "[ForceFreeStates]\n" => "[ForceFreeStates]\nfixed_axis = $(fixed_axis)\n")
    write(toml, s)
    r = GeneralizedPerturbedEquilibrium.main([dir])
    fb = r.ffs.free_boundary
    wp = Matrix(fb.wp)
    wpe = eigvals(Hermitian((wp + wp') / 2))
    dp = real.(diag(Matrix(r.ffs.delta_prime.matrix)))
    return (et1 = minimum(real.(fb.et)), wp_min = minimum(wpe), wp_max = maximum(wpe),
            n_neg = count(<(0), wpe), dprime = dp)
end

@testset "Fixed-axis default (DCON axis condition)" begin
    # The default itself is the fix: pin it.
    @test ForceFreeStatesControl().fixed_axis == true

    fixed = run_fixed_axis_case(true)
    @test fixed.et1 > 0                      # physical fundamental, not a pole
    @test fixed.n_neg == 0                   # W_p has no negative eigenvalue (DCON: none)
    @test isapprox(fixed.wp_max, WP_STIFF_DCON; rtol=0.02)  # stiff eigenvalue on DCON's value (this fixture gives 1.6090e4; 2 % covers 129^2 / mpsi 128 vs DCON's 257^2 / mpsi 256)

    # The Frobenius free-axis start, kept as an opt-in: record its defect.
    free = run_fixed_axis_case(false)
    @test_broken free.et1 > 0
    @test_broken free.n_neg == 0
    @test_broken isapprox(free.wp_max, WP_STIFF_DCON; rtol=0.02)

    # The Delta' BVP does not depend on the axis condition.
    @test length(fixed.dprime) == length(free.dprime)
    @test isapprox(fixed.dprime, free.dprime; rtol=1e-3)
end
