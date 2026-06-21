# runtests_innerlayer.jl
#
# Inner-layer GGJ Δ(Q) solver tests on the published Glasser & Wang (2020, Eq. 55) benchmark.
# Pins the Galerkin inner-layer Δ as a numerical guard against accidental drift in the Δ(Q) path
# that gal_match_rpec feeds. The forced eigenvalue is built in the RPEC convention γ = 2πi·n·f
# (gal_rotation is f in Hz), so this also exercises that convention.
#
# The reference Δ values are current Julia outputs (reproducible to machine precision), pinned pending
# a Fortran rmatch deltac_run / paper cross-check. The :shooting solver returns NaN on this benchmark,
# so solver-vs-solver cross-validation is left as a follow-up.

const IL = GeneralizedPerturbedEquilibrium.InnerLayer

@testset "InnerLayer GGJ Δ(Q)" begin
    p = IL.glasser_wang_2020_eq55()
    gal = IL.GGJModel(; solver=:galerkin)
    n = 1

    @testset "f=$(f) Hz" for (f, Δ_ref) in (
        (1.0e3, (-5.9565220175e-06 - 2.9037648679e-08im, -2.7749660199e-05 + 1.0841989446e-05im)),
        (1.0e4, (3.7251353050e-08 + 4.0060105204e-07im, 1.0531076405e-07 - 9.4368319597e-08im))
    )
        γ = 2π * im * n * f
        Δ = IL.solve_inner(gal, p, γ)
        @test all(isfinite, Δ)
        @test Δ[1] ≈ Δ_ref[1] rtol = 1e-5
        @test Δ[2] ≈ Δ_ref[2] rtol = 1e-5
    end
end
