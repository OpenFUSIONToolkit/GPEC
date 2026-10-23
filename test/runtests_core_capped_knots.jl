using Test
using GeneralizedPerturbedEquilibrium

@testset "core_capped_knots" begin
    FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
    R = GeneralizedPerturbedEquilibrium.Equilibrium.RATIONAL_RES_RADIUS

    # Densely packed grid, one rational far from the core.
    xs = collect(range(1e-3, 0.99; length=2000))
    keep = FFS.core_capped_knots(xs, [0.5])
    kept = xs[keep]
    @test keep[1] == 1 && keep[end] == length(xs)
    @test issorted(keep) && allunique(keep)
    @test length(keep) < length(xs)
    # Nothing at or above ψ = 0.1 is dropped.
    @test count(>=(0.1), kept) == count(>=(0.1), xs)
    # Below it, kept knots are at least 5% of ψ apart.
    core = kept[kept .< 0.1]
    @test all(diff(core) .>= 0.05 .* core[2:end])

    # A core rational ends the capped region before its resolution window.
    keep_rat = FFS.core_capped_knots(xs, [0.05])
    @test all(i in keep_rat for i in eachindex(xs) if abs(xs[i] - 0.05) <= R)

    # A grid already sparser than the cap is returned whole.
    sparse = collect(range(0.01, 0.99; length=20))
    @test FFS.core_capped_knots(sparse, [0.5]) == collect(1:20)
    @test FFS.core_capped_knots([0.1, 0.2], Float64[]) == [1, 2]
end
