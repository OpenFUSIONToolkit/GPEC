# runtests_islands_grids.jl
#
# Islands module — phase-space grid / discretization unit tests (src/Islands/phasespace).
# Pure numerics: spectral and finite-difference differentiation and quadrature.
# No physics coefficients here (nothing [VERIFY]-tagged); these back the MMS
# ladder A1 checks in runtests_islands_operators.jl.

const PS = GeneralizedPerturbedEquilibrium.Islands.PhaseSpace

@testset "Islands phase-space grids" begin

    @testset "Fourier spectral ∂ξ is exact for bandlimited data" begin
        fg = PS.FourierGrid(16; L=2π)
        x = fg.nodes
        g = @. sin(3x) + cos(2x) - 0.5 * sin(x)
        dg_exact = @. 3cos(3x) - 2sin(2x) - 0.5cos(x)
        @test maximum(abs, fg.D1 * g .- dg_exact) < 1e-12
        # odd node count is rejected
        @test_throws ArgumentError PS.FourierGrid(15)
    end

    @testset "Mapped FD converges at design order (uniform grid)" begin
        # smooth decaying test function on [-6, 6]
        f(x) = exp(-x^2 / 2)
        f′(x) = -x * exp(-x^2 / 2)
        f″(x) = (x^2 - 1) * exp(-x^2 / 2)
        err1 = Float64[]
        err2 = Float64[]
        ns = [17, 33, 65]
        for n in ns
            g = PS.MappedFDGrid(n; halfwidth=6.0, order=4)
            xn = g.nodes
            push!(err1, maximum(abs, g.D1 * f.(xn) .- f′.(xn)))
            push!(err2, maximum(abs, g.D2 * f.(xn) .- f″.(xn)))
        end
        # 4th-order: halving h cuts error by ≳ 2^4 (allow margin for pre-asymptotics)
        @test log(err1[2] / err1[3]) / log(ns[3] / ns[2]) > 3.7
        @test log(err2[2] / err2[3]) / log(ns[3] / ns[2]) > 3.7
        @test err1[end] < 1e-3
        @test err2[end] < 1e-3
    end

    @testset "Layer-clustered map preserves order and packs the center" begin
        # a strongly clustered grid still differentiates a smooth function accurately
        g = PS.MappedFDGrid(65; halfwidth=6.0, clustering=2.0, order=4)
        # node spacing is smallest near the clustering center (x = 0)
        Δ = diff(g.nodes)
        icenter = argmin(abs.(g.nodes[1:(end-1)] .+ g.nodes[2:end]) ./ 2)
        @test Δ[icenter] < Δ[1]
        @test Δ[icenter] < Δ[end]
        f(x) = sin(x) * exp(-x^2 / 8)
        f′(x) = (cos(x) - x / 4 * sin(x)) * exp(-x^2 / 8)
        @test maximum(abs, g.D1 * f.(g.nodes) .- f′.(g.nodes)) < 1e-3
    end

    @testset "Half-domain grid packs at y_c and spans [0, y_max]" begin
        g = PS.MappedFDGrid(17; halfwidth=4.0, clustering=1.0, center=1.0, domain=:half, order=4)
        @test g.nodes[1] ≈ 0.0 atol = 1e-12
        @test g.nodes[end] ≈ 4.0 atol = 1e-12
        @test issorted(g.nodes)
    end

    @testset "Simpson quadrature weights integrate at design order" begin
        # ∫_0^4 exp(-(y-1)^2/2) dy — self-convergence (no closed form needed):
        # successive-refinement differences must shrink at ≳ 4th order.
        quad(n) =
            let g = PS.MappedFDGrid(n; halfwidth=4.0, clustering=1.0, center=1.0, domain=:half, order=4)
                sum(g.wq .* exp.(-(g.nodes .- 1) .^ 2 ./ 2))
            end
        q = quad.([17, 33, 65, 129])
        d1, d2, d3 = abs(q[2] - q[1]), abs(q[3] - q[2]), abs(q[4] - q[3])
        @test log(d1 / d2) / log(2) > 3.3
        @test log(d2 / d3) / log(2) > 3.3
    end

    @testset "Gauss–Laguerre quadrature is exact on polynomials × e^{-E}" begin
        gg = PS.GaussGrid(6; kind=:laguerre)
        # ∫_0^∞ E^k e^{-E} dE = k!
        for (k, want) in ((0, 1.0), (1, 1.0), (2, 2.0), (3, 6.0), (4, 24.0))
            @test sum(gg.weights .* gg.nodes .^ k) ≈ want rtol = 1e-10
        end
    end

    @testset "IslandGrid assembles all five coordinates" begin
        ig = PS.IslandGrid(; nx=17, nxi=16, ny=9, nE=4, halfwidth_x=6.0, clustering_x=1.5,
            y_max=4.0, y_c=1.0, clustering_y=1.0)
        @test PS.nnodes(ig) == (17, 16, 9, 4, 2)
        @test ig.σ == [1.0, -1.0]
        @test length(ig.x.wq) == 17
        # even nx is rejected (Simpson needs odd)
        @test_throws ArgumentError PS.MappedFDGrid(16; halfwidth=6.0)
    end

    @testset "island-resolution protocol: Δx(0) ≤ w/K (04 §2)" begin
        # island_clustering_x picks β so the central spacing hits the w/K target exactly
        for (w, Lx, nx, K) in ((0.5, 3.0, 21, 8), (1.0, 6.0, 31, 8), (0.3, 2.0, 41, 10), (2.0, 12.0, 25, 6))
            β = PS.island_clustering_x(w, Lx, nx; K=K)
            gx = PS.MappedFDGrid(nx; halfwidth=Lx, clustering=β, center=0.0, domain=:symmetric, order=4)
            Δ0 = minimum(diff(gx.nodes))
            @test Δ0 <= w / K * (1 + 1e-6)                 # target met (exact two-node spacing)
            @test Δ0 > w / K * 0.98                        # and not wastefully over-resolved
        end
        # a domain already fine at uniform spacing needs no clustering
        @test PS.island_clustering_x(5.0, 6.0, 9; K=1) == 0.0
        # over-clustering (too few nodes for the requested w/K) is refused, not silently starved
        @test_throws ArgumentError PS.island_clustering_x(0.01, 20.0, 9; K=10)
        @test_throws ArgumentError PS.island_clustering_x(-1.0, 6.0, 21)

        # resolved_island_grid + the is_island_resolved diagnostic
        g = PS.resolved_island_grid(; w=0.5, nx=25, K=8, Lx_over_w=6.0, nxi=8, ny=9, nE=2, y_max=4.0, clustering_y=0.8)
        @test PS.nnodes(g) == (25, 8, 9, 2, 2)
        @test g.x.nodes[end] ≈ 6.0 * 0.5                   # Lx = Lx_over_w · w
        chk = PS.is_island_resolved(g, 0.5; K=8)
        @test chk.resolved
        @test chk.central_spacing <= 0.5 / 8
        @test chk.Lx_over_w ≈ 6.0
        @test chk.nodes_per_halfwidth >= 8
        @test PS.central_x_spacing(g) == minimum(diff(g.x.nodes))
        # an under-resolved grid (coarse, near far field) is flagged not resolved
        coarse = PS.IslandGrid(; nx=9, nxi=8, ny=9, nE=2, halfwidth_x=1.0, clustering_x=0.0, y_max=4.0, clustering_y=0.8)
        @test !PS.is_island_resolved(coarse, 0.5; K=8).resolved   # both Δx≪w and Lx/w≥5 fail
    end

    @testset "drift-island band grid: uniform envelope band + geometric tails (04 §1)" begin
        w, K, R, Lx = 0.05, 8, 0.25, 0.6
        h = w / K
        nodes = PS.banded_x_nodes(; R=R, h=h, Lx=Lx, growth=1.2)
        # ascending, strictly increasing, symmetric, includes 0 and lands on ±Lx
        @test issorted(nodes) && all(diff(nodes) .> 0)
        @test maximum(abs.(nodes .+ reverse(nodes))) < 1e-12       # symmetric about 0
        @test any(x -> abs(x) < 1e-14, nodes)                      # includes x=0
        @test nodes[1] ≈ -Lx && nodes[end] ≈ Lx
        # the central band resolves the envelope at ≤ w/K (uniform spacing there)
        band = filter(x -> abs(x) <= R + 1e-12, nodes)
        @test maximum(diff(band)) <= h * (1 + 1e-9)
        @test all(≈(h; atol=1e-12), diff(band))                    # band is uniform
        # tails coarsen (spacing grows outward); the final interval is clamped to
        # land exactly on Lx so it may be smaller — check the geometric interior.
        right = filter(x -> x > 0, nodes)
        dtail = diff(right[right.>=R])
        @test dtail[1] > h                                         # first tail step coarser than the band
        @test issorted(dtail[1:(end - 1)])                         # geometric growth (interior)
        @test maximum(dtail) > h                                   # tail is coarser than the band
        # guards
        @test_throws ArgumentError PS.banded_x_nodes(; R=0.25, h=h, Lx=0.2)   # Lx ≤ R
        @test_throws ArgumentError PS.banded_x_nodes(; R=-1.0, h=h, Lx=Lx)
        @test_throws ArgumentError PS.banded_x_nodes(; R=R, h=h, Lx=Lx, growth=1.0)

        # MappedFDGrid from explicit nodes: Fornberg D1/D2 exact on polynomials to order
        g = PS.MappedFDGrid(nodes; order=4)
        @test g.n == length(nodes) && g.nodes == nodes
        f3 = g.nodes .^ 3
        @test maximum(abs, g.D1 * f3 .- 3 .* g.nodes .^ 2) / maximum(abs, 3 .* g.nodes .^ 2) < 1e-8
        @test maximum(abs, g.D2 * f3 .- 6 .* g.nodes) < 1e-6
        @test all(g.wq .> 0) && sum(g.wq) ≈ (nodes[end] - nodes[1])   # trapezoidal weights sum to the interval
        @test_throws ArgumentError PS.MappedFDGrid([0.0, 1.0, 0.5]; order=4)   # not sorted

        # drift_island_grid assembles an IslandGrid whose band resolves [-R, R]
        dg = PS.drift_island_grid(; R=R, w=w, K=K, Lx_over_w=12.0, nxi=8, ny=9, nE=3, y_max=4.0, clustering_y=0.8)
        @test dg.x.nodes[end] ≈ 12.0 * w                           # Lx = Lx_over_w · w
        @test PS.central_x_spacing(dg) <= h * (1 + 1e-9)           # central spacing ≤ w/K
        @test any(x -> abs(abs(x) - R) < h, dg.x.nodes)            # nodes sit at the shifted drift island ±R
        @test dg.y_c == 1.0 && PS.nnodes(dg)[2:end] == (8, 9, 3, 2)
    end
end
