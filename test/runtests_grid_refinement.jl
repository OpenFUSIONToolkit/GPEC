using Test
using GeneralizedPerturbedEquilibrium
const GridRef = GeneralizedPerturbedEquilibrium.Equilibrium

@testset "Grid Refinement" begin

    @testset "_fourth_derivative_nodes" begin
        # Exact on a quartic over nonuniform nodes: f'''' = 24 everywhere
        xs = [0.0, 0.1, 0.25, 0.33, 0.5, 0.62, 0.8, 0.93, 1.0]
        d4 = GridRef._fourth_derivative_nodes(xs, xs .^ 4)
        @test all(isapprox.(d4, 24.0; rtol=1e-8))

        # Zero on a cubic (divided differences of order 4 vanish)
        d4c = GridRef._fourth_derivative_nodes(xs, 2.0 .* xs .^ 3 .- xs)
        @test all(abs.(d4c) .< 1e-8)

        # Fewer than 5 nodes: returns zeros
        @test GridRef._fourth_derivative_nodes([0.0, 0.5, 1.0], [1.0, 2.0, 3.0]) == zeros(3)
    end

    @testset "_equidistribute" begin
        xs = collect(range(0.0, 1.0; length=101))

        # Constant density -> uniform grid
        nodes = GridRef._equidistribute(xs, fill(10.0, 101), 20)
        @test length(nodes) == 21
        @test nodes[1] == 0.0 && nodes[end] == 1.0
        @test all(isapprox.(diff(nodes), 0.05; atol=1e-10))

        # Localized density bump attracts a proportional share of the knots
        rho = fill(1.0, 101)
        rho[41:60] .= 100.0  # bump on [0.4, 0.6)
        nodes = GridRef._equidistribute(xs, rho, 50)
        @test all(diff(nodes) .> 0)
        in_bump = count(x -> 0.4 <= x <= 0.6, nodes)
        @test in_bump > 40  # bump carries ~95% of the density integral
    end

    @testset "merge_mandatory_nodes" begin
        grid = collect(range(0.0, 1.0; length=11))  # spacing 0.1, delta_min = 0.025

        # Empty mandatory is the identity
        @test GridRef.merge_mandatory_nodes(grid, Float64[]) == grid

        # Plain insertion away from existing nodes
        merged = GridRef.merge_mandatory_nodes(grid, [0.55])
        @test 0.55 in merged
        @test length(merged) == 12
        @test all(diff(merged) .> 0)

        # Snap: existing node within delta_min of the mandatory one is dropped
        merged = GridRef.merge_mandatory_nodes(grid, [0.51])
        @test (0.51 in merged) && !(0.5 in merged)
        @test length(merged) == 11

        # Endpoints always win: out-of-span and near-endpoint mandatory nodes are discarded
        merged = GridRef.merge_mandatory_nodes(grid, [-0.1, 0.0, 1.0, 1.2, 0.001, 0.999])
        @test merged == grid

        # Near-duplicate mandatory pair collapses to the first (same physical surface)
        merged = GridRef.merge_mandatory_nodes(grid, [0.55, 0.55 + 1e-15])
        @test count(x -> abs(x - 0.55) < 0.01, merged) == 1
        @test minimum(diff(merged)) > 1e-3

        # Distinct mandatory nodes farther apart than delta_min both survive
        merged = GridRef.merge_mandatory_nodes(grid, [0.42, 0.47])
        @test (0.42 in merged) && (0.47 in merged)
        @test all(diff(merged) .> 0)
    end

    @testset "_validate_psi_nodes" begin
        grid = collect(range(0.01, 0.99; length=10))
        @test GridRef._validate_psi_nodes(grid, 0.01, 0.99) === grid
        @test_throws ErrorException GridRef._validate_psi_nodes(reverse(grid), 0.01, 0.99)
        @test_throws ErrorException GridRef._validate_psi_nodes(grid, 0.02, 0.99)
        @test_throws ErrorException GridRef._validate_psi_nodes(grid, 0.01, 0.995)
        @test_throws ErrorException GridRef._validate_psi_nodes([0.5], 0.5, 0.5)
    end

    @testset "density concentrates knots at a synthetic layer" begin
        # tanh pedestal at psi=0.9 on a nonuniform grid; density must peak in the layer
        xs = collect(range(0.01, 0.9995; length=201)) .^ 0.8
        sort!(xs)
        vals = tanh.((xs .- 0.9) ./ 0.05)
        rho = zeros(length(xs))
        GridRef._density_from_curvature!(rho, GridRef._fourth_derivative_nodes(xs, vals), 1.0, 3e-3)
        # Density peaks inside the layer and vanishes (noise floor) far from it
        layer = [0.8 <= x <= 0.99 for x in xs]
        far = [x < 0.5 for x in xs]
        @test maximum(rho[layer]) == maximum(rho)
        @test maximum(rho[layer]) > 100.0
        @test maximum(rho[far]) == 0.0

        # tau^(-1/3) scaling of the density (h^3 derivative error model); taus chosen so
        # neither endpoint hits the H_TARGET_MIN/MAX clamps
        rho_tight = zeros(length(xs))
        GridRef._density_from_curvature!(rho_tight, GridRef._fourth_derivative_nodes(xs, vals), 1.0, 3e-6)
        ratio = maximum(rho_tight) / maximum(rho)
        @test isapprox(ratio, 10.0; rtol=0.3)  # (3e-3/3e-6)^(1/3) = 10
    end

    @testset "Solovev two-pass integration" begin
        eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(;
            eq_type="sol", grid_type="auto", mpsi=0, psi_accuracy=1e-3,
            psilow=0.01, psihigh=0.995)
        sol_config = GeneralizedPerturbedEquilibrium.Equilibrium.SolovevConfig()

        @test GridRef.wants_two_pass(eq_config)
        # Legacy alias still selects the two-pass grid
        alias_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(;
            eq_type="sol", grid_type="log_asymptotic", mpsi=0, psi_accuracy=1e-3)
        @test GridRef.wants_two_pass(alias_config)

        equil1 = GridRef.setup_equilibrium(eq_config, sol_config)

        # Rational nodes land exactly on q = m/n surfaces
        rat = GeneralizedPerturbedEquilibrium.ForceFreeStates.rational_psi_nodes(equil1; nlow=1, nhigh=2)
        @test !isempty(rat) && issorted(rat)
        for p in rat
            q = equil1.profiles.q_spline(p)
            @test isapprox(q, round(2q) / 2; atol=1e-6)  # m/n with n in 1:2 is a half-integer
        end

        grid = GridRef.refined_psi_grid(equil1; tau=eq_config.psi_accuracy, mandatory=rat)
        @test all(diff(grid) .> 0)
        @test grid[1] == eq_config.psilow && grid[end] == eq_config.psihigh
        @test all(p -> p in grid, rat)

        # Every rational surface appears as a mandatory knot in the refined grid, and the
        # min-spacing snap in merge_mandatory_nodes keeps the grid strictly increasing even
        # when multi-n (n=1,2) rationals cluster.
        @test all(p -> any(g -> abs(g - p) < 1e-12, grid), rat)
        @test minimum(diff(grid)) > 0

        # Pass 2 honors the refined grid exactly
        equil2 = GridRef.setup_equilibrium(eq_config, sol_config; override_psi_nodes=grid)
        @test equil2.profiles.xs == grid
        @test equil2.rzphi_xs == grid
    end
end
