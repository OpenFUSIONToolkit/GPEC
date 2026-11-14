using TOML
using HDF5

# Static pressure flattening across islands of given width, on the coarse TJ fixture equilibrium.
@testset "island flattening" begin
    GPEC = GeneralizedPerturbedEquilibrium
    IF = GPEC.Tearing.IslandFlattening
    template = joinpath(@__DIR__, "test_data", "regression_tj_ideal_example")
    deck = TOML.parsefile(joinpath(template, "gpec.toml"))
    equil = GPEC.Equilibrium.PlasmaEquilibrium(GPEC.equilibrium_config(deck, template), GPEC.analytic_config(deck))
    prof = equil.profiles
    p(psi) = prof.P_spline(psi) / GPEC.Equilibrium.mu0
    dV(psi) = prof.dVdpsi_spline(psi)
    one = island_flattening(equil; widths=Dict("2/1" => 0.05))
    psi_s = only(one.islands).psi
    # Volume inside ψ_N = x, by the midpoint rule.
    volume_inside(x; n=4000) = sum(dV(prof.xs[1] + (x - prof.xs[1]) * (k - 0.5) / n) for k in 1:n) * (x - prof.xs[1]) / n

    @testset "the Fortran trapezoid rule converges to the closed form" begin
        # Fortran GPEC: zero dp/dψ on the island, anchor at the edge, integrate inward on a grid.
        grid = range(prof.xs[1], prof.xs[end]; length=200_001)
        dp = [abs(x - psi_s) <= 0.025 ? 0.0 : prof.P_deriv(x) / GPEC.Equilibrium.mu0 for x in grid]
        pm = similar(dp)
        pm[end] = p(grid[end])
        for i in (length(grid)-1):-1:1
            pm[i] = pm[i+1] - 0.5 * (dp[i+1] + dp[i]) * step(grid)
        end
        @test pm[1] ≈ IF.flattened_pressure(p, one.intervals, grid[1]) rtol = 1e-6
        dW = 1.5 * sum((p(x) - y) * dV(x) for (x, y) in zip(grid, pm)) * step(grid)
        @test dW ≈ one.delta_W rtol = 1e-3
    end

    @testset "narrow islands: ΔW → 0 and ΔW ∝ w" begin
        @test island_flattening(equil; widths=Dict("2/1" => 0.0)).delta_W == 0
        small = [island_flattening(equil; widths=Dict("2/1" => w)).delta_W for w in (1e-3, 2e-3)]
        @test small[2] / small[1] ≈ 2 rtol = 1e-2
        # Leading order: the drop −p′w across the island lowers everything inward of it.
        @test small[1] ≈ 1.5 * (-prof.P_deriv(psi_s) / GPEC.Equilibrium.mu0) * 1e-3 * volume_inside(psi_s) rtol = 2e-2
    end

    @testset "overlapping islands are flattened once" begin
        # The 2/1 (ψ_N ≈ 0.48) and 5/2 (≈ 0.79) islands overlap, so they flatten one interval.
        widths = Dict("2/1" => 0.4, "5/2" => 0.3)
        two = island_flattening(equil; widths)
        (a, b) = only(two.intervals)
        @test (a, b) == (two.islands[1].psi_in, two.islands[2].psi_out)
        alone = [island_flattening(equil; widths=Dict(k => v)).delta_W for (k, v) in widths]
        @test two.delta_W < sum(alone)
        n = 200_000
        grid = range(prof.xs[1], prof.xs[end]; length=n + 1)
        mids = (grid[1:end-1] .+ grid[2:end]) ./ 2
        @test two.delta_W ≈ 1.5 * sum((p(x) - IF.flattened_pressure(p, [(a, b)], x)) * dV(x) for x in mids) * step(grid) rtol = 1e-6
        @test two.delta_p0 ≈ only(two.delta_p) ≈ p(a) - p(b)
        # Inside the island the pressure sits at its outer-edge value.
        @test IF.flattened_pressure(p, two.intervals, (a + b) / 2) ≈ p(b)
    end

    @testset "edges, bootstrap fraction and bad input" begin
        wide = island_flattening(equil; widths=Dict("2/1" => 4.0))
        @test only(wide.islands).clipped
        # The whole profile flattens to its boundary value.
        @test wide.delta_W ≈ wide.W - 1.5 * p(prof.xs[end]) * volume_inside(prof.xs[end]) rtol = 1e-8
        # A uniform ⟨j_bs·B⟩ loses the volume fraction of the flattened interval.
        frac = island_flattening(equil; widths=Dict("2/1" => 0.05), jbs_B=_ -> 1.0)
        (a, b) = only(frac.intervals)
        @test frac.jbs_fraction_lost ≈ (volume_inside(b) - volume_inside(a)) / volume_inside(prof.xs[end]) rtol = 1e-3
        @test all(iszero, frac.jbs_B_mod[a.<frac.psi.<b])
        @test isnan(one.jbs_fraction_lost) && isempty(one.jbs_B)
        @test_throws ArgumentError island_flattening(equil; widths=Dict("2-1" => 0.05))
        @test_throws ArgumentError island_flattening(equil; widths=Dict("2/1" => -0.1))
    end

    @testset "HDF5 output keeps the rest of Tearing/" begin
        mktempdir() do dir
            path = joinpath(dir, "gpec.h5")
            h5open(path, "w") do f
                f["Tearing/enabled"] = 1
                f["Input/gpec_toml_raw"] = sprint(TOML.print, Dict("Tearing" => Dict("inner_model" => "SLAYER")))
            end
            save_to_hdf5(path, one)
            h5open(path, "r") do f
                @test read(f["Tearing/enabled"]) == 1
                @test read(f["Tearing/IslandFlattening/delta_W"]) == one.delta_W
                @test read(f["Tearing/IslandFlattening/Profiles/p_mod"]) == one.p_mod
                recorded = TOML.parse(read(f["Input/gpec_toml_raw"]))["Tearing"]
                @test recorded["inner_model"] == "SLAYER"
                @test recorded["IslandFlattening"]["widths"] == Dict("2/1" => 0.05)
            end
        end
    end
end
