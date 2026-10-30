using TOML
using Plots

# Analysis.EnergyDecomposition on the coarse Solovev fixture (mpsi = 16, forward integrator, vac_flag = true):
# one solve and one decomposition, shared by every testset.
@testset "Analysis.EnergyDecomposition" begin
    ENV["GKSwstype"] = "100"
    GPEC = GeneralizedPerturbedEquilibrium
    ED = GPEC.Analysis.EnergyDecomposition

    template = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
    deck = TOML.parsefile(joinpath(template, "gpec.toml"))
    equil = GPEC.Equilibrium.PlasmaEquilibrium(GPEC.Equilibrium.EquilibriumConfig(deck["Equilibrium"], template),
        GPEC.Equilibrium.SolovevConfig(deck["SOL_INPUT"]))
    wall = GPEC.Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in deck["Wall"])...)
    ffs_kwargs = Dict(Symbol(k) => v for (k, v) in deck["ForceFreeStates"] if !(k in ("integrator", "nn_low", "nn_high")))
    ffs = solve(equil, Forward(); nn=1, wall=wall, dir_path=mktempdir(), ffs_kwargs...)

    ed = ED.decompose_plasma_energy(ffs; eigenmodes=[1, 2])
    psi = ed.psi
    mtheta = length(ed.theta)
    knots = findall(in(equil.rzphi_xs), psi)
    geo = ED.surface_geometry(ffs, psi)
    states = ED.node_states(ffs, psi)

    @testset "standard form equals the DCON Lagrangian on the knots" begin
        # On a knot the standard-form surface integral is 2/ψ₀² times the Euler-Lagrange quadratic form ξ'†Fξ' + 2Re(ξ'†Kξ) + ξ†Gξ.
        N = ffs.numpert_total
        F, K, G = (zeros(ComplexF64, N, N) for _ in 1:3)
        u1, du1, _ = states
        sol = ffs.solution
        for (m, k) in enumerate(ed.eigenmodes)
            alpha = sol.u_store[:, :, 1, sol.step] \ Vector{ComplexF64}(ffs.free_boundary.wt[:, k])
            for i in knots
                sf = (ffs.mlow:ffs.mhigh) .- ffs.nlow * equil.profiles.q_spline(psi[i])
                ffs.mats.ideal.F_spline_lower(vec(G), psi[i])
                F .= LowerTriangular(G) * LowerTriangular(G)' .* sf .* sf'
                ffs.mats.ideal.K_spline(vec(K), psi[i])
                K .*= sf
                ffs.mats.ideal.G_spline(vec(G), psi[i])
                xi, dxi = u1[:, :, i] * alpha, du1[:, :, i] * alpha
                lagrangian = real(dot(dxi, F, dxi) + 2 * dot(dxi, K, xi) + dot(xi, G, xi))
                standard = real(ed.profile[i, 5, m] - ed.profile[i, 6, m] + ed.profile[i, 7, m])
                @test standard ≈ 2 / equil.psio^2 * lagrangian rtol = 1e-9
            end
        end
    end

    @testset "success checks: δW_p against ep and the curl residual" begin
        # ψ-quadrature error of the mpsi = 16 knot grid (measured 1.4 % and 2.0 %); it converges with mpsi.
        @test all(abs.(ed.relative_error_effective) .< 0.025)
        @test all(abs.(ed.relative_error_standard) .< 0.025)
        # The forms agree only after integration, so they differ by the knot-grid quadrature (measured 1.3e-5 and 8.3e-5).
        @test ed.dW_effective ≈ ed.dW_standard rtol = 2e-4
        # (∇×b_eff)·∇ψ is an algebraic identity on the knots (measured 8e-13), a check of the b_eff assembly.
        @test all(ed.curl_residual .< 1e-10)
    end

    @testset "flux-surface average of the shear" begin
        # The θ-derivative term of S averages out, so ⟨J S⟩ = χ'² q'.
        i = knots[end÷2]
        @test sum(geo.shear[i, :] .* geo.jac[i, :]) / mtheta ≈ (2π * equil.psio)^2 * equil.profiles.q_deriv(psi[i]) rtol = 1e-12
    end

    @testset "metric between the knots" begin
        # Off the knots the metric is splined in ψ; it must match the rzphi bicubic to interpolation accuracy (measured 1.2e-5).
        xs = equil.rzphi_xs
        off = [(xs[end÷2] + xs[end÷2+1]) / 2, psi[end]]
        g = ED.surface_geometry(ffs, off)
        ref = [GPEC.Equilibrium.flux_surface_metric(equil, p, t) for p in off, t in ed.theta]
        @test maximum(abs, g.jac ./ getfield.(ref, :jac) .- 1) < 1e-4
        @test maximum(abs, sqrt.(g.delpsi2) ./ getfield.(ref, :delpsi) .- 1) < 1e-4
        @test maximum(abs, g.R ./ getfield.(ref, :r) .- 1) < 1e-4
    end

    @testset "solution state between saved steps" begin
        sol = ffs.solution
        @test ForceFreeStates.solution_state_at(ffs, sol.psi_store[5]) == sol.u_store[:, :, :, 5]
        # Just left of a mid-interval point is integrated from the left step, just right of it from the right one; they must meet.
        sing = [s.psifac for s in ffs.surfaces]
        i = 9 + findfirst(j -> !any(s -> sol.psi_store[j] < s < sol.psi_store[j+1], sing), 10:(sol.step-1))
        a, b = sol.psi_store[i], sol.psi_store[i+1]
        from_left = ForceFreeStates.solution_state_at(ffs, (a + b) / 2 - 1e-9 * (b - a))
        from_right = ForceFreeStates.solution_state_at(ffs, (a + b) / 2 + 1e-9 * (b - a))
        @test maximum(abs, from_left - from_right) / maximum(abs, from_left) < 1e-6
    end

    @testset "plot" begin
        path = joinpath(mktempdir(), "energy_decomposition.png")
        @test ED.plot_energy_decomposition(ed; eigenmode=2, save_path=path) isa Plots.Plot
        @test isfile(path)
    end
end
