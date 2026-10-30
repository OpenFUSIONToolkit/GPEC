using TOML
using Plots

# Analysis.EnergyDecomposition on the coarse Solovev fixture (mpsi = 16, forward integrator, vac_flag = true) in Boozer
# coordinates, which unlike the deck's PEST coordinates have nonzero g_θζ and g_ψζ: one solve and one decomposition,
# shared by every testset but the last two.
@testset "Analysis.EnergyDecomposition" begin
    ENV["GKSwstype"] = "100"
    GPEC = GeneralizedPerturbedEquilibrium
    ED = GPEC.Analysis.EnergyDecomposition

    # Forward solve of a deck for toroidal mode number 1, with `[Equilibrium]` and `[ForceFreeStates]` overrides.
    function forward_solve(dir; eq_overrides=Dict{String,Any}(), ffs_overrides=Dict{Symbol,Any}())
        deck = TOML.parsefile(joinpath(dir, "gpec.toml"))
        merge!(deck["Equilibrium"], eq_overrides)
        config = GPEC.Equilibrium.EquilibriumConfig(deck["Equilibrium"], dir)
        equil =
            haskey(deck, "SOL_INPUT") ? GPEC.Equilibrium.PlasmaEquilibrium(config, GPEC.Equilibrium.SolovevConfig(deck["SOL_INPUT"])) :
            GPEC.Equilibrium.PlasmaEquilibrium(config)
        wall = GPEC.Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in deck["Wall"])...)
        ffs_kwargs = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in deck["ForceFreeStates"] if !(k in ("integrator", "nn_low", "nn_high")))
        return solve(equil, Forward(); nn=1, wall=wall, dir_path=mktempdir(), merge(ffs_kwargs, ffs_overrides)...)
    end

    # Standard-form surface integrals and 2/ψ₀² times the Euler-Lagrange quadratic form ξ'†Fξ' + 2Re(ξ'†Kξ) + ξ†Gξ,
    # (knot, eigenmode) matrices over the knots among the radial nodes.
    function standard_and_lagrangian(ffs, ed)
        knots = findall(in(ffs.equil.rzphi_xs), ed.psi)
        u1, du1, _ = ED.node_states(ffs, ed.psi)
        N = ffs.numpert_total
        F, F_lower, K, G = (zeros(ComplexF64, N, N) for _ in 1:4)
        standard = [real(ed.profile[i, 5, m] - ed.profile[i, 6, m] + ed.profile[i, 7, m]) for i in knots, m in eachindex(ed.eigenmodes)]
        lagrangian = similar(standard)
        for (m, k) in enumerate(ed.eigenmodes)
            alpha = ffs.solution.u_store[:, :, 1, ffs.solution.step] \ Vector{ComplexF64}(ffs.free_boundary.wt[:, k])
            for (j, i) in enumerate(knots)
                m_nq = (ffs.mlow:ffs.mhigh) .- ffs.nlow * ffs.equil.profiles.q_spline(ed.psi[i])
                ffs.mats.ideal.F_spline_lower(vec(F_lower), ed.psi[i])
                F .= LowerTriangular(F_lower) * LowerTriangular(F_lower)' .* m_nq .* m_nq'
                ffs.mats.ideal.K_spline(vec(K), ed.psi[i])
                K .*= m_nq
                ffs.mats.ideal.G_spline(vec(G), ed.psi[i])
                xi, dxi = u1[:, :, i] * alpha, du1[:, :, i] * alpha
                lagrangian[j, m] = 2 / ffs.equil.psio^2 * real(dot(dxi, F, dxi) + 2 * dot(dxi, K, xi) + dot(xi, G, xi))
            end
        end
        return standard, lagrangian
    end

    template = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
    boozer = Dict{String,Any}("jac_type" => "boozer")
    ffs = forward_solve(template; eq_overrides=boozer)
    equil = ffs.equil
    ed = ED.decompose_plasma_energy(ffs; eigenmodes=[1, 2])
    psi = ed.psi
    mtheta = length(ed.theta)
    knots = findall(in(equil.rzphi_xs), psi)
    geo = ED.surface_geometry(ffs, psi)

    @testset "standard form equals the DCON Lagrangian on the knots" begin
        for (standard, lagrangian) in zip(standard_and_lagrangian(ffs, ed)...)
            @test standard ≈ lagrangian rtol = 1e-9
        end
    end

    @testset "success checks: δW_p against ep and the curl residual" begin
        # ψ-quadrature error of the mpsi = 16 knot grid (measured 1.4 % and 1.9 %); it converges with mpsi.
        @test all(abs.(ed.relative_error_effective) .< 0.025)
        @test all(abs.(ed.relative_error_standard) .< 0.025)
        # The forms agree only after integration, so they differ by the knot-grid quadrature (measured 1.3e-5 and 8.2e-5).
        @test ed.dW_effective ≈ ed.dW_standard rtol = 2e-4
        # (∇×b_eff)·∇ψ is an algebraic identity on the knots (measured 4e-14), a check of the b_eff assembly.
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

    @testset "ψ integrals start at the ODE start" begin
        # A qlow start falls between knots, and the sweep saves its first state one ODE step after it.
        qffs = forward_solve(template; eq_overrides=boozer, ffs_overrides=Dict{Symbol,Any}(:qlow => 2.5))
        qed = ED.decompose_plasma_energy(qffs; eigenmodes=[1, 2])
        @test qed.psi[1] == qffs.psilow < qffs.solution.psi_store[1]
        @test all(abs.(qed.relative_error_standard) .< 0.025)
    end

    @testset "F′ ≠ 0: the DIII-D-like reconstruction" begin
        # The Solovev fixture has F′ = 0, so the poloidal current and the qF′ part of j^ζ are exercised only here.
        dffs = forward_solve(joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example"))
        ded = ED.decompose_plasma_energy(dffs)
        standard, lagrangian = standard_and_lagrangian(dffs, ded)
        # Judged against the largest density (measured 7e-11), since single knots where the mode is weak agree less closely.
        @test maximum(abs, standard .- lagrangian) < 1e-8 * maximum(abs, lagrangian)
        @test all(ded.curl_residual .< 1e-10)  # measured 4e-12
    end
end
