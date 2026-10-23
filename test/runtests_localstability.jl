using TOML

# The LocalStability stage on the coarse Solovev fixture deck (mpsi=16): the named profiles of
# solve(LocalStabilityProblem, Ballooning()) against their kernels.
@testset "LocalStability" begin
    GPEC = GeneralizedPerturbedEquilibrium
    LS = GPEC.LocalStability
    template = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
    deck = TOML.parsefile(joinpath(template, "gpec.toml"))
    equil = GPEC.Equilibrium.setup_equilibrium(
        GPEC.Equilibrium.EquilibriumConfig(deck["Equilibrium"], template),
        GPEC.Equilibrium.SolovevConfig(deck["SOL_INPUT"])
    )
    xs = equil.profiles.xs
    res = solve(LocalStabilityProblem(equil), Ballooning(; verbose=false))

    @testset "profiles match their kernels" begin
        @test res isa LocalStabilityResult
        @test res.psi == xs
        for i in eachindex(xs)
            di = LS.prepare_ballooning_coefficients(i, equil).di
            h = LS.resistive_interchange_h(i, equil)
            @test res.D_I[i] ≈ di rtol = 1e-14
            @test res.D_R[i] ≈ di + (h - 0.5)^2 rtol = 1e-12
            @test LS.mercier_D_I(res, xs[i]) ≈ res.D_I[i] rtol = 1e-12
        end
        @test all(isfinite, res.ballooning_Delta_prime)
    end

    @testset "α boundary is NaN outside its scan window" begin
        @test res.ballooning_psi == xs
        inside = [LS._in_ballooning_scan_window(p, xs[end]) for p in xs]
        @test any(inside) && !all(inside)
        @test all(isfinite, res.alpha[inside])
        @test all(isnan, res.alpha[.!inside])
        @test all(isnan, res.alpha_critical[.!inside])
    end

    @testset "the α scan can be switched off" begin
        quick = solve(LocalStabilityProblem(equil), Ballooning(; alpha_boundary_scan=false, verbose=false))
        @test isempty(quick.ballooning_psi) && isempty(quick.alpha) && isempty(quick.alpha_critical)
        @test quick.D_I == res.D_I
        @test quick.ballooning_Delta_prime == res.ballooning_Delta_prime
    end

    @testset "solve writes no file" begin
        mktempdir() do dir
            cd(dir) do
                solve(LocalStabilityProblem(equil), Ballooning(; alpha_boundary_scan=false, verbose=false))
            end
            @test isempty(readdir(dir))
        end
    end
end
