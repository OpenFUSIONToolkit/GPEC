using HDF5
using TOML

# The LocalStability stage on the coarse Solovev fixture deck (mpsi=16): the named profiles of
# solve(LocalStabilityProblem, Ballooning()) against their kernels.
@testset "LocalStability" begin
    GPEC = GeneralizedPerturbedEquilibrium
    LS = GPEC.LocalStability
    template = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
    deck = TOML.parsefile(joinpath(template, "gpec.toml"))
    equil = GPEC.Equilibrium.PlasmaEquilibrium(
        GPEC.equilibrium_config(deck, template),
        GPEC.analytic_config(deck)
    )
    xs = equil.profiles.xs
    res = solve(LocalStabilityProblem(equil), Ballooning())

    @testset "problem settings reach the kernels" begin
        problem = LocalStabilityProblem(equil; theta_k=0.25, alpha_boundary_scan=false)
        shifted = solve(problem, Ballooning(); verbose=false)
        expected = zeros(Float64, length(xs), 5)
        LS.compute_ballooning_stability!(expected, equil; theta_k=problem.theta_k, ode_tol=1e-8, verbose=false)
        @test shifted.ballooning_Delta_prime == expected[:, 4]
        @test isempty(shifted.alpha_critical)
        @test_throws MethodError Ballooning(; theta_k=0.25)
        @test_throws MethodError Ballooning(; alpha_boundary_scan=false)
        @test_throws MethodError Ballooning(; verbose=true)
    end

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
        quick = solve(LocalStabilityProblem(equil; alpha_boundary_scan=false), Ballooning())
        @test isempty(quick.ballooning_psi) && isempty(quick.alpha) && isempty(quick.alpha_critical)
        @test quick.D_I == res.D_I
        @test quick.ballooning_Delta_prime == res.ballooning_Delta_prime
    end

    @testset "numerical knobs reach the kernels" begin
        narrow = solve(LocalStabilityProblem(equil), Ballooning(; psi_window=(0.3, 0.7)); verbose=false)
        inside = [0.3 <= p <= 0.7 for p in xs]
        @test all(isfinite, narrow.alpha[inside])
        @test all(isnan, narrow.alpha[.!inside])
        @test narrow.alpha[inside] == res.alpha[inside]

        loose = solve(LocalStabilityProblem(equil; alpha_boundary_scan=false), Ballooning(; ode_tol=1e-6))
        @test loose.D_I == res.D_I
        @test loose.ballooning_Delta_prime ≈ res.ballooning_Delta_prime rtol = 1e-3
        @test loose.ballooning_Delta_prime != res.ballooning_Delta_prime
    end

    @testset "solve writes no file" begin
        mktempdir() do dir
            cd(dir) do
                solve(LocalStabilityProblem(equil; alpha_boundary_scan=false), Ballooning())
            end
            @test isempty(readdir(dir))
        end
    end

    # Copy the fixture deck into `dir` with HDF5 output on, no [LocalStability] section, and the
    # given `[ForceFreeStates]` keys and extra sections.
    function _deck(dir; ffs_extra=Dict{String,Any}(), sections=Dict{String,Any}())
        for name in readdir(template)
            cp(joinpath(template, name), joinpath(dir, name))
        end
        toml_path = joinpath(dir, "gpec.toml")
        inputs = TOML.parsefile(toml_path)
        foreach(section -> delete!(inputs, section), ("LocalStability", "Ballooning"))
        merge!(inputs["ForceFreeStates"], ffs_extra)
        merge!(inputs, sections)
        open(io -> TOML.print(io, inputs), toml_path, "w")
        return dir
    end
    _h5_local_stability(dir) =
        h5open(joinpath(dir, "gpec.h5"), "r") do f
            Dict(k => read(f, "LocalStability/$k") for k in keys(f["LocalStability"]))
        end
    ls_section = Dict{String,Any}("LocalStability" => Dict{String,Any}())

    @testset "deck section and solve give identical results" begin
        mktempdir() do dir
            run = GPEC.main([_deck(dir; sections=ls_section)])
            for name in fieldnames(LocalStabilityResult)
                name === :di_psi && continue
                @test isequal(getfield(run.local_stability, name), getfield(res, name))
            end
            h5 = _h5_local_stability(dir)
            @test h5["D_I"] == res.D_I
            @test isequal(h5["alpha_critical"], res.alpha_critical)
            sing_di = h5open(f -> read(f, "SingularSurfaces/D_I"), joinpath(dir, "gpec.h5"))
            @test !isempty(sing_di)
            @test sing_di == [LS.mercier_D_I(res, s.psifac) for s in run.ffs.surfaces]
        end
    end

    @testset "deck preserves problem settings" begin
        section = Dict{String,Any}("theta_k" => 0.25, "alpha_boundary_scan" => false)
        expected = solve(LocalStabilityProblem(equil; theta_k=0.25, alpha_boundary_scan=false), Ballooning())
        mktempdir() do dir
            run = GPEC.main([_deck(dir; sections=Dict("LocalStability" => section))])
            @test run.local_stability.ballooning_Delta_prime == expected.ballooning_Delta_prime
            @test isempty(run.local_stability.alpha_critical)
        end
    end

    @testset "no section, no stage" begin
        mktempdir() do dir
            run = GPEC.main([_deck(dir)])
            @test run.local_stability === nothing
            @test all(isempty, values(_h5_local_stability(dir)))
        end
    end

    @testset "deck section parsing" begin
        parse(sections...) = GPEC.local_stability_from_deck(Dict{String,Any}(sections...))
        @test_throws ArgumentError parse("LocalStability" => Dict{String,Any}(), "Ballooning" => Dict{String,Any}("n_scna" => 32))
        @test_throws ArgumentError parse("LocalStability" => Dict{String,Any}("n_scan" => 32))
        @test_throws ErrorException parse("Ballooning" => Dict{String,Any}("n_scan" => 32))
        alg = parse("LocalStability" => Dict{String,Any}(), "Ballooning" => Dict{String,Any}("psi_window" => [0.2, 0.8], "n_scan" => 32)).algorithm
        @test alg.psi_window === (0.2, 0.8)
        @test alg.n_scan == 32
        @test parse() === nothing
        ls = parse("LocalStability" => Dict{String,Any}("theta_k" => 0.25, "alpha_boundary_scan" => false))
        @test ls.algorithm == Ballooning()
        @test ls.problem == Dict{Symbol,Any}(:theta_k => 0.25, :alpha_boundary_scan => false)
    end
end
