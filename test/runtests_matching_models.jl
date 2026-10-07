using TOML

# Inner-layer model capability gates and option forwarding, and the shared layer-parameter
# builder, which must reproduce the resistivity/density closures from `equil.kinetic`.
@testset "Matching models and layer parameters" begin
    GPEC = GeneralizedPerturbedEquilibrium
    FFS = GPEC.ForceFreeStates
    NCR = GPEC.Utilities.NeoclassicalResistivity
    E_CHG = GPEC.Utilities.PhysicalConstants.E_CHG
    M_P = GPEC.Utilities.PhysicalConstants.M_P

    @testset "model capability gates and option forwarding" begin
        IL = GPEC.InnerLayer
        @test IL.GGJModel() isa IL.GGJModel{:ray}
        @test FFS.closure_capable(IL.GGJModel())
        @test !FFS.closure_capable(IL.SLAYERModel())
        @test IL.GGJModel(; solver=:galerkin, nx=640).options == (nx=640,)
        # Carried options reach the backend exactly as call-site keywords do.
        p = IL.glasser_wang_2020_eq55()
        γ = 1e-3im
        @test IL.solve_inner(IL.GGJModel(; solver=:galerkin, nx=256), p, γ) == IL.solve_inner(IL.GGJModel(; solver=:galerkin), p, γ; nx=256)
        @test IL.solve_inner(IL.GGJModel(; solver=:galerkin, nx=256), p, γ; nx=512) == IL.solve_inner(IL.GGJModel(; solver=:galerkin), p, γ)
    end

    @testset "override precedence needs no kinetic data" begin
        fake_surfaces = [(psifac=0.3,), (psifac=0.6,)]
        out = FFS.layer_parameters(fake_surfaces, nothing;
            eta=[1e-7, 2e-7], rho=[1e-7, 1e-7], rotation=[0.0, 100.0])
        @test out.eta == [1e-7, 2e-7]
        @test out.rho == [1e-7, 1e-7]
        @test out.rotation == [0.0, 100.0]
        @test_throws ErrorException FFS.layer_parameters(fake_surfaces, nothing;
            eta=[1e-7], rho=[1e-7, 1e-7], rotation=[0.0, 0.0])
    end

    @testset "derivation from equil.kinetic matches the shared closures" begin
        template = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
        deck = TOML.parsefile(joinpath(template, "gpec.toml"))
        equil = GPEC.Equilibrium.setup_equilibrium(
            GPEC.Equilibrium.EquilibriumConfig(deck["Equilibrium"], template),
            GPEC.Equilibrium.SolovevConfig(deck["SOL_INPUT"])
        )
        ffs_kwargs = Dict(Symbol(k) => v for (k, v) in deck["ForceFreeStates"]
                          if !(k in ("integrator", "nn_low", "nn_high")))
        kin_file = joinpath(@__DIR__, "..", "examples", "Solovev_kinetic_NTV_example", "kinetic.dat")
        attach_kinetic_profiles!(equil, kin_file; zi=1)

        mktempdir() do dir
            ffs = solve(equil, Forward(); nn=1, dir_path=dir, ffs_kwargs...)
            surfaces = ffs.surfaces
            @test !isempty(surfaces)

            # A derivation without kinetic data must fail loudly.
            bare = GPEC.Equilibrium.setup_equilibrium(
                GPEC.Equilibrium.EquilibriumConfig(deck["Equilibrium"], template),
                GPEC.Equilibrium.SolovevConfig(deck["SOL_INPUT"])
            )
            @test_throws ErrorException FFS.layer_parameters(surfaces, bare)

            kp = ffs.equil.kinetic
            out = FFS.layer_parameters(surfaces, ffs.equil)
            for (k, sing) in enumerate(surfaces)
                ψ = sing.psifac
                n_e = kp.ne_spline(ψ)
                t_e = kp.Te_spline(ψ) / E_CHG
                lnLamb = NCR.coulomb_log_e(n_e, t_e; form=:nrl)
                @test out.eta[k] ≈ NCR.eta_spitzer(n_e, t_e, 1.0; lnLamb=lnLamb) rtol = 1e-12
                @test out.rho[k] ≈ 2.0 * M_P * n_e rtol = 1e-12
                @test out.rotation[k] ≈ kp.omegaE_spline(ψ) / (2π) rtol = 1e-12
            end

            # Partial override: eta passed through verbatim, the rest still derived.
            mixed = FFS.layer_parameters(surfaces, ffs.equil; eta=fill(3e-8, length(surfaces)))
            @test all(mixed.eta .== 3e-8)
            @test mixed.rho == out.rho

            # The neoclassical closure reads the surface's trapped fraction and stays physical.
            neo = FFS.layer_parameters(surfaces, ffs.equil; resistivity_model=NCR.SauterNeoModel())
            @test all(isfinite, neo.eta) && all(>(0), neo.eta)
            @test neo.eta != out.eta
        end
    end
end
