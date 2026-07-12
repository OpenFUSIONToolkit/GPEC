# runtests_islands_configure.jl
#
# Islands M2c — the Level-0 configuration-assembly gates
# (docs/src/islands/design/03 §2; M2c milestone contract, deliverable #1):
#   - configure_level0 builds a well-formed IslandStack + far-field BCs + Δ prefactors;
#   - the CLEARED coefficients are wired faithfully onto the operator stack
#     (c_D ≡ Coefficients.magnetic_drift_frequency node-for-node; the :improved
#     toggle; the pitch diffusivity/deflection shapes; the Δ prefactors);
#   - the still-gated coefficient families (QUESTIONS Q5) are the supplied inputs;
#   - the assembled residual runs and Newton–Krylov converges STRUCTURALLY on the
#     documented non-physics placeholder config (never a physics result — the
#     gated coefficients are uncleared).
#
# STRUCTURAL check: the placeholder gated inputs are explicitly non-physics
# (Configure.level0_placeholders); a physics run supplies cleared inputs once the
# Q5 derivation lane clears the remaining coefficient families.

using LinearAlgebra
using Test

const IslC = GeneralizedPerturbedEquilibrium.Islands
const PSc = IslC.PhaseSpace
const Opc = IslC.Operators
const Soc = IslC.Solvers
const Coc = IslC.Coefficients
const Spc = IslC.SpeciesLists
const Cfg = IslC.Configure

# small physical grid: y_max spans the full pitch domain (forbidden region +
# y_c layer) so the assembly's robustness there is exercised.
_grid() = PSc.IslandGrid(; nx=9, nxi=8, ny=9, nE=3, halfwidth_x=6.0, clustering_x=1.0,
    y_max=4.0, y_c=1.0, clustering_y=0.8, order=4)

_phys(; variant=:original, model=:chandrasekhar) = Cfg.Level0Physics(; epsilon=0.1,
    inv_Lq=1.0, inv_LB=1.0, q_s=2.0, dq_dpsi=0.5, w_psi=0.05, mu0_R=1.0, tau=1.0,
    variant=variant, collision_model=model)

_ion() = [Spc.Species(; name=:i, Z=1.0, m=1.0, background=Spc.Maxwellian(; n=1.0, T=1.0), role=Spc.Bulk)]

@testset "Islands Configure — Level-0 assembly (M2c)" begin
    grid = _grid()
    phys = _phys()
    species = _ion()
    gated = Cfg.level0_placeholders(grid)
    cfg = Cfg.configure_level0(grid, phys, species; gated=gated)

    @testset "well-formed stack + provenance" begin
        @test cfg.stack isa Opc.IslandStack
        @test length(cfg.stack.kinetic) == 5              # streaming, drift, E×B, collisions, drive
        @test cfg.stack.field isa Opc.Quasineutrality
        @test cfg.bc isa Opc.FarFieldConditions
        # provenance tuples name exactly which coefficients are cleared vs gated
        @test :magnetic_drift in cfg.cleared
        @test :delta_prefactors in cfg.cleared
        @test :quasineutrality_alpha in cfg.gated        # QUESTIONS Q5 structural gap
        @test :streaming in cfg.gated
    end

    @testset "CLEARED c_D wired faithfully vs magnetic_drift_frequency" begin
        cD = Cfg.drift_coefficient_table(grid, phys)
        nx, nξ, ny, nE, nσ = PSc.nnodes(grid)
        y_forbidden = (1 + 0.1) / (1 - 0.1)
        # every well-defined node equals the direct cleared call, node-for-node
        for iσ in 1:nσ, iE in 1:nE, iy in 1:ny
            y = grid.y.nodes[iy]
            (y >= y_forbidden) && continue                # forbidden region ⇒ 0 (no particles)
            (abs(y - 1.0) < 5e-2) && continue             # skip the gated y_c layer
            v̂ = sqrt(grid.E.nodes[iE])
            σ = grid.σ[iσ]
            direct = Coc.magnetic_drift_frequency(; y=y, v_hat=v̂, sigma=σ, epsilon=0.1,
                inv_Lq=1.0, inv_LB=1.0, variant=:original)
            @test cD[4, 3, iy, iE, iσ] ≈ direct atol = 1e-12
        end
        # forbidden region carries no particles ⇒ c_D ≡ 0
        for iy in 1:ny
            if grid.y.nodes[iy] >= y_forbidden
                @test all(iszero, @view cD[:, :, iy, :, :])
            end
        end
        # σ-odd: reversing the sign of v_∥ flips the drift
        @test cD[4, 3, 2, 2, 1] ≈ -cD[4, 3, 2, 2, 2] atol = 1e-12
    end

    @testset ":improved drift toggle zeroes the ∇B term" begin
        cD_orig = Cfg.drift_coefficient_table(grid, _phys(; variant=:original))
        cD_imp = Cfg.drift_coefficient_table(grid, _phys(; variant=:improved))
        iy, iE, iσ = 2, 2, 1
        y = grid.y.nodes[iy]
        v̂ = sqrt(grid.E.nodes[iE])
        σ = grid.σ[iσ]
        A, G = Coc.orbit_average_drift_brackets(; y=y, epsilon=0.1)
        @test cD_imp[4, 3, iy, iE, iσ] ≈ (σ * v̂ / 1.1) * (1.0 * A) atol = 1e-10   # LB → 0
        @test cD_orig[4, 3, iy, iE, iσ] ≈ (σ * v̂ / 1.1) * (1.0 * A - 0.5 * 1.0 * G) atol = 1e-10
        @test cD_imp[4, 3, iy, iE, iσ] != cD_orig[4, 3, iy, iE, iσ]               # the toggle bites
    end

    @testset "CLEARED collision shapes wired from Coefficients" begin
        # c is y-independent, energy-dependent = nu_tilde · deflection_frequency(√E)
        c = Cfg.collision_coefficient(grid, phys, 1.0)
        nx, nξ, nE, nσ = size(c)
        for iE in 1:nE
            v̂ = sqrt(grid.E.nodes[iE])
            ν = Coc.deflection_frequency(v̂; nu_tilde=1.0, model=:chandrasekhar)
            @test c[3, 2, iE, 1] ≈ ν atol = 1e-12
        end
        # P profile from the cleared pitch_diffusivity, in-domain and nonnegative
        P, wmeas = Cfg.pitch_diffusivity_profile(grid, gated.B_profile)
        @test all(>=(0), P)
        @test all(>(0), wmeas)
        iy = 4
        λ = grid.y.nodes[iy]
        @test P[iy] ≈ Coc.pitch_diffusivity(λ, gated.B_profile[iy]) atol = 1e-12
    end

    @testset "CLEARED Δ prefactors (symmetric, from delta_moment_prefactors)" begin
        direct = Coc.delta_moment_prefactors(; mu0_R=1.0, w_psi=0.05, dq_dpsi=0.5, q_s=2.0)
        @test cfg.delta_prefactors.cos ≈ direct.cos atol = 1e-9
        @test cfg.delta_prefactors.sin ≈ direct.sin atol = 1e-9
        @test cfg.delta_prefactors.cos ≈ -cfg.delta_prefactors.sin atol = 1e-9   # symmetric pin
    end

    @testset "structural solve converges on the placeholder config" begin
        f! = Soc.flat_residual(cfg.stack, grid; bc=cfg.bc)
        N = Opc.statelength(grid)
        sol = Soc.newton_krylov(f!, zeros(N); rtol=1e-8, atol=1e-12, max_iter=30, memory=200)
        @test sol.converged
        @test sol.residual_norms[end] < 1e-7
        # residual is finite everywhere at a nonzero state
        U = Opc.IslandState(grid)
        Opc.fill_state!(U, 0.3)
        R = Opc.IslandState(grid)
        cache = Opc.IslandCache(grid)
        Opc.residual!(R, U, cfg.stack, grid, cache, cfg.bc)
        @test all(isfinite, R.g) && all(isfinite, R.Φ)
    end

    @testset "assembly validates the species list" begin
        @test_throws ArgumentError Cfg.configure_level0(grid, phys, Spc.Species[]; gated=gated)
    end
end

@testset "Islands anchor-sync (docs/07 §1.1)" begin
    Vfy = IslC.Verify
    ops_file = normpath(joinpath(@__DIR__, "..", "src", "Islands", "operators", "Operators.jl"))

    @testset "the operator stack and the as-implemented docs are in sync" begin
        r = Vfy.check_anchor_sync()
        @test isempty(r.undocumented)      # every AbstractTerm operator is documented
        @test isempty(r.dangling)          # every `Implemented by:` symbol resolves
    end

    @testset "the check CATCHES drift (negative controls)" begin
        # a doc missing an operator ⇒ that operator is flagged undocumented
        missing_doc = tempname() * ".md"
        write(missing_doc, "Implemented by: `Operators.MagneticDrift`.\n")
        r1 = Vfy.check_anchor_sync(; docfiles=[missing_doc])
        @test "ParallelStreaming" in r1.undocumented
        @test isempty(r1.dangling)         # the one symbol it names does resolve
        rm(missing_doc; force=true)

        # a doc naming a nonexistent symbol ⇒ that symbol is flagged dangling
        bogus_doc = tempname() * ".md"
        write(bogus_doc, "Implemented by: `Operators.NotARealOperator`.\n")
        r2 = Vfy.check_anchor_sync(; docfiles=[bogus_doc])
        @test "Operators.NotARealOperator" in r2.dangling
        rm(bogus_doc; force=true)
    end
end
