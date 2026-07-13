# runtests_islands_configure.jl
#
# Islands M2c — the Level-0 configuration-assembly gates
# (docs/src/islands/design/03 §2; M2c milestone contract, deliverable #1):
#   - configure_level0 builds a well-formed IslandStack + far-field BCs + Δ prefactors;
#   - the CLEARED coefficients are wired faithfully onto the operator stack
#     (c_D ≡ Coefficients.magnetic_drift_frequency node-for-node; the :improved
#     toggle; the pitch diffusivity/deflection shapes; the Δ prefactors);
#   - QUESTIONS Q5 is now fully cleared: every Level-0 operator coefficient is
#     built from `phys` via a cleared `Coefficients.*` builder (no gated inputs),
#     including the full orbit-averaged collision operator (orbit-averaged-collision.md);
#   - the assembled residual runs and Newton–Krylov converges.

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

# ρ̂_θi is order-unity here so the island-streaming a_xi = (inv_Lq/ρ̂_θi)·x stays
# commensurate with the (structural, non-physics) test domain and the naive
# Newton–Krylov converges without the physics preconditioner; the cleared
# coefficient *structure* (the {Ω,g} advection) is ρ̂_θi-independent in form.
_phys(; variant=:original, model=:chandrasekhar) = Cfg.Level0Physics(; epsilon=0.1,
    inv_Lq=1.0, inv_LB=1.0, q_s=2.0, dq_dpsi=0.5, w_psi=0.05, mu0_R=1.0, inv_Ln0=1.0,
    rho_hat_theta_i=1.0, eta_i=0.5, nu_star=0.01, m=2.0, tau=1.0, variant=variant, collision_model=model)

_ion() = [Spc.Species(; name=:i, Z=1.0, m=1.0, background=Spc.Maxwellian(; n=1.0, T=1.0), role=Spc.Bulk)]

@testset "Islands Configure — Level-0 assembly (M2c)" begin
    grid = _grid()
    phys = _phys()
    species = _ion()
    cfg = Cfg.configure_level0(grid, phys, species)

    @testset "well-formed stack + provenance" begin
        @test cfg.stack isa Opc.IslandStack
        # streaming, drift, E×B, pitch D+E, drag A, neoclassical B, cross C, drive
        @test length(cfg.stack.kinetic) == 8
        @test cfg.stack.field isa Opc.Quasineutrality
        @test cfg.bc isa Opc.FarFieldConditions
        # provenance tuples name exactly which coefficients are cleared vs gated
        @test :magnetic_drift in cfg.cleared
        @test :delta_prefactors in cfg.cleared
        @test :quasineutrality in cfg.cleared            # closure now wired (01 §3)
        @test !(:quasineutrality_alpha in cfg.gated)     # no longer a structural gap
        @test :streaming in cfg.cleared                  # island streaming now wired (01 §2)
        @test :gradient_drive in cfg.cleared             # far-field drive, zero source (01 §2)
        @test :far_field in cfg.cleared
        @test :exb in cfg.cleared                        # E×B now wired (01 §2, exb-coupling.md)
        # the full orbit-averaged collision operator (orbit-averaged-collision.md)
        @test :pitch_diffusion in cfg.cleared            # D+E (σ-odd, orbit-averaged P_oa)
        @test :collisional_drag in cfg.cleared           # A (∂_x)
        @test :neoclassical_diffusion in cfg.cleared     # B (∂²_x)
        @test :collisional_cross in cfg.cleared          # C (∂²_xy)
        @test :collision_magnitude in cfg.cleared        # ε^{3/2}ν_★
        @test cfg.gated == ()                            # every L0 operator coefficient is cleared
    end

    @testset "gradient drive = zero source + diamagnetic far field (01 §2)" begin
        nx, nξ, ny, nE, nσ = PSc.nnodes(grid)
        # I19 Formulation A: the interior GradientDrive source is zero
        gd = cfg.stack.kinetic[8]
        @test gd isa Opc.GradientDrive
        @test all(iszero, gd.drive)
        # far field g_far = x·L̂_n0⁻¹·[1+(E−3/2)η_i] at the boundaries, Φ_far = 0
        bc = Cfg.gradient_far_field(grid, phys)
        xL, xR = grid.x.nodes[1], grid.x.nodes[nx]
        for iE in 1:nE
            temp = 1 + (grid.E.nodes[iE] - 1.5) * phys.eta_i
            @test bc.g_left[2, 3, iE, 1] ≈ xL * phys.inv_Ln0 * temp atol = 1e-12
            @test bc.g_right[2, 3, iE, 1] ≈ xR * phys.inv_Ln0 * temp atol = 1e-12
        end
        @test all(iszero, bc.Φ_left) && all(iszero, bc.Φ_right)   # ω_E = 0
        # isotropic in ξ, y, σ (leading order)
        @test bc.g_left[1, 1, 2, 1] == bc.g_left[nξ, ny, 2, nσ]
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

    @testset "CLEARED orbit-averaged pitch diffusion D+E (σ-odd, P_oa=y⟨√(1−yb)⟩)" begin
        nx, nξ, ny, nE, nσ = PSc.nnodes(grid)
        # P_oa = y·S(y), flat measure (wmeas ≡ 1); node-for-node vs the cleared bracket
        P_oa, wmeas = Cfg.pitch_diffusivity_profile(grid, phys)
        @test all(>=(0), P_oa)
        @test all(==(1.0), wmeas)                         # flat measure (divergence form)
        # A4 for the shipped orbit-averaged P_oa: the mimetic K conserves ∫g dy and
        # has the correct entropy sign (dissipative) — for any test g
        K, Wq = Opc.conservative_pitch_operator(grid.y, P_oa, wmeas)
        gtest = sin.(range(0.3, 2.7; length=grid.y.n))
        @test abs(dot(Wq, K * gtest)) < 1e-10             # particle conservation
        @test dot(gtest, Diagonal(Wq) * (K * gtest)) <= 1e-12   # entropy sign ≤ 0
        y_forbidden = (1 + 0.1) / (1 - 0.1)
        for iy in 1:ny
            y = grid.y.nodes[iy]
            (y >= y_forbidden || abs(y - 1.0) < 5e-2) && continue   # forbidden / y_c layer → 0
            S, _ = Coc.orbit_average_pitch_brackets(; y=y, epsilon=0.1)
            @test P_oa[iy] ≈ y * S atol = 1e-10
        end
        # the σ-odd pitch coefficient = −2ν̂ii(1+ε)/(m ρ̂_θ σ√E); reversing σ flips sign
        cp = cfg.stack.kinetic[4].c
        nu_tilde = phys.epsilon^1.5 * phys.nu_star
        for iE in 1:nE
            v̂ = sqrt(grid.E.nodes[iE])
            ν̂ii = Coc.deflection_frequency(v̂; nu_tilde=nu_tilde, model=phys.collision_model)
            want = 2 * ν̂ii * (1 + 0.1) / (phys.m * phys.rho_hat_theta_i * (1) * v̂)  # σ=+1 (+: ÷−mρ̂θ flip)
            @test cp[3, 2, iE, 1] ≈ want atol = 1e-12
            @test cp[3, 2, iE, 1] ≈ -cp[3, 2, iE, 2] atol = 1e-14   # σ-odd
        end
    end

    @testset "CLEARED collision drag / neoclassical / cross (A/B/C)" begin
        nx, nξ, ny, nE, nσ = PSc.nnodes(grid)
        nu_tilde = phys.epsilon^1.5 * phys.nu_star
        a_drag = cfg.stack.kinetic[5].a_x                 # A: −ν̂ii/m, passing-only, σ-even
        c_neo = cfg.stack.kinetic[6].c                    # B: σ-odd, uses T=⟨1/√(1−yb)⟩
        c_cross = cfg.stack.kinetic[7].c                  # C: −2ν̂ii y/m, passing-only, σ-even
        @test cfg.stack.kinetic[5] isa Opc.CollisionalDrag
        @test cfg.stack.kinetic[6] isa Opc.NeoclassicalDiffusion
        @test cfg.stack.kinetic[7] isa Opc.CollisionalCross
        ip = findfirst(y -> 0 < y < 0.9, grid.y.nodes)    # a passing node
        v̂ = sqrt(grid.E.nodes[1]); ν̂ii = Coc.deflection_frequency(v̂; nu_tilde=nu_tilde, model=phys.collision_model)
        @test a_drag[4, 3, ip, 1, 1] ≈ ν̂ii / phys.m atol = 1e-12     # +: ÷−mρ̂θ flip of L23's leading −
        @test a_drag[4, 3, ip, 1, 1] == a_drag[4, 3, ip, 1, 2]        # σ-even
        @test c_cross[4, 3, ip, 1, 1] ≈ 2 * ν̂ii * grid.y.nodes[ip] / phys.m atol = 1e-12
        @test c_cross[4, 3, ip, 1, 1] == c_cross[4, 3, ip, 1, 2]      # σ-even
        # B: neoclassical σ-odd, node-for-node vs the T bracket
        _, T = Coc.orbit_average_pitch_brackets(; y=grid.y.nodes[ip], epsilon=0.1)
        wantB = ν̂ii * (1 * v̂) * phys.rho_hat_theta_i / (2 * phys.m * (1 + 0.1)) * grid.y.nodes[ip] * T
        @test c_neo[4, 3, ip, 1, 1] ≈ wantB atol = 1e-12
        @test c_neo[4, 3, ip, 1, 1] ≈ -c_neo[4, 3, ip, 1, 2] atol = 1e-14   # σ-odd
        # passing-only masks: A, C vanish for trapped
        itrap = findfirst(>=(grid.y_c), grid.y.nodes)
        if itrap !== nothing
            @test all(iszero, @view a_drag[:, :, itrap, :, :])
            @test all(iszero, @view c_cross[:, :, itrap, :, :])
        end
    end

    @testset "CLEARED collision magnitude nu_tilde = ε^{3/2}ν_★ (collision-magnitude.md)" begin
        # the momentum-restoring average reproduces L23 Eq. 4.1.6 to its quoted digits
        nu_avg = Coc.momentum_restoring_average(; epsilon=0.1, nu_star=0.01)
        @test nu_avg ≈ 1.267537e-4 rtol = 1e-5            # L23 p. 88 unit-test value
        @test nu_avg ≈ (4 * 0.1^1.5 * 0.01 / (3 * sqrt(π))) * (sqrt(2) - log(1 + sqrt(2))) atol = 1e-15
        # scales linearly in ν_★ and as ε^{3/2}
        @test Coc.momentum_restoring_average(; epsilon=0.1, nu_star=0.02) ≈ 2 * nu_avg atol = 1e-14
        @test Coc.momentum_restoring_average(; epsilon=0.4, nu_star=0.01) ≈ nu_avg * (0.4 / 0.1)^1.5 atol = 1e-14
        @test Coc.momentum_restoring_average(; epsilon=0.0, nu_star=0.01) == 0.0
    end

    @testset "CLEARED Δ prefactors (symmetric, from delta_moment_prefactors)" begin
        direct = Coc.delta_moment_prefactors(; mu0_R=1.0, w_psi=0.05, dq_dpsi=0.5, q_s=2.0)
        @test cfg.delta_prefactors.cos ≈ direct.cos atol = 1e-9
        @test cfg.delta_prefactors.sin ≈ direct.sin atol = 1e-9
        @test cfg.delta_prefactors.cos ≈ -cfg.delta_prefactors.sin atol = 1e-9   # symmetric pin
    end

    @testset "structural solve converges on the fully-cleared config" begin
        # every operator coefficient is now physical (incl. the full orbit-averaged
        # collision operator); assert the grid-independent per-equation max-norm
        # (04 §5), the √N-scaled L2 floors slightly higher across ~N unknowns.
        f! = Soc.flat_residual(cfg.stack, grid; bc=cfg.bc)
        N = Opc.statelength(grid)
        sol = Soc.newton_krylov(f!, zeros(N); rtol=1e-9, atol=1e-13, max_iter=40, memory=200)
        @test sol.converged
        @test sol.residual_max < 1e-7
        # residual is finite everywhere at a nonzero state
        U = Opc.IslandState(grid)
        Opc.fill_state!(U, 0.3)
        R = Opc.IslandState(grid)
        cache = Opc.IslandCache(grid)
        Opc.residual!(R, U, cfg.stack, grid, cache, cfg.bc)
        @test all(isfinite, R.g) && all(isfinite, R.Φ)
        # allocation regression: the full stack (incl. the new collision operators
        # CollisionalDrag/NeoclassicalDiffusion/CollisionalCross) is allocation-free.
        # Measure through a function barrier so testset-scope boxing isn't counted.
        _ralloc(R, U, st, g, c, b) = (Opc.residual!(R, U, st, g, c, b); @allocated Opc.residual!(R, U, st, g, c, b))
        @test _ralloc(R, U, cfg.stack, grid, cache, cfg.bc) == 0
    end

    @testset "island streaming = advection along Ω (the {Ω,g} structure)" begin
        a_xi, a_x = Cfg.streaming_coefficients(grid, phys)
        nx, nξ, ny, nE, nσ = PSc.nnodes(grid)
        w = phys.w_psi
        pref = phys.inv_Lq * w^2 / (4 * phys.rho_hat_theta_i)
        for iy in 1:ny
            Θ = grid.y.nodes[iy] < grid.y_c ? 1.0 : 0.0
            for iξ in 1:nξ, ix in 1:nx
                x = grid.x.nodes[ix]
                ξ = grid.ξ.nodes[iξ]
                # a_xi must equal pref·Θ·∂ₓΩ = pref·Θ·(4x/w²); a_x = pref·Θ·(−∂_ξΩ) = pref·Θ·(−sinξ)
                @test a_xi[ix, iξ, iy, 1, 1] ≈ pref * Θ * (4 * x / w^2) atol = 1e-12
                @test a_x[ix, iξ, iy, 1, 1] ≈ pref * Θ * (-sin(ξ)) atol = 1e-12
            end
        end
        # passing-only: trapped nodes (y ≥ y_c) carry zero streaming
        itrap = findfirst(>=(grid.y_c), grid.y.nodes)
        if itrap !== nothing
            @test all(iszero, @view a_xi[:, :, itrap, :, :])
            @test all(iszero, @view a_x[:, :, itrap, :, :])
        end
        # E, σ independence (broadcast)
        @test a_xi[3, 2, 2, 1, 1] == a_xi[3, 2, 2, nE, nσ]
    end

    @testset "CLEARED E×B coupling c_E = ½⟨1/v̂_∥⟩ (passing σ-odd, trapped ≡ 0)" begin
        cE = Cfg.exb_coupling_table(grid, phys)
        nx, nξ, ny, nE, nσ = PSc.nnodes(grid)
        # passing nodes (y < y_c, away from the y_c layer): c_E = (σ/2√E)·B₁(y),
        # node-for-node vs the cleared bracket
        for iσ in 1:nσ, iE in 1:nE, iy in 1:ny
            y = grid.y.nodes[iy]
            (y >= grid.y_c) && continue                   # passing-only
            (abs(y - 1.0) < 5e-2) && continue             # skip the gated y_c layer
            v̂ = sqrt(grid.E.nodes[iE])
            σ = grid.σ[iσ]
            B1 = Coc.orbit_average_exb_bracket(; y=y, epsilon=0.1)
            @test cE[4, 3, iy, iE, iσ] ≈ (σ / (2 * v̂)) * B1 atol = 1e-10
        end
        # trapped nodes (y ≥ y_c): c_E ≡ 0 exactly (σ-odd banana-leg cancellation)
        for iy in 1:ny
            if grid.y.nodes[iy] >= grid.y_c
                @test all(iszero, @view cE[:, :, iy, :, :])
            end
        end
        # σ-odd: reversing v_∥ flips the coupling (like the drift x_D ∝ σ)
        ipass = findfirst(y -> 0 < y < 0.9, grid.y.nodes)
        @test ipass !== nothing
        @test cE[4, 3, ipass, 2, 1] ≈ -cE[4, 3, ipass, 2, 2] atol = 1e-12
        @test cE[4, 3, ipass, 2, 1] != 0.0                # nonzero for passing
        # x, ξ independence (broadcast over the solve plane)
        @test cE[2, 5, ipass, 2, 1] == cE[7, 1, ipass, 2, 1]
        # the coupling is the velocity-dependent (array) ExBDrift coefficient in the stack
        @test cfg.stack.kinetic[3] isa Opc.ExBDrift
        @test cfg.stack.kinetic[3].c_E isa AbstractArray  # array, not scalar (velocity-dependent)
        @test cfg.stack.kinetic[3].c_E == cE
    end

    @testset "quasineutrality drive makes Φ nonzero (the Q5 field fix)" begin
        # the cleared L̂_{n0}⁻¹(x−ĥ) source drives Φ; without it Φ collapses to 0.
        S = Cfg.quasineutrality_source(grid, phys)
        @test any(!iszero, S)                            # the drive is nontrivial
        @test cfg.stack.field.source !== nothing         # wired into the operator
        @test cfg.stack.field.α ≈ (phys.tau + 1) / phys.tau  # α = (τ+1)/τ, not τ/(τ+1)
        # solved Φ is nonzero in the interior (boundaries pinned by the far field)
        f! = Soc.flat_residual(cfg.stack, grid; bc=cfg.bc)
        N = Opc.statelength(grid)
        sol = Soc.newton_krylov(f!, zeros(N); rtol=1e-9, atol=1e-13, max_iter=40, memory=200)
        Usol = Opc.IslandState(grid)
        Opc.unflatten!(Usol, sol.u)
        @test maximum(abs, Usol.Φ) > 1e-6                # Φ no longer trivially zero
    end

    @testset "assembly validates the species list" begin
        @test_throws ArgumentError Cfg.configure_level0(grid, phys, Spc.Species[])
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
