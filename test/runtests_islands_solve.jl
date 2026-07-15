# runtests_islands_solve.jl
#
# Islands M2 — Level-0 solve machinery, structural verification gates
# (docs/src/islands/design/05 §A; the M2 milestone contract):
#   A5  zero-drive null: g ≡ 0 ⇒ residual = machine zero; Newton converges trivially.
#   A1+ assembled solve-MMS: Newton–Krylov recovers the manufactured state at design order.
#   A8  y_c matching-block smallest-singular-value conditioning monitor.
#   A4  (L0) exact discrete particle conservation + entropy sign of the mimetic
#       pitch-angle collision operator.
#   A3  Δ_cos even / Δ_sin odd parity under ξ-reflection (manufactured J̄_∥).
#   A7  the coefficient-free closure identity ⟨∂²h/∂x²⟩_Ω = 0.
#   +   preconditioner quality (GMRES iteration reduction), far-field BC rows,
#       pseudo-arclength fold detection, species/frames plumbing.
#
# STRUCTURAL (pre-physics) checks: manufactured order-unity coefficients only.
# Every physics coefficient in src/ is a [VERIFY]-gated supplied parameter
# (QUESTIONS Q2–Q4); the York-gate physics benchmarks stay skipped in
# benchmarks/islands/ until human clearance.

using LinearAlgebra

const IslM2 = GeneralizedPerturbedEquilibrium.Islands
const PS2 = IslM2.PhaseSpace
const Op2 = IslM2.Operators
const So2 = IslM2.Solvers
const V2 = IslM2.Verify
const Mo2 = IslM2.Moments
const Fi2 = IslM2.Fields
const Sp2 = IslM2.SpeciesLists
const Fr2 = IslM2.Frames

_sgrid(n; ny=n) = PS2.IslandGrid(; nx=n, nxi=8, ny=ny, nE=2, halfwidth_x=6.0, clustering_x=1.0,
    y_max=4.0, y_c=1.0, clustering_y=0.8, order=4)

@testset "Islands L0 solve machinery (M2)" begin

    @testset "species plumbing (02 §1, D3)" begin
        bg = Sp2.Maxwellian(; n=1.0, T=1.0, dlnn_dr=-1.0)
        ion = Sp2.Species(; name=:D, Z=1.0, m=1.0, background=bg, role=Sp2.Bulk)
        trc = Sp2.Species(; name=:Dtrace, Z=1.0, m=1.0, background=Sp2.Maxwellian(; n=1e-4, T=1.0), role=Sp2.Trace)
        @test Sp2.validate_species([ion, trc]) == [ion, trc]
        @test Sp2.is_bulk(ion) && Sp2.is_trace(trc)
        @test Sp2.bulk_species([ion, trc]) == [ion]
        # the L0 test pair: trace deuterium copy of the bulk passes the criteria
        @test isempty(Sp2.check_trace_criteria([ion, trc]))
        # a heavy trace at bulk-like density violates and is flagged (warn, never degrade)
        w = Sp2.Species(; name=:W, Z=40.0, m=92.0, background=Sp2.Maxwellian(; n=0.01, T=1.0), role=Sp2.Trace)
        @test Sp2.check_trace_criteria([ion, w]) == [:W]
        # structural validation
        @test_throws ArgumentError Sp2.validate_species([trc])            # no Bulk
        @test_throws ArgumentError Sp2.validate_species([ion, ion])       # duplicate names
    end

    @testset "frames: gated conventions poison, mechanics work (01 §5)" begin
        conv = Fr2.FrameConvention()                    # all NaN until human-cleared (Q3)
        @test !Fr2.is_cleared(conv)
        @test isnan(Fr2.omega_dia_form(2, 1.0, -1.0, 2.0, conv))
        @test isnan(Fr2.effective_dlnn_form(-1.0, 1.0, 0.5, conv))
        # the frame-invariant combination is pure bookkeeping
        @test Fr2.frame_shift(1.7, 0.4) ≈ 1.3
        # parameter-vector validation
        p = Fr2.Level0Parameters(; w_hat=1.0, omega_E_hat=0.0, epsilon=0.1, inv_Lq_hat=1.0, q_s=2.0)
        @test p.tau == 1.0
        @test_throws ArgumentError Fr2.Level0Parameters(-1.0, 0.0, 0.1, 1.0, 2.0, 1.0, Dict{Symbol,Float64}())
    end

    @testset "A4 — mimetic pitch operator: exact conservation + entropy sign" begin
        g = _sgrid(9)
        P = @. g.y.nodes * (4.0 - g.y.nodes) + 0.1     # arbitrary positive test profile
        wm = @. 1.0 + 0.1 * g.y.nodes
        K, Wq = Op2.conservative_pitch_operator(g.y, P, wm)
        for gv in (sin.(g.y.nodes) .+ 0.3 .* g.y.nodes .^ 2, exp.(-g.y.nodes), g.y.nodes)
            @test abs(dot(Wq, K * gv)) < 1e-11          # particle conservation, machine level
            @test dot(gv .* Wq, K * gv) <= 1e-13        # entropy sign (0 only for constants)
        end
        @test abs(dot(ones(g.y.n) .* Wq, K * ones(g.y.n))) < 1e-12   # constants in the null space
        @test_throws ArgumentError Op2.conservative_pitch_operator(g.y, -P, wm)
        # the term applies c ⋅ (K g) and is allocation-free
        nx, nξ, ny, nE, nσ = PS2.nnodes(g)
        c4 = fill(2.0, nx, nξ, nE, nσ)
        term = Op2.PitchAngleDiffusion(K, c4)
        U, = V2.manufactured_state(g)
        R = Op2.IslandState(g)
        cache = Op2.IslandCache(g)
        Op2.apply!(R, term, U, g, cache)
        @test R.g[3, 2, :, 1, 1] ≈ 2.0 .* (K * U.g[3, 2, :, 1, 1])
        @test (@allocated Op2.apply!(R, term, U, g, cache)) == 0
    end

    @testset "A5 — zero-drive null test" begin
        g = _sgrid(7; ny=7)
        setup = V2.zero_drive_setup(g)
        r = ones(setup.N)
        setup.f(r, zeros(setup.N))
        @test maximum(abs, r) == 0.0                    # residual is EXACTLY machine zero
        # Newton from a small perturbation falls back to the zero state
        sol = So2.newton_krylov(setup.f, 1e-3 .* sin.(1:setup.N); rtol=1e-12, atol=1e-12)
        @test sol.converged
        @test norm(sol.u) < 1e-9
    end

    @testset "A1 — assembled solve-MMS at design order" begin
        r17 = V2.solve_mms(17)
        r33 = V2.solve_mms(33)
        @test r17.converged && r33.converged
        # solution error against the manufactured state converges at design order
        @test log(r17.err / r33.err) / log(33 / 17) > 3.3
        @test r33.err < 5e-3
    end

    @testset "preconditioner: y-block Jacobi with TSVD cuts GMRES iterations (04 §5)" begin
        g = _sgrid(9)
        nx, nξ, ny, nE, nσ = PS2.nnodes(g)
        P = @. g.y.nodes * (4.0 - g.y.nodes)            # degenerate endpoints: zero-flux built in
        K, = Op2.conservative_pitch_operator(g.y, P, ones(ny))
        cstiff = fill(30.0, nx, nξ, nE, nσ)             # stiff collisional pencil
        shift = fill(-1.0, nx, nξ, ny, nE, nσ)
        stack = Op2.IslandStack((Op2.PitchAngleDiffusion(K, cstiff), Op2.RadiationSink(shift)),
            Op2.Quasineutrality(1.3))
        f0! = So2.flat_residual(stack, g)
        N = Op2.statelength(g)
        b = sin.((1:N) ./ 7)
        f!(out, u) = (f0!(out, u); out .-= b; out)
        pc = So2.YBlockJacobi(g, (ix, iξ, iE, iσ) -> I(ny) + cstiff[ix, iξ, iE, iσ] .* K; phi_scale=-1.3)
        s0 = So2.newton_krylov(f!, zeros(N); rtol=1e-10, memory=300)
        s1 = So2.newton_krylov(f!, zeros(N); rtol=1e-10, memory=300, precond=pc)
        @test s0.converged && s1.converged
        @test maximum(abs, s0.u .- s1.u) < 1e-8         # same solution
        @test s1.gmres_iters < s0.gmres_iters ÷ 2       # preconditioner earns its keep
    end

    @testset "newton_direct: exact-solve Newton matches newton_krylov + robust when it stalls" begin
        g = _sgrid(9)
        nx, nξ, ny, nE, nσ = PS2.nnodes(g)
        P = @. g.y.nodes * (4.0 - g.y.nodes)
        K, = Op2.conservative_pitch_operator(g.y, P, ones(ny))
        cstiff = fill(30.0, nx, nξ, nE, nσ)
        shift = fill(-1.0, nx, nξ, ny, nE, nσ)
        stack = Op2.IslandStack((Op2.PitchAngleDiffusion(K, cstiff), Op2.RadiationSink(shift)),
            Op2.Quasineutrality(1.3))
        f0! = So2.flat_residual(stack, g)
        N = Op2.statelength(g)
        b = sin.((1:N) ./ 7)
        f!(out, u) = (f0!(out, u); out .-= b; out)
        sd = So2.newton_direct(f!, zeros(N); rtol=1e-10, atol=1e-13)
        sk = So2.newton_krylov(f!, zeros(N); rtol=1e-10, memory=300)
        @test sd.converged
        @test sd.gmres_iters == 0                          # direct solve, no Krylov
        @test maximum(abs, sd.u .- sk.u) < 1e-7            # same solution as matrix-free
        @test sd.iterations <= 8                           # quadratic: few Newton steps
        # robust on an ill-conditioned advective stack where naive GMRES needs a preconditioner:
        # a stiff streaming coefficient (large a_xi) that conditions J poorly
        astiff = fill(0.0, nx, nξ, ny, nE, nσ)
        axstiff = [40.0 * g.x.nodes[ix] for ix in 1:nx, iξ in 1:nξ, iy in 1:ny, iE in 1:nE, iσ in 1:nσ]
        stack2 = Op2.IslandStack((Op2.ParallelStreaming(astiff, axstiff),
                Op2.PitchAngleDiffusion(K, cstiff), Op2.RadiationSink(shift)), Op2.Quasineutrality(1.3))
        g2!(out, u) = (So2.flat_residual(stack2, g)(out, u); out .-= b; out)
        sd2 = So2.newton_direct(g2!, zeros(N); rtol=1e-9, atol=1e-12)
        @test sd2.converged                                # exact solve converges regardless of conditioning
    end

    @testset "far-field BCs replace the boundary residual rows (01 §3)" begin
        g = _sgrid(7; ny=7)
        nx, nξ, ny, nE, nσ = PS2.nnodes(g)
        U, = V2.manufactured_state(g)
        bc = Op2.FarFieldConditions(0.1 .+ zeros(nξ, ny, nE, nσ), 0.2 .+ zeros(nξ, ny, nE, nσ),
            fill(0.3, nξ), fill(0.4, nξ))
        stack = V2.build_stack(g)
        R = Op2.IslandState(g)
        cache = Op2.IslandCache(g)
        Op2.residual!(R, U, stack, g, cache, bc)
        @test R.g[1, 2, 3, 1, 1] ≈ U.g[1, 2, 3, 1, 1] - 0.1
        @test R.g[nx, 2, 3, 1, 2] ≈ U.g[nx, 2, 3, 1, 2] - 0.2
        @test R.Φ[1, 5] ≈ U.Φ[1, 5] - 0.3
        @test R.Φ[nx, 5] ≈ U.Φ[nx, 5] - 0.4
        # interior rows are untouched by the BC application
        R2 = Op2.IslandState(g)
        Op2.residual!(R2, U, stack, g, cache)
        @test R.g[2:(nx-1), :, :, :, :] == R2.g[2:(nx-1), :, :, :, :]
    end

    @testset "A8 — y_c matching-block conditioning monitor" begin
        g = _sgrid(7; ny=7)
        setup = V2.zero_drive_setup(g)
        J = So2.dense_jacobian(setup.f, zeros(setup.N))
        mon = V2.yc_block_sigma_min(J, g)
        @test isfinite(mon.sigma_min) && mon.sigma_min > 0
        @test all(mon.pencil .>= 1)
        # the monitor detects an artificially singularized pencil block (the
        # silent-noise regression of L23 §4.2 must be *tested for*)
        idx = [Op2.g_flat_index(g, 1, 1, iy, 1, 1) for iy in 3:5]
        Jsing = copy(J)
        Jsing[idx, :] .= 0.0
        @test V2.yc_block_sigma_min(Jsing, g).sigma_min < 1e-14
    end

    @testset "A3 — Δ_cos even / Δ_sin odd under ξ-reflection" begin
        g = _sgrid(9)
        J = [exp(-x^2) * (2.0 + 1.5 * cos(ξ) + 0.7 * sin(ξ)) for x in g.x.nodes, ξ in g.ξ.nodes]
        Jr = [exp(-x^2) * (2.0 + 1.5 * cos(-ξ) + 0.7 * sin(-ξ)) for x in g.x.nodes, ξ in g.ξ.nodes]
        d = Mo2.delta_moments(J, g; prefactor_cos=1.0, prefactor_sin=1.0)
        dr = Mo2.delta_moments(Jr, g; prefactor_cos=1.0, prefactor_sin=1.0)
        @test d.Δcos ≈ dr.Δcos atol = 1e-12             # even
        @test d.Δsin ≈ -dr.Δsin atol = 1e-12            # odd
        @test abs(d.Δsin) > 0.1                         # the projection actually sees the sin part
        # ξ-projection is spectrally exact: a pure cos ξ current has zero sin moment
        Jc = [cos(ξ) for x in g.x.nodes, ξ in g.ξ.nodes]
        @test abs(Mo2.delta_moments(Jc, g; prefactor_cos=1.0, prefactor_sin=1.0).Δsin) < 1e-13
    end

    @testset "moments: J̄_∥ assembly and Ω-average diagnostics" begin
        g = _sgrid(9)
        nx, nξ, ny, nE, nσ = PS2.nnodes(g)
        U, = V2.manufactured_state(g)
        W = ones(ny, nE, nσ)
        ion = Sp2.Species(; name=:D, Z=1.0, m=1.0, background=Sp2.Maxwellian(; n=1.0, T=1.0), role=Sp2.Bulk)
        anti = Sp2.Species(; name=:A, Z=-1.0, m=1.0, background=Sp2.Maxwellian(; n=1.0, T=1.0), role=Sp2.Bulk)
        Jp = zeros(nx, nξ)
        # equal & opposite charges with identical g cancel exactly
        Mo2.parallel_current!(Jp, [U.g, U.g], [ion, anti], [W, W], g)
        @test maximum(abs, Jp) < 1e-14
        # single species with W ≡ 1 reduces to the plain velocity moment
        Mo2.parallel_current!(Jp, [U.g], [ion], [W], g)
        Mref = zeros(nx, nξ)
        Op2.velocity_moment!(Mref, U.g, g)
        @test Jp ≈ Mref
        # ⟨1⟩_Ω = 1 outside and inside the separatrix; constant J has no polarization part
        @test Mo2.omega_average((x, ξ) -> 1.0, 1.5, 1.0) ≈ 1.0 rtol = 1e-8
        @test Mo2.omega_average((x, ξ) -> 1.0, 0.2, 1.0) ≈ 1.0 rtol = 1e-6
        cs = Mo2.channel_split((x, ξ) -> 3.0, 2.0, 1.0)
        @test cs.bs ≈ 3.0 rtol = 1e-8
        @test abs(cs.pol(0.5, 1.0)) < 1e-8
        @test Mo2.omega_label(0.0, 0.0, 1.0) ≈ -1.0     # O-point
        @test Mo2.omega_label(1.0, float(π), 1.0) ≈ 3.0
    end

    @testset "grid_interpolant + Ω-decomposition diagnostics (01 §4)" begin
        g = _sgrid(17)
        # analytic smooth field on the grid; the interpolant is nodally exact and
        # accurate off-node, periodic in ξ.
        Ffun(x, ξ) = cos(ξ) * exp(-x^2 / 8)                                    # smooth, mode-1 in ξ
        F = [Ffun(x, ξ) for x in g.x.nodes, ξ in g.ξ.nodes]
        itp = Mo2.grid_interpolant(F, g)
        for ix in (1, 5, 9, g.x.n), iξ in (1, 3, 6)
            @test itp(g.x.nodes[ix], g.ξ.nodes[iξ]) ≈ F[ix, iξ] atol = 1e-12   # nodal exactness
        end
        xm = (g.x.nodes[8] + g.x.nodes[9]) / 2
        ξm = (g.ξ.nodes[3] + g.ξ.nodes[4]) / 2
        @test itp(xm, ξm) ≈ Ffun(xm, ξm) rtol = 5e-3                            # off-node accuracy
        @test itp(1.3, 0.7 + 2π) ≈ itp(1.3, 0.7) atol = 1e-12                   # ξ-periodic
        @test_throws ArgumentError Mo2.grid_interpolant(zeros(3, 3), g)

        # a flux-surface-constant current J = f(Ω) is pure bootstrap+curvature:
        # its ⟨·⟩_Ω reconstruction reproduces it, so Δ_pol → 0.
        w = 2.0
        fΩ(Ω) = exp(-0.2 * Ω)
        J = [fΩ(Mo2.omega_label(x, ξ, w)) for x in g.x.nodes, ξ in g.ξ.nodes]
        Δneo = Mo2.delta_moments(J, g; prefactor_cos=1.0, prefactor_sin=0.0).Δcos
        dec = Mo2.channel_decomposition(J, g, w; prefactor_cos=1.0)
        @test dec.bootstrap_curvature + dec.polarization ≈ Δneo                 # additive split
        @test abs(dec.polarization) < 5e-2 * abs(Δneo)                          # flux function ⇒ ~no polarization
        # the ⟨J̄_∥⟩_Ω profile of a flux function recovers f(Ω) at each Ω sample
        prof = dec.omega_average_profile
        for i in eachindex(prof.Ω)
            isnan(prof.value[i]) && continue
            @test prof.value[i] ≈ fΩ(prof.Ω[i]) rtol = 2e-2
        end
        # a non-flux current has a genuine polarization channel
        Jnf = [exp(-x^2 / 4) * (2.0 + cos(ξ)) for x in g.x.nodes, ξ in g.ξ.nodes]
        decnf = Mo2.channel_decomposition(Jnf, g, w; prefactor_cos=1.0)
        Δnf = Mo2.delta_moments(Jnf, g; prefactor_cos=1.0, prefactor_sin=0.0).Δcos
        @test decnf.bootstrap_curvature + decnf.polarization ≈ Δnf
        @test abs(decnf.polarization) > 1e-8                                     # not flux-constant
    end

    @testset "island_flux_amplitude — cleared ψ̃ = (w_ψ²/4)(q_s'/q_s) relation" begin
        # cleared physics relation (sign-off 2026-07-11; derivations/psi-tilde-amplitude.md)
        w, dq, q = 0.3, 0.8, 1.2
        ψ̃ = Mo2.island_flux_amplitude(; w_psi=w, dq_dpsi=dq, q_s=q)
        @test ψ̃ ≈ (w^2 / 4) * (dq / q)
        # round-trip: ψ̃ inverts the half-width relation w_ψ = 2√(ψ̃/|χ₀''|), χ₀'' = q_s'/q_s
        χ0pp = dq / q
        @test 2 * sqrt(ψ̃ / χ0pp) ≈ w                    # recovers the input half-width
        # scales as w²
        @test Mo2.island_flux_amplitude(; w_psi=2w, dq_dpsi=dq, q_s=q) ≈ 4 * ψ̃
        @test_throws ArgumentError Mo2.island_flux_amplitude(; w_psi=w, dq_dpsi=dq, q_s=0.0)
    end

    @testset "magnetic_drift_frequency — cleared ω̂_D + :original/:improved toggle" begin
        Co = IslM2.Coefficients
        # cleared physics (sign-off 2026-07-11; derivations/omega-D-drift-frequency.md)
        # ε→0 analytic limit: b→1 ⟹ A→√(1−y), G→(2−y)/√(1−y)
        for y in (0.2, 0.5, 0.8)
            A, G = Co.orbit_average_drift_brackets(; y=y, epsilon=1e-5)
            @test A ≈ sqrt(1 - y) rtol = 1e-3
            @test G ≈ (2 - y) / sqrt(1 - y) rtol = 1e-3
        end
        # the :improved toggle forces the L̂_B term to zero
        kw = (; y=0.5, v_hat=1.2, sigma=1.0, epsilon=0.1, inv_Lq=1.0)
        ωimp = Co.magnetic_drift_frequency(; kw..., inv_LB=0.7, variant=:improved)
        ωlb0 = Co.magnetic_drift_frequency(; kw..., inv_LB=0.0, variant=:original)
        @test ωimp ≈ ωlb0                                # :improved == :original with L̂_B⁻¹=0
        # σ-odd, v̂-linear (the σv̂/(1+ε) prefactor)
        ωp = Co.magnetic_drift_frequency(; kw..., inv_LB=0.7, variant=:original)
        ωm = Co.magnetic_drift_frequency(; y=0.5, v_hat=1.2, sigma=-1.0, epsilon=0.1, inv_Lq=1.0, inv_LB=0.7)
        @test ωp ≈ -ωm
        @test Co.magnetic_drift_frequency(; y=0.5, v_hat=2.4, sigma=1.0, epsilon=0.1, inv_Lq=1.0, inv_LB=0.7) ≈ 2 * ωp
        # the toggle is a real, large effect here (grad-B nearly cancels the shear term)
        @test !isapprox(ωp, ωimp; rtol=0.5)
        # trapped particles (1 < y < (1+ε)/(1−ε)) give finite brackets; forbidden y rejected
        At, Gt = Co.orbit_average_drift_brackets(; y=1.1, epsilon=0.1)
        @test isfinite(At) && isfinite(Gt) && At > 0 && Gt > 0
        @test_throws ArgumentError Co.orbit_average_drift_brackets(; y=1.5, epsilon=0.1)
        @test_throws ArgumentError Co.magnetic_drift_frequency(; kw..., inv_LB=0.7, variant=:bogus)
    end

    @testset "collision operator — cleared diffusivity P(λ) + deflection frequency ν(v̂)" begin
        Co = IslM2.Coefficients
        # P(λ) = λ√(1−λB) ≥ 0, vanishing at both endpoints (zero-flux)
        @test Co.pitch_diffusivity(0.0, 2.0) == 0.0
        @test Co.pitch_diffusivity(0.5, 2.0) == 0.0        # λ = 1/B endpoint
        @test Co.pitch_diffusivity(0.25, 2.0) ≈ 0.25 * sqrt(1 - 0.5)
        @test Co.pitch_diffusivity(0.3, 2.0) > 0
        @test_throws ArgumentError Co.pitch_diffusivity(0.6, 2.0)   # λ > 1/B
        # deflection frequency: high-v → ν̃/v̂³ (φ−G → 1 − 1/2v̂² + …); low-v → (4/3√π)/v̂²
        @test Co.deflection_frequency(6.0) * 6.0^3 ≈ 1.0 rtol = 2e-2       # φ−G → 1 (slow 1/2v̂² tail)
        @test Co.deflection_frequency(30.0) * 30.0^3 ≈ 1.0 rtol = 1e-3     # tighter at larger v̂
        @test Co.deflection_frequency(0.02) * 0.02^2 ≈ 4 / (3 * sqrt(π)) rtol = 1e-2  # 1/v̂² divergence
        # the Chandrasekhar form diverges slower than the reduced v̂⁻³ at low v̂
        @test Co.deflection_frequency(0.05; model=:chandrasekhar) < Co.deflection_frequency(0.05; model=:vcubed)
        @test Co.deflection_frequency(1.0; model=:vcubed) == 1.0
        @test_throws ArgumentError Co.deflection_frequency(1.0; model=:bogus)
        @test_throws ArgumentError Co.deflection_frequency(-1.0)
    end

    @testset "A7 — flattened-electron identity ⟨∂²h/∂x²⟩_Ω = 0 (coefficient-free)" begin
        for Ω in (1.2, 2.0, 5.0), pref in (1.0, 3.7)     # prefactor-independent
            @test abs(Fi2.flat_average_d2h_dx2(Ω, 1.0; prefactor=pref)) < 1e-10
        end
        @test Fi2.h_profile(0.5; prefactor=1.0) == 0.0   # exactly flat inside the separatrix
        @test Fi2.h_profile(2.0; prefactor=1.0) > 0.0
        @test Fi2.Q_omega(3.0) > Fi2.Q_omega(1.5)        # Q grows with Ω
        @test !Fi2.is_cleared(Fi2.ElectronClosure())     # closure constants (k, f_p) stay NaN-gated (Q3)
        @test isnan(Fi2.ElectronClosure().k_HS)
        # the h(Ω) amplitude C = w_ψ/2√2 is cleared (sign-off 2026-07-11); feeds h_profile's prefactor
        Co = IslM2.Coefficients
        @test Co.h_amplitude(0.3) ≈ 0.3 / (2 * sqrt(2))
        # far-field: with C = w_ψ/2√2 and Q → √Ω, h = C ∫₁^Ω dΩ'/Q → 2C(√Ω − 1)
        # = (w_ψ/√2)(√Ω − 1), approaching x = (w_ψ/√2)√Ω (derivation §3)
        w = 0.4
        Ω = 400.0
        @test Fi2.h_profile(Ω; prefactor=Co.h_amplitude(w)) ≈ (w / sqrt(2)) * (sqrt(Ω) - 1) rtol = 2e-2
        # quasineutrality closure coefficient τ/(τ+1) → 1/2 at τ=1 (cleared 2026-07-11)
        @test Co.quasineutrality_coefficient(1.0) ≈ 0.5
        @test Co.quasineutrality_coefficient(2.0) ≈ 2 / 3
        @test Co.quasineutrality_coefficient(1e6) ≈ 1.0 rtol = 1e-5   # τ → ∞ (cold ions)
        @test_throws ArgumentError Co.quasineutrality_coefficient(0.0)
        # passing fraction f_p = 1 − 1.4624√ε (cleared 2026-07-11; = quoted 1.46 to 3 s.f.)
        @test Co.passing_fraction(0.0) == 1.0                          # no trapping at ε=0
        @test Co.passing_fraction(0.1) ≈ 1 - 1.4624 * sqrt(0.1)
        @test Co.passing_fraction(0.01) < Co.passing_fraction(0.001)   # f_p decreases with ε
        @test isapprox(1 - Co.passing_fraction(0.1), 1.46 * sqrt(0.1); rtol=2e-3)  # matches 1.46
        @test_throws ArgumentError Co.passing_fraction(-0.1)
        # Δ-moment prefactors ∓μ₀R/2ψ̃ (cleared 2026-07-11), ψ̃ = (w²/4)(q'/q)
        pf = Co.delta_moment_prefactors(; mu0_R=3.0, w_psi=0.3, dq_dpsi=0.8, q_s=1.2)
        ψt = Mo2.island_flux_amplitude(; w_psi=0.3, dq_dpsi=0.8, q_s=1.2)
        @test pf.cos ≈ -3.0 / (2 * ψt)
        @test pf.sin ≈ +3.0 / (2 * ψt)
        @test pf.sin ≈ -pf.cos              # symmetric [DERIVED] pin
    end

    @testset "pseudo-arclength continuation detects the toy fold (03 §3)" begin
        ftoy!(out, u, p) = (out[1] = u[1]^2 + p; out)
        pa = So2.pseudo_arclength(ftoy!, [1.0], -1.0; ds=0.3, nsteps=15, rtol=1e-12, atol=1e-12)
        @test length(pa.ps) > 10                         # stepped through, no stall at the fold
        @test !isempty(pa.folds)                         # the fold at p = 0 is detected
        @test maximum(pa.ps) < 0.05                      # never steps past the fold parameter
        # both branches visited: u > 0 before the fold, u < 0 after
        @test any(z -> z[1] > 0.5, pa.us) && any(z -> z[1] < -0.5, pa.us)
    end
end
