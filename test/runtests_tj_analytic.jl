using Test
using Printf
using GeneralizedPerturbedEquilibrium.Equilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium: TJAnalyticConfig, EquilibriumConfig,
    setup_equilibrium, tj_analytic_run, tj_analytic_run_direct

# Two-path smoke tests for the TJ-analytic equilibrium model
# (GPEC adaptation of R. Fitzpatrick's TJ code,
# https://github.com/rfitzp/TJ).
#
# `tj_analytic_run` (inverse) is exercised at a low-εa point where the
# first-order Shafranov-shifted-circle geometry is faithful;
# `tj_analytic_run_direct` (Option B direct-GS) is exercised at a moderate-εa
# point where the εa³·L terms in the (R,Z)→(r,w) Newton inversion matter.
# These cover the two dispatch branches (`eq_type = "tj_analytic"` /
# `"tj_analytic_direct"`) that are otherwise only run end-to-end via the LAR_*
# scan scripts.

@testset "TJ-analytic model" begin
    @testset "tj_analytic_run (inverse) — basic invariants at ε = 0.25" begin
        # Keep ε, mpsi, mtheta modest so the whole block runs in ~1 s.
        tj = TJAnalyticConfig(lar_r0=1.0 / 0.25, lar_a=1.0,
            qc=1.5, qa=3.6, pc=0.001, mu=2.0, B0=12.0,
            ma=64, mtau=64)
        eq = EquilibriumConfig(eq_type="tj_analytic",
            psilow=0.01, psihigh=0.995,
            mpsi=64, mtheta=128, etol=1e-7)
        pe = setup_equilibrium(eq, tj)

        # psio is a physical-scale ψ; regressions in the a→a² normalization
        # or the dψ/dr construction would change it by factors of a.
        @test pe.psio > 0
        @test isfinite(pe.psio)

        # ν root-find pins q₂(x=1) = qa; qmax at psihigh=0.995 lands ~0.04 below.
        @test pe.params.q0 ≈ 1.5 rtol = 1e-3
        @test pe.params.qmax > 3.5
        @test pe.params.qmax < 3.7

        # Magnetic axis at R = R0, Z = 0 for the shifted-circle benchmark.
        @test pe.ro ≈ 4.0 rtol = 1e-3
        @test abs(pe.zo) < 1e-8
    end

    @testset "tj_analytic_run_direct (Option B) — pole-approach physics at ε = 0.60" begin
        # ε = 0.60 sits on the stable side of the ideal-external-kink pole at
        # ε ≈ 0.665 for this (qc, qa, pc, μ) combination.  Pole-approach shape
        # (δW_t small, Δ' > 0 and growing) is the Option B success criterion.
        tj = TJAnalyticConfig(lar_r0=1.0 / 0.60, lar_a=1.0,
            qc=1.5, qa=3.6, pc=0.001, mu=2.0, B0=12.0,
            ma=64, mtau=64)
        eq = EquilibriumConfig(eq_type="tj_analytic_direct",
            psilow=0.01, psihigh=0.995,
            mpsi=64, mtheta=128, etol=1e-7)
        pe = setup_equilibrium(eq, tj)

        @test pe.psio > 0
        @test isfinite(pe.psio)

        # Direct-GS line integration at ε=0.60 gives qmax between 3.8 and 4.0.
        # If the εa³·L shape terms in f_R / f_Z regress, qmax jumps above 5.
        @test pe.params.q0 ≈ 1.5 rtol = 1e-2
        @test pe.params.qmax > 3.75
        @test pe.params.qmax < 4.1

        # Magnetic axis at R = R0.  Shafranov shift of the O-point itself is
        # zero by construction (H₁(0) = 0).
        @test pe.ro ≈ (1.0 / 0.60) rtol = 1e-3
        @test abs(pe.zo) < 1e-4
    end

    @testset "tj_analytic_run_direct — ψ(R,Z) endpoint consistency" begin
        # At the magnetic axis ψ_in should equal psio (axis convention: ψ
        # positive at axis, zero at LCFS); sampling well outside the LCFS should
        # give a negative value (the vacuum branch of psi_rz).
        tj = TJAnalyticConfig(lar_r0=1.0 / 0.25, lar_a=1.0,
            qc=1.5, qa=3.6, pc=0.001, mu=2.0, B0=12.0,
            ma=64, mtau=64)
        eq = EquilibriumConfig(eq_type="tj_analytic_direct",
            psilow=0.01, psihigh=0.995,
            mpsi=64, mtheta=128, etol=1e-7)
        inp = tj_analytic_run_direct(eq, tj)

        # ψ at the geometric axis matches psio (see DirectRunInput docstring for
        # the sign convention: psi_in is positive at axis, zero at LCFS).
        R0 = 1.0 / 0.25
        @test inp.psi_in((R0, 0.0)) ≈ inp.psio rtol = 1e-3

        # Well outside the LCFS → negative ψ_in (vacuum branch of the grid).
        R_out = R0 + 1.05   # plasma LCFS is at R ≈ R0 + 0.94
        @test inp.psi_in((R_out, 0.0)) < 0
    end
end
