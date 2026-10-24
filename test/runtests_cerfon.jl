using Test
using FastInterpolations: deriv1
using GeneralizedPerturbedEquilibrium.Equilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium: CerfonConfig, EquilibriumConfig, setup_equilibrium,
    cerfon_run, cerfon_basis, cerfon_basis_dx, cerfon_basis_dy, cerfon_basis_dxx, cerfon_basis_dyy,
    cerfon_basis_dxy, cerfon_psi_p, cerfon_dpsi_p_dx, cerfon_d2psi_p_dx2, cerfon_psihat, cerfon_grad, cerfon_hessian,
    cerfon_solve_coeffs, cerfon_find_axis, cerfon_shape_points

# Cerfon-Freidberg diverted analytic equilibrium (eq_type = "cerfon").
#
# The basis and derivative tests are the load-bearing ones: every boundary condition is
# assembled from these closed forms, and a single dropped product-rule term silently shifts
# the plasma shape rather than raising an error. The Δ* check catches a wrong basis function
# and the finite-difference checks catch a wrong derivative of a right one.

const CERFON_PTS = [(0.75, -0.5), (1.0, 0.2), (1.3, 0.6), (0.9, -0.55)]

"""
Δ*f = f_xx − f_x/x + f_yy by central differences.
"""
function delta_star_fd(f, x, y; h=1e-4)
    fxx = (f(x + h, y) - 2f(x, y) + f(x - h, y)) / h^2
    fx = (f(x + h, y) - f(x - h, y)) / (2h)
    fyy = (f(x, y + h) - 2f(x, y) + f(x, y - h)) / h^2
    return fxx - fx / x + fyy
end

@testset "Cerfon-Freidberg analytic equilibrium" begin
    @testset "homogeneous basis solves Δ*ψᵢ = 0" begin
        for i in 1:12, (x, y) in CERFON_PTS
            @test abs(delta_star_fd((a, b) -> cerfon_basis(a, b)[i], x, y)) < 1e-3
        end
    end

    @testset "particular solution solves Δ*ψ_p = (1−A)x² + A" begin
        for A in (-0.155, 0.0, 0.5), (x, y) in CERFON_PTS
            @test delta_star_fd((a, _) -> cerfon_psi_p(a, A), x, y) ≈ (1 - A) * x^2 + A atol = 1e-5
        end
    end

    @testset "hand-coded derivatives match finite differences" begin
        for (x, y) in CERFON_PTS, i in 1:12
            f = (a, b) -> cerfon_basis(a, b)[i]
            h1, h2 = 1e-6, 1e-4
            @test cerfon_basis_dx(x, y)[i] ≈ (f(x + h1, y) - f(x - h1, y)) / (2h1) rtol = 1e-5 atol = 1e-5
            @test cerfon_basis_dy(x, y)[i] ≈ (f(x, y + h1) - f(x, y - h1)) / (2h1) rtol = 1e-5 atol = 1e-5
            @test cerfon_basis_dxx(x, y)[i] ≈ (f(x + h2, y) - 2f(x, y) + f(x - h2, y)) / h2^2 rtol = 1e-4 atol = 1e-4
            @test cerfon_basis_dyy(x, y)[i] ≈ (f(x, y + h2) - 2f(x, y) + f(x, y - h2)) / h2^2 rtol = 1e-4 atol = 1e-4
            g = (a, b) -> cerfon_basis_dx(a, b)[i]
            @test cerfon_basis_dxy(x, y)[i] ≈ (g(x, y + h1) - g(x, y - h1)) / (2h1) rtol = 1e-5 atol = 1e-5
        end
        for A in (-0.155, 0.4), (x, _) in CERFON_PTS
            h = 1e-6
            @test cerfon_dpsi_p_dx(x, A) ≈ (cerfon_psi_p(x + h, A) - cerfon_psi_p(x - h, A)) / (2h) rtol = 1e-5
            @test cerfon_d2psi_p_dx2(x, A) ≈
                  (cerfon_psi_p(x + 1e-4, A) - 2cerfon_psi_p(x, A) + cerfon_psi_p(x - 1e-4, A)) / 1e-8 rtol = 1e-4
        end
    end

    @testset "boundary conditions are satisfied by the solved coefficients" begin
        for null in ("lsn", "dn")
            cfg = CerfonConfig(; null=null)
            c, (xn, yn) = cerfon_solve_coeffs(cfg)
            A = cfg.A
            p = cerfon_shape_points(cfg)
            # the single null sits below the midplane; the double null returns its upper null
            @test sign(yn) == (null == "dn" ? 1 : -1)
            # boundary passes through the equatorial points and the null
            @test abs(cerfon_psihat(p.xout, 0.0, c, A)) < 1e-12
            @test abs(cerfon_psihat(p.xin, 0.0, c, A)) < 1e-12
            @test abs(cerfon_psihat(xn, yn, c, A)) < 1e-12
            # the null is a true magnetic null
            @test hypot(cerfon_grad(xn, yn, c, A)...) < 1e-10
            # the axis is an interior extremum with ψ̂ < 0 (so ψ = −Pψ̂ peaks on axis)
            xa, ya = cerfon_find_axis(cfg, c, A)
            @test p.xin < xa < p.xout
            @test cerfon_psihat(xa, ya, c, A) < 0
            @test hypot(cerfon_grad(xa, ya, c, A)...) < 1e-9
        end
        # the single null additionally pins the high point; the double null is up-down symmetric
        cfg = CerfonConfig(; null="lsn")
        c, _ = cerfon_solve_coeffs(cfg)
        p = cerfon_shape_points(cfg)
        @test abs(cerfon_psihat(p.xhigh, p.yhigh, c, cfg.A)) < 1e-12
        cfg_dn = CerfonConfig(; null="dn")
        cdn, _ = cerfon_solve_coeffs(cfg_dn)
        @test all(abs.(cdn[8:12]) .< 1e-12)
        @test cerfon_find_axis(cfg_dn, cdn, cfg_dn.A)[2] ≈ 0 atol = 1e-9
    end

    @testset "[CERFON_INPUT] reads A and rejects the old a_solovev key" begin
        @test CerfonConfig(Dict{String,Any}("A" => -0.2)).A == -0.2
        @test_throws MethodError CerfonConfig(Dict{String,Any}("a_solovev" => -0.2))
    end

    @testset "cerfon_run's F/μ₀p table satisfies Grad-Shafranov with its flux map" begin
        # Differentiates the profile columns cerfon_run hands downstream, so a sign slip in F² fails here.
        for null in ("lsn", "dn")
            cfg = CerfonConfig(; null=null)
            run_input = cerfon_run(EquilibriumConfig(; eq_type="cerfon"), cfg)
            c, _ = cerfon_solve_coeffs(cfg)
            A, R0, psio = cfg.A, cfg.r0, run_input.psio
            xa, ya = cerfon_find_axis(cfg, c, A)
            P = -psio / cerfon_psihat(xa, ya, c, A)  # flux scale of ψ = −P·ψ̂, recovered from cerfon_run's ψ_axis
            sq_deriv = deriv1(run_input.sq_in)
            sq, dsq = zeros(4), zeros(4)
            for (x, y) in ((0.85, -0.2), (1.1, -0.1), (1.15, 0.2), (0.95, 0.4))
                psi_norm = 1 - (-P * cerfon_psihat(x, y, c, A)) / psio
                @test 0 < psi_norm < 1
                run_input.sq_in(sq, psi_norm)
                sq_deriv(dsq, psi_norm)
                # table columns are F and μ₀p against ψ_N = 1 − ψ/psio, so d/dψ = −(1/psio) d/dψ_N
                mu0_dpdpsi = -dsq[2] / psio
                FdFdpsi = -sq[1] * dsq[1] / psio
                gx, _ = cerfon_grad(x, y, c, A)
                hxx, hyy, _ = cerfon_hessian(x, y, c, A)
                dstar = -P * (hxx - gx / x + hyy) / R0^2  # Δ*ψ in (R, Z)
                residual = dstar + mu0_dpdpsi * (R0 * x)^2 + FdFdpsi
                @test abs(residual) / abs(dstar) < 1e-5
            end
        end
    end

    @testset "unknown null topology is rejected" begin
        eq = EquilibriumConfig(; eq_type="cerfon")
        @test_throws ErrorException cerfon_run(eq, CerfonConfig(; null="snowflake"))
    end

    @testset "q diverges logarithmically toward ψ_N = 1" begin
        # The whole point of this equilibrium: the null on the boundary makes q unbounded, so
        # raising psihigh by a decade adds a fixed increment to q forever and no psihigh
        # converges. Compare against the plain Solovev model, whose q_edge is finite.
        qedge = Float64[]
        for psihigh in (0.999, 0.9999, 0.99999)
            eq = EquilibriumConfig(; eq_type="cerfon", jac_type="pest", grid_type="ldp",
                psilow=1e-4, psihigh=psihigh, mpsi=128, mtheta=128,
                etol=1e-8, force_termination=true)
            cfg = CerfonConfig(; null="lsn", mr=256, mz=384, ma=128, q0=1.1)
            pe = setup_equilibrium(eq, cerfon_run(eq, cfg))
            # the closed-form flux scale must deliver the requested on-axis q
            @test pe.params.q0 ≈ cfg.q0 rtol = 1e-3
            push!(qedge, pe.profiles.q_spline(last(pe.profiles.xs)))
        end
        @test issorted(qedge)
        # increments per decade settle to a constant — the signature of a logarithm
        d1, d2 = qedge[2] - qedge[1], qedge[3] - qedge[2]
        @test d1 > 0.1
        @test d2 ≈ d1 rtol = 0.1
    end

    @testset "double null diverges about twice as fast as single null" begin
        function edge_increment(null)
            qs = map((0.999, 0.99999)) do psihigh
                eq = EquilibriumConfig(; eq_type="cerfon", jac_type="pest", grid_type="ldp",
                    psilow=1e-4, psihigh=psihigh, mpsi=128, mtheta=128,
                    etol=1e-8, force_termination=true)
                cfg = CerfonConfig(; null=null, mr=256, mz=384, ma=128, q0=1.1)
                pe = setup_equilibrium(eq, cerfon_run(eq, cfg))
                @test pe.params.q0 ≈ cfg.q0 rtol = 1e-3
                pe.profiles.q_spline(last(pe.profiles.xs))
            end
            qs[2] - qs[1]
        end
        @test edge_increment("dn") / edge_increment("lsn") ≈ 2 rtol = 0.15
    end
end
