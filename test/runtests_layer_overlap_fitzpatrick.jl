# External-reference check of the diffusive-resistive layer width against a published
# result: Fitzpatrick, Nucl. Fusion 2025 (doi 10.1088/1741-4326/ae4fdd), Sect. 5.8-5.9.
#
# The paper reports that for its model JET equilibrium the resistive layers of adjacent
# rational surfaces first overlap at Psi = 0.9985 for n = 1 (its Fig. 9) and Psi = 0.9952
# for n = 4 (its Fig. 10). This rebuilds that equilibrium from the paper's own definitions
# and drives GPEC's `delta_dr` through `slayer_parameters`, so the width formula and the
# tau_R / tau_A / d_beta chain feeding it are checked against a number in print rather than
# against GPEC's own history.
@testset "Layer overlap vs Fitzpatrick (2025) JET model" begin
    using GeneralizedPerturbedEquilibrium.InnerLayer.SLAYER: slayer_parameters,
        slayer_layer_thickness, SpitzerHarmModel
    using QuadGK: quadgk
    using Roots: find_zero, Bisection

    # Sect. 5.8 parameters.
    B0, R0, A_MIN = 3.45, 2.96, 1.25
    ZEFF, MNUM, CHI = 10.0, 2.0, 1.0
    Q0, Q95, Q105 = 1.01, 3.5, 4.0

    # Model safety factor, Eqs. (32)-(35). alpha- depends on rhat95, which depends on the
    # profile through Eq. (36), so the pair is solved self-consistently.
    q_in(r, am) = Q0 - am * log(1 - r^2)
    q_out(r, ap) = -ap * log(r^2 - 1)
    q_any(r, am, ap) = r < 1 ? q_in(r, am) : q_out(r, ap)
    psi_raw(r, am, ap) = quadgk(x -> x / q_any(x, am, ap), 0.0, r; rtol=1e-11)[1]

    am, ap = 1.0, 1.0
    for _ in 1:200
        psi_sep = psi_raw(1.0 - 1e-12, am, ap)
        r95 = find_zero(r -> psi_raw(r, am, ap) / psi_sep - 0.95, (0.5, 1 - 1e-9), Bisection())
        f105(r) = (psi_sep + quadgk(x -> x / q_out(x, ap), 1 + 1e-12, r; rtol=1e-10)[1]) / psi_sep - 1.05
        hi = 1.0 + 1e-9
        while hi < 1.40 && f105(hi) < 0     # q_out turns negative past r = sqrt(2)
            hi += 0.005
        end
        r105 = hi >= 1.40 ? 1.20 : find_zero(f105, (1 + 1e-9, hi), Bisection())
        am_new = -(Q95 - Q0) / log(1 - r95^2)
        ap_new = -Q105 / log(r105^2 - 1)
        converged = abs(am_new - am) < 1e-12 && abs(ap_new - ap) < 1e-12
        am, ap = am_new, ap_new
        converged && break
    end
    psi_sep = psi_raw(1.0 - 1e-12, am, ap)
    q_of(r) = q_any(r, am, ap)
    Psi_of(r) = r < 1 ? psi_raw(r, am, ap) / psi_sep :
                (psi_sep + quadgk(x -> x / q_out(x, ap), 1 + 1e-12, r; rtol=1e-10)[1]) / psi_sep
    dq_dr(r) = r < 1 ? am * 2r / (1 - r^2) : -ap * 2r / (r^2 - 1)
    s_of(r) = r * dq_dr(r) / q_of(r)

    # mtanh edge profiles read off the paper's Fig. 8, anchored at Psi = 0.96 and at the
    # separatrix (Sect. 5.9 states Te ~ 100 eV there).
    function make_tanh(f096, f100, f_ped, f_sol)
        a = (f_ped - f_sol) / 2
        x1 = atanh(clamp(1 - (f096 - f_sol) / a, -0.999, 0.999))
        x2 = atanh(clamp(1 - (f100 - f_sol) / a, -0.999, 0.999))
        d = 0.04 / (x2 - x1)
        p0 = 0.96 - x1 * d
        return P -> f_sol + a * (1 - tanh((P - p0) / d))
    end
    te_of = make_tanh(400.0, 100.0, 600.0, 20.0)
    ne_of = make_tanh(3.5e19, 1.2e19, 5.0e19, 3.0e18)

    # Width in units of rhat, through the shipped code path.
    function width_hat(r, n)
        P = Psi_of(r)
        te = te_of(P)
        ne = ne_of(P)
        p = slayer_parameters(; n_e=ne, t_e=te, t_i=te,
            omega=0.0, omega_e=1.0e4, omega_i=5.0e3,
            qval=q_of(r), sval_r=s_of(r), bt=B0,
            rs=r * A_MIN, R0=R0, mu_i=MNUM, zeff=ZEFF,
            chi_perp=CHI, chi_tor=CHI, m=1, n=n,
            resistivity_model=SpitzerHarmModel(), lnLambda_form=:nrl)
        return slayer_layer_thickness(p).delta_dr / A_MIN
    end

    function surfaces(n; rmin=0.90, rmax=1.35)
        out = Tuple{Int,Float64}[]
        for m in 1:400
            qt = m / n
            qt < q_of(rmin) && continue
            r = if qt < q_of(1 - 1e-13)
                find_zero(x -> q_of(x) - qt, (rmin, 1 - 1e-13), Bisection())
            elseif qt > q_of(1 + 1e-13) && qt < q_of(rmax)
                find_zero(x -> q_of(x) - qt, (1 + 1e-13, rmax), Bisection())
            else
                nothing
            end
            r === nothing || push!(out, (m, r))
        end
        sort!(out; by=t -> t[2])
        return out
    end

    # Inner boundary of the overlap region: the first surface whose layer runs into its
    # inner neighbour, reported at that neighbour's location.
    function overlap_psi(n)
        S = surfaces(n)
        for k in 2:length(S)
            gap = S[k][2] - S[k-1][2]
            (width_hat(S[k-1][2], n) + width_hat(S[k][2], n)) / 2 >= gap && return Psi_of(S[k-1][2])
        end
        return nothing
    end

    # The self-consistent profile must reproduce the paper's own inputs before the widths
    # mean anything.
    @test isapprox(q_of(0.0), Q0; atol=1e-10)
    @test am > 0 && ap > 0

    psi1 = overlap_psi(1)
    @test psi1 !== nothing
    psi1 === nothing || @info "Fitzpatrick 2025 overlap, n = 1" psi=psi1 paper=0.9985 deviation=abs(psi1 - 0.9985) atol=1e-4
    # Paper: Psi = 0.9985 (n = 1); measured deviation 1.1e-5. The bound covers the pedestal read
    # off Fig. 8 and the resistivity closure (Spitzer-Harm here against the paper's Eqs. 70-72),
    # and still leaves ~9x margin. Pure quadrature and root-finding, no BLAS, so platform spread
    # sits far below this.
    @test isapprox(psi1, 0.9985; atol=1e-4)

    psi4 = overlap_psi(4)
    @test psi4 !== nothing
    psi4 === nothing || @info "Fitzpatrick 2025 overlap, n = 4" psi=psi4 paper=0.9952 deviation=abs(psi4 - 0.9952) atol=1e-3
    # Paper: Psi = 0.9952 (n = 4); measured deviation 2.3e-4. Looser than n = 1 because these
    # surfaces sit further in, where the digitised pedestal shape matters more.
    @test isapprox(psi4, 0.9952; atol=1e-3)
    # The overlap region must move inward with n, which is the paper's reported scaling.
    @test psi4 < psi1
end
