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
        slayer_layer_thickness, slayer_algebraic_widths, SpitzerHarmModel
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
    fp_converged = false
    for _ in 1:200
        psi_sep = psi_raw(1.0 - 1e-12, am, ap)
        r95 = find_zero(r -> psi_raw(r, am, ap) / psi_sep - 0.95, (0.5, 1 - 1e-9), Bisection())
        f105(r) = (psi_sep + quadgk(x -> x / q_out(x, ap), 1 + 1e-12, r; rtol=1e-10)[1]) / psi_sep - 1.05
        hi = 1.0 + 1e-9
        while hi < 1.40 && f105(hi) < 0     # q_out turns negative past r = sqrt(2)
            hi += 0.005
        end
        hi < 1.40 || error("Fitzpatrick JET model: failed to bracket the Psi = 1.05 surface before q_out degenerates")
        r105 = find_zero(f105, (1 + 1e-9, hi), Bisection())
        am_new = -(Q95 - Q0) / log(1 - r95^2)
        ap_new = -Q105 / log(r105^2 - 1)
        fp_converged = abs(am_new - am) < 1e-12 && abs(ap_new - ap) < 1e-12
        am, ap = am_new, ap_new
        fp_converged && break
    end
    fp_converged || error("Fitzpatrick JET model: the alpha-/alpha+ fixed point did not converge in 200 iterations")
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

    # Width in units of rhat, through the shipped code path. Eq. (100) is algebraic, so this takes
    # the no-ODE path; the two are asserted equal once below.
    function params_at(r, n)
        P = Psi_of(r)
        te = te_of(P)
        ne = ne_of(P)
        p = slayer_parameters(; n_e=ne, t_e=te, t_i=te,
            omega=0.0, omega_e=1.0e4, omega_i=5.0e3,
            qval=q_of(r), sval_r=s_of(r), bt=B0,
            rs=r * A_MIN, R0=R0, mu_i=MNUM, zeff=ZEFF,
            chi_perp=CHI, chi_tor=CHI, m=1, n=n,
            resistivity_model=SpitzerHarmModel(), lnLambda_form=:nrl)
        return p
    end
    width_hat(r, n) = slayer_algebraic_widths(params_at(r, n)).delta_dr / A_MIN

    # The scan's criterion channel must be exactly what the full width routine reports, so the
    # no-ODE path cannot drift from it.
    let p = params_at(0.9, 1)
        @test slayer_algebraic_widths(p).delta_dr == slayer_layer_thickness(p).delta_dr
        @test slayer_algebraic_widths(p).delta_visco == slayer_layer_thickness(p).delta_visco
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

# Wiring of the overlap cap into the integration domain. The scan itself is checked against the
# paper above; this checks that the cap reaches `sing_lim!` and behaves as an upper bound.
@testset "Layer-overlap cap wiring" begin
    using GeneralizedPerturbedEquilibrium.ForceFreeStates: sing_lim!, ForceFreeStatesInternal,
        ForceFreeStatesControl
    using GeneralizedPerturbedEquilibrium
    using GeneralizedPerturbedEquilibrium.Equilibrium
    using TOML

    dir_path = joinpath(dirname(@__DIR__), "examples", "Solovev_ideal_example")
    inputs = TOML.parsefile(joinpath(dir_path, "gpec.toml"))
    equil = Equilibrium.setup_equilibrium(
        Equilibrium.EquilibriumConfig(inputs["Equilibrium"], dir_path),
        Equilibrium.SolovevConfig(inputs["SOL_INPUT"]))

    _fresh() = (i = ForceFreeStatesInternal(); i.nlow = 1; i.nhigh = 1; i)
    ctrl = ForceFreeStatesControl(; truncate_at_rational_offset=false, qhigh=1e3)

    # Baseline: no cap.
    base = _fresh()
    sing_lim!(base, ctrl, equil)

    # A cap inside the domain must pull qlim down and move psilim inward.
    inside = _fresh()
    cap = 0.5 * (equil.profiles.xs[1] + base.psilim)
    sing_lim!(inside, ctrl, equil; psilim_cap=cap)
    @test inside.qlim < base.qlim
    @test inside.psilim < base.psilim
    @test isapprox(inside.psilim, cap; rtol=1e-6)

    # A cap beyond psihigh is inert by construction -- the bound never widens the domain.
    beyond = _fresh()
    sing_lim!(beyond, ctrl, equil; psilim_cap=base.psilim + 1e-3)
    @test beyond.qlim == base.qlim
    @test beyond.psilim == base.psilim

    # nothing is the same as not passing it at all.
    none = _fresh()
    sing_lim!(none, ctrl, equil; psilim_cap=nothing)
    @test none.qlim == base.qlim
    @test none.psilim == base.psilim

    # With rational_offset_fraction on, the (last rational + rational_offset_fraction)/n offset must never step past qhigh or the cap.
    ctrl_dm = ForceFreeStatesControl(; truncate_at_rational_offset=true, rational_offset_fraction=0.2, qhigh=1e3)
    dm_base = _fresh()
    sing_lim!(dm_base, ctrl_dm, equil)
    N = floor(Int, equil.params.qmax - 0.15)
    @test N >= 2
    for qh in (Float64(N), N + 0.1)
        c = ForceFreeStatesControl(; truncate_at_rational_offset=true, rational_offset_fraction=0.2, qhigh=qh)
        i = _fresh()
        sing_lim!(i, c, equil)
        @test i.qlim ≈ N - 1 + 0.2
        @test i.qlim <= qh
    end

    # A cap placed exactly on a rational must exclude it on either side of the float knife-edge.
    q_of = equil.profiles.q_spline
    lo, hi = equil.profiles.xs[1], equil.params.psihigh_resolved
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        q_of(mid) < N ? (lo = mid) : (hi = mid)
    end
    psi_N = 0.5 * (lo + hi)
    for cap_k in (prevfloat(psi_N, 1000), psi_N, nextfloat(psi_N, 1000))
        i = _fresh()
        sing_lim!(i, ctrl_dm, equil; psilim_cap=cap_k)
        @test i.qlim ≈ N - 1 + 0.2
        @test i.psilim <= cap_k
    end
    @test dm_base.qlim > N - 1 + 0.2

    # The control flag exists and is opt-in.
    @test ForceFreeStatesControl().psilim_from_layer_overlap == false

    # Multi-n: the innermost overlap point binds; scans with no overlap never do.
    Scan = GeneralizedPerturbedEquilibrium.Tearing.LayerOverlapScan
    mkscan(ph) = Scan(; psihigh_dr=ph, psihigh=ph, first_overlap=(ph === nothing ? nothing : 1))
    binding = GeneralizedPerturbedEquilibrium._binding_overlap
    s1, s2, s_none = mkscan(0.9997), mkscan(0.9952), mkscan(nothing)
    @test binding([s1, s2, s_none]) === s2
    @test binding([s_none, s1]) === s1
    @test binding([s_none, mkscan(nothing)]) === s_none
    @test binding([s1]) === s1

    # The scan must see the run's own plasma: every [SLAYER] setting the layer width depends on
    # reaches it, and changing one changes the widths.
    trunc_dir = joinpath(dirname(@__DIR__), "examples", "DIIID-like_truncation_example")
    slayer_section = TOML.parsefile(joinpath(trunc_dir, "gpec.toml"))["SLAYER"]
    slayer_inputs(over) = Dict{String,Any}("SLAYER" => merge(slayer_section, over))
    overlap_inputs(over; dir=trunc_dir) = GeneralizedPerturbedEquilibrium._overlap_slayer_inputs(slayer_inputs(over), dir)
    oi = overlap_inputs(Dict("ion_mass" => 3.0, "bt" => 1.7, "theta_sample" => 0.3, "lnLambda_form" => "sauter"))
    @test (oi.mu_i, oi.bt, oi.theta, oi.lnLambda_form) == (3.0, 1.7, 0.3, :sauter)

    # A disabled section feeds the scan whatever its inner model, and a missing or empty file only skips the scan;
    # with tearing enabled the same file is an error, raised here rather than after every other stage.
    @test overlap_inputs(Dict("inner_model" => "ggj_ray")) !== nothing
    @test (@test_logs (:warn, r"Layer-overlap scan skipped") match_mode = :any overlap_inputs(Dict("profile_file" => "missing.h5"))) === nothing
    @test (@test_logs overlap_inputs(Dict("profile_file" => ""))) === nothing
    @test_throws ErrorException overlap_inputs(Dict("enabled" => true, "profile_file" => "missing.h5"))
    @test_throws ErrorException overlap_inputs(Dict("enabled" => true, "profile_file" => ""))

    # Z_eff(psi) comes from the kinetic file's own n_i/n_e with the [SLAYER] impurity charge.
    kin_path = joinpath(trunc_dir, slayer_section["profile_file"])
    kin = Equilibrium.read_kinetic_file(kin_path)
    j = length(kin.psi) ÷ 2
    @test oi.zeff(kin.psi[j]) ≈ 6.0 - kin.n_i[j] / kin.n_e[j] * 5.0
    @test overlap_inputs(Dict("impurity_charge" => 10.0)).zeff(kin.psi[j]) ≈ 10.0 - kin.n_i[j] / kin.n_e[j] * 9.0
    @test 1.0 < oi.zeff(kin.psi[j]) < 6.0
    # A zeff profile supplied in the file overrides the n_i/n_e closure; a file with neither falls
    # back to the scalar [SLAYER] zeff.
    mktempdir() do tmp
        own = fill(3.2, length(kin.psi))
        base = (; psi=kin.psi, n_e=kin.n_e, T_i=kin.T_i, T_e=kin.T_e, omega_E=kin.omega_E, chi_e=kin.chi_e, chi_phi=kin.chi_phi)
        Equilibrium.write_kinetic_h5(joinpath(tmp, "own_zeff.h5"), Equilibrium.KineticProfileData(; base..., n_i=kin.n_i, zeff=own))
        @test Equilibrium.read_kinetic_file(joinpath(tmp, "own_zeff.h5")).zeff == own
        @test overlap_inputs(Dict("profile_file" => "own_zeff.h5", "impurity_charge" => 10.0, "zeff" => 2.5); dir=tmp).zeff(kin.psi[j]) ≈ 3.2
        Equilibrium.write_kinetic_h5(joinpath(tmp, "no_ni.h5"), Equilibrium.KineticProfileData(; base...))
        @test overlap_inputs(Dict("profile_file" => "no_ni.h5", "zeff" => 2.5); dir=tmp).zeff == 2.5
    end

    widths(over) = GeneralizedPerturbedEquilibrium._layer_overlap_scan(overlap_inputs(over), _fresh(), equil).width_dr
    w_ref = widths(Dict{String,Any}())
    @test !isempty(w_ref) && all(isfinite, w_ref)
    for over in (Dict("impurity_charge" => 10.0), Dict("resistivity_model" => "spitzer"), Dict("lnLambda_form" => "sauter"))
        @test all(.!isapprox.(widths(over), w_ref; rtol=1e-3))
    end
    # Ion mass and field strength cancel between the Lundquist number and the ion sound radius in
    # Eq. (100), so the criterion width does not depend on them; the scalar zeff is only a
    # fallback, so it is inert while the file carries n_i.
    for over in (Dict("ion_mass" => 1.0), Dict("bt" => 1.0), Dict("zeff" => 3.0))
        @test all(isapprox.(widths(over), w_ref; rtol=1e-10))
    end

    # The no-surfaces path must return an empty scan with a note, not throw (m_max=0 with
    # extrapolation off forces it deterministically).
    @testset "resistive_layer_overlap: empty surface list" begin
        using GeneralizedPerturbedEquilibrium: Tearing, Utilities
        npsi = 8
        psi_grid = collect(range(0.0, 1.0; length=npsi))
        profs = Utilities.KineticProfiles(; psi=psi_grid,
            n_e=fill(1.0e19, npsi), T_e=fill(500.0, npsi), T_i=fill(500.0, npsi),
            omega=zeros(npsi), omega_e=zeros(npsi), omega_i=zeros(npsi))
        scan = Tearing.resistive_layer_overlap(equil, profs; n_tor=1, m_max=0, extrapolate=false)
        @test scan isa Tearing.LayerOverlapScan
        @test isempty(scan.psi)
        @test scan.psihigh === nothing && scan.first_overlap === nothing
        @test any(contains("no q ="), scan.notes)
    end
end

# Surfaces beyond the grid are placed on the X-point law. Checked against the equilibrium's own
# traced q: a law built on a short grid must land where a longer grid actually finds the surfaces.
@testset "Separatrix q-law" begin
    using GeneralizedPerturbedEquilibrium.Equilibrium: separatrix_q_law, separatrix_q, separatrix_q_psi, separatrix_q_dqdpsi
    using GeneralizedPerturbedEquilibrium.ForceFreeStates: ForceFreeStatesInternal

    # The functions themselves, on a two-X-point law with known parameters.
    law2 = (psi_x=[1.0002, 1.01], A=[0.5, 0.3], psi_anchor=0.995, q_anchor=4.3)
    @test separatrix_q(law2, law2.psi_anchor) == law2.q_anchor
    for q_target in (4.5, 6.0, 9.0)
        psi_q = separatrix_q_psi(law2, q_target)
        @test law2.psi_anchor < psi_q < law2.psi_x[1]
        @test separatrix_q(law2, psi_q) ≈ q_target rtol = 1e-10
    end
    @test separatrix_q_dqdpsi(law2, 0.999) ≈ 0.5 / (1.0002 - 0.999) + 0.3 / (1.01 - 0.999)
    @test separatrix_q_psi(law2, 4.3) === nothing            # not beyond the anchor
    @test separatrix_q_psi(law2, 4.0) === nothing
    @test separatrix_q_psi(law2, 40.0) === nothing           # closer to the separatrix than psi_gap
    @test separatrix_q_psi(law2, 40.0; psi_gap=1e-40) !== nothing

    trunc_dir = joinpath(dirname(@__DIR__), "examples", "DIIID-like_truncation_example")
    trunc_inputs = TOML.parsefile(joinpath(trunc_dir, "gpec.toml"))
    diiid(psihigh) = Equilibrium.setup_equilibrium(
        Equilibrium.EquilibriumConfig(merge(trunc_inputs["Equilibrium"], Dict("psihigh" => psihigh)), trunc_dir), nothing)
    long, short = diiid(0.9999), diiid(0.995)

    # One lower X-point, 1e-4 outside this file's boundary; the law reproduces the traced shear.
    law = separatrix_q_law(short)
    @test law !== nothing
    @test law.psi_x[1] ≈ 1.0001 atol = 2e-5
    @test law.A[1] ≈ 0.640 rtol = 1e-2
    @test law.z_x[1] < 0
    @test law.shear_ratio ≈ 1.0 atol = 0.05
    @test separatrix_q_law(short; shear_ratio=1.001) === nothing

    # Anchored at psi = 0.995, it must place q = 7 and q = 8 where the grid to 0.9999 traces them.
    for q_target in (7.0, 8.0)
        psi_traced = find_zero(p -> long.profiles.q_spline(p) - q_target, (0.99, 0.9999), Bisection())
        gap_traced = law.psi_x[1] - psi_traced
        @test law.psi_x[1] - separatrix_q_psi(law, q_target) ≈ gap_traced rtol = 0.05
    end

    # The scan's overlap point must therefore not depend on where the grid stops, and no
    # rational may fall between the in-grid search and the law.
    oi = GeneralizedPerturbedEquilibrium._overlap_slayer_inputs(trunc_inputs, trunc_dir)
    one_n() = (i = ForceFreeStatesInternal(); i.nlow = 1; i.nhigh = 1; i)
    scan_long = GeneralizedPerturbedEquilibrium._layer_overlap_scan(oi, one_n(), long)
    scan_short = GeneralizedPerturbedEquilibrium._layer_overlap_scan(oi, one_n(), short)
    @test scan_short.psihigh ≈ scan_long.psihigh atol = 5e-5
    @test scan_short.first_overlap == scan_long.first_overlap
    @test all(diff(scan_short.m) .== 1) && all(diff(scan_long.m) .== 1)
    @test any(scan_short.extrapolated) && any(occursin("separatrix law", note) for note in scan_short.caveats)
    # psihigh stays the upper limit: at most one rational is placed past the equilibrium boundary,
    # and the recommendation is never beyond it.
    for scan in (scan_long, scan_short)
        @test count(note -> occursin("skipped", note), scan.notes) <= 1
        @test scan.psihigh < 1.0
    end

    # No 2D flux, no X-point: an analytic equilibrium is not extrapolated.
    sol_dir = joinpath(dirname(@__DIR__), "examples", "Solovev_ideal_example")
    sol_inputs = TOML.parsefile(joinpath(sol_dir, "gpec.toml"))
    solovev = Equilibrium.setup_equilibrium(
        Equilibrium.EquilibriumConfig(sol_inputs["Equilibrium"], sol_dir), Equilibrium.SolovevConfig(sol_inputs["SOL_INPUT"]))
    @test separatrix_q_law(solovev) === nothing
end
