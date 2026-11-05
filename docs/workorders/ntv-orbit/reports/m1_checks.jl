# M1 physics checks for the guiding-center orbits (KineticForces/Orbits.jl).
# Run from gpec/: julia --project=. docs/workorders/ntv-orbit/reports/m1_checks.jl
# Reads results/m0_diiid_n1 (DIII-D-like, kinetic profiles) and results/m1_lar (TJ analytic, ε_a = 0.05);
# writes results/m1_orbits/{m1_numbers.json, zow.csv, orbits_rz.csv, lar.csv}.

using GeneralizedPerturbedEquilibrium, TOML, Printf, SpecialFunctions
const GPEC = GeneralizedPerturbedEquilibrium
const KF = GPEC.KineticForces
const EQ = GPEC.Equilibrium

const ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", "..", ".."))
const OUT = joinpath(ROOT, "results", "m1_orbits")
mkpath(OUT)

function load(run)
    dir = joinpath(ROOT, "results", run)
    deck = TOML.parsefile(joinpath(dir, "gpec.toml"))
    return EQ.PlasmaEquilibrium(GPEC.equilibrium_config(deck, dir), GPEC.analytic_config(deck))
end

numbers = Dict{String,Any}()
const mass, chrg = 2 * KF.mp, KF.e

# ---------------------------------------------------------------- DIII-D-like
t_eq = @elapsed equil = load("m0_diiid_n1")
prof = KF.ntv_profiles(KF.KineticForcesControl(), equil)
g = KF.guiding_center_field(equil)
gE = KF.guiding_center_field(equil; omegaE=prof.omegaE_spline)
intr = KF.KineticForcesInternal(equil)
intr.mfac = collect(-5:5)

θs = range(0, 1, 513)[1:end-1]
numbers["construction"] = Dict(
    "q_rel_err_max" => maximum(abs(sum(θ -> g.F(ψ) * g.jac((ψ, θ)) / (g.s * g.psio * g.R((ψ, θ))^2), θs) / 512 / 2π / equil.profiles.q_spline(ψ) - 1) for ψ in 0.05:0.05:0.95),
    "B_rel_err_max" => maximum(
        abs(g.bcov_theta((ψ, θ)) * g.s * g.psio / g.jac((ψ, θ)) + (g.F(ψ) / g.R((ψ, θ)))^2 / g.B((ψ, θ)) - g.B((ψ, θ))) / g.B((ψ, θ)) for ψ in 0.05:0.05:0.95, θ in θs[1:8:end]
    ),
    "jac_ratio_range" => extrema(g.jac((ψ, θ)) / equil.rzphi_jac((ψ, θ)) for ψ in 0.05:0.1:0.95, θ in θs[1:16:end]),
    "s" => g.s, "F_axis" => g.F(0.0), "offset_theta0" => maximum(abs(equil.rzphi_offset((ψ, 0.0))) for ψ in 0.1:0.1:0.9))

function bounce_geo(psi0)
    st = KF._setup_surface_state(psi0, 1, 2, false, equil, intr, prof)
    nm = length(intr.mfac)
    geo = KF.compute_bounce_geometry(psi0, 1, st.q, intr.bo, st.bmax, st.bmin, st.theta_bmax, st.tspl, st.B_vpar, intr.mfac, intr.chi1,
        intr.ro, zeros(ComplexF64, nm), zeros(ComplexF64, nm), 1.0, 1.0, mass, chrg, st.T_s, "fgar"; nlmda=64, ntheta=256)
    return st, geo
end

# ZOW limit (checks 1 exponent, 2 frequencies, 5 conservation), potential off.
xs = 10.0 .^ (-4:0.5:0)
zow = IOBuffer()
println(zow, "psi0,lambda,code_sigma,x,sigma,status,wb_rel_err,prec_over_wd_x,psibar_minus_psi0,energy_err,pphi_err,wb,wphi")
cons = Float64[]
t_orb = @elapsed for psi0 in (0.3, 0.5, 0.7)
    st, geo = bounce_geo(psi0)
    for il in (5, 30, 41, 50, 60)
        lam = geo.lambda[il]
        for x in xs, sv in (1, -1)
            o = KF.guiding_center_orbit(g, psi0, lam, x * st.T_s, sv; mass, chrg)
            o === nothing && continue
            push!(cons, o.energy_error, o.pphi_error)
            @printf(zow, "%.3f,%.6f,%d,%.3e,%d,%s,%.6e,%.6e,%.6e,%.2e,%.2e,%.6e,%.6e\n", psi0, lam, geo.sigma[il], x, sv, o.status,
                o.wb / (geo.wb[il] * sqrt(x)) - 1, o.wphi / (geo.wd[il] * x), o.psibar - psi0, o.energy_error, o.pphi_error, o.wb, o.wphi)
        end
    end
end
write(joinpath(OUT, "zow.csv"), take!(zow))
numbers["timing"] = Dict("equilibrium_s" => t_eq, "zow_table_s" => t_orb)

# Check 2: trapped/passing boundary at low energy versus BounceAveraging's λ = B₀/B_max.
st, geo = bounce_geo(0.5)
function boundary(sv; x=1e-4)
    lo, hi = 0.5, intr.bo / st.bmin * 0.999
    for _ in 1:40
        mid = (lo + hi) / 2
        o = KF.guiding_center_orbit(g, 0.5, mid, x * st.T_s, sv; mass, chrg)
        (o !== nothing && o.status == :trapped) ? (hi = mid) : (lo = mid)
    end
    return (lo + hi) / 2
end
numbers["boundary"] = Dict("lambda_tpb_code" => intr.bo / st.bmax, "lambda_orbit_plus" => boundary(1), "lambda_orbit_minus" => boundary(-1),
    "lambda_orbit_plus_x1" => boundary(1; x=1.0), "lambda_orbit_minus_x1" => boundary(-1; x=1.0))

# Check 3: co-current passing ions are displaced outward in R. Co-current ⇔ v∥ F s > 0.
co = Dict{String,Any}()
for sv in (1, -1)
    o = KF.guiding_center_orbit(g, 0.5, 0.3, st.T_s, sv; mass, chrg)
    ts = range(0, o.sol.t[end], 4001)
    us = o.sol.(ts)
    Ro = sum(g.R((u[1], u[2])) for u in us) / length(us)
    Rs = sum(g.R((0.5, u[2])) for u in us) / length(us)
    co[sv * sign(g.F(0.5)) * g.s > 0 ? "co" : "counter"] = Dict("sigma" => sv, "R_orbit_minus_R_surface" => Ro - Rs, "psibar_minus_psi0" => o.psibar - 0.5)
end
numbers["cocurrent"] = co

# Check 4: halving the ODE tolerance.
chg = 0.0
for (lam, sv, x) in ((0.3, 1, 1.0), (0.3, -1, 1.0), (0.95, 1, 1.0), (0.95, -1, 10.0), (0.80, 1, 1.0))
    a = KF.guiding_center_orbit(g, 0.5, lam, x * st.T_s, sv; mass, chrg, rtol=1e-10)
    b = KF.guiding_center_orbit(g, 0.5, lam, x * st.T_s, sv; mass, chrg, rtol=5e-12)
    global chg = max(chg, abs(a.wb / b.wb - 1), abs(a.wphi / b.wphi - 1), abs(a.psibar - b.psibar))
end
numbers["tolerance"] = Dict("max_rel_change" => chg)
numbers["conservation"] = Dict("max_rel_drift" => maximum(cons), "n_orbits" => length(cons) ÷ 2)

# Check 6: with Φ, the trapped ⟨φ̇⟩ shifts by s·ω_E (the same sign map as the precession). The formula is
# tested at 0.1·ω_E (v_E/v ≤ 0.03); the example's full ω_E is recorded as well, where v_E/v reaches 0.3.
pot = Any[]
g01 = KF.guiding_center_field(equil; omegaE=ψ -> 0.1 * prof.omegaE_spline(ψ))
for (f, gf) in ((0.1, g01), (1.0, gE)), psi0 in (0.3, 0.5, 0.7), x in (0.3, 1.0)
    T = prof.Ti_spline(psi0)
    a = KF.guiding_center_orbit(g, psi0, 0.97, x * T, 1; mass, chrg)
    b = KF.guiding_center_orbit(gf, psi0, 0.97, x * T, 1; mass, chrg)
    (a === nothing || b === nothing) && continue
    wE = f * prof.omegaE_spline(b.psibar)
    vE = abs(wE) * equil.ro
    push!(
        pot,
        Dict("omegaE_factor" => f, "psi0" => psi0, "x" => x, "dwphi" => b.wphi - a.wphi, "s_omegaE" => g.s * wE, "ratio" => (b.wphi - a.wphi) / (g.s * wE),
            "vE_over_v" => vE / sqrt(2x * T / mass), "status" => string(b.status))
    )
end
numbers["potential"] = pot

# Figure data: orbits in (R, Z) at x = 1, 10, 40 (2, 21, 86 keV D at ψ = 0.5), trapped and co/counter passing.
rz = IOBuffer()
println(rz, "x,lambda,sigma,status,R,Z")
for x in (1.0, 10.0, 40.0), (lam, sv) in ((0.95, 1), (0.3, 1), (0.3, -1))
    o = KF.guiding_center_orbit(g, 0.5, lam, x * st.T_s, sv; mass, chrg)
    for τ in range(0, o.sol.t[end], 600)
        u = o.sol(τ)
        @printf(rz, "%g,%g,%d,%s,%.5f,%.5f\n", x, lam, sv, o.status, g.R((u[1], u[2])), g.Z((u[1], u[2])))
    end
end
write(joinpath(OUT, "orbits_rz.csv"), take!(rz))
surf = IOBuffer()
println(surf, "psi,R,Z")
for ψ in (0.5, 0.995), θ in range(0, 1, 200)
    @printf(surf, "%g,%.5f,%.5f\n", ψ, g.R((ψ, θ)), g.Z((ψ, θ)))
end
write(joinpath(OUT, "surfaces_rz.csv"), take!(surf))

# ---------------------------------------------------------------- large aspect ratio (check 1)
lar = load("m1_lar")
gl = KF.guiding_center_field(lar)
lb = IOBuffer()
println(lb, "psi0,kind,kappa2,energy_eV,lambda,wb,wb_pendulum,status")
pend = Dict{String,Any}()
for psi0 in (0.1, 0.3, 0.5)
    Bt = [gl.B((psi0, θ)) for θ in θs]
    Bmax, Bmin = extrema(Bt) |> reverse
    B0e, eps = (Bmax + Bmin) / 2, (Bmax - Bmin) / (Bmax + Bmin)
    # connection length per poloidal turn: dl = (B/B^θ) dθ
    L = sum(θ -> gl.B((psi0, θ)) * abs(gl.jac((psi0, θ))) / gl.psio, θs) / length(θs)
    qR = L / 2π
    E = 100 * KF.eV
    errs = Float64[]
    for k2 in vcat(range(0.05, 0.95, 10), range(1.05, 4.0, 10))
        mu = E / (B0e * (1 - eps + 2eps * k2))
        lam = mu * gl.bo / E
        o = KF.guiding_center_orbit(gl, psi0, lam, E, 1; mass, chrg)
        o === nothing && continue
        a = sqrt(mu * B0e * eps / mass)
        wp = k2 < 1 ? π * a / (2qR * ellipk(k2)) : π * a * sqrt(k2) / (qR * ellipk(1 / k2))
        push!(errs, o.wb / wp - 1)
        @printf(lb, "%.2f,%s,%.4f,%.1f,%.6f,%.6e,%.6e,%s\n", psi0, k2 < 1 ? "trapped" : "passing", k2, E / KF.eV, lam, o.wb, wp, o.status)
    end
    # ω_b ∝ E^{1/2} at fixed λ: fitted exponent over 1–100 eV
    lam = gl.bo / (B0e * (1 - eps + 2eps * 0.5))
    Es = (1.0, 10.0, 100.0) .* KF.eV
    wbs = [KF.guiding_center_orbit(gl, psi0, lam, Ei, 1; mass, chrg).wb for Ei in Es]
    slope = log(wbs[end] / wbs[1]) / log(Es[end] / Es[1])
    pend[string(psi0)] = Dict("eps_eff" => eps, "max_abs_rel_err" => maximum(abs.(errs)), "exponent" => slope)
end
write(joinpath(OUT, "lar.csv"), take!(lb))
numbers["pendulum"] = pend

# Reflection of the starting θ on the up-down symmetric LAR equilibrium (check 3).
refl = 0.0
for (lam, sv, x) in ((0.3, 1, 1e4), (1.02, 1, 1e4), (1.02, -1, 1e4))
    E = x * KF.eV
    a = KF.guiding_center_orbit(gl, 0.4, lam, E, sv; mass, chrg, theta0=0.03)
    b = KF.guiding_center_orbit(gl, 0.4, lam, E, sv; mass, chrg, theta0=0.97)
    (a === nothing || b === nothing) && continue
    global refl = max(refl, abs(a.wb / b.wb - 1), abs(a.wphi / b.wphi - 1), abs(a.psibar - b.psibar))
end
numbers["reflection"] = Dict("max_rel_diff" => refl)

tojson(x::AbstractDict) = "{" * join(("\"$k\": " * tojson(v) for (k, v) in sort(collect(x); by=first)), ", ") * "}"
tojson(x::Union{AbstractVector,Tuple}) = "[" * join(tojson.(x), ", ") * "]"
tojson(x::AbstractString) = "\"$x\""
tojson(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
write(joinpath(OUT, "m1_numbers.json"), tojson(numbers))
println(tojson(numbers))
