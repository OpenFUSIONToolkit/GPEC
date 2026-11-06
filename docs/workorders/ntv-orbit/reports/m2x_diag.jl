# Data for the M2 explainer figures (m2x_figures.py) on the DIII-D-like n = 1 case, potential off as in the M2 scan.
# Usage (from gpec/): julia --project=. -t 8 docs/workorders/ntv-orbit/reports/m2x_diag.jl <out_dir>
# (1) orbit traces in (R, Z) versus charge; (2) orbit class on the table grid and on a fine pitch scan, with the
# bisected class boundary λc per energy node; (3) the 3- and 6-point ψ node sets; (4) Σℓ|δJ_ℓ|² on pitches clustered at λc.
using GeneralizedPerturbedEquilibrium, Serialization, Printf, QuadGK
const GPEC = GeneralizedPerturbedEquilibrium
const KF = GPEC.KineticForces
out = ARGS[1]  # holds state.jls (m2_checks.jl cache)
equil, intr, prof = deserialize(joinpath(out, "state.jls"))
n, L, mi = 1, 6, 2
g = KF.guiding_center_field(equil)
code(o) = o === nothing ? -1 : Int(KF.ORBIT_STATUS_CODES[o.status])
function rz(ψ, θ)
    r = sqrt(max(equil.rzphi_rsquared((ψ, mod(θ, 1.0))), 0.0))
    η = 2π * (θ + equil.rzphi_offset((ψ, mod(θ, 1.0))))
    return equil.ro + r * cos(η), equil.zo + r * sin(η)
end
trace(o) = [(rz(u[1], u[2])..., u[1]) for u in o.sol(range(0, o.sol.t[end]; length=400)).u]

# (1) Orbit traces.
open(joinpath(out, "explain_orbits.csv"), "w") do io
    println(io, "case,zi,status,R,Z,psi")
    for ψs in (0.5, 0.96, 1 - 1e-4), θ in range(0, 1; length=200)
        println(io, join(("surf$(ψs)", 0, "", rz(ψs, θ)..., ψs), ","))
    end
    sf = KF._setup_surface_state(0.5, 1, mi, false, equil, intr, prof)
    lam = (intr.bo / sf.bmax + intr.bo / sf.bmin) / 2
    for zi in (1, 4, 16, 64)
        s = KF._setup_surface_state(0.5, zi, mi, false, equil, intr, prof)
        o = KF.guiding_center_orbit(g, 0.5, lam, s.T_s, 1; mass=s.mass, chrg=s.chrg, theta0=s.theta_bmin)
        foreach(p -> println(io, join(("mid", zi, o.status, p...), ",")), trace(o))
    end
end

# (2) Orbit class on the table grid and on a fine pitch scan; λc bisected as in `_fill_separatrix_layer!`.
us = KF.orbit_energy_nodes(12)
brows, blk = Tuple[], ReentrantLock()
open(joinpath(out, "explain_classes.csv"), "w") do io
    println(io, "psi,zi,grid,lambda,k,u,slot,status")
    for (ψ, zi) in ((0.96, 1), (0.96, 16), (0.70, 1))
        sf = KF._torque_surface(ψ, zi, mi, false, "fgar", equil, intr, prof; n)
        gm = KF.compute_bounce_geometry(ψ, n, sf.q, intr.bo, sf.bmax, sf.bmin, sf.theta_bmax, sf.tspl, sf.B_vpar, intr.mfac,
            intr.chi1, intr.ro, sf.dbob_m_f, sf.divx_m_f, 1.0, 1.0, sf.mass, sf.chrg, sf.T_s, "forb"; nlmda=128, surface_field=sf.bf)
        λmax = intr.bo / g.B((ψ, sf.theta_bmin))
        λtp = intr.bo / sf.bmax
        cls(λ, k, s) = code(KF.guiding_center_orbit(g, ψ, λ, us[k]^2 * sf.T_s, s == 1 ? 1 : -1; mass=sf.mass, chrg=sf.chrg, theta0=sf.theta_bmin))
        fine = [λmax * (i - 0.5) / 400 for i in 1:400]
        for (grid, lams) in (("table", gm.lambda[1:gm.nlmda]), ("fine", fine))
            res = Array{Int}(undef, length(lams), length(us), 2)
            Threads.@threads :dynamic for il in eachindex(lams)
                for k in eachindex(us), s in 1:2
                    res[il, k, s] = cls(lams[il], k, s)
                end
            end
            for il in eachindex(lams), k in eachindex(us), s in 1:2
                println(io, join((ψ, zi, grid, lams[il], k, us[k], s, res[il, k, s]), ","))
            end
            grid == "fine" || continue
            # λc: the trapped/passing change nearest λtp on the fine scan, bisected.
            Threads.@threads for ks in [(k, s) for k in eachindex(us), s in 1:2]
                k, s = ks
                c = res[:, k, s]
                pairs = [i for i in 1:length(c)-1 if c[i] in (1, 2) && c[i+1] in (1, 2) && c[i] != c[i+1]]
                λc = NaN
                if !isempty(pairs)
                    i = pairs[argmin([abs(fine[j] + fine[j+1] - 2λtp) for j in pairs])]
                    lo, hi, clo = fine[i], fine[i+1], c[i]
                    for _ in 1:30
                        mid = (lo + hi) / 2
                        cm = cls(mid, k, s)
                        cm in (1, 2) || break
                        cm == clo ? (lo = mid) : (hi = mid)
                    end
                    λc = (lo + hi) / 2
                end
                lock(blk) do
                    push!(brows, (ψ, zi, k, us[k], s, λc, λtp, λmax))
                end
            end
        end
        @printf("ψ=%.2f zi=%d done\n", ψ, zi)
        flush(stdout)
    end
end
open(joinpath(out, "explain_boundary.csv"), "w") do io
    println(io, "psi,zi,k,u,slot,lambda_c,lambda_tp,lambda_max")
    foreach(r -> println(io, join(r, ",")), sort(brows))
end

# (4) Σℓ|δJ_ℓ|² on pitches clustered at the orbit's own boundary λc (ψ = 0.96, Z = 1): what a λc-normalized grid resolves.
let ψ = 0.96, zi = 1
    sf = KF._torque_surface(ψ, zi, mi, false, "fgar", equil, intr, prof; n)
    open(joinpath(out, "explain_dj_lc.csv"), "w") do io
        println(io, "k,u,slot,lambda,lambda_c,status,dj")
        for r in brows, kk in (6, 8)
            (r[1] == ψ && r[2] == zi && r[3] == kk && !isnan(r[6])) || continue
            k, s, λc = r[3], r[5], r[6]
            lams = sort(vcat(λc .- λc .* 10 .^ range(-6, -0.7; length=24), λc .+ λc .* 10 .^ range(-6, -0.7; length=24)))
            rows = Vector{Any}(undef, length(lams))
            Threads.@threads :dynamic for i in eachindex(lams)
                λ = lams[i]
                o = KF.guiding_center_orbit(g, ψ, λ, us[k]^2 * sf.T_s, s == 1 ? 1 : -1; mass=sf.mass, chrg=sf.chrg, theta0=sf.theta_bmin)
                dj = 0.0
                if o !== nothing && o.status in (:trapped, :passing)
                    buf = (dbob=zeros(ComplexF64, intr.mpert), divx=zeros(ComplexF64, intr.mpert), A=zeros(ComplexF64, KF.ORBIT_NSAMPLE), u=zeros(6))
                    v = zeros(2L + 1)
                    KF._orbit_action!(v, o, g, intr, λ, n, L, 1.0, o.status == :passing && o.dir_theta > 0, buf)
                    dj = sum(v)
                end
                rows[i] = (k, us[k], s, λ, λc, code(o), dj)
            end
            foreach(x -> println(io, join(x, ",")), rows)
        end
    end
end

# (3) ψ nodes, as in m2_checks.jl.
psi_c = KF.axis_validity_boundary(KF.KineticForcesControl(), nothing, prof, equil, intr.sing_psis)
x0, xout = max(equil.config.psilow, psi_c), min(intr.psilim, 1 - 1e-6)
res = KF.kinetic_resonance_psi_nodes(prof, equil; n, nl=L, zi=1, mi, electron=false, wdfac=1.0)
panels = KF.psi_panel_points(vcat(intr.sing_psis, res, [2psi_c]), x0, xout)
open(joinpath(out, "explain_psinodes.csv"), "w") do io
    println(io, "set,psi")
    foreach(p -> println(io, "panel,$p"), panels)
    foreach(p -> println(io, "rational,$p"), intr.sing_psis)
    for np in (3, 6)
        gx, _ = QuadGK.gauss(np)
        for i in 1:length(panels)-1
            a, b = panels[i], panels[i+1]
            foreach(x -> println(io, "gl$np,$((a + b) / 2 + (b - a) / 2 * x)"), gx)
        end
    end
end
println("EXPLAIN_DONE")
