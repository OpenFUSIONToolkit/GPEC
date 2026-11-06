# M2 attempt 2, step 1: how orbit quantities vary with energy at fixed pitch, against the 12-node table.
# Usage (from gpec/): julia --project=. -t 8 docs/workorders/ntv-orbit/reports/m2a2_ures.jl <state_dir> <out_dir>
# DIII-D-like n = 1, thermal D, Z = 1, ψ = 0.96 (and 0.70), potential off as in the M2 scan.
using GeneralizedPerturbedEquilibrium, Serialization
const KF = GeneralizedPerturbedEquilibrium.KineticForces
equil, intr, prof = deserialize(joinpath(ARGS[1], "state.jls"))
out = ARGS[2]
n, L, mi = 1, 6, 2
g = KF.guiding_center_field(equil)
ufine = collect(range(0.03, 6.0; length=150))
unodes = KF.orbit_energy_nodes(12)
rows, lk = String[], ReentrantLock()
for ψ in (0.96, 0.70)
    sf = KF._torque_surface(ψ, 1, mi, false, "fgar", equil, intr, prof; n)
    λtp, λmax = intr.bo / sf.bmax, intr.bo / g.B((ψ, sf.theta_bmin))
    lams = sort(unique(round.([0.19, 0.45, 0.6, λtp - 0.01, λtp + 0.01, 0.75, 0.9, 1.05, λmax - 0.02] .* 1e4) ./ 1e4))
    jobs = [(λ, s, u, tag) for λ in lams for s in (1, -1) for (u, tag) in Iterators.flatten((((u, "fine") for u in ufine), ((u, "node") for u in unodes)))]
    Threads.@threads :dynamic for (λ, s, u, tag) in jobs
        o = KF.guiding_center_orbit(g, ψ, λ, u^2 * sf.T_s, s; mass=sf.mass, chrg=sf.chrg, theta0=sf.theta_bmin)
        st = o === nothing ? -1 : Int(KF.ORBIT_STATUS_CODES[o.status])
        dj = fill(NaN, 2L + 1)
        wb = wphi = dir = NaN
        if o !== nothing && o.status in (:trapped, :passing)
            buf = (dbob=zeros(ComplexF64, intr.mpert), divx=zeros(ComplexF64, intr.mpert), A=zeros(ComplexF64, KF.ORBIT_NSAMPLE), u=zeros(6))
            KF._orbit_action!(dj, o, g, intr, λ, n, L, 1.0, false, buf)
            wb, wphi, dir = o.wb, o.wphi, o.dir_theta
        end
        line = join((ψ, λ, λtp, λmax, s, tag, u, st, wb, wphi, dir, sf.q, dj...), ",")
        lock(lk) do
            push!(rows, line)
        end
    end
end
open(joinpath(out, "ures.csv"), "w") do io
    println(io, join(("psi,lambda,lambda_tp,lambda_max,sigma,grid,u,status,wb,wphi,dir,q", ("dj$(l)" for l in -L:L)...), ","))
    foreach(r -> println(io, r), sort(rows))
end
println("wrote ", length(rows), " rows")
