# The refined forb table on the two pitch columns of m2a2_ures.jl's figure, in the same columns as ures.csv (grid = "refined").
# Usage (from gpec/): julia --project=. -t 1 docs/workorders/ntv-orbit/reports/m2a2_ures_table.jl <state_dir> <out_dir>
using GeneralizedPerturbedEquilibrium, Serialization
const KF = GeneralizedPerturbedEquilibrium.KineticForces
equil, intr, prof = deserialize(joinpath(ARGS[1], "state.jls"))
n, L, mi, ψ = 1, 6, 2, 0.70
g = KF.guiding_center_field(equil)
sf = KF._torque_surface(ψ, 1, mi, false, "fgar", equil, intr, prof; n)
λtp, λmax = intr.bo / sf.bmax, intr.bo / g.B((ψ, sf.theta_bmin))
open(joinpath(ARGS[2], "ures_table.csv"), "w") do io
    println(io, join(("psi,lambda,lambda_tp,lambda_max,sigma,grid,u,status,wb,wphi,dir,q", ("dj$(l)" for l in -L:L)...), ","))
    for (λ, sv) in ((0.75, -1), (1.05, 1))
        nstatus = zeros(Int, 6)
        buf = (dbob=zeros(ComplexF64, intr.mpert), divx=zeros(ComplexF64, intr.mpert), A=zeros(ComplexF64, KF.ORBIT_NSAMPLE), u=zeros(6))
        gco(u; kw...) = KF.guiding_center_orbit(g, ψ, λ, u^2 * sf.T_s, sv; mass=sf.mass, chrg=sf.chrg, theta0=sf.theta_bmin, rtol=KF.ORBIT_RTOL, kw...)
        function orbit(u; dense=true)
            o = gco(u; dense)
            o !== nothing && o.status == :unclosed && (o = gco(u; dense, tau_max=2e4 * KF.ORBIT_TAU_RETRY))
            return o
        end
        code(o) = o === nothing ? Int8(-1) : KF.ORBIT_STATUS_CODES[o.status]
        for (ua, ub, c) in KF._class_segments(u -> code(orbit(u; dense=false)), u -> code(gco(u; dense=false)), KF.ORBIT_U_MAX, nstatus)
            c in (1, 2) || continue
            seg = KF._sample_segment(orbit, c == 2, ua, ub, λ, sf.q, sf.welec, false, n, L, 1.0, g, intr, buf, 12, 5e-3, nstatus)
            seg === nothing && continue
            for (j, u) in enumerate(seg.u)
                println(io, join((ψ, λ, λtp, λmax, sv, "refined", u, c, seg.wb[j] * u, NaN, seg.dir, sf.q, seg.dJdJ[j, :]...), ","))
            end
            println("λ=$λ σ=$sv class $c on [$(round(ua; digits=3)), $(round(ub; digits=3))]: $(length(seg.u)) nodes")
        end
    end
end
