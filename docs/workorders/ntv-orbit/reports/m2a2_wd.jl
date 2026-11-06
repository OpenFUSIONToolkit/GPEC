# Trapped precession: forb's table value over the surface ω_D per pitch column, at several Z (variant-B floor check).
# Usage (from gpec/): julia --project=. -t 1 docs/workorders/ntv-orbit/reports/m2a2_wd.jl <state_dir> <out_csv>
using GeneralizedPerturbedEquilibrium, Serialization, Statistics
const KF = GeneralizedPerturbedEquilibrium.KineticForces
equil, intr, prof = deserialize(joinpath(ARGS[1], "state.jls"))
intr.orbit_field, intr.orbit_lmax, intr.orbit_nlambda, intr.orbit_nenergy, intr.orbit_potential = KF.guiding_center_field(equil), 1, 64, 12, false
intr.orbit_energy_tol = 5e-3
open(ARGS[2], "w") do io
    println(io, "zi,psi,lambda,lambda_tp,u,wd_orbit,wd_surface")
    for zi in (1024, 16384), ψ in (0.5, 0.9179367691935629)
        so = KF._torque_surface(ψ, zi, 2, false, "forb", equil, intr, prof; n=1)
        gm = so.geometry
        r = Float64[]
        for (il, segs) in enumerate(so.orbit.segments), s in segs
            s.passing && continue
            foreach(j -> println(io, join((zi, ψ, gm.lambda[il], intr.bo / so.bmax, s.u[j], s.wd[j], gm.wd[il]), ",")), eachindex(s.u))
            push!(r, median(s.wd) / gm.wd[il])
        end
        println("Z=$zi ψ=$ψ trapped columns $(length(r)): wd_orbit/wd_surface median $(round(median(r); sigdigits=5)), range $(round.(extrema(r); sigdigits=4))")
        flush(stdout)
    end
end
