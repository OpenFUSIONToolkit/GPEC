# Dump one forb table (segments and their nodes) for the report's schematic of the per-column scheme.
# Usage (from gpec/): julia --project=. -t 1 docs/workorders/ntv-orbit/reports/m2a2_table.jl <run_dir> <out_csv> <psi> <zi> <nlambda>
using GeneralizedPerturbedEquilibrium, Serialization
const KF = GeneralizedPerturbedEquilibrium.KineticForces
equil, intr, prof = deserialize(joinpath(ARGS[1], "state.jls"))
ψ, zi, nl = parse(Float64, ARGS[3]), parse(Int, ARGS[4]), parse(Int, ARGS[5])
intr.orbit_field, intr.orbit_lmax, intr.orbit_nlambda, intr.orbit_potential, intr.orbit_energy_tol = KF.guiding_center_field(equil), 2, nl, false, 5e-3
so = KF._torque_surface(ψ, zi, 2, false, "forb", equil, intr, prof; n=1)
open(ARGS[2], "w") do io
    println(io, "il,lambda,seg,passing,dir,ua,ub,u,wb,wd,dj0")
    for (il, segs) in enumerate(so.orbit.segments), (k, s) in enumerate(segs), j in eachindex(s.u)
        println(io, join((il, so.orbit.lambda[il], k, s.passing, s.dir, s.ua, s.ub, s.u[j], s.wb[j], s.wd[j], s.dJdJ[j, 3]), ","))
    end
end
println("bo/bmax = $(intr.bo / so.bmax); nstatus $(so.orbit.nstatus)")
