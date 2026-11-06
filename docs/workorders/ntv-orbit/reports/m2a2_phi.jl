# Orbit-class counts of the forb table with and without Φ in the orbit Hamiltonian (DIII-D-like or TJ state), Z = 1.
# Usage (from gpec/): julia --project=. -t 1 docs/workorders/ntv-orbit/reports/m2a2_phi.jl <state_dir> <psi>...
using GeneralizedPerturbedEquilibrium, Serialization
const KF = GeneralizedPerturbedEquilibrium.KineticForces
equil, intr, prof = deserialize(joinpath(ARGS[1], "state.jls"))
intr.orbit_lmax, intr.orbit_nlambda, intr.orbit_nenergy, intr.orbit_energy_tol = 1, 16, 12, 1e-2
for pot in (false, true), ψ in parse.(Float64, ARGS[2:end])
    intr.orbit_field = pot ? KF.guiding_center_field(equil; omegaE=prof.omegaE_spline) : KF.guiding_center_field(equil)
    intr.orbit_potential = pot
    t0 = time()
    so = KF._torque_surface(ψ, 1, 2, false, "forb", equil, intr, prof; n=1)
    println(
        "potential=$pot ψ=$ψ nstatus (unclosed,trapped,passing,lost,axis,dropped) $(so.orbit.nstatus) in $(round(time() - t0; digits=1)) s; welec $(so.welec), wd-scale $(maximum(abs, so.geometry.wd[1:so.geometry.nlmda]))"
    )
    flush(stdout)
end
