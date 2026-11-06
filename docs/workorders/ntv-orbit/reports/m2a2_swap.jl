# Floor attribution: forb with its own table, and with the surface ω_D, ω_b or |δJ_ℓ|² swapped in for the orbit's,
# per class, against the same-grid replay. Usage (from gpec/): julia --project=. -t 1 docs/workorders/ntv-orbit/reports/m2a2_swap.jl <run_dir> <out_csv> <zi> <psi>...
using GeneralizedPerturbedEquilibrium, Serialization
const KF = GeneralizedPerturbedEquilibrium.KineticForces
equil, intr, prof = deserialize(joinpath(ARGS[1], "state.jls"))
zi = parse(Int, ARGS[3])
const n, L, mi = 1, 6, 2
intr.orbit_field, intr.orbit_lmax, intr.orbit_nlambda, intr.orbit_nenergy, intr.orbit_potential = KF.guiding_center_field(equil), L, 64, 12, false
intr.orbit_energy_tol = 5e-3
tot(tab, sf, f, c) = sum(
    l -> KF.calculate_orbit(tab, n, l, sf.q, sf.epsr, sf.wdian, sf.wdiat, sf.welec * f, sf.nuk * f, intr.bo, sf.bmax, sf.n_s, sf.T_s;
        chi1=intr.chi1, ro=intr.ro, energy_atol=1e-11, energy_rtol=1e-7, pitch_atol=1e-9, pitch_rtol=1e-5, classes=c), -L:L)
function replay_table(gm, sf)
    sdj = reduce(hcat, [KF.bounce_harmonic(gm, l, n, sf.q, intr.ro, intr.mpert).dJdJ for l in -L:L])
    us = [0.5, 1.0, 2.0, 4.0]
    segs = map(1:gm.nlmda) do il
        gm.valid[il] || return KF.OrbitSegment[]
        p = gm.sigma[il] != 0
        [KF.OrbitSegment(p, d, 0.0, sqrt(KF.X_ENERGY_MAX), us, fill(gm.wb[il], 4), fill(gm.wd[il], 4), repeat(sdj[il:il, :], 4)) for d in (p ? (-1, 1) : (0, 0))]
    end
    return KF.OrbitSurfaceTable(gm.lambda[1:gm.nlmda], segs, L, zeros(Int, 6))
end
function swap(tab, gm, which, sdj)
    segs = map(enumerate(tab.segments)) do (il, ss)
        map(ss) do s
            m = length(s.u)
            KF.OrbitSegment(s.passing, s.dir, s.ua, s.ub, s.u, which == :wb ? fill(gm.wb[il], m) : s.wb, which == :wd ? fill(gm.wd[il], m) : s.wd,
                which == :dj ? repeat(sdj[il:il, :], m) : s.dJdJ)
        end
    end
    return KF.OrbitSurfaceTable(tab.lambda, segs, tab.lmax, tab.nstatus)
end
open(ARGS[2], "w") do io
    println(io, "zi,psi,variant,class,forb_re,forb_im,wdsurf_re,wdsurf_im,wbsurf_re,wbsurf_im,djsurf_re,djsurf_im,replay_re,replay_im")
    for ψ in parse.(Float64, ARGS[4:end])
        so = KF._torque_surface(ψ, zi, mi, false, "forb", equil, intr, prof; n)
        gm = so.geometry
        sdj = reduce(hcat, [KF.bounce_harmonic(gm, l, n, so.q, intr.ro, intr.mpert).dJdJ for l in -L:L])
        tabs = (so.orbit, swap(so.orbit, gm, :wd, sdj), swap(so.orbit, gm, :wb, sdj), swap(so.orbit, gm, :dj, sdj), replay_table(gm, so))
        for (v, f) in (("A", 1.0), ("B", 1 / zi)), c in (:passing, :trapped)
            T = [tot(t, so, f, c) for t in tabs]
            println(io, join((zi, ψ, v, c, (x for t in T for x in (real(t), imag(t)))...), ","))
            println("ψ=$ψ $v $c: " * join((string(k, " ", round(abs(t / T[5] - 1); sigdigits=2)) for (k, t) in zip(("forb", "wdsurf", "wbsurf", "djsurf"), T[1:4])), ", "))
            flush(io)
            flush(stdout)
        end
    end
end
