# M2 checks: forb (orbit-averaged) against fgar (surface) on the DIII-D-like n = 1 case.
# Both paths are evaluated on the same fixed Gauss–Legendre ψ nodes, so the ψ quadrature error is common.
# Usage (from gpec/): julia --project=. -t 16 docs/workorders/ntv-orbit/reports/m2_checks.jl <run_dir> <out_dir>
using GeneralizedPerturbedEquilibrium, Serialization, Printf, QuadGK
const GPEC = GeneralizedPerturbedEquilibrium
const KF = GPEC.KineticForces

run_dir, out = ARGS[1], ARGS[2]
mkpath(out)
cache = joinpath(run_dir, "state.jls")
if isfile(cache)
    equil, intr, prof = deserialize(cache)
else
    r = GPEC.main([run_dir])
    ffs = r.pe.response.ffs
    equil = ffs.equil
    prof = KF.ntv_profiles(r.kinetic.control, equil)
    intr = KF.KineticForcesInternal(equil)
    KF.set_perturbation_data!(intr, r.pe, ffs, equil, ffs.metric)
    serialize(cache, (equil, intr, prof))
end
const n, L, mi = 1, 6, 2
g = KF.guiding_center_field(equil)
intr.orbit_field, intr.orbit_lmax, intr.orbit_nlambda, intr.orbit_nenergy, intr.orbit_potential = g, L, 128, 12, false

# Fixed ψ nodes: 6-point Gauss–Legendre on the panels the adaptive quadrature uses.
psi_c = KF.axis_validity_boundary(KF.KineticForcesControl(), nothing, prof, equil, intr.sing_psis)
x0, xout = max(equil.config.psilow, psi_c), min(intr.psilim, 1 - 1e-6)
res = KF.kinetic_resonance_psi_nodes(prof, equil; n, nl=L, zi=1, mi, electron=false, wdfac=1.0)
panels = KF.psi_panel_points(vcat(intr.sing_psis, res, [2psi_c]), x0, xout)
gx, gw = QuadGK.gauss(6)
nodes, weights = Float64[], Float64[]
for i in 1:length(panels)-1
    a, b = panels[i], panels[i+1]
    append!(nodes, (a + b) / 2 .+ (b - a) / 2 .* gx)
    append!(weights, (b - a) / 2 .* gw)
end
env = [KF.kinetic_axis_validity_envelope(ψ, psi_c) for ψ in nodes]
@printf("ψ nodes: %d on %d panels, domain [%.4f, %.4f], psi_c = %.4f, threads = %d\n", length(nodes), length(panels) - 1, x0, xout, psi_c, Threads.nthreads())

sumℓ(surf, ψ, zi, method) = sum(l -> (r = Ref{ComplexF64}(0); KF.tpsi!(r, ψ, n, l, zi, mi, 1.0, 1.0, false, method, equil, intr, prof; surface=surf); r[]), -L:L)
orbit_total(tab, sf) = sum(
    l -> KF.calculate_orbit(tab, n, l, sf.q, sf.epsr, sf.wdian, sf.wdiat, sf.welec, sf.nuk, intr.bo, sf.bmax, sf.n_s, sf.T_s;
        chi1=intr.chi1, ro=intr.ro, energy_atol=1e-11, energy_rtol=1e-8, pitch_atol=1e-9, pitch_rtol=1e-6), -L:L)
function replay_table(gm, sf)
    us = KF.orbit_energy_nodes(12)
    sdj = reduce(hcat, [KF.bounce_harmonic(gm, l, n, sf.q, intr.ro, intr.mpert).dJdJ for l in -L:L])
    dJ = zeros(gm.nlmda, length(us), 2, 2L + 1)
    for k in eachindex(us), s in 1:2
        dJ[:, k, s, :] .= sdj
    end
    return KF.OrbitSurfaceTable(gm.lambda, BitVector(gm.sigma .== 0), us, repeat(gm.wb, 1, length(us), 2), repeat(gm.wd, 1, length(us), 2), dJ,
        trues(gm.nlmda, length(us), 2), L, zeros(Int, 6))
end

# Orbit width: the trapped orbit at ψ = 0.5, x = 1, mid-trapped pitch, as a fraction of the minor radius.
function width_over_a(zi)
    sf = KF._setup_surface_state(0.5, zi, mi, false, equil, intr, prof)
    lam = (intr.bo / sf.bmax + intr.bo / sf.bmin) / 2
    o = KF.guiding_center_orbit(g, 0.5, lam, sf.T_s, 1; mass=sf.mass, chrg=sf.chrg, theta0=sf.theta_bmin)
    rr = equil.geometry.avg_r_spline
    return (rr(o.psi_range[2]) - rr(o.psi_range[1])) / rr(xout)
end

zis = [1, 4, 16, 64, 256, 1024, 4096]
scan = Dict{Int,Any}()
for zi in zis
    tf, to = zeros(ComplexF64, length(nodes)), zeros(ComplexF64, length(nodes))
    ttab = zeros(length(nodes))
    nst = [zeros(Int, 6) for _ in nodes]
    t0 = time()
    Threads.@threads :dynamic for j in eachindex(nodes)
        ψ = nodes[j]
        sf = KF._torque_surface(ψ, zi, mi, false, "fgar", equil, intr, prof; n)
        t1 = time()
        so = KF._torque_surface(ψ, zi, mi, false, "forb", equil, intr, prof; n)
        ttab[j] = time() - t1
        tf[j], to[j] = sumℓ(sf, ψ, zi, "fgar") * env[j], sumℓ(so, ψ, zi, "forb") * env[j]
        nst[j] = so.orbit.nstatus
    end
    Tf, To = sum(weights .* tf), sum(weights .* to)
    w = width_over_a(zi)
    scan[zi] = (; width=w, Tf, To, rel=abs(To / Tf - 1), rel_re=abs(real(To) / real(Tf) - 1), tf, to, nstatus=sum(nst), wall=time() - t0, table_s=sum(ttab))
    @printf("zi=%5d  Δb/a=%.3e  T_fgar=%+.6e%+.6ei  T_forb=%+.6e%+.6ei  |rel|=%.3e  rel(Re)=%.3e  wall %.0fs  tables %.0f core-s  status %s\n",
        zi, w, real(Tf), imag(Tf), real(To), imag(To), scan[zi].rel, scan[zi].rel_re, scan[zi].wall, scan[zi].table_s, scan[zi].nstatus)
    flush(stdout)
end

# zi = 1 variants: replay identity, grid doubling (pitch, energy), and the zero-width layer's share.
variants = (:replay, :base, :lam256, :en24, :layer3, :layer30)
vt = Dict(v => zeros(ComplexF64, length(nodes)) for v in variants)
t0 = time()
Threads.@threads :dynamic for j in eachindex(nodes)
    ψ = nodes[j]
    sf = KF._torque_surface(ψ, 1, mi, false, "fgar", equil, intr, prof; n)
    geo(nl) = KF.compute_bounce_geometry(ψ, n, sf.q, intr.bo, sf.bmax, sf.bmin, sf.theta_bmax, sf.tspl, sf.B_vpar, intr.mfac,
        intr.chi1, intr.ro, sf.dbob_m_f, sf.divx_m_f, 1.0, 1.0, sf.mass, sf.chrg, sf.T_s, "forb"; nlmda=nl, surface_field=sf.bf)
    g128, g256 = geo(128), geo(256)
    tab(gm; kw...) = KF.orbit_surface_table(g, gm, intr, ψ, sf.theta_bmin, sf.q, sf.welec, sf.mass, sf.chrg, sf.T_s, n; lmax=L, kw...)
    vt[:replay][j] = orbit_total(replay_table(g128, sf), sf) * env[j]
    vt[:base][j] = orbit_total(tab(g128), sf) * env[j]
    vt[:lam256][j] = orbit_total(tab(g256), sf) * env[j]
    vt[:en24][j] = orbit_total(tab(g128; nenergy=24), sf) * env[j]
    vt[:layer3][j] = orbit_total(tab(g128; layer_factor=3.0), sf) * env[j]
    vt[:layer30][j] = orbit_total(tab(g128; layer_factor=30.0), sf) * env[j]
end
VT = Dict(v => sum(weights .* vt[v]) for v in variants)
replay_dev = maximum(abs.(vt[:replay] .- scan[1].tf)) / maximum(abs.(scan[1].tf))
@printf("zi=1 variants (wall %.0fs): replay vs fgar max node dev %.2e; total replay/fgar-1 = %.2e\n", time() - t0, replay_dev, abs(VT[:replay] / scan[1].Tf - 1))
for v in (:lam256, :en24, :layer3, :layer30)
    @printf("  %-8s T=%+.6e%+.6ei  change vs base %.3e\n", v, real(VT[v]), imag(VT[v]), abs(VT[v] / VT[:base] - 1))
end

open(joinpath(out, "zow_scan.csv"), "w") do io
    println(io, "zi,width_over_a,Tf_re,Tf_im,To_re,To_im,rel,rel_re,wall_s,table_core_s")
    for zi in zis
        s = scan[zi]
        println(io, join((zi, s.width, real(s.Tf), imag(s.Tf), real(s.To), imag(s.To), s.rel, s.rel_re, s.wall, s.table_s), ","))
    end
end
open(joinpath(out, "profiles.csv"), "w") do io
    println(io, "psi,weight,fgar_z1_re,fgar_z1_im,forb_z1_re,forb_z1_im,fgar_z1024_re,forb_z1024_re")
    for j in eachindex(nodes)
        println(
            io,
            join((nodes[j], weights[j], real(scan[1].tf[j]), imag(scan[1].tf[j]), real(scan[1].to[j]), imag(scan[1].to[j]),
                    real(scan[1024].tf[j]), real(scan[1024].to[j])), ",")
        )
    end
end
open(joinpath(out, "variants.csv"), "w") do io
    println(io, "variant,T_re,T_im,change_vs_base")
    for v in variants
        println(io, join((v, real(VT[v]), imag(VT[v]), abs(VT[v] / VT[:base] - 1)), ","))
    end
    println(io, "replay_max_node_dev,$replay_dev,0,0")
end
println("CHECKS_DONE")
