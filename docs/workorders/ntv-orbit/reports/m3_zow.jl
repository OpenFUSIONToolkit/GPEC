# M3, check 1 with the M3 drive: as m3_zow.jl, but forb evaluates the drive at each orbit's ψ̄ with the factor
# ∂ψ̄/∂ψ₀ at fixed (E, μ) and the launch-section weight 1/ncross; the replay keeps the surface drive. Variant B scales
# ω_E by 1/Z in the orbit drive too, as it does in the replay's numerator.
# forb is compared with the surface values replayed through the orbit integrals on the same pitch grid (so the pitch
# resolution is common), and with fgar; variant B divides ω_E and ν by Z, so every frequency except ω_b scales as 1/Z.
# ψ nodes: 3-point Gauss–Legendre on the adaptive quadrature's panels, with extra panels above ψ = 0.9.
# Usage (from gpec/): julia --project=. -t 1 docs/workorders/ntv-orbit/reports/m3_zow.jl <run_dir> <out_dir> <zi> <chunk> <nchunks> [classes]
# With "classes", each row also carries the passing and trapped shares of forb and replay in both variants.
using GeneralizedPerturbedEquilibrium, Serialization, Printf, QuadGK
const GPEC = GeneralizedPerturbedEquilibrium
const KF = GPEC.KineticForces

run_dir, out = ARGS[1], ARGS[2]
zi, chunk, nchunks = parse(Int, ARGS[3]), parse(Int, ARGS[4]), parse(Int, ARGS[5])
per_class = length(ARGS) > 5 && ARGS[6] == "classes"
mkpath(out)
equil, intr, prof = deserialize(joinpath(run_dir, "state.jls"))
const n, L, mi = 1, 6, 2
g = KF.guiding_center_field(equil)
intr.orbit_field, intr.orbit_lmax, intr.orbit_nlambda, intr.orbit_nenergy, intr.orbit_potential = g, L, 64, 12, false
intr.orbit_energy_tol = 5e-3

psi_c = KF.axis_validity_boundary(KF.KineticForcesControl(), nothing, prof, equil, intr.sing_psis)
x0, xout = max(equil.config.psilow, psi_c), min(intr.psilim, 1 - 1e-6)
res = KF.kinetic_resonance_psi_nodes(prof, equil; n, nl=L, zi=1, mi, electron=false, wdfac=1.0)
panels = KF.psi_panel_points(vcat(intr.sing_psis, res, [2psi_c], [0.9, 0.93, 0.95, 0.97, 0.98, 0.99]), x0, xout)
gx, gw = QuadGK.gauss(3)
nodes, weights = Float64[], Float64[]
for i in 1:length(panels)-1
    a, b = panels[i], panels[i+1]
    append!(nodes, (a + b) / 2 .+ (b - a) / 2 .* gx)
    append!(weights, (b - a) / 2 .* gw)
end
mine = chunk:nchunks:length(nodes)

orbit_total(tab, sf, f; classes=:all, drive=nothing) = sum(
    l -> KF.calculate_orbit(tab, n, l, sf.q, sf.epsr, sf.wdian, sf.wdiat, sf.welec * f, sf.nuk * f, intr.bo, sf.bmax, sf.n_s, sf.T_s;
        chi1=intr.chi1, ro=intr.ro, energy_atol=1e-11, energy_rtol=1e-7, pitch_atol=1e-9, pitch_rtol=1e-5, classes, drive), -L:L)
# The M3 orbit drive with ω_E scaled by f (Maxwellian, no potential in the orbit).
function scaled_drive(ψ0, chrg, f)
    d = KF.orbit_drive(prof, g, ψ0, chrg, intr.chi1; potential=false)
    f == 1 && return d
    return function (x, dψ, gs)
        ψ = ψ0 + dψ
        nb, Tb = prof.ni_spline(ψ), prof.Ti_spline(ψ)
        wn = -2π * Tb * prof.ni_deriv(ψ) / (chrg * intr.chi1 * nb)
        wt = -2π * prof.Ti_deriv(ψ) / (chrg * intr.chi1)
        xb = x * d.T0 / Tb
        return gs * (nb / d.n0) * (d.T0 / Tb)^2.5 * exp(-xb) * (f * prof.omegaE_spline(ψ) + wn + wt * (xb - 1.5))
    end
end
function replay_table(gm, sf)
    sdj = reduce(hcat, [KF.bounce_harmonic(gm, l, n, sf.q, intr.ro, intr.mpert).dJdJ for l in -L:L])
    us = [0.5, 1.0, 2.0, 4.0]
    segs = map(1:gm.nlmda) do il
        gm.valid[il] || return KF.OrbitSegment[]
        passing = gm.sigma[il] != 0
        [KF.OrbitSegment(passing, dir, 0.0, sqrt(KF.X_ENERGY_MAX), us, fill(gm.wb[il], 4), fill(gm.wd[il], 4), repeat(sdj[il:il, :], 4))
            for dir in (passing ? (-1, 1) : (0, 0))]
    end
    return KF.OrbitSurfaceTable(gm.lambda[1:gm.nlmda], segs, L, zeros(Int, 6))
end
fgar_total(sf, ψ) = sum(l -> (r = Ref{ComplexF64}(0); KF.tpsi!(r, ψ, n, l, zi, mi, 1.0, 1.0, false, "fgar", equil, intr, prof; surface=sf); r[]), -L:L)

file = joinpath(out, @sprintf("zow_z%05d_c%02d%s.csv", zi, chunk, per_class ? "_classes" : ""))
open(file, "w") do io
    println(
        io,
        "zi,psi,weight,envelope,forb_re,forb_im,replay_re,replay_im,fgar_re,fgar_im,forbB_re,forbB_im,replayB_re,replayB_im,table_s,integrals_s,segments,nstatus" *
        (per_class ? ",cls" * join(("_$(m)$(v)_$(c)_$(p)" for m in ("forb", "replay") for v in ("", "B") for c in ("passing", "trapped") for p in ("re", "im")), ",cls") : "")
    )
    for j in mine
        ψ = nodes[j]
        env = KF.kinetic_axis_validity_envelope(ψ, psi_c)
        sf = KF._torque_surface(ψ, zi, mi, false, "fgar", equil, intr, prof; n)
        t0 = time()
        so = KF._torque_surface(ψ, zi, mi, false, "forb", equil, intr, prof; n)
        tb = time() - t0
        t0 = time()
        gref = KF.compute_bounce_geometry(ψ, n, so.q, intr.bo, so.bmax, so.bmin, so.theta_bmax, so.tspl, so.B_vpar, intr.mfac, intr.chi1, intr.ro,
            so.dbob_m_f, so.divx_m_f, 1.0, 1.0, so.mass, so.chrg, so.T_s, "fgar"; nlmda=intr.orbit_nlambda, ntheta=512, surface_field=so.bf)
        rep = replay_table(gref, so)
        dA, dB = scaled_drive(ψ, so.chrg, 1.0), scaled_drive(ψ, so.chrg, 1 / zi)
        To, Tr, Tf = orbit_total(so.orbit, so, 1.0; drive=dA), orbit_total(rep, so, 1.0), fgar_total(sf, ψ)
        ToB, TrB = orbit_total(so.orbit, so, 1 / zi; drive=dB), orbit_total(rep, so, 1 / zi)
        cls =
            per_class ?
            [orbit_total(t, so, f; classes=c, drive=(t === rep ? nothing : f == 1 ? dA : dB)) for t in (so.orbit, rep) for f in (1.0, 1 / zi) for c in (:passing, :trapped)] :
            ComplexF64[]
        ti = time() - t0
        println(
            io,
            join(
                (zi, ψ, weights[j], env, real(To), imag(To), real(Tr), imag(Tr), real(Tf), imag(Tf), real(ToB), imag(ToB), real(TrB), imag(TrB),
                    round(tb; digits=1), round(ti; digits=1), sum(length, so.orbit.segments), join(so.orbit.nstatus, ";"), (x for c in cls for x in (real(c), imag(c)))...), ",")
        )
        flush(io)
        @printf("zi=%d ψ=%.4f table %.0f s, integrals %.0f s, |forb/replay−1| = %.2e\n", zi, ψ, tb, ti, abs(To / Tr - 1))
        flush(stdout)
    end
end
