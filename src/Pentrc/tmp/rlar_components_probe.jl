include(joinpath(@__DIR__, "..", "..", "JPEC.jl"))

using .JPEC
using Printf

const P = JPEC.PENTRC

function main(case_dir::String=joinpath(@__DIR__, "runtime_rlar_case"), psi::Float64=0.5;
              conjugate_modes::Bool=false)
    eq = JPEC.Equilibrium.setup_equilibrium(joinpath(case_dir, "equil.toml"))
    P.load_equilibrium!(eq)
    P.load_kinetic_profiles!(joinpath(case_dir, "g147131.02300_DIIID_KEFIT.kin"))
    P.load_pmodb!(joinpath(case_dir, "gpec_pmodb_n1.out"))
    P.load_fnml!("/Users/iseonjae/Desktop/GPEC/pentrc/fkmnl.dat")

    if conjugate_modes
        P.dbob_m = P.Spl.CubicSpline(collect(P.dbob_m.xs), conj.(P.dbob_m.fs); bctype="extrap")
        P.divx_m = P.Spl.CubicSpline(collect(P.divx_m.xs), conj.(P.divx_m.fs); bctype="extrap")
    end

    ts = Ref{ComplexF64}(0.0 + 0.0im)
    P.tpsi!(ts, psi, 1, 1, 1, 2, 1.0, 1.0, false, "rlar")

    dbm = P.Spl.spline_eval!(P.dbob_m, psi)
    sqf = P.Spl.spline_eval!(P.sq, psi)
    kinf, kinf1 = P.Spl.spline_deriv1!(P.kin, psi)
    geomf = P.Spl.spline_eval!(P.geom, psi)

    s = 1
    chrg = P.e
    mass = 2 * P.mp
    q = sqf[4]
    epsr = geomf[2] / geomf[3]
    wdian = -P.twopi * kinf[s + 2] * kinf1[s] / (chrg * P.chi1 * kinf[s])
    wdiat = -P.twopi * kinf1[s + 2] / (chrg * P.chi1)
    welec = kinf[5]
    wtran = sqrt(2 * kinf[s + 2] / mass) / (q * P.ro)
    wgyro = chrg * P.bo / mass
    wbhat = (π / 4) * sqrt(epsr / 2) * wtran
    wdhat = q^3 * wtran^2 / (4 * epsr * wgyro)
    nueff = kinf[s + 6] / (2 * epsr)
    lmdamax = min(1 / (1 - epsr), P.bo / 0.5)

    xint = P.xintgrl_lsode(wdian, wdiat, welec, wdhat, wbhat, nueff,
                           1, 1.0, 1, psi, lmdamax, "rlar")
    kint = P.kappaintgrl(1, 1, q, P.mfac, dbm)

    @printf("psi=%.6f\n", psi)
    @printf("tpsi=%.12e %+.12ei\n", real(ts[]), imag(ts[]))
    @printf("xint=%.12e %+.12ei\n", real(xint), imag(xint))
    @printf("kappaint=%.12e\n", kint)
    @printf("epsr=%.12e\n", epsr)
    @printf("sq3=%.12e\n", sqf[3])
    @printf("wdian=%.12e\n", wdian)
    @printf("wdiat=%.12e\n", wdiat)
    @printf("welec=%.12e\n", welec)
    @printf("wdhat=%.12e\n", wdhat)
    @printf("wbhat=%.12e\n", wbhat)
    @printf("nueff=%.12e\n", nueff)
end

conjugate_modes = lowercase(get(ENV, "PENTRC_CONJUGATE_MODES", "false")) in ("1", "true", "t", "yes", "y")

main(; conjugate_modes=conjugate_modes)
