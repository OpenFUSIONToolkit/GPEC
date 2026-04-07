include(joinpath(@__DIR__, "..", "..", "JPEC.jl"))

using .JPEC
using Printf

const P = JPEC.PENTRC

function main(case_dir::String=joinpath(@__DIR__, "runtime_rlar_case"))
    eq = JPEC.Equilibrium.setup_equilibrium(joinpath(case_dir, "equil.toml"))
    P.load_equilibrium!(eq)
    P.load_kinetic_profiles!(joinpath(case_dir, "g147131.02300_DIIID_KEFIT.kin"))
    P.load_pmodb!(joinpath(case_dir, "gpec_pmodb_n1.out"))
    P.load_fnml!("/Users/iseonjae/Desktop/GPEC/pentrc/fkmnl.dat")

    ts = Ref{ComplexF64}(0.0 + 0.0im)
    for psi in (0.1, 0.5, 0.9)
        P.tpsi!(ts, psi, 1, 1, 1, 2, 1.0, 1.0, false, "rlar")
        @printf("psi=%.1f tpsi=%.12e %+.12ei\n", psi, real(ts[]), imag(ts[]))
    end
end

main()
