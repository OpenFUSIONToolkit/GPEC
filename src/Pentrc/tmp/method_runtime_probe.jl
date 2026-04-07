include(joinpath(@__DIR__, "..", "..", "JPEC.jl"))

using .JPEC
using Printf

const P = JPEC.PENTRC

function _parse_psi_list(raw::AbstractString)
    isempty(strip(raw)) && return nothing
    return parse.(Float64, split(raw, ','))
end

function main(case_dir::String, method::String, ell::Int=1; psi_values=nothing)
    eq = JPEC.Equilibrium.setup_equilibrium(joinpath(case_dir, "equil.toml"))
    P.load_equilibrium!(eq)
    P.load_kinetic_profiles!(joinpath(case_dir, "g147131.02300_DIIID_KEFIT.kin"))
    P.load_pmodb!(joinpath(case_dir, "gpec_pmodb_n1.out"))
    if method in ("rlar", "clar")
        P.load_fnml!("/Users/iseonjae/Desktop/GPEC/pentrc/fkmnl.dat")
    end

    ts = Ref{ComplexF64}(0.0 + 0.0im)
    psi_grid = isnothing(psi_values) ? (0.1, 0.5, 0.9) : psi_values
    for psi in psi_grid
        P.tpsi!(ts, psi, 1, ell, 1, 2, 1.0, 1.0, false, method)
        @printf("%s psi=%.12f tpsi=%.12e %+.12ei\n", method, psi, real(ts[]), imag(ts[]))
    end
end

case_dir = get(ENV, "PENTRC_CASE", joinpath(@__DIR__, "runtime_clar_case"))
method = get(ENV, "PENTRC_METHOD", "clar")
ell = parse(Int, get(ENV, "PENTRC_ELL", method == "fcgl" ? "0" : "1"))
psi_values = _parse_psi_list(get(ENV, "PENTRC_PSI_LIST", ""))

main(case_dir, method, ell; psi_values=psi_values)
