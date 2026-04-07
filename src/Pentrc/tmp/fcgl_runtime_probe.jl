include(joinpath(@__DIR__, "..", "..", "JPEC.jl"))

using .JPEC
using Printf

const P = JPEC.PENTRC

function parse_bool_env(key::String, default::Bool=false)
    return lowercase(get(ENV, key, default ? "true" : "false")) in ("1", "true", "t", "yes", "y")
end

function parse_idconfile(case_dir::String)
    pentrc_in = joinpath(case_dir, "pentrc.in")
    if isfile(pentrc_in)
        for line in eachline(pentrc_in)
            m = match(r"idconfile\s*=\s*\"([^\"]+)\"", line)
            m === nothing || return joinpath(case_dir, m.captures[1])
        end
    end
    fallback = joinpath(case_dir, "euler.bin")
    return fallback
end

function main(case_dir::String;
              method::String,
              psi::Float64,
              ell::Int,
              electron::Bool=false,
              zi::Int=1,
              mi::Int=2,
              wdfac::Float64=1.0,
    divxfac::Float64=1.0)
    eq = JPEC.Equilibrium.setup_equilibrium(joinpath(case_dir, "equil.toml"))
    P.load_equilibrium!(eq; idconfile=parse_idconfile(case_dir))
    P.load_kinetic_profiles!(joinpath(case_dir, "g147131.02300_DIIID_KEFIT.kin"))
    P.load_pmodb!(joinpath(case_dir, "gpec_pmodb_n1.out"))

    ts = Ref{ComplexF64}(0.0 + 0.0im)
    P.set_trace!(enabled=true, method=method, psi=psi, ell=ell)
    try
        P.tpsi!(ts, psi, 1, ell, zi, mi, wdfac, divxfac, electron, method)
        @printf("TRACE_PROBE method=%s psi=%.15e ell=%d tpsi=%.12e %+.12ei\n",
                method, psi, ell, real(ts[]), imag(ts[]))
    finally
        P.clear_trace!()
    end
    return nothing
end

case_dir = get(ENV, "PENTRC_CASE", joinpath(@__DIR__, "runtime_fcgl_case"))
method = lowercase(strip(get(ENV, "PENTRC_METHOD", "fcgl")))
psi_str = get(ENV, "PENTRC_PSI", "")
isempty(psi_str) && error("Set PENTRC_PSI to the exact psi value to trace.")
psi = parse(Float64, psi_str)
ell = parse(Int, get(ENV, "PENTRC_ELL", method == "fcgl" ? "0" : "1"))
electron = parse_bool_env("PENTRC_ELECTRON", false)
zi = parse(Int, get(ENV, "PENTRC_ZI", "1"))
mi = parse(Int, get(ENV, "PENTRC_MI", "2"))
wdfac = parse(Float64, get(ENV, "PENTRC_WDFAC", "1.0"))
divxfac = parse(Float64, get(ENV, "PENTRC_DIVXFAC", "1.0"))

main(case_dir;
     method=method,
     psi=psi,
     ell=ell,
     electron=electron,
     zi=zi,
     mi=mi,
     wdfac=wdfac,
     divxfac=divxfac)
