include(joinpath(@__DIR__, "..", "..", "JPEC.jl"))

using .JPEC
using Printf

const P = JPEC.PENTRC

function extract_block(text::String, start_label::String, stop_label::Union{String,Nothing})
    start_idx = findfirst(start_label, text)
    start_idx === nothing && error("Could not find block starting with $(repr(start_label))")
    from = last(start_idx) + 1
    to =
        if isnothing(stop_label)
            lastindex(text)
        else
            stop_idx = findfirst(stop_label, text[from:end])
            stop_idx === nothing && error("Could not find block stopping at $(repr(stop_label))")
            from + first(stop_idx) - 2
        end
    return text[from:to]
end

function parse_ncdump_vector(text::String, label::String; stop_label::Union{String,Nothing}=nothing, int_mode::Bool=false)
    block = extract_block(text, label * " =", stop_label)
    rx = int_mode ? r"[-+]?\d+" : r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"
    vals = [m.match for m in eachmatch(rx, block)]
    return int_mode ? parse.(Int, vals) : parse.(Float64, vals)
end

function choose_sample_indices(n::Int, nsample::Int)
    nsel = min(n, nsample)
    idx = round.(Int, range(1, n, length=nsel))
    out = Int[]
    for i in idx
        isempty(out) || out[end] != i || continue
        push!(out, i)
    end
    return out
end

function load_fortran_profiles(case_dir::String, method::String, ell::Int)
    ncfile = joinpath(case_dir, "pentrc_output_n1.nc")
    var_psi = "psi_$(method)"
    var_d = "dTdpsi_$(method)"
    var_t = "T_$(method)"
    text = read(`ncdump -v $(var_psi),ell,$(var_d),$(var_t) $ncfile`, String)

    data_idx = findfirst("data:", text)
    data_idx === nothing && error("ncdump output missing data section")
    data = text[first(data_idx):end]

    ells = parse_ncdump_vector(data, "ell"; stop_label="$(var_psi) =", int_mode=true)
    psi = parse_ncdump_vector(data, var_psi; stop_label="$(var_d) =")
    dvals = parse_ncdump_vector(data, var_d; stop_label="$(var_t) =")
    tvals = parse_ncdump_vector(data, var_t; stop_label="}")

    ell_idx = findfirst(==(ell), ells)
    ell_idx === nothing && error("Could not find ell=$ell in Fortran output")

    ni = 2
    nell = length(ells)
    expected = length(psi) * nell * ni
    length(dvals) == expected || error("Unexpected dTdpsi block length")
    length(tvals) == expected || error("Unexpected T block length")

    dprof = ComplexF64[]
    tprof = ComplexF64[]
    for ipsi in eachindex(psi)
        base = ((ipsi - 1) * nell + (ell_idx - 1)) * ni + 1
        push!(dprof, ComplexF64(dvals[base], dvals[base + 1]))
        push!(tprof, ComplexF64(tvals[base], tvals[base + 1]))
    end

    return psi, dprof, tprof
end

function load_julia_state(case_dir::String, method::String)
    eq = JPEC.Equilibrium.setup_equilibrium(joinpath(case_dir, "equil.toml"))
    P.load_equilibrium!(eq)
    P.load_kinetic_profiles!(joinpath(case_dir, "g147131.02300_DIIID_KEFIT.kin"))
    P.load_pmodb!(joinpath(case_dir, "gpec_pmodb_n1.out"))
    if method in ("rlar", "clar")
        P.load_fnml!("/Users/iseonjae/Desktop/GPEC/pentrc/fkmnl.dat")
    end
end

function main(case_dir::String, method::String, ell::Int, nsample::Int, outfile::String)
    psi, fortran_local, fortran_cumulative = load_fortran_profiles(case_dir, method, ell)
    sample_idx = choose_sample_indices(length(psi), nsample)

    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            load_julia_state(case_dir, method)
        end
    end

    ts = Ref{ComplexF64}(0.0 + 0.0im)
    open(outfile, "w") do io
        println(io, "psi,fortran_local_re,fortran_local_im,julia_local_re,julia_local_im,fortran_T_re,fortran_T_im,julia_T_re,julia_T_im")
        for idx in sample_idx
            ψ = psi[idx]
            julia_local = redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    P.tpsi!(ts, ψ, 1, ell, 1, 2, 1.0, 1.0, false, method)
                    ts[]
                end
            end
            julia_cumulative = redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    P.tintgrl_lsode([0.0, ψ], 1, ell, 1, 2, 1.0, 1.0, false, method)
                end
            end

            fl = fortran_local[idx]
            ft = fortran_cumulative[idx]
            @printf(
                io,
                "%.15f,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                ψ,
                real(fl), imag(fl),
                real(julia_local), imag(julia_local),
                real(ft), imag(ft),
                real(julia_cumulative), imag(julia_cumulative),
            )
        end
    end

    println(outfile)
end

case_dir = get(ENV, "PENTRC_CASE", joinpath(@__DIR__, "runtime_clar_case"))
method = get(ENV, "PENTRC_METHOD", "clar")
ell = parse(Int, get(ENV, "PENTRC_ELL", method == "fcgl" ? "0" : "1"))
nsample = parse(Int, get(ENV, "PENTRC_POINTS", "20"))
outfile = get(ENV, "PENTRC_OUTFILE", joinpath(@__DIR__, "$(method)_profile_compare.csv"))

main(case_dir, method, ell, nsample, outfile)
