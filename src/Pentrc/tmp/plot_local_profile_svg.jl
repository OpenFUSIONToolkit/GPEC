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

function load_fortran_local(case_dir::String, method::String, ell::Int)
    ncfile = joinpath(case_dir, "pentrc_output_n1.nc")
    var_psi = "psi_$(method)"
    var_d = "dTdpsi_$(method)"
    text = read(`ncdump -v $(var_psi),ell,$(var_d) $ncfile`, String)

    data_idx = findfirst("data:", text)
    data_idx === nothing && error("ncdump output missing data section")
    data = text[first(data_idx):end]

    ells = parse_ncdump_vector(data, "ell"; stop_label="$(var_psi) =", int_mode=true)
    psi = parse_ncdump_vector(data, var_psi; stop_label="$(var_d) =")
    dvals = parse_ncdump_vector(data, var_d; stop_label="}")

    ell_idx = findfirst(==(ell), ells)
    ell_idx === nothing && error("Could not find ell=$ell in Fortran output")

    ni = 2
    nell = length(ells)
    expected = length(psi) * nell * ni
    length(dvals) == expected || error("Unexpected dTdpsi block length")

    dprof = ComplexF64[]
    for ipsi in eachindex(psi)
        base = ((ipsi - 1) * nell + (ell_idx - 1)) * ni + 1
        push!(dprof, ComplexF64(dvals[base], dvals[base + 1]))
    end

    return psi, dprof
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

function svg_polyline_points(xs, ys, xmap, ymap)
    return join((@sprintf("%.2f,%.2f", xmap(x), ymap(y)) for (x, y) in zip(xs, ys)), " ")
end

function write_panel(io, x, y_fortran, y_julia, x0, y0, w, h, title)
    xmin, xmax = minimum(x), maximum(x)
    ymin = min(minimum(y_fortran), minimum(y_julia))
    ymax = max(maximum(y_fortran), maximum(y_julia))
    if isapprox(ymin, ymax; atol=0, rtol=1e-12)
        δ = max(abs(ymin), 1.0) * 0.05
        ymin -= δ
        ymax += δ
    else
        δ = 0.08 * (ymax - ymin)
        ymin -= δ
        ymax += δ
    end

    xmap(v) = x0 + (v - xmin) / max(xmax - xmin, eps()) * w
    ymap(v) = y0 + h - (v - ymin) / max(ymax - ymin, eps()) * h

    println(io, @sprintf("""<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="white" stroke="#cccccc"/>""", x0, y0, w, h))
    println(io, @sprintf("""<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#333" stroke-width="1"/>""", x0, y0+h, x0+w, y0+h))
    println(io, @sprintf("""<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#333" stroke-width="1"/>""", x0, y0, x0, y0+h))
    println(io, @sprintf("""<text x="%.1f" y="%.1f" font-size="18" font-family="Helvetica" fill="#111">%s</text>""", x0, y0-12, title))

    fpts = svg_polyline_points(x, y_fortran, xmap, ymap)
    jpts = svg_polyline_points(x, y_julia, xmap, ymap)
    println(io, """<polyline fill="none" stroke="#1f77b4" stroke-width="2.5" points="$fpts"/>""")
    println(io, """<polyline fill="none" stroke="#d95f02" stroke-width="2.5" points="$jpts"/>""")
    for (xp, yp) in zip(x, y_fortran)
        println(io, @sprintf("""<circle cx="%.2f" cy="%.2f" r="3.2" fill="#1f77b4"/>""", xmap(xp), ymap(yp)))
    end

    println(io, @sprintf("""<text x="%.1f" y="%.1f" font-size="12" font-family="Helvetica" fill="#333">psi min = %.3f, psi max = %.3f</text>""", x0, y0+h+22, xmin, xmax))
    println(io, @sprintf("""<text x="%.1f" y="%.1f" font-size="12" font-family="Helvetica" fill="#333">y min = %.3e, y max = %.3e</text>""", x0+230, y0+h+22, ymin, ymax))
end

function write_svg(outfile::String, method::String, ell::Int, psi::Vector{Float64}, fortran::Vector{ComplexF64}, julia::Vector{ComplexF64})
    width = 960
    height = 760
    panel_x = 80
    panel_w = 820
    panel_h = 250

    open(outfile, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="#f8f8f8"/>""")
        println(io, @sprintf("""<text x="80" y="40" font-size="24" font-family="Helvetica" fill="#111">%s local profile comparison (ell = %d)</text>""", uppercase(method), ell))
        println(io, """<text x="80" y="68" font-size="14" font-family="Helvetica" fill="#444">Blue: Fortran points + line, Orange: Julia line, x-axis: psi</text>""")

        write_panel(io, psi, real.(fortran), real.(julia), panel_x, 110, panel_w, panel_h, "Real part")
        write_panel(io, psi, imag.(fortran), imag.(julia), panel_x, 430, panel_w, panel_h, "Imaginary part")

        println(io, """</svg>""")
    end
end

function main(case_dir::String, method::String, ell::Int, nsample::Int, outfile::String, electron::Bool)
    psi_all, fortran_all = load_fortran_local(case_dir, method, ell)
    idx = choose_sample_indices(length(psi_all), nsample)
    psi = psi_all[idx]
    fortran = fortran_all[idx]

    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            load_julia_state(case_dir, method)
        end
    end

    ts = Ref{ComplexF64}(0.0 + 0.0im)
    julia = ComplexF64[]
    for ψ in psi
        val = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                P.tpsi!(ts, ψ, 1, ell, 1, 2, 1.0, 1.0, electron, method)
                ts[]
            end
        end
        push!(julia, val)
    end

    write_svg(outfile, method, ell, collect(psi), fortran, julia)
    println(outfile)
end

case_dir = get(ENV, "PENTRC_CASE", joinpath(@__DIR__, "runtime_clar_case"))
method = get(ENV, "PENTRC_METHOD", "clar")
ell = parse(Int, get(ENV, "PENTRC_ELL", method == "fcgl" ? "0" : "1"))
nsample = parse(Int, get(ENV, "PENTRC_POINTS", "20"))
outfile = get(ENV, "PENTRC_OUTFILE", joinpath(case_dir, "$(method)_local_profile_compare.svg"))
electron = lowercase(get(ENV, "PENTRC_ELECTRON", "false")) in ("1", "true", "t", "yes", "y")

main(case_dir, method, ell, nsample, outfile, electron)
